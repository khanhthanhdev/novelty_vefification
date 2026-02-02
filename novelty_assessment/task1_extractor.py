from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from paper_novelty_pipeline.utils.text_cleaning import (
    sanitize_for_llm,
    truncate_at_references,
)


class Task1ExtractionError(RuntimeError):
    pass


def build_task1_messages(
    *,
    paper_text: str,
    review_text: str,
    paper_title: Optional[str] = None,
    max_paper_chars: int = 200_000,
    max_review_chars: int = 60_000,
) -> List[Dict[str, str]]:
    """
    Build the single-call prompt for Task 1 (paper+review extraction).

    This function does NOT call any LLM; it only builds messages.
    """
    cleaned_paper = _prepare_paper_text(paper_text, max_chars=max_paper_chars)
    cleaned_review = _prepare_review_text(review_text, max_chars=max_review_chars)
    cleaned_title = sanitize_for_llm((paper_title or "").strip())

    system = (
        "You are extracting structured targets for verifiable novelty checking.\n\n"
        "You will receive TWO sources in the user message:\n"
        "1) PAPER TEXT (the submission)\n"
        "2) REVIEW TEXT (a peer review of that submission)\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT invent citations, titles, author names, years, arXiv IDs, URLs, or any prior-work references.\n"
        "- For any citation/title/prior-work mention, ONLY copy strings that appear in the REVIEW TEXT.\n"
        "- For novelty claims, the 'text' field MUST be verbatim from the REVIEW TEXT (1–2 sentences max).\n"
        "- Return STRICT JSON only. No markdown, no code fences, no extra keys.\n\n"
        "OUTPUT JSON SCHEMA (must match exactly):\n"
        "{\n"
        "  \"paper\": {\n"
        "    \"core_task\": \"string (<=20 words)\",\n"
        "    \"contributions\": [\"string (<=25 words)\", \"string (<=25 words)\", \"string (<=25 words)\"],\n"
        "    \"key_terms\": [\"5-12 short phrases\"],\n"
        "    \"must_have_entities\": [\"model/dataset/metric names if any\"]\n"
        "  },\n"
        "  \"review\": {\n"
        "    \"novelty_claims\": [\n"
        "      {\n"
        "        \"claim_id\": \"C1\",\n"
        "        \"text\": \"verbatim review claim (1-2 sentences max)\",\n"
        "        \"stance\": \"not_novel | somewhat_novel | novel | unclear\",\n"
        "        \"confidence_lang\": \"high | medium | low\",\n"
        "        \"mentions_prior_work\": true,\n"
        "        \"prior_work_strings\": [\"author-year strings or titles as written\"],\n"
        "        \"evidence_expected\": \"method_similarity | task_similarity | results_similarity | theory_overlap | dataset_overlap\"\n"
        "      }\n"
        "    ],\n"
        "    \"all_citations_raw\": [\"everything that looks like a citation/title/arxiv id/url\"]\n"
        "  }\n"
        "}\n\n"
        "PAPER-SIDE GUIDELINES:\n"
        "- Do NOT summarize the paper; extract only the core task and atomic contributions as query anchors.\n"
        "- core_task must be specific and concrete (e.g., 'visual question answering for chest X-rays'), not generic.\n"
        "- contributions must be atomic (method change / training recipe / benchmark / theory claim).\n"
        "- contributions list must contain 1–3 items; prefer 2-3 if clearly supported.\n"
        "- key_terms must contain 5–12 short technical phrases.\n"
        "- must_have_entities must list explicit names (models/datasets/metrics) if mentioned; else []\n\n"
        "REVIEW-SIDE GUIDELINES:\n"
        "- novelty_claims: include ONLY novelty-related statements (incremental/similar to X, main novelty is..., combines A and B, differs from prior work by...).\n"
        "- Exclude general weaknesses not about novelty.\n"
        "- prior_work_strings: if the claim mentions prior work, copy the exact strings as written in the review.\n"
        "- all_citations_raw: include everything in the review that looks like a citation/title/arXiv ID/URL/DOI.\n"
        "- If the review contains no novelty claims, return an empty novelty_claims list and still fill all_citations_raw.\n"
    )

    user_parts: List[str] = []
    if cleaned_title:
        user_parts.append(f"PAPER TITLE (if known):\n{cleaned_title}")
    user_parts.append("PAPER TEXT:\n<<<PAPER_TEXT_START>>>\n" + cleaned_paper + "\n<<<PAPER_TEXT_END>>>")
    user_parts.append("REVIEW TEXT:\n<<<REVIEW_TEXT_START>>>\n" + cleaned_review + "\n<<<REVIEW_TEXT_END>>>")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def extract_task1(
    *,
    paper_text: str,
    review_text: str,
    paper_title: Optional[str] = None,
    llm_client: Any = None,
    max_paper_chars: int = 200_000,
    max_review_chars: int = 60_000,
    max_tokens: int = 1400,
    temperature: float = 0.0,
    use_cache: bool = False,
    cache_ttl: str = "1h",
    strict_review_verbatim: bool = True,
    augment_citations_regex: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run Task 1 extraction using ONE LLM call.

    Returns a dict with exactly two top-level keys: {"paper": ..., "review": ...}.
    """
    log = logger or logging.getLogger(__name__)
    messages = build_task1_messages(
        paper_text=paper_text,
        review_text=review_text,
        paper_title=paper_title,
        max_paper_chars=max_paper_chars,
        max_review_chars=max_review_chars,
    )

    client = llm_client
    if client is None:
        try:
            from paper_novelty_pipeline.services.llm_client import create_llm_client

            client = create_llm_client()
        except AssertionError as e:
            raise Task1ExtractionError(
                "LLM is not configured. Set LLM_API_KEY (and optionally LLM_MODEL_NAME / LLM_API_ENDPOINT) "
                "then retry."
            ) from e

    if client is None:
        raise Task1ExtractionError("LLM client could not be initialized (create_llm_client returned None).")

    data = client.generate_json(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
    )
    if not isinstance(data, dict):
        raise Task1ExtractionError("LLM did not return a JSON object.")

    normalized = normalize_task1_output(
        data,
        review_text=review_text,
        strict_review_verbatim=strict_review_verbatim,
        augment_citations_regex=augment_citations_regex,
        logger=log,
    )
    return normalized


def normalize_task1_output(
    raw: Dict[str, Any],
    *,
    review_text: str,
    strict_review_verbatim: bool = True,
    augment_citations_regex: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Normalize/validate the Task 1 output to match the process.md schema.

    This function does not add any new information; it only:
    - enforces field presence/types
    - clamps list sizes and word limits
    - filters citation strings to those present in the review (if strict)
    """
    log = logger or logging.getLogger(__name__)

    paper_raw = raw.get("paper") if isinstance(raw, dict) else None
    review_raw = raw.get("review") if isinstance(raw, dict) else None

    paper: Dict[str, Any] = paper_raw if isinstance(paper_raw, dict) else {}
    review: Dict[str, Any] = review_raw if isinstance(review_raw, dict) else {}

    core_task = _ensure_str(paper.get("core_task"))
    core_task = _limit_words(core_task, 20)

    contributions = _ensure_str_list(paper.get("contributions"))
    contributions = [c for c in (c.strip() for c in contributions) if c]
    if len(contributions) > 3:
        contributions = contributions[:3]
    contributions = [_limit_words(c, 25) for c in contributions]
    if not contributions:
        # Keep schema stable while making the failure obvious (no new info added).
        contributions = []

    key_terms = _ensure_str_list(paper.get("key_terms"))
    key_terms = _dedupe_preserve_order([t.strip() for t in key_terms if t and t.strip()])
    if len(key_terms) > 12:
        key_terms = key_terms[:12]

    must_have_entities = _ensure_str_list(paper.get("must_have_entities"))
    must_have_entities = _dedupe_preserve_order([e.strip() for e in must_have_entities if e and e.strip()])

    novelty_claims_in = review.get("novelty_claims")
    novelty_claims = _normalize_novelty_claims(
        novelty_claims_in,
        review_text=review_text,
        strict_review_verbatim=strict_review_verbatim,
        logger=log,
    )

    all_citations_raw = _ensure_str_list(review.get("all_citations_raw"))
    all_citations_raw = _dedupe_preserve_order([c.strip() for c in all_citations_raw if c and c.strip()])

    if augment_citations_regex:
        extracted = _extract_citations_regex(review_text)
        all_citations_raw = _dedupe_preserve_order(all_citations_raw + extracted)

    if strict_review_verbatim:
        all_citations_raw = [c for c in all_citations_raw if _in_source(c, review_text)]

    out: Dict[str, Any] = {
        "paper": {
            "core_task": core_task,
            "contributions": contributions,
            "key_terms": key_terms,
            "must_have_entities": must_have_entities,
        },
        "review": {
            "novelty_claims": novelty_claims,
            "all_citations_raw": all_citations_raw,
        },
    }

    _log_schema_warnings(out, log)
    return out


def _prepare_paper_text(text: str, *, max_chars: int) -> str:
    base = sanitize_for_llm(text or "")
    try:
        trimmed = truncate_at_references(base)
        if trimmed:
            base = trimmed
    except Exception:
        pass
    if max_chars > 0 and len(base) > max_chars:
        base = base[:max_chars]
    return base


def _prepare_review_text(text: str, *, max_chars: int) -> str:
    base = sanitize_for_llm(text or "")
    if max_chars > 0 and len(base) > max_chars:
        base = base[:max_chars]
    return base


def _normalize_novelty_claims(
    novelty_claims_in: Any,
    *,
    review_text: str,
    strict_review_verbatim: bool,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    claims_in: List[Any] = novelty_claims_in if isinstance(novelty_claims_in, list) else []

    normalized: List[Dict[str, Any]] = []
    for item in claims_in:
        if not isinstance(item, dict):
            continue
        text = _ensure_str(item.get("text"))
        if not text:
            continue
        text = _limit_sentences(text, 2)
        if strict_review_verbatim and not _in_source(text, review_text):
            continue

        stance = _normalize_enum(
            _ensure_str(item.get("stance")),
            allowed={"not_novel", "somewhat_novel", "novel", "unclear"},
            default="unclear",
        )
        confidence_lang = _normalize_enum(
            _ensure_str(item.get("confidence_lang")),
            allowed={"high", "medium", "low"},
            default="low",
        )
        evidence_expected = _normalize_enum(
            _ensure_str(item.get("evidence_expected")),
            allowed={
                "method_similarity",
                "task_similarity",
                "results_similarity",
                "theory_overlap",
                "dataset_overlap",
            },
            default="method_similarity",
        )

        prior_work_strings = _ensure_str_list(item.get("prior_work_strings"))
        prior_work_strings = _dedupe_preserve_order([s.strip() for s in prior_work_strings if s and s.strip()])
        if strict_review_verbatim:
            prior_work_strings = [s for s in prior_work_strings if _in_source(s, review_text)]

        mentions_prior_work = item.get("mentions_prior_work")
        if isinstance(mentions_prior_work, bool):
            mentions_prior_work_bool = mentions_prior_work
        else:
            mentions_prior_work_bool = bool(prior_work_strings)

        if not mentions_prior_work_bool and prior_work_strings:
            mentions_prior_work_bool = True

        if not mentions_prior_work_bool:
            prior_work_strings = []

        normalized.append(
            {
                "claim_id": "",  # filled later
                "text": text,
                "stance": stance,
                "confidence_lang": confidence_lang,
                "mentions_prior_work": mentions_prior_work_bool,
                "prior_work_strings": prior_work_strings,
                "evidence_expected": evidence_expected,
            }
        )

    # Re-number claim IDs deterministically: C1, C2, ...
    for idx, claim in enumerate(normalized, start=1):
        claim["claim_id"] = f"C{idx}"

    # Warn if we filtered everything (helps users tune prompts)
    if claims_in and not normalized:
        logger.warning(
            "All novelty_claims were filtered out during normalization. "
            "If this is unexpected, consider setting strict_review_verbatim=false."
        )
    return normalized


def _extract_citations_regex(review_text: str) -> List[str]:
    """
    Extract citation-like substrings directly from the review text.

    This is intentionally conservative: it aims to recover obvious IDs/URLs/DOIs
    that an LLM might miss, without inventing anything.
    """
    if not review_text:
        return []

    patterns: Sequence[Tuple[str, int]] = (
        (r"\barXiv:\s*\d{4}\.\d{4,5}\b", re.IGNORECASE),
        (r"\b\d{4}\.\d{4,5}\b", 0),  # bare arXiv id
        (r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE),  # DOI
        (r"https?://[^\s\)\]\}]+", re.IGNORECASE),  # URL (stop at common closers)
        (r"\[[^\]]{1,80}\]", 0),  # bracket citations like [12] or [Smith2020]
    )

    found: List[str] = []
    for pat, flags in patterns:
        try:
            for m in re.finditer(pat, review_text, flags=flags):
                s = (m.group(0) or "").strip()
                if not s:
                    continue
                found.append(s)
        except Exception:
            continue

    return _dedupe_preserve_order(found)


def _in_source(snippet: str, source: str) -> bool:
    """
    Loosely check whether `snippet` appears in `source`, ignoring whitespace runs.
    """
    if not snippet or not source:
        return False
    sn = _normalize_ws(snippet)
    so = _normalize_ws(source)
    if not sn or not so:
        return False
    return sn.lower() in so.lower()


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def _limit_words(s: str, max_words: int) -> str:
    if not s or max_words <= 0:
        return (s or "").strip()
    words = (s or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _limit_sentences(s: str, max_sentences: int) -> str:
    if not s:
        return ""
    if max_sentences <= 0:
        return ""
    # Simple heuristic: split on sentence terminators.
    # Keep terminators by splitting with regex capture.
    parts = re.split(r"([.!?])", s.strip())
    sentences: List[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        current += part
        if part in ".!?":
            sentences.append(current.strip())
            current = ""
        if len(sentences) >= max_sentences:
            break
    if len(sentences) < max_sentences and current.strip():
        sentences.append(current.strip())
    return " ".join(sentences[:max_sentences]).strip()


def _ensure_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _ensure_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item)
        return out
    return []


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_enum(value: str, *, allowed: set, default: str) -> str:
    if value in allowed:
        return value
    lower = (value or "").strip().lower()
    # Accept case-insensitive matches
    for a in allowed:
        if lower == str(a).lower():
            return a
    return default


def _log_schema_warnings(out: Dict[str, Any], log: logging.Logger) -> None:
    try:
        paper = out.get("paper") or {}
        review = out.get("review") or {}
        core_task = (paper.get("core_task") or "").strip()
        if not core_task:
            log.warning("Task1 output: paper.core_task is empty")
        key_terms = paper.get("key_terms") or []
        if isinstance(key_terms, list) and len(key_terms) < 5:
            log.warning("Task1 output: paper.key_terms has <5 items (len=%s)", len(key_terms))
        contributions = paper.get("contributions") or []
        if isinstance(contributions, list) and len(contributions) == 0:
            log.warning("Task1 output: paper.contributions is empty")
        novelty_claims = review.get("novelty_claims") or []
        if isinstance(novelty_claims, list) and len(novelty_claims) == 0:
            log.info("Task1 output: review.novelty_claims is empty")
    except Exception:
        return
