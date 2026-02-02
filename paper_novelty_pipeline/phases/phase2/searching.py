"""
Phase 2: Paper Searching.

This module handles the academic paper search phase of the novelty analysis pipeline.

Responsibilities:
  1. Prepare search queries from Phase 1 extracted content (core_task + contributions)
  2. Execute concurrent searches via Semantic Scholar API
  3. Save raw API responses to phase2/raw_responses/
  4. Return execution statistics
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, Optional, List

from paper_novelty_pipeline.config import PHASE2_SEMANTIC_SCHOLAR_LIMIT
from paper_novelty_pipeline.models import ExtractedContent, ContributionClaim
from paper_novelty_pipeline.services.semantic_scholar_client import SemanticScholarClient
from paper_novelty_pipeline.utils.text_cleaning import sanitize_unicode


logger = logging.getLogger(__name__)

DEFAULT_FIELDS = (
    "title,abstract,year,venue,authors,externalIds,url,openAccessPdf,"
    "citationCount,influentialCitationCount"
)


@dataclass
class QuerySpec:
    scope: str
    query: str
    index: int
    source: str


def _first_non_empty(values: List[Optional[str]]) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _normalize_meta_text(title: str, abstract: str) -> str:
    base = f"{title} {abstract}".lower()
    base = re.sub(r"[^a-z0-9 ]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def _score_value(paper: Dict[str, Any]) -> float:
    for key in ("relevance_score", "score"):
        val = paper.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def _dedup_approx(
    papers: List[Dict[str, Any]],
    *,
    threshold: float = 0.96,
) -> List[Dict[str, Any]]:
    if len(papers) <= 1:
        return papers

    kept: List[Dict[str, Any]] = []
    kept_norms: List[str] = []

    for paper in papers:
        norm = _normalize_meta_text(paper.get("title", ""), paper.get("abstract", ""))
        dup_idx = None
        for i, existing in enumerate(kept_norms):
            if not norm or not existing:
                continue
            if norm == existing:
                dup_idx = i
                break
            if SequenceMatcher(None, norm, existing).ratio() >= threshold:
                dup_idx = i
                break
        if dup_idx is None:
            kept.append(paper)
            kept_norms.append(norm)
        else:
            if _score_value(paper) > _score_value(kept[dup_idx]):
                kept[dup_idx] = paper
                kept_norms[dup_idx] = norm

    return kept


def _normalize_semantic_scholar_item(
    item: Dict[str, Any],
    rank: int,
    total: int,
) -> Dict[str, Any]:
    external_ids = item.get("externalIds") or {}
    doi = external_ids.get("DOI") or external_ids.get("doi")
    arxiv_id = (
        external_ids.get("ArXiv")
        or external_ids.get("arXiv")
        or external_ids.get("arxiv")
    )

    authors: List[str] = []
    for author in item.get("authors") or []:
        if isinstance(author, dict):
            name = author.get("name")
            if name:
                authors.append(str(name))

    year_val = item.get("year")
    year = None
    if year_val is not None:
        try:
            year = int(year_val)
        except (TypeError, ValueError):
            year = None

    open_access = item.get("openAccessPdf") or {}
    pdf_url = open_access.get("url") if isinstance(open_access, dict) else None

    title = sanitize_unicode(item.get("title") or "")
    abstract = sanitize_unicode(item.get("abstract") or "")
    score = item.get("score")
    if score is None:
        score = (float(total - rank) / float(max(total, 1)))

    venue = item.get("venue") or ""
    publication_venue = item.get("publicationVenue")
    if not venue and isinstance(publication_venue, dict):
        venue = publication_venue.get("name") or ""

    return {
        "paper_id": item.get("paperId"),
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "venue": venue,
        "year": year,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "relevance_score": score,
        "pdf_url": pdf_url,
        "source_url": item.get("url"),
        "raw_metadata": item,
        "flags": {"perfect": True, "partial": False, "no": False},
    }


class PaperSearcher:
    """Phase 2: Search for related papers via Semantic Scholar API."""

    def __init__(self, concurrency: Optional[int] = None, limit_per_query: Optional[int] = None):
        self.concurrency = max(1, int(concurrency)) if concurrency else 1
        self.limit_per_query = int(limit_per_query or PHASE2_SEMANTIC_SCHOLAR_LIMIT)
        self.client = SemanticScholarClient()

    def search_all(
        self,
        extracted: ExtractedContent,
        out_dir: Path,
    ) -> Dict[str, Any]:
        """Execute all searches for a paper."""
        raw_dir = Path(out_dir) / "raw_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)

        query_specs = self._build_query_specs(extracted)
        stats = {
            "total_queries": len(query_specs),
            "succeeded": 0,
            "failed": 0,
            "queries": [],
        }

        if not query_specs:
            logger.warning("Phase2: no valid queries built from extracted content.")
            return stats

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {
                executor.submit(self._run_query, spec, raw_dir): spec for spec in query_specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                    stats["queries"].append(result)
                    if result.get("status") == "ok":
                        stats["succeeded"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as exc:
                    stats["failed"] += 1
                    stats["queries"].append(
                        {
                            "scope": spec.scope,
                            "query": spec.query,
                            "status": "error",
                            "error": str(exc),
                        }
                    )

        return stats

    def _build_query_specs(self, extracted: ExtractedContent) -> List[QuerySpec]:
        specs: List[QuerySpec] = []

        core_task_text = ""
        if extracted.core_task:
            if extracted.core_task.query_variants:
                core_task_text = _first_non_empty(extracted.core_task.query_variants)
            if not core_task_text:
                core_task_text = extracted.core_task.text.strip()

        if core_task_text:
            specs.append(QuerySpec(scope="core_task", query=core_task_text, index=1, source="core_task"))

        for idx, contrib in enumerate(extracted.contributions or [], start=1):
            scope = contrib.id or f"contribution_{idx}"
            contrib_query = self._build_contribution_query(core_task_text, contrib)
            if not contrib_query:
                continue
            specs.append(
                QuerySpec(
                    scope=scope,
                    query=contrib_query,
                    index=1,
                    source="contribution",
                )
            )

        return specs

    def _build_contribution_query(self, core_task: str, contrib: ContributionClaim) -> str:
        base = _first_non_empty(
            [
                contrib.prior_work_query,
                *(contrib.query_variants or []),
                contrib.description,
                contrib.author_claim_text,
                contrib.name,
            ]
        )
        if not base:
            return ""
        if core_task:
            return f"{core_task} {base}".strip()
        return base

    def _run_query(self, spec: QuerySpec, raw_dir: Path) -> Dict[str, Any]:
        start = time.time()
        response = self.client.search(
            query=spec.query,
            limit=self.limit_per_query,
            offset=0,
            fields=DEFAULT_FIELDS,
        )

        raw_items = response.get("data") or []
        papers = [
            _normalize_semantic_scholar_item(item, rank=idx, total=len(raw_items))
            for idx, item in enumerate(raw_items)
        ]
        papers = _dedup_approx(papers)

        payload = {
            "backend": "semantic_scholar",
            "scope": spec.scope,
            "query": spec.query,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "total": response.get("total"),
            "papers": papers,
            "response_meta": {
                "offset": response.get("offset"),
                "next": response.get("next"),
            },
        }

        raw_path = raw_dir / f"raw_{spec.scope}_{spec.index}.json"
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return {
            "scope": spec.scope,
            "query": spec.query,
            "status": "ok",
            "count": len(papers),
            "raw_path": str(raw_path),
            "duration_s": round(time.time() - start, 2),
        }


def run_phase2_search(
    extracted: ExtractedContent,
    out_dir: Path,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Phase2 search (API calls only)."""
    searcher = PaperSearcher(concurrency=concurrency)
    return searcher.search_all(extracted, out_dir)
