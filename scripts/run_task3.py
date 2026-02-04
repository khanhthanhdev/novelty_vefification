#!/usr/bin/env python3
"""
Run Task 3 review-claim verification (LLM Judge).

Usage:
  python scripts/run_task3.py --task1 output/task1_result.json --task2 task2_result.json \
    --paper ICLR_2024/paper_nougat_mmd/XXXX.mmd --output task3_result.json

  python scripts/run_task3.py --task1 output/task1_result.json --task2 task2_result.json \
    --paper-context abstract_intro.txt --output task3_result.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from novelty_assessment.task3_judge import (
    extract_abstract_intro_from_text,
    run_task3_verification,
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    return data


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_output(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_review_sentences(task1: Dict[str, Any]) -> List[Dict[str, str]]:
    review = task1.get("review") or {}
    claims = review.get("novelty_claims") or []
    sentences: List[Dict[str, str]] = []
    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        text = (claim.get("text") or "").strip()
        if not text:
            continue
        sentence_id = claim.get("claim_id") or f"S_{idx + 1:03d}"
        sentences.append({"review_sentence_id": str(sentence_id), "text": text})
    return sentences


def extract_related_works(task2: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("candidate_pool_top30", "candidate_pool_topN", "candidate_pool", "candidates"):
        value = task2.get(key)
        if isinstance(value, list):
            return value
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 3 review-claim verification (LLM Judge)."
    )
    parser.add_argument(
        "--task1",
        type=Path,
        required=True,
        help="Path to Task 1 output JSON",
    )
    parser.add_argument(
        "--task2",
        type=Path,
        required=True,
        help="Path to Task 2 output JSON",
    )
    paper_group = parser.add_mutually_exclusive_group(required=True)
    paper_group.add_argument(
        "--paper",
        type=Path,
        help="Full paper text file (used to extract Abstract + Introduction)",
    )
    paper_group.add_argument(
        "--paper-context",
        type=Path,
        help="Pre-extracted Abstract + Introduction text file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path (prints to stdout if not set)",
    )
    parser.add_argument(
        "--max-review-chars",
        type=int,
        default=1500,
        help="Max chars for each review sentence (default: 1500)",
    )
    parser.add_argument(
        "--max-paper-chars",
        type=int,
        default=12000,
        help="Max chars for paper context (default: 12000)",
    )
    parser.add_argument(
        "--max-related-chars",
        type=int,
        default=8000,
        help="Max chars for related work text (default: 8000)",
    )
    parser.add_argument(
        "--max-related-per-sentence",
        type=int,
        help="Limit related works per review sentence",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Cap total judge pairs",
    )
    parser.add_argument(
        "--aggregate-policy",
        type=str,
        choices=["max", "mean", "weighted"],
        default="max",
        help="Aggregation policy for final_score (default: max)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable LLM cache (if supported by provider)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=str,
        default="1h",
        help="LLM cache TTL (default: 1h)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Max tokens for Judge response (default: 800)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        help="LLM provider (openai, openrouter, azure)",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        help="LLM API key",
    )
    parser.add_argument(
        "--llm-api-endpoint",
        type=str,
        help="LLM API endpoint URL",
    )
    parser.add_argument(
        "--llm-model-name",
        type=str,
        help="LLM model name",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider
    if args.llm_api_key:
        os.environ["LLM_API_KEY"] = args.llm_api_key
    if args.llm_api_endpoint:
        os.environ["LLM_API_ENDPOINT"] = args.llm_api_endpoint
    if args.llm_model_name:
        os.environ["LLM_MODEL_NAME"] = args.llm_model_name

    log = setup_logging(verbose=args.verbose)

    task1 = load_json(args.task1)
    task2 = load_json(args.task2)

    review_sentences = extract_review_sentences(task1)
    if not review_sentences:
        log.warning("No novelty claims found in Task 1 output.")

    related_works = extract_related_works(task2)
    if not related_works:
        log.warning("No related works found in Task 2 output.")

    if args.paper_context:
        paper_context = load_text(args.paper_context)
    else:
        paper_text = load_text(args.paper)
        paper_context = extract_abstract_intro_from_text(
            paper_text,
            max_chars=args.max_paper_chars,
        )

    result = run_task3_verification(
        review_sentences=review_sentences,
        paper_context=paper_context,
        related_works=related_works,
        max_review_chars=args.max_review_chars,
        max_paper_chars=args.max_paper_chars,
        max_related_chars=args.max_related_chars,
        max_related_per_sentence=args.max_related_per_sentence,
        max_pairs=args.max_pairs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_cache=args.cache,
        cache_ttl=args.cache_ttl,
        aggregate_policy=args.aggregate_policy,
        logger=log,
    )

    if args.output:
        save_output(args.output, result)
        log.info("✓ Output saved to: %s", args.output)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
