#!/usr/bin/env python3
"""
Run Task 2 related-works retrieval (Semantic Scholar only).

Usage:
  # From Task 1 output
  python scripts/run_task2.py --task1 output/task1_result.json --output task2_result.json

  # Optional paper year filter and fixed query mode
  python scripts/run_task2.py --task1 output/task1_result.json --paper-year 2024 --mode fixed -o task2.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from novelty_assessment.task2_related_works import retrieve_related_works


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_task1(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Task1 JSON must be an object")
    if "paper" not in data:
        raise ValueError("Task1 JSON missing 'paper' section")
    return data


def save_output(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 2 related-works retrieval (Semantic Scholar)."
    )
    parser.add_argument(
        "--task1",
        type=Path,
        required=True,
        help="Path to Task 1 output JSON",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path (prints to stdout if not set)",
    )
    parser.add_argument(
        "--paper-year",
        type=int,
        help="Filter out candidates published after this year",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["per_contribution", "fixed"],
        default="per_contribution",
        help="Query mode: per_contribution (default) or fixed (Q1/Q2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Per-query Semantic Scholar limit (default: 10)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Diversified candidate pool size (default: 30)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=30,
        help="Cap final candidate count (default: 30)",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.96,
        help="Approx dedup threshold (default: 0.96)",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="MMR relevance/diversity tradeoff (default: 0.7)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable on-disk caching of Task2 results",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory (default: output/task2_cache)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    log = setup_logging(verbose=args.verbose)

    task1 = load_task1(args.task1)

    result = retrieve_related_works(
        task1,
        paper_year=args.paper_year,
        mode=args.mode,
        limit_per_query=args.limit,
        top_k=args.top_k,
        max_total=args.max_total,
        dedup_threshold=args.dedup_threshold,
        mmr_lambda=args.mmr_lambda,
        use_cache=args.cache,
        cache_dir=args.cache_dir,
        logger=log,
    )

    if args.output:
        save_output(args.output, result)
        log.info("✓ Output saved to: %s", args.output)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
