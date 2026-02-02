#!/usr/bin/env python3
"""
Run Phase 1 and Phase 2 continuously for a list of papers.

This script performs, for each paper:
  1) Phase 1 extraction (writes to <paper_dir>/phase1)
  2) Phase 2 search + postprocess (writes to <paper_dir>/phase2)
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from paper_novelty_pipeline.entrypoints import run_phase1_only, run_phase2_full
from paper_novelty_pipeline.config import PHASE2_QUERY_CONCURRENCY
from paper_novelty_pipeline.utils.paths import safe_dir_name


def _read_papers_from_file(path: Path) -> List[str]:
    papers: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        papers.append(line)
    return papers


def _collect_papers(args: argparse.Namespace) -> List[str]:
    papers: List[str] = []
    if args.papers:
        papers.extend(args.papers)
    if args.paper_file:
        papers.extend(_read_papers_from_file(Path(args.paper_file)))

    if papers:
        return papers

    # Interactive fallback
    print("Enter paper ids/URLs (empty line to finish):")
    while True:
        try:
            line = input("paper: ").strip()
        except EOFError:
            break
        if not line:
            break
        papers.append(line)
    return papers


def _make_paper_root(out_root: Path, run_prefix: str, paper_id: str, date_str: str) -> Path:
    base_root = out_root / run_prefix if run_prefix else out_root
    base_root.mkdir(parents=True, exist_ok=True)
    safe_name = safe_dir_name(paper_id)
    return base_root / f"{safe_name}_{date_str}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 then Phase 2 continuously for a list of papers."
    )
    parser.add_argument("--papers", nargs="+", help="Paper ids or URLs")
    parser.add_argument("--paper-file", type=str, help="File with one paper id/URL per line")
    parser.add_argument("--out-root", type=str, default="output", help="Root output directory")
    parser.add_argument("--run-prefix", type=str, default="", help="Optional run prefix under out-root")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Override date suffix (YYYYMMDD). Default: today.",
    )
    parser.add_argument(
        "--phase2-concurrency",
        type=int,
        default=int(PHASE2_QUERY_CONCURRENCY),
        help="Phase2 concurrent queries (threads)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have phase2/final/citation_index.json",
    )
    args = parser.parse_args()

    papers = _collect_papers(args)
    if not papers:
        print("No papers provided.")
        raise SystemExit(1)

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    out_root = Path(args.out_root)

    log = logging.getLogger("run_phase1_phase2")
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    for paper_id in papers:
        paper_root = _make_paper_root(out_root, args.run_prefix, paper_id, date_str)
        phase1_dir = paper_root / "phase1"
        phase2_dir = paper_root / "phase2"
        phase1_done = (phase1_dir / "phase1_extracted.json").exists()
        phase2_done = (phase2_dir / "final" / "citation_index.json").exists()

        if args.skip_existing and phase2_done:
            log.info(f"[skip] Phase2 exists: {paper_root}")
            continue

        if not phase1_done:
            phase1_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"[Phase1] {paper_id} -> {phase1_dir}")
            rc1 = run_phase1_only(
                paper_url=paper_id,
                out_dir=str(phase1_dir),
                log_level=args.log_level,
            )
            if rc1 != 0:
                log.error(f"[Phase1] failed: {paper_id}")
                continue
        else:
            log.info(f"[Phase1] already done: {phase1_dir}")

        log.info(f"[Phase2] {paper_id} -> {paper_root}")
        rc2 = run_phase2_full(
            phase1_dir=phase1_dir,
            out_dir=paper_root,
            concurrency=int(args.phase2_concurrency),
            log_level=args.log_level,
        )
        if rc2 != 0:
            log.error(f"[Phase2] failed: {paper_id}")

    print("Done.")


if __name__ == "__main__":
    main()
