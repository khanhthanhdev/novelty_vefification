#!/usr/bin/env python3
"""
Run Task 1 and Task 2 for all papers in ICLR_2024 that have both paper content and review.

Usage:
    # Process all papers
    python scripts/run_task1_task2_iclr2024.py

    # Process specific papers only
    python scripts/run_task1_task2_iclr2024.py --paper-ids 1BuWv9poWz 1JtTPYBKqt

    # Resume from where it left off (skip existing outputs)
    python scripts/run_task1_task2_iclr2024.py --skip-existing

    # Dry run to see what would be processed
    python scripts/run_task1_task2_iclr2024.py --dry-run
"""

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from novelty_assessment.task1_extractor import extract_task1, Task1ExtractionError
from novelty_assessment.task2_related_works import retrieve_related_works


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# Thread lock for thread-safe logging
_log_lock = threading.Lock()

# Task 2 rate limiting (1.5 seconds between requests)
_task2_lock = threading.Lock()
_task2_last_call = 0.0
_task2_min_interval = 1.5


def log_thread_safe(logger: logging.Logger, level: str, message: str) -> None:
    """Log message in a thread-safe manner."""
    with _log_lock:
        getattr(logger, level)(message)


def rate_limit_task2() -> None:
    """Enforce rate limiting for Task 2: minimum 1.5 seconds between calls."""
    global _task2_last_call
    with _task2_lock:
        now = time.time()
        elapsed = now - _task2_last_call
        if elapsed < _task2_min_interval:
            sleep_time = _task2_min_interval - elapsed
            time.sleep(sleep_time)
        _task2_last_call = time.time()


def find_matching_papers(
    paper_dir: Path,
    review_dir: Path,
) -> List[Tuple[str, Path, Path]]:
    """
    Find all paper IDs that have both paper content (.mmd) and review (.txt).
    
    Returns:
        List of tuples: (paper_id, paper_path, review_path)
    """
    # Get all paper IDs from .mmd files
    paper_files = {f.stem: f for f in paper_dir.glob("*.mmd")}
    
    # Get all review IDs from .txt files
    review_files = {f.stem: f for f in review_dir.glob("*.txt")}
    
    # Find intersection
    common_ids = set(paper_files.keys()) & set(review_files.keys())
    
    # Return sorted list of tuples
    matches = [
        (paper_id, paper_files[paper_id], review_files[paper_id])
        for paper_id in sorted(common_ids)
    ]
    
    return matches


def load_text_file(path: Path) -> str:
    """Load text from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save data to JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_paper(
    *,
    paper_id: str,
    paper_path: Path,
    review_path: Path,
    output_dir: Path,
    paper_year: int = 2024,
    mode: str = "per_contribution",
    skip_existing: bool = False,
    logger: logging.Logger,
) -> Tuple[str, bool, bool]:
    """
    Process a single paper: run Task 1 and Task 2.
    
    Returns:
        (paper_id, task1_success, task2_success)
    """
    task1_output_path = output_dir / paper_id / "task1_result.json"
    task2_output_path = output_dir / paper_id / "task2_result.json"
    
    # Check if we should skip
    if skip_existing and task1_output_path.exists() and task2_output_path.exists():
        log_thread_safe(logger, "info", f"[{paper_id}] Skipping (outputs already exist)")
        return (paper_id, True, True)
    
    log_thread_safe(logger, "info", f"[{paper_id}] Processing paper...")
    
    # Load paper and review text
    try:
        paper_text = load_text_file(paper_path)
        review_text = load_text_file(review_path)
    except Exception as e:
        log_thread_safe(logger, "error", f"[{paper_id}] Failed to load files: {e}")
        return (paper_id, False, False)
    
    log_thread_safe(logger, "info", f"[{paper_id}] Paper: {len(paper_text)} chars, Review: {len(review_text)} chars")
    
    # Task 1: Extraction
    task1_success = False
    task1_result = None
    
    if skip_existing and task1_output_path.exists():
        log_thread_safe(logger, "info", f"[{paper_id}] Task 1 output exists, loading...")
        try:
            with task1_output_path.open("r", encoding="utf-8") as f:
                task1_result = json.load(f)
            task1_success = True
        except Exception as e:
            log_thread_safe(logger, "warning", f"[{paper_id}] Failed to load existing Task 1 output: {e}")
    
    if not task1_success:
        log_thread_safe(logger, "info", f"[{paper_id}] Running Task 1 extraction...")
        try:
            task1_result = extract_task1(
                paper_text=paper_text,
                review_text=review_text,
                paper_title=None,
                strict_review_verbatim=True,
                augment_citations_regex=True,
                logger=logger,
            )
            
            save_json(task1_output_path, task1_result)
            log_thread_safe(logger, "info", f"[{paper_id}] ✓ Task 1 completed, saved to {task1_output_path}")
            
            # Log summary
            paper_data = task1_result.get("paper", {})
            review_data = task1_result.get("review", {})
            log_thread_safe(logger, "info", f"[{paper_id}]   - Core task: {paper_data.get('core_task', 'N/A')}")
            log_thread_safe(logger, "info", f"[{paper_id}]   - Contributions: {len(paper_data.get('contributions', []))}")
            log_thread_safe(logger, "info", f"[{paper_id}]   - Novelty claims: {len(review_data.get('novelty_claims', []))}")
            
            task1_success = True
            
        except Task1ExtractionError as e:
            log_thread_safe(logger, "error", f"[{paper_id}] Task 1 extraction failed: {e}")
            return (paper_id, False, False)
        except Exception as e:
            log_thread_safe(logger, "error", f"[{paper_id}] Task 1 unexpected error: {e}", exc_info=True)
            return (paper_id, False, False)
    
    # Task 2: Related works retrieval
    task2_success = False
    
    if skip_existing and task2_output_path.exists():
        log_thread_safe(logger, "info", f"[{paper_id}] Task 2 output exists, skipping...")
        task2_success = True
    else:
        # Rate limit Task 2 to avoid API throttling (1.5s between requests)
        rate_limit_task2()
        
        log_thread_safe(logger, "info", f"[{paper_id}] Running Task 2 related works retrieval...")
        try:
            task2_result = retrieve_related_works(
                task1_output=task1_result,
                paper_year=paper_year,
                mode=mode,
                limit_per_query=10,
                top_k=30,
                max_total=30,
                dedup_threshold=0.96,
                mmr_lambda=0.7,
                use_cache=False,
                logger=logger,
            )
            
            save_json(task2_output_path, task2_result)
            log_thread_safe(logger, "info", f"[{paper_id}] ✓ Task 2 completed, saved to {task2_output_path}")
            
            # Log summary
            stats = task2_result.get("stats", {})
            log_thread_safe(logger, "info", f"[{paper_id}]   - Queries: {len(task2_result.get('queries', []))}")
            log_thread_safe(logger, "info", f"[{paper_id}]   - Candidates found: {stats.get('final', 0)}")
            
            task2_success = True
            
        except Exception as e:
            log_thread_safe(logger, "error", f"[{paper_id}] Task 2 failed: {e}", exc_info=True)
            return (paper_id, task1_success, False)
    
    return (paper_id, task1_success, task2_success)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 1 and Task 2 for all ICLR 2024 papers with both paper and review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("ICLR_2024/paper_nougat_mmd"),
        help="Directory containing paper .mmd files (default: ICLR_2024/paper_nougat_mmd)",
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("ICLR_2024/review_raw_txt"),
        help="Directory containing review .txt files (default: ICLR_2024/review_raw_txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/iclr2024"),
        help="Output directory for results (default: output/iclr2024)",
    )
    parser.add_argument(
        "--paper-year",
        type=int,
        default=2024,
        help="Paper year for filtering related works (default: 2024)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["per_contribution", "fixed"],
        default="per_contribution",
        help="Task 2 query mode (default: per_contribution)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of papers to process in each batch (default: 10)",
    )
    parser.add_argument(
        "--paper-ids",
        type=str,
        nargs="+",
        help="Process only specific paper IDs (e.g., 1BuWv9poWz 1JtTPYBKqt)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List papers that would be processed without actually processing them",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    log = setup_logging(verbose=args.verbose)
    
    # Find matching papers
    log.info("Scanning directories for matching papers...")
    matches = find_matching_papers(args.paper_dir, args.review_dir)
    
    if not matches:
        log.error("No papers found with both paper content and review!")
        sys.exit(1)
    
    log.info(f"Found {len(matches)} papers with both paper and review content")
    
    # Filter by paper IDs if specified
    if args.paper_ids:
        filter_set = set(args.paper_ids)
        matches = [(pid, pp, rp) for pid, pp, rp in matches if pid in filter_set]
        log.info(f"Filtered to {len(matches)} papers based on --paper-ids")
    
    if not matches:
        log.error("No papers match the criteria!")
        sys.exit(1)
    
    # Dry run: just list papers
    if args.dry_run:
        log.info("DRY RUN: Papers that would be processed:")
        for paper_id, paper_path, review_path in matches:
            log.info(f"  - {paper_id}")
            log.info(f"      Paper: {paper_path}")
            log.info(f"      Review: {review_path}")
        log.info(f"Total: {len(matches)} papers")
        return
    
    # Process all papers in batches with parallel workers
    log.info(f"Starting parallel batch processing of {len(matches)} papers...")
    log.info(f"Batch size: {args.batch_size} papers, Workers per batch: {args.workers}")
    log.info("=" * 80)
    
    results = {
        "total": len(matches),
        "task1_success": 0,
        "task2_success": 0,
        "both_success": 0,
        "failed": [],
    }
    
    # Process papers in batches
    total_completed = 0
    num_batches = (len(matches) + args.batch_size - 1) // args.batch_size  # Ceiling division
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(matches))
        batch = matches[batch_start:batch_end]
        
        log.info(f"\n{'='*80}")
        log.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch)} papers)...")
        log.info(f"{'='*80}")
        
        # Submit batch tasks to thread pool
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Create a mapping of future to paper_id for tracking
            future_to_paper_id = {}
            
            for paper_id, paper_path, review_path in batch:
                future = executor.submit(
                    process_paper,
                    paper_id=paper_id,
                    paper_path=paper_path,
                    review_path=review_path,
                    output_dir=args.output_dir,
                    paper_year=args.paper_year,
                    mode=args.mode,
                    skip_existing=args.skip_existing,
                    logger=log,
                )
                future_to_paper_id[future] = paper_id
            
            # Collect results as they complete
            batch_completed = 0
            for future in as_completed(future_to_paper_id):
                batch_completed += 1
                total_completed += 1
                paper_id, task1_ok, task2_ok = future.result()
                
                if task1_ok:
                    results["task1_success"] += 1
                if task2_ok:
                    results["task2_success"] += 1
                if task1_ok and task2_ok:
                    results["both_success"] += 1
                else:
                    results["failed"].append(paper_id)
                
                # Print progress
                log_thread_safe(
                    log,
                    "info",
                    f"[Batch {batch_idx + 1}/{num_batches} | {batch_completed}/{len(batch)}] "
                    f"[Overall {total_completed}/{len(matches)}] {paper_id}: "
                    f"Task1={'✓' if task1_ok else '✗'} Task2={'✓' if task2_ok else '✗'}",
                )
    
    # Summary
    log.info("\n" + "=" * 80)
    log.info("BATCH PROCESSING COMPLETE")
    log.info(f"Total papers: {results['total']}")
    log.info(f"Task 1 success: {results['task1_success']}/{results['total']}")
    log.info(f"Task 2 success: {results['task2_success']}/{results['total']}")
    log.info(f"Both success: {results['both_success']}/{results['total']}")
    
    if results["failed"]:
        log.info(f"\nFailed papers ({len(results['failed'])}):")
        for paper_id in results["failed"]:
            log.info(f"  - {paper_id}")
    
    # Save summary
    summary_path = args.output_dir / "batch_summary.json"
    save_json(summary_path, results)
    log.info(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
