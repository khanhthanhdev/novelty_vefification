#!/usr/bin/env python3
"""
Run Task 1 extraction in batch mode for multiple papers.

Usage:
    # Process multiple input files
    python scripts/run_task1_batch.py --inputs input1.json input2.json input3.json --output-dir results/

    # Process from a list file
    python scripts/run_task1_batch.py --input-list papers.txt --output-dir results/

    # With parallel processing
    python scripts/run_task1_batch.py --inputs *.json --output-dir results/ --workers 4
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from novelty_assessment.task1_extractor import extract_task1, Task1ExtractionError


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_input_json(input_path: Path) -> Dict[str, Any]:
    """
    Load input from JSON file.

    Expected format:
    {
        "paper_text": "...",
        "review_text": "...",
        "paper_title": "..." (optional),
        "paper_id": "..." (optional, for naming)
    }
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{input_path}: Input JSON must be an object/dict")
    if "paper_text" not in data:
        raise ValueError(f"{input_path}: Missing required field: paper_text")
    if "review_text" not in data:
        raise ValueError(f"{input_path}: Missing required field: review_text")

    return data


def save_output_json(output_path: Path, data: Dict[str, Any]) -> None:
    """Save output to JSON file with pretty formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_single_paper(
    input_path: Path,
    output_dir: Path,
    strict_review_verbatim: bool,
    augment_citations_regex: bool,
    log: logging.Logger,
) -> Tuple[Path, bool, Optional[str]]:
    """
    Process a single paper.

    Returns:
        (input_path, success, error_message)
    """
    try:
        log.info(f"Processing: {input_path}")

        # Load input
        input_data = load_input_json(input_path)
        paper_text = input_data["paper_text"]
        review_text = input_data["review_text"]
        paper_title = input_data.get("paper_title")
        paper_id = input_data.get("paper_id", input_path.stem)

        # Run extraction
        result = extract_task1(
            paper_text=paper_text,
            review_text=review_text,
            paper_title=paper_title,
            strict_review_verbatim=strict_review_verbatim,
            augment_citations_regex=augment_citations_regex,
            logger=log,
        )

        # Save output
        output_path = output_dir / f"{paper_id}_task1.json"
        save_output_json(output_path, result)

        # Log summary
        paper_data = result.get("paper", {})
        review_data = result.get("review", {})
        log.info(
            f"✓ {input_path.name}: "
            f"{len(paper_data.get('contributions', []))} contributions, "
            f"{len(review_data.get('novelty_claims', []))} novelty claims"
        )

        return (input_path, True, None)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        log.error(f"✗ {input_path.name}: {error_msg}")
        return (input_path, False, error_msg)


def run_batch_extraction(
    input_paths: List[Path],
    output_dir: Path,
    workers: int = 1,
    verbose: bool = False,
    strict_review_verbatim: bool = True,
    augment_citations_regex: bool = True,
) -> Dict[str, Any]:
    """
    Run Task 1 extraction on multiple papers.

    Returns:
        Summary dict with statistics and results
    """
    log = setup_logging(verbose=verbose)
    log.info(f"Starting batch processing of {len(input_paths)} papers")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Workers: {workers}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    success_count = 0
    error_count = 0

    if workers > 1:
        # Parallel processing
        log.info("Using parallel processing...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_single_paper,
                    input_path,
                    output_dir,
                    strict_review_verbatim,
                    augment_citations_regex,
                    log,
                ): input_path
                for input_path in input_paths
            }

            for future in as_completed(futures):
                input_path, success, error = future.result()
                results.append(
                    {
                        "input": str(input_path),
                        "success": success,
                        "error": error,
                    }
                )
                if success:
                    success_count += 1
                else:
                    error_count += 1
    else:
        # Sequential processing
        log.info("Using sequential processing...")
        for input_path in input_paths:
            input_path_obj, success, error = process_single_paper(
                input_path,
                output_dir,
                strict_review_verbatim,
                augment_citations_regex,
                log,
            )
            results.append(
                {
                    "input": str(input_path_obj),
                    "success": success,
                    "error": error,
                }
            )
            if success:
                success_count += 1
            else:
                error_count += 1

    # Save summary
    summary = {
        "total": len(input_paths),
        "success": success_count,
        "errors": error_count,
        "results": results,
    }

    summary_path = output_dir / "_batch_summary.json"
    save_output_json(summary_path, summary)

    log.info("=" * 60)
    log.info(f"Batch processing complete!")
    log.info(f"  Total: {len(input_paths)}")
    log.info(f"  Success: {success_count}")
    log.info(f"  Errors: {error_count}")
    log.info(f"  Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run Task 1 extraction in batch mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process multiple input files
  python scripts/run_task1_batch.py --inputs input1.json input2.json --output-dir results/

  # Process all JSON files in a directory
  python scripts/run_task1_batch.py --inputs data/*.json --output-dir results/

  # Process from a list file (one path per line)
  python scripts/run_task1_batch.py --input-list papers.txt --output-dir results/

  # With parallel processing (4 workers)
  python scripts/run_task1_batch.py --inputs *.json --output-dir results/ --workers 4

Input JSON format (same as run_task1.py):
  {
    "paper_text": "full paper text...",
    "review_text": "review text...",
    "paper_title": "optional title",
    "paper_id": "optional_id_for_naming"
  }

Output:
  - results/<paper_id>_task1.json (one per paper)
  - results/_batch_summary.json (overall summary)
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        help="List of input JSON files",
    )
    input_group.add_argument(
        "--input-list",
        type=Path,
        help="Text file containing paths to input JSON files (one per line)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output files",
    )

    # Processing options
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential processing)",
    )
    parser.add_argument(
        "--no-strict-verbatim",
        action="store_true",
        help="Disable strict review verbatim checking",
    )
    parser.add_argument(
        "--no-augment-citations",
        action="store_true",
        help="Disable regex-based citation augmentation",
    )

    # Other options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Load input paths
    if args.inputs:
        input_paths = [p.resolve() for p in args.inputs if p.exists()]
        if len(input_paths) < len(args.inputs):
            missing = len(args.inputs) - len(input_paths)
            print(f"Warning: {missing} input file(s) not found", file=sys.stderr)
    else:
        # Load from list file
        with open(args.input_list, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        input_paths = [Path(line).resolve() for line in lines if Path(line).exists()]
        if len(input_paths) < len(lines):
            missing = len(lines) - len(input_paths)
            print(f"Warning: {missing} file(s) from list not found", file=sys.stderr)

    if not input_paths:
        print("Error: No valid input files found", file=sys.stderr)
        sys.exit(1)

    # Run batch extraction
    summary = run_batch_extraction(
        input_paths=input_paths,
        output_dir=args.output_dir.resolve(),
        workers=args.workers,
        verbose=args.verbose,
        strict_review_verbatim=not args.no_strict_verbatim,
        augment_citations_regex=not args.no_augment_citations,
    )

    # Exit with error code if any failures
    if summary["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
