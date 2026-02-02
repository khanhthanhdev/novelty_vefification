#!/usr/bin/env python3
"""
Run Task 1 extraction (paper + review → structured output) for a single paper.

Usage:
    # From paper text file and review text file
    python scripts/run_task1.py --paper paper.txt --review review.txt --output result.json

    # From JSON input
    python scripts/run_task1.py --input input.json --output result.json

    # With optional paper title
    python scripts/run_task1.py --paper paper.txt --review review.txt --title "My Paper" --output result.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path to allow imports
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
        "paper_title": "..." (optional)
    }
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object/dict")
    if "paper_text" not in data:
        raise ValueError("Input JSON missing required field: paper_text")
    if "review_text" not in data:
        raise ValueError("Input JSON missing required field: review_text")

    return data


def load_text_file(path: Path) -> str:
    """Load text from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_output_json(output_path: Path, data: Dict[str, Any]) -> None:
    """Save output to JSON file with pretty formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_task1_extraction(
    *,
    paper_text: str,
    review_text: str,
    paper_title: Optional[str] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
    strict_review_verbatim: bool = True,
    augment_citations_regex: bool = True,
) -> Dict[str, Any]:
    """
    Run Task 1 extraction and optionally save to file.

    Returns:
        The extracted data as a dict with keys: "paper", "review"
    """
    log = setup_logging(verbose=verbose)
    log.info("Starting Task 1 extraction...")

    if paper_title:
        log.info(f"Paper title: {paper_title}")
    log.info(f"Paper text length: {len(paper_text)} chars")
    log.info(f"Review text length: {len(review_text)} chars")

    try:
        result = extract_task1(
            paper_text=paper_text,
            review_text=review_text,
            paper_title=paper_title,
            strict_review_verbatim=strict_review_verbatim,
            augment_citations_regex=augment_citations_regex,
            logger=log,
        )

        log.info("✓ Task 1 extraction completed successfully")

        # Log summary
        paper_data = result.get("paper", {})
        review_data = result.get("review", {})
        log.info(f"  Core task: {paper_data.get('core_task', 'N/A')}")
        log.info(f"  Contributions: {len(paper_data.get('contributions', []))}")
        log.info(f"  Key terms: {len(paper_data.get('key_terms', []))}")
        log.info(f"  Novelty claims: {len(review_data.get('novelty_claims', []))}")
        log.info(f"  Citations found: {len(review_data.get('all_citations_raw', []))}")

        if output_path:
            save_output_json(output_path, result)
            log.info(f"✓ Output saved to: {output_path}")

        return result

    except Task1ExtractionError as e:
        log.error(f"Task 1 extraction failed: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run Task 1 extraction for a single paper + review pair.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From separate text files
  python scripts/run_task1.py --paper paper.txt --review review.txt --output result.json

  # From JSON input file
  python scripts/run_task1.py --input input.json --output result.json

  # With paper title
  python scripts/run_task1.py --paper paper.txt --review review.txt --title "My Paper" -o result.json

  # Print to stdout (no output file)
  python scripts/run_task1.py --paper paper.txt --review review.txt

Input JSON format:
  {
    "paper_text": "full paper text...",
    "review_text": "review text...",
    "paper_title": "optional title"
  }

Output JSON format:
  {
    "paper": {
      "core_task": "...",
      "contributions": ["...", "...", "..."],
      "key_terms": ["...", ...],
      "must_have_entities": ["...", ...]
    },
    "review": {
      "novelty_claims": [
        {
          "claim_id": "C1",
          "text": "...",
          "stance": "not_novel | somewhat_novel | novel | unclear",
          "confidence_lang": "high | medium | low",
          "mentions_prior_work": true/false,
          "prior_work_strings": ["...", ...],
          "evidence_expected": "method_similarity | ..."
        }
      ],
      "all_citations_raw": ["...", ...]
    }
  }
        """,
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=Path,
        help="Path to input JSON file containing paper_text, review_text, and optionally paper_title",
    )
    input_group.add_argument(
        "--paper",
        type=Path,
        help="Path to paper text file (use with --review)",
    )

    parser.add_argument(
        "--review",
        type=Path,
        help="Path to review text file (required when using --paper)",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Paper title (optional, only used with --paper/--review)",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to output JSON file (prints to stdout if not specified)",
    )

    # Extraction options
    parser.add_argument(
        "--no-strict-verbatim",
        action="store_true",
        help="Disable strict review verbatim checking (allows more flexible matching)",
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

    # Validate input combinations
    if args.paper and not args.review:
        parser.error("--paper requires --review")
    if args.review and not args.paper:
        parser.error("--review requires --paper")
    if args.title and not args.paper:
        parser.error("--title can only be used with --paper/--review")

    # Load input data
    if args.input:
        # Load from JSON
        input_data = load_input_json(args.input)
        paper_text = input_data["paper_text"]
        review_text = input_data["review_text"]
        paper_title = input_data.get("paper_title")
    else:
        # Load from separate files
        paper_text = load_text_file(args.paper)
        review_text = load_text_file(args.review)
        paper_title = args.title

    # Run extraction
    result = run_task1_extraction(
        paper_text=paper_text,
        review_text=review_text,
        paper_title=paper_title,
        output_path=args.output,
        verbose=args.verbose,
        strict_review_verbatim=not args.no_strict_verbatim,
        augment_citations_regex=not args.no_augment_citations,
    )

    # If no output file specified, print to stdout
    if not args.output:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
