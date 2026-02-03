#!/bin/bash
# Run Task 1 and Task 2 for all ICLR 2024 papers with both paper and review content.
#
# Usage:
#   bash scripts/run_iclr2024_batch.sh
#   bash scripts/run_iclr2024_batch.sh --skip-existing
#   bash scripts/run_iclr2024_batch.sh --paper-ids "1BuWv9poWz 1JtTPYBKqt"

set -e

# Default settings
PAPER_DIR="ICLR_2024/paper_nougat_mmd"
REVIEW_DIR="ICLR_2024/review_raw_txt"
OUTPUT_DIR="output/iclr2024"
PAPER_YEAR=2024
MODE="per_contribution"
SKIP_EXISTING=""
PAPER_IDS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-existing)
            SKIP_EXISTING="--skip-existing"
            shift
            ;;
        --paper-ids)
            PAPER_IDS="--paper-ids $2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --paper-year)
            PAPER_YEAR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-existing] [--paper-ids \"id1 id2 ...\"] [--mode per_contribution|fixed] [--paper-year YEAR] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ICLR 2024 Batch Processing"
echo "=========================================="
echo "Paper directory: $PAPER_DIR"
echo "Review directory: $REVIEW_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Paper year: $PAPER_YEAR"
echo "Query mode: $MODE"
echo ""

# Run the Python batch script
python scripts/run_task1_task2_iclr2024.py \
    --paper-dir "$PAPER_DIR" \
    --review-dir "$REVIEW_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --paper-year "$PAPER_YEAR" \
    --mode "$MODE" \
    $SKIP_EXISTING \
    $PAPER_IDS

echo ""
echo "=========================================="
echo "Batch processing complete!"
echo "Check outputs in: $OUTPUT_DIR"
echo "=========================================="
