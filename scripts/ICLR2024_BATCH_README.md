# ICLR 2024 Batch Processing

This directory contains scripts for batch processing all ICLR 2024 papers that have both paper content and review data.

## Data Structure

- **Papers**: `ICLR_2024/paper_nougat_mmd/*.mmd` - Paper content in Markdown format
- **Reviews**: `ICLR_2024/review_raw_txt/*.txt` - Review text files
- **Outputs**: `output/iclr2024/<paper_id>/` - Task 1 and Task 2 results per paper

## Available Scripts

### 1. Python Batch Script (Recommended)

The main Python script with full control and logging:

```bash
python scripts/run_task1_task2_iclr2024.py
```

**Features:**
- Automatically finds papers with both paper content and review
- Runs Task 1 (extraction) and Task 2 (related works retrieval) for each paper
- Saves results to `output/iclr2024/<paper_id>/task1_result.json` and `task2_result.json`
- Provides detailed logging and error handling
- Supports resume functionality with `--skip-existing`

**Options:**

```bash
# Process all matching papers
python scripts/run_task1_task2_iclr2024.py

# Skip papers that already have outputs (resume)
python scripts/run_task1_task2_iclr2024.py --skip-existing

# Process specific papers only
python scripts/run_task1_task2_iclr2024.py --paper-ids 1BuWv9poWz 1JtTPYBKqt

# Use fixed query mode for Task 2
python scripts/run_task1_task2_iclr2024.py --mode fixed

# Custom output directory
python scripts/run_task1_task2_iclr2024.py --output-dir output/my_run

# Dry run (see what would be processed)
python scripts/run_task1_task2_iclr2024.py --dry-run

# Verbose logging
python scripts/run_task1_task2_iclr2024.py -v
```

**Full Usage:**

```
usage: run_task1_task2_iclr2024.py [-h] [--paper-dir PAPER_DIR]
                                    [--review-dir REVIEW_DIR]
                                    [--output-dir OUTPUT_DIR]
                                    [--paper-year PAPER_YEAR]
                                    [--mode {per_contribution,fixed}]
                                    [--paper-ids PAPER_IDS [PAPER_IDS ...]]
                                    [--skip-existing] [--dry-run] [-v]

Options:
  --paper-dir PAPER_DIR      Directory with .mmd files (default: ICLR_2024/paper_nougat_mmd)
  --review-dir REVIEW_DIR    Directory with .txt files (default: ICLR_2024/review_raw_txt)
  --output-dir OUTPUT_DIR    Output directory (default: output/iclr2024)
  --paper-year PAPER_YEAR    Filter related works by year (default: 2024)
  --mode {per_contribution,fixed}
                             Task 2 query mode (default: per_contribution)
  --paper-ids PAPER_IDS [PAPER_IDS ...]
                             Process only specific paper IDs
  --skip-existing            Skip papers with existing outputs
  --dry-run                  List papers without processing
  -v, --verbose              Enable verbose logging
```

### 2. Shell Wrapper Script

Simple bash wrapper for common use cases:

```bash
bash scripts/run_iclr2024_batch.sh
```

**Options:**

```bash
# Skip existing outputs
bash scripts/run_iclr2024_batch.sh --skip-existing

# Process specific papers
bash scripts/run_iclr2024_batch.sh --paper-ids "1BuWv9poWz 1JtTPYBKqt"

# Use fixed query mode
bash scripts/run_iclr2024_batch.sh --mode fixed

# Custom settings
bash scripts/run_iclr2024_batch.sh --output-dir output/my_run --paper-year 2024
```

## Output Structure

For each processed paper, the following files are created:

```
output/iclr2024/
├── <paper_id_1>/
│   ├── task1_result.json    # Extracted paper+review structured data
│   └── task2_result.json    # Related works candidates
├── <paper_id_2>/
│   ├── task1_result.json
│   └── task2_result.json
├── ...
└── batch_summary.json       # Overall batch processing summary
```

### Task 1 Output Schema

```json
{
  "paper": {
    "core_task": "string",
    "contributions": [
      {
        "name": "string",
        "author_claim_text": "string",
        "description": "string",
        "source_hint": "string"
      }
    ],
    "key_terms": ["string", ...],
    "must_have_entities": ["string", ...]
  },
  "review": {
    "novelty_claims": [
      {
        "claim_id": "string",
        "text": "string",
        "stance": "not_novel | somewhat_novel | novel | unclear",
        "confidence_lang": "high | medium | low",
        "mentions_prior_work": true,
        "prior_work_strings": ["string", ...],
        "evidence_expected": "string"
      }
    ],
    "all_citations_raw": ["string", ...]
  }
}
```

### Task 2 Output Schema

```json
{
  "mode": "per_contribution",
  "paper_year": 2024,
  "queries": [
    {
      "id": "string",
      "query": "string",
      "status": "ok",
      "count": 10
    }
  ],
  "candidate_pool_top30": [
    {
      "cand_id": "string",
      "title": "string",
      "year": 2023,
      "venue": "string",
      "abstract": "string",
      "url": "string",
      "embedding": null
    }
  ],
  "stats": {
    "total_candidates": 50,
    "after_dedup": 45,
    "after_nontechnical_filter": 42,
    "after_year_filter": 40,
    "final": 30
  }
}
```

## Workflow Examples

### Full Batch Processing

Process all papers with both content and review:

```bash
python scripts/run_task1_task2_iclr2024.py
```

### Resume After Interruption

If the process is interrupted, resume without reprocessing completed papers:

```bash
python scripts/run_task1_task2_iclr2024.py --skip-existing
```

### Test on Specific Papers

Test the pipeline on specific papers before running the full batch:

```bash
python scripts/run_task1_task2_iclr2024.py --paper-ids 1BuWv9poWz 1JtTPYBKqt
```

### Preview Before Running

See which papers will be processed without actually running:

```bash
python scripts/run_task1_task2_iclr2024.py --dry-run
```

## Monitoring Progress

The script provides detailed logging for each paper:

```
[1/2] Processing 1BuWv9poWz...
[1BuWv9poWz] Processing paper...
[1BuWv9poWz] Paper: 45231 chars, Review: 3421 chars
[1BuWv9poWz] Running Task 1 extraction...
[1BuWv9poWz] ✓ Task 1 completed, saved to output/iclr2024/1BuWv9poWz/task1_result.json
[1BuWv9poWz]   - Core task: visual question answering for medical images
[1BuWv9poWz]   - Contributions: 2
[1BuWv9poWz]   - Novelty claims: 3
[1BuWv9poWz] Running Task 2 related works retrieval...
[1BuWv9poWz] ✓ Task 2 completed, saved to output/iclr2024/1BuWv9poWz/task2_result.json
[1BuWv9poWz]   - Queries: 2
[1BuWv9poWz]   - Candidates found: 30
```

## Summary Report

After batch processing completes, check the summary:

```bash
cat output/iclr2024/batch_summary.json
```

Example summary:

```json
{
  "total": 2,
  "task1_success": 2,
  "task2_success": 2,
  "both_success": 2,
  "failed": []
}
```

## Troubleshooting

### No papers found

If you see "No papers found with both paper content and review":

1. Check that paper files exist in `ICLR_2024/paper_nougat_mmd/`
2. Check that review files exist in `ICLR_2024/review_raw_txt/`
3. Verify that paper and review files share the same base name (e.g., `1BuWv9poWz.mmd` and `1BuWv9poWz.txt`)

### Task 1 extraction errors

- Check that `.env` file has valid API keys for LLM service
- Ensure paper and review files are valid text files
- Try with `--verbose` flag for detailed error messages

### Task 2 retrieval errors

- Semantic Scholar API may rate limit; script includes 1-second delays between papers
- Use `--skip-existing` to resume after rate limiting
- Check network connectivity

### Out of memory

If processing large batches causes memory issues:

1. Process in smaller batches using `--paper-ids`
2. Process papers one at a time in a loop
3. Restart the process with `--skip-existing` to resume

## Environment Setup

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

And a valid `.env` file with:

```
OPENAI_API_KEY=your_key_here
# or other LLM provider keys
```

## Current Status

Run the dry-run command to see how many papers are available:

```bash
python scripts/run_task1_task2_iclr2024.py --dry-run
```

Based on the current ICLR_2024 data:
- **Papers with content**: 16 files in `paper_nougat_mmd/`
- **Papers with reviews**: 33 files in `review_raw_txt/`
- **Papers with both**: 2 papers (1BuWv9poWz, 1JtTPYBKqt)
