# Task 1 Quick Reference

## Single Paper

```bash
# Basic usage
python scripts/run_task1.py --paper paper.txt --review review.txt -o result.json

# With JSON input
python scripts/run_task1.py --input input.json -o result.json

# With title
python scripts/run_task1.py --paper paper.txt --review review.txt --title "My Paper" -o result.json
```

## Batch Processing

```bash
# Multiple files
python scripts/run_task1_batch.py --inputs *.json --output-dir results/

# From list file
python scripts/run_task1_batch.py --input-list papers.txt --output-dir results/

# Parallel (4 workers)
python scripts/run_task1_batch.py --inputs *.json --output-dir results/ --workers 4
```

## Test with Examples

```bash
# Single paper test
python scripts/run_task1.py --input examples/task1/input.json -o test.json

# Batch test
python scripts/run_task1_batch.py \
    --input-list examples/task1/batch_list.txt \
    --output-dir test_results/
```

## Input Format

### JSON Input
```json
{
  "paper_text": "paper content...",
  "review_text": "review content...",
  "paper_title": "optional",
  "paper_id": "optional"
}
```

## Output Format

```json
{
  "paper": {
    "core_task": "...",
    "contributions": ["...", "...", "..."],
    "key_terms": [...],
    "must_have_entities": [...]
  },
  "review": {
    "novelty_claims": [{
      "claim_id": "C1",
      "text": "...",
      "stance": "not_novel|somewhat_novel|novel|unclear",
      "confidence_lang": "high|medium|low",
      "mentions_prior_work": true/false,
      "prior_work_strings": [...],
      "evidence_expected": "method_similarity|..."
    }],
    "all_citations_raw": [...]
  }
}
```

## Options

- `--no-strict-verbatim` - Allow flexible matching
- `--no-augment-citations` - Disable regex citation extraction
- `-v, --verbose` - Debug logging
- `--workers N` - Parallel processing (batch mode)

## Full Documentation

See [docs/TASK1_USAGE.md](../docs/TASK1_USAGE.md) for complete documentation.
