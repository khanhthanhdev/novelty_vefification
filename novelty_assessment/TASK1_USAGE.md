# Task 1 Extraction - Usage Guide

Task 1 performs structured extraction from paper and review texts to identify:
- **From Paper**: Core task, contributions (1-3), key terms, must-have entities
- **From Review**: Novelty-related claims with stance, confidence, and prior work mentions

---

## Quick Start

### Single Paper

```bash
# From separate text files
python scripts/run_task1.py \
    --paper examples/task1/paper.txt \
    --review examples/task1/review.txt \
    --output result.json

# From JSON input
python scripts/run_task1.py \
    --input examples/task1/input.json \
    --output result.json

# With paper title
python scripts/run_task1.py \
    --paper paper.txt \
    --review review.txt \
    --title "My Paper Title" \
    --output result.json
```

### Multiple Papers (Batch Mode)

```bash
# Process multiple JSON files
python scripts/run_task1_batch.py \
    --inputs input1.json input2.json input3.json \
    --output-dir results/

# Process all JSON files in a directory
python scripts/run_task1_batch.py \
    --inputs data/*.json \
    --output-dir results/

# From a list file (one path per line)
python scripts/run_task1_batch.py \
    --input-list papers_list.txt \
    --output-dir results/

# Parallel processing with 4 workers
python scripts/run_task1_batch.py \
    --inputs data/*.json \
    --output-dir results/ \
    --workers 4
```

---

## Input Formats

### Option 1: Separate Text Files

**paper.txt** - Full paper text (plain text):
```
Introduction

Deep learning has revolutionized computer vision...

[rest of paper content]
```

**review.txt** - Review text (plain text):
```
Summary:
This paper proposes a new architecture for...

Strengths:
- Novel combination of attention mechanisms...

Weaknesses:
- The approach is similar to Smith et al. (2023)...
- Limited novelty compared to recent work...

[rest of review]
```

### Option 2: JSON Input

**input.json**:
```json
{
  "paper_text": "Full paper text here...",
  "review_text": "Review text here...",
  "paper_title": "Optional: Paper Title",
  "paper_id": "optional_id_for_naming"
}
```

---

## Output Format

The output is a JSON file with two main sections:

```json
{
  "paper": {
    "core_task": "Visual question answering for medical chest X-rays",
    "contributions": [
      "A novel attention mechanism that combines spatial and channel-wise attention",
      "A new training strategy using progressive curriculum learning",
      "State-of-the-art results on VQA-RAD and PathVQA benchmarks"
    ],
    "key_terms": [
      "visual question answering",
      "medical imaging",
      "attention mechanism",
      "chest X-ray",
      "multi-modal learning",
      "transformer",
      "curriculum learning"
    ],
    "must_have_entities": [
      "VQA-RAD",
      "PathVQA",
      "BERT",
      "ResNet-50"
    ]
  },
  "review": {
    "novelty_claims": [
      {
        "claim_id": "C1",
        "text": "The proposed attention mechanism is similar to the concurrent work by Smith et al. (2023), which also combines spatial and channel attention.",
        "stance": "not_novel",
        "confidence_lang": "high",
        "mentions_prior_work": true,
        "prior_work_strings": [
          "Smith et al. (2023)",
          "Smith et al."
        ],
        "evidence_expected": "method_similarity"
      },
      {
        "claim_id": "C2",
        "text": "The main novelty lies in the progressive curriculum learning strategy adapted specifically for medical VQA.",
        "stance": "novel",
        "confidence_lang": "medium",
        "mentions_prior_work": false,
        "prior_work_strings": [],
        "evidence_expected": "method_similarity"
      }
    ],
    "all_citations_raw": [
      "Smith et al. (2023)",
      "Johnson & Lee (2022)",
      "arXiv:2301.12345",
      "https://github.com/user/repo"
    ]
  }
}
```

### Output Schema Details

#### Paper Section

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `core_task` | string | The main problem the paper addresses | ≤20 words, specific and concrete |
| `contributions` | array[string] | Main contributions (atomic claims) | 1-3 items, ≤25 words each |
| `key_terms` | array[string] | Technical terms and phrases | 5-12 items |
| `must_have_entities` | array[string] | Explicit model/dataset/metric names | Variable length |

#### Review Section

| Field | Type | Description |
|-------|------|-------------|
| `novelty_claims` | array[object] | Novelty-related statements from review |
| `all_citations_raw` | array[string] | All citation-like strings found |

**Novelty Claim Object:**

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `claim_id` | string | C1, C2, ... | Unique identifier |
| `text` | string | | Verbatim claim from review (1-2 sentences max) |
| `stance` | enum | `not_novel`, `somewhat_novel`, `novel`, `unclear` | Reviewer's novelty assessment |
| `confidence_lang` | enum | `high`, `medium`, `low` | Confidence level in language |
| `mentions_prior_work` | boolean | | Whether claim references prior work |
| `prior_work_strings` | array[string] | | Exact citation strings from review |
| `evidence_expected` | enum | `method_similarity`, `task_similarity`, `results_similarity`, `theory_overlap`, `dataset_overlap` | Type of evidence needed |

---

## Command-Line Options

### run_task1.py (Single Paper)

```
Required (choose one):
  --input PATH              Input JSON file
  --paper PATH              Paper text file (requires --review)

Optional:
  --review PATH             Review text file (with --paper)
  --title TEXT              Paper title
  -o, --output PATH         Output JSON file (prints to stdout if omitted)
  --no-strict-verbatim      Allow flexible matching (less strict)
  --no-augment-citations    Disable regex citation extraction
  -v, --verbose             Enable debug logging
```

### run_task1_batch.py (Multiple Papers)

```
Required (choose one):
  --inputs PATH [PATH ...]  List of input JSON files
  --input-list PATH         Text file with input paths (one per line)

Required:
  --output-dir PATH         Output directory

Optional:
  --workers N               Number of parallel workers (default: 1)
  --no-strict-verbatim      Allow flexible matching
  --no-augment-citations    Disable regex citation extraction
  -v, --verbose             Enable debug logging
```

---

## Examples

### Example 1: Single Paper with Separate Files

```bash
# Create input files
cat > paper.txt << 'EOF'
Introduction

This paper addresses the problem of visual question answering
for medical chest X-rays. We propose three main contributions:

1. A novel dual-attention mechanism combining spatial and channel attention
2. A progressive curriculum learning strategy for medical domain
3. New state-of-the-art results on VQA-RAD and PathVQA benchmarks
EOF

cat > review.txt << 'EOF'
Summary: The paper proposes a VQA system for chest X-rays.

Strengths:
- Good empirical results

Weaknesses:
- The attention mechanism is similar to Smith et al. (2023)
- Limited novelty in the overall approach
- The curriculum learning differs from prior work mainly in the scheduling
EOF

# Run extraction
python scripts/run_task1.py \
    --paper paper.txt \
    --review review.txt \
    --title "Medical VQA with Dual Attention" \
    --output result.json

# Check output
cat result.json
```

### Example 2: Batch Processing with JSON Inputs

```bash
# Create input directory with multiple papers
mkdir -p data/inputs

# Create paper 1
cat > data/inputs/paper1.json << 'EOF'
{
  "paper_id": "paper_001",
  "paper_text": "Introduction\n\nThis paper proposes...",
  "review_text": "Summary: The paper...\nWeaknesses: Similar to...",
  "paper_title": "Novel Approach to X"
}
EOF

# Create paper 2
cat > data/inputs/paper2.json << 'EOF'
{
  "paper_id": "paper_002",
  "paper_text": "Abstract\n\nWe introduce...",
  "review_text": "Strengths: Innovative...\nWeaknesses: Incremental...",
  "paper_title": "Improved Method for Y"
}
EOF

# Process all papers
python scripts/run_task1_batch.py \
    --inputs data/inputs/*.json \
    --output-dir data/outputs/ \
    --workers 2 \
    --verbose

# Check results
ls data/outputs/
# Output:
# paper_001_task1.json
# paper_002_task1.json
# _batch_summary.json
```

### Example 3: Using Input List File

```bash
# Create list of input files
cat > papers_list.txt << 'EOF'
# Papers to process
data/paper1.json
data/paper2.json
data/paper3.json
# data/paper4.json  # commented out
EOF

# Process from list
python scripts/run_task1_batch.py \
    --input-list papers_list.txt \
    --output-dir results/ \
    --workers 3
```

### Example 4: Integration with Existing Pipeline

If you have papers from Phase 1 of the main pipeline:

```bash
# Extract paper text from Phase 1 output
jq -r '.paper_text' output/paper_xyz/phase1/phase1_extracted.json > paper.txt

# Extract review text (assuming you have it)
cat review.txt

# Run Task 1
python scripts/run_task1.py \
    --paper paper.txt \
    --review review.txt \
    --output task1_result.json
```

---

## Batch Summary Output

When using `run_task1_batch.py`, a summary file `_batch_summary.json` is created:

```json
{
  "total": 3,
  "success": 2,
  "errors": 1,
  "results": [
    {
      "input": "/path/to/paper1.json",
      "success": true,
      "error": null
    },
    {
      "input": "/path/to/paper2.json",
      "success": true,
      "error": null
    },
    {
      "input": "/path/to/paper3.json",
      "success": false,
      "error": "Task1ExtractionError: LLM API timeout"
    }
  ]
}
```

---

## Configuration

### Environment Variables

Task 1 uses the LLM client from the main pipeline. Set these environment variables:

```bash
# Required
export LLM_API_KEY="your-api-key"

# Optional
export LLM_MODEL_NAME="gpt-4"  # default depends on provider
export LLM_API_ENDPOINT="https://api.custom.com/v1"
export LLM_API_TEMPERATURE="0.0"
```

See [.env_example](../.env_example) in the root directory for all available options.

### Extraction Parameters

You can tune extraction behavior:

- `--no-strict-verbatim`: Allows claims/citations not verbatim in review (more lenient)
- `--no-augment-citations`: Disables regex-based citation extraction (LLM-only)

**Recommended**: Use defaults for most cases. Use `--no-strict-verbatim` only if you notice too many valid claims being filtered out.

---

## Troubleshooting

### Common Issues

**1. "LLM is not configured"**
```
Solution: Set LLM_API_KEY environment variable
export LLM_API_KEY="your-key"
```

**2. "No novelty claims extracted"**
```
Possible causes:
- Review doesn't contain novelty-related statements
- Claims filtered due to strict verbatim checking
Solution: Try --no-strict-verbatim
```

**3. "Empty contributions list"**
```
Possible causes:
- Paper doesn't explicitly list contributions
- LLM failed to extract
Solution: Check input paper text quality; may need manual review
```

**4. Parallel processing fails**
```
Possible causes:
- LLM API rate limits
- Insufficient system resources
Solution: Reduce --workers count or use sequential (--workers 1)
```

### Validation

Check output quality:

```bash
# Check if output has required fields
python -c "
import json
import sys

with open('result.json') as f:
    data = json.load(f)

assert 'paper' in data
assert 'review' in data
assert data['paper']['core_task']
assert len(data['paper']['contributions']) >= 1
assert len(data['paper']['contributions']) <= 3
print('✓ Output valid')
"
```

---

## Advanced Usage

### Custom LLM Parameters

Modify `extract_task1()` call in the scripts to adjust:

```python
result = extract_task1(
    paper_text=paper_text,
    review_text=review_text,
    paper_title=paper_title,
    max_tokens=2000,          # default: 1400
    temperature=0.1,           # default: 0.0
    max_paper_chars=300000,    # default: 200000
    max_review_chars=80000,    # default: 60000
)
```

### Programmatic Usage

```python
from novelty_assessment.task1_extractor import extract_task1

result = extract_task1(
    paper_text="...",
    review_text="...",
    paper_title="Optional Title",
)

print(f"Core task: {result['paper']['core_task']}")
print(f"Contributions: {len(result['paper']['contributions'])}")
print(f"Novelty claims: {len(result['review']['novelty_claims'])}")
```

---

## Integration Notes

### Differences from Phase 1 Pipeline

Task 1 is a standalone extraction module that differs from the main pipeline's Phase 1:

| Aspect | Phase 1 (Pipeline) | Task 1 (Standalone) |
|--------|-------------------|---------------------|
| Input | PDF URL or path | Plain text files |
| Review | Not used | Required input |
| Output | Multiple files | Single JSON |
| Focus | Retrieval queries | Novelty analysis |

Task 1 is designed for scenarios where you already have paper and review texts and want to extract structured novelty information.

---

## Performance

Typical processing times (single paper):
- **LLM call**: ~10-30 seconds (depends on API)
- **Total**: ~30-60 seconds

Batch processing with workers:
- **Sequential (1 worker)**: N × 30-60 seconds
- **Parallel (4 workers)**: ~(N/4) × 30-60 seconds + overhead

---

## License & Citation

This tool is part of the OpenNovelty project.

If you use Task 1 extraction in your research, please cite:

```bibtex
@article{opennovelty2025,
  title={OpenNovelty: Transparent and Verifiable Scholarly Novelty Assessment},
  author={...},
  journal={arXiv preprint arXiv:2601.01576},
  year={2025}
}
```

---

## Support

- **Issues**: https://github.com/Zhangbeibei1991/OpenNovelty/issues
- **Documentation**: https://www.opennovelty.org
- **Paper**: https://arxiv.org/abs/2601.01576
