# Task 1 Examples

This directory contains example input files for testing Task 1 extraction.

## Files

### Single Paper Example

- **paper.txt** - Full paper text for a medical VQA paper
- **review.txt** - Review text with novelty concerns
- **input.json** - Combined JSON input (same content as paper.txt + review.txt)

### Batch Processing Examples

- **batch_input1.json** - Example paper 1 (transformer models)
- **batch_input2.json** - Example paper 2 (graph neural networks)
- **batch_list.txt** - List file for batch processing

## Quick Test

### Test Single Paper

```bash
# Using separate text files
python scripts/run_task1.py \
    --paper examples/task1/paper.txt \
    --review examples/task1/review.txt \
    --output test_output.json

# Using JSON input
python scripts/run_task1.py \
    --input examples/task1/input.json \
    --output test_output.json
```

### Test Batch Processing

```bash
# Process all example inputs
python scripts/run_task1_batch.py \
    --inputs examples/task1/batch_input*.json \
    --output-dir test_results/

# Or use the list file
python scripts/run_task1_batch.py \
    --input-list examples/task1/batch_list.txt \
    --output-dir test_results/
```

## Expected Output Structure

The output JSON will have this structure:

```json
{
  "paper": {
    "core_task": "visual question answering for chest X-ray images",
    "contributions": [
      "A dual-attention mechanism that captures spatial regions and channel-wise features",
      "A progressive curriculum learning training strategy for medical VQA",
      "State-of-the-art results on VQA-RAD and PathVQA benchmarks"
    ],
    "key_terms": [
      "visual question answering",
      "chest X-ray",
      "dual-attention mechanism",
      "curriculum learning",
      "medical imaging",
      ...
    ],
    "must_have_entities": [
      "VQA-RAD",
      "PathVQA",
      "ResNet-50",
      "BERT"
    ]
  },
  "review": {
    "novelty_claims": [
      {
        "claim_id": "C1",
        "text": "The proposed dual-attention mechanism is very similar to the concurrent work by Smith et al. (2023) \"Dual Attention Networks for Visual Question Answering\"...",
        "stance": "not_novel",
        "confidence_lang": "high",
        "mentions_prior_work": true,
        "prior_work_strings": ["Smith et al. (2023)"],
        "evidence_expected": "method_similarity"
      },
      ...
    ],
    "all_citations_raw": [
      "Smith et al. (2023)",
      "Zhang et al. (2023)",
      "Johnson & Lee, 2022",
      "Bengio et al. (2009)",
      ...
    ]
  }
}
```

## Notes

- The example paper is synthetic but realistic
- Review contains typical novelty concerns
- Multiple citation formats are included
- Good for testing extraction quality
