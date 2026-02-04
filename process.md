0) Preprocess (no LLM)
0.1 Parse the paper PDF into structured text

Input: PDF
Output: paper_struct with sections + sentence IDs

Extract:

title, abstract, intro, related work, method, experiments, conclusion

Split into sentences and assign stable IDs:

S_abs_001, S_int_023, S_rw_112, …

Also store a short “paper card”:

paper_card = {title, year(if known), authors(if available), venue(if available), abstract}

Why: later, you can point to spans without quoting.

0.2 Normalize the review text

Input: raw review
Output: review_struct with sentence IDs, paragraph blocks

Sentence split + IDs: R_001, R_002, …

Keep original formatting and also a plain version.

1) Extract (LLM ONCE per paper–review pair)

This step produces structured targets for retrieval and claim-checking.

Output JSON (single object; BOTH sections are required):

{
  "paper": {
    "core_task": "string (<=20 words)",
    "contributions": [
      "string (<=25 words)",
      "string (<=25 words)"
    ],
    "key_terms": ["5-12 short phrases"],
    "must_have_entities": ["model/dataset/metric names if any"]
  },
  "review": {
    "novelty_claims": [
      {
        "claim_id": "C1",
        "text": "verbatim review claim (1-2 sentences max)",
        "stance": "not_novel | somewhat_novel | novel | unclear",
        "confidence_lang": "high | medium | low",
        "mentions_prior_work": true,
        "prior_work_strings": ["author-year strings or titles as written"],
        "evidence_expected": "method_similarity | task_similarity | results_similarity | theory_overlap | dataset_overlap"
      }
    ],
    "all_citations_raw": ["everything that looks like a citation/title/arxiv id/url"]
  }
}

1.1 Paper-side extraction (short, constrained)

Goal: get 1 core task + 1–2 contributions as query anchors (not a summary).

Paper fields (within "paper" key)

{
  "paper": {
    "core_task": "string (<=20 words)",
    "contributions": [
      "string (<=25 words)",
      "string (<=25 words)"
    ],
    "key_terms": ["5-12 short phrases"],
    "must_have_entities": ["model/dataset/metric names if any"]
  }
}


Guidelines

“core_task” should look like: “visual question answering for chest X-rays” not “improving healthcare”

contributions should be atomic: method change / training recipe / benchmark / theory claim

1.2 Review-side extraction (the critical part)

Goal: extract novelty-related claims from the review, and parse citations.

Review fields (within "review" key)

{
  "review": {
    "novelty_claims": [
      {
        "claim_id": "C1",
        "text": "verbatim review claim (1-2 sentences max)",
        "stance": "not_novel | somewhat_novel | novel | unclear",
        "confidence_lang": "high | medium | low",
        "mentions_prior_work": true,
        "prior_work_strings": ["author-year strings or titles as written"],
        "evidence_expected": "method_similarity | task_similarity | results_similarity | theory_overlap | dataset_overlap"
      }
    ],
    "all_citations_raw": ["everything that looks like a citation/title/arxiv id/url"]
  }
}


How to define “novelty claim”
Include only statements like:

“Incremental / similar to X / already studied”

“Main novelty is …”

“Essentially combines A and B”

“Differs from prior work by …”
Exclude general weaknesses not about novelty.

1.3 Recommended prompt style (to reduce hallucination)

Tell the model: “Do not invent citations; only copy what is in the review text.”

Force JSON-only output.

Temperature 0.

Efficiency note: This is the only “big” call.

2) Retrieve candidate prior work (no LLM)

This step is paper-level and can be cached across many reviews.

2.1 Build queries (fixed, low variance)

From Step 1 output:

For each contribution i:

Qi (contrib query): paper.core_task + contribution_i (prefer prior_work_query; append must-have entities if available)

Example:

Q1: "medical VQA chest X-ray radiology report generation report-guided VQA"

Q2: "selective prediction abstain mechanism hidden representation steering"

2.2 Retrieve Top-K (K=10 per query) via Semantic Scholar

Use Semantic Scholar search (Graph API), deterministic settings only.

Candidate record format

{
  "cand_id": "P123",
  "title": "...",
  "year": 2023,
  "venue": "...",
  "abstract": "...",
  "url": "...",
  "embedding": "vector-id-or-precomputed"
}

2.3 Clean + filter (deterministic)

Apply cheap filters:

dedup by approximate metadata similarity (title + abstract)

drop obvious non-technical matches

time filter: remove papers later than the target paper's release year

optional diversify by MMR so results aren’t near-duplicates


Output: candidate_pool_topN (N typically < 30)

3) Verify review claims against evidence (LLM Judge)

This step verifies each novelty-review sentence against each related paper using an LLM Judge.

**Input:**

- `S = {s}` — set of novelty-review sentences determined by Phase 1
- `a` — Abstract + Introduction of the paper being reviewed
- `B = {b}` — set of (Abstract + Introduction) of related papers found in Phase 2

**Process:**

```
for s in S:
    for b in B:
        Query = "Review sentence:" + s + "\nPaper being reviewed:" + a + "\nRelated work:" + b + INSTRUCTION_PROMPT
        result = Judge(Query)
```

3.1 Judge Classification (required)

The Judge must first classify the review sentence:

| Field | Values | Description |
|-------|--------|-------------|
| `Claim` | 0 / 1 | Whether `s` is a claim made by the Reviewer about paper `a` |
| `Proof` | 0 / 1 | Whether `s` is a proof/support of a claim made by the Reviewer about paper `a` |

3.2 Judge Score Output

The Judge assigns a score based on how the review sentence relates to the evidence:

| Score | Label | Description |
|-------|-------|-------------|
| **+2** | `SUPPORTED` | Reviewer claim aligns with retrieved evidence |
| **+1** | `OVERSTATED` | Some relation exists but reviewer claims "same / not novel" too strongly |
| **0** | `AMBIGUOUS` | Claim is too vague or unverifiable |
| **-1** | `UNDERSTATED` | Reviewer misses very close prior work present in candidates |
| **-2** | `UNSUPPORTED` | Evidence contradicts or no evidence found in candidate set |

3.3 Judge Output Format

```json
{
  "review_sentence_id": "S_001",
  "related_paper_id": "P123",
  "classification": {
    "claim": 1,
    "proof": 0
  },
  "score": 2,
  "label": "SUPPORTED",
  "explanation": "Short explanation: why picking that class and why that score"
}
```

3.4 Aggregation (per review sentence)

After processing all `(s, b)` pairs, aggregate results per review sentence:

```json
{
  "review_sentence_id": "S_001",
  "text": "verbatim review sentence",
  "classification": {"claim": 1, "proof": 0},
  "evidence_results": [
    {"related_paper_id": "P123", "score": 2, "label": "SUPPORTED", "explanation": "..."},
    {"related_paper_id": "P045", "score": 1, "label": "OVERSTATED", "explanation": "..."}
  ],
  "final_score": "max or weighted aggregate of evidence_results",
  "best_evidence": ["P123"]
}
```

4) Score the reviewer (deterministic scoring + optional small LLM)

You now have: extracted novelty claims + claim-level verification labels + citation parsing.

For each claim Ci, decide:

Label set

SUPPORTED: evidence clearly supports reviewer claim

UNSUPPORTED: evidence contradicts or no support found

OVERSTATED: there is related prior work, but reviewer’s “not novel / same as” is too strong

AMBIGUOUS: evidence insufficient or claim too vague to verify

Also output:

which candidate(s) mattered

which sentence IDs from candidate abstracts were used (or snippet indices)

optional: what additional evidence would be needed (1 line)

4.2 Batch multiple claims per call

To reduce calls:

Group 3–6 claims per verifier call (depending on context window)

Provide each claim with its 5-candidate evidence pack

Verifier output JSON

{
  "results": [
    {
      "claim_id": "C1",
      "label": "OVERSTATED",
      "key_candidates": ["P123","P045"],
      "evidence_spans": [
        {"cand_id":"P123","span_ref":"abstract_sent_2"},
        {"cand_id":"P045","span_ref":"abstract_sent_4"}
      ],
      "notes": "1 short sentence"
    }
  ]
}

4.3 Guardrails to keep verifier honest

Force: “If you cannot point to a span, return AMBIGUOUS.”

Force: JSON-only.

Temperature 0.

If you worry about verifier bias, do 2 verifiers only on disputed cases:

run second verifier only when label is UNSUPPORTED or OVERSTATED with low confidence

Cost control: Most claims will be decided in one pass.

4.1 Core novelty-grounding metrics (no LLM)

Let N = number of novelty claims in the review.

(A) Grounding Precision

How often the reviewer’s novelty statements are supportable.

GP = (# claims labeled SUPPORTED) / N

(B) Overreach Rate

How often the reviewer over-asserts “not novel”.

OR = (# OVERSTATED + # UNSUPPORTED) / N

(C) Verifiability Rate

Penalize vague claims.

VR = 1 - (# AMBIGUOUS / N)

These three alone already separate “good novelty reviewing” from hand-wavy takes.

5.2 Citation correctness (mostly deterministic)

From review.all_citations_raw:

Resolve each cited string to a candidate paper (exact/approx title match, author-year match, arXiv ID)

Mark:

RESOLVED: match found

UNRESOLVED: cannot find

Optional: if resolved, check whether the cited paper appears in evidence sets for claims where it’s mentioned.

Metrics:

CR = #RESOLVED / (#RESOLVED + #UNRESOLVED)

Citation-Relevance: fraction of resolved citations that were actually used by verifier as relevant evidence.

5.3 Coverage: did the reviewer miss the obvious neighbors? (no LLM)

This is review quality, not paper novelty.

For each claim Ci with stance “not_novel” or “somewhat_novel”:

Look at the Top-5 evidence set.

If any candidate has similarity above a threshold (e.g., 0.80) and reviewer does not mention any prior work, count as “missed neighbor”.

Metric:

MN = (# missed-neighbor claims) / (# claims where strong neighbor exists)

5.4 Calibration: confidence language vs support (lightweight)

From confidence_lang:
Map high=3, medium=2, low=1
Compute:

average confidence on SUPPORTED vs UNSUPPORTED/OVERSTATED
Good reviewers: high confidence mostly when supported.

Metric:

CalGap = mean(conf | supported) - mean(conf | unsupported_or_overstated)
(or a rank correlation)

5.5 Usefulness / actionability (two options)
Option 1 (no extra LLM, heuristic)

Score each novelty claim for specificity:

+1 if it names a concrete mechanism/assumption (“uses contrastive loss”, “missing-modality reconstruction”, “risk-coverage metric”)

+1 if it includes a clear comparison axis (“same setting/dataset”, “same architecture”, “same objective”)

+1 if it proposes a fix (“compare to X”, “add ablation Y”)

UseScore = avg per-claim (0–3)

Option 2 (small LLM, single call per review)

Ask a model to rate novelty critique usefulness given the extracted claims + verification labels, output 1–5 with 2 bullet justifications.
(Do this only if you can afford it.)

Putting it together: outputs you should store

For each (paper, review, reviewer-model), store a single record:

{
  "paper_id": "...",
  "reviewer_id": "...",
  "novelty_claims": [...],
  "evidence_sets": { "C1": [...], "C2": [...] },
  "verification": { "C1": {...}, "C2": {...} },
  "scores": {
    "GP": 0.50,
    "OR": 0.33,
    "VR": 0.83,
    "CR": 0.40,
    "MN": 0.60,
    "CalGap": 0.8,
    "UseScore": 1.9
  }
}

Practical knobs to control cost (recommended defaults)

Candidate pool: K=30

Evidence per claim: M=5

Batch size in verification: 3–6 claims per call

Second verifier: only for low-confidence / disputed

Cache: retrieval and candidate embeddings per paper (reused across all reviewers)

This gives you a pipeline that is:

review-centric (claims come from the review)

evidence-grounded (all judgments must cite spans)

cheap (one extraction call + a few small batched verification calls)

scorable (deterministic metrics aligned with “novelty judgment quality”)

If you want, I can also give you (1) a concrete prompt pack for Step 1 and Step 4, and (2) a minimal reference implementation skeleton (Python classes + JSON I/O + caching).
