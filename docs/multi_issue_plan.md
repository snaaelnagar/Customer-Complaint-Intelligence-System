# Multi-Issue Classification — Scientific Plan

## The question

Can one complaint narrative genuinely contain more than one Issue?

```
"I was charged twice and the refund hasn't arrived."
      ->  [Billing, Refund]
```

The CFPB schema only records a single `Issue` per complaint, so there
is no ground-truth multi-label data anywhere in this dataset. Before
building a "multi-label classifier," the honest first step is to find
out, empirically, whether this is a real pattern worth modeling or a
plausible-sounding idea that doesn't hold up against real complaint
text.

**This has not been done yet.** It requires the raw narrative dataset
(`data/processed/narratives_training.parquet` or the raw CFPB CSV),
which isn't available in this environment. What follows is the plan
for doing it once that data exists, plus a heuristic placeholder
(`src/multi_issue.py`) that approximates the idea today using the
already-trained single-label model, clearly flagged as unvalidated.

## Step 1 — Data exploration (no model yet)

Before training anything, quantify the signal:

1. **Narrative length vs. single-issue narratives.** Compare the
   length distribution of narratives whose `Issue` label the current
   classifier predicts with high confidence (likely single-topic)
   against low-confidence ones (possibly multi-topic). A meaningfully
   longer/more complex distribution among low-confidence narratives is
   a first signal.
2. **Coordinating-conjunction scan.** Count how often narratives
   contain patterns like `"... and also ..."`, `"... in addition ..."`,
   multiple distinct dollar amounts, or multiple distinct dates —
   proxies for "more than one thing happened here."
3. **Confusion-pair cross-check.** Cross-reference
   `src/nlp_evaluation.py`'s `error_analysis()` confused-pair output
   (once the Issue classifier is retrained on canonical labels) against
   a random sample of the actual misclassified narratives: is the
   model "wrong," or is the narrative genuinely describing two issues
   and the single-label ground truth only captured one of them?
4. **Manual sample audit.** Pull a stratified random sample (e.g. 200
   narratives across the confidence spectrum) and have a human reviewer
   tag each as single-issue or multi-issue, using the canonical Issue
   taxonomy from `src/taxonomy.py` as the tagging vocabulary. This
   produces the first small labeled evaluation set.

**Decision gate:** if the manual audit finds multi-issue narratives are
rare (e.g. <5-10%) or not distinguishable from single-issue narratives
in a principled way, multi-label modeling isn't worth the added
complexity — the heuristic in `src/multi_issue.py`, or nothing at all,
is the right level of investment. This plan should not proceed past
Step 1 without that check.

## Step 2 — Weak-label construction

If Step 1 supports the idea, build a *weak* multi-label training set
without waiting on more manual labeling:

1. For every canonical Issue category, gather its distinctive
   vocabulary (e.g. via TF-IDF top terms per class, already computable
   from the existing vectorizer/model).
2. Scan each narrative for keyword/keyphrase matches against every
   canonical Issue's vocabulary, not just the one tied to its official
   `Issue` label.
3. Treat a narrative as weakly multi-labeled when it matches a second
   canonical Issue's vocabulary strongly enough (threshold to be tuned
   against the Step 1 manual sample, used here as a small validation
   set — not for training).

This is a weak-supervision approach (labels are noisy, not
ground-truth), which is the honest way to bootstrap multi-label data
from a single-label schema.

## Step 3 — Model training

Once a weak-labeled multi-label dataset exists:

1. Train a multi-label model — e.g. One-vs-Rest logistic regression per
   canonical Issue, or a single model with a sigmoid output per class
   instead of softmax — on the weak labels.
2. Evaluate against the Step 1 manual sample (real human labels, small
   but trustworthy) using multi-label-appropriate metrics:
   - **Subset accuracy** (exact match of the full label set — a strict,
     often low, metric)
   - **Hamming loss** (fraction of individual label mismatches)
   - **Per-label F1 / Macro-F1 over labels** (same imbalance concerns
     as the single-label case, see `src/nlp_evaluation.py`)
   - **Jaccard similarity** between predicted and true label sets

## Step 4 — Productionize

Only after Step 3 shows the weak-labeled multi-label model actually
outperforms the heuristic in `src/multi_issue.py` on the human-labeled
sample should it replace `extract_candidate_issues()`. The function
signature in `src/multi_issue.py` is deliberately structured so this
swap doesn't require changes anywhere else (`nlp_predictor.py`, the
dashboard) — only the internals of
`analyze_complaint_multi_issue()` change.

## What exists today

| Piece | Status |
|---|---|
| Heuristic multi-issue extraction from the existing single-label model | Implemented (`src/multi_issue.py`), works today, clearly flagged as unvalidated |
| Data exploration (Step 1) | Not done — requires raw narrative dataset |
| Weak-label construction (Step 2) | Not done — depends on Step 1 |
| Trained multi-label model (Step 3) | Not done — depends on Step 2 |
| Production swap-in (Step 4) | Not done — depends on Step 3 |
