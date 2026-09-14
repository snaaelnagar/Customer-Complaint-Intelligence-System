# Evaluation

This document explains how each reported metric was measured, what it
does and doesn't tell you, and where the known weak points are. The goal
is to let a reader judge the numbers rather than just trust them.

---

## Dataset & Storage Context

| | |
|---|---|
| Raw Dataset Size | 8–9 GB CSV |
| Processed Dataset Size | ~1.3 GB Parquet |
| Storage Reduction | ~85% |
| Total Complaints | 15.95M |

The analytics pipeline uses Parquet storage and pre-aggregated dashboard outputs to reduce memory usage and improve dashboard responsiveness.

## NLP Classifiers

### Product Classifier

| | |
|---|---|
| Model | TF-IDF + Logistic Regression |
| Accuracy | 75.28% |
| Best parameters | `C=2.0`, `class_weight=None`, `max_features=20000` |
| Training data | Up to 300,000 sampled narratives (configurable), classes with fewer than 50 samples dropped |
| Validation | 80/20 stratified train/test split, fixed `random_state` |

### Issue Classifier

| | |
|---|---|
| Model | TF-IDF + Logistic Regression |
| Accuracy | 62.39% |
| Best parameters | `C=2.0`, `class_weight=None`, `max_features=20000` |
| Training data | Same source as above, classes with fewer than 50 samples dropped |
| Validation | 80/20 stratified train/test split, fixed `random_state` |

### Why Accuracy Alone Isn't Enough Here

`Issue` has a large number of distinct classes, several of which are
semantically close to each other (for example, multiple issue categories
that all describe variations of "incorrect information on a credit
report"). A model can be wrong in a way that's still operationally
useful (predicting a closely related issue category) or wrong in a way
that's actively misleading. Accuracy alone collapses that distinction.

Per-class precision/recall/F1 (saved to `product_classifier_metrics.txt`
and `issue_classifier_metrics.txt` after training) gives a more complete
picture, particularly for the rarer issue classes, which is why those
files — not just the headline accuracy number — should be checked before
trusting the model on a specific category.

### Taxonomy Canonicalization (New)

The CFPB `Product` and `Issue` fields have been renamed and consolidated
several times since 2012 (e.g. `Credit reporting` ->
`Credit reporting, credit repair services, or other personal consumer
reports` -> `Credit reporting or other personal consumer reports`, all
three referring to the same real-world category). Training directly on
raw labels fragments the signal for a single category across 2-3
classes, which disproportionately hurts Macro F1.

`src/taxonomy.py` adds a canonicalization layer (`canonicalize_product`,
`canonicalize_issue`) that maps historically-fragmented raw labels to a
single canonical category, **without deleting the original raw label**.
`src/nlp_model_training.py` now trains on the canonical label while
keeping the raw label alongside it (`{column}_raw`), and
`src/nlp_predictor.py` applies the same canonicalization at inference
time to the *already-shipped* models by summing predicted probability
mass across raw classes that map to the same canonical category — this
part works today, with the current models, without retraining.

Running the audit against this project's own already-trained
classifiers (`python -m src.taxonomy_audit`, using the real per-class
support counts already saved in `product_classifier_metrics.txt` /
`issue_classifier_metrics.txt`) shows:

| | Raw classes | Canonical classes | Classes removed |
|---|---|---|---|
| Product | 18 | 11 | 7 |
| Issue | 75 | 63 | 12 |

Full methodology, every merged group, and the confidence rating
(`high` / `medium`) behind each merge decision are in
[`docs/taxonomy_audit.md`](taxonomy_audit.md).

**Important caveat:** this mapping was built from publicly documented
CFPB taxonomy history and cross-referenced against the raw labels this
project's classifiers were actually evaluated on — not from mining the
full raw CFPB dataset directly, since that ~8-9 GB file isn't available
in this environment. Medium-confidence merges are flagged as such and
should be validated against real label co-occurrence/frequency once the
raw dataset is accessible (`audit_dataframe()` in
`src/taxonomy_audit.py` is built for that).

The headline accuracy/Macro F1 numbers on this page still reflect the
currently-shipped models, which were trained on **raw** labels (they
haven't been retrained yet — that requires the raw narrative dataset).
The canonicalization is already live for the two things that don't
require retraining: prediction-time confidence merging (see next
section) and the training script itself, so the next retraining run
will pick it up automatically.

### Data Leakage Prevention

Both `nlp_model_training.py` and `nlp_tuning.py` follow the same
ordering: the train/test split happens first, then `TfidfVectorizer` is
**fit only on the training split** and applied (`.transform()`, not
`.fit_transform()`) to the held-out test split. This means vocabulary
and term-frequency statistics from the test set never influence the
vectorizer, which is what makes the reported test-set accuracy a
genuine out-of-sample measurement rather than an optimistic one.

### Known Weaknesses

- **[Resolved] Confidence + Human Review.** Product and Issue
  predictions now expose a canonical-label confidence score, the top
  alternative categories, and an `AUTO_ACCEPT` / `HUMAN_REVIEW` decision
  (threshold configurable via `nlp.confidence_threshold` in
  `config.yaml`, default 0.55) — see `src/nlp_predictor.py` and the
  Complaint Analyzer tab. Low-confidence predictions are now flagged for
  human review instead of being silently presented as if certain.
- **New evaluation methodology.** Training (`src/nlp_model_training.py`)
  now reports Macro F1, Weighted F1, Top-K accuracy, and an automatic
  "top confused category pairs" error analysis (`src/nlp_evaluation.py`)
  instead of accuracy alone. This will apply the next time the models
  are retrained; see "Taxonomy Canonicalization" above for why this
  matters most on the Issue classifier's long tail of rare, overlapping
  classes.
- **Class imbalance.** Some products and issues have orders of magnitude
  more training examples than others. The `class_weight` parameter was
  tuned (alongside `C` and `max_features`) as part of `nlp_tuning.py`'s
  search, and `class_weight=None` won for both classifiers in this run —
  which suggests the imbalance, while present, wasn't severe enough for
  class weighting to help on the held-out test set.
- **Rare classes are dropped, not handled.** Classes below the minimum
  sample threshold (50) are excluded from training entirely rather than
  being oversampled or merged into a broader category. This avoids
  unreliable per-class metrics on tiny classes, but it also means the
  classifier simply cannot predict those rare categories at all.

---

## Topic Modeling (LDA)

10 topics, fit via `LatentDirichletAllocation` on the cleaned, vectorized
narrative text. Topic names (e.g. "Identity Theft & Fraud", "Mortgage &
Loan Servicing") were assigned manually by inspecting each topic's
top-weighted keywords — they are not learned or validated against any
ground-truth label, since the CFPB dataset doesn't provide one for this
purpose.

The topic prediction does expose a confidence score
(`topic_probs[topic_id]` from LDA's `transform()` output), but this is
**not a calibrated probability** — it reflects the model's internal
topic-mixture weight for the input text, not a validated likelihood that
the assigned topic is "correct" in any ground-truth sense.

---

## Forecasting (Prophet)

| | |
|---|---|
| Model | Facebook Prophet, monthly granularity |
| Validation MAPE | 3.57% |
| Validation MAE | 17,748 complaints |
| Validation method | Last 6 months held out, trained on prior months, predicted and compared against actuals |
| Training window | Configurable number of most recent **complete** years (the current, possibly-partial year is excluded from training) |

### How Validation Actually Works

`forecasting.py`'s `validate_forecast()` does not just report the
training-set fit — it trains a **separate** Prophet model on all but the
last 6 months of the training window, forecasts those 6 held-out months,
and compares the forecast against the actual values using MAE and MAPE.
The final production model (used for the actual future forecast shown in
the dashboard) is then retrained on the full training window. This means
the reported 3.57% MAPE reflects genuine out-of-sample performance on
recent history, not in-sample fit.

### Known Weaknesses

- **Single signal.** The forecast is based purely on historical complaint
  volume. It does not incorporate external factors (regulatory changes,
  economic conditions, company-specific events) that could shift volume
  in ways the historical pattern wouldn't predict.
- **Validation window is short relative to typical forecast horizon.** A
  6-month holdout is used to validate a model that's then asked to
  forecast up to 6 months forward — reasonable, but it means the
  validation error estimate is itself based on a single 6-month period,
  not multiple independent holdout windows.
- **Partial-year exclusion is a heuristic.** The current year is always
  excluded from training on the assumption that it's incomplete. If the
  pipeline happens to run very early or very late in a year, this could
  exclude more or less real data than intended.

---

## Unit Tests

19/19 tests passing, covering core utility functions (configuration
loading, exception handling, safe growth-rate calculation, risk-score
scaling, MAPE calculation). The test suite validates individual
functions in isolation rather than running an end-to-end integration
test against real data — there is currently no test that runs the full
pipeline against a small fixture dataset and checks the final dashboard
outputs.

---

## Metric Scope

All NLP metrics reported in this document are offline evaluation metrics measured on a held-out test split.

These metrics do not represent:
- Real-world production monitoring
- Live user feedback
- Model drift performance
- Future complaint distributions

As a result, actual performance may vary when the model is exposed to complaint patterns that differ from the training data.

---

## Out of Scope

This evaluation covers offline accuracy and validation metrics only.
The following are explicitly **not** covered, because they haven't been
built or measured:

- Real-time monitoring of live prediction quality
- Data or model drift detection over time
- Online/continuous learning
- Human-in-the-loop review or correction workflows

## Executive Interpretation

A 75.28% Product accuracy means the model identifies the correct
broad product category in roughly three out of four complaints. A
62.39% Issue accuracy reflects a harder problem — `Issue` has more
classes than `Product`, and several of them describe closely related
or overlapping situations (see "Why Accuracy Alone Isn't Enough Here"
above), so a meaningful share of the "incorrect" predictions are likely
near-misses rather than unrelated categories.

Practically, this means the classifiers are appropriate for complaint
triage, routing, and exploratory trend analysis — surfacing the likely
category of a new complaint quickly — but **should not be used as the
sole basis for a regulatory, legal, or compliance determination** about
a specific complaint. A wrong classification on an individual complaint
is an expected and non-rare outcome at this accuracy level.

## What These Metrics Should Not Be Used For

- As a substitute for a compliance or legal review of any individual
  complaint
- As evidence that a specific company did or did not commit a specific
  violation
- As a confidence-calibrated probability of correctness (see "Topic
  Modeling" above — even the one model that exposes a numeric
  confidence score is not calibrated against ground truth)

---

## Honest Summary

The numbers above are real, measured results from this project's actual
data and code — not illustrative placeholders. They should be read with
the limitations above in mind: confidence-aware NLP predictions have
since been added (see "Taxonomy Canonicalization" and the Confidence +
Human Review bullet above), so the single-signal forecast and the
still-unretrained, raw-label-based headline accuracy numbers are now
the two areas most worth improving before treating this as more than a
portfolio-grade analytics platform.