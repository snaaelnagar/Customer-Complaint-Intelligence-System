# Architecture Decision Records (ADR)

This file records the significant technical decisions made in the Machine Learning / NLP Customer Complaint Intelligence System.

Each entry documents:

* The decision made
* The context behind it
* Alternatives considered
* Why the chosen approach was selected
* Current status

These decisions reflect real trade-offs made during development, not retrofitted justifications.

---

## ADR-001: Use Parquet for Intermediate and Dashboard Storage

**Decision:** Store the processed dataset and dashboard summaries as Parquet instead of CSV.

**Context:**
The raw CFPB dataset is approximately 8-9 GB as CSV. Repeatedly reading CSV during development and pipeline execution is slow and memory-heavy. CSV also does not support efficient column pruning.

**Alternatives considered:**

* **CSV throughout the pipeline** — rejected because every downstream module would repeatedly parse large row-oriented files even when only a few columns were needed.
* **Pickle files** — rejected because they are Python-specific and less portable for analytics workflows.
* **Database-first storage** — not needed for this portfolio-scale batch pipeline.

**Decision outcome:**
Use Parquet for:

* `complaints_processed.parquet`
* Dashboard summary outputs
* Growth, forecast, risk, driver, and recommendation outputs

**Result:**
Raw CSV size was reduced from approximately 8-9 GB to about 1.3 GB as processed Parquet, with an approximate storage reduction of 85%.

**Status:** Implemented.

---

## ADR-002: Use TF-IDF + Logistic Regression Instead of Transformer Fine-Tuning

**Decision:** Use TF-IDF vectorization with Logistic Regression for Product and Issue classification.

**Context:**
The dataset contains approximately 3.8M complaint narratives. A transformer model could potentially improve accuracy but would increase training time, memory usage, inference latency, and Docker image size.

**Alternatives considered:**

* **BERT / RoBERTa fine-tuning** — rejected for this version because the deployment target is CPU-only and single-machine.
* **Zero-shot classification** — rejected because it would be slow and expensive for large-scale inference.
* **Classical ML with TF-IDF** — selected because it is fast, reproducible, and deployment-friendly.

**Decision outcome:**
Use:

```text
TF-IDF + Logistic Regression
```

with a small tuning grid over:

```text
C
class_weight
max_features
```

**Result:**

| Classifier         | Accuracy |
| ------------------ | -------: |
| Product classifier |   75.28% |
| Issue classifier   |   62.39% |

**Status:** Implemented.

---

## ADR-003: Do Not Add Generic Sentiment Analysis

**Decision:** Do not include sentiment analysis as a dashboard feature.

**Context:**
Complaint narratives are naturally negative because users submit them when something went wrong. Generic sentiment analysis would likely classify most complaints as negative and add little business value.

**Alternatives considered:**

* **VADER sentiment analysis** — rejected because output would likely be uniformly negative.
* **Pretrained sentiment classifier** — rejected for the same reason.
* **Complaint-specific classification** — preferred, because Product, Issue, and Topic predictions map more directly to operational decisions.

**Decision outcome:**
Skip sentiment analysis and focus NLP work on:

* Product classification
* Issue classification
* Topic modeling

**Status:** Implemented as a deliberate omission.

---

## ADR-004: Pre-Aggregate Dashboard Data Instead of Reading Full Dataset in Streamlit

**Decision:** Streamlit should read only pre-computed summary files, not the full processed dataset.

**Context:**
Streamlit re-runs the script on interactions. Reading or aggregating 15.95M rows inside the app would make the dashboard slow and unstable.

**Alternatives considered:**

* **Read full Parquet directly inside Streamlit** — rejected because first load and cache invalidation would still be expensive.
* **Use Streamlit cache only** — insufficient because caching does not remove the initial heavy computation.
* **Pre-compute dashboard summaries** — selected.

**Decision outcome:**
Create small Parquet summaries under:

```text
data/processed/dashboard/
```

The dashboard reads summary outputs such as:

```text
kpis.parquet
top_products.parquet
company_risk_score.parquet
product_growth.parquet
forecast_summary.parquet
recommendations.parquet
```

**Status:** Implemented.

---

## ADR-005: Keep NLP Training Separate from the Main Analytics Pipeline

**Decision:** Do not run NLP training inside `src/pipeline.py`.

**Context:**
NLP training over hundreds of thousands of narratives is slower and less frequently needed than analytics summary generation.

**Alternatives considered:**

* **Run NLP training every time the pipeline runs** — rejected because routine analytics refreshes would become unnecessarily slow.
* **Train models once and load saved artifacts** — selected.

**Decision outcome:**
Separate commands are used:

```bash
python -m src.create_narrative_training_data
python -m src.nlp_model_training
python -m src.nlp_tuning
```

The main analytics pipeline remains:

```bash
python -m src.pipeline
```

**Status:** Implemented.

---

## ADR-006: Use Config-Driven Design with `config.yaml`

**Decision:** Move thresholds, paths, forecasting settings, risk weights, and NLP parameters into `config.yaml`.

**Context:**
Hardcoded values make experiments and deployment changes harder. A config-driven design makes the pipeline easier to tune without editing source code.

**Alternatives considered:**

* **Hardcoded constants** — rejected for most modules because values like risk thresholds and sample sizes should be adjustable.
* **Environment variables only** — rejected because nested pipeline configuration is easier to manage in YAML.
* **Central YAML config** — selected.

**Decision outcome:**
Use:

```text
config.yaml
src/config_loader.py
```

Most modules read settings through the centralized config, including
`nlp_model_training.py` and `nlp_tuning.py`, which were updated to read
their per-classifier hyperparameters (`max_features`, `min_df`,
`ngram_range`, `max_iter`, `min_class_samples`) from
`nlp.product_classifier` / `nlp.issue_classifier` / `nlp.topic_model`
in `config.yaml` instead of hardcoding them.

**Status:** Implemented.

---

## ADR-007: Add Minimum Complaint-Count Floor to Company Risk Scoring

**Decision:** Exclude companies below a configurable minimum complaint-count threshold before risk scoring.

**Context:**
Small-sample companies can produce misleading percentages. For example, a company with 2 complaints and 1 late response would have a 50% untimely rate, but that does not necessarily indicate systemic risk.

**Alternatives considered:**

* **Score every company** — rejected because low-volume companies could dominate risk rankings due to noise.
* **Score all companies and hide low-volume results in the UI** — rejected because the scoring output itself would still contain misleading rankings.
* **Apply a minimum complaint-count floor before scoring** — selected.

**Decision outcome:**
Companies below the threshold are excluded before risk-score computation.

**Status:** Implemented.

---

## ADR-008: Use Custom Exceptions with File and Line Context

**Decision:** Wrap pipeline errors in a custom exception class that includes file and line information.

**Context:**
The project contains multiple pipeline modules. When chained together, raw exceptions can be difficult to trace back to the failing stage.

**Alternatives considered:**

* **Let raw exceptions propagate** — simpler but harder to debug in a multi-stage pipeline.
* **Use custom exception wrapper** — selected for clearer debugging.

**Decision outcome:**
Use:

```text
src/exception.py
```

for consistent error messages and debugging context.

**Status:** Implemented.

---

## ADR-009: Use Prophet for Monthly Complaint Forecasting

**Decision:** Use Prophet for monthly complaint-volume forecasting.

**Context:**
The forecasting problem is monthly complaint volume with trend and seasonality. The goal is not to build the most complex forecasting system, but to provide a reliable planning signal for dashboard users.

**Alternatives considered:**

* **Naive moving average** — simple but weaker for trend and seasonality.
* **ARIMA / SARIMA** — valid, but requires more stationarity and parameter handling.
* **XGBoost regression** — possible, but would require more manual feature engineering for time effects.
* **Prophet** — selected because it handles trend and seasonality well with minimal setup.

**Decision outcome:**
Use Prophet with a holdout-based validation strategy.

**Result:**

| Metric          |             Value |
| --------------- | ----------------: |
| Validation MAPE |             3.57% |
| Validation MAE  | 17,748 complaints |

**Status:** Implemented.

---

## ADR-010: Use a Rule-Based Recommendation Engine

**Decision:** Build a rule-based recommendation engine instead of a generative or ML-based recommender.

**Context:**
The goal is to convert risk, growth, forecast, and driver signals into executive recommendations. For this project, explainability is more important than model complexity.

**Alternatives considered:**

* **Generative recommendations with an LLM** — rejected because it would introduce dependency on external APIs and make outputs harder to reproduce.
* **ML-based recommendation model** — rejected because there is no labeled training data for "correct recommendation."
* **Rule-based recommendation logic** — selected because it is transparent and auditable.

**Decision outcome:**
The recommendation engine reads outputs from:

```text
risk_analysis.py
growth_analysis.py
driver_analysis.py
forecasting.py
```

and produces:

```text
recommendations.parquet
executive_action_plan.parquet
```

**Status:** Implemented.

---

## ADR-011: Use Docker for Reproducible Dashboard Deployment

**Decision:** Package the application as a Docker image.

**Context:**
The project needs to run consistently across environments without asking users to manually recreate the local Python setup.

**Alternatives considered:**

* **Local-only setup** — rejected because dependency mismatches are common.
* **Streamlit-only deployment** — useful for demo hosting, but does not capture the full local artifact environment.
* **Docker container** — selected for reproducibility and one-command startup.

**Decision outcome:**
The image bundles:

```text
app/
src/
config.yaml
.streamlit/
data/processed/dashboard/
models/nlp/
```

and excludes:

```text
data/raw/
large training intermediates
local environment files
```

**Status:** Implemented.

---

## ADR-012: Bundle Pre-Computed Artifacts in the Docker Image

**Decision:** Include processed dashboard artifacts and trained NLP models in the Docker image.

**Context:**
The Docker image is intended for one-command dashboard execution, not for running full preprocessing and model training from scratch inside the container.

**Alternatives considered:**

* **Generate all artifacts on container startup** — rejected because startup would become slow and require the raw 8-9 GB CSV.
* **Require user to mount artifacts manually** — rejected for portfolio usability.
* **Bundle pre-computed outputs** — selected.

**Decision outcome:**
The image includes:

```text
data/processed/dashboard/
models/nlp/
```

but excludes:

```text
data/raw/
complaints_processed.parquet
narratives_training.parquet
```

**Trade-off:**
The image is larger, but the dashboard works immediately after `docker run`.

**Status:** Implemented.

---

## ADR-013: Use GitHub Actions for CI Validation

**Decision:** Use GitHub Actions to run tests and validate Docker builds.

**Context:**
Manual local testing is easy to forget. CI provides an automated check that the project still installs, tests, and builds successfully after changes.

**Alternatives considered:**

* **Manual testing only** — rejected because it does not scale well and gives no public validation signal.
* **Full deployment pipeline** — not necessary for this portfolio project.
* **GitHub Actions for test + Docker build validation** — selected.

**Decision outcome:**
CI runs:

```text
pytest tests
docker build
```

on push / pull request.

**Status:** Implemented.

---

## ADR-014: Use Streamlit Instead of a Custom React Frontend

**Decision:** Use Streamlit as the dashboard frontend.

**Context:**
The project focuses on data engineering, analytics, NLP, forecasting, and MLOps-style deployment rather than frontend engineering.

**Alternatives considered:**

* **React frontend + FastAPI backend** — more flexible but much heavier and slower to build.
* **Dash** — valid option, but Streamlit was faster for rapid analytics dashboard development.
* **Streamlit** — selected for fast dashboard development and direct Python integration.

**Decision outcome:**
Use:

```text
app/streamlit_app.py
```

as the dashboard entry point.

**Status:** Implemented.

---

## ADR-015: Keep the Project Single-Machine Instead of Spark/Airflow

**Decision:** Use a single-machine Pandas/PyArrow/Parquet pipeline instead of Spark or Airflow.

**Context:**
The project runs on a consumer laptop and handles the target 15.95M-row dataset successfully with chunking and Parquet. Adding Spark or Airflow would increase operational complexity without being necessary for the current scale.

**Alternatives considered:**

* **Spark** — rejected because current data volume is manageable after optimization and does not require distributed execution.
* **Airflow** — rejected because the workflow is batch-oriented but simple enough to run through scripts and documented commands.
* **Single-machine modular pipeline** — selected.

**Decision outcome:**
Use modular Python scripts under `src/`, executed with:

```bash
python -m src.pipeline
```

**Status:** Implemented.

---

## ADR-016: Canonicalize the CFPB Product/Issue Taxonomy Before Training

**Decision:** Add a canonical taxonomy layer (`src/taxonomy.py`) that maps historically-fragmented raw CFPB Product/Issue labels to stable categories before training, while always preserving the original raw label alongside it.

**Context:**
The CFPB has renamed and consolidated its Product/Issue taxonomy several times since 2012. The same real-world category (e.g. "Credit reporting") ends up split across 2-3 raw labels across different taxonomy eras purely due to wording changes. Training directly on raw labels fragments the signal for these categories, which disproportionately hurts Macro F1.

**Alternatives considered:**

* **Train on raw labels as-is** — the original approach; rejected once the fragmentation was identified, since it artificially inflates the effective number of rare classes.
* **Mine the raw label crosswalk from the full raw dataset directly** — the more rigorous approach, but not possible in this environment (no raw CFPB CSV, no network egress). Deferred to `src/taxonomy_audit.py`'s `audit_dataframe()` for when raw data is available.
* **Manually curated mapping from documented CFPB taxonomy history, cross-referenced against this project's actual trained-model label lists** — selected as the pragmatic option given the data constraint, with every merge tagged `high`/`medium` confidence and the raw label never discarded.

**Decision outcome:**
`src/nlp_model_training.py` trains on `{column}_canonical` while keeping `{column}_raw` in the working frame. `src/nlp_predictor.py` applies the same mapping at inference time to the already-shipped (raw-label-trained) models, by summing predicted probability across raw classes that share a canonical label — this works today without retraining.

**Result (measured against this project's own model metrics, see `docs/taxonomy_audit.md`):**
Product: 18 → 11 classes. Issue: 75 → 63 classes.

**Status:** Implemented for inference (works today). Training-time canonicalization implemented but not yet exercised against real data (no raw dataset in this environment).

---

## ADR-017: Add Confidence + Human Review Instead of Returning a Bare Top Label

**Decision:** Product and Issue predictions now return a canonical-label confidence score, the top alternative categories, and an `AUTO_ACCEPT` / `HUMAN_REVIEW` decision, instead of a single label with no indication of how certain the model was.

**Context:**
The dashboard's complaint analyzer previously showed only `model.predict()`'s top label. There was no way to tell, from the UI, whether a prediction was a confident call or a near-coin-flip — a genuine gap for anything positioned as a decision-support tool rather than a novelty demo.

**Alternatives considered:**
* **Leave as top-label-only** — rejected; this is the exact gap flagged in `case_study.md`'s "What I Would Do Differently" section before this change.
* **Return raw `predict_proba()` output directly** — rejected on its own, because raw-class probability is split across taxonomy-duplicate labels (see ADR-016), understating true confidence for fragmented categories.
* **Canonical-label confidence via `predict_proba()` summed by taxonomy group, with a configurable accept/review threshold** — selected.

**Decision outcome:**
`src/nlp_predictor.predict_with_confidence()` implements this; the threshold is configurable via `nlp.confidence_threshold` in `config.yaml` (default 0.55). The raw pre-canonicalization label is still returned alongside the canonical one, so nothing about the underlying model output is hidden.

**Status:** Implemented and working against the currently-shipped models (verified with real predictions, no retraining required).

---

## ADR-018: Multi-Issue Detection as an Explicitly-Flagged Heuristic, Not a Trained Multi-Label Model

**Decision:** Ship a heuristic multi-issue extractor (`src/multi_issue.py`) built on top of the existing single-label Issue classifier, clearly tagged `is_heuristic: True` in its output, rather than presenting it as a validated multi-label classifier.

**Context:**
A complaint narrative can plausibly describe more than one issue (e.g. "I was charged twice and the refund hasn't arrived"), but the CFPB schema only records one `Issue` per complaint — there is no ground-truth multi-label data to train or validate a real multi-label model against in this environment.

**Alternatives considered:**
* **Train a multi-label model on invented/assumed labels** — rejected; would produce an unvalidated model presented as if it were a real capability, which this project's benchmarking philosophy (`benchmark.md` §1: "avoids fake performance claims") explicitly rules out.
* **Skip multi-issue entirely until real data is available** — a reasonable option, but discards a usable approximation that's honestly labeled.
* **Threshold-based heuristic over the existing single-label classifier's probability distribution, explicitly flagged as unvalidated, with a documented plan for a real model** — selected.

**Decision outcome:**
`src/multi_issue.py` implements the heuristic. `docs/multi_issue_plan.md` documents the actual scientific plan (data exploration → weak-label construction → model training → validated productionization) for building a real multi-label classifier once the raw narrative dataset is available.

**Status:** Heuristic implemented and tested against the currently-shipped model. Real multi-label model: not started, blocked on raw data availability.

---

## ADR-019: Evaluate NLP Classifiers with Macro F1 / Top-K / Error Analysis Instead of Accuracy Alone

**Decision:** Standardize NLP classifier evaluation (`src/nlp_evaluation.py`) around Macro F1, Weighted F1, Top-K accuracy, and automatic plain-language error analysis of which category pairs the model confuses, instead of a single accuracy number.

**Context:**
Both the Product and Issue label sets are heavily imbalanced (see `docs/taxonomy_audit.md`'s rare-class counts). Accuracy alone rewards a classifier for being good at the two or three dominant classes while being nearly useless on the long tail — exactly the failure mode this project's own Issue classifier already showed (62.39% accuracy with many near-zero-recall classes, per `models/nlp/issue_classifier_metrics.txt`).

**Alternatives considered:**
* **Keep accuracy as the headline metric** — rejected as insufficient given the class imbalance already documented in `evaluation.md`.
* **Macro F1 as a drop-in replacement, without further diagnostics** — an improvement, but still leaves the reader asking "which classes, and why" without a next step.
* **Macro F1 + Weighted F1 + Top-K + automatic confused-pair explanation** — selected, so the training output directly names *which* categories the model struggles to separate and offers a candidate reason (shared label wording — a taxonomy signal per ADR-016 — vs. overlapping narrative vocabulary).

**Decision outcome:**
`src/nlp_model_training.py` now calls `src/nlp_evaluation.py`'s `evaluate_classifier()` / `format_evaluation_report()` and writes the richer report to `models/nlp/*_metrics.txt`. The dashboard's Model Performance tab (`src/model_metrics_reader.py`) parses either the legacy (accuracy-only) or new format, so it works whether or not a given model has been retrained yet.

**Status:** Implemented and unit-tested against synthetic data. Not yet exercised against the real dataset (currently-shipped models were trained before this change and haven't been retrained — no raw dataset available in this environment).

---

## Summary

The core architecture choices prioritize:

* Memory-safe processing
* Reproducible outputs
* CPU-only deployment
* Transparent modeling
* Fast dashboard interaction
* Dockerized execution
* Honest documentation of limitations
* An ML/NLP core (canonical taxonomy, confidence-aware prediction, rigorous evaluation) treated as the project's identity rather than an add-on feature

The result is a production-style portfolio system rather than a notebook-only analysis project.
