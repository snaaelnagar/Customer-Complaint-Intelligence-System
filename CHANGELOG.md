# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog principles and follows semantic versioning.

---

# [v2.0.0] - 2026-09-07

## ML/NLP Repositioned as Core Identity; Taxonomy, Confidence & Discovery Layers

This release restructures the project so that ML/NLP classification and
discovery are the primary identity, with the existing executive
analytics, risk scoring, growth detection, and forecasting retained in
full as a supporting "Business Analytics" layer.

### Added

* **Canonical taxonomy layer** (`src/taxonomy.py`, `src/taxonomy_audit.py`) — maps historically-fragmented raw CFPB Product/Issue labels (renamed/consolidated by CFPB several times since 2012) to stable canonical categories, while always preserving the original raw label. Measured against this project's own trained-model metrics: Product 18 → 11 classes, Issue 75 → 63 classes. Full methodology and every merge decision documented in `docs/taxonomy_audit.md`.
* **Confidence + human review** (`src/nlp_predictor.py`) — Product/Issue predictions now return a taxonomy-aware confidence score, top alternative categories, and an `AUTO_ACCEPT` / `HUMAN_REVIEW` decision (configurable threshold in `config.yaml`), instead of a bare top label. Works against the already-shipped models without retraining.
* **Rigorous NLP evaluation** (`src/nlp_evaluation.py`) — Macro F1, Weighted F1, Top-K accuracy, and automatic plain-language error analysis of which category pairs the model confuses and a candidate reason why, replacing accuracy-only reporting.
* **Multi-issue detection infrastructure** (`src/multi_issue.py`) — heuristic extraction of additional plausible issues from the existing single-label classifier, explicitly flagged as unvalidated, with a documented scientific plan (`docs/multi_issue_plan.md`) for building and validating a real multi-label model.
* **Topic Discovery Layer** (`src/topic_discovery.py`) — elevates the existing LDA topic model from a per-complaint feature into recurring-theme and emerging-issue detection across a text corpus.
* **Model metrics reader** (`src/model_metrics_reader.py`) — parses both the legacy (accuracy-only) and new (Macro F1 / Top-K) classifier metrics file formats for dashboard display.
* Four new dashboard tabs: Classification, Similar Complaints, Issue Discovery, Model Performance.
* `tests/test_taxonomy.py` — unit tests for the canonical taxonomy mapping.
* Architecture Decision Records ADR-016 through ADR-019 documenting the above decisions (`docs/adr.md`).

### Changed

* `src/nlp_model_training.py` now trains on the canonical label (raw label preserved alongside it) and reports the new evaluation metrics.
* Dashboard tab structure reorganized: `Overview`, `Complaint Analyzer`, `Classification`, `Similar Complaints`, `Issue Discovery`, `Trends & Emerging Issues`, `Model Performance`, `Business Analytics` (nested — houses the original Risk/Growth/Forecasting/Recommendations/Product/Issue/Company/Resolution/Consumer/Narrative/Channels/Geography tabs unchanged).
* `docs/evaluation.md`, `docs/architecture.md`, `docs/case_study.md`, `docs/benchmark.md`, `docs/INTERVIEW_QA.md`, `README.md`, and `PROJECT_SUMMARY.md` updated to reflect the ML/NLP-first identity and the new capabilities.

### Fixed

* Stale unit-test count references across documentation (14/14 → 19/19, reflecting the new taxonomy tests).
* Stale "returns only a top label" / "no confidence score" limitation notes that no longer applied once the confidence layer shipped.

### Breaking Changes

* Dashboard tab names and order changed (see "Changed" above). Any bookmarked tab position or automation relying on the previous 16-tab flat layout will need updating.

---



## Tuned NLP Pipeline, Recommendation Engine, Docker & CI/CD

### Added

* NLP hyperparameter tuning pipeline (`nlp_tuning.py`)
* Fixed 6-combination search grid over:

  * `C`
  * `class_weight`
  * `max_features`
* Tuned Product classifier
* Tuned Issue classifier
* Recommendation Engine (`recommendation_engine.py`)
* Executive Action Plan output
* GitHub Actions CI pipeline
* Automated Docker build validation
* Docker Hub deployment
* Streamlit custom dark theme (`.streamlit/config.toml`)
* Pytest test suite covering core utility functions
* Forecast validation using holdout-based MAE and MAPE metrics
* Minimum complaint-count floor in company risk scoring
* "New / Emerging" growth label for products and issues with zero complaints in the prior year

### Changed

* Product classifier updated to best-performing parameters:

  * `C=2.0`
  * `class_weight=None`
  * `max_features=20000`

* Issue classifier updated to best-performing parameters:

  * `C=2.0`
  * `class_weight=None`
  * `max_features=20000`

* Configuration migrated from hardcoded constants to `config.yaml`

* Centralized configuration loading via `config_loader.py`

* Improved recommendation prioritization logic

* Improved growth classification logic

### Results

* Product Classifier Accuracy: **75.28%**
* Issue Classifier Accuracy: **62.39%**
* Forecast Validation MAPE: **3.57%**
* Forecast Validation MAE: **17,748**
* Unit Tests: **14/14 Passed**

### Fixed

* Growth-analysis edge case where new categories were incorrectly labeled as Stable
* Risk-scoring noise caused by extremely low-volume companies
* Docker deployment reproducibility issues through dependency pinning

### Breaking Changes

None.

---

# [v1.1.0] - 2026-06-20

## Forecasting, Risk Intelligence & Driver Analytics

### Added

* Prophet-based complaint forecasting
* Monthly complaint forecasting pipeline
* Next-month forecast output
* Next-3-month forecast output
* Next-6-month forecast output
* Forecast validation framework
* 6-month holdout evaluation window
* Complaint driver analysis
* Product → Issue → Sub-Issue hierarchy analysis
* Product driver summary
* Growth analysis
* Product-level growth tracking
* Issue-level growth tracking
* Executive Risk Dashboard

### Changed

* Risk scoring thresholds refined
* Dashboard expanded with advanced analytics outputs
* Growth classification framework standardized

### Results

* Forecast Validation MAPE: **3.57%**
* Forecast Validation MAE: **17,748**

### Breaking Changes

None.

---

# [v1.0.0] - 2026-06-15

## Initial Data Platform, Analytics Dashboard & Parquet Optimization

### Added

* Chunk-based CSV preprocessing
* Required-column validation
* Missing-value handling
* Date parsing and cleaning
* Duplicate complaint removal
* Feature engineering:

  * Year
  * Month
  * Quarter
  * Day
  * Day_Name
* Parquet-based storage architecture
* Dashboard summary generation
* Executive KPI dashboard
* Product analysis
* Issue analysis
* Company analysis
* Resolution analysis
* Geography analysis
* Narrative intelligence dashboard

### Engineering Notes

* Raw Dataset:

  * 15.95M complaint records
  * 8-9 GB CSV

* Processed Dataset:

  * ~1.3 GB Parquet

* Storage Reduction:

  * ~85% reduction versus raw CSV

* Dashboard Optimization:

  * Streamlit reads pre-aggregated summary files
  * Avoids loading the full processed dataset at runtime

### Architecture Decisions

* Adopted Parquet as the primary storage format
* Separated preprocessing from dashboard serving
* Implemented pre-aggregation strategy for dashboard performance
* Designed modular pipeline architecture for future expansion

### Breaking Changes

None.
