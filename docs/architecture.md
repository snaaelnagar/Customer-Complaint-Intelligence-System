# Architecture

This document explains the architecture of the Machine Learning / NLP Customer Complaint Intelligence System: how raw CFPB complaint data moves through preprocessing, canonical taxonomy, NLP classification, unsupervised discovery, analytics, recommendation generation, dashboard serving, testing, and Docker deployment.

---

## 1. Overview

The platform is designed as a production-style, single-machine ML/NLP system with an analytics layer built on top of it.

It is split into four major layers:

1. **Pipeline Layer** (`src/`)
   Batch jobs that read raw or processed data and write small, pre-aggregated Parquet outputs.

2. **ML/NLP Layer** (`src/taxonomy.py`, `src/nlp_model_training.py`, `src/nlp_predictor.py`, `src/nlp_evaluation.py`, `src/multi_issue.py`, `src/topic_discovery.py`)
   Canonical taxonomy, classification, confidence-aware prediction, evaluation, multi-issue infrastructure, and unsupervised topic discovery — the project's core identity. See Section 5.

3. **Presentation Layer** (`app/streamlit_app.py`)
   Streamlit dashboard that reads only pre-computed summary files, never the raw CSV or full processed dataset. The ML/NLP layer above is the dashboard's primary identity (Complaint Analyzer, Classification, Similar Complaints, Issue Discovery, Model Performance tabs); the pre-aggregated analytics from the Pipeline Layer are presented as a supporting "Business Analytics" tab.

4. **Deployment Layer** (`Docker`)
   Packages the dashboard, source code, processed dashboard artifacts, and trained NLP models for one-command execution.

This separation keeps the dashboard fast and memory-light even though the source dataset contains **15.95M complaints** and starts as an **8-9 GB CSV**.

---

## 2. Technology Stack

| Layer            | Technology                           |
| ---------------- | ------------------------------------ |
| Language         | Python 3.11                          |
| Data Processing  | Pandas, PyArrow                      |
| Storage          | Parquet                              |
| Dashboard        | Streamlit                            |
| Visualization    | Plotly                               |
| NLP Features     | TF-IDF                               |
| NLP Models       | Logistic Regression, LDA             |
| Forecasting      | Prophet                              |
| Testing          | Pytest                               |
| CI/CD            | GitHub Actions                       |
| Containerization | Docker                               |
| Configuration    | YAML (`config.yaml`)                 |
| Logging          | Python logging + RotatingFileHandler |

---

## 3. Repository Structure

```text
customer-complaint-intelligence/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── config_loader.py
│   ├── logger.py
│   ├── exception.py
│   ├── preprocessing.py
│   ├── dashboard_data.py
│   ├── risk_analysis.py
│   ├── driver_analysis.py
│   ├── growth_analysis.py
│   ├── forecasting.py
│   ├── recommendation_engine.py
│   ├── create_narrative_training_data.py
│   ├── taxonomy.py
│   ├── taxonomy_audit.py
│   ├── nlp_model_training.py
│   ├── nlp_tuning.py
│   ├── nlp_evaluation.py
│   ├── nlp_predictor.py
│   ├── multi_issue.py
│   ├── topic_discovery.py
│   ├── model_metrics_reader.py
│   └── pipeline.py
├── data/
│   ├── raw/
│   └── processed/
│       └── dashboard/
├── models/
│   └── nlp/
├── tests/
├── docs/
├── assets/
├── .github/
│   └── workflows/
├── .streamlit/
│   └── config.toml
├── config.yaml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── CHANGELOG.md
└── LICENSE
```

---

## 4. End-to-End Flow

```text
data/raw/complaints.csv
8-9 GB CSV / 15.95M rows
        │
        ▼
src/preprocessing.py
  - chunked CSV reading
  - required-column validation
  - missing-value handling
  - date parsing
  - feature engineering
  - duplicate removal
        │
        ▼
data/processed/complaints_processed.parquet
~1.3 GB Parquet
        │
        ├──────────────► src/dashboard_data.py
        │                  └── data/processed/dashboard/*.parquet
        │
        ├──────────────► src/risk_analysis.py
        │                  └── company_risk_score.parquet
        │
        ├──────────────► src/driver_analysis.py
        │                  ├── driver_analysis.parquet
        │                  ├── top_complaint_drivers.parquet
        │                  └── product_driver_summary.parquet
        │
        └──────────────► src/growth_analysis.py
                           ├── product_growth.parquet
                           ├── issue_growth.parquet
                           └── monthly_complaint_trend.parquet
                                    │
                                    ▼
                           src/forecasting.py
                           ├── complaint_forecast.parquet
                           └── forecast_summary.parquet
                                    │
                                    ▼
                           src/recommendation_engine.py
                           ├── recommendations.parquet
                           └── executive_action_plan.parquet
                                    │
                                    ▼
                           app/streamlit_app.py
```

---

## 5. ML / NLP Flow

NLP training is intentionally separated from the main analytics pipeline. This is the project's core identity layer.

```text
data/processed/complaints_processed.parquet
        │
        ▼
src/create_narrative_training_data.py
        │
        ▼
data/processed/narratives_training.parquet
        │
        ▼
src/taxonomy.py  ──►  canonical Product/Issue labels
        │              (raw label kept alongside, never deleted;
        │               see docs/taxonomy_audit.md)
        │
        ├──► src.nlp_model_training.py
        │       ├── Product classifier  (trained on canonical labels)
        │       ├── Issue classifier    (trained on canonical labels)
        │       ├── LDA topic model
        │       └── src.nlp_evaluation.py
        │             └── Macro F1, Weighted F1, Top-K, error analysis
        │
        └──► src.nlp_tuning.py
                ├── Tuned Product classifier
                └── Tuned Issue classifier
        │
        ▼
models/nlp/*.pkl
        │
        ▼
src/nlp_predictor.py
        │  ├── canonical-label confidence + top alternatives
        │  ├── AUTO_ACCEPT / HUMAN_REVIEW decision
        │  └── src/multi_issue.py (heuristic candidate additional issues)
        │
        ├──► Streamlit "Complaint Analyzer" tab
        ├──► Streamlit "Classification" tab   (src/model_metrics_reader.py)
        └──► Streamlit "Model Performance" tab

src/topic_discovery.py  ──►  Streamlit "Issue Discovery" tab
        (recurring themes, clusters, emerging issues over time)
```

### Confidence + Human Review

`src/nlp_predictor.py` groups each raw predicted class's probability by
its canonical taxonomy label and sums probability mass within each
group, so near-duplicate raw labels (e.g. two historical spellings of
the same Issue) no longer artificially split confidence. The top
canonical label's summed confidence is compared against
`nlp.confidence_threshold` in `config.yaml` (default 0.55) to decide
`AUTO_ACCEPT` vs `HUMAN_REVIEW`. This works with the currently-shipped
models, without retraining — see docs/taxonomy_audit.md for why.

### Multi-Issue Infrastructure

`src/multi_issue.py` is infrastructure, not a validated multi-label
model: it surfaces additional canonical Issues whose merged
probability clears a relative threshold against the top prediction,
explicitly flagged `is_heuristic: True`. A documented plan for
building and validating a real multi-label classifier is in
`docs/multi_issue_plan.md`; it has not been executed because it needs
the raw narrative dataset for empirical validation.

### Topic Discovery Layer

`src/topic_discovery.py` elevates the existing LDA topic model from a
single-complaint feature into a discovery tool:
`compute_topic_prevalence` reports what share of a set of complaints
falls into each topic, and `detect_emerging_topics` compares
prevalence between two periods/subsets to flag growing themes. Both
work on any text corpus handed to them; the dashboard's static topic
catalog (10 topics, real keywords from the trained model) works today,
while the time-based "emerging issues" view needs a narrative corpus
with timestamps that isn't available in this environment.

### Why NLP is Separate

`pipeline.py` does not call:

```text
create_narrative_training_data.py
nlp_model_training.py
nlp_tuning.py
```

because NLP training is slower and does not need to run on every analytics refresh.

Analytics outputs may refresh frequently, but trained NLP models change only when retraining is intentionally performed.

### Operational Note

`nlp_model_training.py` and `nlp_tuning.py` both write to overlapping model filenames such as:

```text
product_classifier_model.pkl
issue_classifier_model.pkl
```

Whichever script runs last becomes the source of truth loaded by `nlp_predictor.py`.

---

## 6. Module Dependency Order

`src/pipeline.py` runs analytics modules in a fixed sequence:

```text
dashboard_data
      ↓
risk_analysis
      ↓
driver_analysis
      ↓
growth_analysis
      ↓
forecasting
      ↓
recommendation_engine
```

However, not every stage depends on the previous stage.

### Independent Jobs

Each of these reads `complaints_processed.parquet` directly:

```text
dashboard_data
risk_analysis
driver_analysis
growth_analysis
```

These modules could run in any relative order.

### Dependent Jobs

```text
growth_analysis
      ↓
monthly_complaint_trend.parquet
      ↓
forecasting
```

```text
risk_analysis ─────┐
growth_analysis ───┼──► recommendation_engine
forecasting ───────┤
driver_analysis ───┘
```

`forecasting.py` and `recommendation_engine.py` validate required input files at startup and fail clearly if upstream outputs are missing.

---

## 7. Pipeline Layer

The pipeline layer contains batch jobs under `src/`.

Responsibilities:

* Read raw or processed data
* Validate inputs
* Transform data
* Generate Parquet outputs
* Log stage-level progress
* Raise clear errors through custom exceptions

The pipeline layer is intentionally separated from the Streamlit app so that expensive computation happens offline, not during dashboard interaction.

---

## 8. Presentation Layer

The dashboard is implemented in:

```text
app/streamlit_app.py
```

The dashboard reads only:

```text
data/processed/dashboard/*.parquet
models/nlp/*.pkl
```

It does **not** read:

```text
data/raw/complaints.csv
data/processed/complaints_processed.parquet
```

at runtime.

This keeps dashboard startup and interaction speed independent of the full dataset size.

---

## 9. Deployment Layer

The Docker image bundles:

```text
app/
src/
config.yaml
.streamlit/
data/processed/dashboard/
models/nlp/
```

It excludes:

```text
data/raw/
data/processed/complaints_processed.parquet
data/processed/narratives_training.parquet
logs/
.env
```

The image is intentionally larger than a minimal Streamlit app image because it includes pre-computed dashboard artifacts and trained NLP models.

This enables:

```text
docker run -p 8501:8501 <your-dockerhub-username>/customer-complaint-intelligence:latest
```

to launch a working dashboard immediately.

---

## 10. CI/CD Layer

GitHub Actions validates the project automatically.

```text
Developer Push
      │
      ▼
GitHub Actions
      │
      ├── Run Unit Tests
      │
      └── Validate Docker Build
      │
      ▼
Fail fast if tests or Docker build break
```

The workflow checks:

* Python dependency installation
* Pytest execution
* Docker image build

This reduces the chance of pushing broken code or a broken Docker build to the main branch.

---

## 11. Configuration Layer

Configuration is centralized in:

```text
config.yaml
```

Loaded by:

```text
src/config_loader.py
```

Configurable items include:

* File paths
* Preprocessing chunk size
* Risk-score weights
* Risk thresholds
* Growth thresholds
* Forecast horizon
* NLP sample sizes and per-classifier hyperparameters (`nlp.product_classifier`, `nlp.issue_classifier`, `nlp.topic_model`)
* Confidence / human-review threshold (`nlp.confidence_threshold`)
* NLP tuning parameters

All modules, including `src/nlp_model_training.py` and
`src/nlp_tuning.py`, read their settings from `config.yaml` rather than
hardcoding them.

---

## 12. Logging Layer

Logging is centralized in:

```text
src/logger.py
```

The shared logger writes to:

* Console
* Rotating log file

Log file behavior:

```text
logs/app.log
Max size: 10 MB
Backups: 5
Maximum retained logs: ~50 MB
```

The logger records:

* Stage start
* Stage completion
* Important intermediate counts
* Errors and exceptions

This makes pipeline failures easier to trace.

---

## 13. Exception Handling Layer

Custom exceptions are defined in:

```text
src/exception.py
```

Each pipeline module wraps failures with contextual error information.

The goal is to make errors easier to debug by surfacing:

* Original exception message
* File name
* Line number
* Pipeline stage context

This is especially useful when `pipeline.py` runs multiple modules sequentially.

---

## 14. Testing Layer

Tests live under:

```text
tests/
```

The current test suite covers:

* Config loading
* Custom exception behavior
* Growth-rate calculation
* Risk-score scaling
* Forecast metric calculation
* Recommendation helper logic
* Required-column validation
* Canonical taxonomy mapping (`tests/test_taxonomy.py`)

Current status:

```text
19/19 tests passing
```

Testing limitation:

The current suite contains unit tests, but not a full end-to-end integration test against a small fixture dataset.

---

## 15. Memory Optimization Strategy

The project uses multiple memory-control techniques.

### Chunked CSV Ingestion

The raw 8-9 GB CSV is read in chunks instead of loading the full file into memory.

### Parquet Storage

The processed dataset is stored as Parquet.

```text
Raw CSV: 8-9 GB
Processed Parquet: ~1.3 GB
Approximate reduction: ~85%
```

### Column Pruning

Downstream modules read only the columns they need:

```python
pd.read_parquet(path, columns=[...])
```

### PyArrow Dataset Scanning

`create_narrative_training_data.py` uses PyArrow scanning to extract narrative rows without loading the full processed dataset into memory.

### Bounded NLP Sampling

NLP training uses a capped sample size instead of vectorizing all ~3.8M narratives.

### Pre-Aggregation

The dashboard reads small summary Parquet files instead of aggregating millions of rows during user interaction.

---

## 16. Data Handling and Privacy

The raw CFPB CSV is excluded from:

* Git repository
* Docker image

The Docker container serves only:

* Pre-computed dashboard summaries
* Trained NLP model artifacts

The CFPB dataset itself redacts many direct identifiers in complaint narratives, commonly replacing sensitive values with `XXXX`.

This project does not perform an additional custom PII-redaction pass beyond the source dataset.

---

## 17. Scaling Limits

The current architecture is designed for single-machine execution.

It works for the current project scale:

```text
15.95M rows
8-9 GB raw CSV
~1.3 GB processed Parquet
```

Likely scaling bottlenecks at larger scale:

* Sequential module execution
* Single-machine CPU processing
* Single-machine memory limits
* Local filesystem storage
* Manual pipeline orchestration

---

## 18. Future Scalability Options

Possible future upgrades:

| Area             | Upgrade                               |
| ---------------- | ------------------------------------- |
| Data processing  | Spark or DuckDB                       |
| Orchestration    | Airflow, Prefect, Dagster             |
| Storage          | PostgreSQL, DuckDB, object storage    |
| NLP serving      | FastAPI                               |
| Model governance | MLflow model registry                 |
| Monitoring       | Evidently, Prometheus, Grafana        |
| Deployment       | Kubernetes or cloud container service |
| Data quality     | Great Expectations / Pandera          |

These are not required for the current portfolio version but represent logical next steps for a larger enterprise deployment.

---

## 19. Design Summary

The architecture prioritizes:

* An ML/NLP core (canonical taxonomy, confidence-aware classification, unsupervised discovery) as the project's identity, with analytics as a supporting layer
* Memory-safe processing
* Pre-computed dashboard artifacts
* CPU-only NLP
* Transparent models
* Config-driven behavior
* Clear pipeline boundaries
* Docker-based reproducibility
* Honest documentation of limitations

The result is a production-style ML/NLP system rather than a notebook-only analysis project.
