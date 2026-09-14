# Runbook

Step-by-step operational commands for running every part of this platform.

All commands assume you are in the project root directory with the virtual environment activated.

---

## 1. First-Time Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the raw CFPB CSV at the path configured in `config.yaml`.

Default path:

```text
data/raw/complaints.csv
```

---

## 2. Preprocessing

Run this once per raw data update.

```bash
python -m src.preprocessing
```

### What it does

* Reads the raw CFPB CSV in chunks
* Validates required columns
* Handles missing values
* Parses date columns
* Creates date features
* Removes duplicate complaints by `Complaint ID`
* Writes the processed Parquet dataset

Output:

```text
data/processed/complaints_processed.parquet
```

### When to run

* First project setup
* Whenever the raw CSV changes
* Whenever preprocessing logic changes

### Expected failure modes

| Error                                  | Cause                                                         | Recovery                                                                |
| -------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `FileNotFoundError`                    | Raw CSV is not at the configured path                         | Place the file at `data/raw/complaints.csv` or update `config.yaml`     |
| `ValueError: Missing required columns` | Raw CSV schema does not match expected CFPB columns           | Check column names against `REQUIRED_COLUMNS` in `src/preprocessing.py` |
| Empty processed output                 | Raw file loaded but rows were invalid or filtered incorrectly | Re-check raw data path, CSV delimiter, and preprocessing logs           |

---

## 3. Run the Full Analytics Pipeline

```bash
python -m src.pipeline
```

### What it does

Runs the analytics pipeline in this order:

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

Outputs are written to:

```text
data/processed/dashboard/
```

### When to run

* After preprocessing
* When new processed data is available
* When dashboard summaries need to be refreshed
* When risk, growth, forecast, or recommendation outputs need updating

### Expected failure modes

| Error                                               | Cause                                                 | Recovery                                                                    |
| --------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------- |
| `FileNotFoundError: Processed data not found`       | Preprocessing has not been run                        | Run `python -m src.preprocessing` first                                     |
| `FileNotFoundError: Monthly trend file not found`   | Forecasting ran before growth analysis                | Run the full pipeline, or run `python -m src.growth_analysis` first         |
| `ValueError: Processed complaints dataset is empty` | Processed Parquet exists but has no rows              | Re-check preprocessing output                                               |
| Missing recommendation inputs                       | Risk, growth, forecast, or driver outputs are missing | Run `python -m src.pipeline` instead of running recommendation engine alone |

### Note

This command does not train or retrain NLP models. NLP training is separate because it is slower and does not need to run on every analytics refresh.

---

## 4. Run Individual Pipeline Stages

Use these commands when debugging one stage without re-running the full pipeline.

```bash
python -m src.dashboard_data
python -m src.risk_analysis
python -m src.driver_analysis
python -m src.growth_analysis
python -m src.forecasting
python -m src.recommendation_engine
```

### Stage dependencies

| Stage                   | Required input                                             |
| ----------------------- | ---------------------------------------------------------- |
| `dashboard_data`        | `data/processed/complaints_processed.parquet`              |
| `risk_analysis`         | `data/processed/complaints_processed.parquet`              |
| `driver_analysis`       | `data/processed/complaints_processed.parquet`              |
| `growth_analysis`       | `data/processed/complaints_processed.parquet`              |
| `forecasting`           | `data/processed/dashboard/monthly_complaint_trend.parquet` |
| `recommendation_engine` | risk, growth, forecast, and driver outputs                 |

---

## 5. Train or Retrain NLP Models

NLP training is separate from the main analytics pipeline.

Run it only when you want to retrain the Product classifier, Issue classifier, or topic model.

---

### Step 5a: Build Narrative Training Data

```bash
python -m src.create_narrative_training_data
```

### What it does

* Reads `complaints_processed.parquet`
* Keeps rows with valid complaint narrative, Product, and Issue
* Writes narrative training data

Output:

```text
data/processed/narratives_training.parquet
```

---

### Step 5b: Train Fixed-Hyperparameter NLP Models

```bash
python -m src.nlp_model_training
```

### What it does

* Trains Product classifier
* Trains Issue classifier
* Trains LDA topic model
* Saves model artifacts to `models/nlp/`

Outputs:

```text
models/nlp/product_classifier_model.pkl
models/nlp/product_classifier_vectorizer.pkl
models/nlp/issue_classifier_model.pkl
models/nlp/issue_classifier_vectorizer.pkl
models/nlp/topic_model.pkl
models/nlp/topic_vectorizer.pkl
models/nlp/topic_words.pkl
```

---

### Step 5c: Hyperparameter Tuning Alternative

```bash
python -m src.nlp_tuning
```

### What it does

* Searches a fixed 6-combination grid
* Tunes Product classifier
* Tunes Issue classifier
* Saves the best Product and Issue models

### Important

`nlp_model_training.py` and `nlp_tuning.py` write to the same output filenames for Product and Issue models.

Whichever script runs last becomes the model loaded by `nlp_predictor.py`.

`nlp_tuning.py` does not train the topic model. If the topic model is missing, run `nlp_model_training.py` at least once.

---

## 6. Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Default URL:

```text
http://localhost:8501
```

### Requirements

The dashboard expects:

```text
data/processed/dashboard/*.parquet
models/nlp/*.pkl
```

If dashboard files or model artifacts are missing, run the relevant pipeline or training step first.

---

## 7. Run Tests

```bash
python -m pytest tests -v
```

Expected result:

```text
19 passed
```

The test suite covers core utility functions, including:

* Growth-rate calculation
* Risk-score scaling
* Recommendation helper logic
* Required-column validation

---

## 8. Docker

### Build Image

```bash
docker build -t customer-complaint-intelligence:v1 .
```

### Run Local Image

```bash
docker run -p 8501:8501 customer-complaint-intelligence:v1
```

### Pull Pre-Built Image

Docker Hub:

```text
https://hub.docker.com/repository/docker/<your-dockerhub-username>/customer-complaint-intelligence
```

```bash
docker pull <your-dockerhub-username>/customer-complaint-intelligence:latest
docker run -p 8501:8501 <your-dockerhub-username>/customer-complaint-intelligence:latest
```

### Verify Container

```bash
docker ps
```

Then use the container ID:

```bash
docker logs -f <container_id>
```

### Common Port Conflict

If port `8501` is already allocated:

```bash
docker run -p 8502:8501 customer-complaint-intelligence:v1
```

Then open:

```text
http://localhost:8502
```

Full Docker troubleshooting is documented in:

```text
docs/docker.md
```

---

## 9. Verify Outputs

After running the analytics pipeline, verify that expected outputs exist.

### Dashboard Outputs

```bash
ls data/processed/dashboard/
```

Expected important files:

```text
kpis.parquet
top_products.parquet
top_issues.parquet
top_companies.parquet
company_risk_score.parquet
product_growth.parquet
issue_growth.parquet
monthly_complaint_trend.parquet
complaint_forecast.parquet
forecast_summary.parquet
recommendations.parquet
executive_action_plan.parquet
```

### NLP Models

```bash
ls models/nlp/
```

Expected files:

```text
product_classifier_model.pkl
product_classifier_vectorizer.pkl
issue_classifier_model.pkl
issue_classifier_vectorizer.pkl
topic_model.pkl
topic_vectorizer.pkl
topic_words.pkl
```

### Verify Tests

```bash
python -m pytest tests -v
```

Expected:

```text
19 passed
```

### Verify Docker Image

```bash
docker images
```

Expected repository:

```text
customer-complaint-intelligence
```

---

## 10. Recovery Guide

What to do when a stage fails.

---

### Pipeline Fails

| Error                                               | Cause                                          | Recovery                                                              |
| --------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------- |
| `FileNotFoundError: Processed data not found`       | Preprocessing was not run                      | Run `python -m src.preprocessing`                                     |
| `FileNotFoundError: Monthly trend file not found`   | Forecasting ran before growth analysis         | Run `python -m src.growth_analysis`, then `python -m src.forecasting` |
| `ValueError: Processed complaints dataset is empty` | Processed dataset has zero rows                | Re-check raw CSV and preprocessing logs                               |
| `FileNotFoundError: Required file not found`        | Recommendation engine dependencies are missing | Run `python -m src.pipeline`                                          |

---

### NLP Training Fails

| Error                                                        | Cause                                        | Recovery                                                   |
| ------------------------------------------------------------ | -------------------------------------------- | ---------------------------------------------------------- |
| `FileNotFoundError: Processed data not found`                | Processed Parquet does not exist             | Run preprocessing first                                    |
| `ValueError: No valid narrative rows found for NLP training` | No valid narrative rows after filtering      | Check narrative availability and preprocessing output      |
| `ValueError: Not enough classes`                             | Too few valid target classes after filtering | Increase NLP sample size or reduce minimum class threshold |
| Stale predictions in dashboard                               | Different NLP script overwrote model files   | Re-run the intended training script                        |

---

### Docker Fails

| Error                                                     | Cause                                                    | Recovery                                        |
| --------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------- |
| `Bind for 0.0.0.0:8501 failed: port is already allocated` | Port 8501 already in use                                 | Use `docker run -p 8502:8501 ...`               |
| Container exits immediately                               | Missing model/dashboard artifacts or dependency mismatch | Check `docker logs -f <container_id>`           |
| NLP model loading error                                   | `numpy` / `scikit-learn` / `joblib` mismatch             | Pin versions and rebuild image                  |
| Dashboard shows missing summary files                     | Pipeline outputs were not copied into the image          | Run pipeline locally, then rebuild Docker image |
| Streamlit theme missing                                   | `.streamlit/config.toml` missing from image              | Confirm file exists and rebuild image           |

---

## 11. Typical Fresh Setup Sequence

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Place raw CSV at data/raw/complaints.csv

# 3. Process raw data
python -m src.preprocessing

# 4. Run analytics pipeline
python -m src.pipeline

# 5. Build narrative training data
python -m src.create_narrative_training_data

# 6. Train NLP models
python -m src.nlp_tuning

# 7. Launch dashboard
streamlit run app/streamlit_app.py
```

Windows activation alternative:

```bash
venv\Scripts\activate
```

---

## 12. Quick Local Validation Checklist

Before pushing or recording a demo, run:

```bash
python -m pytest tests -v
docker build -t customer-complaint-intelligence:v1 .
docker run -p 8501:8501 customer-complaint-intelligence:v1
```

Then verify:

```text
Dashboard opens at http://localhost:8501
Executive dashboard loads
Risk dashboard loads
Forecasting tab loads
NLP prediction tab loads
Recommendation tab loads
```
