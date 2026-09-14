# Data Dictionary

This document describes every column used by the Machine Learning / NLP Customer Complaint Intelligence System — both the raw CFPB columns and the derived / engineered columns created during preprocessing.

Source module:

```text
src/preprocessing.py
```

---

## 1. Data Lineage

The main analytics pipeline follows this flow:

```text
CFPB complaints.csv
        ↓
src/preprocessing.py
        ↓
data/processed/complaints_processed.parquet
        ↓
src/dashboard_data.py
src/risk_analysis.py
src/driver_analysis.py
src/growth_analysis.py
src/forecasting.py
src/recommendation_engine.py
        ↓
data/processed/dashboard/*.parquet
        ↓
app/streamlit_app.py
```

The NLP pipeline branches from the processed dataset:

```text
data/processed/complaints_processed.parquet
        ↓
src/create_narrative_training_data.py
        ↓
data/processed/narratives_training.parquet
        ↓
src/nlp_model_training.py / src/nlp_tuning.py
        ↓
models/nlp/*.pkl
        ↓
src/nlp_predictor.py
        ↓
Complaint Analyzer tab in Streamlit
```

---

## 2. Dataset Storage Summary

| Property                         | Value           |
| -------------------------------- | --------------- |
| Raw dataset size                 | 8–9 GB CSV      |
| Processed dataset size           | ~1.3 GB Parquet |
| Storage reduction                | ~85%            |
| Total complaints                 | 15.95M          |
| Companies                        | 7.97K           |
| Products                         | 21              |
| States / territories             | 64              |
| Narrative availability           | 23.84%          |
| Approximate narratives available | ~3.8M           |

The platform uses Parquet storage, column pruning, and pre-aggregated dashboard outputs to reduce memory pressure and improve dashboard responsiveness.

---

## 3. Raw Columns

These are the columns required by `validate_columns()` before preprocessing begins.

| Column                         | Type              | Description                                                | Business Meaning                                                             | Missing-Value Handling                                                          |
| ------------------------------ | ----------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `Date received`                | string → datetime | Date the CFPB received the complaint                       | Main time-series field used for yearly/monthly trends and forecasting inputs | Parsed with `format="mixed"` and `errors="coerce"`; invalid values become `NaT` |
| `Product`                      | string            | High-level financial product category                      | Primary business dimension for product-level complaint analysis              | Required for most analytics; retained as-is when present                        |
| `Sub-product`                  | string            | More specific category under `Product`                     | Finer product segmentation                                                   | Missing values filled with `"Unknown"`                                          |
| `Issue`                        | string            | Main issue category                                        | Root-cause category for complaint analysis                                   | Missing values filled with `"Unknown"`                                          |
| `Sub-issue`                    | string            | More specific issue category                               | Fine-grained complaint driver analysis                                       | Missing values filled with `"Unknown"`                                          |
| `Consumer complaint narrative` | string            | Consumer-provided free-text complaint narrative            | Input for NLP classification and topic modeling                              | Left missing when not provided; narrative availability is tracked separately    |
| `Company public response`      | string            | Public response provided by the company                    | Transparency / public-response signal                                        | Missing values filled with `"No public response"`                               |
| `Company`                      | string            | Company the complaint was filed against                    | Entity used for company-level analytics and risk scoring                     | Missing company values are not useful for risk scoring                          |
| `State`                        | string            | Consumer state or territory                                | Geographic analysis dimension                                                | Missing values filled with `"Unknown"`                                          |
| `ZIP code`                     | string            | Consumer ZIP code, often partially redacted                | Fine geographic field, currently not heavily used in analytics               | Missing values filled with `"Unknown"`                                          |
| `Tags`                         | string            | Consumer group tag such as Older American or Servicemember | Consumer segment / vulnerable-population signal                              | Missing values mapped to `"Normal Consumer"`                                    |
| `Submitted via`                | string            | Channel used to submit the complaint                       | Channel mix analysis                                                         | Retained as-is                                                                  |
| `Date sent to company`         | string → datetime | Date the complaint was sent to the company                 | Used to calculate routing / resolution delay                                 | Parsed with `format="mixed"` and `errors="coerce"`                              |
| `Company response to consumer` | string            | Final response category from company to consumer           | Resolution outcome signal                                                    | Missing values filled with `"Unknown"`                                          |
| `Timely response?`             | string            | Whether the company responded on time                      | Compliance-related KPI and risk-score input                                  | Retained as-is                                                                  |
| `Complaint ID`                 | integer/string    | Unique complaint identifier                                | Deduplication key                                                            | Used to remove duplicate complaint records                                      |

---

## 4. Derived / Engineered Columns

These columns are created during preprocessing and do not exist in the raw CSV.

| Column                 | Derived From                           | Logic                                       | Business Meaning                               |
| ---------------------- | -------------------------------------- | ------------------------------------------- | ---------------------------------------------- |
| `Year`                 | `Date received`                        | Extract year using `.dt.year`               | Yearly trend analysis                          |
| `Month`                | `Date received`                        | Extract month using `.dt.month`             | Monthly trend and seasonality                  |
| `Quarter`              | `Date received`                        | Extract quarter using `.dt.quarter`         | Quarterly reporting                            |
| `Day`                  | `Date received`                        | Extract day of month using `.dt.day`        | Fine-grained date analysis                     |
| `Day_Name`             | `Date received`                        | Extract weekday name using `.dt.day_name()` | Day-of-week submission pattern                 |
| `Resolution_Delay`     | `Date sent to company - Date received` | Difference in days                          | Operational delay proxy used in risk scoring   |
| `Has_Narrative`        | `Consumer complaint narrative`         | `1` if narrative exists, else `0`           | NLP eligibility and narrative availability KPI |
| `Narrative_Length`     | `Consumer complaint narrative`         | Character count; `0` if missing             | Complaint complexity proxy                     |
| `Narrative_Word_Count` | `Consumer complaint narrative`         | Word count; `0` if missing                  | Narrative complexity indicator                 |
| `Consumer_Group`       | `Tags`                                 | Standardized group label                    | Consumer segment reporting                     |

---

## 5. Consumer Group Mapping

`Consumer_Group` is created from `Tags`.

| Raw Tag Condition                                  | Standardized `Consumer_Group`    |
| -------------------------------------------------- | -------------------------------- |
| Missing tag                                        | `Normal Consumer`                |
| Contains `Servicemember` only                      | `Servicemember`                  |
| Contains `Older American` only                     | `Older American`                 |
| Contains both `Older American` and `Servicemember` | `Older American + Servicemember` |

This simplified field is used for consumer-segment analysis in the dashboard.

---

## 6. Dataset-Level Statistics

Measured values from the processed dataset:

| Statistic                  | Value     |
| -------------------------- | --------- |
| Total complaints           | 15.95M    |
| Companies                  | 7.97K     |
| Products                   | 21        |
| States / territories       | 64        |
| Narrative availability     | 23.84%    |
| Approximate narrative rows | ~3.8M     |
| Timely response rate       | 99.37%    |
| Average resolution delay   | 0.63 days |

---

## 7. Cardinality Summary

| Field            | Approximate Cardinality | Why It Matters                                          |
| ---------------- | ----------------------: | ------------------------------------------------------- |
| `Product`        |                      21 | Low-cardinality business dimension                      |
| `State`          |                      64 | Low-cardinality geography dimension                     |
| `Company`        |                   7.97K | High-cardinality entity dimension used for risk scoring |
| `Issue`          |                    High | NLP and driver-analysis target                          |
| `Sub-issue`      |                    High | Fine-grained root-cause analysis                        |
| `Submitted via`  |                     Low | Channel distribution analysis                           |
| `Consumer_Group` |                       4 | Consumer-segment reporting                              |

Low-cardinality columns are good candidates for categorical optimization in memory-constrained environments.

---

## 8. Columns Used by Each Module

| Module                              | Columns Read                                                                                                                                                                                                    |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dashboard_data.py`                 | `Year`, `Company`, `Product`, `State`, `Timely response?`, `Resolution_Delay`, `Has_Narrative`, `Narrative_Word_Count`, `Issue`, `Sub-issue`, `Company response to consumer`, `Submitted via`, `Consumer_Group` |
| `risk_analysis.py`                  | `Company`, `Timely response?`, `Resolution_Delay`                                                                                                                                                               |
| `driver_analysis.py`                | `Product`, `Issue`, `Sub-issue`                                                                                                                                                                                 |
| `growth_analysis.py`                | `Year`, `Month`, `Product`, `Issue`                                                                                                                                                                             |
| `forecasting.py`                    | Reads `monthly_complaint_trend.parquet` generated by `growth_analysis.py`                                                                                                                                       |
| `create_narrative_training_data.py` | `Consumer complaint narrative`, `Product`, `Issue`                                                                                                                                                              |
| `nlp_model_training.py`             | `Consumer complaint narrative`, `Product`, `Issue`                                                                                                                                                              |
| `nlp_tuning.py`                     | `Consumer complaint narrative`, `Product`, `Issue`                                                                                                                                                              |
| `nlp_predictor.py`                  | Loads trained model artifacts from `models/nlp/`                                                                                                                                                                |
| `recommendation_engine.py`          | Reads risk, growth, driver, and forecast Parquet outputs                                                                                                                                                        |

---

## 9. Important Business KPIs

| KPI                      | Source Field(s)                                   | Meaning                                            |
| ------------------------ | ------------------------------------------------- | -------------------------------------------------- |
| Total Complaints         | `Complaint ID`                                    | Overall complaint volume                           |
| Timely Response Rate     | `Timely response?`                                | Share of complaints responded to on time           |
| Average Resolution Delay | `Date received`, `Date sent to company`           | Average routing / response delay proxy             |
| Narrative Availability   | `Consumer complaint narrative`                    | Share of complaints eligible for NLP analysis      |
| Company Risk Score       | `Company`, `Timely response?`, `Resolution_Delay` | Composite company-level operational risk indicator |
| Product Growth           | `Product`, `Year`, `Month`                        | Product-level trend signal                         |
| Issue Growth             | `Issue`, `Year`, `Month`                          | Issue-level trend signal                           |

---

## 10. Data Quality Notes

### Narrative Availability

Narrative availability is only **23.84%**, which means the NLP pipeline runs on a subset of the full dataset.

The analytics modules operate on the full 15.95M complaint records, while NLP modules operate only on complaints with valid narrative text.

### ZIP Code Redaction

`ZIP code` is often partially redacted in CFPB data. This is expected and is not introduced by the pipeline.

This project currently does not rely on ZIP-level analysis.

### State-Level Analysis

State-level complaint counts are raw totals and are not population-normalized.

This means larger states may naturally show higher complaint counts even if their per-resident complaint rate is lower.

### Company Name Variation

Company names may include naming variations or legal suffixes.

This project uses the CFPB-provided `Company` field directly and does not perform advanced company-entity resolution.

### Rare NLP Classes

Rare Product or Issue classes below the configured sample threshold are dropped during NLP training.

This improves metric stability but means the model cannot predict classes that were excluded from training.

---

## 11. Interpretation Caveats

* Complaint volume does not equal confirmed wrongdoing.
* High complaint count does not automatically mean high complaint rate.
* NLP predictions are probabilistic model outputs, not legal judgments.
* Topic labels are manually interpreted from LDA keywords.
* Forecasting is based only on historical complaint volume.
* Geographic totals should not be interpreted as per-capita complaint rates.

---

## 12. Summary

This data dictionary defines the raw fields, engineered fields, business meaning, module usage, and known data-quality constraints behind the Machine Learning / NLP Customer Complaint Intelligence System.

It is intended to make the pipeline easier to audit, reproduce, maintain, and explain during technical review or interview discussion.
