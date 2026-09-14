# Machine Learning / NLP Customer Complaint Intelligence System — Project Summary

## 1. Project Overview

**Machine Learning / NLP Customer Complaint Intelligence System** is an ML/NLP project built on the CFPB Consumer Complaint dataset, where classification, taxonomy canonicalization, confidence-aware prediction, and unsupervised topic discovery are the core identity. An executive analytics, risk-scoring, and forecasting layer supports the ML/NLP core rather than being the project's headline.

The project processes a large raw complaint dataset and converts it into:

- canonical Product/Issue taxonomy that fixes CFPB label fragmentation
- NLP-based Product and Issue classification with confidence scoring and human-review flagging
- multi-issue detection infrastructure (heuristic today, with a documented plan for a validated multi-label model)
- unsupervised topic discovery (recurring themes, clusters, emerging issues)
- rigorous Macro-F1 / Top-K evaluation with automatic error analysis
- executive dashboard insights
- company risk scores
- complaint driver analysis
- product and issue growth signals
- complaint volume forecasting
- recommendation engine outputs

The goal is not only to show charts, but to build a practical, ML-first decision-support system for understanding complaint patterns and operational risk.

---

## 2. Problem Statement

Consumer complaint data is difficult to use directly because it is large, messy, and operationally complex.

A basic dashboard can show complaint counts, but decision-makers usually need deeper answers:

- Which companies show higher complaint risk?
- Which products or issues are growing quickly?
- Which complaint categories may need operational attention?
- What complaint volume may look like in the near future?
- Can incoming complaint narratives be classified automatically?
- Can analytical outputs be converted into recommendations?

This project solves these problems through a modular analytics and NLP pipeline.

---

## 3. Dataset

| Item | Details |
|---|---|
| Dataset | CFPB Consumer Complaint Database |
| Total complaints | ~15.95M |
| Raw data size | ~8–9 GB CSV |
| Narrative availability | ~23.84% |
| Companies | ~7.97K |
| Products | 21 |
| States / territories | 64 |

Important note: Complaint volume should not be treated as a legal or regulatory conclusion by itself. A high complaint count may be affected by company size, customer base, product mix, and population.

---

## 4. What the System Does

### ML / NLP Core

- **Canonical taxonomy layer** — collapses CFPB Product/Issue labels fragmented by historical taxonomy renames (18 → 11 Product classes, 75 → 63 Issue classes) while preserving the original raw label (see `docs/taxonomy_audit.md`)
- **Product classification** from complaint narratives, with confidence score, top alternatives, and an AUTO_ACCEPT / HUMAN_REVIEW decision
- **Issue classification** from complaint narratives, same confidence-aware treatment
- **Multi-issue detection infrastructure** — heuristic extraction of additional plausible issues from the existing classifier, clearly flagged as unvalidated, plus a documented plan for building and validating a real multi-label model (`docs/multi_issue_plan.md`)
- **Topic Discovery Layer** — LDA topic modeling elevated from a per-complaint feature into discovery of recurring themes, clusters, and (once timestamped narrative data is available) emerging issues over time
- **Rigorous evaluation** — Macro F1, Weighted F1, Top-K accuracy, and automatic error analysis of which categories the model confuses and why, instead of accuracy alone
- Interactive complaint analyzer (Streamlit)

### Business Analytics (Supporting Layer)

Everything below is unchanged from the original platform and remains fully functional — it now supports the ML/NLP core instead of being the project's main identity.

**Executive Analytics**

- Executive KPI dashboard
- Product-level analysis
- Issue-level analysis
- Company-level analysis
- Resolution and timeliness analysis
- State/geography analysis
- Consumer segment analysis
- Narrative availability and text insights

**Advanced Intelligence**

- Company Risk Score
- Complaint Driver Analysis
- Product and Issue Growth Analysis
- Monthly complaint forecasting
- Executive recommendation inputs

**Recommendation Engine**

The recommendation engine combines analytical signals such as company risk, growth trends, forecasts, and complaint drivers into prioritized action recommendations.

---

## 5. Architecture

```text
Raw CFPB CSV
15.95M complaints / 8–9 GB
        |
        v
Chunked Preprocessing
Validation + cleaning + feature engineering
        |
        v
Processed Parquet Outputs
        |
        v
Dashboard Aggregations
        |
        +--> Executive Analytics
        +--> Company Risk Scoring
        +--> Growth Analysis
        +--> Driver Analysis
        +--> Forecasting
        +--> Canonical Taxonomy --> NLP Classification --> Confidence / Human Review
        +--> Topic Modeling --> Issue Discovery Layer
        |
        v
Recommendation Engine
        |
        v
Streamlit Dashboard
        |
        v
Docker Deployment
```

The dashboard reads pre-aggregated Parquet outputs instead of loading the full raw dataset at runtime. This keeps the Streamlit app faster and more memory-efficient.

---

## 6. Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data processing | Pandas, PyArrow, Parquet |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Machine Learning | scikit-learn |
| NLP | TF-IDF, Logistic Regression, LDA |
| Forecasting | Prophet |
| Testing | pytest |
| Deployment | Docker, Docker Hub |
| CI/CD | GitHub Actions |
| Config | YAML-based configuration |

---

## 7. Key Results

| Metric | Result |
|---|---:|
| Product classifier accuracy | 75.28% |
| Issue classifier accuracy | 62.39% |
| Forecast validation MAPE | 3.57% |
| Forecast validation MAE | 17,748 |
| Unit tests | 19/19 passing |
| Timely response rate | 99.37% |
| Average resolution delay | 0.63 days |
| Narrative availability | 23.84% |

These results should be interpreted with the documented limitations. For example, NLP performance applies only to complaints with available narrative text.

---

## 8. Key Engineering Decisions

| Decision | Why It Was Used |
|---|---|
| Chunked preprocessing | To process a large raw CSV without memory failures |
| Parquet storage | Smaller storage and faster column-based reads |
| Pre-aggregated dashboard files | Keeps Streamlit responsive |
| TF-IDF + Logistic Regression | CPU-friendly, explainable, fast baseline |
| Prophet forecasting | Suitable for monthly trend and seasonality forecasting |
| Config-driven pipeline | Keeps thresholds and paths outside code |
| Docker deployment | Reproducible local and portfolio demo setup |
| GitHub Actions | Basic CI validation before changes reach main branch |

---

## 9. Why Not Deep Learning First?

A transformer model could be used, but this project prioritizes:

- CPU-friendly deployment
- faster training and inference
- lower memory usage
- reproducibility on a normal laptop
- easier explanation during interviews

TF-IDF + Logistic Regression is a practical baseline for this project because it gives reasonable performance while keeping the system deployable without GPU dependency.

---

## 10. Current Limitations

This project is intentionally honest about its limitations:

- Geographic complaint counts are not normalized per capita.
- The currently-shipped Product/Issue models are trained on raw CFPB labels; canonical-taxonomy training is implemented (`src/nlp_model_training.py`) but hasn't been exercised on real data since the raw narrative dataset isn't bundled with this repository.
- Multi-issue detection is an explicitly-flagged heuristic, not a validated multi-label model (see `docs/multi_issue_plan.md`).
- Similar Complaints search and time-based emerging-issue detection are implemented but need a narrative-text corpus to run against.
- NLP confidence scores are taxonomy-aware but not formally calibrated probabilities.
- Forecasting uses complaint volume only and does not include external economic or regulatory signals.
- Topic modeling uses LDA with manually interpreted topics.
- FastAPI serving layer is planned, not part of the current final version.

---

## 11. Future Improvements

The most useful future improvements are:

1. Retrain Product/Issue classifiers on the canonical taxonomy against real data.
2. Validate multi-issue detection against real narrative data (`docs/multi_issue_plan.md`).
3. Populate the narrative corpus needed for Similar Complaints search and emerging-issue detection.
4. Formally calibrate confidence scores.
5. Add per-capita geographic normalization.
6. Add FastAPI model serving.
7. Add model metadata and versioning.
8. Add model drift monitoring.
9. Add stronger runtime and memory benchmarks.

---

## 12. Interview Pitch

This project processes a large real-world complaint dataset and turns it into an ML/NLP-first decision-support system. I found that CFPB's Product/Issue taxonomy had been renamed several times since 2012, fragmenting the same real category across multiple raw labels — I built a canonical taxonomy layer that fixes this, backed by real numbers from the project's own trained models (Product 18→11 classes, Issue 75→63). Product and Issue classification return a confidence score and an auto-accept/human-review decision instead of a bare label, and an LDA topic model acts as a discovery layer for recurring and emerging complaint themes. All of this sits on top of a supporting analytics layer — company risk scoring, growth detection, Prophet-based forecasting, and recommendations — built with chunked preprocessing and Parquet outputs to handle scale on a normal laptop. The result is reproducible through Docker and documented for evaluation, including an explicit, honest line between what's validated and what's a flagged heuristic with a documented plan.

---

## 13. Repository Documents

| File | Purpose |
|---|---|
| `README.md` | Main project explanation and setup |
| `PROJECT_SUMMARY.md` | Short recruiter/interviewer overview |
| `docs/INTERVIEW_QA.md` | Interview preparation questions and answers |
| `docs/architecture.md` | System architecture and module flow |
| `docs/evaluation.md` | Model metrics and validation details |
| `docs/taxonomy_audit.md` | CFPB label taxonomy audit (canonical mapping) |
| `docs/multi_issue_plan.md` | Multi-issue detection: current heuristic and validation plan |
| `docs/case_study.md` | Engineering journey and trade-offs |
| `docs/docker.md` | Docker build and deployment guide |
| `docs/data_dictionary.md` | Dataset columns and engineered features |
| `docs/runbook.md` | Operational commands |
| `docs/benchmark.md` | Performance and benchmark notes |
| `docs/adr.md` | Architecture decision records |

---

## Author

**Snaa Alnaggar** — Author
