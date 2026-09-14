# Machine Learning / NLP Customer Complaint Intelligence System — Interview Q&A

## 1. What problem does this project solve?

This project turns a large consumer complaint dataset into an ML/NLP-first decision-support system, with a supporting business analytics layer.

Instead of only showing complaint counts, it helps answer business questions such as:

- Can incoming complaint narratives be classified automatically, with a confidence score you can trust?
- Which raw category labels are actually the same category renamed by taxonomy changes over time?
- What recurring or emerging complaint themes exist that no fixed label captures?
- Which companies have higher complaint risk?
- Which products or issues are growing?
- What complaint volume may look like in the future?
- What actions should an operations or compliance team prioritize?

---

## 2. Why did you choose the CFPB Consumer Complaint dataset?

I chose it because it is large, messy, public, and realistic. It contains millions of complaint records across products, companies, issues, geographies, responses, and complaint narratives.

This made it suitable for showing both data engineering and machine learning skills.

---

## 3. What was the biggest engineering challenge?

The biggest challenge was processing a very large raw CSV on a normal laptop.

The raw dataset is around 8–9 GB and contains about 15.95M complaints. Loading everything directly into memory can cause performance or memory issues.

I handled this by using:

- chunked ingestion
- column selection
- cleaning and validation
- Parquet outputs
- pre-aggregated dashboard files

This allowed the Streamlit dashboard to stay lightweight.

---

## 4. Why did you use Parquet instead of only CSV?

Parquet is columnar and compressed, so it is usually more efficient for analytical workloads.

In this project, Parquet helped with:

- smaller processed storage
- faster reads for selected columns
- better dashboard performance
- easier separation of raw and processed data

CSV is good for raw exchange, but Parquet is better for repeated analytical reads.

---

## 5. Why did you pre-aggregate dashboard files?

The full dataset is too large to load every time the dashboard refreshes.

So the heavy computation is done in the pipeline stage, and the dashboard reads smaller processed summary files. This makes the Streamlit app faster and more stable.

---

## 6. What are the main modules in the project?

The project is built around an ML/NLP core, with a supporting analytics layer:

**ML/NLP Core**
1. Canonical Taxonomy — collapses CFPB label fragmentation before classification
2. Classification — Product/Issue prediction with confidence + human review
3. Multi-Issue Infrastructure — heuristic candidate extraction, plus a documented plan for a real multi-label model
4. Topic Discovery — unsupervised recurring/emerging theme detection

**Business Analytics (Supporting Layer)**
5. Executive Analytics — KPIs and dashboard views
6. Advanced Intelligence — risk scoring, drivers, growth, forecasting
7. Recommendation Engine — converts signals into suggested actions

The dashboard's tab order reflects this: Complaint Analyzer, Classification, Similar Complaints, Issue Discovery, and Model Performance come before the "Business Analytics" tab that houses the rest.

---

## 7. How does the company risk score work?

The company risk score combines multiple complaint-related signals instead of using complaint volume alone.

The main signals are:

- complaint volume
- untimely response rate
- average resolution delay

This is more useful than raw complaint count alone because a large company may naturally receive more complaints. Risk scoring gives a more structured way to compare companies, although it is still not a legal or regulatory conclusion.

---

## 8. Why did you add a minimum complaint threshold for risk scoring?

Without a minimum threshold, small companies with very few complaints can appear risky due to random noise.

For example, if a company has only 2 complaints and 1 is delayed, its delay rate may look very high. A minimum sample threshold reduces this kind of distortion.

---

## 9. What is driver analysis?

Driver analysis identifies the combinations of Product, Issue, and Sub-issue that contribute most to complaint volume.

It helps answer:

- What is driving complaints in a product category?
- Which issue areas should be prioritized?
- Are there repeated operational patterns?

---

## 10. What is growth analysis?

Growth analysis compares complaint volume over time and labels products or issues based on their year-over-year movement.

Example labels include:

- Stable
- Rising
- Rising Fast
- Declining
- New / Emerging

This helps identify early signals before they become major complaint volume problems.

---

## 11. Why did you use Prophet for forecasting?

Prophet is useful for time-series forecasting with trend and seasonality. Complaint volume is naturally time-based, so Prophet was a practical choice for monthly forecasting.

I also validated the forecast on a holdout period instead of only fitting on all historical data.

---

## 12. What does Forecast MAPE mean?

MAPE stands for Mean Absolute Percentage Error. It measures forecast error as a percentage.

A lower MAPE means the forecast is closer to the actual values. In this project, the validation MAPE is documented as 3.57%, which means the model performed well on the selected validation setup.

---

## 13. Why did you use TF-IDF + Logistic Regression instead of BERT?

I used TF-IDF + Logistic Regression because it is:

- CPU-friendly
- fast to train
- fast at inference
- easier to explain
- reproducible on consumer hardware
- strong as a baseline for text classification

A transformer model like BERT may improve performance, but it would increase training time, inference cost, and deployment complexity.

For this project, the goal was a practical end-to-end platform, not only the highest possible NLP score.

---

## 14. What were the NLP results?

The documented results (currently-shipped models, trained on raw labels) are:

| Model | Metric |
|---|---:|
| Product classifier | 75.28% accuracy |
| Issue classifier | 62.39% accuracy |

The Product classifier performs better because Product categories are broader and easier to separate. Issue classification is harder because issue labels are more detailed and semantically overlapping.

A meaningful chunk of that overlap turned out to be a labeling artifact, not a genuine modeling difficulty: several raw labels are the same real category renamed across CFPB taxonomy versions (e.g. three different historical names all meaning "credit reporting"). A canonical taxonomy layer now collapses these (Product: 18 → 11 classes, Issue: 75 → 63 classes — see `docs/taxonomy_audit.md`), which the training pipeline uses going forward, and which the *already-shipped* models benefit from today by merging predicted probability across taxonomy-duplicate raw classes at inference time.

---

## 15. Why is Issue classifier accuracy lower than Product classifier accuracy?

Issue classification is more difficult because:

- there are more issue classes
- many issue labels have similar language — some are literally the same issue renamed (see Q14, `docs/taxonomy_audit.md`)
- complaint narratives can be noisy
- some classes may have fewer examples
- one complaint can contain signals for multiple issues (see Q28 on multi-issue)

So 62.39% is not perfect, but it is a reasonable baseline for a CPU-friendly model, and the accuracy-only framing understates it somewhat since it doesn't separate "genuinely hard" confusion from "duplicate label" confusion. `src/nlp_evaluation.py` now reports Macro F1, Weighted F1, Top-K accuracy, and an automatic "which categories get confused and why" breakdown for exactly this reason, and will apply to the Issue classifier the next time it's retrained on the canonical label set.

---

## 16. Why did you use LDA topic modeling?

LDA helps discover repeated themes in complaint narratives without needing labels.

It is useful for exploratory analysis and for understanding what kinds of words/topics appear frequently in complaint text. The topic names are manually interpreted, so they should be treated as analytical support, not perfect labels.

`src/topic_discovery.py` builds on this to compute topic prevalence across a set of complaints and detect topics whose share is growing between two periods — turning the topic model from a per-complaint side-feature into an actual discovery tool (recurring themes + emerging issues), not just "this complaint is Topic 3."

---

## 17. Why not use sentiment analysis?

Most complaint narratives are negative by nature. Generic sentiment analysis would mostly say “negative” and would not add much business value.

Classification, topic modeling, drivers, growth, and recommendations are more useful for this dataset.

---

## 18. What does the recommendation engine do?

The recommendation engine combines signals from:

- company risk scores
- product growth
- issue growth
- complaint drivers
- forecasts

It converts those signals into prioritized recommendations and executive action guidance.

This makes the project more useful than a dashboard that only shows charts.

---

## 19. How is this project different from a normal dashboard?

A normal dashboard mostly describes what happened.

This project also tries to answer:

- what is risky
- what is growing
- what may happen next
- what should be prioritized
- how to classify new complaint narratives

So it moves from descriptive analytics toward decision intelligence.

---

## 20. How did you evaluate the project?

The project evaluation includes:

- NLP accuracy metrics
- forecasting validation metrics
- unit tests
- dashboard output validation
- documented limitations
- Docker build/run validation
- CI/CD workflow validation

The README reports 19/19 unit tests passing, Product classifier accuracy of 75.28%, Issue classifier accuracy of 62.39%, and forecast validation MAPE of 3.57%.

---

## 21. What are the main limitations?

The main limitations, as of the current version:

- Geographic complaint counts are not normalized by population.
- The currently-shipped Product/Issue models are still trained on raw (pre-canonicalization) labels — the training pipeline supports canonical labels now, but retraining needs the raw narrative dataset, which isn't available in this development environment.
- Multi-issue detection is a heuristic (probability-threshold based on the existing single-label model), explicitly flagged as unvalidated — not a trained multi-label classifier. The real version needs raw narrative data to build and validate against (see `docs/multi_issue_plan.md`).
- "Similar Complaints" search and the time-based "emerging issues" view are fully implemented but need a narrative-text corpus to run against, which isn't available in this environment.
- Forecasting does not include external economic or regulatory signals.
- Topic modeling uses manually interpreted LDA topics.
- Some NLP training constants should be fully moved to config.
- FastAPI serving layer is planned for future improvement.

Resolved since the previous version of this document:
- ~~NLP predictions currently return only the top label.~~ Now returns canonical-label confidence, top alternatives, and an AUTO_ACCEPT / HUMAN_REVIEW decision (`src/nlp_predictor.py`).
- ~~NLP confidence scores are not calibrated probabilities.~~ Confidence is now taxonomy-aware (merged across duplicate raw labels) rather than raw softmax output split across near-duplicate classes — still not formally calibrated (e.g. via Platt scaling), but meaningfully more honest than before.

---

## 22. What would you improve next?

The best next improvements are:

1. Retrain the Product/Issue classifiers on the canonical taxonomy once the raw narrative dataset is available, to get real Macro F1 / Weighted F1 numbers instead of the current raw-label accuracy.
2. Run the multi-issue data-exploration step (`docs/multi_issue_plan.md` Step 1) against real narratives to decide whether a trained multi-label model is worth building.
3. Populate the narrative corpus needed for Similar Complaints search and time-based emerging-issue detection.
4. Formally calibrate confidence scores (e.g. Platt scaling / isotonic regression) rather than relying on taxonomy-merged softmax output.
5. Normalize state-level complaints per capita.
6. Add FastAPI model serving.
7. Add model versioning and metadata.
8. Add drift monitoring for NLP models.
9. Add stronger runtime and memory benchmarks.

---

## 23. Why is per-capita normalization important?

Raw complaint count by state can be misleading because larger states naturally have more people and more customers.

Per-capita normalization would make geographic comparison fairer by adjusting for population.

---

## 24. What is the most impressive part of this project for placement?

The strongest part is treating ML/NLP as the actual product, not a bolted-on feature.

It shows:

- identifying and fixing a real labeling problem (taxonomy fragmentation) using evidence from the project's own model outputs, not just training a model on whatever labels showed up
- turning raw softmax output into an honest confidence + human-review decision, instead of presenting a bare top-label prediction as if it were certain
- being explicit about the difference between a validated capability and a documented, flagged heuristic (multi-issue detection) — and having a real scientific plan for closing that gap rather than faking the result
- large-data handling
- analytics pipeline design
- forecasting
- recommendation logic
- dashboarding
- Docker deployment
- testing and documentation

This makes it stronger than a single notebook-based ML project, and stronger than a project that only reports a headline accuracy number without engaging with why the model gets things wrong.

---

## 25. How would you explain this project in 60 seconds?

This is an ML/NLP-first system built on the CFPB Consumer Complaint dataset with around 15.95M complaints, with an executive analytics layer supporting it. I found that CFPB's Product/Issue taxonomy has been renamed several times since 2012, fragmenting the same real category across multiple raw labels — I built a canonical taxonomy layer that fixes this (Product: 18→11 classes, Issue: 75→63), backed by real numbers from the project's own trained models. On top of that, Product and Issue classification return a confidence score, alternatives, and an auto-accept/human-review decision instead of just a bare label, and an LDA topic model acts as a discovery layer for recurring and emerging complaint themes. All of this sits on chunked preprocessing, Parquet storage, company risk scoring, growth analysis, and Prophet-based forecasting, packaged with Streamlit, Docker, tests, CI/CD, and documentation that's explicit about what's validated versus what's a flagged heuristic.

---

## 26. What should you not overclaim in an interview?

Do not overclaim that:

- the risk score is a legal or regulatory judgment
- the NLP model is perfect
- the multi-issue detection is a validated multi-label model — it's an explicitly-flagged heuristic on top of the single-label classifier until real ground-truth data is available (see `docs/multi_issue_plan.md`)
- the currently-shipped models were retrained on the new canonical taxonomy — they weren't yet; the canonicalization is live at inference time (real, working) but training-time retraining needs data this environment doesn't have
- the forecast includes all external factors
- topic modeling gives exact human-level categories
- complaint count alone proves company quality
- the system is a full enterprise SaaS product

The honest explanation is stronger: this is a production-style portfolio project with clear next improvements, and it's explicit about exactly which claims are backed by measured results versus which are documented plans for future work.

---

## 27. Final interview answer: why should this project be considered strong?

This project is strong because it treats ML/NLP as the actual product — finding and fixing a real labeling problem in the data, adding honest confidence and human-review to predictions instead of a bare label, and being explicit about the line between a validated capability and a flagged heuristic with a documented path forward — while still combining that with data engineering, analytics, forecasting, dashboarding, Docker deployment, testing, and documentation on a real large-scale public dataset. It explains its limitations clearly, which shows practical engineering maturity instead of only showing polished charts or an inflated accuracy number.

---

## 28. How does multi-issue detection work, and is it a real multi-label model?

No — and saying so directly is part of the answer. A complaint like "I was charged twice and the refund hasn't arrived" plausibly describes two issues, but the CFPB schema only records one `Issue` field, so there's no ground-truth multi-label data to train or validate a real classifier against in this environment.

What exists today is a heuristic: it reuses the existing single-label Issue classifier's probability distribution, groups it by canonical taxonomy category, and surfaces any additional category whose merged probability is within a configurable relative threshold of the top prediction — every result is tagged `is_heuristic: True` so it's never confused with a validated capability.

What doesn't exist yet is the real thing, and `docs/multi_issue_plan.md` documents exactly how to get there: a data-exploration step to check whether the phenomenon is even common enough to be worth modeling, weak-label construction from per-category vocabulary, training a proper multi-label model, and only swapping it in once it beats the heuristic on a small human-labeled validation sample. This is the kind of thing that's easy to fake with a plausible-sounding architecture diagram and much harder to actually validate — so the honest answer here is "here's the heuristic, here's why it isn't the real thing yet, and here's the exact plan to make it real."
