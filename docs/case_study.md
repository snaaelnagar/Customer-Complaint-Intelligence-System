# Case Study: Scalable NLP and Analytics on Resource-Constrained Infrastructure

## Problem

The CFPB dataset contains **15.95M complaints** and approximately **8-9 GB**
of raw CSV data. A normal notebook-based workflow struggles at this scale —
loading everything into memory repeatedly leads to slow iteration, RAM
pressure, and an unstable development loop.

The challenge was to build a system that behaves like a production
analytics platform while still running on a single machine with no GPU
and limited RAM.

---

## Challenge 1: Raw CSV Size

Direct, repeated Pandas loading of an 8-9 GB CSV is slow and risks
exceeding available memory, especially during iterative development where
the same file gets reloaded many times.

**Solution:**
- Chunk-based preprocessing (`pd.read_csv(..., chunksize=...)`)
- Column selection downstream (each analytics module reads only the
  columns it needs from the processed Parquet file)
- Parquet conversion for all intermediate storage
- A PyArrow dataset scanner (not plain `pandas.read_parquet`) for
  extracting the narrative-only training subset, so that step never
  loads the full processed dataset into memory at once

**Result:** Raw CSV (~8-9 GB) → processed Parquet (~1.3 GB).

---

## Challenge 2: Streamlit Performance

Streamlit re-runs the script on every interaction. Reading 15.95M rows on
every rerun would make the dashboard unusable.

**Solution:** A dedicated aggregation stage (`dashboard_data.py` and the
Business Analytics scripts) pre-computes small, dashboard-ready Parquet summaries
— `kpis.parquet`, `top_products.parquet`, `company_risk_score.parquet`,
`product_growth.parquet`, `forecast_summary.parquet`,
`recommendations.parquet`, and others. Streamlit's `load_summary()` reads
only these small files, with `@st.cache_data` on top.

**Result:** dashboard startup and interaction speed are decoupled from the
size of the underlying raw dataset.

---

## Challenge 3: NLP at 3.8M-Narrative Scale

Narrative availability is 23.84% of 15.95M complaints — still roughly
3.8M text rows. A transformer-based approach (fine-tuning BERT or
similar) was considered, but the cost in training time, RAM, Docker image
size, and CPU-only inference latency did not seem justified for this
project's goals.

**Solution:** TF-IDF + tuned Logistic Regression.

**Result:**
- Product accuracy: 75.28%
- Issue accuracy: 62.39%

This is a deliberate trade-off favoring deployability and reproducibility
over the marginal accuracy a larger model might add — see
[Rejected Approaches](#rejected-approaches) below.

**Follow-up:** a chunk of the raw label set turned out to be
taxonomy-era duplicates rather than genuinely different categories
(e.g. `Credit reporting` / `Credit reporting, credit repair services,
or other personal consumer reports` / `Credit reporting or other
personal consumer reports` are the same product, renamed twice). A
canonical taxonomy layer (`src/taxonomy.py`) now collapses these before
training (Product: 18 → 11 classes, Issue: 75 → 63 classes — see
[`taxonomy_audit.md`](taxonomy_audit.md)), and applies the same
collapsing to the already-shipped models at inference time by merging
predicted probability across raw classes. The accuracy numbers above
are still the pre-canonicalization numbers, since retraining requires
the raw narrative dataset; the next retraining run will report Macro
F1 / Weighted F1 on the canonical label set instead (see
`src/nlp_evaluation.py`).

---

## Challenge 4: Avoiding Low-Value Sentiment Analysis

Complaint narratives are structurally negative — people write a narrative
specifically because something went wrong. Generic sentiment analysis on
this corpus would overwhelmingly return "Negative" and add little
decision-useful signal.

**Decision:** skip sentiment analysis. Focus instead on Product
prediction, Issue prediction, topic modeling, complaint driver analysis,
and recommendations — signals that map more directly to an operational
decision.

---

## Challenge 5: Forecasting Without Overengineering

Complaint volume is a monthly, trend-and-seasonality-driven time series,
which is exactly the use case Prophet is designed for — no need for a
more complex custom model.

**Validation approach:** rather than reporting in-sample fit, a 6-month
holdout window is used — train on earlier months, forecast the holdout,
compare against actuals.

**Result:** Validation MAPE 3.57%, Validation MAE 17,748. Full discussion
of what this does and doesn't prove: see [`evaluation.md`](evaluation.md).

---

## Challenge 6: From Analytics to Action

A dashboard full of charts tells you *what* happened. It doesn't tell you
*what to do*. The recommendation engine was built specifically
to close that gap — it reads the risk, growth, forecast, and driver
outputs and converts them into a prioritized executive action plan
(e.g. a high-risk company gets a recommended operational audit; a
fast-rising fraud-related issue gets a recommended fraud-monitoring
action).

---

## Production Incidents & Fixes

Real issues hit during development, and how they were resolved:

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Running scripts directly (`python src/preprocessing.py`) instead of as a package | Switched to `python -m src.preprocessing` style execution everywhere, with absolute imports (`from src.logger import logger`) throughout |
| Memory pressure when vectorizing all narratives for the "top words" feature | Running `CountVectorizer` over the full ~3.8M-narrative corpus at once | Sample a bounded subset (configurable size, capped at the available row count) before vectorizing, rather than processing every narrative |
| NLP model loading failures inside Docker | `numpy`/`scikit-learn`/`joblib` version mismatch between the training environment and the container's installed versions | Pinned dependency versions in `requirements.txt` to match the environment the models were trained in |
| Docker image larger than expected | The image intentionally bundles pre-computed dashboard artifacts and trained NLP models so the container works with a single `docker run`, without requiring the user to run the full pipeline first | Accepted as a deliberate trade-off — see `docker.md` for the reasoning |
| Port already allocated when running the container | Another process already bound to host port 8501 | Documented the `-p 8502:8501` remap as a standard troubleshooting step |

---

## Rejected Approaches

Approaches that were considered and explicitly **not** used, with reasons:

| Rejected | Reason |
|---|---|
| Full transformer fine-tuning (BERT/RoBERTa) for Product/Issue classification | Training time, RAM, and Docker image size cost did not seem justified relative to the accuracy gain over tuned TF-IDF + Logistic Regression, for a CPU-only deployment target |
| Generic sentiment analysis on narratives | Complaint text is structurally negative; sentiment scores would mostly read "Negative" and add little operational value |
| Dashboard reading the full 15.95M-row processed dataset directly | Would make every Streamlit interaction slow and memory-heavy; pre-aggregation was used instead |
| Retraining NLP models on every dashboard/pipeline run | Training over hundreds of thousands of narratives is too slow to run on every data refresh; NLP training was kept as a separate, manually-triggered step |
| Open-ended hyperparameter search (e.g. full grid or Bayesian search) for NLP tuning | A small, fixed parameter grid (6 combinations) was used instead, to keep tuning runtime bounded and results reproducible |
| Experiment-tracking tooling (e.g. MLflow) | The project's focus is the end-to-end complaint-intelligence pipeline, not a research environment with many parallel experiments to track |

---

## Lessons Learned

- **Not every NLP problem needs a transformer.** A well-tuned classical
  model on TF-IDF features can be a better fit than a larger model when
  deployability (CPU-only, bounded image size, fast inference) matters
  as much as raw accuracy.
- **Parquet + chunked processing is close to mandatory at this data
  scale** on consumer hardware — repeatedly re-reading an 8-9 GB CSV
  during development is not a sustainable workflow.
- **Pre-aggregation matters more than dashboard-side optimization.** No
  amount of caching in Streamlit fixes a dashboard that's reading the
  full dataset on every page load — the fix has to happen upstream, in
  the pipeline.
- **A diagnostic dashboard and a decision-support tool are different
  products.** Charts answer "what happened." A recommendation engine is
  needed to answer "what should we do about it" — and that has to be
  built deliberately, it doesn't fall out of the analytics for free.

---

## Infrastructure Footprint

| Resource | Value |
|---|---|
| Raw dataset | 15.95M rows, 8-9 GB CSV |
| Processed dataset | ~1.3 GB Parquet |
| Narrative subset used for NLP | up to 300,000 sampled rows (configurable) |
| GPU required | No |
| RAM | Runs on consumer-grade hardware (no specialized infrastructure) |

### Development Hardware

This project was developed and run on a single consumer laptop:

| Component | Spec |
|---|---|
| OS | Windows |
| RAM | 16 GB |
| GPU | GTX 1650 (not required by the final pipeline — no stage uses GPU acceleration) |

The GPU is listed for completeness; it played no role in training or
inference, since TF-IDF + Logistic Regression and Prophet both run
CPU-only. This is itself part of the point: the entire platform,
including NLP training on hundreds of thousands of narratives, runs on
hardware with no dedicated ML accelerator.

---

## Business Impact

The platform is intended to reduce the manual effort needed to make
sense of a complaint dataset too large to review by hand:

- **Complaint triage** — Product/Issue classification gives a starting
  category for an incoming complaint without a human reading the full
  narrative first.
- **Risk surfacing** — the company risk score turns a 7.97K-company
  dataset into a short, ranked list of companies worth a closer
  operational look, instead of requiring someone to scan all of them.
- **Trend detection** — growth analysis and the "New / Emerging" label
  specifically call out categories that are growing fast or appearing
  for the first time, which a static report would bury among hundreds
  of stable categories.
- **Forward planning** — the forecast gives a near-term volume estimate
  (with a validated error rate, not just a guess) that could inform
  staffing or capacity decisions for a complaint-handling team.
- **From signal to action** — the recommendation engine's main point is
  that none of the above is useful on its own if it just produces more
  charts. Converting a risk score or growth signal into a specific
  suggested action is what makes the difference between a reporting
  tool and a decision-support tool.

This is a portfolio project, so these are intended capabilities rather
than measured outcomes — there's no real organization currently using
this platform to validate that, say, triage time actually drops by some
percentage. The claim being made here is about what the system is
designed to do, not a measured before/after result.

---

## Scaling Limits

This architecture was built and tested at the 15.95M-row scale of the
CFPB dataset, on a single machine. It has not been tested at
significantly larger scale (e.g. hundreds of millions of rows), so any
claim about exactly where it would break down would be speculation
rather than something measured here.

What can be said with reasonable confidence, based on how the current
pipeline works:

- The chunked-CSV-to-Parquet preprocessing step would still function
  at larger scale, just take proportionally longer — it doesn't load
  the full raw file into memory at once.
- The NLP training step already samples a bounded subset (configurable,
  currently up to 300,000 narratives) rather than training on every
  available narrative, so NLP training time would not grow much even if
  the underlying narrative count grew substantially.
- The part most likely to need rethinking first is single-machine,
  single-process execution generally — every pipeline stage here runs
  sequentially on one machine. A distributed or parallelized execution
  model would likely become worth the added complexity at some scale
  well beyond what this project currently processes, though the exact
  crossover point hasn't been benchmarked.

This project did not evaluate alternative processing engines (e.g.
Spark, DuckDB) against the current Pandas + Parquet approach — the
current approach was sufficient for the dataset size at hand, and
introducing another tool wasn't something this iteration tested or
required.

---

## What I Would Do Differently

With the benefit of hindsight, a few things would be worth doing in a
different order or differently if starting over:

- **[Done since] Confidence-aware NLP outputs.** Product and Issue
  classifiers used to return only a top label. This has since been
  addressed: `src/nlp_predictor.py` now exposes canonical-label
  confidence, top alternatives, and an AUTO_ACCEPT / HUMAN_REVIEW
  decision — see `docs/evaluation.md` and `docs/taxonomy_audit.md`.
  With hindsight, this would have been roughly the same implementation
  effort if added during initial development, instead of as a
  follow-up improvement later.
- **[Done since] Migrate every module to `config.yaml` in one pass.**
  Most modules were moved to config-driven settings together, but
  `nlp_model_training.py` was initially missed and used hardcoded
  constants. This has since been fixed — it and `nlp_tuning.py` now
  read their per-classifier hyperparameters from `config.yaml`
  (`nlp.product_classifier` / `nlp.issue_classifier` /
  `nlp.topic_model`). With hindsight, a single consistent migration
  pass across every module at once would have avoided the
  inconsistency in the first place, rather than fixing it later.
- **Decide the `nlp_model_training.py` vs `nlp_tuning.py` relationship
  upfront.** Both scripts currently write to the same output filenames
  in `models/nlp/`. This overlap wasn't a deliberate design — it's a
  result of `nlp_tuning.py` being added later without resolving what
  should happen to the original training script. Picking one
  canonical training entry point from the start (or clearly splitting
  their responsibilities) would have avoided the current "only run one
  of these per cycle" caveat documented in `runbook.md` and
  `architecture.md`.
- **Measure runtime and memory usage as features were built, not
  after.** `benchmark.md` currently documents several timing and memory
  metrics as "not yet measured" rather than reporting real numbers,
  because timing instrumentation wasn't added during development.
  Wrapping pipeline stages with simple timers from the start would have
  made this section complete instead of a to-do list.

This section reflects actual known gaps in the current code and
documentation (see `README.md` → Known Limitations and `benchmark.md`),
not a generic list of "best practices" — every item above is something
that's specifically true of this project right now.

---

## Final Outcome

```
Raw CFPB Data
      ↓
Preprocessing & Parquet Optimization
      ↓
Canonical Taxonomy (docs/taxonomy_audit.md)
      ↓
NLP Classification + Confidence / Human Review
      ↓
Unsupervised Topic Discovery
      ↓
Business Analytics (Risk, Growth, Forecasting)
      ↓
Recommendation Engine
      ↓
Streamlit Dashboard
      ↓
Docker Deployment
```

The end result is a decision-support platform rather than a static
report: it identifies risk, detects growth signals, forecasts near-term
volume, classifies incoming complaints with confidence-aware human
review, discovers recurring and emerging themes, and translates all of
that into a ranked set of recommended actions.