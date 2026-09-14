"""
topic_discovery.py

Elevates the existing LDA topic model (models/nlp/topic_model.pkl)
from a single-complaint side-feature ("this complaint is Topic 3")
into a Discovery Layer:

    Classification (Product / Issue)
              +
    Unsupervised Discovery (this module)

What this module adds on top of the raw LDA model:

- A single, shared TOPIC_LABELS mapping (previously duplicated between
  the dashboard and the analysis notebook -- now defined once, here).
- `compute_topic_prevalence`: what share of a set of complaints falls
  into each topic -- the "recurring complaint themes" view.
- `detect_emerging_topics`: compares topic prevalence between two
  periods (or two subsets) and flags topics whose share grew the most
  -- the "emerging issues" / "changes over time" view.
- `build_discovery_report`: combines both into one structure the
  dashboard's Issue Discovery / Trends & Emerging Issues tabs can
  render directly.

Data availability note
-----------------------
`compute_topic_prevalence` and `detect_emerging_topics` work on any
DataFrame with a text column -- they don't require the raw CFPB
dataset, only *some* narrative text. In this environment we don't have
`data/processed/narratives_training.parquet` (see docs/taxonomy_audit.md
for the same limitation), so the dashboard's Issue Discovery tab can
render the static topic definitions (topic_words.pkl, real, already
trained) today, but the *time-based* emerging-issues view needs a
narrative corpus with timestamps to actually run against -- the
dashboard shows a clear message instead of fabricated numbers when
that data isn't present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Single source of truth for topic labels (previously duplicated
# between app/streamlit_app.py and notebooks/02_nlp_analysis.ipynb).
# Order matches the trained LDA model's component order
# (models/nlp/topic_model.pkl, n_components=10).
TOPIC_LABELS: dict = {
    0: "Student Loan / Personal Information",
    1: "FCRA / Legal Credit Reporting",
    2: "Identity Theft & Fraud",
    3: "Credit Report Disputes",
    4: "Debt Collection",
    5: "Banking, Cards & Payments",
    6: "Late Payment / Account History",
    7: "Mortgage & Loan Servicing",
    8: "Credit Bureau Investigation",
    9: "Consumer Reporting Agencies",
}


@dataclass
class TopicInfo:
    topic_id: int
    label: str
    top_words: list


def get_all_topics(topic_words: dict) -> list:
    """
    Static topic catalog: id, human-readable label, and top keywords
    for every trained topic. `topic_words` is the dict loaded from
    models/nlp/topic_words.pkl (topic_id -> list of top terms).
    """
    return [
        TopicInfo(
            topic_id=topic_id,
            label=TOPIC_LABELS.get(topic_id, f"Topic {topic_id}"),
            top_words=words,
        )
        for topic_id, words in sorted(topic_words.items())
    ]


def compute_topic_prevalence(
    texts,
    topic_model,
    topic_vectorizer,
    clean_text_fn,
) -> dict:
    """
    Compute what fraction of `texts` falls into each topic (its
    argmax topic under the trained LDA model).

    texts            : iterable of raw complaint narrative strings
    topic_model       : fitted sklearn LatentDirichletAllocation
    topic_vectorizer  : fitted CountVectorizer used to build topic_model
    clean_text_fn     : text-cleaning function to apply before vectorizing
                        (kept as a parameter rather than imported, so
                        this module has no hard dependency on
                        nlp_predictor's specific implementation)

    Returns {topic_id: prevalence_fraction}, prevalence fractions sum
    to 1.0 across all topics (unless `texts` is empty, in which case
    an empty dict is returned).
    """
    texts = list(texts)

    if not texts:
        return {}

    cleaned = [clean_text_fn(t) for t in texts]
    vectors = topic_vectorizer.transform(cleaned)
    topic_distributions = topic_model.transform(vectors)

    dominant_topics = topic_distributions.argmax(axis=1)

    n_topics = topic_model.components_.shape[0]
    counts = np.bincount(dominant_topics, minlength=n_topics)

    total = counts.sum()
    return {topic_id: float(count) / total for topic_id, count in enumerate(counts)}


def detect_emerging_topics(
    current_prevalence: dict,
    baseline_prevalence: dict,
    min_absolute_growth: float = 0.03,
) -> list:
    """
    Compare topic prevalence between a current period/subset and a
    baseline period/subset, and flag topics whose share grew the most.

    current_prevalence / baseline_prevalence : {topic_id: fraction},
        as returned by compute_topic_prevalence(), ideally over the
        same topic_id set.
    min_absolute_growth : only topics whose prevalence grew by at
        least this many percentage points (e.g. 0.03 = 3pp) are
        considered "emerging" -- filters out noise from small samples.

    Returns a list of:
        {
            "topic_id": ...,
            "current_share": ...,
            "baseline_share": ...,
            "absolute_growth": ...,
        }
    sorted by absolute_growth descending, restricted to topics that
    clear `min_absolute_growth`.
    """
    all_topic_ids = set(current_prevalence) | set(baseline_prevalence)

    rows = []
    for topic_id in all_topic_ids:
        current_share = current_prevalence.get(topic_id, 0.0)
        baseline_share = baseline_prevalence.get(topic_id, 0.0)
        growth = current_share - baseline_share

        if growth >= min_absolute_growth:
            rows.append(
                {
                    "topic_id": topic_id,
                    "current_share": round(current_share, 4),
                    "baseline_share": round(baseline_share, 4),
                    "absolute_growth": round(growth, 4),
                }
            )

    rows.sort(key=lambda row: row["absolute_growth"], reverse=True)
    return rows


def build_discovery_report(
    topic_words: dict,
    current_prevalence: dict | None = None,
    baseline_prevalence: dict | None = None,
    min_absolute_growth: float = 0.03,
) -> dict:
    """
    Combine the static topic catalog with (optional) prevalence and
    emerging-topic analysis into one structure for the dashboard.

    If `current_prevalence` is None, only the static catalog is
    returned (`recurring_themes` / `emerging_issues` are empty lists)
    -- this is the honest state when no narrative corpus with
    timestamps is available to compute real prevalence over time.
    """
    topics = get_all_topics(topic_words)

    recurring_themes = []
    if current_prevalence:
        ranked = sorted(
            current_prevalence.items(), key=lambda item: item[1], reverse=True
        )
        for topic_id, share in ranked:
            recurring_themes.append(
                {
                    "topic_id": topic_id,
                    "label": TOPIC_LABELS.get(topic_id, f"Topic {topic_id}"),
                    "share": round(share, 4),
                }
            )

    emerging_issues = []
    if current_prevalence and baseline_prevalence:
        emerging = detect_emerging_topics(
            current_prevalence, baseline_prevalence, min_absolute_growth
        )
        for row in emerging:
            row = dict(row)
            row["label"] = TOPIC_LABELS.get(row["topic_id"], f"Topic {row['topic_id']}")
            emerging_issues.append(row)

    return {
        "topics": topics,
        "recurring_themes": recurring_themes,
        "emerging_issues": emerging_issues,
    }
