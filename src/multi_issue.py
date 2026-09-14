"""
multi_issue.py

Infrastructure for detecting more than one Issue in a single complaint
narrative (e.g. "I was charged twice and the refund hasn't arrived"
-> Billing AND Refund).

Status: INFRASTRUCTURE ONLY, not a validated multi-label classifier.
--------------------------------------------------------------------
The CFPB complaint schema only records a single `Issue` value per
complaint. There is no ground-truth multi-label data to train or
validate a true multi-label classifier against, and we don't currently
have access to the raw narrative dataset in this environment to
investigate the question empirically ("how often does a narrative
plausibly describe more than one issue?").

Rather than fabricate a "multi-label model" that hasn't been validated
against real co-occurrence data, this module does two honest things:

1. Provides a **heuristic, clearly-labeled-as-heuristic** multi-issue
   extractor (`extract_candidate_issues`) that reuses the existing,
   already-trained single-label Issue classifier: instead of only
   returning the single argmax prediction, it returns every canonical
   issue whose merged probability clears a configurable threshold.
   This is a legitimate, commonly used baseline for turning a softmax
   classifier into an approximate multi-label signal, but it is NOT
   the same as a classifier trained on real multi-label ground truth,
   and every output is tagged accordingly so it is never confused for
   a validated result.

2. Documents the actual scientific plan for building and validating a
   real multi-label Issue classifier once the raw narrative dataset is
   available -- see docs/multi_issue_plan.md. This module's functions
   are structured so that plan's later steps (weak-label mining,
   proper multi-label model training) slot in without changing the
   calling code in nlp_predictor.py / the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.taxonomy import canonicalize_issue


@dataclass
class MultiIssueConfig:
    # A canonical issue is included as a "candidate additional issue"
    # if its merged probability is at least this fraction of the top
    # prediction's probability (relative threshold, not absolute) --
    # this avoids surfacing candidates just because a classifier has
    # many classes and residual probability mass spread thin.
    relative_threshold: float = 0.35
    # Never surface more than this many issues per complaint, even if
    # more clear the relative threshold, since additional low-signal
    # candidates add noise rather than value.
    max_issues: int = 3


DEFAULT_MULTI_ISSUE_CONFIG = MultiIssueConfig()


def extract_candidate_issues(
    canonical_scores: dict,
    config: MultiIssueConfig = DEFAULT_MULTI_ISSUE_CONFIG,
) -> list:
    """
    Turn a {canonical_issue: probability} map (as produced by
    `nlp_predictor.predict_with_confidence`'s internal scoring) into a
    ranked list of *candidate* issues for a complaint, using a
    relative-probability heuristic.

    This is intentionally a thin, swappable function: once a real
    multi-label classifier exists (see docs/multi_issue_plan.md), it
    can replace the body of this function without changing anything
    that calls it, as long as it keeps returning the same shape.

    Returns a list of:
        {
            "label": canonical issue label,
            "confidence": float,
            "is_heuristic": True,
        }
    sorted by confidence descending, top prediction included.
    """
    if not canonical_scores:
        return []

    ranked = sorted(canonical_scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]

    if top_score <= 0:
        return [{"label": top_label, "confidence": top_score, "is_heuristic": True}]

    candidates = []
    for label, score in ranked:
        if len(candidates) >= config.max_issues:
            break
        if score / top_score >= config.relative_threshold:
            candidates.append(
                {
                    "label": label,
                    "confidence": round(float(score), 4),
                    "is_heuristic": True,
                }
            )

    return candidates


def analyze_complaint_multi_issue(text: str) -> dict:
    """
    Convenience wrapper: run the existing single-label Issue model and
    return both its normal top-1 prediction (from
    src.nlp_predictor.predict_issue) and a heuristic multi-issue
    candidate list on top of it.

    Import is done lazily inside the function to avoid a hard
    dependency / import-order coupling between this module and the
    already-loaded models in nlp_predictor at module import time.
    """
    from src.nlp_predictor import (
        issue_model,
        issue_vectorizer,
        validate_input_text,
        _canonical_probabilities,
    )

    cleaned = validate_input_text(text)
    vector = issue_vectorizer.transform([cleaned])
    proba_row = issue_model.predict_proba(vector)[0]

    canonical_scores = _canonical_probabilities(proba_row, issue_model.classes_, canonicalize_issue)
    candidates = extract_candidate_issues(canonical_scores)

    return {
        "primary_issue": candidates[0] if candidates else None,
        "additional_candidate_issues": candidates[1:],
        "note": (
            "Additional issues beyond the primary prediction are a "
            "heuristic derived from the existing single-label "
            "classifier's probability distribution, not a validated "
            "multi-label model. See docs/multi_issue_plan.md."
        ),
    }
