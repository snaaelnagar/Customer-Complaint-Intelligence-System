"""
nlp_evaluation.py

Evaluation layer for the NLP classifiers (Product / Issue).

Why this exists
----------------
The previous evaluation for these classifiers reported a single number
(accuracy) plus a raw sklearn classification_report dump. That is not
enough on a dataset this imbalanced: a classifier can score a high
accuracy just by being good at the two or three dominant classes while
being useless on everything else.

This module standardizes evaluation around:

- Macro F1      (treats every class equally -- the right headline
                  metric on an imbalanced label set)
- Weighted F1    (accounts for class frequency -- useful alongside
                  Macro F1, not instead of it)
- Precision / Recall (macro and per-class)
- Confusion matrix
- Top-K accuracy (is the correct label in the model's top K guesses,
                  even when it isn't the single top prediction)
- Per-class report (already available via sklearn, kept for detail)
- Plain-language error analysis: which class *pairs* the model
  confuses most, and a candidate reason why (shared vocabulary
  overlap, or one class being a near-duplicate label of the other
  after taxonomy canonicalization)

Nothing in this module requires the raw CFPB dataset -- it operates on
whatever y_true / y_pred / y_proba arrays are handed to it, so it is
usable immediately once real narrative training data is available, and
is unit-testable today with synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    top_k_accuracy: dict  # {k: accuracy}
    classification_report: str
    confusion_matrix: np.ndarray
    labels: list
    confused_pairs: list  # see `error_analysis`


def top_k_accuracy(y_true, y_proba: np.ndarray, classes: np.ndarray, k: int) -> float:
    """
    Fraction of samples where the true label appears in the model's
    top-k predicted classes (by predicted probability).

    y_true   : array-like of true labels (raw label values, matching
               entries in `classes`)
    y_proba  : (n_samples, n_classes) probability matrix, e.g. from
               model.predict_proba(X)
    classes  : array of class labels in the same column order as
               y_proba (i.e. model.classes_)
    k        : how many top predictions to consider
    """
    y_true = np.asarray(y_true)
    k = min(k, y_proba.shape[1])

    top_k_idx = np.argsort(y_proba, axis=1)[:, -k:]
    top_k_labels = classes[top_k_idx]

    hits = [y_true[i] in top_k_labels[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def error_analysis(
    y_true,
    y_pred,
    labels: list,
    vectorizer=None,
    top_n_pairs: int = 10,
) -> list:
    """
    Identify the class pairs the model confuses most often, and give a
    plain-language candidate reason for each.

    Returns a list of dicts, sorted by confusion count descending:
        {
            "true_label": ...,
            "predicted_label": ...,
            "count": ...,
            "explanation": "Model struggles to distinguish these two
                complaint categories because ...",
        }

    Reason heuristics (in priority order):
    1. If a fitted TF-IDF-style vectorizer is provided, the shared
       top vocabulary between the two classes' misclassified samples
       is not available without the raw text at this layer, so we
       fall back to a structural explanation based on label wording
       similarity (e.g. shared words in the label strings themselves,
       which is often a strong signal for near-duplicate categories).
    2. Otherwise, a generic explanation naming the imbalance context
       (support count) is used.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)  # only care about off-diagonal confusion

    pair_counts = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j or cm[i, j] == 0:
                continue
            pair_counts.append((true_label, pred_label, int(cm[i, j])))

    pair_counts.sort(key=lambda x: x[2], reverse=True)

    results = []
    for true_label, pred_label, count in pair_counts[:top_n_pairs]:
        explanation = _explain_confusion(true_label, pred_label)
        results.append(
            {
                "true_label": true_label,
                "predicted_label": pred_label,
                "count": count,
                "explanation": explanation,
            }
        )

    return results


def _explain_confusion(true_label: str, pred_label: str) -> str:
    """
    Best-effort, human-readable candidate explanation for why the
    model confuses `true_label` with `pred_label`. This is a heuristic,
    not a certainty -- it is meant to give a human reviewer a starting
    point, not a final diagnosis.
    """
    true_words = set(str(true_label).lower().split())
    pred_words = set(str(pred_label).lower().split())
    shared = true_words & pred_words

    overlap_ratio = len(shared) / max(len(true_words | pred_words), 1)

    if overlap_ratio >= 0.5:
        return (
            f"Model struggles to distinguish '{true_label}' from "
            f"'{pred_label}' because the category labels themselves share "
            f"most of their wording ({', '.join(sorted(shared)) or 'similar phrasing'}); "
            "these may be near-duplicate categories from different "
            "taxonomy eras rather than truly distinct complaint types -- "
            "check src/taxonomy.py for whether they should be merged."
        )

    return (
        f"Model struggles to distinguish '{true_label}' from "
        f"'{pred_label}'. These categories don't share obvious wording, "
        "so the confusion is more likely driven by overlapping complaint "
        "narrative vocabulary (e.g. both categories are commonly "
        "described using similar consumer language) rather than a "
        "labeling artifact. Recommend inspecting misclassified examples "
        "directly once raw text is available."
    )


def evaluate_classifier(
    y_true,
    y_pred,
    y_proba: np.ndarray | None = None,
    classes: np.ndarray | None = None,
    top_k_values: tuple = (1, 3, 5),
) -> EvaluationResult:
    """
    Full evaluation of a single classifier's predictions.

    y_proba/classes are optional; when provided, Top-K accuracy is
    computed for each value in `top_k_values`.
    """
    labels = sorted(set(y_true) | set(y_pred))

    top_k_scores = {}
    if y_proba is not None and classes is not None:
        for k in top_k_values:
            top_k_scores[k] = top_k_accuracy(y_true, y_proba, classes, k)
    else:
        top_k_scores[1] = accuracy_score(y_true, y_pred)

    confused = error_analysis(y_true, y_pred, labels=labels)

    return EvaluationResult(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        macro_precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        macro_recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        top_k_accuracy=top_k_scores,
        classification_report=classification_report(y_true, y_pred, zero_division=0),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels),
        labels=labels,
        confused_pairs=confused,
    )


def format_evaluation_report(title: str, result: EvaluationResult) -> str:
    """Human-readable evaluation report combining all the metrics above."""
    lines = [f"=== {title} ===", ""]
    lines.append(f"Accuracy:          {result.accuracy:.4f}")
    lines.append(f"Macro F1:          {result.macro_f1:.4f}")
    lines.append(f"Weighted F1:       {result.weighted_f1:.4f}")
    lines.append(f"Macro Precision:   {result.macro_precision:.4f}")
    lines.append(f"Macro Recall:      {result.macro_recall:.4f}")

    for k, score in sorted(result.top_k_accuracy.items()):
        lines.append(f"Top-{k} Accuracy:    {score:.4f}")

    lines.append("")
    lines.append("Where the model struggles most (top confused category pairs):")
    lines.append("")

    for pair in result.confused_pairs:
        lines.append(
            f"  True='{pair['true_label']}'  ->  Predicted='{pair['predicted_label']}'  "
            f"(x{pair['count']})"
        )
        lines.append(f"      {pair['explanation']}")
        lines.append("")

    lines.append("Full per-class report:")
    lines.append(result.classification_report)

    return "\n".join(lines)
