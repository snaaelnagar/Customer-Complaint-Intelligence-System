"""
model_metrics_reader.py

Parses saved classifier metrics files (models/nlp/*_metrics.txt) for
display in the dashboard's Classification and Model Performance tabs.

Two file formats can exist side by side, since the currently-shipped
models were trained before the evaluation upgrade in
src/nlp_evaluation.py and have not been retrained yet (no raw dataset
available in this environment -- see docs/taxonomy_audit.md):

- **Legacy format** (currently shipped): "Best Accuracy: 0.xxxx" +
  "Best Params: {...}" + a raw sklearn classification_report.
- **New format** (written by src/nlp_model_training.py going forward):
  Accuracy / Macro F1 / Weighted F1 / Top-K lines, a "top confused
  category pairs" error-analysis section, then the same style of
  classification_report.

`read_metrics_file` detects which format a file is in and returns a
single normalized structure so the dashboard doesn't need to care
which one it's looking at.
"""

from __future__ import annotations

import re
from pathlib import Path


_FLOAT_LINE_RE = re.compile(r"^(?P<key>[A-Za-z0-9 \-]+):\s+(?P<value>[\d.]+)\s*$")

_REPORT_ROW_RE = re.compile(
    r"^(?P<label>.+?)\s+"
    r"(?P<precision>\d+\.\d+)\s+"
    r"(?P<recall>\d+\.\d+)\s+"
    r"(?P<f1>\d+\.\d+)\s+"
    r"(?P<support>\d+)\s*$"
)

_SUMMARY_LABELS = {"accuracy", "macro avg", "weighted avg"}


def _parse_per_class_rows(lines: list) -> list:
    rows = []
    for line in lines:
        match = _REPORT_ROW_RE.match(line.strip())
        if not match:
            continue
        label = match.group("label").strip()
        if label.lower() in _SUMMARY_LABELS:
            continue
        rows.append(
            {
                "label": label,
                "precision": float(match.group("precision")),
                "recall": float(match.group("recall")),
                "f1_score": float(match.group("f1")),
                "support": int(match.group("support")),
            }
        )
    return rows


def read_metrics_file(path: Path) -> dict:
    """
    Parse a classifier metrics file, legacy or new format.

    Returns:
        {
            "format": "legacy" | "new",
            "accuracy": float | None,
            "macro_f1": float | None,
            "weighted_f1": float | None,
            "macro_precision": float | None,
            "macro_recall": float | None,
            "top_k_accuracy": {k: float, ...},
            "confused_pairs": [{"true_label", "predicted_label", "count", "explanation"}, ...],
            "per_class": [{"label", "precision", "recall", "f1_score", "support"}, ...],
        }
    """
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines()

    is_new_format = any(line.startswith("Macro F1:") for line in lines)

    result = {
        "format": "new" if is_new_format else "legacy",
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "macro_precision": None,
        "macro_recall": None,
        "top_k_accuracy": {},
        "confused_pairs": [],
        "per_class": _parse_per_class_rows(lines),
    }

    if is_new_format:
        for line in lines:
            stripped = line.strip()

            if stripped.startswith("Accuracy:"):
                result["accuracy"] = float(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Macro F1:"):
                result["macro_f1"] = float(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Weighted F1:"):
                result["weighted_f1"] = float(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Macro Precision:"):
                result["macro_precision"] = float(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Macro Recall:"):
                result["macro_recall"] = float(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Top-"):
                match = re.match(r"Top-(\d+) Accuracy:\s+([\d.]+)", stripped)
                if match:
                    result["top_k_accuracy"][int(match.group(1))] = float(match.group(2))

        # Confused pairs: lines shaped like
        #   True='X'  ->  Predicted='Y'  (x12)
        # followed on the next line by an indented explanation.
        pair_re = re.compile(r"True='(.+?)'\s+->\s+Predicted='(.+?)'\s+\(x(\d+)\)")
        for i, line in enumerate(lines):
            match = pair_re.search(line)
            if not match:
                continue
            explanation = ""
            if i + 1 < len(lines):
                explanation = lines[i + 1].strip()
            result["confused_pairs"].append(
                {
                    "true_label": match.group(1),
                    "predicted_label": match.group(2),
                    "count": int(match.group(3)),
                    "explanation": explanation,
                }
            )
    else:
        for line in lines:
            if line.startswith("Best Accuracy:"):
                result["accuracy"] = float(line.split(":", 1)[1].strip())
                break

    return result
