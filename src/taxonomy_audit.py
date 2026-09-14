"""
taxonomy_audit.py

Audits how much label fragmentation the canonical taxonomy (src/taxonomy.py)
removes.

This script supports two modes:

1. `audit_dataframe(df, column, canonicalize_fn)`
   Generic mode for real data: given a DataFrame with a raw label column
   (e.g. the full `Issue` or `Product` column from
   data/processed/narratives_training.parquet), reports how many raw
   classes collapse into how many canonical classes, and which raw
   classes gained the most usable training support after merging.
   This is the mode to run once the raw CFPB dataset is available.

2. `audit_existing_model_metrics()`
   Works right now, with zero raw data. This project already shipped
   two trained classifiers (models/nlp/product_classifier_metrics.txt,
   models/nlp/issue_classifier_metrics.txt), each containing a full
   sklearn classification report with real per-class `support` counts
   from an actual held-out test set. This function parses those
   support counts and simulates what canonicalization would have done
   to class support -- using real numbers, not invented ones.

Run directly to print both a Product and an Issue audit against the
model metrics files:

    python -m src.taxonomy_audit
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from src.logger import logger
from src.exception import CustomException
from src.taxonomy import canonicalize_product, canonicalize_issue


MODEL_DIR = Path("models/nlp")

# Matches a classification_report data row:
#   "<label ...>   <precision>  <recall>  <f1-score>  <support>"
# Labels can contain spaces/commas/apostrophes, so we anchor on the four
# trailing numeric columns instead of trying to parse the label format.
_REPORT_ROW_RE = re.compile(
    r"^(?P<label>.+?)\s+"
    r"(?P<precision>\d+\.\d+)\s+"
    r"(?P<recall>\d+\.\d+)\s+"
    r"(?P<f1>\d+\.\d+)\s+"
    r"(?P<support>\d+)\s*$"
)

_SUMMARY_LABELS = {"accuracy", "macro avg", "weighted avg"}


def parse_classification_report(path: Path) -> dict:
    """
    Parse per-class support counts out of a saved sklearn
    classification_report text file.

    Returns {raw_label: support_count}, excluding the accuracy/
    macro avg/weighted avg summary rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    support_by_label: dict = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            match = _REPORT_ROW_RE.match(line.strip())

            if not match:
                continue

            label = match.group("label").strip()

            if label.lower() in _SUMMARY_LABELS:
                continue

            support_by_label[label] = int(match.group("support"))

    return support_by_label


def audit_support_counts(support_by_label: dict, canonicalize_fn) -> dict:
    """
    Given {raw_label: support} and a canonicalize_fn, compute:
    - raw class count vs canonical class count
    - canonical support totals (sum of merged raw supports)
    - which canonical groups merged more than one raw label, and how
      much support each gained versus its weakest raw component
    """
    canonical_support: dict = {}
    canonical_members: dict = {}

    for raw_label, support in support_by_label.items():
        canonical_label = canonicalize_fn(raw_label)
        canonical_support[canonical_label] = (
            canonical_support.get(canonical_label, 0) + support
        )
        canonical_members.setdefault(canonical_label, []).append(
            (raw_label, support)
        )

    merged_groups = {
        canonical: members
        for canonical, members in canonical_members.items()
        if len(members) > 1
    }

    rare_before = sum(1 for s in support_by_label.values() if s < 50)
    rare_after = sum(1 for s in canonical_support.values() if s < 50)

    return {
        "raw_class_count": len(support_by_label),
        "canonical_class_count": len(canonical_support),
        "classes_removed": len(support_by_label) - len(canonical_support),
        "rare_classes_before (<50 support)": rare_before,
        "rare_classes_after (<50 support)": rare_after,
        "merged_groups": {
            canonical: {
                "raw_components": members,
                "combined_support": canonical_support[canonical],
            }
            for canonical, members in merged_groups.items()
        },
    }


def format_audit_report(title: str, audit: dict) -> str:
    lines = [f"=== {title} ===", ""]
    lines.append(f"Raw classes:               {audit['raw_class_count']}")
    lines.append(f"Canonical classes:         {audit['canonical_class_count']}")
    lines.append(f"Classes removed:           {audit['classes_removed']}")
    lines.append(
        f"Rare classes before (<50): {audit['rare_classes_before (<50 support)']}"
    )
    lines.append(
        f"Rare classes after  (<50): {audit['rare_classes_after (<50 support)']}"
    )
    lines.append("")
    lines.append("Merged groups:")

    for canonical, info in audit["merged_groups"].items():
        lines.append(f"  - {canonical}  (combined support: {info['combined_support']})")
        for raw_label, support in info["raw_components"]:
            lines.append(f"        <- \"{raw_label}\"  (support={support})")

    return "\n".join(lines)


def audit_existing_model_metrics() -> str:
    """
    Run the audit against the real support counts saved by this
    project's already-trained Product and Issue classifiers.

    This works today, without the raw CFPB dataset, because the
    support counts come from a real evaluation the models already ran.
    """
    try:
        product_support = parse_classification_report(
            MODEL_DIR / "product_classifier_metrics.txt"
        )
        issue_support = parse_classification_report(
            MODEL_DIR / "issue_classifier_metrics.txt"
        )

        product_audit = audit_support_counts(product_support, canonicalize_product)
        issue_audit = audit_support_counts(issue_support, canonicalize_issue)

        report = "\n\n".join(
            [
                format_audit_report("Product Taxonomy Audit", product_audit),
                format_audit_report("Issue Taxonomy Audit", issue_audit),
            ]
        )

        logger.info("Taxonomy audit against existing model metrics completed")
        return report

    except Exception as e:
        logger.exception("Taxonomy audit failed")
        raise CustomException(e, sys)


def audit_dataframe(df, column: str, canonicalize_fn) -> dict:
    """
    Generic audit for a real raw label column (e.g. the full `Issue`
    or `Product` column once data/processed/narratives_training.parquet
    or the raw CFPB CSV is available).

    Intended usage once real data is present:

        import pandas as pd
        from src.taxonomy import canonicalize_issue
        from src.taxonomy_audit import audit_dataframe, format_audit_report

        df = pd.read_parquet("data/processed/narratives_training.parquet")
        audit = audit_dataframe(df, "Issue", canonicalize_issue)
        print(format_audit_report("Issue Taxonomy Audit (full dataset)", audit))
    """
    support_by_label = df[column].dropna().value_counts().to_dict()
    return audit_support_counts(support_by_label, canonicalize_fn)


if __name__ == "__main__":
    print(audit_existing_model_metrics())
    print()
    print("Note: this audit runs against the support counts already saved by")
    print("the existing trained classifiers (models/nlp/*_metrics.txt), since")
    print("the raw CFPB dataset is not available in this environment. Once")
    print("data/processed/narratives_training.parquet exists, run")
    print("audit_dataframe() on the full column for a complete audit.")
