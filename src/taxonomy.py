"""
taxonomy.py

Canonical taxonomy layer for the CFPB "Product" and "Issue" fields.

Why this exists
----------------
The CFPB complaint schema has been revised several times since 2012.
Every revision renamed some Product/Issue values without changing what
they actually mean, so the *same* real-world category ends up split
across multiple raw string labels in the historical data. For example,
all three of the following raw "Product" values refer to the same
underlying category, just from different taxonomy eras:

    "Credit reporting"
    "Credit reporting, credit repair services, or other personal consumer reports"
    "Credit reporting or other personal consumer reports"

When a classifier is trained directly on the raw labels, the training
signal for that one real category gets fragmented across 2-3 classes.
This hurts Macro-F1 the most, because macro-averaging weights rare
classes equally with common ones, and fragmented classes are
artificially rare.

This module does NOT delete or overwrite the original label. It adds a
canonicalization step:

    raw label  ->  canonical_label(raw label)

`Issue_raw` / `Product_raw` should always be kept alongside
`Issue_canonical` / `Product_canonical` in any dataset built from this
module, so the original CFPB taxonomy is fully auditable.

Where the mapping below comes from
-----------------------------------
We do not currently have access to the raw CFPB CSV in this
environment (no `data/raw/complaints.csv`, no network egress), so this
mapping was NOT built by mining the raw data directly. It was built
from:

1. Publicly documented CFPB Product/Issue taxonomy revisions (the
   CFPB has periodically renamed and consolidated Product/Issue
   values; the groupings below track those documented renames).
2. Cross-referencing every raw label that this project's *own*
   already-trained classifiers were evaluated against (see
   `models/nlp/product_classifier_metrics.txt` and
   `models/nlp/issue_classifier_metrics.txt`), so every raw string in
   this map is a label that has actually appeared in this project's
   training data.

Each group below carries a `confidence` flag:

- "high"   -> near-certain rename/consolidation of the same concept
              (wording or punctuation changed, meaning did not).
- "medium" -> very likely the same concept, but the assumption should
              be validated against real co-occurrence/frequency data
              once the raw dataset is available (see
              `src/taxonomy_audit.py`).

Nothing here is destructive: if a label is not found in the map, it is
returned unchanged as its own canonical group, so unseen or future
labels degrade gracefully instead of erroring out.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaxonomyGroup:
    canonical: str
    raw_labels: tuple
    confidence: str  # "high" | "medium"
    rationale: str


# ---------------------------------------------------------------------------
# Product taxonomy groups
# ---------------------------------------------------------------------------

PRODUCT_GROUPS: tuple = (
    TaxonomyGroup(
        canonical="Credit Reporting",
        raw_labels=(
            "Credit reporting",
            "Credit reporting, credit repair services, or other personal consumer reports",
            "Credit reporting or other personal consumer reports",
        ),
        confidence="high",
        rationale=(
            "CFPB renamed this product twice (2017, then again later) while "
            "keeping the same underlying complaint population: consumer "
            "reporting agencies and credit report accuracy."
        ),
    ),
    TaxonomyGroup(
        canonical="Credit Card / Prepaid Card",
        raw_labels=(
            "Credit card",
            "Prepaid card",
            "Credit card or prepaid card",
        ),
        confidence="high",
        rationale=(
            "CFPB merged the standalone 'Credit card' and 'Prepaid card' "
            "products into 'Credit card or prepaid card' starting in 2018."
        ),
    ),
    TaxonomyGroup(
        canonical="Payday / Title / Personal Loan",
        raw_labels=(
            "Payday loan",
            "Payday loan, title loan, or personal loan",
            "Payday loan, title loan, personal loan, or advance loan",
        ),
        confidence="high",
        rationale=(
            "CFPB progressively broadened this product label to cover "
            "title loans, personal loans, and (most recently) advance/"
            "earned-wage-access loans, without changing the core category."
        ),
    ),
    TaxonomyGroup(
        canonical="Checking / Savings Account",
        raw_labels=(
            "Bank account or service",
            "Checking or savings account",
        ),
        confidence="high",
        rationale=(
            "'Bank account or service' was renamed to 'Checking or savings "
            "account' in the 2017 taxonomy revision."
        ),
    ),
    TaxonomyGroup(
        canonical="Consumer Loan (Legacy)",
        raw_labels=("Consumer Loan",),
        confidence="medium",
        rationale=(
            "'Consumer Loan' was retired as a product around 2017 and its "
            "complaints were redistributed across several newer, more "
            "specific products (e.g. vehicle loan/lease, payday/title/"
            "personal loan). There is no single clean 1:1 replacement, so "
            "this legacy label is kept as its own canonical bucket rather "
            "than force-merged. Flagged for sub-issue-level disambiguation "
            "once the raw dataset is available."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Issue taxonomy groups
# ---------------------------------------------------------------------------

ISSUE_GROUPS: tuple = (
    TaxonomyGroup(
        canonical="Incorrect Information on Report",
        raw_labels=(
            "Incorrect information on credit report",
            "Incorrect information on your report",
        ),
        confidence="high",
        rationale="Same issue, wording updated in the 2017 taxonomy revision.",
    ),
    TaxonomyGroup(
        canonical="Company Investigation Problem",
        raw_labels=(
            "Problem with a company's investigation into an existing issue",
            "Problem with a company's investigation into an existing problem",
            "Problem with a credit reporting company's investigation into an existing problem",
            "Credit reporting company's investigation",
        ),
        confidence="high",
        rationale=(
            "All four labels describe a consumer disputing how a company "
            "(most often a credit bureau) handled its investigation of a "
            "prior complaint or dispute; wording changed across taxonomy "
            "versions, the underlying issue did not."
        ),
    ),
    TaxonomyGroup(
        canonical="Debt Collection Attempts on Debt Not Owed",
        raw_labels=(
            "Attempts to collect debt not owed",
            "Cont'd attempts collect debt not owed",
        ),
        confidence="high",
        rationale="Abbreviated legacy wording ('Cont'd attempts...') replaced by full phrasing.",
    ),
    TaxonomyGroup(
        canonical="Closing an Account",
        raw_labels=(
            "Closing an account",
            "Closing your account",
        ),
        confidence="high",
        rationale="Pronoun-only wording change ('an' -> 'your').",
    ),
    TaxonomyGroup(
        canonical="Dealing with Lender or Servicer",
        raw_labels=(
            "Dealing with my lender or servicer",
            "Dealing with your lender or servicer",
        ),
        confidence="high",
        rationale="Pronoun-only wording change ('my' -> 'your').",
    ),
    TaxonomyGroup(
        canonical="Unable to Get Credit Report or Score",
        raw_labels=(
            "Unable to get credit report/credit score",
            "Unable to get your credit report or credit score",
        ),
        confidence="high",
        rationale="Punctuation/wording normalization, same issue.",
    ),
    TaxonomyGroup(
        canonical="Trouble Using the Card",
        raw_labels=(
            "Trouble using the card",
            "Trouble using your card",
        ),
        confidence="high",
        rationale="Pronoun-only wording change ('the' -> 'your').",
    ),
    TaxonomyGroup(
        canonical="Funds Being Low Caused a Problem",
        raw_labels=(
            "Problem caused by your funds being low",
            "Problems caused by my funds being low",
        ),
        confidence="high",
        rationale="Pluralization/pronoun wording change only.",
    ),
    TaxonomyGroup(
        canonical="Struggling to Repay Loan",
        raw_labels=(
            "Struggling to pay your loan",
            "Struggling to repay your loan",
        ),
        confidence="high",
        rationale="'Pay' vs 'repay' wording change, same issue.",
    ),
    TaxonomyGroup(
        canonical="Getting a Loan or Lease",
        raw_labels=(
            "Getting a loan or lease",
            "Getting the loan",
        ),
        confidence="medium",
        rationale=(
            "Likely the same intake issue ('trouble during the loan "
            "application/origination process') under different taxonomy "
            "eras, but 'Getting the loan' is worded narrowly enough that "
            "this should be confirmed against real co-occurrence data."
        ),
    ),
)


def _build_lookup(groups: tuple) -> dict:
    lookup: dict = {}
    for group in groups:
        for raw_label in group.raw_labels:
            lookup[raw_label] = group.canonical
    return lookup


PRODUCT_CANONICAL_MAP: dict = _build_lookup(PRODUCT_GROUPS)
ISSUE_CANONICAL_MAP: dict = _build_lookup(ISSUE_GROUPS)


def canonicalize_product(raw_label: str) -> str:
    """
    Map a raw CFPB 'Product' label to its canonical category.

    Unknown labels are returned unchanged (they become their own
    canonical group) so new/unseen Product values never crash this
    function.
    """
    if raw_label is None:
        return raw_label

    label = str(raw_label).strip()
    return PRODUCT_CANONICAL_MAP.get(label, label)


def canonicalize_issue(raw_label: str) -> str:
    """
    Map a raw CFPB 'Issue' label to its canonical category.

    Unknown labels are returned unchanged (they become their own
    canonical group) so new/unseen Issue values never crash this
    function.
    """
    if raw_label is None:
        return raw_label

    label = str(raw_label).strip()
    return ISSUE_CANONICAL_MAP.get(label, label)


def taxonomy_report(groups: tuple) -> list:
    """
    Return a plain-data audit summary of a taxonomy group set:
    one row per canonical category, listing every raw label folded
    into it and the confidence of that merge.

    Used by src/taxonomy_audit.py and docs generation; kept dependency
    -free (no pandas) so it can be imported anywhere cheaply.
    """
    rows = []
    for group in groups:
        rows.append(
            {
                "canonical_label": group.canonical,
                "raw_labels": list(group.raw_labels),
                "raw_label_count": len(group.raw_labels),
                "confidence": group.confidence,
                "rationale": group.rationale,
            }
        )
    return rows


def full_taxonomy_report() -> dict:
    """Audit summary for both Product and Issue taxonomies."""
    return {
        "product": taxonomy_report(PRODUCT_GROUPS),
        "issue": taxonomy_report(ISSUE_GROUPS),
    }
