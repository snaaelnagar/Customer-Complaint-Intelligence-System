from src.taxonomy import (
    canonicalize_product,
    canonicalize_issue,
    full_taxonomy_report,
)


def test_canonicalize_issue_merges_known_duplicates():
    assert canonicalize_issue("Incorrect information on credit report") == \
        canonicalize_issue("Incorrect information on your report")


def test_canonicalize_product_merges_known_duplicates():
    assert canonicalize_product("Credit reporting") == \
        canonicalize_product("Credit reporting or other personal consumer reports")


def test_canonicalize_unknown_label_returns_itself():
    unseen_label = "Some brand new issue label from a future taxonomy version"
    assert canonicalize_issue(unseen_label) == unseen_label


def test_canonicalize_handles_none():
    assert canonicalize_issue(None) is None
    assert canonicalize_product(None) is None


def test_full_taxonomy_report_structure():
    report = full_taxonomy_report()

    assert "product" in report
    assert "issue" in report

    for row in report["product"] + report["issue"]:
        assert row["raw_label_count"] == len(row["raw_labels"])
        assert row["confidence"] in {"high", "medium"}
