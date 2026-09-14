from src.recommendation_engine import (
    get_issue_specific_recommendation,
    get_product_specific_recommendation,
)


def test_fraud_issue_recommendation():
    result = get_issue_specific_recommendation("fraud unauthorized transaction")

    assert "fraud" in result.lower()


def test_credit_reporting_product_recommendation():
    result = get_product_specific_recommendation("Credit reporting")

    assert "credit reporting" in result.lower()


def test_unknown_issue_recommendation():
    result = get_issue_specific_recommendation("random unknown issue")

    assert isinstance(result, str)
    assert len(result) > 0