import pandas as pd

from src.risk_analysis import min_max_scale, assign_risk_level


def test_min_max_scale_normal_values():
    series = pd.Series([10, 20, 30])

    result = min_max_scale(series)

    assert result.min() == 0
    assert result.max() == 1


def test_min_max_scale_same_values():
    series = pd.Series([5, 5, 5])

    result = min_max_scale(series)

    assert result.tolist() == [0, 0, 0]


def test_min_max_scale_empty_series():
    series = pd.Series(dtype=float)

    result = min_max_scale(series)

    assert len(result) == 0


def test_assign_risk_level():
    assert assign_risk_level(80) == "High"
    assert assign_risk_level(50) == "Medium"
    assert assign_risk_level(20) == "Low"