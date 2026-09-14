from src.growth_analysis import safe_growth, growth_label


def test_safe_growth_normal_case():
    assert safe_growth(150, 100) == 50.0


def test_safe_growth_previous_zero_current_positive():
    assert safe_growth(10, 0) == 999.0


def test_safe_growth_previous_zero_current_zero():
    assert safe_growth(0, 0) == 0.0


def test_growth_label():
    assert growth_label(999.0) == "New / Emerging"
    assert growth_label(30) == "Rising Fast"
    assert growth_label(10) == "Rising"
    assert growth_label(0) == "Stable"
    assert growth_label(-10) == "Declining"
    assert growth_label(-30) == "Declining Fast"


def test_growth_label_boundaries():
    assert growth_label(25) == "Rising Fast"
    assert growth_label(5) == "Rising"
    assert growth_label(-5) == "Declining"
    assert growth_label(-25) == "Declining Fast"