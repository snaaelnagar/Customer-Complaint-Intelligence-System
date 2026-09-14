import pandas as pd
import pytest

from src.preprocessing import validate_columns, REQUIRED_COLUMNS


def test_validate_columns_success():
    data = {col: ["sample"] for col in REQUIRED_COLUMNS}
    df = pd.DataFrame(data)

    validate_columns(df)


def test_validate_columns_missing_column():
    data = {col: ["sample"] for col in REQUIRED_COLUMNS}
    data.pop("Product")

    df = pd.DataFrame(data)

    with pytest.raises(ValueError):
        validate_columns(df)