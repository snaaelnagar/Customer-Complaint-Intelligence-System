"""
growth_analysis.py

Builds product and issue growth analytics.

Outputs:
1. product_growth.parquet
2. issue_growth.parquet
3. monthly_complaint_trend.parquet
"""

import sys
from pathlib import Path

import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.config_loader import config


PROCESSED_DATA_PATH = Path(config["paths"]["processed_data_path"])
DASHBOARD_DIR = Path(config["paths"]["dashboard_dir"])

PRODUCT_GROWTH_PATH = DASHBOARD_DIR / "product_growth.parquet"
ISSUE_GROWTH_PATH = DASHBOARD_DIR / "issue_growth.parquet"
MONTHLY_TREND_PATH = DASHBOARD_DIR / "monthly_complaint_trend.parquet"

MIN_CURRENT_COMPLAINTS = config["growth"]["min_current_complaints"]
RISING_FAST_THRESHOLD = config["growth"]["rising_fast_threshold"]
RISING_THRESHOLD = config["growth"]["rising_threshold"]
DECLINING_THRESHOLD = config["growth"]["declining_threshold"]
DECLINING_FAST_THRESHOLD = config["growth"]["declining_fast_threshold"]


def safe_growth(current: float, previous: float) -> float:
    """Calculate YoY growth safely."""
    if pd.isna(previous):
        return 0.0

    if previous == 0:
        if current > 0:
            return 999.0
        return 0.0

    return ((current - previous) / previous) * 100


def growth_label(value: float) -> str:
    """Convert growth percentage into business-friendly label."""
    if value == 999.0:
        return "New / Emerging"

    if value >= RISING_FAST_THRESHOLD:
        return "Rising Fast"

    if value >= RISING_THRESHOLD:
        return "Rising"

    if value <= DECLINING_FAST_THRESHOLD:
        return "Declining Fast"

    if value <= DECLINING_THRESHOLD:
        return "Declining"

    return "Stable"


def get_complete_years(df: pd.DataFrame) -> tuple[int, int]:
    """
    Pick last two complete years for YoY comparison.

    Max year is treated as incomplete because the current year may not
    have all 12 months available yet.
    """
    year_counts = df["Year"].value_counts().sort_index()

    if year_counts.empty:
        raise ValueError("No valid year values available for growth analysis.")

    max_year = int(year_counts.index.max())
    candidate_years = year_counts.index[year_counts.index < max_year]

    if len(candidate_years) < 2:
        raise ValueError("Not enough complete years available for growth analysis.")

    previous_year = int(candidate_years[-2])
    latest_year = int(candidate_years[-1])

    logger.info(
        f"Growth analysis using complete years: "
        f"previous_year={previous_year}, latest_year={latest_year}"
    )

    return previous_year, latest_year


def build_product_growth(
    df: pd.DataFrame,
    previous_year: int,
    latest_year: int,
) -> pd.DataFrame:
    """Build YoY growth table for products."""
    product_yearly = (
        df.dropna(subset=["Product"])
        .groupby(["Product", "Year"])
        .size()
        .reset_index(name="Complaints")
    )

    if product_yearly.empty:
        return pd.DataFrame(
            columns=[
                "Product",
                "Previous_Year_Complaints",
                "Current_Year_Complaints",
                "Previous_Year",
                "Current_Year",
                "YoY_Growth_Pct",
                "Growth_Label",
            ]
        )

    product_pivot = (
        product_yearly
        .pivot_table(
            index="Product",
            columns="Year",
            values="Complaints",
            fill_value=0,
        )
        .reset_index()
    )

    if latest_year not in product_pivot.columns:
        product_pivot[latest_year] = 0

    if previous_year not in product_pivot.columns:
        product_pivot[previous_year] = 0

    product_growth = product_pivot[
        ["Product", previous_year, latest_year]
    ].copy()

    product_growth.columns = [
        "Product",
        "Previous_Year_Complaints",
        "Current_Year_Complaints",
    ]

    product_growth["Previous_Year"] = previous_year
    product_growth["Current_Year"] = latest_year

    product_growth["YoY_Growth_Pct"] = product_growth.apply(
        lambda row: safe_growth(
            row["Current_Year_Complaints"],
            row["Previous_Year_Complaints"],
        ),
        axis=1,
    )

    product_growth["Growth_Label"] = product_growth["YoY_Growth_Pct"].apply(
        growth_label
    )

    product_growth = product_growth[
        product_growth["Current_Year_Complaints"] >= MIN_CURRENT_COMPLAINTS
    ]

    product_growth = product_growth.sort_values(
        "YoY_Growth_Pct",
        ascending=False,
    )

    return product_growth


def build_issue_growth(
    df: pd.DataFrame,
    previous_year: int,
    latest_year: int,
) -> pd.DataFrame:
    """Build YoY growth table for issues."""
    issue_yearly = (
        df.dropna(subset=["Issue"])
        .groupby(["Issue", "Year"])
        .size()
        .reset_index(name="Complaints")
    )

    if issue_yearly.empty:
        return pd.DataFrame(
            columns=[
                "Issue",
                "Previous_Year_Complaints",
                "Current_Year_Complaints",
                "Previous_Year",
                "Current_Year",
                "YoY_Growth_Pct",
                "Growth_Label",
            ]
        )

    issue_pivot = (
        issue_yearly
        .pivot_table(
            index="Issue",
            columns="Year",
            values="Complaints",
            fill_value=0,
        )
        .reset_index()
    )

    if latest_year not in issue_pivot.columns:
        issue_pivot[latest_year] = 0

    if previous_year not in issue_pivot.columns:
        issue_pivot[previous_year] = 0

    issue_growth = issue_pivot[
        ["Issue", previous_year, latest_year]
    ].copy()

    issue_growth.columns = [
        "Issue",
        "Previous_Year_Complaints",
        "Current_Year_Complaints",
    ]

    issue_growth["Previous_Year"] = previous_year
    issue_growth["Current_Year"] = latest_year

    issue_growth["YoY_Growth_Pct"] = issue_growth.apply(
        lambda row: safe_growth(
            row["Current_Year_Complaints"],
            row["Previous_Year_Complaints"],
        ),
        axis=1,
    )

    issue_growth["Growth_Label"] = issue_growth["YoY_Growth_Pct"].apply(
        growth_label
    )

    issue_growth = issue_growth[
        issue_growth["Current_Year_Complaints"] >= MIN_CURRENT_COMPLAINTS
    ]

    issue_growth = issue_growth.sort_values(
        "YoY_Growth_Pct",
        ascending=False,
    )

    return issue_growth


def build_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Build monthly complaint trend."""
    monthly_trend = (
        df.groupby(["Year", "Month"])
        .size()
        .reset_index(name="Complaints")
        .sort_values(["Year", "Month"])
    )

    if monthly_trend.empty:
        return pd.DataFrame(columns=["Year", "Month", "Complaints", "Period"])

    monthly_trend["Period"] = (
        monthly_trend["Year"].astype(str)
        + "-"
        + monthly_trend["Month"].astype(str).str.zfill(2)
    )

    return monthly_trend


def build_growth_analysis() -> None:
    """Run complete growth analysis pipeline."""
    try:
        logger.info("Growth analysis started")

        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

        if not PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}")

        columns = [
            "Year",
            "Month",
            "Product",
            "Issue",
        ]

        df = pd.read_parquet(PROCESSED_DATA_PATH, columns=columns)

        if df.empty:
            raise ValueError("Processed complaints dataset is empty")

        df = df.dropna(subset=["Year", "Month"])

        if df.empty:
            raise ValueError("No valid Year/Month rows available for growth analysis")

        df["Year"] = df["Year"].astype(int)
        df["Month"] = df["Month"].astype(int)

        previous_year, latest_year = get_complete_years(df)

        product_growth = build_product_growth(
            df=df,
            previous_year=previous_year,
            latest_year=latest_year,
        )

        issue_growth = build_issue_growth(
            df=df,
            previous_year=previous_year,
            latest_year=latest_year,
        )

        monthly_trend = build_monthly_trend(df)

        product_growth.to_parquet(PRODUCT_GROWTH_PATH, index=False)
        issue_growth.to_parquet(ISSUE_GROWTH_PATH, index=False)
        monthly_trend.to_parquet(MONTHLY_TREND_PATH, index=False)

        logger.info(f"Product growth saved: {PRODUCT_GROWTH_PATH}")
        logger.info(f"Issue growth saved: {ISSUE_GROWTH_PATH}")
        logger.info(f"Monthly complaint trend saved: {MONTHLY_TREND_PATH}")
        logger.info("Growth analysis completed")

    except Exception as e:
        logger.exception("Growth analysis failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    build_growth_analysis()