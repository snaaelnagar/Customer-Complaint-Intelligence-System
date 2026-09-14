"""
driver_analysis.py

Builds complaint driver analysis.

This file identifies which Product + Issue + Sub-issue combinations
are responsible for the highest complaint volume.
"""

import sys
from pathlib import Path

import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.config_loader import config


PROCESSED_DATA_PATH = Path(config["paths"]["processed_data_path"])
DASHBOARD_DIR = Path(config["paths"]["dashboard_dir"])

DRIVER_OUTPUT_PATH = DASHBOARD_DIR / "driver_analysis.parquet"
TOP_DRIVER_OUTPUT_PATH = DASHBOARD_DIR / "top_complaint_drivers.parquet"
PRODUCT_DRIVER_OUTPUT_PATH = DASHBOARD_DIR / "product_driver_summary.parquet"

TOP_PRODUCTS_FOR_DRIVER = config.get("driver_analysis", {}).get("top_products", 10)
TOP_COMPLAINT_DRIVERS = config.get("driver_analysis", {}).get(
    "top_complaint_drivers",
    25,
)
TOP_DRIVERS_PER_PRODUCT = config.get("driver_analysis", {}).get(
    "top_drivers_per_product",
    3,
)


def build_driver_analysis() -> None:
    """Build complaint driver summaries and save them as parquet files."""
    try:
        logger.info("Complaint driver analysis started")

        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

        if not PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}")

        columns = [
            "Product",
            "Issue",
            "Sub-issue",
        ]

        df = pd.read_parquet(PROCESSED_DATA_PATH, columns=columns)

        if df.empty:
            raise ValueError("Processed complaints dataset is empty")

        df["Sub-issue"] = df["Sub-issue"].fillna("Unknown")

        df = df.dropna(subset=["Product", "Issue"])

        if df.empty:
            raise ValueError("No valid Product/Issue rows available for driver analysis")

        driver_analysis = (
            df.groupby(["Product", "Issue", "Sub-issue"])
            .size()
            .reset_index(name="Complaint_Count")
            .sort_values("Complaint_Count", ascending=False)
        )

        product_totals = (
            df.groupby("Product")
            .size()
            .reset_index(name="Product_Total_Complaints")
        )

        driver_analysis = driver_analysis.merge(
            product_totals,
            on="Product",
            how="left",
        )

        driver_analysis["Driver_Contribution_Pct"] = (
            driver_analysis["Complaint_Count"]
            / driver_analysis["Product_Total_Complaints"]
            * 100
        )

        top_products = (
            df["Product"]
            .value_counts()
            .head(TOP_PRODUCTS_FOR_DRIVER)
            .index
        )

        driver_analysis_top_products = driver_analysis[
            driver_analysis["Product"].isin(top_products)
        ]

        driver_analysis_top_products.to_parquet(
            DRIVER_OUTPUT_PATH,
            index=False,
        )

        top_complaint_drivers = (
            driver_analysis
            .head(TOP_COMPLAINT_DRIVERS)
            .reset_index(drop=True)
        )

        top_complaint_drivers.to_parquet(
            TOP_DRIVER_OUTPUT_PATH,
            index=False,
        )

        product_driver_summary = (
            driver_analysis
            .sort_values(["Product", "Complaint_Count"], ascending=[True, False])
            .groupby("Product")
            .head(TOP_DRIVERS_PER_PRODUCT)
            .reset_index(drop=True)
        )

        product_driver_summary.to_parquet(
            PRODUCT_DRIVER_OUTPUT_PATH,
            index=False,
        )

        logger.info(f"Driver analysis saved: {DRIVER_OUTPUT_PATH}")
        logger.info(f"Top complaint drivers saved: {TOP_DRIVER_OUTPUT_PATH}")
        logger.info(f"Product driver summary saved: {PRODUCT_DRIVER_OUTPUT_PATH}")
        logger.info("Complaint driver analysis completed")

    except Exception as e:
        logger.exception("Complaint driver analysis failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    build_driver_analysis()