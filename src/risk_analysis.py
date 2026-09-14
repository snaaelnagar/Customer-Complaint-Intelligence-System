"""
risk_analysis.py

Builds company-level risk scores from CFPB complaint data.

Risk Score is calculated using:
1. Complaint volume
2. Untimely response percentage
3. Average resolution delay

Output:
data/processed/dashboard/company_risk_score.parquet
"""

import sys
from pathlib import Path

import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.config_loader import config


PROCESSED_DATA_PATH = Path(config["paths"]["processed_data_path"])
DASHBOARD_DIR = Path(config["paths"]["dashboard_dir"])
OUTPUT_PATH = DASHBOARD_DIR / "company_risk_score.parquet"

MIN_COMPLAINTS_FOR_RISK = config["risk"]["min_company_complaints"]
MEDIUM_RISK_THRESHOLD = config["risk"]["medium_risk_threshold"]
HIGH_RISK_THRESHOLD = config["risk"]["high_risk_threshold"]


def min_max_scale(series: pd.Series) -> pd.Series:
    """
    Scale values between 0 and 1.

    Example:
    lowest value  -> 0
    highest value -> 1

    If all values are same, return 0 for all rows.
    """
    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def assign_risk_level(score: float) -> str:
    """
    Convert numeric risk score into business-friendly risk level.

    Thresholds come from config.yaml:
    - Low    : below medium threshold
    - Medium : medium threshold to high threshold
    - High   : high threshold and above
    """
    if score >= HIGH_RISK_THRESHOLD:
        return "High"

    if score >= MEDIUM_RISK_THRESHOLD:
        return "Medium"

    return "Low"


def build_company_risk_score() -> None:
    """
    Build company-level risk score and save it as parquet.
    """
    try:
        logger.info("Company risk score generation started")

        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

        if not PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}")

        columns = [
            "Company",
            "Timely response?",
            "Resolution_Delay",
        ]

        df = pd.read_parquet(PROCESSED_DATA_PATH, columns=columns)

        # Production guard.
        if df.empty:
            raise ValueError("Processed complaints dataset is empty")

        # Remove rows where company name is missing.
        df = df.dropna(subset=["Company"])

        if df.empty:
            raise ValueError("No valid company records found after dropping missing companies")

        # Aggregate company-level risk features.
        company_risk = (
            df.groupby("Company")
            .agg(
                Complaint_Count=("Company", "size"),
                Untimely_Response_Pct=(
                    "Timely response?",
                    lambda x: (x == "No").mean() * 100,
                ),
                Avg_Resolution_Delay=("Resolution_Delay", "mean"),
            )
            .reset_index()
        )

        # Remove small companies to avoid small-sample bias.
        company_risk = company_risk[
            company_risk["Complaint_Count"] >= MIN_COMPLAINTS_FOR_RISK
        ].copy()

        logger.info(
            f"Companies kept for risk scoring after minimum complaint filter: {len(company_risk)}"
        )

        # If no company passes the minimum complaint filter, save empty output.
        if company_risk.empty:
            empty_output = pd.DataFrame(
                columns=[
                    "Company",
                    "Complaint_Count",
                    "Untimely_Response_Pct",
                    "Avg_Resolution_Delay",
                    "Complaint_Score",
                    "Untimely_Score",
                    "Delay_Score",
                    "Risk_Score",
                    "Risk_Level",
                ]
            )

            empty_output.to_parquet(OUTPUT_PATH, index=False)
            logger.warning("No companies passed the minimum complaint filter")
            return

        # Clean resolution delay.
        # Missing delay becomes 0.
        # Negative delay is clipped to 0 because negative days are invalid.
        company_risk["Avg_Resolution_Delay"] = (
            company_risk["Avg_Resolution_Delay"]
            .fillna(0)
            .clip(lower=0)
        )

        # Convert raw features into 0-1 normalized scores.
        company_risk["Complaint_Score"] = min_max_scale(
            company_risk["Complaint_Count"]
        )

        company_risk["Untimely_Score"] = min_max_scale(
            company_risk["Untimely_Response_Pct"]
        )

        company_risk["Delay_Score"] = min_max_scale(
            company_risk["Avg_Resolution_Delay"]
        )

        # Final weighted risk score.
        # Complaint volume has highest weight because high-volume companies
        # create the largest customer impact.
        company_risk["Risk_Score"] = (
            company_risk["Complaint_Score"] * 0.50
            + company_risk["Untimely_Score"] * 0.30
            + company_risk["Delay_Score"] * 0.20
        ) * 100

        # Risk level thresholds are controlled from config.yaml.
        company_risk["Risk_Level"] = company_risk["Risk_Score"].apply(
            assign_risk_level
        )

        company_risk = company_risk.sort_values(
            "Risk_Score",
            ascending=False,
        )

        company_risk = company_risk[
            [
                "Company",
                "Complaint_Count",
                "Untimely_Response_Pct",
                "Avg_Resolution_Delay",
                "Complaint_Score",
                "Untimely_Score",
                "Delay_Score",
                "Risk_Score",
                "Risk_Level",
            ]
        ]

        company_risk.to_parquet(OUTPUT_PATH, index=False)

        logger.info(f"Company risk score saved: {OUTPUT_PATH}")
        logger.info("Company risk score generation completed")

    except Exception as e:
        logger.exception("Company risk score generation failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    build_company_risk_score()