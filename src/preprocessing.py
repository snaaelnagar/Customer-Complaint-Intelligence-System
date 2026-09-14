"""
preprocessing.py

Reads raw CFPB complaint CSV data, cleans it, creates useful features,
removes duplicates, and saves the final processed dataset as Parquet.
"""

import sys
from pathlib import Path

import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.config_loader import config


RAW_DATA_PATH = Path(config["paths"]["raw_data_path"])
PROCESSED_DIR = Path(config["paths"]["processed_dir"])
PROCESSED_DATA_PATH = Path(config["paths"]["processed_data_path"])

CHUNKSIZE = config["preprocessing"]["chunksize"]


REQUIRED_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Complaint ID",
]


def validate_columns(df: pd.DataFrame) -> None:
    """Check whether all required columns exist in the raw dataset."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extra spaces from text columns."""
    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values based on business meaning."""
    df["Tags"] = df["Tags"].fillna("Normal Consumer")
    df["Sub-product"] = df["Sub-product"].fillna("Unknown")
    df["Sub-issue"] = df["Sub-issue"].fillna("Unknown")
    df["State"] = df["State"].fillna("Unknown")
    df["Company public response"] = df["Company public response"].fillna(
        "No public response"
    )
    df["Company response to consumer"] = df["Company response to consumer"].fillna(
        "Unknown"
    )
    df["Issue"] = df["Issue"].fillna("Unknown")
    df["ZIP code"] = df["ZIP code"].fillna("Unknown")

    return df


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns into datetime format."""
    df["Date received"] = pd.to_datetime(
        df["Date received"],
        format="mixed",
        errors="coerce",
        utc=True,
    )

    df["Date sent to company"] = pd.to_datetime(
        df["Date sent to company"],
        format="mixed",
        errors="coerce",
        utc=True,
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create dashboard-friendly features."""
    df["Year"] = df["Date received"].dt.year
    df["Month"] = df["Date received"].dt.month
    df["Quarter"] = df["Date received"].dt.quarter
    df["Day"] = df["Date received"].dt.day
    df["Day_Name"] = df["Date received"].dt.day_name()

    df["Resolution_Delay"] = (
        df["Date sent to company"] - df["Date received"]
    ).dt.days

    df["Has_Narrative"] = df["Consumer complaint narrative"].notna().astype(int)

    df["Narrative_Length"] = (
        df["Consumer complaint narrative"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["Narrative_Word_Count"] = (
        df["Consumer complaint narrative"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    return df


def standardize_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Create simplified consumer group feature from Tags."""
    df["Consumer_Group"] = df["Tags"].replace(
        {
            "Servicemember": "Servicemember",
            "Older American": "Older American",
            "Older American, Servicemember": "Older American + Servicemember",
            "Normal Consumer": "Normal Consumer",
        }
    )

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate complaints using Complaint ID."""
    before = len(df)

    df = df.drop_duplicates(subset=["Complaint ID"])

    after = len(df)

    logger.info(f"Removed duplicate complaints: {before - after}")

    return df


def preprocess_data() -> None:
    """Run complete preprocessing pipeline."""
    try:
        logger.info("Chunk-based preprocessing pipeline started")

        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        chunks = pd.read_csv(
            RAW_DATA_PATH,
            chunksize=CHUNKSIZE,
            low_memory=False,
            on_bad_lines="skip",
        )

        processed_chunks = []
        total_rows = 0

        for i, chunk in enumerate(chunks, start=1):
            logger.info(f"Processing chunk {i} | shape: {chunk.shape}")

            if chunk.empty:
                logger.warning(f"Chunk {i} is empty, skipping")
                continue

            validate_columns(chunk)

            chunk = clean_text_columns(chunk)
            chunk = handle_missing_values(chunk)
            chunk = convert_date_columns(chunk)
            chunk = create_features(chunk)
            chunk = standardize_tags(chunk)

            processed_chunks.append(chunk)
            total_rows += len(chunk)

            logger.info(f"Chunk {i} processed | total rows: {total_rows}")

        if not processed_chunks:
            raise ValueError("No valid chunks were processed from raw data")

        logger.info("Combining processed chunks")

        df = pd.concat(processed_chunks, ignore_index=True)

        if df.empty:
            raise ValueError("Processed dataframe is empty after combining chunks")

        df = remove_duplicates(df)

        df.to_parquet(PROCESSED_DATA_PATH, index=False)

        logger.info(f"Processed data saved at: {PROCESSED_DATA_PATH}")
        logger.info(f"Final processed shape: {df.shape}")
        logger.info("Preprocessing pipeline completed successfully")

    except Exception as e:
        logger.exception("Preprocessing pipeline failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    preprocess_data()