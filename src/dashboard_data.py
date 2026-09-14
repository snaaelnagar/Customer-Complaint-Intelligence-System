"""
dashboard_data.py

Generates all aggregated Parquet summaries used by the Streamlit dashboard.

This file reads the processed complaints dataset once and creates small
dashboard-ready parquet files so Streamlit does not need to load the full
15M+ row dataset repeatedly.
"""

import gc
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.logger import logger
from src.exception import CustomException
from src.config_loader import config


PROCESSED_DATA_PATH = Path(config["paths"]["processed_data_path"])
DASHBOARD_DIR = Path(config["paths"]["dashboard_dir"])

TOP_PRODUCTS = config["dashboard"]["top_products"]
TOP_ISSUES = config["dashboard"]["top_issues"]
TOP_COMPANIES = config["dashboard"]["top_companies"]
TOP_STATES = config["dashboard"]["top_states"]

RANDOM_STATE = config["nlp"]["random_state"]


def save_summary(df: pd.DataFrame, filename: str) -> None:
    """Save a dashboard summary dataframe as parquet."""
    path = DASHBOARD_DIR / filename
    df.to_parquet(path, index=False)
    logger.info(f"Saved dashboard summary: {path}")


def safe_idxmax(series: pd.Series, default: str = "Not Available") -> str:
    """Return most frequent value safely."""
    series = series.dropna()

    if series.empty:
        return default

    return series.value_counts().idxmax()


def build_dashboard_data() -> None:
    try:
        logger.info("Dashboard summary generation started")

        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

        if not PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}")

        df = pd.read_parquet(PROCESSED_DATA_PATH)

        if df.empty:
            raise ValueError("Processed complaints dataset is empty")

        if "Narrative_Word_Count" not in df.columns:
            if "Narrative_Length" in df.columns:
                df["Narrative_Word_Count"] = df["Narrative_Length"]
            else:
                df["Narrative_Word_Count"] = pd.NA

        # ==========================================================
        # CORE KPI SUMMARY
        # ==========================================================

        yearly_trend = (
            df.groupby("Year")
            .size()
            .reset_index(name="Complaints")
            .sort_values("Year")
        )

        if yearly_trend.empty:
            raise ValueError("Yearly trend is empty. Check Year column in processed data.")

        peak_row = yearly_trend.loc[yearly_trend["Complaints"].idxmax()]
        peak_year = int(peak_row["Year"])
        peak_year_complaints = int(peak_row["Complaints"])

        kpis = pd.DataFrame([{
            "total_complaints": len(df),
            "total_companies": df["Company"].nunique(),
            "total_products": df["Product"].nunique(),
            "total_states": df["State"].nunique(),
            "timely_response_pct": df["Timely response?"].eq("Yes").mean() * 100,
            "avg_resolution_delay": df["Resolution_Delay"].dropna().mean(),
            "narrative_availability_pct": df["Has_Narrative"].mean() * 100,
            "top_product": safe_idxmax(df["Product"]),
            "top_company": safe_idxmax(df["Company"]),
            "top_issue": safe_idxmax(df["Issue"]),
            "top_channel": safe_idxmax(df["Submitted via"]),
            "peak_year": peak_year,
            "peak_year_complaints": peak_year_complaints,
        }])

        save_summary(kpis, "kpis.parquet")
        save_summary(yearly_trend, "yearly_trend.parquet")

        # ==========================================================
        # RESPONSE / CHANNEL / CONSUMER SUMMARY
        # ==========================================================

        timely_response = df["Timely response?"].value_counts().reset_index()
        timely_response.columns = ["Timely Response", "Count"]
        save_summary(timely_response, "timely_response.parquet")

        submitted_via = df["Submitted via"].value_counts().reset_index()
        submitted_via.columns = ["Channel", "Count"]
        save_summary(submitted_via, "submitted_via.parquet")

        consumer_group = df["Consumer_Group"].value_counts().reset_index()
        consumer_group.columns = ["Consumer Group", "Count"]
        save_summary(consumer_group, "consumer_group.parquet")

        # ==========================================================
        # PRODUCT ANALYSIS
        # ==========================================================

        top_products = df["Product"].dropna().value_counts().head(TOP_PRODUCTS).reset_index()
        top_products.columns = ["Product", "Count"]
        save_summary(top_products, "top_products.parquet")

        product_share = (
            df["Product"]
            .dropna()
            .value_counts(normalize=True)
            .mul(100)
            .head(10)
            .reset_index()
        )
        product_share.columns = ["Product", "Share_Pct"]
        save_summary(product_share, "product_share.parquet")

        year_product_trend = (
            df[["Year", "Product"]]
            .dropna()
            .groupby(["Year", "Product"])
            .size()
            .reset_index(name="Complaints")
            .sort_values(["Year", "Complaints"], ascending=[True, False])
        )
        save_summary(year_product_trend, "year_product_trend.parquet")

        # ==========================================================
        # ISSUE ANALYSIS
        # ==========================================================

        top_issues = df["Issue"].dropna().value_counts().head(TOP_ISSUES).reset_index()
        top_issues.columns = ["Issue", "Count"]
        save_summary(top_issues, "top_issues.parquet")

        top_sub_issues = df["Sub-issue"].dropna().value_counts().head(TOP_ISSUES).reset_index()
        top_sub_issues.columns = ["Sub Issue", "Count"]
        save_summary(top_sub_issues, "top_sub_issues.parquet")

        top_issue_names = df["Issue"].dropna().value_counts().head(10).index
        top_product_names = df["Product"].dropna().value_counts().head(10).index

        issue_source = df[["Issue", "Product"]].dropna()

        issue_product_heatmap = (
            issue_source[
                issue_source["Issue"].isin(top_issue_names)
                & issue_source["Product"].isin(top_product_names)
            ]
            .groupby(["Issue", "Product"])
            .size()
            .reset_index(name="Count")
        )

        save_summary(issue_product_heatmap, "issue_product_heatmap.parquet")

        del issue_source
        gc.collect()

        issue_trend_source = df[["Year", "Issue"]].dropna()

        issue_trend = (
            issue_trend_source[
                issue_trend_source["Issue"].isin(top_issue_names)
            ]
            .groupby(["Year", "Issue"])
            .size()
            .reset_index(name="Complaints")
            .sort_values(["Year", "Complaints"], ascending=[True, False])
        )

        save_summary(issue_trend, "issue_trend.parquet")

        del issue_trend_source
        gc.collect()

        # ==========================================================
        # COMPANY ANALYSIS
        # ==========================================================

        top_companies = df["Company"].dropna().value_counts().head(TOP_COMPANIES).reset_index()
        top_companies.columns = ["Company", "Count"]
        save_summary(top_companies, "top_companies.parquet")

        company_share = (
            df["Company"]
            .dropna()
            .value_counts(normalize=True)
            .mul(100)
            .head(10)
            .reset_index()
        )
        company_share.columns = ["Company", "Share_Pct"]
        save_summary(company_share, "company_share.parquet")

        top_company_names = df["Company"].dropna().value_counts().head(10).index

        company_timely_source = df[["Company", "Timely response?"]].dropna()

        company_timely_response = (
            company_timely_source[
                company_timely_source["Company"].isin(top_company_names)
            ]
            .groupby(["Company", "Timely response?"])
            .size()
            .reset_index(name="Count")
        )

        save_summary(company_timely_response, "company_timely_response.parquet")

        del company_timely_source
        gc.collect()

        company_resolution_source = df[
            ["Company", "Company response to consumer"]
        ].dropna()

        company_resolution = (
            company_resolution_source[
                company_resolution_source["Company"].isin(top_company_names)
            ]
            .groupby(["Company", "Company response to consumer"])
            .size()
            .reset_index(name="Count")
        )

        save_summary(company_resolution, "company_resolution.parquet")

        del company_resolution_source
        gc.collect()

        credit_source = df[["Company", "Timely response?"]].dropna()

        credit_bureau_df = credit_source[
            credit_source["Company"].str.contains(
                "Experian|Equifax|TransUnion|TRANSUNION",
                case=False,
                regex=True,
            )
        ]

        credit_bureau_summary = (
            credit_bureau_df
            .groupby("Company")
            .agg(
                Total_Complaints=("Company", "size"),
                Timely_Response_Pct=(
                    "Timely response?",
                    lambda x: (x == "Yes").mean() * 100,
                ),
            )
            .reset_index()
            .sort_values("Total_Complaints", ascending=False)
            .head(10)
        )

        save_summary(credit_bureau_summary, "credit_bureau_summary.parquet")

        del credit_source, credit_bureau_df
        gc.collect()

        # ==========================================================
        # RESOLUTION ANALYSIS
        # ==========================================================

        resolution_summary = (
            df["Company response to consumer"]
            .dropna()
            .value_counts()
            .reset_index()
        )
        resolution_summary.columns = ["Resolution Type", "Count"]
        save_summary(resolution_summary, "resolution_summary.parquet")

        resolution_by_year = (
            df[["Year", "Company response to consumer"]]
            .dropna()
            .groupby(["Year", "Company response to consumer"])
            .size()
            .reset_index(name="Count")
        )
        save_summary(resolution_by_year, "resolution_by_year.parquet")

        delay_distribution = (
            df[["Resolution_Delay"]]
            .dropna()
            .query("Resolution_Delay >= 0")
        )

        delay_distribution = delay_distribution[
            delay_distribution["Resolution_Delay"] <= 365
        ]

        save_summary(delay_distribution, "delay_distribution.parquet")

        # ==========================================================
        # NARRATIVE ANALYSIS
        # ==========================================================

        narrative_availability = df["Has_Narrative"].value_counts().reset_index()
        narrative_availability.columns = ["Has Narrative", "Count"]
        save_summary(narrative_availability, "narrative_availability.parquet")

        narrative_word_count = df[["Narrative_Word_Count"]].dropna()
        narrative_word_count = narrative_word_count[
            narrative_word_count["Narrative_Word_Count"] <= 1000
        ]
        save_summary(narrative_word_count, "narrative_word_count.parquet")

        narrative_source = df[
            (df["Has_Narrative"] == 1)
            & (df["Narrative_Word_Count"].notna())
        ].copy()

        narrative_length_trend = (
            narrative_source
            .groupby("Year")["Narrative_Word_Count"]
            .mean()
            .reset_index()
        )

        narrative_length_trend.columns = ["Year", "Avg_Word_Count"]
        save_summary(narrative_length_trend, "narrative_length_trend.parquet")

        def complexity_bucket(word_count: float) -> str:
            """Convert word count into simple complexity bucket."""
            if word_count <= 50:
                return "Low"
            elif word_count <= 150:
                return "Medium"
            else:
                return "High"

        complexity_df = narrative_source[["Narrative_Word_Count"]].copy()
        complexity_df["Complexity"] = complexity_df["Narrative_Word_Count"].apply(
            complexity_bucket
        )

        complexity_summary = complexity_df["Complexity"].value_counts().reset_index()
        complexity_summary.columns = ["Complexity", "Count"]
        save_summary(complexity_summary, "narrative_complexity.parquet")

        narrative_product_length = (
            narrative_source
            .groupby("Product")["Narrative_Word_Count"]
            .mean()
            .sort_values(ascending=False)
            .head(TOP_PRODUCTS)
            .reset_index()
        )

        narrative_product_length.columns = ["Product", "Avg_Word_Count"]
        save_summary(narrative_product_length, "narrative_product_length.parquet")

        narrative_product_availability = (
            df.groupby("Product")["Has_Narrative"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .head(TOP_PRODUCTS)
            .reset_index()
        )

        narrative_product_availability.columns = [
            "Product",
            "Narrative_Availability_Pct",
        ]

        save_summary(
            narrative_product_availability,
            "narrative_product_availability.parquet",
        )

        # ==========================================================
        # TOP NARRATIVE WORDS
        # ==========================================================

        if "Consumer complaint narrative" in df.columns:
            narrative_text = df["Consumer complaint narrative"].dropna()

            if not narrative_text.empty:
                sample_size = min(50_000, len(narrative_text))

                narrative_sample = narrative_text.sample(
                    n=sample_size,
                    random_state=RANDOM_STATE,
                )

                vectorizer = CountVectorizer(
                    stop_words="english",
                    max_features=30,
                    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
                )

                X_words = vectorizer.fit_transform(narrative_sample)

                top_words = pd.DataFrame({
                    "Word": vectorizer.get_feature_names_out(),
                    "Count": X_words.sum(axis=0).A1,
                }).sort_values("Count", ascending=False)

                save_summary(top_words, "top_narrative_words.parquet")
            else:
                empty_words = pd.DataFrame(columns=["Word", "Count"])
                save_summary(empty_words, "top_narrative_words.parquet")
        else:
            empty_words = pd.DataFrame(columns=["Word", "Count"])
            save_summary(empty_words, "top_narrative_words.parquet")

        del narrative_source, complexity_df
        gc.collect()

        # ==========================================================
        # GEOGRAPHIC ANALYSIS
        # ==========================================================

        top_states = (
            df["State"]
            .dropna()
            .value_counts()
            .head(TOP_STATES)
            .reset_index()
        )

        top_states.columns = ["State", "Count"]
        save_summary(top_states, "top_states.parquet")

        state_map = (
            df["State"]
            .dropna()
            .value_counts()
            .reset_index()
        )

        state_map.columns = ["State", "Count"]
        save_summary(state_map, "state_map.parquet")

        logger.info("Dashboard summary generation completed")

    except Exception as e:
        logger.exception("Dashboard summary generation failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    build_dashboard_data()