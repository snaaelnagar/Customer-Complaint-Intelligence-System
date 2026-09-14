"""
recommendation_engine.py

Builds prescriptive recommendations and executive action plan
from risk, growth, forecasting, and complaint driver outputs.
"""

import sys
from pathlib import Path

import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.config_loader import config


DASHBOARD_DIR = Path(config["paths"]["dashboard_dir"])

COMPANY_RISK_PATH = DASHBOARD_DIR / "company_risk_score.parquet"
PRODUCT_GROWTH_PATH = DASHBOARD_DIR / "product_growth.parquet"
ISSUE_GROWTH_PATH = DASHBOARD_DIR / "issue_growth.parquet"
FORECAST_SUMMARY_PATH = DASHBOARD_DIR / "forecast_summary.parquet"
TOP_DRIVERS_PATH = DASHBOARD_DIR / "top_complaint_drivers.parquet"

RECOMMENDATIONS_PATH = DASHBOARD_DIR / "recommendations.parquet"
ACTION_PLAN_PATH = DASHBOARD_DIR / "executive_action_plan.parquet"

TOP_HIGH_RISK_COMPANIES = config["recommendation_engine"]["top_high_risk_companies"]
TOP_PRODUCT_GROWTH = config["recommendation_engine"]["top_product_growth"]
TOP_ISSUE_GROWTH = config["recommendation_engine"]["top_issue_growth"]
TOP_COMPLAINT_DRIVERS = config["recommendation_engine"]["top_complaint_drivers"]


def get_issue_specific_recommendation(text: str) -> str:
    """Return recommendation based on issue keywords."""
    text = str(text).lower()

    if "fraud" in text or "unauthorized" in text or "scam" in text:
        return (
            "Strengthen fraud detection rules, improve transaction verification, "
            "monitor suspicious activity patterns, and create a faster fraud-dispute workflow."
        )

    if "credit report" in text or "incorrect information" in text or "report" in text:
        return (
            "Audit credit reporting data pipelines, improve bureau reporting accuracy, "
            "reduce dispute resolution delays, and strengthen data quality controls."
        )

    if "debt" in text or "collect" in text:
        return (
            "Review debt validation workflows, monitor collection communication practices, "
            "and ensure collectors follow proper documentation and compliance procedures."
        )

    if "mortgage" in text or "foreclosure" in text:
        return (
            "Improve mortgage servicing communication, review foreclosure prevention workflows, "
            "and strengthen hardship assistance support."
        )

    if "payment" in text or "funds" in text or "transaction" in text:
        return (
            "Investigate payment processing failures, improve transaction status visibility, "
            "and reduce delays in fund availability and dispute handling."
        )

    if "disclosure" in text or "confusing" in text:
        return (
            "Simplify customer disclosures, improve fee and policy communication, "
            "and review customer-facing documentation for clarity."
        )

    if "service" in text or "account" in text:
        return (
            "Improve customer service escalation paths, reduce response time, "
            "and create clearer account-resolution playbooks."
        )

    return (
        "Perform root-cause analysis, identify process gaps, and create a targeted "
        "resolution playbook for this complaint category."
    )


def get_product_specific_recommendation(product: str) -> str:
    """Return recommendation based on product category."""
    product = str(product).lower()

    if "credit reporting" in product:
        return (
            "Prioritize credit reporting quality audits, improve dispute handling, "
            "and monitor bureau-related complaint drivers."
        )

    if "debt collection" in product:
        return (
            "Review collection practices, strengthen debt validation checks, "
            "and monitor communication-related complaints."
        )

    if "mortgage" in product:
        return (
            "Improve loan servicing transparency, strengthen foreclosure assistance, "
            "and review hardship support workflows."
        )

    if "money transfer" in product or "virtual currency" in product:
        return (
            "Strengthen transaction monitoring, fraud controls, refund workflows, "
            "and digital wallet support processes."
        )

    if "checking" in product or "savings" in product:
        return (
            "Review overdraft, account access, fee, and fund availability processes."
        )

    return (
        "Investigate top issues within this product and allocate monitoring resources "
        "to the fastest-growing complaint areas."
    )


def risk_recommendations(company_risk: pd.DataFrame) -> pd.DataFrame:
    """Create recommendations for highest-risk companies."""
    rows = []

    if company_risk.empty:
        logger.warning("Company risk data is empty")
        return pd.DataFrame(rows)

    high_risk = company_risk[
        company_risk["Risk_Level"].astype(str) == "High"
    ].sort_values(
        "Risk_Score",
        ascending=False,
    ).head(TOP_HIGH_RISK_COMPANIES)

    for _, row in high_risk.iterrows():
        rows.append({
            "Category": "Company Risk",
            "Priority": "High",
            "Entity": row["Company"],
            "Signal": (
                f"Risk Score: {row['Risk_Score']:.2f} | "
                f"Complaints: {int(row['Complaint_Count'])} | "
                f"Untimely Response: {row['Untimely_Response_Pct']:.2f}%"
            ),
            "Recommendation": (
                "Prioritize this company for operational review, audit response delays, "
                "improve SLA compliance, and investigate complaint handling bottlenecks."
            ),
        })

    return pd.DataFrame(rows)


def growth_recommendations(
    product_growth: pd.DataFrame,
    issue_growth: pd.DataFrame,
) -> pd.DataFrame:
    """Create product and issue growth recommendations."""
    rows = []

    if not product_growth.empty:
        product_growth = product_growth.sort_values(
            "YoY_Growth_Pct",
            ascending=False,
        )

        for _, row in product_growth.head(TOP_PRODUCT_GROWTH).iterrows():
            rows.append({
                "Category": "Product Growth",
                "Priority": "Medium",
                "Entity": row["Product"],
                "Signal": (
                    f"YoY Growth: {row['YoY_Growth_Pct']:.2f}% | "
                    f"Current Complaints: {int(row['Current_Year_Complaints'])}"
                ),
                "Recommendation": get_product_specific_recommendation(row["Product"]),
            })

    if not issue_growth.empty:
        issue_growth = issue_growth.sort_values(
            "YoY_Growth_Pct",
            ascending=False,
        )

        for _, row in issue_growth.head(TOP_ISSUE_GROWTH).iterrows():
            rows.append({
                "Category": "Issue Growth",
                "Priority": "High",
                "Entity": row["Issue"],
                "Signal": (
                    f"YoY Growth: {row['YoY_Growth_Pct']:.2f}% | "
                    f"Current Complaints: {int(row['Current_Year_Complaints'])}"
                ),
                "Recommendation": get_issue_specific_recommendation(row["Issue"]),
            })

    return pd.DataFrame(rows)


def forecast_recommendations(forecast_summary: pd.DataFrame) -> pd.DataFrame:
    """Create recommendation based on forecast trend."""
    if forecast_summary.empty:
        raise ValueError("Forecast summary is empty")

    row = forecast_summary.iloc[0]

    trend = row["Forecast_Trend"]
    change_pct = row["Expected_Change_Pct"]
    mape = row.get("Validation_MAPE", 0)

    if trend == "Increasing":
        priority = "High"
        recommendation = (
            "Complaint volume is expected to increase. Increase complaint handling capacity, "
            "prepare support teams for higher workload, and monitor fast-growing products and issues."
        )

    elif trend == "Decreasing":
        priority = "Low"
        recommendation = (
            "Complaint volume is expected to decrease. Maintain current monitoring, "
            "but continue focusing on high-risk companies and major complaint drivers."
        )

    else:
        priority = "Medium"
        recommendation = (
            "Complaint volume is expected to remain stable. Maintain current capacity "
            "and continue monitoring risk and growth signals."
        )

    return pd.DataFrame([{
        "Category": "Forecast",
        "Priority": priority,
        "Entity": "Overall Complaint Volume",
        "Signal": (
            f"{trend} | Expected Change: {change_pct:.2f}% | "
            f"Validation MAPE: {mape:.2f}%"
        ),
        "Recommendation": recommendation,
    }])


def driver_recommendations(top_drivers: pd.DataFrame) -> pd.DataFrame:
    """Create recommendations for top complaint drivers."""
    rows = []

    if top_drivers.empty:
        logger.warning("Top complaint drivers data is empty")
        return pd.DataFrame(rows)

    driver_df = (
        top_drivers
        .sort_values("Complaint_Count", ascending=False)
        .drop_duplicates(subset=["Product", "Issue"])
        .head(TOP_COMPLAINT_DRIVERS)
    )

    for _, row in driver_df.iterrows():
        entity = f"{row['Product']} → {row['Issue']}"

        rows.append({
            "Category": "Complaint Driver",
            "Priority": "High",
            "Entity": entity,
            "Signal": f"Complaints: {int(row['Complaint_Count'])}",
            "Recommendation": (
                get_issue_specific_recommendation(row["Issue"])
                + " This driver has high complaint volume, so it should be treated as a priority root-cause area."
            ),
        })

    return pd.DataFrame(rows)


def build_executive_action_plan(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Build final executive action plan from recommendations."""
    category_order = [
        "Forecast",
        "Company Risk",
        "Complaint Driver",
        "Issue Growth",
        "Product Growth",
    ]

    rows = []
    rank = 1

    if recommendations.empty:
        return pd.DataFrame(
            columns=[
                "Action_Rank",
                "Focus_Area",
                "Target",
                "Why_It_Matters",
                "Recommended_Action",
            ]
        )

    for category in category_order:
        subset = recommendations[recommendations["Category"] == category]

        if subset.empty:
            continue

        row = subset.iloc[0]

        rows.append({
            "Action_Rank": rank,
            "Focus_Area": row["Category"],
            "Target": row["Entity"],
            "Why_It_Matters": row["Signal"],
            "Recommended_Action": row["Recommendation"],
        })

        rank += 1

    return pd.DataFrame(rows)


def build_recommendations() -> None:
    """Main function to build recommendations and action plan."""
    try:
        logger.info("Recommendation engine started")

        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

        required_files = [
            COMPANY_RISK_PATH,
            PRODUCT_GROWTH_PATH,
            ISSUE_GROWTH_PATH,
            FORECAST_SUMMARY_PATH,
            TOP_DRIVERS_PATH,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        company_risk = pd.read_parquet(COMPANY_RISK_PATH)
        product_growth = pd.read_parquet(PRODUCT_GROWTH_PATH)
        issue_growth = pd.read_parquet(ISSUE_GROWTH_PATH)
        forecast_summary = pd.read_parquet(FORECAST_SUMMARY_PATH)
        top_drivers = pd.read_parquet(TOP_DRIVERS_PATH)

        recommendation_parts = [
            forecast_recommendations(forecast_summary),
            risk_recommendations(company_risk),
            driver_recommendations(top_drivers),
            growth_recommendations(product_growth, issue_growth),
        ]

        recommendation_parts = [
            part for part in recommendation_parts if not part.empty
        ]

        if recommendation_parts:
            recommendations = pd.concat(
                recommendation_parts,
                ignore_index=True,
            )
        else:
            recommendations = pd.DataFrame(
                columns=[
                    "Category",
                    "Priority",
                    "Entity",
                    "Signal",
                    "Recommendation",
                ]
            )

        priority_order = {
            "High": 1,
            "Medium": 2,
            "Low": 3,
        }

        if not recommendations.empty:
            recommendations["Priority_Rank"] = recommendations["Priority"].map(
                priority_order
            )

            recommendations = recommendations.sort_values(
                ["Priority_Rank", "Category"]
            ).drop(columns=["Priority_Rank"])

        action_plan = build_executive_action_plan(recommendations)

        recommendations.to_parquet(RECOMMENDATIONS_PATH, index=False)
        action_plan.to_parquet(ACTION_PLAN_PATH, index=False)

        logger.info(f"Recommendations saved: {RECOMMENDATIONS_PATH}")
        logger.info(f"Executive action plan saved: {ACTION_PLAN_PATH}")
        logger.info("Recommendation engine completed")

    except Exception as e:
        logger.exception("Recommendation engine failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    build_recommendations()