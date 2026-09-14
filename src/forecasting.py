"""
forecasting.py

Builds complaint volume forecast using Prophet.

Input:
data/processed/dashboard/monthly_complaint_trend.parquet

Outputs:
1. complaint_forecast.parquet
2. forecast_summary.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

from src.logger import logger
from src.exception import CustomException
from src.config_loader import config


DASHBOARD_DIR = Path(config["paths"]["dashboard_dir"])

MONTHLY_TREND_PATH = DASHBOARD_DIR / "monthly_complaint_trend.parquet"
FORECAST_OUTPUT_PATH = DASHBOARD_DIR / "complaint_forecast.parquet"
FORECAST_SUMMARY_PATH = DASHBOARD_DIR / "forecast_summary.parquet"

FORECAST_PERIODS = config["forecasting"]["periods"]
TRAINING_YEARS = config["forecasting"]["training_years"]
YEARLY_SEASONALITY = config["forecasting"]["yearly_seasonality"]
WEEKLY_SEASONALITY = config["forecasting"]["weekly_seasonality"]
DAILY_SEASONALITY = config["forecasting"]["daily_seasonality"]
SEASONALITY_MODE = config["forecasting"]["seasonality_mode"]

TREND_THRESHOLD = config["forecasting"].get("trend_threshold", 5)


def calculate_mape(actual, predicted) -> float:
    """Calculate MAPE safely by ignoring zero actual values."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return 0.0

    return np.mean(
        np.abs((actual[mask] - predicted[mask]) / actual[mask])
    ) * 100


def create_prophet_model() -> Prophet:
    """Create Prophet model using config.yaml settings."""
    return Prophet(
        yearly_seasonality=YEARLY_SEASONALITY,
        weekly_seasonality=WEEKLY_SEASONALITY,
        daily_seasonality=DAILY_SEASONALITY,
        seasonality_mode=SEASONALITY_MODE,
    )


def validate_forecast(train_df: pd.DataFrame) -> tuple[float, float]:
    """
    Validate Prophet model using last 6 months as holdout.

    Train on older months, predict holdout months,
    then calculate MAE and MAPE.
    """
    if len(train_df) < 18:
        logger.warning("Not enough data for forecast validation")
        return 0.0, 0.0

    validation_periods = 6

    validation_train = train_df.iloc[:-validation_periods].copy()
    validation_test = train_df.iloc[-validation_periods:].copy()

    validation_model = create_prophet_model()
    validation_model.fit(validation_train)

    future = validation_model.make_future_dataframe(
        periods=validation_periods,
        freq="MS",
    )

    validation_forecast = validation_model.predict(future)

    predicted = (
        validation_forecast
        .tail(validation_periods)["yhat"]
        .clip(lower=0)
        .values
    )

    actual = validation_test["y"].values

    mae = mean_absolute_error(actual, predicted)
    mape = calculate_mape(actual, predicted)

    return float(mae), float(mape)


def build_forecast(
    periods: int = FORECAST_PERIODS,
    training_years: int = TRAINING_YEARS,
) -> None:
    """Build Prophet forecast and forecast summary."""
    try:
        logger.info("Prophet complaint forecasting started")

        if not MONTHLY_TREND_PATH.exists():
            raise FileNotFoundError(
                f"Monthly trend file not found: {MONTHLY_TREND_PATH}. "
                "Run growth_analysis.py first."
            )

        monthly = pd.read_parquet(MONTHLY_TREND_PATH)

        if monthly.empty:
            raise ValueError("Monthly trend data is empty")

        required_columns = ["Year", "Month", "Complaints"]

        missing_columns = [
            col for col in required_columns if col not in monthly.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in monthly trend data: {missing_columns}"
            )

        monthly = monthly.dropna(subset=["Year", "Month", "Complaints"])

        if monthly.empty:
            raise ValueError("No valid monthly rows available for forecasting")

        monthly["Year"] = monthly["Year"].astype(int)
        monthly["Month"] = monthly["Month"].astype(int)

        max_year = int(monthly["Year"].max())

        # Exclude max year because it may be partial/incomplete.
        monthly_complete = monthly[monthly["Year"] < max_year].copy()

        if monthly_complete.empty:
            raise ValueError("No complete historical years available for forecasting")

        monthly_complete["ds"] = pd.to_datetime(
            monthly_complete["Year"].astype(str)
            + "-"
            + monthly_complete["Month"].astype(str).str.zfill(2)
            + "-01"
        )

        monthly_complete = monthly_complete.sort_values("ds").reset_index(drop=True)

        latest_complete_year = int(monthly_complete["Year"].max())
        start_training_year = latest_complete_year - training_years + 1

        train = monthly_complete[
            monthly_complete["Year"] >= start_training_year
        ].copy()

        train = train[["ds", "Complaints"]].rename(
            columns={"Complaints": "y"}
        )

        if len(train) < 12:
            raise ValueError("Not enough monthly data available for forecasting.")

        validation_mae, validation_mape = validate_forecast(train)

        model = create_prophet_model()
        model.fit(train)

        future = model.make_future_dataframe(
            periods=periods,
            freq="MS",
        )

        forecast = model.predict(future)

        forecast_result = forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()

        forecast_result = forecast_result.rename(
            columns={
                "ds": "Date",
                "yhat": "Complaints",
                "yhat_lower": "Lower_Bound",
                "yhat_upper": "Upper_Bound",
            }
        )

        for col in ["Complaints", "Lower_Bound", "Upper_Bound"]:
            forecast_result[col] = (
                forecast_result[col]
                .clip(lower=0)
                .round()
                .astype(int)
            )

        actual_dates = set(train["ds"])

        forecast_result["Type"] = forecast_result["Date"].apply(
            lambda x: "Actual" if x in actual_dates else "Forecast"
        )

        actual_values = train.rename(
            columns={
                "ds": "Date",
                "y": "Actual_Complaints",
            }
        )

        forecast_result = forecast_result.merge(
            actual_values,
            on="Date",
            how="left",
        )

        forecast_result["Complaints"] = forecast_result.apply(
            lambda row: int(row["Actual_Complaints"])
            if pd.notna(row["Actual_Complaints"])
            else int(row["Complaints"]),
            axis=1,
        )

        forecast_result = forecast_result.drop(columns=["Actual_Complaints"])

        forecast_result["Period"] = forecast_result["Date"].dt.strftime("%Y-%m")

        forecast_result.to_parquet(FORECAST_OUTPUT_PATH, index=False)

        forecast_only = forecast_result[
            forecast_result["Type"] == "Forecast"
        ].copy()

        if forecast_only.empty:
            raise ValueError("Forecast output is empty")

        next_month_forecast = int(forecast_only.iloc[0]["Complaints"])
        next_3_month_forecast = int(forecast_only.head(3)["Complaints"].sum())
        next_6_month_forecast = int(forecast_only.head(6)["Complaints"].sum())

        latest_actual = int(train.iloc[-1]["y"])

        expected_change_pct = (
            (next_month_forecast - latest_actual) / latest_actual * 100
            if latest_actual != 0
            else 0
        )

        if expected_change_pct >= TREND_THRESHOLD:
            trend_label = "Increasing"
        elif expected_change_pct <= -TREND_THRESHOLD:
            trend_label = "Decreasing"
        else:
            trend_label = "Stable"

        forecast_summary = pd.DataFrame(
            [
                {
                    "Model": config["forecasting"]["model"],
                    "Training_Start_Year": start_training_year,
                    "Training_End_Year": latest_complete_year,
                    "Excluded_Partial_Year": max_year,
                    "Latest_Actual_Complaints": latest_actual,
                    "Next_Month_Forecast": next_month_forecast,
                    "Next_3_Month_Forecast": next_3_month_forecast,
                    "Next_6_Month_Forecast": next_6_month_forecast,
                    "Expected_Change_Pct": expected_change_pct,
                    "Forecast_Trend": trend_label,
                    "Validation_MAE": validation_mae,
                    "Validation_MAPE": validation_mape,
                }
            ]
        )

        forecast_summary.to_parquet(FORECAST_SUMMARY_PATH, index=False)

        logger.info(
            f"Prophet forecast trained on {start_training_year}-{latest_complete_year}, "
            f"excluded partial year {max_year}"
        )
        logger.info(f"Validation MAE: {validation_mae:.2f}")
        logger.info(f"Validation MAPE: {validation_mape:.2f}%")
        logger.info(f"Forecast saved: {FORECAST_OUTPUT_PATH}")
        logger.info(f"Forecast summary saved: {FORECAST_SUMMARY_PATH}")
        logger.info("Prophet complaint forecasting completed")

    except Exception as e:
        logger.exception("Complaint forecasting failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    build_forecast()