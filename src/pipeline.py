import sys

from src.logger import logger
from src.exception import CustomException

from src.dashboard_data import build_dashboard_data
from src.risk_analysis import build_company_risk_score
from src.driver_analysis import build_driver_analysis
from src.growth_analysis import build_growth_analysis
from src.forecasting import build_forecast
from src.recommendation_engine import build_recommendations


def run_pipeline() -> None:
    """
    Run the complete Business Analytics pipeline (dashboard aggregation,
    risk scoring, driver analysis, growth analysis, forecasting, and
    recommendations).

    Execution Order:

    1. Dashboard Aggregations
    2. Company Risk Scoring
    3. Complaint Driver Analysis
    4. Growth Analysis
    5. Forecasting
    6. Recommendation Engine

    Note:
    NLP model training is intentionally not included in this pipeline,
    because training on millions of narratives is computationally
    expensive. This pipeline refreshes analytics outputs using
    already-trained NLP artifacts (models/nlp/*.pkl). Run NLP training
    separately (`python -m src.nlp_model_training`) only when model
    retraining is required.
    """
    try:
        logger.info("Full analytics pipeline started")

        build_dashboard_data()
        build_company_risk_score()
        build_driver_analysis()
        build_growth_analysis()
        build_forecast()
        build_recommendations()

        logger.info("Full analytics pipeline completed successfully")

    except Exception as e:
        logger.exception("Full analytics pipeline failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_pipeline()