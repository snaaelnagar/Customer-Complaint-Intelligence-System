"""
nlp_tuning.py

Small hyperparameter tuning for NLP classifiers.
Runs limited tuning on sample data to avoid very long training time.
"""

import sys
import re
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.logger import logger
from src.exception import CustomException
from src.config_loader import config


TRAINING_DATA_PATH = Path(
    config["paths"].get(
        "narratives_training_path",
        "data/processed/narratives_training.parquet",
    )
)

MODEL_DIR = Path(config["paths"]["model_dir"])

RANDOM_STATE = config["nlp"]["random_state"]
TUNING_SAMPLE_SIZE = config["nlp"].get("tuning_sample_size", 100000)

# Fixed hyperparameters that aren't part of the tuning grid below,
# read from config.yaml (nlp.product_classifier / nlp.issue_classifier)
# so they stay consistent with src/nlp_model_training.py.
CLASSIFIER_CONFIG = {
    "Product": config["nlp"]["product_classifier"],
    "Issue": config["nlp"]["issue_classifier"],
}


PARAM_GRID = [
    {"C": 0.5, "class_weight": None, "max_features": 20000},
    {"C": 1.0, "class_weight": None, "max_features": 20000},
    {"C": 2.0, "class_weight": None, "max_features": 20000},
    {"C": 1.0, "class_weight": "balanced", "max_features": 20000},
    {"C": 2.0, "class_weight": "balanced", "max_features": 20000},
    {"C": 1.0, "class_weight": None, "max_features": 30000},
]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\bxx+\b", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_data() -> pd.DataFrame:
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_DATA_PATH}")

    df = pd.read_parquet(TRAINING_DATA_PATH)

    if df.empty:
        raise ValueError("Training data is empty")

    if len(df) > TUNING_SAMPLE_SIZE:
        df = df.sample(
            n=TUNING_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        ).copy()

    df["clean_text"] = df["Consumer complaint narrative"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()

    return df


def tune_classifier(df: pd.DataFrame, target_col: str, model_prefix: str) -> None:
    logger.info(f"Tuning started for {model_prefix}")

    clf_cfg = CLASSIFIER_CONFIG[target_col]

    model_df = df[["clean_text", target_col]].dropna().copy()

    class_counts = model_df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= clf_cfg["min_class_samples"]].index

    model_df = model_df[model_df[target_col].isin(valid_classes)].copy()

    if model_df[target_col].nunique() < 2:
        raise ValueError(f"Not enough classes for {model_prefix}")

    X = model_df["clean_text"]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    best_accuracy = 0
    best_model = None
    best_vectorizer = None
    best_params = None
    best_report = None

    for params in PARAM_GRID:
        logger.info(f"Testing params for {model_prefix}: {params}")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=params["max_features"],
            min_df=clf_cfg["min_df"],
            ngram_range=(clf_cfg["ngram_min"], clf_cfg["ngram_max"]),
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=clf_cfg["max_iter"],
            n_jobs=-1,
        )

        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"{model_prefix} params={params} accuracy={accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_vectorizer = vectorizer
            best_params = params
            best_report = classification_report(
                y_test,
                y_pred,
                zero_division=0,
            )

    logger.info(f"Best {model_prefix} accuracy: {best_accuracy}")
    logger.info(f"Best {model_prefix} params: {best_params}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / f"{model_prefix}_model.pkl")
    joblib.dump(best_vectorizer, MODEL_DIR / f"{model_prefix}_vectorizer.pkl")

    with open(MODEL_DIR / f"{model_prefix}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Best Accuracy: {best_accuracy}\n")
        f.write(f"Best Params: {best_params}\n\n")
        f.write(best_report)

    logger.info(f"Tuned model saved: {model_prefix}")


def run_nlp_tuning() -> None:
    try:
        logger.info("NLP tuning pipeline started")

        df = load_data()

        tune_classifier(
            df=df,
            target_col="Product",
            model_prefix="product_classifier",
        )

        tune_classifier(
            df=df,
            target_col="Issue",
            model_prefix="issue_classifier",
        )

        logger.info("NLP tuning pipeline completed")

    except Exception as e:
        logger.exception("NLP tuning failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_nlp_tuning()