import re
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.config_loader import config
from src.taxonomy import canonicalize_product, canonicalize_issue
from src.nlp_evaluation import evaluate_classifier, format_evaluation_report


TRAINING_DATA_PATH = Path(
    config["paths"].get(
        "narratives_training_path",
        "data/processed/narratives_training.parquet",
    )
)
MODEL_DIR = Path(config["paths"]["model_dir"])

_NLP_CONFIG = config["nlp"]

RANDOM_STATE = _NLP_CONFIG["random_state"]
SAMPLE_SIZE = _NLP_CONFIG.get("sample_size", 300_000)
TOP_K_VALUES = tuple(_NLP_CONFIG.get("top_k_evaluation_values", [1, 3, 5]))

# Per-classifier hyperparameters, read from config.yaml (nlp.product_classifier /
# nlp.issue_classifier / nlp.topic_model) instead of being hardcoded here, so
# tuning the model doesn't require touching this file.
_PRODUCT_CFG = _NLP_CONFIG["product_classifier"]
_ISSUE_CFG = _NLP_CONFIG["issue_classifier"]
_TOPIC_CFG = _NLP_CONFIG["topic_model"]

CLASSIFIER_CONFIG = {
    "Product": _PRODUCT_CFG,
    "Issue": _ISSUE_CFG,
}

# Which canonicalizer applies to which raw target column. The raw
# column itself is never modified -- canonicalization only affects
# what the model is trained/evaluated against, and the raw label is
# always kept alongside it (see `train_classifier`).
CANONICALIZERS = {
    "Product": canonicalize_product,
    "Issue": canonicalize_issue,
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\bxx+\b", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_training_data() -> pd.DataFrame:
    logger.info("Loading narrative training data")

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Narrative training data not found: {TRAINING_DATA_PATH}. "
            "Run `python -m src.create_narrative_training_data` first."
        )

    df = pd.read_parquet(TRAINING_DATA_PATH)

    if df.empty:
        raise ValueError(f"Training data is empty: {TRAINING_DATA_PATH}")

    logger.info(f"Total narrative rows: {len(df)}")

    if len(df) > SAMPLE_SIZE:
        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        ).copy()

    logger.info(f"Sampled training rows: {len(df)}")

    df["clean_text"] = (
        df["Consumer complaint narrative"]
        .apply(clean_text)
    )

    return df


def train_classifier(df: pd.DataFrame, target_col: str, model_prefix: str) -> None:
    """
    Train a classifier on the canonical version of `target_col`.

    The original raw label is preserved as `{target_col}_raw` in the
    working frame (never deleted, never overwritten) so every
    prediction can always be traced back to the exact CFPB taxonomy
    label it came from. The model itself is trained on the canonical
    label, since that is what fixes the label-fragmentation problem
    described in src/taxonomy.py.
    """
    logger.info(f"Training started: {model_prefix}")

    clf_cfg = CLASSIFIER_CONFIG[target_col]
    canonicalize_fn = CANONICALIZERS.get(target_col)

    model_df = df[["clean_text", target_col]].dropna().copy()
    model_df = model_df.rename(columns={target_col: f"{target_col}_raw"})

    if canonicalize_fn is not None:
        model_df[target_col] = model_df[f"{target_col}_raw"].apply(canonicalize_fn)
        raw_class_count = model_df[f"{target_col}_raw"].nunique()
        canonical_class_count = model_df[target_col].nunique()
        logger.info(
            f"{model_prefix}: taxonomy canonicalization collapsed "
            f"{raw_class_count} raw '{target_col}' labels into "
            f"{canonical_class_count} canonical labels"
        )
    else:
        model_df[target_col] = model_df[f"{target_col}_raw"]

    class_counts = model_df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= clf_cfg["min_class_samples"]].index

    model_df = model_df[
        model_df[target_col].isin(valid_classes)
    ].copy()

    if model_df[target_col].nunique() < 2:
        raise ValueError(f"Not enough classes remain to train {model_prefix}")

    X = model_df["clean_text"]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=clf_cfg["max_features"],
        ngram_range=(clf_cfg["ngram_min"], clf_cfg["ngram_max"]),
        min_df=clf_cfg["min_df"],
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=clf_cfg["max_iter"],
        n_jobs=-1,
    )

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)

    result = evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=model.classes_,
        top_k_values=TOP_K_VALUES,
    )

    report_text = format_evaluation_report(
        title=f"{model_prefix} (canonical labels, target='{target_col}')",
        result=result,
    )

    logger.info(f"{model_prefix} macro F1: {result.macro_f1:.4f}")
    logger.info(f"{model_prefix} weighted F1: {result.weighted_f1:.4f}")
    logger.info(f"\n{report_text}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / f"{model_prefix}_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / f"{model_prefix}_vectorizer.pkl")

    with open(MODEL_DIR / f"{model_prefix}_metrics.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Training completed: {model_prefix}")


def train_topic_model(df: pd.DataFrame) -> None:
    logger.info("Topic modeling started")

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=_TOPIC_CFG["max_features"],
        min_df=_TOPIC_CFG["min_df"],
        ngram_range=(_TOPIC_CFG["ngram_min"], _TOPIC_CFG["ngram_max"]),
    )

    X_topic = vectorizer.fit_transform(df["clean_text"])

    lda = LatentDirichletAllocation(
        n_components=_TOPIC_CFG["n_topics"],
        random_state=RANDOM_STATE,
        learning_method="batch",
    )

    lda.fit(X_topic)

    feature_names = vectorizer.get_feature_names_out()

    topic_words = {}

    for topic_idx, topic in enumerate(lda.components_):
        words = [
            feature_names[i]
            for i in topic.argsort()[-15:][::-1]
        ]

        topic_words[topic_idx] = words
        logger.info(f"Topic {topic_idx}: {words}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(lda, MODEL_DIR / "topic_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "topic_vectorizer.pkl")
    joblib.dump(topic_words, MODEL_DIR / "topic_words.pkl")

    logger.info("Topic modeling completed")


def train_nlp_models() -> None:
    try:
        logger.info("Final NLP training started")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        df = load_training_data()

        logger.info(f"Training data shape: {df.shape}")

        train_classifier(
            df=df,
            target_col="Product",
            model_prefix="product_classifier",
        )

        train_classifier(
            df=df,
            target_col="Issue",
            model_prefix="issue_classifier",
        )

        train_topic_model(df)

        logger.info("Final NLP training completed")

    except Exception as e:
        logger.exception("Final NLP training failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    train_nlp_models()
