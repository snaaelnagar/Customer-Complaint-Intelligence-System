"""
nlp_predictor.py

Loads trained NLP models and provides prediction functions for:
1. Product prediction (with confidence + alternatives + review decision)
2. Issue prediction (with confidence + alternatives + review decision)
3. Topic detection
4. Full complaint analysis
"""

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config_loader import config
from src.taxonomy import canonicalize_product, canonicalize_issue


MODEL_DIR = Path(config["paths"]["model_dir"])

PRODUCT_MODEL_PATH = MODEL_DIR / "product_classifier_model.pkl"
PRODUCT_VECTORIZER_PATH = MODEL_DIR / "product_classifier_vectorizer.pkl"

ISSUE_MODEL_PATH = MODEL_DIR / "issue_classifier_model.pkl"
ISSUE_VECTORIZER_PATH = MODEL_DIR / "issue_classifier_vectorizer.pkl"

TOPIC_MODEL_PATH = MODEL_DIR / "topic_model.pkl"
TOPIC_VECTORIZER_PATH = MODEL_DIR / "topic_vectorizer.pkl"
TOPIC_WORDS_PATH = MODEL_DIR / "topic_words.pkl"

_NLP_CONFIG = config.get("nlp", {})
CONFIDENCE_THRESHOLD = _NLP_CONFIG.get("confidence_threshold", 0.55)
TOP_K_ALTERNATIVES = _NLP_CONFIG.get("top_k_alternatives", 2)

AUTO_ACCEPT = "AUTO_ACCEPT"
HUMAN_REVIEW = "HUMAN_REVIEW"


def clean_text(text: str) -> str:
    """Clean complaint text before prediction."""
    text = str(text).lower()
    text = re.sub(r"\bxx+\b", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_model(path: Path) -> Any:
    """
    Load a saved joblib model safely and patch legacy attributes 
    for scikit-learn compatibility.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)

    # Patch for scikit-learn version mismatch ('multi_class' attribute)
    # نتحقق الأول إن الكائن مش dictionary قبل ما نضيف الخاصية
    if not isinstance(model, dict) and not hasattr(model, "multi_class"):
        try:
            setattr(model, "multi_class", "auto")
        except AttributeError:
            pass

    return model


product_model = load_model(PRODUCT_MODEL_PATH)
product_vectorizer = load_model(PRODUCT_VECTORIZER_PATH)

issue_model = load_model(ISSUE_MODEL_PATH)
issue_vectorizer = load_model(ISSUE_VECTORIZER_PATH)

topic_model = load_model(TOPIC_MODEL_PATH)
topic_vectorizer = load_model(TOPIC_VECTORIZER_PATH)
topic_words = load_model(TOPIC_WORDS_PATH)


def validate_input_text(text: str) -> str:
    """Validate input complaint text."""
    cleaned = clean_text(text)

    if not cleaned:
        raise ValueError("Complaint text is empty after cleaning")

    return cleaned


def _canonical_probabilities(proba_row: np.ndarray, classes: np.ndarray, canonicalize_fn) -> dict:
    """Collapse raw-class probabilities into canonical-label probabilities."""
    canonical_scores: dict = {}

    for raw_label, prob in zip(classes, proba_row):
        canonical_label = canonicalize_fn(raw_label)
        canonical_scores[canonical_label] = canonical_scores.get(canonical_label, 0.0) + float(prob)

    return canonical_scores


def predict_with_confidence(
    text: str,
    model,
    vectorizer,
    canonicalize_fn,
    top_k_alternatives: int = TOP_K_ALTERNATIVES,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """Predict a label with taxonomy-aware confidence and human-review decision."""
    cleaned = validate_input_text(text)
    vector = vectorizer.transform([cleaned])

    raw_label = model.predict(vector)[0]

    proba_row = model.predict_proba(vector)[0]
    canonical_scores = _canonical_probabilities(proba_row, model.classes_, canonicalize_fn)

    ranked = sorted(canonical_scores.items(), key=lambda item: item[1], reverse=True)

    top_label, top_confidence = ranked[0]
    alternatives = [
        {"label": label, "confidence": round(score, 4)}
        for label, score in ranked[1 : 1 + top_k_alternatives]
    ]

    decision = AUTO_ACCEPT if top_confidence >= confidence_threshold else HUMAN_REVIEW

    return {
        "label": top_label,
        "confidence": round(top_confidence, 4),
        "raw_label": raw_label,
        "alternatives": alternatives,
        "decision": decision,
        "confidence_threshold": confidence_threshold,
    }


def predict_product(text: str) -> dict:
    """Predict Product category."""
    return predict_with_confidence(
        text=text,
        model=product_model,
        vectorizer=product_vectorizer,
        canonicalize_fn=canonicalize_product,
    )


def predict_issue(text: str) -> dict:
    """Predict Issue category."""
    return predict_with_confidence(
        text=text,
        model=issue_model,
        vectorizer=issue_vectorizer,
        canonicalize_fn=canonicalize_issue,
    )


def predict_topic(text: str) -> dict:
    """Predict topic ID, topic keywords, and confidence score."""
    cleaned = validate_input_text(text)
    vector = topic_vectorizer.transform([cleaned])

    topic_probs = topic_model.transform(vector)
    topic_id = int(topic_probs.argmax())

    return {
        "topic_id": topic_id,
        "topic_words": topic_words[topic_id],
        "confidence": float(topic_probs[0][topic_id]),
    }


def analyze_complaint(text: str) -> dict:
    """Run complete NLP analysis on one complaint narrative."""
    return {
        "product": predict_product(text),
        "issue": predict_issue(text),
        "topic": predict_topic(text),
    }


if __name__ == "__main__":
    sample_text = """
    Someone opened fraudulent accounts in my name and there are inquiries
    on my credit report that I do not recognize.
    """

    result = analyze_complaint(sample_text)
    print(result)