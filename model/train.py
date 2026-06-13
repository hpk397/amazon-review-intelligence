"""
ML Pipeline: Text Preprocessing → TF-IDF → Logistic Regression
Run this once to train and save the model.
"""

import re
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from nltk.corpus import stopwords

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATA_PATH  = BASE / "data" / "amazon_reviews.json"
MODEL_PATH = BASE / "model" / "sentiment_model.pkl"
VEC_PATH   = BASE / "model" / "tfidf_vectorizer.pkl"
METRICS_PATH = BASE / "model" / "metrics.pkl"

STOP_WORDS = set(stopwords.words("english"))


# ── Preprocessing ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase → remove special chars → remove stop words."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


# ── Fake review heuristic ─────────────────────────────────────────────────────
def fake_review_score(text: str) -> float:
    """
    Simple rule-based fake review detector.
    Returns a probability 0–1 (higher = more likely fake).
    """
    score = 0.0
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl_count = text.count("!")
    word_count  = len(text.split())
    repeat_words = len(text.split()) - len(set(text.lower().split()))

    if caps_ratio > 0.4:       score += 0.35
    if excl_count >= 3:        score += 0.25
    if word_count < 8:         score += 0.15
    if repeat_words > 3:       score += 0.15
    buy_words = ["buy", "must", "now", "everyone", "best ever"]
    if sum(text.lower().count(w) for w in buy_words) >= 2:
        score += 0.10

    return min(score, 1.0)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    df = pd.read_json(DATA_PATH, lines=True)

    # Label: 4-5 → Positive (1), 1-2 → Negative (0), 3 → neutral (drop)
    df = df[df["overall"] != 3].copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    df["clean"] = df["reviewText"].apply(clean_text)
    df = df[df["clean"].str.strip() != ""]

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred) * 100, 2),
        "f1":        round(f1_score(y_test, y_pred) * 100, 2),
        "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, target_names=["Negative", "Positive"]),
    }

    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(VEC_PATH,   "wb") as f: pickle.dump(vectorizer, f)
    with open(METRICS_PATH, "wb") as f: pickle.dump(metrics, f)

    print("✅ Model trained and saved.")
    print(f"   Accuracy : {metrics['accuracy']}%")
    print(f"   F1 Score : {metrics['f1']}%")
    print(metrics["report"])
    return model, vectorizer, metrics


if __name__ == "__main__":
    train()
