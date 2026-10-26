"""
trainer/model.py

Trains a TF-IDF + Logistic Regression spam classifier on the UCI SMS Spam
Collection and saves it as a single sklearn Pipeline (vectorizer + model).

Usage (standalone):
    python model.py

Usage (from the Flask API):
    from model import train_model
    metrics = train_model()
"""

import json
import os
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Paths can be overridden with environment variables (handy inside Docker).
DATA_PATH = os.getenv("DATA_PATH", "data\SMSSpamCollection")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "models/metrics.json")


def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load the tab-separated SMS Spam Collection file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Mount your data folder or set the DATA_PATH environment variable."
        )

    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["label", "text"],
        encoding="latin-1",
    )
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # ham -> 0, spam -> 1
    df["target"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    return df


def build_pipeline() -> Pipeline:
    """Vectorizer + classifier in one object so they save/load together."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=10,
                    class_weight="balanced",  # spam is the minority class
                ),
            ),
        ]
    )


def train_model(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    metrics_path: str = METRICS_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train, evaluate, save the model and metrics. Returns the metrics dict."""
    start = time.time()

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["target"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    report = classification_report(
        y_test, preds, target_names=["ham", "spam"], output_dict=True
    )

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1_spam": round(f1_score(y_test, preds), 4),
        "precision_spam": round(report["spam"]["precision"], 4),
        "recall_spam": round(report["spam"]["recall"], 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_seconds": round(time.time() - start, 2),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save model + metrics
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(pipeline, model_path)

    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def predict(text: str, model_path: str = MODEL_PATH) -> dict:
    """Load the saved model and classify one message (useful for quick tests)."""
    pipeline = joblib.load(model_path)
    proba = pipeline.predict_proba([text])[0]
    label = "spam" if proba[1] >= 0.5 else "ham"
    return {"label": label, "spam_probability": round(float(proba[1]), 4)}


if __name__ == "__main__":
    results = train_model()
    print("Training complete. Metrics:")
    print(json.dumps(results, indent=2))

    print("\nQuick sanity check:")
    print(predict("WINNER!! You have won a free prize. Call now to claim."))
    print(predict("Hey, are we still meeting for lunch tomorrow?"))