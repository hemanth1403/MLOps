import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def log(level: str, message: str, **kwargs):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "type": "training",
        "message": message,
        **kwargs,
    }
    with open(os.path.join(LOG_DIR, "training.log"), "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry))


def train():
    log("INFO", "Training started")

    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = list(wine.feature_names)

    log(
        "INFO",
        "Dataset loaded",
        dataset="wine_recognition",
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        n_classes=int(len(np.unique(y))),
        class_names=list(wine.target_names),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log(
        "INFO",
        "Data split complete",
        train_samples=int(X_train.shape[0]),
        test_samples=int(X_test.shape[0]),
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=2, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    log(
        "INFO",
        "Model trained",
        model_type="RandomForestClassifier",
        n_estimators=100,
        max_depth=10,
    )

    y_pred = model.predict(X_test_scaled)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))
    cm = confusion_matrix(y_test, y_pred).tolist()

    feature_importance = {
        name: float(imp)
        for name, imp in zip(feature_names, model.feature_importances_)
    }
    top_features = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    )

    log(
        "INFO",
        "Evaluation complete",
        accuracy=round(accuracy, 4),
        f1_score=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        confusion_matrix=cm,
        top_features=top_features,
    )

    with open(os.path.join(MODEL_DIR, "wine_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Baseline stores raw training samples for KS-test drift detection
    baseline = {
        "feature_names": feature_names,
        "train_samples": X_train.tolist(),
    }
    with open(os.path.join(MODEL_DIR, "baseline.pkl"), "wb") as f:
        pickle.dump(baseline, f)

    metrics = {
        "model_type": "RandomForestClassifier",
        "dataset": "wine_recognition",
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "n_classes": int(len(np.unique(y))),
        "feature_names": feature_names,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log(
        "INFO",
        "Training complete — artifacts saved",
        accuracy=round(accuracy, 4),
        f1_score=round(f1, 4),
    )


if __name__ == "__main__":
    train()
