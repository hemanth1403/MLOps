"""
evaluate.py - Model evaluation pipeline stage

Computes classification metrics on the test set and saves them
as JSON files that DVC can track. Also generates a confusion matrix
plot and feature importance visualization.
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib

# Use non-interactive backend so it works headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_params():
    """Load pipeline parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate():
    """Evaluate the trained model and save metrics + plots."""
    params = load_params()
    processed_dir = params["data"]["processed_path"]

    # Load test data and model
    print("Loading test data and model...")
    X_test = pd.read_csv(f"{processed_dir}/X_test.csv")
    y_test = pd.read_csv(f"{processed_dir}/y_test.csv").values.ravel()
    model = joblib.load("models/model.joblib")

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    # --- Compute metrics ---
    metrics = {}
    metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
    metrics["precision"] = round(precision_score(y_test, y_pred, zero_division=0), 4)
    metrics["recall"] = round(recall_score(y_test, y_pred, zero_division=0), 4)
    metrics["f1_score"] = round(f1_score(y_test, y_pred, zero_division=0), 4)

    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)
    else:
        metrics["roc_auc"] = None

    # Additional context
    metrics["model_type"] = params["train"]["model_type"]
    metrics["test_samples"] = int(len(y_test))
    metrics["positive_rate"] = round(float(y_test.mean()), 4)

    print("\n=== Evaluation Results ===")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # --- Save metrics as JSON (tracked by DVC) ---
    os.makedirs("metrics", exist_ok=True)
    metrics_path = "metrics/eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # --- Generate confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {params['train']['model_type'].replace('_', ' ').title()}",
        fontsize=14
    )
    fig.tight_layout()
    cm_path = "metrics/confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    # --- Feature importance plot (for tree-based models) ---
    if hasattr(model, "feature_importances_"):
        # Load original feature names
        feature_names = pd.read_csv(f"{processed_dir}/X_train.csv").columns.tolist()

        importances = pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        importances.plot(kind="barh", ax=ax, color="#2196F3", edgecolor="white")
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(
            f"Feature Importances — {params['train']['model_type'].replace('_', ' ').title()}",
            fontsize=14
        )
        fig.tight_layout()
        fi_path = "metrics/feature_importance.png"
        fig.savefig(fi_path, dpi=150)
        plt.close(fig)
        print(f"Feature importance plot saved to {fi_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate()
