"""
train.py - Model training pipeline stage

Trains a classification model on the preprocessed heart disease data.
Model type and hyperparameters are configured via params.yaml, making
it easy to run experiments by just changing the config file.
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib


def load_params():
    """Load pipeline parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def get_model(params):
    """Instantiate the model specified in params.yaml."""
    model_type = params["train"]["model_type"]
    seed = params["train"]["random_seed"]

    if model_type == "random_forest":
        hp = params["train"]["random_forest"]
        model = RandomForestClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"],
            min_samples_leaf=hp["min_samples_leaf"],
            class_weight=hp["class_weight"],
            random_state=seed,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        hp = params["train"]["gradient_boosting"]
        model = GradientBoostingClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            subsample=hp["subsample"],
            random_state=seed
        )
    elif model_type == "logistic_regression":
        hp = params["train"]["logistic_regression"]
        model = LogisticRegression(
            C=hp["C"],
            max_iter=hp["max_iter"],
            solver=hp["solver"],
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Model: {model_type}")
    print(f"Hyperparameters: {hp}")
    return model


def train():
    """Load processed data, train the model, and save it."""
    params = load_params()
    processed_dir = params["data"]["processed_path"]

    # Load processed data
    print("Loading processed training data...")
    X_train = pd.read_csv(f"{processed_dir}/X_train.csv")
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").values.ravel()
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}")

    # Build and train model
    model = get_model(params)
    print("Training model...")
    model.fit(X_train, y_train)

    # Training accuracy as a sanity check
    train_acc = model.score(X_train, y_train)
    print(f"  Training accuracy: {train_acc:.4f}")

    # Feature importances (for tree-based models)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)
        print("\nTop 5 feature importances:")
        for feat, imp in importances.head(5).items():
            print(f"  {feat}: {imp:.4f}")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
