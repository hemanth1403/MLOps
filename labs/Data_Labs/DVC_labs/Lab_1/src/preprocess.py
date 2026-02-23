"""
preprocess.py - Data preprocessing pipeline stage

Handles missing values, feature scaling, and train/test splitting.
All preprocessing decisions are driven by params.yaml so experiments
can be reproduced and compared through DVC.
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import joblib


def load_params():
    """Load pipeline parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def preprocess():
    """Run full preprocessing pipeline on raw heart disease data."""
    params = load_params()

    # Load raw data
    raw_path = params["data"]["raw_path"]
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # --- Handle missing values ---
    strategy = params["preprocessing"]["missing_strategy"]
    print(f"  Handling missing values with strategy: {strategy}")

    if strategy == "drop":
        mask = X.isnull().any(axis=1)
        X = X[~mask]
        y = y[~mask]
        print(f"  Dropped {mask.sum()} rows with missing values")
    else:
        imputer = SimpleImputer(strategy=strategy)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    missing_after = X.isnull().sum().sum()
    print(f"  Missing values after preprocessing: {missing_after}")

    # --- Train/test split (before scaling to prevent leakage) ---
    test_split = params["data"]["test_split"]
    seed = params["data"]["random_seed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set:  {len(X_test)} samples")

    # --- Feature scaling ---
    scaler_type = params["preprocessing"]["scaler_type"]
    print(f"  Applying {scaler_type} scaling...")

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # --- Optional polynomial features ---
    if params["preprocessing"]["poly_features"]:
        degree = params["preprocessing"]["poly_degree"]
        print(f"  Generating polynomial features (degree={degree})...")
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_scaled = pd.DataFrame(
            poly.fit_transform(X_train_scaled),
            index=X_train_scaled.index
        )
        X_test_scaled = pd.DataFrame(
            poly.transform(X_test_scaled),
            index=X_test_scaled.index
        )
        print(f"  Feature count after polynomial expansion: {X_train_scaled.shape[1]}")

    # --- Save processed data and artifacts ---
    out_dir = params["data"]["processed_path"]
    os.makedirs(out_dir, exist_ok=True)

    X_train_scaled.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test_scaled.to_csv(f"{out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)
    joblib.dump(scaler, f"{out_dir}/scaler.joblib")

    print(f"  Saved processed data to {out_dir}/")
    print("Preprocessing complete!")


if __name__ == "__main__":
    preprocess()
