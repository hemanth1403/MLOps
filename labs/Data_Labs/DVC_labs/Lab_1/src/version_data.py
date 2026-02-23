"""
version_data.py - Demonstrate DVC data versioning

This script simulates real-world data changes by modifying the
heart disease dataset in controlled ways, showing how DVC tracks
each version and lets you switch between them.

Usage:
    python src/version_data.py --version v1   # Original data
    python src/version_data.py --version v2   # Add noise + new samples
    python src/version_data.py --version v3   # Feature engineering update
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yaml


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def version_v1():
    """V1: Original clean dataset (baseline)."""
    params = load_params()
    raw_path = params["data"]["raw_path"]

    if not os.path.exists(raw_path):
        print("ERROR: Run 'python src/download_data.py' first to get the base data.")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    print(f"V1 (Baseline): {len(df)} records, {df.shape[1]} columns")
    print(f"  Target distribution: {dict(df['target'].value_counts())}")
    return df


def version_v2():
    """V2: Simulates receiving new patient records from a partner clinic."""
    params = load_params()
    raw_path = params["data"]["raw_path"]
    df = pd.read_csv(raw_path)

    np.random.seed(123)
    n_new = 50

    # Generate new records that look like real patient data
    new_records = pd.DataFrame({
        "age": np.random.normal(58, 8, n_new).clip(35, 80).astype(int),
        "sex": np.random.binomial(1, 0.60, n_new),
        "chest_pain": np.random.choice([1, 2, 3, 4], n_new, p=[0.20, 0.15, 0.25, 0.40]),
        "resting_bp": np.random.normal(135, 15, n_new).clip(100, 190).astype(int),
        "cholesterol": np.random.normal(250, 45, n_new).clip(150, 500).astype(int),
        "fasting_bs": np.random.binomial(1, 0.18, n_new),
        "resting_ecg": np.random.choice([0, 1, 2], n_new, p=[0.50, 0.03, 0.47]),
        "max_hr": np.random.normal(145, 20, n_new).clip(80, 195).astype(int),
        "exercise_angina": np.random.binomial(1, 0.35, n_new),
        "oldpeak": np.random.exponential(1.1, n_new).clip(0, 5.5).round(1),
        "slope": np.random.choice([1, 2, 3], n_new, p=[0.45, 0.48, 0.07]),
        "ca": np.random.choice([0, 1, 2, 3], n_new, p=[0.55, 0.25, 0.13, 0.07]).astype(float),
        "thal": np.random.choice([3, 6, 7], n_new, p=[0.52, 0.15, 0.33]).astype(float),
        "target": np.random.binomial(1, 0.48, n_new)
    })

    df_v2 = pd.concat([df, new_records], ignore_index=True)
    df_v2.to_csv(raw_path, index=False)

    print(f"V2 (New Clinic Data): {len(df_v2)} records (+{n_new} new)")
    print(f"  Target distribution: {dict(df_v2['target'].value_counts())}")
    return df_v2


def version_v3():
    """V3: Feature engineering - add BMI-proxy and risk category columns."""
    params = load_params()
    raw_path = params["data"]["raw_path"]
    df = pd.read_csv(raw_path)

    # Add derived features
    # Age-HR interaction (cardiac reserve proxy)
    df["cardiac_reserve"] = (df["max_hr"] / df["age"]).round(3)

    # Cholesterol-to-BP ratio (cardiovascular stress indicator)
    df["chol_bp_ratio"] = (df["cholesterol"] / df["resting_bp"]).round(3)

    # Risk category based on key indicators
    df["risk_category"] = 0
    df.loc[df["chest_pain"] == 4, "risk_category"] += 1
    df.loc[df["exercise_angina"] == 1, "risk_category"] += 1
    df.loc[df["oldpeak"] > 2, "risk_category"] += 1
    df.loc[df["ca"] > 1, "risk_category"] += 1

    df.to_csv(raw_path, index=False)

    print(f"V3 (Feature Engineering): {len(df)} records, {df.shape[1]} columns")
    print(f"  New features: cardiac_reserve, chol_bp_ratio, risk_category")
    print(f"  Risk category distribution: {dict(df['risk_category'].value_counts().sort_index())}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Create different data versions")
    parser.add_argument(
        "--version", choices=["v1", "v2", "v3"], required=True,
        help="Which data version to create"
    )
    args = parser.parse_args()

    print(f"\n--- Creating data {args.version.upper()} ---")

    if args.version == "v1":
        version_v1()
    elif args.version == "v2":
        version_v2()
    elif args.version == "v3":
        version_v3()

    print(f"\nData version {args.version} is ready.")
    print("Next steps:")
    print("  1. dvc add data/raw/heart_disease.csv")
    print("  2. git add data/raw/heart_disease.csv.dvc")
    print("  3. git commit -m 'Data version {}'".format(args.version))
    print("  4. dvc push")


if __name__ == "__main__":
    main()
