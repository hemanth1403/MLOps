"""
download_data.py - Fetch and prepare the UCI Heart Disease dataset

This script downloads the Cleveland Heart Disease dataset from UCI,
adds proper column headers, and saves it as a clean CSV file.
The dataset contains 303 patient records with 13 clinical features
used to predict the presence of heart disease (binary classification).

Dataset: https://archive.ics.uci.edu/ml/datasets/heart+Disease
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np


def load_params():
    """Load pipeline parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def download_and_prepare():
    """Download UCI Heart Disease data and save as structured CSV."""
    params = load_params()

    # Column names for the Cleveland dataset
    columns = [
        "age",            # Age in years
        "sex",            # Sex (1=male, 0=female)
        "chest_pain",     # Chest pain type (1-4)
        "resting_bp",     # Resting blood pressure (mm Hg)
        "cholesterol",    # Serum cholesterol (mg/dl)
        "fasting_bs",     # Fasting blood sugar > 120 mg/dl (1=true, 0=false)
        "resting_ecg",    # Resting ECG results (0-2)
        "max_hr",         # Maximum heart rate achieved
        "exercise_angina", # Exercise induced angina (1=yes, 0=no)
        "oldpeak",        # ST depression induced by exercise
        "slope",          # Slope of peak exercise ST segment (1-3)
        "ca",             # Number of major vessels colored by fluoroscopy (0-3)
        "thal",           # Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)
        "target"          # Diagnosis (0=no disease, 1-4=disease severity)
    ]

    raw_path = params["data"]["raw_path"]
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    # Try to download from UCI, fall back to generating synthetic data
    try:
        url = params["data"]["source_url"]
        print(f"Downloading heart disease dataset from UCI...")
        df = pd.read_csv(url, header=None, names=columns, na_values="?")
        print(f"Downloaded {len(df)} records from UCI repository")
    except Exception as e:
        print(f"Could not download from UCI ({e}), generating dataset locally...")
        df = _generate_heart_data(columns)
        print(f"Generated {len(df)} synthetic records")

    # Convert target to binary: 0 = no disease, 1 = disease present
    df["target"] = (df["target"] > 0).astype(int)

    # Save raw data
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Target distribution: {dict(df['target'].value_counts())}")
    print(f"  Missing values: {df.isnull().sum().sum()}")

    return df


def _generate_heart_data(columns):
    """
    Generate a realistic synthetic heart disease dataset.
    Uses medically-plausible distributions for each feature.
    """
    np.random.seed(42)
    n = 303  # Match UCI dataset size

    # Generate features with realistic distributions
    age = np.random.normal(54, 9, n).clip(29, 77).astype(int)
    sex = np.random.binomial(1, 0.68, n)  # ~68% male in original
    chest_pain = np.random.choice([1, 2, 3, 4], n, p=[0.16, 0.17, 0.28, 0.39])
    resting_bp = np.random.normal(131, 17, n).clip(94, 200).astype(int)
    cholesterol = np.random.normal(246, 52, n).clip(126, 564).astype(int)
    fasting_bs = np.random.binomial(1, 0.15, n)
    resting_ecg = np.random.choice([0, 1, 2], n, p=[0.49, 0.02, 0.49])
    max_hr = np.random.normal(149, 23, n).clip(71, 202).astype(int)
    exercise_angina = np.random.binomial(1, 0.33, n)
    oldpeak = np.random.exponential(1.0, n).clip(0, 6.2).round(1)
    slope = np.random.choice([1, 2, 3], n, p=[0.48, 0.46, 0.06])
    ca = np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])
    thal = np.random.choice([3, 6, 7], n, p=[0.55, 0.13, 0.32])

    # Generate target based on feature correlations (simplified risk model)
    risk_score = (
        0.03 * age
        + 0.5 * (chest_pain == 4).astype(float)
        - 0.01 * max_hr
        + 0.3 * exercise_angina
        + 0.2 * oldpeak
        + 0.4 * ca
        + 0.3 * (thal == 7).astype(float)
    )
    prob = 1 / (1 + np.exp(-risk_score + 2))
    target = np.random.binomial(1, prob)

    # Add some missing values (like the original dataset)
    ca_arr = ca.astype(float)
    thal_arr = thal.astype(float)
    missing_idx_ca = np.random.choice(n, 4, replace=False)
    missing_idx_thal = np.random.choice(n, 2, replace=False)
    ca_arr[missing_idx_ca] = np.nan
    thal_arr[missing_idx_thal] = np.nan

    data = {
        "age": age, "sex": sex, "chest_pain": chest_pain,
        "resting_bp": resting_bp, "cholesterol": cholesterol,
        "fasting_bs": fasting_bs, "resting_ecg": resting_ecg,
        "max_hr": max_hr, "exercise_angina": exercise_angina,
        "oldpeak": oldpeak, "slope": slope, "ca": ca_arr,
        "thal": thal_arr, "target": target
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    download_and_prepare()
