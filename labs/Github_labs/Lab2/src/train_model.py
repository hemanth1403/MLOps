"""
Wine Quality Model Training Script
Trains a Gradient Boosting Classifier with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import json
import os
import urllib.request


def download_wine_dataset():
    """Download Wine Quality dataset from UCI"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    os.makedirs('data', exist_ok=True)
    output_file = 'data/winequality-red.csv'
    
    if os.path.exists(output_file):
        print(f"Dataset already exists at {output_file}")
        return output_file
    
    try:
        print(f"Downloading Wine Quality dataset ")
        urllib.request.urlretrieve(url, output_file)
        print(f"Dataset downloaded successfully!")
        return output_file
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def load_data() -> tuple:
    """
    Load wine quality data
    
    Returns:
        Tuple of (X, y)
    """
    # Try to download/load the UCI Wine Quality dataset
    dataset_file = download_wine_dataset()
    
    if dataset_file and os.path.exists(dataset_file):
        # Load UCI Wine Quality dataset
        df = pd.read_csv(dataset_file, sep=';')
        
        # Separate features and target
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        # Convert to binary: quality >= 6 is "good" (1), otherwise "bad" (0)
        y_binary = (y >= 6).astype(int)
        
        print(f"\nDataset loaded successfully!")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Class 0 (bad): {(y_binary == 0).sum()}")
        print(f"   Class 1 (good): {(y_binary == 1).sum()}")
        
        return X.values, y_binary.values
    
    else:
        # Fallback: Use sklearn wine dataset
        print("\nUsing sklearn wine dataset as fallback")
        from sklearn.datasets import load_wine
        data = load_wine()
        X = data.data
        
        # Create balanced binary classification
        # Split cultivars: class 0 vs (classes 1,2)
        y = (data.target == 0).astype(int)
        
        print(f"\nDataset loaded (sklearn wine):")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Class 0: {(y == 0).sum()}")
        print(f"   Class 1: {(y == 1).sum()}")
        
        return X, y


def perform_hyperparameter_tuning(X_train, y_train) -> dict:
    """
    Perform hyperparameter tuning using GridSearchCV
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Best parameters dictionary
    """
    print("\nStarting hyperparameter tuning ")
    
    # Reduced parameter grid for stability
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Create base model
    base_model = GradientBoostingClassifier(random_state=42)
    
    # Perform grid search with stratified k-fold
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


def train_model(X_train, y_train, params: dict = None) -> GradientBoostingClassifier:
    """
    Train Gradient Boosting model with given parameters
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model parameters (if None, uses defaults)
        
    Returns:
        Trained model
    """
    print("\nTraining Gradient Boosting Classifier...")
    
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    else:
        params['random_state'] = 42
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    
    return model


def save_model_and_metadata(model, scaler, best_params, feature_names=None):
    """
    Save trained model, scaler, and metadata
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        best_params: Best hyperparameters
        feature_names: List of feature names
    """
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"models/wine_quality_model_{timestamp}.pkl"
    joblib.dump(model, model_filename)
    print(f"\nModel saved: {model_filename}")
    
    # Save scaler
    scaler_filename = f"models/scaler_{timestamp}.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved: {scaler_filename}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'GradientBoostingClassifier',
        'hyperparameters': best_params,
        'model_filename': model_filename,
        'scaler_filename': scaler_filename,
        'feature_names': feature_names if feature_names else []
    }
    
    metadata_filename = f"models/model_metadata_{timestamp}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved: {metadata_filename}")
    
    return model_filename, timestamp


def main():
    """Main training pipeline"""
    print("Wine Quality Model Training Pipeline")
    
    # Load data
    X, y = load_data()
    
    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    if len(unique) < 2:
        print("\nERROR: Dataset has only one class!")
        print("   This usually means the data wasn't loaded correctly.")
        print("   Please check your internet connection and try again.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nData split:")
    print(f"   Train size: {X_train.shape[0]}")
    print(f"   Test size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Perform hyperparameter tuning
    best_params = perform_hyperparameter_tuning(X_train_scaled, y_train)
    
    # Train model with best parameters
    model = train_model(X_train_scaled, y_train, best_params)
    
    # Save model and metadata
    model_filename, timestamp = save_model_and_metadata(
        model, scaler, best_params
    )
    
    # Save test data for evaluation
    test_data = {
        'X_test': X_test_scaled.tolist(),
        'y_test': y_test.tolist()
    }
    test_data_filename = f"models/test_data_{timestamp}.json"
    with open(test_data_filename, 'w') as f:
        json.dump(test_data, f)
    print(f"Test data saved: {test_data_filename}")
    

    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()