"""
Data Validation Script for Wine Quality Dataset
Validates data quality before model training
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import sys


def validate_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, str]]:
    """
    Validate the wine quality dataset
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = {}
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        errors['missing_values'] = f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}"
    
    # Check for negative values in features that should be positive
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            errors[f'negative_values_{col}'] = f"Column {col} contains negative values"
    
    # Check minimum number of samples
    min_samples = 100
    if len(df) < min_samples:
        errors['insufficient_samples'] = f"Dataset has only {len(df)} samples, minimum {min_samples} required"
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > len(df) * 0.1:  # More than 10% duplicates
        errors['excessive_duplicates'] = f"Found {duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.2f}%)"
    
    # Check data types
    expected_numeric_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 
                            'residual sugar', 'chlorides', 'free sulfur dioxide',
                            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    
    for col in expected_numeric_cols:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            errors[f'invalid_dtype_{col}'] = f"Column {col} should be numeric but is {df[col].dtype}"
    
    is_valid = len(errors) == 0
    return is_valid, errors


def load_and_validate_data(filepath: str = None) -> pd.DataFrame:
    """
    Load wine quality data from sklearn or file and validate it
    
    Args:
        filepath: Optional path to CSV file
        
    Returns:
        Validated dataframe
    """
    if filepath:
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
    else:
        print("Loading wine quality dataset from sklearn")
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['quality'] = data.target
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Validate data
    is_valid, errors = validate_data(df)
    
    if not is_valid:
        print("Data validation failed!")
        for error_type, error_msg in errors.items():
            print(f"  - {error_type}: {error_msg}")
        sys.exit(1)
    else:
        print("Data validation passed!")
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the wine quality data
    
    Args:
        df: Input dataframe
        
    Returns:
        Preprocessed dataframe
    """
    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates()
    if len(df) < original_len:
        print(f"Removed {original_len - len(df)} duplicate rows")
    
    # Create binary quality target (good wine: quality >= 6)
    if 'quality' in df.columns:
        df['quality_binary'] = (df['quality'] >= 6).astype(int)
        print(f"Created binary target: {df['quality_binary'].value_counts().to_dict()}")
    
    return df


if __name__ == "__main__":
    # Load and validate data
    df = load_and_validate_data()
    
    # Preprocess
    df = preprocess_data(df)
    
    # Save processed data
    output_path = "data/wine_quality_processed.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
