"""
Unit tests for Wine Quality MLOps Pipeline
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_validation import validate_data, preprocess_data


class TestDataValidation:
    """Test data validation functions"""
    
    def test_validate_data_valid(self):
        """Test validation with valid data"""
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['quality'] = data.target
        
        is_valid, errors = validate_data(df)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_data_missing_values(self):
        """Test validation detects missing values"""
        df = pd.DataFrame({
            'feature1': [1, 2, None, 4],
            'feature2': [5, 6, 7, 8]
        })
        
        is_valid, errors = validate_data(df)
        assert is_valid == False
        assert 'missing_values' in errors
    
    def test_validate_data_insufficient_samples(self):
        """Test validation detects insufficient samples"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        is_valid, errors = validate_data(df)
        assert is_valid == False
        assert 'insufficient_samples' in errors
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['quality'] = data.target
        
        # Add duplicates
        df = pd.concat([df, df.head(5)], ignore_index=True)
        
        processed_df = preprocess_data(df)
        
        # Check duplicates removed
        assert len(processed_df) == len(df.drop_duplicates())
        
        # Check binary target created
        assert 'quality_binary' in processed_df.columns
        assert processed_df['quality_binary'].isin([0, 1]).all()


class TestModelTraining:
    """Test model training pipeline"""
    
    def test_data_loading(self):
        """Test data can be loaded"""
        from train_model import load_data
        
        X, y = load_data()
        assert X.shape[0] > 0
        assert y.shape[0] > 0
        assert X.shape[0] == y.shape[0]
    
    def test_model_training(self):
        """Test model can be trained"""
        from train_model import train_model
        
        # Create simple dataset
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        
        model = train_model(X_train, y_train)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')


class TestModelEvaluation:
    """Test model evaluation"""
    
    def test_metrics_calculation(self):
        """Test metrics are calculated correctly"""
        from evaluate_model import evaluate_model
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Create simple dataset
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train simple model
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X[:80], y[:80])
        
        # Evaluate
        metrics, y_pred, y_pred_proba = evaluate_model(model, X[80:], y[80:])
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Check metric values are valid
        for key, value in metrics.items():
            if key != 'timestamp':
                assert 0 <= value <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
