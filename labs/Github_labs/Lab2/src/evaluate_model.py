"""
Wine Quality Model Evaluation Script
Evaluates model performance with multiple metrics and visualizations
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for CI/CD
import matplotlib.pyplot as plt
import seaborn as sns


def load_latest_model():
    """
    Load the most recent model and test data
    
    Returns:
        Tuple of (model, X_test, y_test, timestamp)
    """
    models_dir = 'models'
    
    # Find latest model file
    model_files = [f for f in os.listdir(models_dir) if f.startswith('wine_quality_model_') and f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError("No model files found!")
    
    latest_model_file = sorted(model_files)[-1]
    timestamp = latest_model_file.replace('wine_quality_model_', '').replace('.pkl', '')
    
    print(f"Loading model: {latest_model_file}")
    model = joblib.load(os.path.join(models_dir, latest_model_file))
    
    # Load test data
    test_data_file = f"models/test_data_{timestamp}.json"
    with open(test_data_file, 'r') as f:
        test_data = json.load(f)
    
    X_test = np.array(test_data['X_test'])
    y_test = np.array(test_data['y_test'])
    
    print(f"Test data loaded: {X_test.shape[0]} samples")
    
    return model, X_test, y_test, timestamp


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate model with comprehensive metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Print results
    print("EVALUATION RESULTS")
    for metric, value in metrics.items():
        if metric != 'timestamp':
            print(f"{metric.upper():15s}: {value:.4f}")
    
    return metrics, y_pred, y_pred_proba


def generate_confusion_matrix(y_test, y_pred, timestamp):
    """
    Generate and save confusion matrix plot
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        timestamp: Model timestamp
    """
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - Wine Quality Prediction', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('metrics', exist_ok=True)
    plot_filename = f"metrics/confusion_matrix_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved: {plot_filename}")


def generate_feature_importance(model, timestamp):
    """
    Generate and save feature importance plot
    
    Args:
        model: Trained model
        timestamp: Model timestamp
    """
    print("\nGenerating feature importance plot...")
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        n_features = len(importance)
        
        # Try to get feature names
        try:
            # First try Wine Quality dataset feature names
            wine_quality_features = [
                'fixed acidity', 'volatile acidity', 'citric acid',
                'residual sugar', 'chlorides', 'free sulfur dioxide',
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
            ]
            
            if n_features == len(wine_quality_features):
                feature_names = wine_quality_features
            else:
                # Fall back to sklearn wine dataset names
                from sklearn.datasets import load_wine
                sklearn_names = load_wine().feature_names
                
                if n_features == len(sklearn_names):
                    feature_names = sklearn_names
                else:
                    # Generic names if nothing matches
                    feature_names = [f"Feature {i+1}" for i in range(n_features)]
        except:
            feature_names = [f"Feature {i+1}" for i in range(n_features)]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names[:n_features],  # Ensure same length
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x='importance', y='feature', palette='viridis')
        plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"metrics/feature_importance_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved: {plot_filename}")
        
        return importance_df.to_dict('records')
    else:
        print("Model does not have feature_importances_ attribute")
        return None


def save_evaluation_results(metrics, feature_importance, timestamp):
    """
    Save evaluation results to JSON file
    
    Args:
        metrics: Dictionary of metrics
        feature_importance: Feature importance data
        timestamp: Model timestamp
    """
    results = {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'model_timestamp': timestamp
    }
    
    results_filename = f"metrics/evaluation_results_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation results saved: {results_filename}")


def main():
    """Main evaluation pipeline"""
    print("Wine Quality Model Evaluation Pipeline")
    
    # Load model and test data
    model, X_test, y_test, timestamp = load_latest_model()
    
    # Evaluate model
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Generate visualizations
    generate_confusion_matrix(y_test, y_pred, timestamp)
    feature_importance = generate_feature_importance(model, timestamp)
    
    # Save results
    save_evaluation_results(metrics, feature_importance, timestamp)
    
    print("Evaluation pipeline completed successfully! ")


if __name__ == "__main__":
    main()