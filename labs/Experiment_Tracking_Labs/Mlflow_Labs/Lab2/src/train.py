"""
Advanced ML Training with MLflow Experiment Tracking

This script implements comprehensive experiment tracking including:
- Hyperparameter logging
- Metric tracking with multiple evaluation metrics
- Model artifact storage
- Feature importance visualization
- Confusion matrix logging
- ROC curve and PR curve artifacts
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, mean_squared_error, r2_score
)
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLflowExperimentTracker:
    """Handles MLflow experiment tracking and artifact logging"""
    
    # def __init__(self, experiment_name: str, tracking_uri: str = "http://127.0.0.1:5001"):
    #     """
    #     Initialize MLflow tracker
        
    #     Args:
    #         experiment_name: Name of the MLflow experiment
    #         tracking_uri: MLflow tracking server URI
    #     """
    #     self.experiment_name = experiment_name
    #     mlflow.set_tracking_uri(tracking_uri)
    #     mlflow.set_experiment(experiment_name)
    #     logger.info(f"MLflow experiment set: {experiment_name}")
    #     logger.info(f"Tracking URI: {tracking_uri}")

    def __init__(self, experiment_name: str, tracking_uri: str = None):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        
        # Use environment variable or default
        if tracking_uri is None:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set: {experiment_name}")
        logger.info(f"Tracking URI: {tracking_uri}")
    
    def log_dataset_info(self, X_train: np.ndarray, X_test: np.ndarray, 
                        y_train: np.ndarray, y_test: np.ndarray):
        """Log dataset information"""
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(np.unique(y_train)))
        mlflow.log_param("class_distribution", dict(zip(*np.unique(y_train, return_counts=True))))
    
    def log_model_params(self, model_name: str, params: Dict[str, Any]):
        """Log model hyperparameters"""
        mlflow.log_param("model_type", model_name)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log evaluation metrics"""
        for metric_name, metric_value in metrics.items():
            if step is not None:
                mlflow.log_metric(metric_name, metric_value, step=step)
            else:
                mlflow.log_metric(metric_name, metric_value)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: list = None):
        """Create and log confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save and log
        cm_path = "/tmp/confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(cm_path, "visualizations")
        plt.close()
        
        logger.info("Confusion matrix logged to MLflow")
    
    def log_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                     n_classes: int):
        """Create and log ROC curve"""
        plt.figure(figsize=(10, 8))
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            auc = roc_auc_score(y_true, y_proba[:, 1])
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        else:
            # Multi-class
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        roc_path = "/tmp/roc_curve.png"
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(roc_path, "visualizations")
        plt.close()
        
        logger.info("ROC curve logged to MLflow")
    
    def log_feature_importance(self, model: Any, feature_names: list):
        """Log feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importances')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), 
                      [feature_names[i] for i in indices], 
                      rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            
            fi_path = "/tmp/feature_importance.png"
            plt.savefig(fi_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(fi_path, "visualizations")
            plt.close()
            
            # Log as JSON
            importance_dict = {feature_names[i]: float(importances[i]) 
                             for i in range(len(importances))}
            with open("/tmp/feature_importance.json", "w") as f:
                json.dump(importance_dict, f, indent=2)
            mlflow.log_artifact("/tmp/feature_importance.json", "metrics")
            
            logger.info("Feature importance logged to MLflow")


def load_and_prepare_data(test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Load diabetes dataset and prepare for classification
    
    We'll convert the regression target to binary classification:
    - Above median = 1 (high diabetes progression)
    - Below median = 0 (low diabetes progression)
    """
    logger.info("Loading diabetes dataset...")
    
    # Load data
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    # Convert to binary classification
    y_binary = (y > np.median(y)).astype(int)
    
    # Get feature names
    feature_names = diabetes.feature_names
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y_binary, return_counts=True)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


def get_model(model_type: str, **kwargs) -> Any:
    """
    Get model instance based on type
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific hyperparameters
    """
    models = {
        'logistic_regression': LogisticRegression(
            max_iter=kwargs.get('max_iter', 1000),
            C=kwargs.get('C', 1.0),
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 2),
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            max_depth=kwargs.get('max_depth', 3),
            random_state=42
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            max_depth=kwargs.get('max_depth', 3),
            random_state=42,
            eval_metric='logloss'
        ),
        'svm': SVC(
            C=kwargs.get('C', 1.0),
            kernel=kwargs.get('kernel', 'rbf'),
            probability=True,
            random_state=42
        )
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return models[model_type]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
    }
    
    # Add AUC if probabilities available
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, 
                                                  multi_class='ovr', 
                                                  average='weighted')
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
    
    return metrics


def train_model(model_type: str, experiment_name: str, run_name: str = None,
               **model_params) -> Tuple[Any, Dict[str, float]]:
    """
    Train model with comprehensive MLflow tracking
    
    Args:
        model_type: Type of model to train
        experiment_name: MLflow experiment name
        run_name: Optional run name
        **model_params: Model hyperparameters
    
    Returns:
        Trained model and metrics dictionary
    """
    # Initialize tracker
    tracker = MLflowExperimentTracker(experiment_name)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        logger.info(f"Starting training: {model_type}")
        start_time = time.time()
        
        # Log dataset info
        tracker.log_dataset_info(X_train, X_test, y_train, y_test)
        
        # Get model
        model = get_model(model_type, **model_params)
        
        # Log model parameters
        tracker.log_model_params(model_type, model.get_params())
        
        # Train model
        logger.info("Training model...")
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        
        # Predictions
        logger.info("Making predictions...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        
        # Compute metrics
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
        
        # Log training metrics
        for metric_name, value in train_metrics.items():
            tracker.log_metrics({f"train_{metric_name}": value})
        
        # Log test metrics
        for metric_name, value in test_metrics.items():
            tracker.log_metrics({f"test_{metric_name}": value})
        
        # Log timing
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Log artifacts
        logger.info("Logging artifacts...")
        tracker.log_confusion_matrix(y_test, y_test_pred, labels=['Low', 'High'])
        tracker.log_roc_curve(y_test, y_test_proba, n_classes=2)
        
        # Log feature importance if available
        tracker.log_feature_importance(model, feature_names)
        
        # Save classification report
        report = classification_report(y_test, y_test_pred, 
                                      target_names=['Low', 'High'])
        with open("/tmp/classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("/tmp/classification_report.txt", "reports")
        
        # Log model
        logger.info("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"{experiment_name}_{model_type}"
        )
        
        # Log success
        mlflow.set_tag("status", "success")
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("trained_by", "automated_pipeline")
        
        logger.info(f" Training completed successfully!")
        logger.info(f"   Model: {model_type}")
        logger.info(f"   Test Accuracy: {test_metrics['accuracy']:.3f}")
        logger.info(f"   Test F1-Score: {test_metrics['f1_score']:.3f}")
        logger.info(f"   Training Time: {training_time:.2f}s")
        
        return model, test_metrics


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train ML model with MLflow tracking')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['logistic_regression', 'random_forest', 
                               'gradient_boosting', 'xgboost', 'svm'],
                       help='Type of model to train')
    parser.add_argument('--experiment-name', type=str, 
                       default='diabetes-classification',
                       help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default=None,
                       help='MLflow run name')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of estimators (for ensemble models)')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth (for tree-based models)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate (for boosting models)')
    parser.add_argument('--C', type=float, default=1.0,
                       help='Regularization parameter')
    
    args = parser.parse_args()
    
    # Prepare model parameters
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'C': args.C
    }
    
    try:
        # Train model
        model, metrics = train_model(
            model_type=args.model_type,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            **model_params
        )
        
        print("\n")
        print(" Training completed successfully!")

        print(f"\nView results at: http://localhost:5000")
        print(f"Experiment: {args.experiment_name}")
        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
