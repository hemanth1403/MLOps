"""
Utility functions for ML monitoring lab
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import mlflow

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ml_monitoring.log')
        ]
    )


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Save metrics to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def compare_models(experiment_name: str, metric: str = 'test_accuracy') -> Dict:
    """
    Compare all models in an experiment
    
    Args:
        experiment_name: Name of MLflow experiment
        metric: Metric to compare models by
        
    Returns:
        Dictionary with comparison results
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )
    
    comparison = {
        'best_run_id': runs.iloc[0]['run_id'],
        'best_model': runs.iloc[0]['params.model_type'],
        'best_score': runs.iloc[0][f'metrics.{metric}'],
        'all_runs': []
    }
    
    for _, run in runs.iterrows():
        comparison['all_runs'].append({
            'run_id': run['run_id'],
            'model_type': run['params.model_type'],
            'score': run[f'metrics.{metric}']
        })
    
    return comparison


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics as a pretty table"""
    lines = ["Metrics"]
    for name, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"{name:30s}: {value:.4f}")
        else:
            lines.append(f"{name:30s}: {value}")

    return "\n".join(lines)


def calculate_model_drift_score(reference_metrics: Dict[str, float],
                                current_metrics: Dict[str, float]) -> float:
    """
    Calculate overall drift score between model versions
    
    Args:
        reference_metrics: Metrics from reference model
        current_metrics: Metrics from current model
        
    Returns:
        Drift score (0-1, higher means more drift)
    """
    common_metrics = set(reference_metrics.keys()) & set(current_metrics.keys())
    
    if not common_metrics:
        return 1.0
    
    drifts = []
    for metric in common_metrics:
        ref_val = reference_metrics[metric]
        cur_val = current_metrics[metric]
        
        # Normalized difference
        if ref_val != 0:
            drift = abs(cur_val - ref_val) / abs(ref_val)
        else:
            drift = abs(cur_val - ref_val)
        
        drifts.append(drift)
    
    return np.mean(drifts)


def get_latest_model_version(model_name: str) -> str:
    """Get latest version of a registered model"""
    mlflow.set_tracking_uri("http://localhost:5000")
    
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name)
        
        if not versions:
            return "No versions found"
        
        # Get the latest version number
        latest = max(versions, key=lambda x: int(x.version))
        return latest.version
    
    except Exception as e:
        logger.error(f"Error getting model version: {e}")
        return "Error"


def generate_experiment_report(experiment_name: str, output_file: str = None) -> str:
    """
    Generate a comprehensive experiment report
    
    Args:
        experiment_name: Name of the experiment
        output_file: Optional file to save the report
        
    Returns:
        Report as a string
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return f"Experiment '{experiment_name}' not found"
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    report_lines = [

        f"EXPERIMENT REPORT: {experiment_name}",

        f"Experiment ID: {experiment.experiment_id}",
        f"Total Runs: {len(runs)}",
        f"Created: {experiment.creation_time}",
        "",
        "TOP 5 MODELS (by test accuracy)",

    ]
    
    # Get top 5 runs
    top_runs = runs.nsmallest(5, 'metrics.test_accuracy', keep='all')
    
    for i, (_, run) in enumerate(top_runs.iterrows(), 1):
        report_lines.extend([
            f"\n{i}. {run['params.model_type']}",
            f"   Run ID: {run['run_id']}",
            f"   Accuracy: {run['metrics.test_accuracy']:.4f}",
            f"   F1-Score: {run['metrics.test_f1_score']:.4f}",
            f"   Training Time: {run['metrics.training_time_seconds']:.2f}s",
        ])
    
    report_lines.extend([
        "",
        ""
    ])
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_file}")
    
    return report


if __name__ == "__main__":
    # Example usage
    setup_logging()
    
    # Test comparing models
    try:
        comparison = compare_models('diabetes-classification')
        print(f"\nBest model: {comparison['best_model']}")
        print(f"Best score: {comparison['best_score']:.4f}")
    except Exception as e:
        print(f"Could not compare models: {e}")
