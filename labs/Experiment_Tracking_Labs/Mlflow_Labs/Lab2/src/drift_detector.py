"""
Data Drift Detection Module

Implements multiple statistical tests for detecting distribution changes:
- Kolmogorov-Smirnov test
- Population Stability Index (PSI)
- Chi-square test
- Wasserstein distance
"""

import logging
from typing import Tuple, Dict, List
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detect data drift using multiple statistical methods
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize drift detector
        
        Args:
            significance_level: Threshold for statistical significance (default 0.05)
        """
        self.significance_level = significance_level
        self.drift_history = []
        
    def kolmogorov_smirnov_test(self, reference: np.ndarray, 
                                current: np.ndarray) -> Tuple[bool, float, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution similarity
        
        Args:
            reference: Reference distribution (training data)
            current: Current distribution (new data)
            
        Returns:
            (drift_detected, statistic, p_value)
        """
        statistic, p_value = ks_2samp(reference, current)
        drift_detected = p_value < self.significance_level
        
        return drift_detected, statistic, p_value
    
    def population_stability_index(self, reference: np.ndarray,
                                  current: np.ndarray,
                                  bins: int = 10) -> Tuple[bool, float]:
        """
        Calculate Population Stability Index (PSI)
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization
            
        Returns:
            (drift_detected, psi_value)
        """
        # Create bins based on reference distribution
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        # Bin both distributions
        ref_binned = np.digitize(reference, breakpoints)
        cur_binned = np.digitize(current, breakpoints)
        
        # Calculate proportions
        ref_counts = np.bincount(ref_binned, minlength=len(breakpoints) + 1)
        cur_counts = np.bincount(cur_binned, minlength=len(breakpoints) + 1)
        
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)
        
        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        cur_props = np.where(cur_props == 0, 0.0001, cur_props)
        
        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        # PSI >= 0.2 indicates significant drift
        drift_detected = psi >= 0.2
        
        return drift_detected, psi
    
    def wasserstein_distance_test(self, reference: np.ndarray,
                                  current: np.ndarray,
                                  threshold: float = 0.1) -> Tuple[bool, float]:
        """
        Calculate Wasserstein distance (Earth Mover's Distance)
        
        Args:
            reference: Reference distribution
            current: Current distribution
            threshold: Threshold for drift detection
            
        Returns:
            (drift_detected, distance)
        """
        distance = wasserstein_distance(reference, current)
        
        # Normalize by reference std
        normalized_distance = distance / (np.std(reference) + 1e-10)
        
        drift_detected = normalized_distance > threshold
        
        return drift_detected, distance
    
    def chi_square_test(self, reference: np.ndarray, current: np.ndarray,
                       bins: int = 10) -> Tuple[bool, float, float]:
        """
        Perform Chi-square test for categorical or discretized continuous data
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization
            
        Returns:
            (drift_detected, statistic, p_value)
        """
        # Create bins
        all_data = np.concatenate([reference, current])
        bin_edges = np.percentile(all_data, np.linspace(0, 100, bins + 1))
        
        # Bin both distributions
        ref_binned = np.digitize(reference, bin_edges)
        cur_binned = np.digitize(current, bin_edges)
        
        # Create contingency table
        ref_counts = np.bincount(ref_binned, minlength=bins + 2)[:bins + 1]
        cur_counts = np.bincount(cur_binned, minlength=bins + 2)[:bins + 1]
        
        contingency_table = np.array([ref_counts, cur_counts])
        
        # Perform chi-square test
        statistic, p_value, _, _ = chi2_contingency(contingency_table)
        
        drift_detected = p_value < self.significance_level
        
        return drift_detected, statistic, p_value
    
    def detect_multivariate_drift(self, reference: np.ndarray,
                                 current: np.ndarray,
                                 feature_names: List[str] = None) -> Dict:
        """
        Detect drift across multiple features
        
        Args:
            reference: Reference data (n_samples, n_features)
            current: Current data (n_samples, n_features)
            feature_names: Optional feature names
            
        Returns:
            Dictionary with drift results per feature
        """
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        if current.ndim == 1:
            current = current.reshape(-1, 1)
        
        n_features = reference.shape[1]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        results = {
            'overall_drift': False,
            'drift_score': 0.0,
            'features': {}
        }
        
        drift_count = 0
        total_psi = 0.0
        
        for i, feature_name in enumerate(feature_names):
            ref_feature = reference[:, i]
            cur_feature = current[:, i]
            
            # Run multiple tests
            ks_drift, ks_stat, ks_pval = self.kolmogorov_smirnov_test(
                ref_feature, cur_feature
            )
            psi_drift, psi_value = self.population_stability_index(
                ref_feature, cur_feature
            )
            ws_drift, ws_dist = self.wasserstein_distance_test(
                ref_feature, cur_feature
            )
            
            # Feature has drift if any test detects it
            feature_drift = ks_drift or psi_drift or ws_drift
            
            if feature_drift:
                drift_count += 1
            
            total_psi += psi_value
            
            results['features'][feature_name] = {
                'drift_detected': feature_drift,
                'ks_test': {
                    'drift': ks_drift,
                    'statistic': float(ks_stat),
                    'p_value': float(ks_pval)
                },
                'psi': {
                    'drift': psi_drift,
                    'value': float(psi_value)
                },
                'wasserstein': {
                    'drift': ws_drift,
                    'distance': float(ws_dist)
                }
            }
        
        # Overall drift if > 30% of features show drift
        results['overall_drift'] = (drift_count / n_features) > 0.3
        results['drift_score'] = total_psi / n_features
        results['drifted_features'] = drift_count
        results['total_features'] = n_features
        
        # Log results
        if results['overall_drift']:
            logger.warning(
                f"Data drift detected! "
                f"{drift_count}/{n_features} features showing drift "
                f"(Avg PSI: {results['drift_score']:.3f})"
            )
        else:
            logger.info(
                f"No significant drift detected "
                f"(Avg PSI: {results['drift_score']:.3f})"
            )
        
        return results
    
    def get_drift_summary(self, results: Dict) -> str:
        """
        Generate human-readable drift summary
        
        Args:
            results: Results from detect_multivariate_drift
            
        Returns:
            Formatted summary string
        """
        summary = []
        
        summary.append("DATA DRIFT DETECTION REPORT")
        summary.append(f"\nOverall Drift Detected: {results['overall_drift']}")
        summary.append(f"Drift Score (Avg PSI): {results['drift_score']:.4f}")
        summary.append(f"Features with Drift: {results['drifted_features']}/{results['total_features']}")
        summary.append("\nPer-Feature Analysis:")
        
        for feature_name, feature_results in results['features'].items():
            if feature_results['drift_detected']:
                summary.append(f"\n{feature_name}:")
                summary.append(f"  PSI: {feature_results['psi']['value']:.4f}")
                summary.append(f"  KS p-value: {feature_results['ks_test']['p_value']:.4f}")
                summary.append(f"  Wasserstein: {feature_results['wasserstein']['distance']:.4f}")
        
        summary.append("\n")
        
        return "\n".join(summary)


def simulate_drift(data: np.ndarray, drift_type: str = 'mean_shift',
                  severity: float = 0.5) -> np.ndarray:
    """
    Simulate different types of data drift
    
    Args:
        data: Original data
        drift_type: Type of drift ('mean_shift', 'scale_change', 'noise')
        severity: How severe the drift should be (0-1)
        
    Returns:
        Drifted data
    """
    drifted_data = data.copy()
    
    if drift_type == 'mean_shift':
        # Shift mean
        shift = severity * np.std(data, axis=0)
        drifted_data += shift
        
    elif drift_type == 'scale_change':
        # Change scale (variance)
        scale_factor = 1 + severity
        mean = np.mean(data, axis=0)
        drifted_data = mean + (data - mean) * scale_factor
        
    elif drift_type == 'noise':
        # Add noise
        noise = np.random.normal(0, severity * np.std(data, axis=0), data.shape)
        drifted_data += noise
        
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")
    
    return drifted_data


def main():
    """Example usage"""
    from sklearn.datasets import load_diabetes
    
    # Load data
    diabetes = load_diabetes()
    X = diabetes.data
    feature_names = diabetes.feature_names
    
    # Split into reference and current
    n_split = len(X) // 2
    X_reference = X[:n_split]
    X_current = X[n_split:]
    
    # Test 1: No drift
    print("\n")
    print("TEST 1: No Drift (Natural Data)")
  
    
    detector = DriftDetector()
    results_no_drift = detector.detect_multivariate_drift(
        X_reference, X_current, feature_names
    )
    print(detector.get_drift_summary(results_no_drift))
    
    # Test 2: Mean shift drift
    print("\n")
    print("TEST 2: Mean Shift Drift")

    
    X_drifted = simulate_drift(X_current, drift_type='mean_shift', severity=0.8)
    results_drift = detector.detect_multivariate_drift(
        X_reference, X_drifted, feature_names
    )
    print(detector.get_drift_summary(results_drift))
    
    # Test 3: Scale change
    print("\n")
    print("TEST 3: Scale Change Drift")

    
    X_scaled = simulate_drift(X_current, drift_type='scale_change', severity=0.6)
    results_scaled = detector.detect_multivariate_drift(
        X_reference, X_scaled, feature_names
    )
    print(detector.get_drift_summary(results_scaled))


if __name__ == "__main__":
    main()
