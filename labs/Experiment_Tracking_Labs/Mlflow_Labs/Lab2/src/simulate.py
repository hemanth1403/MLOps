"""
Simulation Script for ML Monitoring Demo

This script simulates various scenarios to demonstrate monitoring capabilities:
1. Normal operation with predictions
2. Performance degradation
3. Data drift
4. High load scenarios
"""

import time
import random
import argparse
from typing import List
import numpy as np
import requests
from sklearn.datasets import load_diabetes

from drift_detector import DriftDetector, simulate_drift

# API endpoint
API_URL = "http://localhost:8000"


def generate_sample_features() -> List[float]:
    """Generate realistic feature values"""
    # Load diabetes dataset for realistic ranges
    diabetes = load_diabetes()
    X = diabetes.data
    
    # Pick a random sample
    idx = random.randint(0, len(X) - 1)
    return X[idx].tolist()


def simulate_normal_operation(duration_seconds: int = 60, requests_per_second: int = 2):
    """
    Simulate normal API operation
    
    Args:
        duration_seconds: How long to run simulation
        requests_per_second: Request rate
    """
    print(f"\n Simulating normal operation for {duration_seconds}s...")
    print(f"   Request rate: {requests_per_second} req/s")
    
    start_time = time.time()
    total_requests = 0
    successful = 0
    failed = 0
    
    while time.time() - start_time < duration_seconds:
        try:
            features = generate_sample_features()
            
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=5
            )
            
            if response.status_code == 200:
                successful += 1
                result = response.json()
                if total_requests % 10 == 0:  # Print every 10th request
                    print(f"   Request {total_requests}: prediction={result['prediction']}, "
                          f"confidence={result['confidence']:.3f}, "
                          f"latency={result['latency_ms']:.2f}ms")
            else:
                failed += 1
                print(f"    Request failed: {response.status_code}")
            
            total_requests += 1
            
            # Sleep to maintain request rate
            time.sleep(1.0 / requests_per_second)
            
        except Exception as e:
            failed += 1
            print(f"    Error: {e}")
    
    duration = time.time() - start_time
    print(f"\n Results:")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Total requests: {total_requests}")
    print(f"   Successful: {successful} ({successful/total_requests*100:.1f}%)")
    print(f"   Failed: {failed}")
    print(f"   Actual rate: {total_requests/duration:.2f} req/s")


def simulate_performance_degradation(duration_seconds: int = 30):
    """
    Simulate performance degradation by sending degraded features
    """
    print(f"\n Simulating performance degradation for {duration_seconds}s...")
    
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        # Generate features with added noise (simulating data quality issues)
        features = generate_sample_features()
        
        # Add significant noise
        noisy_features = [f + random.gauss(0, 0.5) for f in features]
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": noisy_features},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Noisy prediction: {result['prediction']}, "
                      f"confidence={result['confidence']:.3f}")
            
        except Exception as e:
            print(f"    Error: {e}")
        
        time.sleep(1)
    
    print("     Check Prometheus/Grafana for confidence drop!")


def simulate_data_drift(duration_seconds: int = 30):
    """
    Simulate data drift by sending systematically shifted features
    """
    print(f"\n Simulating data drift for {duration_seconds}s...")
    
    # Load reference data
    diabetes = load_diabetes()
    X = diabetes.data
    
    # Create drifted version
    X_drifted = simulate_drift(X, drift_type='mean_shift', severity=1.0)
    
    start_time = time.time()
    request_count = 0
    
    while time.time() - start_time < duration_seconds:
        # Pick random drifted sample
        idx = random.randint(0, len(X_drifted) - 1)
        features = X_drifted[idx].tolist()
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                if request_count % 5 == 0:
                    print(f"   Drifted prediction: {result['prediction']}, "
                          f"confidence={result['confidence']:.3f}")
            
            request_count += 1
            
        except Exception as e:
            print(f"    Error: {e}")
        
        time.sleep(1)
    
    print("     Check feature distribution metrics in Prometheus!")


def simulate_high_load(duration_seconds: int = 20, concurrent_requests: int = 50):
    """
    Simulate high load scenario
    """
    import concurrent.futures
    
    print(f"\n⚡ Simulating high load for {duration_seconds}s...")
    print(f"   Concurrent requests: {concurrent_requests}")
    
    def make_request():
        features = generate_sample_features()
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    start_time = time.time()
    total_requests = 0
    successful = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        while time.time() - start_time < duration_seconds:
            # Submit batch of requests
            futures = [executor.submit(make_request) for _ in range(concurrent_requests)]
            
            # Wait for completion
            for future in concurrent.futures.as_completed(futures):
                total_requests += 1
                if future.result():
                    successful += 1
            
            if total_requests % 100 == 0:
                print(f"   Processed {total_requests} requests...")
            
            time.sleep(0.1)  # Brief pause between batches
    
    duration = time.time() - start_time
    print(f"\n Results:")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Total requests: {total_requests}")
    print(f"   Successful: {successful} ({successful/total_requests*100:.1f}%)")
    print(f"   Throughput: {total_requests/duration:.1f} req/s")
    print("     Check latency and active_requests metrics in Prometheus!")


def simulate_error_scenario(duration_seconds: int = 20):
    """
    Simulate errors by sending invalid requests
    """
    print(f"\nSimulating error scenario for {duration_seconds}s...")
    
    start_time = time.time()
    error_count = 0
    
    while time.time() - start_time < duration_seconds:
        # Send invalid requests
        invalid_payloads = [
            {"features": []},  # Empty features
            {"features": [1, 2, 3]},  # Wrong number of features
            {"features": "invalid"},  # Wrong type
            {},  # Missing features
        ]
        
        payload = random.choice(invalid_payloads)
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=5
            )
            
            if response.status_code != 200:
                error_count += 1
                print(f"   Expected error: {response.status_code}")
            
        except Exception as e:
            error_count += 1
            print(f"   Expected exception: {type(e).__name__}")
        
        time.sleep(1)
    
    print(f"   Generated {error_count} errors")
    print("   Check error_count metrics in Prometheus!")


def run_all_scenarios():
    """Run all simulation scenarios in sequence"""

    print("ML MONITORING SIMULATION SUITE")

    print("\nThis will demonstrate various monitoring scenarios.")
    print("Keep your Grafana dashboard open to observe metrics!")
    print("")
    
    input("Press Enter to start...")
    
    # Check API health
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            print("API is not healthy!")
            return
        print("API is healthy\n")
    except Exception as e:
        print(f"Cannot connect to API: {e}")
        return
    
    # Run scenarios
    scenarios = [
        ("Normal Operation", lambda: simulate_normal_operation(duration_seconds=30, requests_per_second=3)),
        ("Performance Degradation", lambda: simulate_performance_degradation(duration_seconds=20)),
        ("Data Drift", lambda: simulate_data_drift(duration_seconds=20)),
        ("High Load", lambda: simulate_high_load(duration_seconds=15, concurrent_requests=30)),
        ("Error Scenario", lambda: simulate_error_scenario(duration_seconds=15)),
        ("Recovery - Normal Operation", lambda: simulate_normal_operation(duration_seconds=30, requests_per_second=2)),
    ]
    
    for name, scenario_func in scenarios:
        print(f"\n")
        print(f"SCENARIO: {name}")

        
        scenario_func()
        
        print(f"\nScenario '{name}' completed")
        print("   Waiting 10s before next scenario...")
        time.sleep(10)
    
    print("\n")
    print("ALL SCENARIOS COMPLETED!")

    print("\nCheck your monitoring dashboards:")
    print("  • Grafana: http://localhost:3000")
    print("  • Prometheus: http://localhost:9090")
    print("  • MLflow: http://localhost:5000")


def main():
    parser = argparse.ArgumentParser(description='ML Monitoring Simulation')
    parser.add_argument('--scenario', type=str, 
                       choices=['normal', 'degradation', 'drift', 'load', 'errors', 'all'],
                       default='all',
                       help='Scenario to simulate')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration in seconds')
    
    args = parser.parse_args()
    
    scenarios = {
        'normal': lambda: simulate_normal_operation(args.duration),
        'degradation': lambda: simulate_performance_degradation(args.duration),
        'drift': lambda: simulate_data_drift(args.duration),
        'load': lambda: simulate_high_load(args.duration),
        'errors': lambda: simulate_error_scenario(args.duration),
        'all': run_all_scenarios
    }
    
    scenarios[args.scenario]()


if __name__ == "__main__":
    main()
