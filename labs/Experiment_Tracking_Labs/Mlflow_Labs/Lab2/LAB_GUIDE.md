# Lab 2: ML Monitoring & Experiment Tracking - Complete Guide

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Detailed Walkthrough](#detailed-walkthrough)
6. [Monitoring Scenarios](#monitoring-scenarios)
7. [Key Concepts](#key-concepts)
8. [Troubleshooting](#troubleshooting)
9. [Assignment Questions](#assignment-questions)

## Overview

This lab implements a production-grade ML monitoring system that combines:

- **MLflow** for experiment tracking and model registry
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **FastAPI** for model serving
- **Data Drift Detection** for data quality monitoring

### Learning Objectives

By completing this lab, we will:

- Understand how to track ML experiments systematically
- Learn to expose and collect custom metrics
- Build real-time monitoring dashboards
- Detect and respond to data drift
- Implement production-ready ML pipelines
- Integrate monitoring into CI/CD workflows

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │         Training Pipeline            │
                    │         (src/train.py)               │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │         MLflow Server                │
                    │  • Experiment Tracking               │
                    │  • Model Registry                    │
                    │  • Artifact Storage                  │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │      Model Serving (FastAPI)         │
                    │  • REST API                          │
                    │  • Health Checks                     │
                    │  • Prometheus Metrics Export         │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │      Prometheus Server               │
                    │  • Metrics Collection                │
                    │  • Alerting Rules                    │
                    │  • Time-Series Storage               │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │         Grafana                      │
                    │  • Dashboards                        │
                    │  • Visualization                     │
                    │  • Alert Management                  │
                    └──────────────────────────────────────┘
```

## Installation

### Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Git
- 8GB RAM (recommended)
- 10GB disk space

### Setup Steps

1. **Navigate to the lab directory:**

```bash
cd labs/Experiment_tracking_labs/Mlflow_labs/Lab2
```

2. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

3. **Start all services:**

```bash
docker-compose up -d
```

4. **Verify services are running:**

```bash
docker-compose ps
```

All services should show "Up" status.

## Quick Start

Use the provided quick start script:

```bash
./start.sh
```

This will:

1. Start all Docker services
2. Install dependencies
3. Train multiple models
4. Test drift detection
5. Verify API is working

**Access the UIs:**

- MLflow: http://localhost:5000
- Grafana: http://localhost:3000 (login: admin/admin)
- Prometheus: http://localhost:9090
- API Docs: http://localhost:8000/docs

## Detailed Walkthrough

### Part 1: Training with MLflow

#### Train Your First Model

```bash
python src/train.py \
    --model-type random_forest \
    --experiment-name my-first-experiment \
    --n-estimators 100 \
    --max-depth 10
```

**What happens:**

1. Data is loaded and preprocessed
2. Model is trained with specified hyperparameters
3. Metrics are logged to MLflow
4. Model artifacts are saved
5. Visualizations are generated

#### View Results in MLflow

1. Open http://localhost:5000
2. Click on "my-first-experiment"
3. See all runs with their metrics
4. Compare different runs
5. View artifacts (confusion matrix, ROC curve, etc.)

#### Train Multiple Models

```bash
# Logistic Regression
python src/train.py --model-type logistic_regression

# Random Forest
python src/train.py --model-type random_forest --n-estimators 200

# XGBoost
python src/train.py --model-type xgboost --learning-rate 0.05

# Gradient Boosting
python src/train.py --model-type gradient_boosting
```

#### Compare Models

In MLflow UI:

1. Select multiple runs (checkbox)
2. Click "Compare"
3. See side-by-side comparison
4. Identify best performing model

### Part 2: Model Serving with Monitoring

#### Start the Serving API

The API is automatically started by Docker Compose. You can verify:

```bash
curl http://localhost:8000/health
```

#### Make Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [0.038, 0.05, 0.061, 0.021, -0.044,
                    -0.034, -0.043, -0.002, 0.019, -0.017]
     }'
```

Response:

```json
{
  "prediction": 1,
  "probability": 0.73,
  "confidence": 0.73,
  "model_version": "latest",
  "timestamp": "2026-03-11T12:00:00",
  "latency_ms": 12.5
}
```

#### View Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

You'll see metrics like:

- `model_requests_total` - Total prediction requests
- `model_prediction_latency_seconds` - Prediction latency
- `prediction_confidence` - Confidence scores
- `active_requests` - Current active requests

### Part 3: Data Drift Detection

#### Test Drift Detection

```bash
python src/drift_detector.py
```

This runs three tests:

1. **No drift** - Normal data vs normal data
2. **Mean shift drift** - Shifted distribution
3. **Scale change drift** - Changed variance

#### Programmatic Drift Detection

```python
from src.drift_detector import DriftDetector
from sklearn.datasets import load_diabetes
import numpy as np

# Load data
X = load_diabetes().data
X_reference = X[:200]
X_current = X[200:400]

# Detect drift
detector = DriftDetector()
results = detector.detect_multivariate_drift(
    X_reference,
    X_current,
    feature_names=['age', 'sex', 'bmi', ...]
)

print(f"Drift detected: {results['overall_drift']}")
print(f"Drift score: {results['drift_score']:.3f}")
print(f"Features with drift: {results['drifted_features']}")
```

### Part 4: Monitoring Dashboards

#### Prometheus Queries

Open http://localhost:9090 and try these queries:

1. **Request rate:**

```promql
rate(model_requests_total[5m])
```

2. **95th percentile latency:**

```promql
histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))
```

3. **Error rate:**

```promql
rate(model_errors_total[5m])
```

4. **Active requests:**

```promql
active_requests
```

#### Grafana Dashboards

1. Open http://localhost:3000
2. Login: admin/admin
3. Go to Dashboards
4. Create a new dashboard or import the provided templates

**Key Panels to Add:**

- Request rate (graph)
- Latency percentiles (graph)
- Error rate (graph)
- Prediction confidence distribution (histogram)
- Feature distributions (heatmap)

## Monitoring Scenarios

### Scenario 1: Normal Operation

Simulate normal traffic:

```bash
python src/simulate.py --scenario normal --duration 60
```

**What to observe:**

- Steady request rate in Prometheus
- Consistent latency
- No errors
- Stable confidence scores

### Scenario 2: Performance Degradation

Simulate model performance issues:

```bash
python src/simulate.py --scenario degradation --duration 30
```

**What to observe:**

- Lower confidence scores
- Increased prediction variance
- Potential accuracy drop
- Check Grafana for alerts

### Scenario 3: Data Drift

Simulate distribution shift:

```bash
python src/simulate.py --scenario drift --duration 30
```

**What to observe:**

- Feature distributions change
- PSI scores increase
- Drift detection alerts fire
- May need model retraining

### Scenario 4: High Load

Test system under load:

```bash
python src/simulate.py --scenario load --duration 20
```

**What to observe:**

- Increased latency
- Higher CPU/memory usage
- Active requests spike
- System stability

### Scenario 5: Complete Suite

Run all scenarios:

```bash
python src/simulate.py --scenario all
```

Keep Grafana open during this to see all metrics in action!

## Key Concepts

### MLflow Concepts

**Experiment**: A collection of related runs
**Run**: A single execution of your training code
**Metrics**: Quantitative measures (accuracy, loss, etc.)
**Parameters**: Input configuration (hyperparameters)
**Artifacts**: Output files (models, plots, etc.)

### Prometheus Concepts

**Metric Types:**

- **Counter**: Only increases (requests, errors)
- **Gauge**: Can go up/down (temperature, active requests)
- **Histogram**: Distribution of values (latency, confidence)
- **Summary**: Similar to histogram with quantiles

### Drift Detection Methods

**Kolmogorov-Smirnov Test**: Tests if two distributions are different
**Population Stability Index (PSI)**: Measures distribution shift
**Wasserstein Distance**: Earth mover's distance between distributions

## Troubleshooting

### Services Won't Start

```bash
# Check Docker
docker info

# View service logs
docker-compose logs mlflow
docker-compose logs prometheus
docker-compose logs grafana

# Restart services
docker-compose restart
```

### MLflow Can't Connect to Database

```bash
# Check PostgreSQL
docker-compose logs postgres

# Restart database
docker-compose restart postgres

# Wait and restart MLflow
sleep 10
docker-compose restart mlflow
```

### API Returns 503 (Model Not Loaded)

```bash
# Check if MLflow is running
curl http://localhost:5000/health

# Check API logs
docker-compose logs ml_api

# Manually reload model
curl -X POST "http://localhost:8000/model/reload?model_name=diabetes-classification_random_forest"
```

### Prometheus Not Scraping Metrics

1. Check Prometheus targets: http://localhost:9090/targets
2. Verify API metrics endpoint: http://localhost:8000/metrics
3. Check prometheus.yml configuration
4. Restart Prometheus: `docker-compose restart prometheus`

### Clean Slate (Reset Everything)

```bash
# Stop and remove all containers, volumes, and data
docker-compose down -v

# Remove all local data
rm -rf mlruns/ mlflow.db prometheus_data/ grafana_data/

# Start fresh
docker-compose up -d
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---
