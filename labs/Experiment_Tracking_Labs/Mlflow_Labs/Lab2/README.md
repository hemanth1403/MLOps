# Lab 2: Advanced ML Monitoring & Experiment Tracking

## Overview

This lab implements a **production-grade ML monitoring system** that combines MLflow for experiment tracking, Prometheus for metrics collection, Grafana for visualization, and FastAPI for model serving. The system includes automated training pipelines, real-time performance monitoring, data drift detection, and comprehensive observability.

## Learning Objectives

This lab covers:

- How to track ML experiments systematically with MLflow
- How to expose and collect custom Prometheus metrics
- How to build real-time monitoring dashboards with Grafana
- How to detect and respond to data drift
- How to implement production-ready ML monitoring pipelines
- How to integrate monitoring into CI/CD workflows

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Training       │────>│   MLflow     │<────│  Grafana    │
│  Pipeline       │     │   Server     │     │  Dashboard  │
└─────────────────┘     └──────────────┘     └─────────────┘
                              │                      │
                              v                      v
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  FastAPI        │────>│  Prometheus  │────>│  Alerting   │
│  Serving        │     │   Server     │     │  System     │
└─────────────────┘     └──────────────┘     └─────────────┘
```

## Prerequisites

### Required Software

- **Docker & Docker Compose** - For containerized services
- **Python 3.9+** - For ML training and serving
- **Conda** (recommended) - For clean environment management
- **Git** - For version control
- **8GB RAM** (minimum) - For running all services
- **10GB disk space** - For Docker images and data

### System Requirements

- **macOS, Linux, or Windows** with WSL2
- **Internet connection** - For downloading Docker images and Python packages

## Complete Setup Guide

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/hemanth1403/MLOps-IE7374.git
cd MLOps-IE7374/labs/Experiment_tracking_labs/Mlflow_labs/Lab2
```

### Step 2: Create Python Environment

We recommend using conda for a clean, isolated environment:

```bash
# Create new conda environment
conda create -n mlops_lab python=3.9 -y

# Activate the environment
conda activate mlops_lab

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
mlflow --version  # Should show: mlflow, version 2.9.2
python --version  # Should show: Python 3.9.x
```

### Step 3: Start Docker Services

We'll run Prometheus, Grafana, and Alertmanager in Docker:

```bash
# Start only the monitoring services (NOT MLflow or API)
docker-compose up -d prometheus grafana alertmanager

# Wait for services to start
sleep 20

# Verify services are running
docker-compose ps

# You should see:
# - prometheus_server    Up
# - grafana_server       Up
# - alertmanager         Up
```

**Why not run everything in Docker?**
On macOS, Docker volume permissions can cause issues with MLflow artifact storage. Running MLflow and the API locally avoids these issues while maintaining full functionality.

### Step 4: Start MLflow Server

Run MLflow locally for reliable artifact storage:

```bash
# Start MLflow in background
mlflow server \
  --backend-store-uri "sqlite:///mlflow.db" \
  --default-artifact-root "./mlruns" \
  --host 127.0.0.1 \
  --port 5001 > mlflow.log 2>&1 &

# Wait for it to start
sleep 10

# Verify MLflow is running
curl http://localhost:5001/
# Should return HTML response

# Check the log if needed
tail -20 mlflow.log
```

**Access MLflow UI:** http://localhost:5001

### Step 5: Train ML Models

Train multiple models to compare their performance:

```bash
# Train Random Forest
python src/train.py \
  --model-type random_forest \
  --experiment-name diabetes-classification \
  --n-estimators 100 \
  --max-depth 10

# Train XGBoost
python src/train.py \
  --model-type xgboost \
  --experiment-name diabetes-classification \
  --n-estimators 100 \
  --learning-rate 0.1

# Train Logistic Regression
python src/train.py \
  --model-type logistic_regression \
  --experiment-name diabetes-classification

# Train Gradient Boosting
python src/train.py \
  --model-type gradient_boosting \
  --experiment-name diabetes-classification \
  --n-estimators 100 \
  --learning-rate 0.1
```

**Each training run will:**

- Load and preprocess the diabetes dataset
- Train the specified model
- Log metrics, parameters, and artifacts to MLflow
- Generate visualizations (confusion matrix, ROC curve, feature importance)
- Take approximately 30-60 seconds

**View Results:** Go to http://localhost:5001 to see all experiments and compare models!

### Step 6: Start the Model Serving API

Run the FastAPI server to serve predictions with monitoring:

```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001

# Start the API in background
python -m uvicorn src.serve:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

# Wait for startup
sleep 15

# Verify API is running
curl http://localhost:8000/health
# Should return: {"status":"healthy","model_loaded":true,...}
```

**Access API Documentation:** http://localhost:8000/docs

### Step 7: Verify All Services

Check that everything is running:

```bash
# MLflow
curl http://localhost:5001/ | head -5

# Prometheus
curl http://localhost:9090/-/healthy

# Grafana
curl http://localhost:3000/api/health

# API
curl http://localhost:8000/health

# All should return successful responses
```

### Step 8: Access the Monitoring Stack

Open these URLs in your browser:

| Service        | URL                        | Credentials | Purpose                              |
| -------------- | -------------------------- | ----------- | ------------------------------------ |
| **MLflow UI**  | http://localhost:5001      | -           | View experiments, compare models     |
| **Grafana**    | http://localhost:3000      | admin/admin | Create dashboards, visualize metrics |
| **Prometheus** | http://localhost:9090      | -           | Query metrics, view targets          |
| **API Docs**   | http://localhost:8000/docs | -           | Interactive API documentation        |

## Testing the System

### Test 1: Make Predictions

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.038, 0.05, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]
  }'

# Expected response:
# {
#   "prediction": 1,
#   "probability": 0.73,
#   "confidence": 0.73,
#   "model_version": "best",
#   "timestamp": "2026-03-12T20:00:00",
#   "latency_ms": 5.2
# }
```

### Test 2: View Prometheus Metrics

```bash
# View all metrics
curl http://localhost:8000/metrics

# You'll see metrics like:
# - model_requests_total
# - model_prediction_latency_seconds
# - prediction_confidence
# - active_requests
```

### Test 3: Run Data Drift Detection

```bash
# Test drift detection algorithms
python src/drift_detector.py

# This runs three tests:
# 1. No drift - Normal data vs normal data
# 2. Mean shift drift - Simulated distribution shift
# 3. Scale change drift - Simulated variance change
```

### Test 4: Run Monitoring Simulations

This demonstrates all monitoring scenarios with real traffic:

```bash
# Run all scenarios (takes ~3 minutes)
python src/simulate.py --scenario all

# Or run individual scenarios:
python src/simulate.py --scenario normal --duration 60
python src/simulate.py --scenario degradation --duration 30
python src/simulate.py --scenario drift --duration 30
python src/simulate.py --scenario load --duration 20
python src/simulate.py --scenario errors --duration 15
```

**Keep Grafana and Prometheus open while running simulations to watch metrics in real-time!**

## Setting Up Grafana Dashboards

### Step 1: Login to Grafana

1. Go to http://localhost:3000
2. Login with: **admin** / **admin**
3. Skip password change or set a new password

### Step 2: Verify Prometheus Data Source

1. Click **☰** menu -> **Connections** -> **Data sources**
2. You should see **Prometheus** listed
3. Click on it and click **"Test"** to verify connection

### Step 3: Create Your Dashboard

1. Click **☰** menu -> **Dashboards** -> **New** -> **New Dashboard**
2. Click **"Add visualization"**
3. Select **"Prometheus"** data source
4. Click **"Code"** tab at the bottom to enter queries directly

### Step 4: Add Panels

**Panel 1: Request Rate**

- Query: `rate(model_requests_total[1m])`
- Title: "Request Rate (per second)"
- Visualization: Time series

**Panel 2: Prediction Latency (P95)**

- Query: `histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))`
- Title: "Prediction Latency (P95)"
- Visualization: Time series

**Panel 3: Total Requests**

- Query: `model_requests_total`
- Title: "Total Requests"
- Visualization: Stat (big number display)

**Panel 4: Average Confidence**

- Query: `avg(prediction_confidence)`
- Title: "Average Prediction Confidence"
- Visualization: Time series

**Panel 5: Active Requests**

- Query: `active_requests`
- Title: "Active Requests"
- Visualization: Time series

**Panel 6: Error Rate**

- Query: `rate(model_errors_total[1m])`
- Title: "Error Rate"
- Visualization: Time series

### Step 5: Save Your Dashboard

1. Click the **Save** icon (top right)
2. Name: "ML Model Monitoring"
3. Click **"Save"**

## Useful Prometheus Queries

```promql
# Request rate (requests per second)
rate(model_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))

# 50th percentile latency (median)
histogram_quantile(0.50, rate(model_prediction_latency_seconds_bucket[5m]))

# Error rate
rate(model_errors_total[5m])

# Success rate percentage
100 * rate(model_requests_total{status="success"}[5m]) / rate(model_requests_total[5m])

# Average prediction confidence
avg(prediction_confidence)

# Requests by status
sum by (status) (rate(model_requests_total[5m]))

# Total active requests
active_requests
```

## Stopping All Services

### Clean Shutdown

```bash
# 1. Stop MLflow server
pkill -f "mlflow server"

# 2. Stop API server
pkill -f "uvicorn src.serve:app"

# 3. Stop Docker services
docker-compose down

# 4. Verify everything is stopped
docker ps  # Should show nothing
ps aux | grep mlflow  # Should show only grep
ps aux | grep uvicorn  # Should show only grep
```

### Complete Cleanup (Remove All Data)

```bash
# Stop services
pkill -f "mlflow server"
pkill -f "uvicorn"
docker-compose down -v

# Remove generated data
rm -rf mlflow.db mlflow.log mlruns/ api.log

# Remove Docker volumes
docker volume prune -f

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
```

## Restart from Scratch

If you want to start fresh:

```bash
# 1. Complete cleanup (run commands above)

# 2. Activate environment
conda activate mlops_lab

# 3. Start Docker services
docker-compose up -d prometheus grafana alertmanager

# 4. Start MLflow
mlflow server --backend-store-uri "sqlite:///mlflow.db" \
  --default-artifact-root "./mlruns" \
  --host 127.0.0.1 --port 5001 > mlflow.log 2>&1 &

# 5. Start API
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
python -m uvicorn src.serve:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

# 6. Train models
python src/train.py --model-type random_forest --experiment-name diabetes-classification
```

## Troubleshooting

### Port Already in Use

```bash
# Find what's using the port
lsof -i :5001  # Or :8000, :9090, :3000

# Kill the process
kill -9 <PID>
```

### MLflow Connection Issues

```bash
# Check MLflow is running
curl http://localhost:5001/

# Check the log
tail -50 mlflow.log

# Restart MLflow
pkill -f "mlflow server"
mlflow server --backend-store-uri "sqlite:///mlflow.db" \
  --default-artifact-root "./mlruns" --host 127.0.0.1 --port 5001 &
```

### API Not Loading Model

```bash
# Ensure MLflow has trained models
curl http://localhost:5001/api/2.0/mlflow/experiments/list

# Restart API with correct environment
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
pkill -f "uvicorn"
python -m uvicorn src.serve:app --host 0.0.0.0 --port 8000 &
```

### Prometheus Not Scraping Metrics

```bash
# Check Prometheus targets
# Go to: http://localhost:9090/targets

# If ml_api is DOWN, the API might not be running locally
# Verify: curl http://localhost:8000/metrics
```

## Quick Reference Commands

```bash
#  SETUP
conda activate mlops_lab
docker-compose up -d prometheus grafana alertmanager
mlflow server --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root "./mlruns" --host 127.0.0.1 --port 5001 &
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
python -m uvicorn src.serve:app --host 0.0.0.0 --port 8000 &

# TRAINING
python src/train.py --model-type random_forest --experiment-name diabetes-classification
python src/train.py --model-type xgboost --experiment-name diabetes-classification

# TESTING
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"features": [0.038, 0.05, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]}'
python src/simulate.py --scenario all
python src/drift_detector.py

# CLEANUP
pkill -f "mlflow server"
pkill -f "uvicorn"
docker-compose down
```

## By completing this lab, we will have:

- Trained and compared 4+ ML models
- Tracked experiments with comprehensive metrics
- Built a production API serving predictions
- Set up real-time monitoring dashboards
- Implemented data drift detection
- Processed 1000+ monitored requests
- Created a complete MLOps monitoring pipeline

## GitHub Actions CI/CD

This lab includes automated testing via GitHub Actions. The workflow:

- Runs on Linux (no Docker volume issues)
- Trains models automatically
- Runs drift detection tests
- Performs integration testing
- Validates the complete pipeline

**Location:** `.github/workflows/monitoring_pipeline.yml`

## Credits

**Author:** Hemanth Sai Mada  
**Course:** IE7374 - MLOps  
**Institution:** Northeastern University

---

For detailed instructions, see:

- **LAB_GUIDE.md** - Step-by-step walkthrough
- **QUICK_REFERENCE.md** - Command cheat sheet
- **SETUP.md** - Platform-specific setup notes
