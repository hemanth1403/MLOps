#!/bin/bash

# Quick Start Script for ML Monitoring Lab
# This script sets up and runs the complete monitoring stack

set -e  # Exit on error

echo "ML Monitoring Lab - Quick Start"

echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED} Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

echo -e "${GREEN} Docker is running${NC}"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED} Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN} Docker Compose is available${NC}"
echo ""

# Step 1: Start services

echo "Step 1: Starting services "

docker-compose up -d

echo ""
echo "Waiting for services to be healthy "
sleep 30

# Check service health
echo ""
echo "Checking service health "

# Check MLflow
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo -e "${GREEN} MLflow is healthy${NC}"
else
    echo -e "${YELLOW}  MLflow may still be starting${NC}"
fi

# Check Prometheus
if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo -e "${GREEN} Prometheus is healthy${NC}"
else
    echo -e "${YELLOW}  Prometheus may still be starting...${NC}"
fi

# Check Grafana
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN} Grafana is healthy${NC}"
else
    echo -e "${YELLOW}  Grafana may still be starting...${NC}"
fi

# Step 2: Install Python dependencies
echo ""

echo "Step 2: Installing dependencies "


if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo -e "${GREEN} Dependencies installed${NC}"
else
    echo -e "${YELLOW}  requirements.txt not found${NC}"
fi

# Step 3: Train models
echo ""

echo "Step 3: Training models "


echo "Training Logistic Regression "
python src/train.py --model-type logistic_regression --experiment-name quickstart

echo ""
echo "Training Random Forest "
python src/train.py --model-type random_forest --experiment-name quickstart --n-estimators 100

echo ""
echo "Training XGBoost "
python src/train.py --model-type xgboost --experiment-name quickstart --n-estimators 100

echo -e "${GREEN} Models trained successfully${NC}"

# Step 4: Test drift detection
echo ""

echo "Step 4: Testing drift detection "


python -c "
from src.drift_detector import DriftDetector, simulate_drift
from sklearn.datasets import load_diabetes
import numpy as np

X = load_diabetes().data
X_ref = X[:200]
X_curr = X[200:400]

detector = DriftDetector()

print('Testing no drift scenario.')
results = detector.detect_multivariate_drift(X_ref, X_curr)
print(f'Drift detected: {results[\"overall_drift\"]}')

print('\nTesting drift scenario.')
X_drifted = simulate_drift(X_curr, 'mean_shift', severity=0.8)
results = detector.detect_multivariate_drift(X_ref, X_drifted)
print(f'Drift detected: {results[\"overall_drift\"]}')
"

echo -e "${GREEN} Drift detection tested${NC}"

# Step 5: Start serving API
echo ""

echo "Step 5: API is running in Docker"


# Wait a bit more for API to be ready
sleep 10

# Test API
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN} API is healthy and ready${NC}"
    
    # Make a test prediction
    echo ""
    echo "Making test prediction..."
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"features": [0.038, 0.05, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]}' \
         | python -m json.tool
    
    echo ""
    echo -e "${GREEN} Test prediction successful${NC}"
else
    echo -e "${YELLOW}  API may need more time to start${NC}"
fi

# Final summary
echo ""

echo " Setup Complete!"

echo ""
echo "Access your monitoring stack:"
echo ""
echo "   MLflow UI:       http://localhost:5000"
echo "   Prometheus:      http://localhost:9090"
echo "   Grafana:         http://localhost:3000 (admin/admin)"
echo "   API Docs:        http://localhost:8000/docs"
echo "   API Health:      http://localhost:8000/health"
echo "   API Metrics:     http://localhost:8000/metrics"
echo ""
echo "Example commands:"
echo ""
echo "  # Train a new model"
echo "  python src/train.py --model-type random_forest --experiment-name my-experiment"
echo ""
echo "  # Make a prediction"
echo "  curl -X POST http://localhost:8000/predict \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"features\": [0.038, 0.05, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]}'"
echo ""
echo "  # Test drift detection"
echo "  python src/drift_detector.py"
echo ""
echo "  # Stop all services"
echo "  docker-compose down"
echo ""
echo "  # Stop and remove all data"
echo "  docker-compose down -v"
echo ""

