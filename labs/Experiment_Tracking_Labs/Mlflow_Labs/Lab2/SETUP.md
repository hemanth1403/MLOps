# Lab 2 Setup Instructions

## Local Development Setup (macOS)

Due to Docker volume permission issues on macOS, this lab uses a hybrid approach:

### Services Running Locally:
- **MLflow** - Port 5001
- **API** - Port 8000

### Services Running in Docker:
- **Prometheus** - Port 9090
- **Grafana** - Port 3000
- **Alertmanager** - Port 9093

## Quick Start

1. Create conda environment:
```bash
conda create -n mlops_lab python=3.9 -y
conda activate mlops_lab
pip install -r requirements.txt
```

2. Start Docker services:
```bash
docker-compose up -d prometheus grafana alertmanager
```

3. Start MLflow locally:
```bash
mlflow server \
  --backend-store-uri "sqlite:///mlflow.db" \
  --default-artifact-root "./mlruns" \
  --host 127.0.0.1 \
  --port 5001 &
```

4. Start API locally:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
python -m uvicorn src.serve:app --host 0.0.0.0 --port 8000 &
```

5. Train models:
```bash
python src/train.py --model-type random_forest --experiment-name diabetes-classification
python src/train.py --model-type xgboost --experiment-name diabetes-classification
python src/train.py --model-type logistic_regression --experiment-name diabetes-classification
python src/train.py --model-type gradient_boosting --experiment-name diabetes-classification
```

6. Run simulations:
```bash
python src/simulate.py --scenario all
```

## For CI/CD (GitHub Actions)

The GitHub Actions workflow runs everything natively on Linux (no Docker volume issues).
All services work perfectly in the automated pipeline.
