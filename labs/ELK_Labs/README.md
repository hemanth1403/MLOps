# ELK Lab — Wine Cultivar Classifier with FastAPI + ELK Stack

## What this lab does

Trains a **Random Forest classifier** on the sklearn Wine Recognition dataset (13 features, 3 classes), serves it via **FastAPI**, and ships all logs into an **ELK stack** for real-time observability. A background **KS-test drift detector** compares live inference data against the training baseline every 60 seconds.

## How it differs from the professor's implementation

| Aspect          | Professor                                        | This lab                                   |
| --------------- | ------------------------------------------------ | ------------------------------------------ |
| Dataset         | Iris (4 features)                                | Wine Recognition (13 features)             |
| Model           | Logistic Regression                              | Random Forest                              |
| Logging source  | Training loop only                               | Training + FastAPI requests + drift        |
| Drift detection | Artificial noise injection                       | Two-sample KS test on real inference data  |
| Log parsing     | Grok regex patterns                              | Native JSON codec (no parsing needed)      |
| Serving layer   | None                                             | FastAPI with Pydantic validation           |
| ES indices      | 2 (`logstashtraining`, `logstashdriftdetection`) | 3 (`elk-training`, `elk-api`, `elk-drift`) |

<!-- ## Architecture

```
  ┌─────────────────────────────────────────┐
  │  Docker Container: api                  │
  │                                         │
  │  train.py  ──►  model/                  │
  │  api.py    ──►  logs/api.log            │
  │  drift_detector.py ─► logs/drift.log   │
  └──────────────┬──────────────────────────┘
                 │ logs/ volume
  ┌──────────────▼──────────────────────────┐
  │  Logstash                               │
  │  training.log ──► elk-training-*        │
  │  api.log      ──► elk-api-*             │
  │  drift.log    ──► elk-drift-*           │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │  Elasticsearch :9200                    │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │  Kibana :5601   (Discover / Dashboards) │
  └─────────────────────────────────────────┘
  Metricbeat collects ES / Logstash / Kibana / Docker metrics
``` -->

## Local setup (without Docker)

```bash
cd labs/ELK_Labs
pip install -r requirements.txt

# Train model
python src/train.py

# Start API
uvicorn src.api:app --reload

# Start drift detector (separate terminal)
python src/drift_detector.py

# Run tests
pytest tests/test_api.py -v
```

## Docker setup (full stack)

```bash
cd labs/ELK_Labs
docker-compose up --build
```

| Service       | URL                   |
| ------------- | --------------------- |
| FastAPI       | http://localhost:8000 |
| Kibana        | http://localhost:5601 |
| Elasticsearch | http://localhost:9200 |

## API endpoints

| Method | Path          | Description                            |
| ------ | ------------- | -------------------------------------- |
| GET    | `/health`     | Liveness check                         |
| GET    | `/model-info` | Model type, accuracy, feature names    |
| POST   | `/predict`    | Predict wine cultivar from 13 features |

### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [13.20,1.78,2.14,11.2,100.0,2.65,2.76,0.26,1.28,4.38,1.05,3.40,1050.0]}'
```

```json
{
  "prediction": 0,
  "class_name": "class_1",
  "confidence": 0.97,
  "probabilities": [0.97, 0.02, 0.01],
  "latency_ms": 3.5
}
```

## Kibana — viewing logs

1. Open http://localhost:5601
2. Go to **Stack Management -> Index Patterns**
3. Create patterns for `elk-training-*`, `elk-api-*`, `elk-drift-*`
4. Go to **Discover** and select an index pattern to explore logs

Useful fields to visualise:

- `elk-api-*`: `confidence`, `latency_ms`, `class_name`, `prediction`
- `elk-drift-*`: `any_drift_detected`, `drifted_features`, `inference_samples`
- `elk-training-*`: `accuracy`, `f1_score`, `top_features`

## CI/CD pipeline

Three sequential jobs triggered on push to `Main` (path-filtered to `labs/ELK_Labs/`):

```
train  ->  test  ->  docker-build
```

- **train**: Trains the model, validates accuracy ≥ 90%, uploads model artifact
- **test**: Downloads artifact, runs 13 pytest tests via FastAPI TestClient
- **docker-build**: Builds image, starts container, polls `/health`, smoke-tests all endpoints

## Log format

All three log files are newline-delimited JSON — no grok parsing required in Logstash.

```json
// training.log
{"timestamp": "...", "level": "INFO", "type": "training", "accuracy": 0.9722, "f1_score": 0.9723}

// api.log
{"timestamp": "...", "level": "INFO", "type": "api_request", "prediction": 0, "confidence": 0.97, "latency_ms": 3.5}

// drift.log
{"timestamp": "...", "level": "WARNING", "type": "drift_detection", "any_drift_detected": true, "drifted_features": ["alcohol"]}
```
