# Wine Cultivar Classifier — Dockerized ML Pipeline

A containerized machine learning pipeline that trains a PyTorch neural network to classify wine cultivars from physicochemical properties, then serves real-time predictions through a FastAPI web application.

## Project Overview

This project demonstrates Docker containerization for ML workflows with:
- **Multi-stage Docker builds** — training and serving in separate stages for smaller final images
- **Docker Compose orchestration** — separate training and serving containers sharing data via volumes
- **PyTorch neural network** with BatchNorm, Dropout, and early stopping
- **FastAPI REST API** with health check and metrics endpoints
- **Interactive web UI** for making predictions

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Docker Compose                                  │
│                                                  │
│  ┌──────────────┐     ┌──────────────────────┐  │
│  │ wine_trainer  │     │   wine_serving       │  │
│  │              │     │                      │  │
│  │ Train Model  │────▶│  FastAPI Server      │  │
│  │ Save Artifacts│     │  /predict endpoint   │  │
│  └──────────────┘     │  /health endpoint    │  │
│         │              │  /metrics endpoint   │  │
│         ▼              └──────────────────────┘  │
│  ┌──────────────┐              │                 │
│  │wine_artifacts│              │                 │
│  │   (volume)   │              │                 │
│  └──────────────┘         Port 8000              │
└───────────────────────────┬─────────────────────┘
                            │
                       localhost:8000
```

## What Makes This Different from the Reference Lab

| Aspect | Reference Lab 2 | This Implementation |
|--------|----------------|---------------------|
| Dataset | Iris (4 features, 3 classes) | Wine (13 features, 3 cultivars) |
| Framework | TensorFlow/Keras | PyTorch |
| Web Framework | Flask | FastAPI |
| API Features | Single /predict endpoint | /predict, /health, /metrics endpoints |
| Model Architecture | Simple 2-layer NN | 3-layer NN with BatchNorm + Dropout |
| Training | Fixed epochs | Early stopping + LR scheduling |
| Data Split | Train/Test only | Train/Validation/Test (70/15/15) |
| Predictions | Class label only | Class + confidence + all probabilities |
| Docker Practices | Basic multi-stage | Multi-stage + HEALTHCHECK + .dockerignore + env vars |
| Compose Features | Basic depends_on | depends_on + healthcheck + environment config |
| CI/CD | None | GitHub Actions: build, test endpoints, push to registry |

## Quick Start

### Option 1: Multi-Stage Dockerfile (single image)

```sh
# Build the image (trains model during build)
docker build -t wine-classifier:v1 .

# Run the container
docker run -p 8000:8000 wine-classifier:v1

# Save image as tar (for submission)
docker save wine-classifier:v1 > wine_classifier.tar
```

### Option 2: Docker Compose (separate services)

```sh
# Build and start both services
docker compose up

# Or run in detached mode
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# Stop services
docker compose down
```

Then open **http://localhost:8000** in your browser.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web prediction interface |
| `/predict` | POST | Make predictions (form data or JSON) |
| `/health` | GET | Health check for container orchestration |
| `/metrics` | GET | View training metrics |

### Example API Call

```sh
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [14.23, 1.71, 2.43, 15.6, 127, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]}'
```

## CI/CD Pipeline (GitHub Actions)

The project includes a GitHub Actions workflow (`.github/workflows/docker_ci.yml`) that runs automatically on every push to `main`:

**What it does:**
1. **Builds** the Docker image using the multi-stage Dockerfile
2. **Starts** the container and waits for the server to be ready
3. **Tests all API endpoints** — `/health`, `/metrics`, `/predict`, and `/` (home page)
4. **Validates prediction output** — checks that response contains `predicted_class`, `confidence`, and `probabilities`
5. **Saves the Docker image** as a GitHub Actions artifact for download
6. **Optionally pushes to Docker Hub** if you configure secrets (`DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`)

To enable Docker Hub push (optional):
1. Go to your repo → Settings → Secrets and variables → Actions
2. Add `DOCKERHUB_USERNAME` (your Docker Hub username)
3. Add `DOCKERHUB_TOKEN` (generate at https://hub.docker.com/settings/security)

## Project Structure

```
wine-quality-docker/
├── .github/
│   └── workflows/
│       └── docker_ci.yml   # GitHub Actions CI/CD pipeline
├── dockerfile              # Multi-stage build (train + serve)
├── docker-compose.yml      # Orchestrate training & serving containers
├── .dockerignore           # Exclude unnecessary files from build
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── src/
    ├── model_training.py   # PyTorch training pipeline
    ├── main.py             # FastAPI serving application
    ├── templates/
    │   └── predict.html    # Web UI for predictions
    └── statics/
        ├── wine_class0.jpg # Cultivar 1 image
        ├── wine_class1.jpg # Cultivar 2 image
        └── wine_class2.jpg # Cultivar 3 image
```

## Technologies Used

- **PyTorch** — Deep learning framework for model training
- **FastAPI** — Modern async web framework for serving
- **scikit-learn** — Data preprocessing and evaluation metrics
- **Docker** — Containerization with multi-stage builds
- **Docker Compose** — Multi-container orchestration with shared volumes
