import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

model = None
scaler = None
metrics = None

CLASS_NAMES = ["class_1", "class_2", "class_3"]


def load_model():
    global model, scaler, metrics
    with open(os.path.join(MODEL_DIR, "wine_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Wine Cultivar Classifier", version="1.0.0", lifespan=lifespan)


def _write_log(entry: dict):
    with open(os.path.join(LOG_DIR, "api.log"), "a") as f:
        f.write(json.dumps(entry) + "\n")


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float
    probabilities: List[float]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    if metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return metrics


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    expected = len(metrics["feature_names"])
    if len(request.features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, got {len(request.features)}",
        )

    start = time.time()

    X = np.array(request.features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0].tolist()
    confidence = round(float(max(proba)), 4)
    latency_ms = round((time.time() - start) * 1000, 2)

    _write_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "type": "api_request",
            "endpoint": "/predict",
            "features": {
                name: val
                for name, val in zip(metrics["feature_names"], request.features)
            },
            "prediction": pred,
            "class_name": CLASS_NAMES[pred],
            "confidence": confidence,
            "latency_ms": latency_ms,
            "status_code": 200,
        }
    )

    return PredictResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        confidence=confidence,
        probabilities=[round(p, 4) for p in proba],
        latency_ms=latency_ms,
    )
