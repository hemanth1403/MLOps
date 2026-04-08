import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add lab root to path so `src.api` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api import app  # noqa: E402

# 13 features matching sklearn wine recognition dataset order
VALID_FEATURES = [
    13.20, 1.78, 2.14, 11.2, 100.0,
    2.65, 2.76, 0.26, 1.28, 4.38,
    1.05, 3.40, 1050.0,
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# Health

def test_health_status_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_model_loaded(client):
    assert client.get("/health").json()["model_loaded"] is True


# Model info 

def test_model_info_returns_200(client):
    assert client.get("/model-info").status_code == 200


def test_model_info_required_fields(client):
    data = client.get("/model-info").json()
    for field in ("model_type", "accuracy", "f1_score", "feature_names", "n_classes"):
        assert field in data, f"Missing field: {field}"


def test_model_info_accuracy_in_range(client):
    accuracy = client.get("/model-info").json()["accuracy"]
    assert 0.0 <= accuracy <= 1.0


# Predict 

def test_predict_returns_200(client):
    response = client.post("/predict", json={"features": VALID_FEATURES})
    assert response.status_code == 200


def test_predict_response_schema(client):
    data = client.post("/predict", json={"features": VALID_FEATURES}).json()
    for field in ("prediction", "class_name", "confidence", "probabilities", "latency_ms"):
        assert field in data, f"Missing field: {field}"


def test_predict_valid_class(client):
    data = client.post("/predict", json={"features": VALID_FEATURES}).json()
    assert data["prediction"] in [0, 1, 2]
    assert data["class_name"] in ["class_1", "class_2", "class_3"]


def test_predict_confidence_in_range(client):
    confidence = client.post("/predict", json={"features": VALID_FEATURES}).json()["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_predict_probabilities_sum_to_one(client):
    probs = client.post("/predict", json={"features": VALID_FEATURES}).json()["probabilities"]
    assert abs(sum(probs) - 1.0) < 1e-3


def test_predict_wrong_feature_count_returns_422(client):
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 422


def test_predict_empty_features_returns_422(client):
    response = client.post("/predict", json={"features": []})
    assert response.status_code == 422


def test_predict_missing_body_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
