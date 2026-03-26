import sys
import os
import pytest
from fastapi.testclient import TestClient

# Path setup — allow importing from src/ and resolve model paths correctly
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

# Change working directory so relative "../model/" paths inside src/ resolve correctly
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from main import app  # noqa: E402 — import after path setup

client = TestClient(app)


# Sample payload — values taken from the first row of the Diabetes dataset
VALID_PAYLOAD = {
    "age": 0.038076,
    "sex": 0.050680,
    "bmi": 0.061696,
    "bp": 0.021872,
    "s1": -0.044223,
    "s2": -0.034821,
    "s3": -0.043401,
    "s4": -0.002592,
    "s5": 0.019907,
    "s6": -0.017646,
}


# Health check


def test_health_check():
    # GET / should return 200 and {"status": "healthy"}
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# Prediction endpoint

def test_predict_returns_200():
    # POST /predict with valid payload should return HTTP 200
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_prediction_key():
    # POST /predict response must contain a 'prediction' key
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "prediction" in response.json()


def test_predict_returns_float():
    # The predicted value must be a float (regression output, not a class label).
    response = client.post("/predict", json=VALID_PAYLOAD)
    prediction = response.json()["prediction"]
    assert isinstance(prediction, float)


def test_predict_value_in_reasonable_range():
    # Predicted disease progression should be between 25 and 350 (dataset range).
    response = client.post("/predict", json=VALID_PAYLOAD)
    prediction = response.json()["prediction"]
    assert 25.0 <= prediction <= 350.0


def test_predict_missing_field_returns_422():
    # POST /predict with a missing field should return HTTP 422 (validation error).
    incomplete_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "bmi"}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    # POST /predict with a non-numeric field should return HTTP 422.
    bad_payload = {**VALID_PAYLOAD, "age": "not_a_number"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_empty_body_returns_422():
    # POST /predict with an empty body should return HTTP 422.
    response = client.post("/predict", json={})
    assert response.status_code == 422


# Model info endpoint


def test_model_info_returns_200():
    # GET /model-info should return HTTP 200 when metrics.json exists
    response = client.get("/model-info")
    assert response.status_code == 200


def test_model_info_schema():
    # GET /model-info should return all required metadata fields.
    response = client.get("/model-info")
    data = response.json()
    assert "model_type" in data
    assert "dataset" in data
    assert "task" in data
    assert "metrics" in data


def test_model_info_task_is_regression():
    # The task field in /model-info must be 'regression'.
    response = client.get("/model-info")
    assert response.json()["task"] == "regression"


def test_model_info_metrics_contain_rmse_and_r2():
    # Metrics in /model-info must include 'rmse' and 'r2_score'.
    response = client.get("/model-info")
    metrics = response.json()["metrics"]
    assert "rmse" in metrics
    assert "r2_score" in metrics
