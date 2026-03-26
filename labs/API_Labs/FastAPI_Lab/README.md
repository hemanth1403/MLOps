# FastAPI Lab — Diabetes Progression Predictor

## Overview

In this lab we expose an ML **regression** model as a REST API using [FastAPI](https://fastapi.tiangolo.com/) and [uvicorn](https://www.uvicorn.org/).

**Key difference from a classification API:**
The professor's lab uses a _Decision Tree Classifier_ on the Iris dataset — it predicts a discrete class label (`0`, `1`, or `2`).
This lab uses a _Random Forest Regressor_ on the Diabetes dataset — it predicts a **continuous float score** representing disease progression. This makes it a **regression** problem, not a classification one.

|                | This Lab                            |
| -------------- | ----------------------------------- |
| Dataset        | Diabetes (442 samples, 10 features) |
| Model          | `RandomForestRegressor`             |
| Task           | **Regression**                      |
| Output         | Continuous score `e.g. 150.32`      |
| Extra endpoint | `GET /model-info`                   |
| Tests          | 12 pytest tests                     |
| CI             | GitHub Actions                      |

---

## Project Structure

```
FastAPI_Lab/
├── .gitignore
├── README.md
├── requirements.txt
├── model/                    # generated at runtime by train.py
│   ├── diabetes_model.pkl
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── data.py               # load & split the Diabetes dataset
│   ├── train.py              # train RandomForestRegressor, save model + metrics
│   ├── predict.py            # load model and run inference
│   └── main.py               # FastAPI app (3 endpoints)
└── tests/
    └── test_api.py           # 12 pytest tests using TestClient
```

---

## Setup

**1. Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Lab

### Step 1 — Train the model

Move into `src/` and run the training script:

```bash
cd src
python train.py
```

This trains a `RandomForestRegressor` on the sklearn Diabetes dataset and saves:

- `../model/diabetes_model.pkl` — the serialised model
- `../model/metrics.json` — RMSE and R² on the test set

Expected output:

```
RMSE : 53.6803
R²   : 0.4561
```

### Step 2 — Start the API

```bash
uvicorn main:app --reload
```

The `--reload` flag restarts the server on code changes (development only).

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Step 3 — Test the endpoints

**Health check**

```bash
curl http://localhost:8000/
# {"status":"healthy"}
```

**Predict disease progression**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.038, "sex": 0.050, "bmi": 0.061, "bp": 0.021,
    "s1": -0.044, "s2": -0.034, "s3": -0.043,
    "s4": -0.002, "s5": 0.019,  "s6": -0.017
  }'
# {"prediction": 150.32}
```

**Model info**

```bash
curl http://localhost:8000/model-info
# {"model_type":"RandomForestRegressor","dataset":"sklearn Diabetes Dataset","task":"regression","metrics":{"rmse":53.6803,"r2_score":0.4561}}
```

---

## API Endpoints

### `GET /`

Health check. Returns `200 OK` when the service is running.

### `POST /predict`

Accepts the 10 standardised baseline features and returns a **continuous regression score**.

**Request body:**

```json
{
  "age": 0.038,
  "sex": 0.05,
  "bmi": 0.061,
  "bp": 0.021,
  "s1": -0.044,
  "s2": -0.034,
  "s3": -0.043,
  "s4": -0.002,
  "s5": 0.019,
  "s6": -0.017
}
```

**Response:**

```json
{ "prediction": 150.32 }
```

> All 10 features are the mean-centred, scaled values exactly as provided by `sklearn.datasets.load_diabetes()`.

### `GET /model-info`

Returns metadata about the deployed model including evaluation metrics saved during training.

**Response:**

```json
{
  "model_type": "RandomForestRegressor",
  "dataset": "sklearn Diabetes Dataset",
  "task": "regression",
  "metrics": { "rmse": 53.6803, "r2_score": 0.4561 }
}
```

---

## Running Tests

From the **repo root**:

```bash
pytest labs/API_Labs/FastAPI_Lab/tests/test_api.py -v
```

The 12 tests cover:

| Test                                          | What it checks                                 |
| --------------------------------------------- | ---------------------------------------------- |
| `test_health_check`                           | `GET /` returns 200 and `{"status":"healthy"}` |
| `test_predict_returns_200`                    | Valid payload -> HTTP 200                      |
| `test_predict_response_has_prediction_key`    | Response contains `prediction` key             |
| `test_predict_returns_float`                  | Prediction is a float, not an int/class label  |
| `test_predict_value_in_reasonable_range`      | Score is between 25 and 350                    |
| `test_predict_missing_field_returns_422`      | Missing feature -> HTTP 422                    |
| `test_predict_wrong_type_returns_422`         | String instead of float -> HTTP 422            |
| `test_predict_empty_body_returns_422`         | Empty body -> HTTP 422                         |
| `test_model_info_returns_200`                 | `GET /model-info` returns 200                  |
| `test_model_info_schema`                      | Response has all required keys                 |
| `test_model_info_task_is_regression`          | `task` field equals `"regression"`             |
| `test_model_info_metrics_contain_rmse_and_r2` | Metrics contain `rmse` and `r2_score`          |

> **Note:** Run `python src/train.py` before running tests — the tests need the model file to exist.

---

## GitHub Actions CI

The workflow `.github/workflows/fastapi_lab_train_and_test.yml` runs automatically on push/PR to `main` when files under `labs/API_Labs/FastAPI_Lab/` change.

**Two sequential jobs:**

```
train -> test
```

1. **Train** — installs dependencies, runs `train.py`, uploads `diabetes_model.pkl` as a GitHub artifact
2. **Test** — downloads the artifact, runs all 12 pytest tests, uploads the XML test report

You can also trigger it manually from the **Actions** tab -> _FastAPI Lab - Train and Test_ -> **Run workflow**.

---

## FastAPI Concepts Covered

**Pydantic models for request/response validation**

```python
class DiabetesFeatures(BaseModel):
    age: float
    bmi: float
    # ...

class DiabetesResponse(BaseModel):
    prediction: float        # float output — not an int class label
```

FastAPI automatically validates incoming JSON against `DiabetesFeatures`. If any field is missing or the wrong type, it returns `422 Unprocessable Entity` with a detailed error — no manual validation code needed.

**Async route handlers**

```python
@app.post("/predict", response_model=DiabetesResponse)
async def predict_diabetes(features: DiabetesFeatures):
    ...
```

`async` allows the server to handle other requests while waiting on I/O (e.g. loading the model from disk), improving concurrency under load.

**HTTPException for error handling**

```python
raise HTTPException(status_code=500, detail=str(e))
```

When something goes wrong inside a route, raising `HTTPException` returns a structured JSON error response with the appropriate HTTP status code.

**Response model documentation**
Specifying `response_model=DiabetesResponse` in the decorator tells FastAPI to:

- Serialise the output as JSON matching that schema
- Include the response schema in the auto-generated `/docs` page
