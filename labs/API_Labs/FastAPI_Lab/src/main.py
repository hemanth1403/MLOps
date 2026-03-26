import json
import os
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from predict import predict_data


app = FastAPI(
    title="Diabetes Progression Predictor",
    description=(
        "A regression API that predicts diabetes disease progression score "
        "using a Random Forest Regressor trained on the sklearn Diabetes dataset."
    ),
    version="1.0.0",
)


class DiabetesFeatures(BaseModel):


    age: float = Field(..., json_schema_extra={"example": 0.038})
    sex: float = Field(..., json_schema_extra={"example": 0.050})
    bmi: float = Field(..., json_schema_extra={"example": 0.061})
    bp: float = Field(..., json_schema_extra={"example": 0.021})
    s1: float = Field(..., json_schema_extra={"example": -0.044})
    s2: float = Field(..., json_schema_extra={"example": -0.034})
    s3: float = Field(..., json_schema_extra={"example": -0.043})
    s4: float = Field(..., json_schema_extra={"example": -0.002})
    s5: float = Field(..., json_schema_extra={"example": 0.019})
    s6: float = Field(..., json_schema_extra={"example": -0.017})


class DiabetesResponse(BaseModel):

    prediction: float


class ModelInfo(BaseModel):

    model_type: str
    dataset: str
    task: str
    metrics: dict


# Routes



@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    # Health check — returns 200 OK when the service is up
    return {"status": "healthy"}


@app.post("/predict", response_model=DiabetesResponse)
async def predict_diabetes(features: DiabetesFeatures):
    try:
        feature_vector = [[
            features.age, features.sex, features.bmi, features.bp,
            features.s1, features.s2, features.s3,
            features.s4, features.s5, features.s6,
        ]]
        prediction = predict_data(feature_vector)
        return DiabetesResponse(prediction=round(float(prediction[0]), 4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    metrics_path = "../model/metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run train.py first.",
        )
    with open(metrics_path) as f:
        metrics = json.load(f)

    return ModelInfo(
        model_type="RandomForestRegressor",
        dataset="sklearn Diabetes Dataset",
        task="regression",
        metrics=metrics,
    )
