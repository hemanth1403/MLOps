"""
Wine Quality Prediction — FastAPI Serving Application
Serves the trained PyTorch model via REST API with a web interface.
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os
from datetime import datetime


# Model Definition [must match training] 
class WineClassifierNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(WineClassifierNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# Load Artifacts 
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

checkpoint = torch.load(
    os.path.join(ARTIFACTS_DIR, "wine_model.pth"),
    map_location=torch.device("cpu"),
    weights_only=False
)

model = WineClassifierNet(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    num_classes=checkpoint["num_classes"]
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "r") as f:
    training_metrics = json.load(f)

FEATURE_NAMES = checkpoint["feature_names"]
TARGET_NAMES = checkpoint["target_names"]

print(f"[INFO] Model loaded — classes: {TARGET_NAMES}")
print(f"[INFO] Training accuracy: {training_metrics['accuracy']:.4f}")

#  FastAPI App 
app = FastAPI(
    title="Wine Quality Classifier API",
    description="Predict wine cultivar from physicochemical properties",
    version="1.0.0"
)

app.mount("/statics", StaticFiles(directory="statics"), name="statics")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the prediction web interface."""
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
    })


@app.post("/predict")
async def predict(request: Request):
    """Make a prediction from form data or JSON payload."""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        data = await request.json()
        features = data.get("features", [])
    else:
        form = await request.form()
        features = [float(form.get(name, 0)) for name in FEATURE_NAMES]

    if len(features) != len(FEATURE_NAMES):
        return JSONResponse(
            status_code=400,
            content={"error": f"Expected {len(FEATURE_NAMES)} features, got {len(features)}"}
        )

    try:
        # Preprocess and predict
        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_tensor = torch.FloatTensor(input_scaled)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            predicted_idx = int(np.argmax(probabilities))

        predicted_class = TARGET_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx]) * 100

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": {
                name: round(float(prob) * 100, 2)
                for name, prob in zip(TARGET_NAMES, probabilities)
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "model": "WineClassifierNet",
        "framework": "PyTorch",
        "classes": TARGET_NAMES,
        "features_expected": len(FEATURE_NAMES),
        "training_accuracy": training_metrics["accuracy"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    """Return training metrics."""
    return training_metrics


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
