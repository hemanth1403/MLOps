"""
ML Model Serving with Prometheus Metrics

This FastAPI application serves ML models with comprehensive monitoring:
- Request/response logging
- Prediction latency tracking
- Model performance metrics
- Health checks
- Prometheus metrics export
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import os

import mlflow
import mlflow.sklearn
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, 
    CONTENT_TYPE_LATEST, CollectorRegistry
)
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="Production ML serving with monitoring",
    version="1.0.0"
)

# Prometheus metrics
registry = CollectorRegistry()

# Request metrics
request_count = Counter(
    'model_requests_total',
    'Total number of prediction requests',
    ['model_version', 'status'],
    registry=registry
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Time spent processing prediction',
    ['model_version'],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

# Model metrics
model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model_version'],
    registry=registry
)

prediction_confidence = Histogram(
    'prediction_confidence',
    'Model prediction confidence scores',
    ['model_version', 'predicted_class'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    registry=registry
)

# System metrics
active_requests = Gauge(
    'active_requests',
    'Number of requests currently being processed',
    registry=registry
)

error_count = Counter(
    'model_errors_total',
    'Total number of errors',
    ['error_type'],
    registry=registry
)

# Data drift metrics
feature_distribution = Histogram(
    'feature_values',
    'Distribution of feature values',
    ['feature_name'],
    registry=registry
)


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[float] = Field(..., description="Input features for prediction")
    model_version: Optional[str] = Field(default="latest", description="Model version to use")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.038, 0.05, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017],
                "model_version": "latest"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Prediction probability")
    confidence: float = Field(..., description="Confidence score")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    timestamp: str


class MetricsResponse(BaseModel):
    """Metrics endpoint response"""
    total_requests: int
    total_errors: int
    average_latency_ms: float
    model_accuracy: float


# Global state
class ModelState:
    """Global model state"""
    def __init__(self):
        self.model = None
        self.model_version = None
        self.model_name = None
        self.scaler = None
        self.feature_names = None
        self.start_time = time.time()
        self.prediction_history = []
        
    def load_model(self, model_name: str = None, model_version: str = "latest"):
        """Load model from MLflow using run ID"""
        try:
            logger.info(f"Loading model from MLflow...")
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
            
            # Instead of using model registry, load from runs
            # Get the experiment
            experiment = mlflow.get_experiment_by_name("diabetes-classification")
            
            if experiment is None:
                logger.error("Experiment 'diabetes-classification' not found")
                return False
            
            # Get the best run (by test_accuracy)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.test_accuracy DESC"],
                max_results=1
            )
            
            if len(runs) == 0:
                logger.error("No runs found in experiment")
                return False
            
            best_run = runs.iloc[0]
            run_id = best_run['run_id']
            
            logger.info(f"Loading best model from run: {run_id}")
            
            # Load model from run
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_version = "best"
            self.model_name = "diabetes-classification"
            
            logger.info(f"✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction with timing"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get confidence (max probability)
        confidence = float(np.max(probabilities))
        probability = float(probabilities[prediction])
        
        # Record metrics
        prediction_latency.labels(model_version=self.model_version).observe(latency / 1000)
        prediction_confidence.labels(
            model_version=self.model_version,
            predicted_class=str(prediction)
        ).observe(confidence)
        
        # Log feature distributions
        for i, val in enumerate(features[0]):
            feature_distribution.labels(feature_name=f'feature_{i}').observe(val)
        
        # Store in history (keep last 1000)
        self.prediction_history.append({
            'prediction': int(prediction),
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)
        
        return {
            'prediction': int(prediction),
            'probability': probability,
            'confidence': confidence,
            'latency_ms': latency
        }


# Initialize model state
model_state = ModelState()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting ML serving API...")
    
    # Load model
    success = model_state.load_model(
        model_name="diabetes-classification_random_forest",
        model_version="latest"
    )
    
    if not success:
        logger.warning("Failed to load model on startup - will try on first request")
    
    logger.info("✅ API startup complete")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model Serving API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - model_state.start_time
    
    return HealthResponse(
        status="healthy" if model_state.model is not None else "degraded",
        model_loaded=model_state.model is not None,
        model_version=model_state.model_version or "none",
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for k8s"""
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction with monitoring"""
    
    active_requests.inc()
    start_time = time.time()
    
    try:
        # Validate input
        if len(request.features) != 10:
            error_count.labels(error_type='invalid_input').inc()
            raise HTTPException(
                status_code=400,
                detail=f"Expected 10 features, got {len(request.features)}"
            )
        
        # Ensure model is loaded
        if model_state.model is None:
            logger.info("Model not loaded, attempting to load...")
            success = model_state.load_model(
                model_name="diabetes-classification_random_forest"
            )
            if not success:
                raise HTTPException(status_code=503, detail="Model not available")
        
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        result = model_state.predict(features)
        
        # Record success
        request_count.labels(
            model_version=model_state.model_version,
            status='success'
        ).inc()
        
        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            confidence=result['confidence'],
            model_version=model_state.model_version,
            timestamp=datetime.now().isoformat(),
            latency_ms=result['latency_ms']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        error_count.labels(error_type='prediction_error').inc()
        request_count.labels(
            model_version=model_state.model_version or 'unknown',
            status='error'
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        active_requests.dec()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(
        generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model/info")
async def model_info():
    """Get current model information"""
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model_state.model_name,
        "model_version": model_state.model_version,
        "model_type": type(model_state.model).__name__,
        "loaded_at": datetime.fromtimestamp(model_state.start_time).isoformat(),
        "predictions_made": len(model_state.prediction_history)
    }


@app.get("/model/history")
async def prediction_history(limit: int = 100):
    """Get recent prediction history"""
    if not model_state.prediction_history:
        return {"history": [], "count": 0}
    
    history = model_state.prediction_history[-limit:]
    
    return {
        "history": history,
        "count": len(history),
        "latest": history[-1] if history else None
    }


@app.post("/model/reload")
async def reload_model(model_name: str, model_version: str = "latest"):
    """Reload model with new version"""
    logger.info(f"Reloading model: {model_name} v{model_version}")
    
    success = model_state.load_model(model_name, model_version)
    
    if success:
        return {
            "status": "success",
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware for request logging"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {duration:.3f}s")
    
    return response


def main():
    """Run the FastAPI server"""
    logger.info("Starting ML Model Serving API...")
    
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
