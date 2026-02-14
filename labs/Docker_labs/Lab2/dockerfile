# ============================================================
# Wine Quality Classifier — Multi-Stage Docker Build
# Stage 1: Train the model
# Stage 2: Serve predictions via FastAPI
# ============================================================

# Stage 1: Model Training
FROM python:3.10-slim AS trainer

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/model_training.py .

# Train model and produce artifacts
RUN python model_training.py


# Stage 2: Serving
FROM python:3.10-slim AS server

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained model artifacts from training stage
COPY --from=trainer /app/artifacts ./artifacts

# Copy application code
COPY src/main.py .
COPY src/templates ./templates
COPY src/statics ./statics

# Environment configuration
ENV ARTIFACTS_DIR=artifacts
ENV PORT=8000

# Expose the serving port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch the FastAPI server
CMD ["python", "main.py"]
