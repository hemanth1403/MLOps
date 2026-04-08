#!/bin/sh
set -e

echo "Training model"
python src/train.py

echo "Starting FastAPI server"
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

echo "Starting drift detector"
python src/drift_detector.py &

# Keep container alive; exit if any background process dies
wait
