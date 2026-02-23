#!/bin/bash
# setup.sh - Initialize the DVC Lab project
# Run this once to set up Git, DVC, and the local remote storage for the project.

set -e

echo "  Heart Disease Prediction - DVC Lab Setup"

# --- Create virtual environment ---
echo ""
echo "[1/5] Creating virtual environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created at ./venv"
else
    echo "  Virtual environment already exists, skipping"
fi

# --- Activate virtual environment ---
echo ""
echo "[2/5] Activating virtual environment"
source venv/bin/activate
echo "  Using Python: $(which python)"
echo "  Python version: $(python --version)"

# --- Install dependencies ---
echo ""
echo "[3/5] Installing Python dependencies"
pip3 install --upgrade pip
pip3 install -r requirements.txt

# --- Initialize DVC ---
echo ""
echo "[4/5] Initializing DVC..."
if [ ! -d ".dvc" ]; then
    dvc init --subdir
    echo "  DVC initialized"
else
    echo "  DVC already initialized, skipping"
fi

# --- Set up local remote storage ---
echo ""
echo "[5/5] Setting up local DVC remote storage"
mkdir -p /tmp/dvc-local-remote
dvc remote add -d local_storage /tmp/dvc-local-remote 2>/dev/null || \
    dvc remote modify local_storage url /tmp/dvc-local-remote
echo "  Local remote configured at /tmp/dvc-local-remote"
echo ""
echo "  NOTE: To switch to Google Cloud Storage, run:"
echo "    dvc remote add -d gcs_storage gs://YOUR_BUCKET_NAME"
echo "    dvc remote modify gcs_storage credentialpath /path/to/credentials.json"

echo ""
echo "  Setup complete! Next steps:"
echo "  1. Activate venv:    source venv/bin/activate"
echo "  2. Run the pipeline: dvc repro"
echo "  3. Push data:        dvc push"
echo "  4. View metrics:     dvc metrics show"
