"""
Drift detector — runs every 60 seconds.
Loads the training baseline and recent inference data from api.log,
then performs a two-sample KS test per feature.
Results are written to logs/drift.log for Logstash ingestion.
"""

import json
import os
import pickle
import time
from datetime import datetime, timezone

import numpy as np
from scipy import stats
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

MIN_INFERENCE_SAMPLES = 10
DRIFT_ALPHA = 0.05
CHECK_INTERVAL_SECONDS = 60


def _write_log(entry: dict):
    with open(os.path.join(LOG_DIR, "drift.log"), "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry))


def _load_baseline() -> dict:
    with open(os.path.join(MODEL_DIR, "baseline.pkl"), "rb") as f:
        return pickle.load(f)


def _load_inference_data(feature_names: list) -> Optional[np.ndarray]:
    api_log = os.path.join(LOG_DIR, "api.log")
    if not os.path.exists(api_log):
        return None

    rows = []
    with open(api_log, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "api_request" and "features" in entry:
                    row = [entry["features"].get(name, 0.0) for name in feature_names]
                    rows.append(row)
            except (json.JSONDecodeError, KeyError):
                continue

    return np.array(rows) if len(rows) >= MIN_INFERENCE_SAMPLES else None


def run_once():
    baseline = _load_baseline()
    feature_names: list = baseline["feature_names"]
    train_data = np.array(baseline["train_samples"])

    inference_data = _load_inference_data(feature_names)

    if inference_data is None:
        _write_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "type": "drift_detection",
                "message": "Insufficient inference data — skipping",
                "inference_samples": 0,
                "min_required": MIN_INFERENCE_SAMPLES,
            }
        )
        return

    results = []
    for i, feature in enumerate(feature_names):
        ks_stat, p_value = stats.ks_2samp(train_data[:, i], inference_data[:, i])
        results.append(
            {
                "feature": feature,
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": bool(p_value < DRIFT_ALPHA),
            }
        )

    drifted = [r["feature"] for r in results if r["drift_detected"]]
    any_drift = len(drifted) > 0

    _write_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "WARNING" if any_drift else "INFO",
            "type": "drift_detection",
            "message": (
                f"Drift detected in {len(drifted)} feature(s): {drifted}"
                if any_drift
                else "No drift detected"
            ),
            "inference_samples": int(inference_data.shape[0]),
            "training_samples": int(train_data.shape[0]),
            "drift_threshold": DRIFT_ALPHA,
            "any_drift_detected": any_drift,
            "drifted_features": drifted,
            "feature_results": results,
        }
    )


def main():
    _write_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "type": "drift_detection",
            "message": "Drift detector started",
            "check_interval_seconds": CHECK_INTERVAL_SECONDS,
            "drift_threshold": DRIFT_ALPHA,
        }
    )
    while True:
        try:
            run_once()
        except Exception as exc:
            _write_log(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "ERROR",
                    "type": "drift_detection",
                    "message": f"Drift detection error: {exc}",
                }
            )
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
