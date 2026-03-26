import json
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data import load_data, split_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def fit_model(X_train, y_train):
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    rf_regressor.fit(X_train, y_train)
    joblib.dump(rf_regressor, "../model/diabetes_model.pkl")
    return rf_regressor


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    metrics = {"rmse": round(rmse, 4), "r2_score": round(r2, 4)}

    with open("../model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    return metrics


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
