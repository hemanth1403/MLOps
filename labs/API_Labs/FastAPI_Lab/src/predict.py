import joblib


def predict_data(X):
    model = joblib.load("../model/diabetes_model.pkl")
    y_pred = model.predict(X)
    return y_pred
