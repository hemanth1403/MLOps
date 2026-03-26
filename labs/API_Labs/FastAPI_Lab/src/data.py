import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def get_feature_names():
    return load_diabetes().feature_names
