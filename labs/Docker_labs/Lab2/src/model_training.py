"""
Wine Quality Prediction - Model Training Pipeline
Uses PyTorch to train a neural network on the UCI Wine Quality dataset.
Includes data preprocessing, validation splits, and model evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import json
import pickle
import numpy as np
import os
from datetime import datetime


class WineClassifierNet(nn.Module):
    """Multi-layer neural network for wine classification."""

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


def load_and_preprocess_data():
    """Load wine dataset and apply preprocessing."""
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = list(wine.target_names)

    # Validate data quality
    assert not np.isnan(X).any(), "Dataset contains NaN values"
    assert len(X) == len(y), "Feature and target length mismatch"
    print(f"[INFO] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[INFO] Classes: {target_names}")
    print(f"[INFO] Class distribution: {dict(zip(target_names, np.bincount(y)))}")

    # Split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, target_names


def train_model(X_train, y_train, X_val, y_val, input_dim, num_classes,
                hidden_dim=64, epochs=150, lr=0.001, patience=15):
    """Train the PyTorch model with early stopping."""

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    model = WineClassifierNet(input_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    print(f"\n[INFO] Training started — {epochs} max epochs, patience={patience}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        scheduler.step(val_loss)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_accuracy"].append(val_acc)

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — "
                  f"Train Loss: {loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, "
                  f"Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [INFO] Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(best_model_state)
    print(f"[INFO] Best validation loss: {best_val_loss:.4f}")
    return model, history


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate model on test set and return metrics."""
    model.eval()
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    with torch.no_grad():
        outputs = model(X_test_t)
        predictions = torch.argmax(outputs, dim=1).numpy()
        probabilities = torch.softmax(outputs, dim=1).numpy()

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names, output_dict=True)

    print(f"\n[RESULTS] Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=target_names))

    return accuracy, report, probabilities


if __name__ == "__main__":
    print("=" * 60)
    print("  Wine Quality Classifier — PyTorch Training Pipeline")
    print("=" * 60)

    # Load and preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, target_names = \
        load_and_preprocess_data()

    # Train
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        input_dim=X_train.shape[1],
        num_classes=len(target_names)
    )

    # Evaluate
    accuracy, report, _ = evaluate_model(model, X_test, y_test, target_names)

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "hidden_dim": 64,
        "num_classes": len(target_names),
        "feature_names": feature_names,
        "target_names": target_names,
    }, "artifacts/wine_model.pth")

    # Save scaler
    with open("artifacts/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "training_epochs": len(history["train_loss"]),
        "best_val_loss": min(history["val_loss"]),
        "timestamp": datetime.now().isoformat(),
        "model_architecture": "WineClassifierNet(13->64->32->3)",
        "framework": "PyTorch",
    }
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[INFO] Artifacts saved to artifacts/")
    print(f"  - wine_model.pth (model weights)")
    print(f"  - scaler.pkl (feature scaler)")
    print(f"  - metrics.json (evaluation results)")
    print("=" * 60)
    print("  Training pipeline completed successfully!")
    print("=" * 60)
