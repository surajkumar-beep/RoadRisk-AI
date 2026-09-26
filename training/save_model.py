"""
Reusable functions to persist trained models and their companion artifacts
(model, scaler, feature column order, label encoders).

Paths all live under the project 'models' directory, matching the convention
already used by the preprocessing pipeline.
"""
import os
import pickle

# Project root (parent of this file's directory).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "accident_severity_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved artifact -> {path}")


def save_model(model, path=MODEL_PATH):
    """Save a fitted sklearn/xgboost model to pickle."""
    _dump(model, path)


def save_scaler(scaler, path=SCALER_PATH):
    """Save the fitted scaler (reused/perpetuated from preprocessing) to pickle."""
    _dump(scaler, path)


def save_feature_columns(columns, path=FEATURE_COLUMNS_PATH):
    """Save the final feature column order used by the model."""
    _dump(list(columns), path)


def save_label_encoders(encoders, path=LABEL_ENCODERS_PATH):
    """Save the label/target-encoding artifacts to pickle."""
    _dump(encoders, path)


def load_model(path=MODEL_PATH):
    """Load a pickled model."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_columns(path=FEATURE_COLUMNS_PATH):
    """Load the saved feature column order."""
    with open(path, "rb") as f:
        return pickle.load(f)
