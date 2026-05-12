"""KNN-only training and inference helpers for the Streamlit demo.

The website demo intentionally uses one model: KNN.  This keeps the demo simple:
users enter the 23 NASA93/COCOMO features and the app returns one predicted
software-effort value.
"""
from __future__ import annotations

from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import COCOMO_FEATURES, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from .data_loader import load_nasa93_csv, select_features_and_target
from .preprocessing import build_preprocessor, transform_target_log, inverse_log_prediction
from .metrics import evaluate_regression
from .models.knn_model import build_knn


ARTIFACT_MODEL = "knn_model.joblib"
ARTIFACT_PREPROCESSOR = "preprocessor.joblib"
ARTIFACT_METADATA = "metadata.json"
ARTIFACT_TRAINING_DATA = "clean_training_data.csv"
MODEL_NAME = "KNN"


def _set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_and_save_artifacts(
    data_path: str | Path,
    artifact_dir: str | Path = "artifacts",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Train only the KNN model and save deployable files."""
    _set_seed(random_state)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = load_nasa93_csv(data_path)
    X, y, clean_df = select_features_and_target(df, COCOMO_FEATURES, TARGET_COLUMN)
    y_log = transform_target_log(y)

    X_train, X_test, y_train, y_test, y_train_log, _ = train_test_split(
        X,
        y,
        y_log,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    model = build_knn()
    model.fit(X_train_scaled, y_train_log)

    pred_log = model.predict(X_test_scaled)
    pred = inverse_log_prediction(pred_log)
    metrics = evaluate_regression(y_test, pred)
    metrics["Model"] = MODEL_NAME

    joblib.dump(model, artifact_dir / ARTIFACT_MODEL)
    joblib.dump(preprocessor, artifact_dir / ARTIFACT_PREPROCESSOR)
    clean_df.to_csv(artifact_dir / ARTIFACT_TRAINING_DATA, index=False)

    feature_defaults = clean_df[COCOMO_FEATURES].median(numeric_only=True).to_dict()
    feature_min = clean_df[COCOMO_FEATURES].min(numeric_only=True).to_dict()
    feature_max = clean_df[COCOMO_FEATURES].max(numeric_only=True).to_dict()

    metadata = {
        "model_name": MODEL_NAME,
        # kept for compatibility with older app text or notebooks
        "best_model_name": MODEL_NAME,
        "features": COCOMO_FEATURES,
        "target": TARGET_COLUMN,
        "feature_defaults": feature_defaults,
        "feature_min": feature_min,
        "feature_max": feature_max,
        "metrics": [metrics],
        "n_samples": int(clean_df.shape[0]),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "artifact_model_file": ARTIFACT_MODEL,
    }
    with open(artifact_dir / ARTIFACT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_artifacts(artifact_dir: str | Path = "artifacts"):
    """Load the saved KNN model, preprocessor, and metadata."""
    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / ARTIFACT_MODEL
    preprocessor_path = artifact_dir / ARTIFACT_PREPROCESSOR
    metadata_path = artifact_dir / ARTIFACT_METADATA

    missing = [p.name for p in [model_path, preprocessor_path, metadata_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing KNN artifact file(s): " + ", ".join(missing) +
            ". Run `python train_artifacts.py --data path/to/NASA_93_Sheet.csv` "
            "or train from the Streamlit sidebar first."
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, preprocessor, metadata


def predict_effort(feature_values: dict, model, preprocessor, metadata: dict) -> float:
    """Predict effort from one dictionary of COCOMO feature values."""
    features = metadata.get("features", COCOMO_FEATURES)
    row = {feature: float(feature_values[feature]) for feature in features}
    X = pd.DataFrame([row], columns=features)
    X_scaled = preprocessor.transform(X)
    pred_log = model.predict(X_scaled)
    pred = inverse_log_prediction(pred_log)
    return float(pred[0])


def predict_effort_batch(input_df: pd.DataFrame, model, preprocessor, metadata: dict) -> pd.DataFrame:
    """Predict effort for a CSV/DataFrame containing all required features."""
    features = metadata.get("features", COCOMO_FEATURES)
    missing = [feature for feature in features if feature not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = input_df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X_scaled = preprocessor.transform(X)
    pred_log = model.predict(X_scaled)
    pred = inverse_log_prediction(pred_log)

    output = input_df.copy()
    output["predicted_effort_knn"] = pred
    return output
