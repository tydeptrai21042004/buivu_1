"""Preprocessing utilities."""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> Pipeline:
    """Build the numerical preprocessing pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def transform_target_log(y):
    """Apply log1p transform to effort."""
    return np.log1p(y)


def inverse_log_prediction(y_pred_log):
    """Convert predicted log effort back to positive effort."""
    y_pred = np.expm1(y_pred_log)
    return np.maximum(y_pred, 1e-6)
