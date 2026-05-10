"""Evaluation metrics used in the paper and additional metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mmre(y_true, y_pred) -> float:
    """Mean Magnitude Relative Error."""
    eps = 1e-8
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)))


def bre(y_true, y_pred) -> float:
    """Balanced Relative Error."""
    eps = 1e-8
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.minimum(np.abs(y_true), np.abs(y_pred)), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denominator))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Evaluate predictions using paper metrics and extra regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-6)
    mmre_value = mmre(y_true, y_pred)

    return {
        "MMRE": mmre_value,
        "MMRE (%)": mmre_value * 100,
        "RMSE": rmse(y_true, y_pred),
        "BRE": bre(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Accuracy (%)": max(0.0, 1.0 - mmre_value) * 100,
    }
