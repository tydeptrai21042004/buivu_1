"""Shared neural-network helpers."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def build_target_scaler(y_train_log):
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(np.asarray(y_train_log).reshape(-1, 1)).ravel()
    return scaler, y_train_scaled


def inverse_nn_prediction(pred_scaled, target_scaler, inverse_log_fn):
    pred_log = target_scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()
    return inverse_log_fn(pred_log)


def train_neural_model(
    model,
    model_name: str,
    X_train_nn,
    X_test_nn,
    y_train_scaled,
    target_scaler,
    inverse_log_fn,
    evaluate_fn,
    y_test,
    epochs: int = 800,
    batch_size: int = 8,
    patience: int = 80,
):
    """Train a Keras neural model and return metrics, predictions, and history."""
    try:
        from tensorflow.keras.callbacks import EarlyStopping
    except Exception as exc:
        raise RuntimeError("TensorFlow/Keras is required for neural-network models.") from exc

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_nn,
        y_train_scaled,
        validation_split=0.20,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    pred_scaled = model.predict(X_test_nn, verbose=0).ravel()
    pred = inverse_nn_prediction(pred_scaled, target_scaler, inverse_log_fn)
    metrics = evaluate_fn(y_test, pred)
    metrics["Model"] = model_name
    metrics["Epochs"] = len(history.history["loss"])

    return metrics, pred, history
