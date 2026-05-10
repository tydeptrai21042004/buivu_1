"""Step-by-step pipeline following the paper methodology."""
from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
    COCOMO_FEATURES,
    OUTPUT_RESULT_FILE,
    OUTPUT_PREDICTION_FILE,
)
from .data_loader import load_nasa93_csv, select_features_and_target
from .preprocessing import build_preprocessor, transform_target_log, inverse_log_prediction
from .metrics import evaluate_regression
from .evaluation import rank_results, build_prediction_table
from .plotting import save_all_plots
from .models.classical_models import build_classical_models
from .models.tree_models import build_tree_models


def _set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


def print_step(step: int, title: str) -> None:
    print("\n" + "=" * 80)
    print(f"STEP {step}. {title}")
    print("=" * 80)


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path = "outputs",
    skip_neural: bool = False,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Run the complete experiment and save outputs."""
    _set_seed(random_state)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Problem definition
    print_step(1, "PROBLEM DEFINITION")
    print("Goal: predict software development effort/cost from COCOMO/NASA project features.")
    print("Task type: regression.")

    # Step 2: Dataset
    print_step(2, "DATASET LOADED")
    df = load_nasa93_csv(data_path)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Step 3: Preprocessing and feature selection
    print_step(3, "CLEAN COCOMO/NASA FEATURES SELECTED")
    X, y, clean_df = select_features_and_target(df, COCOMO_FEATURES, TARGET_COLUMN)
    print("Selected features:", COCOMO_FEATURES)
    print("Target column:", TARGET_COLUMN)
    print("Clean dataset shape:", clean_df.shape)
    print("Target summary:")
    print(y.describe())

    # Step 4: log target
    print_step(4, "TARGET TRANSFORMATION")
    y_log = transform_target_log(y)
    print("Training target: log1p(effort).")
    print("Prediction inverse: expm1(predicted_log_effort).")

    # Step 5: train/test split
    print_step(5, "TRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test, y_train_log, _ = train_test_split(
        X,
        y,
        y_log,
        test_size=test_size,
        random_state=random_state,
    )
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # Step 6: feature preprocessing
    print_step(6, "PREPROCESSING")
    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    print("Scaled training shape:", X_train_scaled.shape)
    print("Scaled testing shape:", X_test_scaled.shape)

    # Step 7: metrics
    print_step(7, "METRICS READY")
    print("Metrics: MMRE, RMSE, BRE, MAE, R2, Accuracy = 1 - MMRE.")

    results: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    trained_classical_models = {}

    # Step 8: classical models
    print_step(8, "CLASSICAL AND TREE MODELS")
    classical_models = build_classical_models()
    classical_models.update(build_tree_models(random_state=random_state))

    for model_name, model in classical_models.items():
        print(f"Training: {model_name}")
        model.fit(X_train_scaled, y_train_log)
        pred_log = model.predict(X_test_scaled)
        pred = inverse_log_prediction(pred_log)
        metrics = evaluate_regression(y_test, pred)
        metrics["Model"] = model_name
        results.append(metrics)
        predictions[model_name] = pred
        trained_classical_models[model_name] = model
        print(f"  MMRE={metrics['MMRE']:.6f}, RMSE={metrics['RMSE']:.6f}, BRE={metrics['BRE']:.6f}")

    # Step 9-17: neural networks
    if not skip_neural:
        print_step(9, "NEURAL NETWORK MODELS")
        try:
            from .models.neural_base import build_target_scaler, train_neural_model
            from .models.mlp_model import build_mlp, build_deep_nn
            from .models.cascade_model import build_cascade_forward_nn
            from .models.sequence_models import (
                reshape_for_sequence,
                build_elman_nn,
                build_lstm_nn,
                build_gru_nn,
                build_cnn_1d,
            )

            target_scaler, y_train_log_scaled = build_target_scaler(y_train_log)
            input_dim = X_train_scaled.shape[1]

            neural_jobs = [
                ("MLP Neural Network", build_mlp(input_dim), X_train_scaled, X_test_scaled),
                ("Deep Neural Network", build_deep_nn(input_dim), X_train_scaled, X_test_scaled),
                ("Cascade Forward Neural Network", build_cascade_forward_nn(input_dim), X_train_scaled, X_test_scaled),
            ]

            X_train_seq = reshape_for_sequence(X_train_scaled)
            X_test_seq = reshape_for_sequence(X_test_scaled)
            timesteps = X_train_seq.shape[1]

            neural_jobs.extend([
                ("Elman Neural Network", build_elman_nn(timesteps), X_train_seq, X_test_seq),
                ("LSTM Neural Network", build_lstm_nn(timesteps), X_train_seq, X_test_seq),
                ("GRU Neural Network", build_gru_nn(timesteps), X_train_seq, X_test_seq),
                ("1D-CNN Neural Network", build_cnn_1d(timesteps), X_train_seq, X_test_seq),
            ])

            for model_name, model, X_tr, X_te in neural_jobs:
                metrics, pred, _history = train_neural_model(
                    model=model,
                    model_name=model_name,
                    X_train_nn=X_tr,
                    X_test_nn=X_te,
                    y_train_scaled=y_train_log_scaled,
                    target_scaler=target_scaler,
                    inverse_log_fn=inverse_log_prediction,
                    evaluate_fn=evaluate_regression,
                    y_test=y_test,
                )
                results.append(metrics)
                predictions[model_name] = pred
                print(f"  {model_name}: MMRE={metrics['MMRE']:.6f}, RMSE={metrics['RMSE']:.6f}, BRE={metrics['BRE']:.6f}")

        except Exception as exc:
            print("Neural-network section skipped due to error:", repr(exc))
    else:
        print_step(9, "NEURAL NETWORK MODELS SKIPPED")

    # Step 18: final comparison
    print_step(18, "FINAL MODEL COMPARISON")
    results_df = rank_results(results)
    print(results_df)
    results_df.to_csv(output_dir / OUTPUT_RESULT_FILE, index=False)

    # Step 19: best model prediction table
    print_step(19, "BEST MODEL PREDICTION TABLE")
    best_model_name = results_df.iloc[0]["Model"]
    best_pred = predictions[best_model_name]
    prediction_table = build_prediction_table(y_test, best_pred)
    print("Best model:", best_model_name)
    print(prediction_table)
    prediction_table.to_csv(output_dir / OUTPUT_PREDICTION_FILE, index=False)

    # Step 20: visualizations
    print_step(20, "VISUALIZATION OUTPUTS")
    save_all_plots(results_df, y_test, best_pred, best_model_name, output_dir)
    print("Plots saved to:", output_dir)

    return {
        "results_df": results_df,
        "prediction_table": prediction_table,
        "best_model_name": best_model_name,
        "output_dir": output_dir,
    }
