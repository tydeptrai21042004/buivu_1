"""Plotting helpers for model comparison."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_actual_vs_predicted(y_test, best_pred, best_model_name: str, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, best_pred)
    min_value = min(np.min(y_test), np.min(best_pred))
    max_value = max(np.max(y_test), np.max(best_pred))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Actual Effort")
    plt.ylabel("Predicted Effort")
    plt.title(f"Actual vs Predicted - {best_model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=200)
    plt.close()


def save_metric_bar(results_df, metric: str, filename: str, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.bar(results_df["Model"], results_df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def save_all_plots(results_df, y_test, best_pred, best_model_name: str, output_dir: str | Path) -> None:
    save_actual_vs_predicted(y_test, best_pred, best_model_name, output_dir)
    save_metric_bar(results_df, "MMRE (%)", "mmre_comparison.png", output_dir)
    save_metric_bar(results_df, "RMSE", "rmse_comparison.png", output_dir)
    save_metric_bar(results_df, "BRE", "bre_comparison.png", output_dir)
