"""Model evaluation helpers."""
from __future__ import annotations

import pandas as pd


def rank_results(results: list[dict]) -> pd.DataFrame:
    """Rank models using the three paper metrics: MMRE, RMSE, and BRE."""
    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "Model",
            "MMRE",
            "MMRE (%)",
            "RMSE",
            "BRE",
            "MAE",
            "R2",
            "Accuracy (%)",
        ] + (["Epochs"] if "Epochs" in results_df.columns else [])
    ]

    results_df["Rank_MMRE"] = results_df["MMRE"].rank(method="min")
    results_df["Rank_RMSE"] = results_df["RMSE"].rank(method="min")
    results_df["Rank_BRE"] = results_df["BRE"].rank(method="min")
    results_df["Average_Rank"] = (
        results_df["Rank_MMRE"] + results_df["Rank_RMSE"] + results_df["Rank_BRE"]
    ) / 3

    return results_df.sort_values(by="Average_Rank").reset_index(drop=True)


def build_prediction_table(y_test, best_pred) -> pd.DataFrame:
    import numpy as np

    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(best_pred, dtype=float)
    return pd.DataFrame({
        "Actual_Effort": actual,
        "Predicted_Effort": predicted,
        "Absolute_Error": np.abs(actual - predicted),
        "Relative_Error": np.abs(actual - predicted) / np.maximum(np.abs(actual), 1e-8),
    })
