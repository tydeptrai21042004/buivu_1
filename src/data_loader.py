"""Dataset loading utilities."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def load_nasa93_csv(path: str | Path) -> pd.DataFrame:
    """Load the NASA93 CSV file and perform light column cleaning."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.replace(["?", "NA", "N/A", "na", "null", "NULL", ""], np.nan)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_required_columns(df: pd.DataFrame, features: list[str], target: str) -> None:
    """Validate that all required COCOMO/NASA features and target exist."""
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required COCOMO/NASA features: {missing_features}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")


def select_features_and_target(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Select clean COCOMO/NASA features and the effort target."""
    validate_required_columns(df, features, target)

    clean_df = df[features + [target]].copy()
    for col in features + [target]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df = clean_df.dropna().reset_index(drop=True)
    X = clean_df[features].copy()
    y = clean_df[target].astype(float).copy()

    return X, y, clean_df
