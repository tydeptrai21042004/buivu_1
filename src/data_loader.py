"""Dataset loading utilities for the NASA93 KNN web demo."""
from __future__ import annotations

from pathlib import Path
from typing import IO
import re

import numpy as np
import pandas as pd


def normalize_column_name(name: object) -> str:
    """Normalize column names from Kaggle/Excel-style CSV files.

    Examples:
        " Effort " -> "effort"
        "Team Size" -> "team size"
        "\ufeffprec" -> "prec"
    """
    text = str(name).replace("\ufeff", "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def read_csv_safely(path_or_buffer: str | Path | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Read a CSV robustly and normalize its columns.

    The previous version used ``pd.to_numeric(..., errors='ignore')``.  Newer
    pandas versions may reject that deprecated value and raise
    ``ValueError: invalid error value specified``.  This function avoids that
    incompatibility and leaves type conversion to ``select_features_and_target``.
    """
    try:
        df = pd.read_csv(path_or_buffer)
    except UnicodeDecodeError:
        # Some CSV files exported from Excel use latin1/cp1252 encoding.
        df = pd.read_csv(path_or_buffer, encoding="latin1")

    df.columns = [normalize_column_name(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.replace(["?", "NA", "N/A", "na", "null", "NULL", "", " "], np.nan)
    return df


def load_nasa93_csv(path: str | Path) -> pd.DataFrame:
    """Load the NASA93 CSV file and perform light column cleaning."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    return read_csv_safely(data_path)


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

    before = len(clean_df)
    clean_df = clean_df.dropna().reset_index(drop=True)
    if clean_df.empty:
        raise ValueError(
            "No usable rows remain after converting required columns to numeric values. "
            "Please check that the NASA93 feature columns and effort column contain numbers."
        )

    if (clean_df[target] <= 0).any():
        removed = int((clean_df[target] <= 0).sum())
        clean_df = clean_df[clean_df[target] > 0].reset_index(drop=True)
        if clean_df.empty:
            raise ValueError("All effort values are non-positive, so log1p training is not valid.")
        print(f"Removed {removed} row(s) with non-positive effort values.")

    dropped = before - len(clean_df)
    if dropped:
        print(f"Dropped {dropped} incomplete/non-numeric row(s).")

    X = clean_df[features].astype(float).copy()
    y = clean_df[target].astype(float).copy()

    return X, y, clean_df
