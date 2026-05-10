from pathlib import Path

import pandas as pd

from config import COCOMO_FEATURES, TARGET_COLUMN
from src.data_loader import load_nasa93_csv, select_features_and_target


def test_sample_data_schema():
    path = Path(__file__).parent / "data" / "sample_nasa93_small.csv"
    df = load_nasa93_csv(path)
    X, y, clean_df = select_features_and_target(df, COCOMO_FEATURES, TARGET_COLUMN)
    assert X.shape[1] == len(COCOMO_FEATURES)
    assert len(X) == len(y)
    assert TARGET_COLUMN in clean_df.columns
