"""Train and save KNN model artifacts for the Streamlit demo."""
from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_DATA_FILE, RANDOM_STATE, TEST_SIZE
from src.inference import train_and_save_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Train deployable KNN NASA93 effort-prediction artifacts")
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="Path to NASA_93_Sheet.csv")
    parser.add_argument("--artifact-dir", default="artifacts", help="Folder for saved model artifacts")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = train_and_save_artifacts(
        data_path=args.data,
        artifact_dir=args.artifact_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("Saved artifacts to:", Path(args.artifact_dir).resolve())
    print("Model:", metadata["model_name"])
    print("Training samples after cleaning:", metadata["n_samples"])


if __name__ == "__main__":
    main()
