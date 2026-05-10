"""Command-line entry point for NASA93 software effort prediction."""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from config import DEFAULT_DATA_FILE, RANDOM_STATE, TEST_SIZE
from src.pipeline import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="NASA93 software effort prediction")
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="Path to NASA_93_Sheet.csv")
    parser.add_argument("--output", default="outputs", help="Output folder")
    parser.add_argument("--skip-neural", action="store_true", help="Skip TensorFlow neural-network models")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(
        data_path=args.data,
        output_dir=args.output,
        skip_neural=args.skip_neural,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
