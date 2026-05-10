"""Project configuration for NASA93 software effort prediction."""

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_DATA_FILE = "/kaggle/input/datasets/asmasadaqat/nasa93/NASA_93_Sheet.csv"
TARGET_COLUMN = "effort"

COCOMO_FEATURES = [
    "prec", "flex", "resl", "team", "pmat",
    "rely", "data", "cplx", "ruse", "docu",
    "time", "stor", "pvol",
    "acap", "pcap", "pcon", "apex", "plex", "ltex",
    "tool", "site", "sced",
    "kloc",
]

OUTPUT_RESULT_FILE = "corrected_log_effort_model_comparison.csv"
OUTPUT_PREDICTION_FILE = "corrected_best_model_predictions.csv"
