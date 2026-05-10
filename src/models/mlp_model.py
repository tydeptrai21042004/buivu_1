"""MLP and deep neural-network models."""
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
except Exception:  # pragma: no cover
    Sequential = None


def _require_tf():
    if Sequential is None:
        raise RuntimeError("TensorFlow is required to build neural-network models.")


def build_mlp(input_dim: int):
    _require_tf()
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dropout(0.15),
        Dense(32, activation="relu"),
        Dropout(0.10),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_deep_nn(input_dim: int):
    _require_tf()
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.20),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.15),
        Dense(32, activation="relu"),
        Dropout(0.10),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model
