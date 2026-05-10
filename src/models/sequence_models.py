"""Sequence-style neural-network models for tabular feature sequences."""
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, GRU, Conv1D, Flatten, Dropout
    from tensorflow.keras.optimizers import Adam
except Exception:  # pragma: no cover
    Sequential = None


def _require_tf():
    if Sequential is None:
        raise RuntimeError("TensorFlow is required to build sequence neural-network models.")


def reshape_for_sequence(X):
    return X.reshape(X.shape[0], X.shape[1], 1)


def build_elman_nn(timesteps: int):
    _require_tf()
    model = Sequential([
        Input(shape=(timesteps, 1)),
        SimpleRNN(24, activation="tanh"),
        Dense(12, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_lstm_nn(timesteps: int):
    _require_tf()
    model = Sequential([
        Input(shape=(timesteps, 1)),
        LSTM(24, activation="tanh"),
        Dense(12, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_gru_nn(timesteps: int):
    _require_tf()
    model = Sequential([
        Input(shape=(timesteps, 1)),
        GRU(24, activation="tanh"),
        Dense(12, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_cnn_1d(timesteps: int):
    _require_tf()
    model = Sequential([
        Input(shape=(timesteps, 1)),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        Conv1D(16, kernel_size=3, activation="relu", padding="same"),
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.15),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model
