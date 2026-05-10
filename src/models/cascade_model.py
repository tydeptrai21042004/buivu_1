"""Cascade Forward Neural Network model.

This imitates the paper's Cascade Neural Network idea using Keras skip/cascade
connections from the input to later hidden layers and output.
"""
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
    from tensorflow.keras.optimizers import Adam
except Exception:  # pragma: no cover
    Model = None


def build_cascade_forward_nn(input_dim: int):
    if Model is None:
        raise RuntimeError("TensorFlow is required to build Cascade Forward Neural Network.")

    inp = Input(shape=(input_dim,))

    h1 = Dense(64, activation="relu")(inp)
    h1 = Dropout(0.15)(h1)

    c1 = Concatenate()([inp, h1])

    h2 = Dense(32, activation="relu")(c1)
    h2 = Dropout(0.10)(h2)

    c2 = Concatenate()([inp, h1, h2])

    h3 = Dense(16, activation="relu")(c2)
    c3 = Concatenate()([inp, h1, h2, h3])

    out = Dense(1, activation="linear")(c3)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model
