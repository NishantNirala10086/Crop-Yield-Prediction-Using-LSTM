"""Keras model builders: stacked LSTM (proposed) plus RNN / GRU baselines."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def _build(cell: str, timesteps: int, n_features: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    Rec = {"lstm": layers.LSTM, "gru": layers.GRU, "rnn": layers.SimpleRNN}[cell]
    u1, u2 = config.LSTM_UNITS
    model = models.Sequential([
        layers.Input(shape=(timesteps, n_features)),
        Rec(u1, return_sequences=True),
        layers.Dropout(config.DROPOUT),
        Rec(u2),
        layers.Dropout(config.DROPOUT),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),
    ], name=f"{cell}_yield")
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


def build_lstm(timesteps, n_features): return _build("lstm", timesteps, n_features)
def build_gru(timesteps, n_features): return _build("gru", timesteps, n_features)
def build_rnn(timesteps, n_features): return _build("rnn", timesteps, n_features)


def callbacks():
    from tensorflow.keras import callbacks as cb
    return [cb.EarlyStopping(monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True),
            cb.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, config.PATIENCE // 3))]
