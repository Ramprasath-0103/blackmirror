"""
forecasting.py - predict future collapse with an LSTM (plan Phase 3.2).

    make_windows(values)              [a, b, c, d, e] -> inputs [a,b,c],[b,c,d]  targets d, e
    train_lstm(ci_table)              one small LSTM trained on all 6 series (G0..G8)
    forecast_table(model, ci_table)   holdout check (G9, G10) and future forecast (G11-G13)

How the LSTM is checked honestly:
    - It is trained on generations G0..G8 only.
    - G9 and G10 are "held out": we predict them and compare with the real values.
    - We also compute a NAIVE forecast: "next value = last value". If the LSTM
      is not more accurate than that, it has not learned anything useful,
      and the report says so.
"""

import os

import numpy as np
import pandas as pd

import config
from src.collapse import all_series, series_values

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's start-up messages
import keras  # imported after the line above so the setting takes effect

WINDOW = config.LSTM_WINDOW
LAST_TRAIN = config.LSTM_LAST_TRAIN_GENERATION


def make_windows(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of 3 values along the series; each window predicts the next value."""
    inputs = [values[i:i + WINDOW] for i in range(len(values) - WINDOW)]
    targets = [values[i + WINDOW] for i in range(len(values) - WINDOW)]
    return np.array(inputs), np.array(targets)


def build_lstm() -> keras.Model:
    """LSTM(16) -> Dense(1). Input shape: (3 time steps, 1 value per step)."""
    model = keras.Sequential([
        keras.Input(shape=(WINDOW, 1)),
        keras.layers.LSTM(config.LSTM_UNITS),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(ci_table: pd.DataFrame) -> keras.Model:
    """
    Pool the windows of all 6 series (G0..G8 only) and train one model.

    One series alone gives only 6 windows - far too few to train on.
    Pooling the 6 series gives 6 x 6 = 36 windows.
    """
    keras.utils.set_random_seed(config.SEED)

    all_inputs, all_targets = [], []
    for algorithm, ratio in all_series():
        training_part = series_values(ci_table, algorithm, ratio)[:LAST_TRAIN + 1]
        inputs, targets = make_windows(training_part)
        all_inputs.append(inputs)
        all_targets.append(targets)

    X = np.concatenate(all_inputs)[..., np.newaxis]   # shape (36, 3, 1)
    y = np.concatenate(all_targets)                   # shape (36,)
    print(f"  training on {len(X)} windows for {config.LSTM_EPOCHS} epochs ...")

    model = build_lstm()
    model.fit(X, y, epochs=config.LSTM_EPOCHS, batch_size=config.LSTM_BATCH_SIZE, verbose=0)
    return model


def predict_next(model: keras.Model, last_values: list[float]) -> float:
    """Predict the value that follows the given 3 values."""
    x = np.array(last_values, dtype="float32").reshape(1, WINDOW, 1)
    return float(model(x, training=False)[0, 0])


def forecast_table(model: keras.Model, ci_table: pd.DataFrame) -> pd.DataFrame:
    """
    For every series, return rows of two kinds:

    "holdout"   G9 and G10, predicted from the REAL previous 3 values,
                next to the real value and the naive guess (previous value).
    "forecast"  G11, G12, G13: each prediction is fed back in as input
                to predict the next one (there is no real value to compare to).
    """
    rows = []
    for algorithm, ratio in all_series():
        values = series_values(ci_table, algorithm, ratio)

        for g in range(LAST_TRAIN + 1, len(values)):   # G9, G10
            rows.append({
                "algorithm": algorithm, "ratio": ratio, "generation": g, "kind": "holdout",
                "actual": values[g],
                "lstm": predict_next(model, values[g - WINDOW:g]),
                "naive": values[g - 1],
            })

        history = list(values)
        for step in range(config.N_FORECAST_STEPS):    # G11, G12, G13
            prediction = predict_next(model, history[-WINDOW:])
            history.append(prediction)
            rows.append({
                "algorithm": algorithm, "ratio": ratio, "generation": len(values) + step,
                "kind": "forecast", "actual": np.nan, "lstm": prediction, "naive": np.nan,
            })

    return pd.DataFrame(rows)


def holdout_mae(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error of LSTM and naive forecasts on G9-G10, per series."""
    holdout = forecasts[forecasts["kind"] == "holdout"].copy()
    holdout["lstm_error"] = (holdout["lstm"] - holdout["actual"]).abs()
    holdout["naive_error"] = (holdout["naive"] - holdout["actual"]).abs()
    return (holdout.groupby(["algorithm", "ratio"], sort=False)[["lstm_error", "naive_error"]]
            .mean()
            .rename(columns={"lstm_error": "lstm_mae", "naive_error": "naive_mae"})
            .reset_index())
