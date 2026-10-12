"""
encoding.py - turn text categories into numbers (plan Phase 2.1).

    fit_encoder(g0)          learn a number for every category seen in G0
    to_xy(df, encoder)       split a table into X (encoded features) and y (income)

A Random Forest can only work with numbers. An OrdinalEncoder replaces each
category with an integer code (e.g. "Private" -> 3). For tree models the
order of the codes does not matter, so this simple encoding is enough.
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import config


def fit_encoder(g0: pd.DataFrame) -> OrdinalEncoder:
    """
    Fit the encoder on G0's categorical columns - once, for the whole experiment.

    Fitting once means every run uses the identical encoding. If a synthetic
    generation ever produces a category G0 never had, it gets the code -1
    instead of crashing the program.
    """
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(g0[config.CATEGORICAL_COLUMNS])
    return encoder


def to_xy(df: pd.DataFrame, encoder: OrdinalEncoder) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y): numeric columns unchanged, categorical columns encoded."""
    X = df[config.FEATURES].copy()
    X[config.CATEGORICAL_COLUMNS] = encoder.transform(df[config.CATEGORICAL_COLUMNS])
    y = df[config.TARGET]
    return X, y
