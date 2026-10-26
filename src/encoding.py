import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import config


def fit_encoder(g0: pd.DataFrame) -> OrdinalEncoder:

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(g0[config.CATEGORICAL_COLUMNS])
    return encoder


def to_xy(df: pd.DataFrame, encoder: OrdinalEncoder) -> tuple[pd.DataFrame, pd.Series]:
    X = df[config.FEATURES].copy()
    X[config.CATEGORICAL_COLUMNS] = encoder.transform(df[config.CATEGORICAL_COLUMNS])
    y = df[config.TARGET]
    return X, y
