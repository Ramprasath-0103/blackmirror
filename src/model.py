import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config


def train_random_forest(params: dict, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(**params, random_state=config.SEED, n_jobs=-1)
    model.fit(X, y)
    return model


def f1(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    return f1_score(y, model.predict(X), pos_label=1)


def test_metrics(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    predictions = model.predict(X)
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y, predictions, pos_label=1),
        "f1": f1_score(y, predictions, pos_label=1),
    }
