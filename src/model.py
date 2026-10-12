"""
model.py - the Random Forest and how it is scored (plan Phase 2.2 and 2.5).

    train_random_forest(params, X, y)     fit a forest with the given hyperparameters
    test_metrics(model, X, y)             accuracy, precision, recall and F1

How F1 is calculated (positive class = 1 = ">50K"):
    Precision = TP / (TP + FP)      of the people we called >50K, how many really are?
    Recall    = TP / (TP + FN)      of the people who really are >50K, how many did we find?
    F1        = 2 * Precision * Recall / (Precision + Recall)
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config


def train_random_forest(params: dict, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Fit a Random Forest. The fixed random_state means the same params and
    data always give the same model, so the search is deterministic.
    """
    model = RandomForestClassifier(**params, random_state=config.SEED, n_jobs=-1)
    model.fit(X, y)
    return model


def f1(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    """F1 score for the >50K class."""
    return f1_score(y, model.predict(X), pos_label=1)


def test_metrics(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """All four metrics the plan reports, computed from one set of predictions."""
    predictions = model.predict(X)
    return {
        "accuracy": accuracy_score(y, predictions),
        # zero_division=0: if the model never predicts >50K, report 0 instead of a warning
        "precision": precision_score(y, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y, predictions, pos_label=1),
        "f1": f1_score(y, predictions, pos_label=1),
    }
