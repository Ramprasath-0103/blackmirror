"""
dataset.py - get the real data ready (plan Phase 1.1 - 1.3).

    load_adult()          download the official UCI copy of Adult Income
    clean_adult(df)       drop/repair columns so we end with 13 clean columns
    split_dataset(df)     subsample 10,000 rows and split into G0 / V / Test
    read_generation(g)    read data/G{g}.csv back from disk
    read_validation()     read data/val.csv
    read_test()           read data/test.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

import config

UCI_ADULT_ID = 2


def load_adult() -> pd.DataFrame:
    """Download Adult from the UCI repository and return features + target together."""
    adult = fetch_ucirepo(id=UCI_ADULT_ID)
    features = adult.data.features
    target = adult.data.targets
    return pd.concat([features, target], axis=1)


def clean_adult(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned copy with 10 features + income (11 columns).

    - Four columns are dropped (config.DROPPED_COLUMNS explains each one):
      fnlwgt, education, capital-gain and capital-loss.
    - Missing values appear as both NaN and "?"; both become the category
      "Unknown" so that no rows are lost.
    - The label appears as "<=50K", "<=50K.", ">50K" and ">50K.". We remove the
      trailing period and map it to 0/1, with 1 meaning ">50K".
    """
    df = df.drop(columns=config.DROPPED_COLUMNS).copy()

    for column in config.COLUMNS_WITH_MISSING:
        df[column] = df[column].fillna("Unknown").replace("?", "Unknown")

    label = df[config.TARGET].str.strip().str.rstrip(".")
    df[config.TARGET] = (label == ">50K").astype(int)

    return df[config.ALL_COLUMNS]


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Subsample and split the clean data into three disjoint sets.

        full data --(stratified)--> 10,000 rows
        10,000    --> Test 2,000  +  Dev 8,000
        Dev 8,000 --> V 1,600     +  G0 6,400

    "Stratified" means every set keeps the same share of >50K rows (~24%),
    so class imbalance can never explain a difference between results.

    Returns (g0, val, test).
    """
    subsample, _ = train_test_split(
        df,
        train_size=config.N_SUBSAMPLE,
        stratify=df[config.TARGET],
        random_state=config.SEED,
    )
    dev, test = train_test_split(
        subsample,
        test_size=config.N_TEST,
        stratify=subsample[config.TARGET],
        random_state=config.SEED,
    )
    g0, val = train_test_split(
        dev,
        test_size=config.N_VAL,
        stratify=dev[config.TARGET],
        random_state=config.SEED,
    )
    return (
        g0.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Reading the saved files back (used by Phases 2 and 3)
# ---------------------------------------------------------------------------
def read_generation(generation: int) -> pd.DataFrame:
    """Read G0 (real) or G1..G10 (synthetic) from the data folder."""
    return pd.read_csv(config.generation_csv(generation))


def read_validation() -> pd.DataFrame:
    return pd.read_csv(config.VAL_CSV)


def read_test() -> pd.DataFrame:
    return pd.read_csv(config.TEST_CSV)
