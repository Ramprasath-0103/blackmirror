import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

import config

UCI_ADULT_ID = 2


def load_adult() -> pd.DataFrame:
    adult = fetch_ucirepo(id=UCI_ADULT_ID)
    features = adult.data.features
    target = adult.data.targets
    return pd.concat([features, target], axis=1)


def clean_adult(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop(columns=config.DROPPED_COLUMNS).copy()

    for column in config.COLUMNS_WITH_MISSING:
        df[column] = df[column].fillna("Unknown").replace("?", "Unknown")

    label = df[config.TARGET].str.strip().str.rstrip(".")
    df[config.TARGET] = (label == ">50K").astype(int)

    return df[config.ALL_COLUMNS]


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

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

def read_generation(generation: int) -> pd.DataFrame:
    return pd.read_csv(config.generation_csv(generation))


def read_validation() -> pd.DataFrame:
    return pd.read_csv(config.VAL_CSV)


def read_test() -> pd.DataFrame:
    return pd.read_csv(config.TEST_CSV)
