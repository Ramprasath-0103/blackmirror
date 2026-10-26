import warnings

import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

import config

TABLE_NAME = "adult"


def build_metadata(g0: pd.DataFrame) -> Metadata:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metadata = Metadata.detect_from_dataframe(g0, table_name=TABLE_NAME, infer_keys=None)

    metadata.update_columns(config.CATEGORICAL_COLUMNS + [config.TARGET], sdtype="categorical")
    metadata.update_columns(config.NUMERICAL_COLUMNS, sdtype="numerical")
    metadata.validate()

    metadata.save_to_json(config.METADATA_JSON, mode="overwrite")
    return metadata


def generate_chain(g0: pd.DataFrame, metadata: Metadata) -> list[pd.DataFrame]:

    np.random.seed(config.SEED)
    previous = g0
    generations = []

    for g in range(1, config.N_GENERATIONS + 1):
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(previous)
        new_generation = synthesizer.sample(num_rows=config.N_ROWS)
        generations.append(new_generation[config.ALL_COLUMNS])

        positive_rate = new_generation[config.TARGET].mean()
        print(f"  G{g:<2} generated  ({len(new_generation):,} rows, {positive_rate:.1%} >50K)")
        previous = new_generation

    return generations
