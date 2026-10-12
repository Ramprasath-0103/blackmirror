"""
generation.py - create the synthetic generations (plan Phase 1.4 - 1.5).

    build_metadata(g0)          describe each column's type to SDV
    generate_chain(g0, meta)    G0 -> G1 -> G2 -> ... -> G10

Each generation is produced by a brand-new synthesizer that only ever sees
the previous generation. That is what makes the chain "recursive": any
error SDV makes in G1 is baked into G2, then G3, and so on.
"""

import warnings

import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

import config

TABLE_NAME = "adult"


def build_metadata(g0: pd.DataFrame) -> Metadata:
    """
    Detect the table layout from G0, then set every column type by hand.

    Setting types by hand matters: income is stored as 0/1, so auto-detection
    could call it numerical and SDV would happily generate values like 0.37.
    capital-gain / capital-loss are ~90% zeros and must stay numerical.

    (Written for SDV 1.38. The metadata API changed during the 1.x releases,
    so check the SDV docs if you use a different version.)
    """
    with warnings.catch_warnings():
        # SDV warns that auto-detected metadata should be reviewed - we do
        # exactly that on the next two lines, so the warning is just noise.
        warnings.simplefilter("ignore")
        metadata = Metadata.detect_from_dataframe(g0, table_name=TABLE_NAME, infer_keys=None)

    metadata.update_columns(config.CATEGORICAL_COLUMNS + [config.TARGET], sdtype="categorical")
    metadata.update_columns(config.NUMERICAL_COLUMNS, sdtype="numerical")
    metadata.validate()

    # Saving the metadata is what SDV recommends for replicability, and it
    # lets anyone open data/metadata.json to see the column types we chose.
    metadata.save_to_json(config.METADATA_JSON, mode="overwrite")
    return metadata


def generate_chain(g0: pd.DataFrame, metadata: Metadata) -> list[pd.DataFrame]:
    """
    Return [G1, G2, ..., G10], each with exactly N_ROWS rows.

    For every generation g:
        1. create a fresh GaussianCopulaSynthesizer
        2. fit it on generation g-1 only
        3. sample 6,400 new rows -> generation g

    SDV's defaults round integer columns and keep values inside the min/max
    seen during fitting, so the synthetic values stay realistic.
    """
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
