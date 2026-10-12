"""
mixing.py - build a training set from real + synthetic rows (plan Phase 1.6).

    make_mix(g, r)    r share of real rows from G0, the rest from generation g

Mixtures are built on demand instead of saved to disk: it saves space and
keeps the mixing logic in one place.
"""

import pandas as pd

import config
from src.dataset import read_generation


def make_mix(generation: int, real_ratio: float) -> pd.DataFrame:
    """
    Return a shuffled training set of N_ROWS (6,400) rows.

    Example: make_mix(3, 0.7) = 4,480 real rows from G0 + 1,920 synthetic rows from G3.
    The baseline is make_mix(0, 1.0), which is simply all of G0.

    Because random_state is fixed, a given ratio always picks the SAME real
    rows, whatever the generation. The only thing that changes between
    generations is where the synthetic rows come from - which is exactly
    the effect we want to measure.
    """
    n_real = round(real_ratio * config.N_ROWS)
    n_synthetic = config.N_ROWS - n_real

    real_rows = read_generation(0).sample(n=n_real, random_state=config.SEED)
    synthetic_rows = read_generation(generation).sample(n=n_synthetic, random_state=config.SEED)

    mixture = pd.concat([real_rows, synthetic_rows])
    # Shuffle so the real rows are not all at the top.
    return mixture.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
