import pandas as pd

import config
from src.dataset import read_generation


def make_mix(generation: int, real_ratio: float) -> pd.DataFrame:

    n_real = round(real_ratio * config.N_ROWS)
    n_synthetic = config.N_ROWS - n_real

    real_rows = read_generation(0).sample(n=n_real, random_state=config.SEED)
    synthetic_rows = read_generation(generation).sample(n=n_synthetic, random_state=config.SEED)

    mixture = pd.concat([real_rows, synthetic_rows])
    return mixture.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
