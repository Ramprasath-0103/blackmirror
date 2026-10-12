"""
collapse.py - measure how much performance was lost (plan Phase 3.1).

    collapse_index_table(results)          the Collapse Index for all 6 series x 11 generations
    series_values(table, algorithm, ratio) one series as a plain list [CI_0, ..., CI_10]

The Collapse Index (CI) of generation g is the relative loss in test F1
compared with training on 100% real data:

    CI_g = (F1_base - F1_g) / F1_base

    CI = 0.10  -> 10% worse than the baseline
    CI = 0     -> no change
    CI < 0     -> slightly BETTER than the baseline (kept as-is, not clipped to 0)

A "series" is one ratio + one algorithm, e.g. "70/30 with Simulated Annealing".
F1_base always comes from the SAME algorithm, so a difference between HC
and SA is never mistaken for collapse.
"""

import pandas as pd

import config


def all_series() -> list[tuple[str, float]]:
    """The 6 (algorithm, ratio) pairs, in a fixed order."""
    return [(algorithm, ratio) for algorithm in config.ALGORITHMS for ratio in config.RATIOS]


def collapse_index_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long table with one row per (algorithm, ratio, generation).

    Generation 0 is the shared baseline, so every series starts at CI = 0.
    """
    rows = []
    for algorithm, ratio in all_series():
        runs = results[results["algorithm"] == algorithm]
        f1_base = runs.loc[runs["generation"] == 0, "test_f1"].item()

        series = runs[(runs["ratio"] == ratio) | (runs["generation"] == 0)]
        for _, run in series.sort_values("generation").iterrows():
            rows.append({
                "algorithm": algorithm,
                "ratio": ratio,
                "generation": int(run["generation"]),
                "test_f1": run["test_f1"],
                "collapse_index": (f1_base - run["test_f1"]) / f1_base,
            })
    return pd.DataFrame(rows)


def series_values(table: pd.DataFrame, algorithm: str, ratio: float) -> list[float]:
    """The Collapse Index of one series, ordered G0, G1, ..., G10."""
    series = table[(table["algorithm"] == algorithm) & (table["ratio"] == ratio)]
    return series.sort_values("generation")["collapse_index"].tolist()
