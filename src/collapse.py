import pandas as pd

import config


def all_series() -> list[tuple[str, float]]:

    return [(algorithm, ratio) for algorithm in config.ALGORITHMS for ratio in config.RATIOS]


def collapse_index_table(results: pd.DataFrame) -> pd.DataFrame:

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
    series = table[(table["algorithm"] == algorithm) & (table["ratio"] == ratio)]
    return series.sort_values("generation")["collapse_index"].tolist()
