import numpy as np
import pandas as pd

import config
from src.collapse import collapse_index_table
from src.report import build_summary
from web.jobs import PHASES

EXPECTED_RUNS = 2 * (1 + config.N_GENERATIONS * len(config.RATIOS))

COLUMN_DESCRIPTIONS = {
    "age": "Age in years",
    "education-num": "Education level, 1-16 (9 = high school, 13 = bachelor's, 16 = doctorate)",
    "hours-per-week": "Hours worked per week",
    "workclass": "Employer type (private, self-employed, government ...)",
    "marital-status": "Marital status",
    "occupation": "Job",
    "relationship": "Role in the household (husband, wife, own child ...)",
    "race": "Race",
    "sex": "Sex",
    "native-country": "Country of birth",
    "income": "The answer: 1 = earns more than $50K a year, 0 = $50K or less",
}

DROP_REASONS = {
    "fnlwgt": "A census sampling weight. It describes how the person was sampled, not the person.",
    "education": "An exact copy of education-num.",
    "capital-gain": "About 92% of people have 0. SDV cannot copy that spike at zero, so synthetic rows "
                    "become easy to spot and the forest simply sets them aside.",
    "capital-loss": "About 95% of people have 0. Same problem as capital-gain.",
}

SPLIT_ROLES = {
    "G0": "Seeds SDV, the real part of every mix, the baseline",
    "V": "Scores settings during the search",
    "Test": "The final score, used once per run",
}


def _results() -> pd.DataFrame | None:
    if not config.RESULTS_CSV.exists():
        return None
    return pd.read_csv(config.RESULTS_CSV, dtype={"max_depth": str}, keep_default_na=False)


def _clean(records: list[dict]) -> list[dict]:
    """Turn numpy numbers into plain Python ones so they can be sent as JSON."""
    return [{key: (value.item() if isinstance(value, np.generic) else value) for key, value in row.items()}
            for row in records]


def _data_ready() -> bool:
    files = [config.generation_csv(g) for g in range(config.N_GENERATIONS + 1)]
    return all(f.exists() for f in files + [config.VAL_CSV, config.TEST_CSV])


def status() -> dict:
    results = _results()
    runs_done = 0 if results is None else len(results)
    outputs = [config.COLLAPSE_INDEX_CSV, config.SUMMARY_CSV, config.FINDINGS_TXT]
    analysed = (results is not None and all(f.exists() for f in outputs)
                and min(f.stat().st_mtime for f in outputs) >= config.RESULTS_CSV.stat().st_mtime)

    details = {
        "prepare": (_data_ready(), "G0-G10, validation and test files are in data/"
                    if _data_ready() else "No data yet"),
        "experiment": (runs_done == EXPECTED_RUNS, f"{runs_done} of {EXPECTED_RUNS} runs in results.csv"),
        "analyze": (analysed, "Summary, findings and plots are up to date"
                    if analysed else "Not run since the latest results"),
    }
    phases = [{"id": key, "title": info["title"], "script": info["script"],
               "done": details[key][0], "detail": details[key][1]} for key, info in PHASES.items()]
    return {"phases": phases, "runs_done": runs_done, "runs_expected": EXPECTED_RUNS}


def settings() -> dict:
    return {
        "seed": config.SEED, "generations": config.N_GENERATIONS, "ratios": config.RATIOS,
        "ratio_labels": [config.ratio_label(r) for r in config.RATIOS],
        "budget": config.EVAL_BUDGET, "grid_size": int(np.prod([len(v) for v in config.GRID.values()])),
        "grid": config.GRID, "start_temperature": config.SA_START_TEMPERATURE,
        "cooling_rate": config.SA_COOLING_RATE, "features": len(config.FEATURES),
    }


def overview() -> dict:
    results = _results()
    if results is None or len(results) != EXPECTED_RUNS:
        return {"ready": False, "runs_done": 0 if results is None else len(results),
                "runs_expected": EXPECTED_RUNS}

    ci_table = collapse_index_table(results)
    summary = build_summary(results, ci_table)
    synthetic = results[results["generation"] > 0]
    paired = synthetic.pivot_table(index=["generation", "ratio"], columns="algorithm", values="val_f1")
    hc = results[results["algorithm"] == "HC"]

    by_ratio = results.groupby("ratio")["test_f1"].mean()
    test_f1_by_ratio = [{"ratio": r, "label": "100% real" if r == 1.0 else config.ratio_label(r),
                         "f1": float(by_ratio[r])} for r in [1.0] + config.RATIOS]

    worst = ci_table[ci_table["generation"] > 0].nlargest(1, "collapse_index").iloc[0]
    other = "SA" if worst["algorithm"] == "HC" else "HC"
    other_ci = ci_table[(ci_table["algorithm"] == other) & (ci_table["ratio"] == worst["ratio"])
                        & (ci_table["generation"] == worst["generation"])]["collapse_index"].item()

    monotone = all(bool(np.all(np.diff(summary[summary["algorithm"] == a]["mean_ci_G1_G10"].to_numpy()) > 0))
                   for a in config.ALGORITHMS)
    generation_means = ci_table[ci_table["generation"] > 0].groupby("generation")["collapse_index"].mean()

    return {
        "ready": True,
        "runs_done": len(results),
        "runs_expected": EXPECTED_RUNS,
        "budget": config.EVAL_BUDGET,
        "baseline": results[results["generation"] == 0].set_index("algorithm")["test_f1"].to_dict(),
        "test_f1_by_ratio": test_f1_by_ratio,
        "series": _clean(summary.to_dict("records")),
        "ratio_effect_monotone": monotone,
        "ci_range": [float(ci_table[ci_table["generation"] > 0]["collapse_index"].min()),
                     float(ci_table[ci_table["generation"] > 0]["collapse_index"].max())],
        "generation_means": [{"generation": int(g), "mean_ci": float(v)} for g, v in generation_means.items()],
        "algorithm": {
            "sa_higher": int((paired["SA"] > paired["HC"]).sum()),
            "same": int((paired["SA"] == paired["HC"]).sum()),
            "hc_higher": int((paired["SA"] < paired["HC"]).sum()),
            "total": len(paired),
            "mean_evals": results.groupby("algorithm")["evals_used"].mean().to_dict(),
            "hc_local_optima": int((hc["evals_used"] < config.EVAL_BUDGET).sum()),
            "hc_runs": len(hc),
        },
        "largest": {"algorithm": worst["algorithm"], "generation": int(worst["generation"]),
                    "ratio_label": config.ratio_label(worst["ratio"]), "ci": float(worst["collapse_index"]),
                    "other_algorithm": other, "other_ci": float(other_ci)},
        "findings": config.FINDINGS_TXT.read_text() if config.FINDINGS_TXT.exists() else None,
    }


def results() -> dict:
    results = _results()
    if results is None or len(results) != EXPECTED_RUNS:
        return {"ready": False}
    ci_table = collapse_index_table(results)
    summary = build_summary(results, ci_table)
    return {
        "ready": True,
        "runs": _clean(results.to_dict("records")),
        "collapse": _clean(ci_table.to_dict("records")),
        "summary": _clean(summary.to_dict("records")),
    }


def dataset(generation: int) -> dict:
    columns = [{"name": c, "type": "number" if c in config.NUMERICAL_COLUMNS else "category",
                "description": COLUMN_DESCRIPTIONS.get(c, "")} for c in config.ALL_COLUMNS]
    dropped = [{"name": c, "reason": DROP_REASONS.get(c, "")} for c in config.DROPPED_COLUMNS]
    payload = {"ready": _data_ready(), "columns": columns, "dropped": dropped,
               "subsample": config.N_SUBSAMPLE, "generation": generation}
    if not payload["ready"]:
        return payload

    split_files = {"G0": config.generation_csv(0), "V": config.VAL_CSV, "Test": config.TEST_CSV}
    splits = []
    for name, path in split_files.items():
        frame = pd.read_csv(path)
        splits.append({"name": name, "rows": len(frame), "positives": int(frame[config.TARGET].sum()),
                       "rate": float(frame[config.TARGET].mean()), "role": SPLIT_ROLES[name]})

    generations = []
    for g in range(config.N_GENERATIONS + 1):
        frame = pd.read_csv(config.generation_csv(g))
        generations.append({"generation": g, "rows": len(frame), "rate": float(frame[config.TARGET].mean())})

    payload["splits"] = splits
    payload["generations"] = generations
    payload["sample_real"] = _clean(pd.read_csv(config.generation_csv(0)).head(6).to_dict("records"))
    payload["sample_synthetic"] = _clean(pd.read_csv(config.generation_csv(generation)).head(6).to_dict("records"))
    return payload
