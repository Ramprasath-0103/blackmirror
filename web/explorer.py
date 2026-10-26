import threading

import pandas as pd

import config
from src.dataset import read_generation, read_test, read_validation
from src.encoding import fit_encoder, to_xy
from src.mixing import make_mix
from src.model import test_metrics, train_random_forest
from src.search import Objective, hill_climbing, simulated_annealing, state_to_params

_cache: dict = {}
_lock = threading.Lock()


def _settings(state) -> dict:
    params = state_to_params(state)
    return {"trees": params["n_estimators"], "depth": params["max_depth"], "split": params["min_samples_split"]}


def _hill_climbing_rounds(log: list, start) -> list:
    seen = {start}
    rounds = []
    for entry in log:
        neighbours = []
        for state, f1 in entry["neighbors"].items():
            neighbours.append({"settings": _settings(state), "f1": f1, "cached": state in seen,
                               "best": state == entry["best"]})
            seen.add(state)
        rounds.append({"settings": _settings(entry["state"]), "f1": entry["f1"], "neighbours": neighbours,
                       "moved": entry["moved"], "evals_used": entry["evals_used"]})
    return rounds


def _annealing_steps(log: list) -> tuple[list, int]:
    """Keep the steps up to the last new evaluation; after that SA only revisits known states."""
    last_new = 0
    for index, entry in enumerate(log):
        if index == 0 or entry["evals_used"] > log[index - 1]["evals_used"]:
            last_new = index
    shown = log[:last_new + 1]
    steps = [{"step": e["step"], "temperature": e["temperature"], "settings": _settings(e["candidate"]),
              "f1": e["f1"], "delta": e["delta"], "probability": e["probability"], "roll": e["roll"],
              "accepted": e["accepted"], "new_best": e["new_best"], "evals_used": e["evals_used"]}
             for e in shown]
    return steps, len(log) - len(shown)


def _progress(start_f1: float, points: list[tuple[int, float]]) -> list[dict]:
    best = start_f1
    curve = [{"evals": 1, "best": best}]
    for evals, f1 in points:
        best = max(best, f1)
        curve.append({"evals": evals, "best": best})
    return curve


def _saved_row(generation: int, ratio: float, algorithm: str) -> dict | None:
    if not config.RESULTS_CSV.exists():
        return None
    results = pd.read_csv(config.RESULTS_CSV, dtype={"max_depth": str}, keep_default_na=False)
    row = results[(results["generation"] == generation) & (results["ratio"] == ratio)
                  & (results["algorithm"] == algorithm)]
    return None if row.empty else row.iloc[0].to_dict()


def _compute(generation: int, ratio: float) -> dict:
    encoder = fit_encoder(read_generation(0))
    X_train, y_train = to_xy(make_mix(generation, ratio), encoder)
    X_val, y_val = to_xy(read_validation(), encoder)
    X_test, y_test = to_xy(read_test(), encoder)

    output = {"generation": generation, "ratio": ratio}
    for name, search in (("HC", hill_climbing), ("SA", simulated_annealing)):
        log = []
        objective = Objective(X_train, y_train, X_val, y_val)
        result = search(objective, log=log)
        model = train_random_forest(result.best_params, X_train, y_train)
        test_f1 = test_metrics(model, X_test, y_test)["f1"]
        start_f1 = objective.cache[config.START_STATE]

        saved = _saved_row(generation, ratio, name)
        matches = saved is not None and (
            int(saved["evals_used"]) == result.evals_used
            and abs(float(saved["val_f1"]) - result.best_val_f1) < 1e-12
            and abs(float(saved["test_f1"]) - test_f1) < 1e-12
        )

        summary = {"settings": _settings(result.best_state), "val_f1": result.best_val_f1,
                   "test_f1": test_f1, "evals_used": result.evals_used,
                   "start_f1": start_f1, "matches_results_csv": matches}
        if name == "HC":
            rounds = _hill_climbing_rounds(log, config.START_STATE)
            points = [(r["evals_used"], max(n["f1"] for n in r["neighbours"])) for r in rounds]
            output["HC"] = {**summary, "rounds": rounds, "progress": _progress(start_f1, points)}
        else:
            steps, frozen = _annealing_steps(log)
            points = [(s["evals_used"], s["f1"]) for s in steps]
            output["SA"] = {**summary, "steps": steps, "frozen_steps": frozen,
                            "total_steps": len(log), "progress": _progress(start_f1, points)}
    return output


def trace(generation: int, ratio: float) -> dict:
    """Run both searches on one training mix and record every decision they make."""
    files = [config.generation_csv(0), config.generation_csv(generation), config.VAL_CSV, config.TEST_CSV]
    key = (generation, ratio, tuple(f.stat().st_mtime for f in files))
    with _lock:
        if key not in _cache:
            _cache[key] = _compute(generation, ratio)
        return _cache[key]
