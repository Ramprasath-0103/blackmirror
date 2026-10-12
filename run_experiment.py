"""
run_experiment.py - PHASE 2: Optimization & Evaluation.     Run:  python run_experiment.py

For every run it:
    1. builds the training mixture  make_mix(generation, ratio)
    2. searches hyperparameters with Hill Climbing or Simulated Annealing (scored on V)
    3. retrains the best forest and evaluates it ONCE on the frozen Test set
    4. appends one row to results/results.csv

The runs are:
    baseline: generation 0, ratio 1.0 (pure real data)          ->  1
    every generation 1..10 x every ratio in {0.9, 0.7, 0.5}     -> 30
    ... each done with both HC and SA                           -> x2 = 62 runs

It is safe to stop and restart: finished runs are skipped, so a crash or a
sleeping laptop only loses the run that was in progress.
"""

import random
import time

import pandas as pd

import config
from src.dataset import read_generation, read_test, read_validation
from src.encoding import fit_encoder, to_xy
from src.mixing import make_mix
from src.model import test_metrics, train_random_forest
from src.search import ALGORITHMS, Objective

RESULT_COLUMNS = [
    "generation", "ratio", "algorithm",
    "n_estimators", "max_depth", "min_samples_split",
    "val_f1", "test_accuracy", "test_precision", "test_recall", "test_f1",
    "evals_used",
]


# ---------------------------------------------------------------------------
# Which runs exist, and which are already done
# ---------------------------------------------------------------------------
def list_runs() -> list[tuple[int, float, str]]:
    """All 62 (generation, ratio, algorithm) combinations, baseline first."""
    mixtures = [(0, 1.0)] + [(g, r) for g in range(1, config.N_GENERATIONS + 1) for r in config.RATIOS]
    return [(g, r, algorithm) for g, r in mixtures for algorithm in ALGORITHMS]


def load_results() -> pd.DataFrame:
    if not config.RESULTS_CSV.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    # pandas treats the text "None" as a missing value by default, which would
    # turn max_depth into floats (8 -> 8.0). Reading it as text keeps it as written.
    return pd.read_csv(config.RESULTS_CSV, dtype={"max_depth": str}, keep_default_na=False)


def finished_runs() -> set[tuple[int, float, str]]:
    done = load_results()
    return set(zip(done["generation"].astype(int), done["ratio"].astype(float), done["algorithm"]))


def append_result(row: dict) -> None:
    """Add one row to results.csv straight away (header only if the file is new)."""
    write_header = not config.RESULTS_CSV.exists()
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
        config.RESULTS_CSV, mode="a", header=write_header, index=False
    )


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run_one(generation: int, ratio: float, algorithm: str, encoder, val_xy, test_xy) -> dict:
    """Search on V, then evaluate the winner once on Test. Returns a results row."""
    X_train, y_train = to_xy(make_mix(generation, ratio), encoder)
    X_val, y_val = val_xy
    X_test, y_test = test_xy

    # Step 2: hyperparameter search (only ever looks at V, never at Test)
    objective = Objective(X_train, y_train, X_val, y_val)
    result = ALGORITHMS[algorithm](objective)

    # Step 3: retrain with the best settings and score on Test exactly once
    model = train_random_forest(result.best_params, X_train, y_train)
    metrics = test_metrics(model, X_test, y_test)

    params = result.best_params
    return {
        "generation": generation,
        "ratio": ratio,
        "algorithm": algorithm,
        "n_estimators": params["n_estimators"],
        "max_depth": "None" if params["max_depth"] is None else params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "val_f1": result.best_val_f1,
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "evals_used": result.evals_used,
    }


# ---------------------------------------------------------------------------
# Checks from the plan's "Phase 2 is done when" list
# ---------------------------------------------------------------------------
def check_results(encoder, val_xy, test_xy) -> None:
    results = load_results()
    expected = len(list_runs())
    print("\nChecks")
    print(f"  rows in results.csv: {len(results)} (expected {expected})")
    print(f"  largest evals_used:  {results['evals_used'].max()} (must be <= {config.EVAL_BUDGET})")

    for algorithm in ALGORITHMS:
        baseline = results[(results["generation"] == 0) & (results["algorithm"] == algorithm)]
        if baseline.empty:
            continue
        f1_base = baseline["test_f1"].item()
        low, high = config.EXPECTED_BASELINE_F1
        verdict = "OK" if low <= f1_base <= high else f"outside the expected {low}-{high}, check encoding/labels"
        print(f"  baseline test F1 ({algorithm}): {f1_base:.3f}  {verdict}")

    # Reproducibility: pick any finished run, do it again, compare the numbers.
    if not results.empty:
        saved = results.sample(n=1, random_state=random.randrange(10_000)).iloc[0]
        g, r, a = int(saved["generation"]), float(saved["ratio"]), saved["algorithm"]
        print(f"  re-running G{g} ratio={r} {a} to check it reproduces ...", end=" ", flush=True)
        again = run_one(g, r, a, encoder, val_xy, test_xy)
        # pandas saves floats at full precision, so comparing as text is exact.
        same = all(str(again[column]) == str(saved[column]) for column in RESULT_COLUMNS)
        print("identical" if same else "DIFFERENT - something is not seeded")


# ---------------------------------------------------------------------------
def main() -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)

    # These never change between runs, so prepare them once.
    encoder = fit_encoder(read_generation(0))
    val_xy = to_xy(read_validation(), encoder)
    test_xy = to_xy(read_test(), encoder)

    runs = list_runs()
    done = finished_runs()
    print(f"{len(runs)} runs in total, {len(done)} already finished.\n")

    for number, (g, r, algorithm) in enumerate(runs, start=1):
        if (g, r, algorithm) in done:
            continue

        start = time.time()
        row = run_one(g, r, algorithm, encoder, val_xy, test_xy)
        append_result(row)

        print(f"[{number:>2}/{len(runs)}] G{g:<2} real={r:.0%}  {algorithm}  "
              f"val F1 {row['val_f1']:.3f}  test F1 {row['test_f1']:.3f}  "
              f"evals {row['evals_used']:>2}  "
              f"(trees={row['n_estimators']}, depth={row['max_depth']}, split={row['min_samples_split']})  "
              f"{time.time() - start:.0f}s")

    check_results(encoder, val_xy, test_xy)


if __name__ == "__main__":
    main()
