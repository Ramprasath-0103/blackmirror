"""
prepare_data.py - PHASE 1: Data & Generation.         Run:  python prepare_data.py

What it does, step by step:
    1. Download the Adult Income dataset from UCI.
    2. Clean it (13 columns, no missing values, income is 0/1).
    3. Subsample 10,000 rows and split them into G0 / V / Test.
    4. Build the SDV metadata.
    5. Generate G1 ... G10, each from the previous generation.
    6. Check the output files against the plan's "Phase 1 is done when" list.

Files written to data/:
    G0.csv       6,400 real rows     (training data / seed for SDV)
    val.csv      1,600 real rows     (scored by the hyperparameter search)
    test.csv     2,000 real rows     (final evaluation only)
    G1..G10.csv  6,400 synthetic rows each
"""

import hashlib

import pandas as pd

import config
from src.dataset import clean_adult, load_adult, split_dataset
from src.generation import build_metadata, generate_chain


def save(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False)
    print(f"  saved {path.relative_to(config.PROJECT_DIR)}  ({len(df):,} rows)")


def check_outputs() -> None:
    """Re-read every file and confirm it matches the plan's checklist."""
    print("\nStep 6/6  Checking the files")

    files = {"test": config.TEST_CSV, "val": config.VAL_CSV}
    for g in range(config.N_GENERATIONS + 1):
        files[f"G{g}"] = config.generation_csv(g)

    problems = []
    print(f"  {'file':<6} {'rows':>6} {'>50K':>7}   fingerprint")
    for name, path in files.items():
        df = pd.read_csv(path)

        if list(df.columns) != config.ALL_COLUMNS:
            problems.append(f"{name}: columns are not the expected 13")
        if df.isna().any().any():
            problems.append(f"{name}: contains NaN values")
        if not set(df[config.TARGET].unique()) <= {0, 1}:
            problems.append(f"{name}: income contains values other than 0/1")
        if name.startswith("G") and len(df) != config.N_ROWS:
            problems.append(f"{name}: has {len(df)} rows, expected {config.N_ROWS}")

        # A short fingerprint of the file's bytes. Run the script twice:
        # identical fingerprints mean identical files (reproducibility check).
        fingerprint = hashlib.md5(path.read_bytes()).hexdigest()[:10]
        print(f"  {name:<6} {len(df):>6,} {df[config.TARGET].mean():>7.1%}   {fingerprint}")

    if problems:
        print("\n  PROBLEMS FOUND:")
        for problem in problems:
            print(f"   - {problem}")
        raise SystemExit(1)
    print("\n  All checks passed: 13 columns, no NaN, income in {0, 1}, 6,400 rows per generation.")
    print("  Watch the >50K column: if it drifts away from G0, that partly explains later F1 changes.")


def main() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)

    print("Step 1/6  Downloading Adult Income from UCI")
    raw = load_adult()
    print(f"  {len(raw):,} rows, {raw.shape[1]} columns")

    print("Step 2/6  Cleaning")
    clean = clean_adult(raw)
    print(f"  {clean.shape[1]} columns left (12 features + {config.TARGET})")

    print("Step 3/6  Subsampling and splitting")
    g0, val, test = split_dataset(clean)
    save(test, config.TEST_CSV)
    save(val, config.VAL_CSV)
    save(g0, config.generation_csv(0))

    print("Step 4/6  Building SDV metadata")
    metadata = build_metadata(g0)
    print(f"  saved {config.METADATA_JSON.relative_to(config.PROJECT_DIR)}")

    print(f"Step 5/6  Generating G1 ... G{config.N_GENERATIONS}")
    for g, generation in enumerate(generate_chain(g0, metadata), start=1):
        generation.to_csv(config.generation_csv(g), index=False)
    print(f"  saved data/G1.csv ... data/G{config.N_GENERATIONS}.csv")

    check_outputs()


if __name__ == "__main__":
    main()
