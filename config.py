"""
config.py - every setting the project uses, in one place.

All three phases import from this file. Treat it as the "contract" between
team members: if a seed, a column name or a file path changes here, it
changes for everyone at the same time, so results stay comparable.

If you want to experiment (for example, a different budget or more
generations), this is the only file you should need to edit.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Core experiment constants (from the plan's "Shared setup")
# ---------------------------------------------------------------------------
SEED = 42                  # one seed for every random step -> reproducible runs
N_SUBSAMPLE = 10_000       # rows taken from the full Adult dataset
N_GENERATIONS = 10         # synthetic generations G1 ... G10
RATIOS = [0.9, 0.7, 0.5]   # share of REAL rows in each training mixture
EVAL_BUDGET = 40           # max unique model evaluations per search run
TARGET = "income"          # 0 = <=50K, 1 = >50K

# ---------------------------------------------------------------------------
# Split sizes (Phase 1.3)
# ---------------------------------------------------------------------------
N_TEST = 2_000             # frozen test set, used once per run at the very end
N_VAL = 1_600              # validation set, scored by Hill Climbing / Sim. Annealing
N_ROWS = 6_400             # rows in G0 and in every synthetic generation

# ---------------------------------------------------------------------------
# Columns after cleaning (Phase 1.2): 10 features + the target = 11 columns
# ---------------------------------------------------------------------------
NUMERICAL_COLUMNS = [
    "age",
    "education-num",
    "hours-per-week",
]
CATEGORICAL_COLUMNS = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
FEATURES = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
ALL_COLUMNS = FEATURES + [TARGET]

# Columns that are dropped during cleaning, and why:
#   fnlwgt        a census sampling weight - describes how a person was sampled, not the person
#   education     an exact copy of education-num
#   capital-gain  ~92% zeros, and capital-loss ~95% zeros. SDV's GaussianCopula cannot copy
#   capital-loss  that spike at zero, so synthetic rows get non-zero values and become trivially
#                 recognisable. The forest then just sets them aside, which hides the effect we
#                 want to measure. (README section 3 shows the evidence.)
DROPPED_COLUMNS = ["fnlwgt", "education", "capital-gain", "capital-loss"]
# Columns where missing values appear as NaN or "?"
COLUMNS_WITH_MISSING = ["workclass", "occupation", "native-country"]

# ---------------------------------------------------------------------------
# Hyperparameter search (Phase 2.2 - 2.4)
# ---------------------------------------------------------------------------
# A "state" is a triple of indices into these three lists.
GRID = {
    "n_estimators":      [50, 100, 150, 200, 250, 300],
    "max_depth":         [2, 4, 6, 8, 10, None],
    "min_samples_split": [2, 4, 6, 8, 10],
}
START_STATE = (1, 1, 0)    # -> n_estimators=100, max_depth=4, min_samples_split=2

SA_START_TEMPERATURE = 0.05
SA_COOLING_RATE = 0.9      # temperature is multiplied by this after every step
# Safety net: if SA ends up surrounded by neighbours it has already scored,
# no new evaluation happens and the budget never runs out. This cap makes
# sure the loop always ends. It is far above what a normal run needs.
SA_MAX_STEPS = 1_000

ALGORITHMS = ["HC", "SA"]  # short names written to results.csv

# Sanity check for the baseline (100% real data) test F1. A value far outside
# this range usually means an encoding or label bug. Without the two capital
# columns the baseline is lower than the plan's 0.65-0.70, which assumed them.
EXPECTED_BASELINE_F1 = (0.55, 0.70)

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

TEST_CSV = DATA_DIR / "test.csv"
VAL_CSV = DATA_DIR / "val.csv"
METADATA_JSON = DATA_DIR / "metadata.json"   # SDV column types, for reference

RESULTS_CSV = RESULTS_DIR / "results.csv"
COLLAPSE_INDEX_CSV = RESULTS_DIR / "collapse_index.csv"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
FINDINGS_TXT = RESULTS_DIR / "findings.txt"


def generation_csv(generation: int) -> Path:
    """Path of the file for one generation, e.g. data/G0.csv or data/G7.csv."""
    return DATA_DIR / f"G{generation}.csv"


def ratio_label(ratio: float) -> str:
    """Human-readable real/synthetic split, e.g. 0.7 -> "70/30"."""
    real = round(ratio * 100)
    return f"{real}/{100 - real}"
