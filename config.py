from pathlib import Path

SEED = 42                  
N_SUBSAMPLE = 10_000       
N_GENERATIONS = 10         
RATIOS = [0.9, 0.7, 0.5]   
EVAL_BUDGET = 40           
TARGET = "income"         


N_TEST = 2_000             
N_VAL = 1_600             
N_ROWS = 6_400             


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

DROPPED_COLUMNS = ["fnlwgt", "education", "capital-gain", "capital-loss"]

COLUMNS_WITH_MISSING = ["workclass", "occupation", "native-country"]

GRID = {
    "n_estimators":      [50, 100, 150, 200, 250, 300],
    "max_depth":         [2, 4, 6, 8, 10, None],
    "min_samples_split": [2, 4, 6, 8, 10],
}

START_STATE = (1, 1, 0)    

SA_START_TEMPERATURE = 0.05
SA_COOLING_RATE = 0.9      
SA_MAX_STEPS = 1_000

ALGORITHMS = ["HC", "SA"]  


EXPECTED_BASELINE_F1 = (0.55, 0.70)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

TEST_CSV = DATA_DIR / "test.csv"
VAL_CSV = DATA_DIR / "val.csv"
METADATA_JSON = DATA_DIR / "metadata.json"   

RESULTS_CSV = RESULTS_DIR / "results.csv"
COLLAPSE_INDEX_CSV = RESULTS_DIR / "collapse_index.csv"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
FINDINGS_TXT = RESULTS_DIR / "findings.txt"


def generation_csv(generation: int) -> Path:
   
    return DATA_DIR / f"G{generation}.csv"


def ratio_label(ratio: float) -> str:
    real = round(ratio * 100)
    return f"{real}/{100 - real}"
