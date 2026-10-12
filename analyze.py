"""
analyze.py - PHASE 3: Degradation & Forecasting.         Run:  python analyze.py

Reads results/results.csv (made by run_experiment.py) and produces:

    results/collapse_index.csv          Collapse Index for 6 series x 11 generations
    results/lstm_forecast.csv           LSTM holdout predictions (G9-G10) and forecast (G11-G13)
    results/summary.csv                 one row per ratio x algorithm
    results/findings.txt                the research question answered in plain English
    results/plots/ci_vs_generation.png  Plot 1
    results/plots/lstm_forecast.png     Plot 2
"""

import pandas as pd

import config
from src.collapse import collapse_index_table
from src.forecasting import forecast_table, holdout_mae, train_lstm
from src.plots import plot_collapse_index, plot_lstm_forecast
from src.report import build_summary, write_findings

EXPECTED_RUNS = 2 * (1 + config.N_GENERATIONS * len(config.RATIOS))   # 62


def main() -> None:
    if not config.RESULTS_CSV.exists():
        raise SystemExit("results/results.csv not found - run `python run_experiment.py` first.")
    results = pd.read_csv(config.RESULTS_CSV)
    if len(results) != EXPECTED_RUNS:
        raise SystemExit(f"results.csv has {len(results)} rows, expected {EXPECTED_RUNS}. "
                         "Finish `python run_experiment.py` first.")
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1/4  Collapse Index")
    ci_table = collapse_index_table(results)
    ci_table.to_csv(config.COLLAPSE_INDEX_CSV, index=False)
    print(f"  saved {config.COLLAPSE_INDEX_CSV.relative_to(config.PROJECT_DIR)}")

    print("Step 2/4  LSTM forecast")
    model = train_lstm(ci_table)
    forecasts = forecast_table(model, ci_table)
    forecasts.to_csv(config.FORECAST_CSV, index=False)
    mae = holdout_mae(forecasts)
    print(f"  saved {config.FORECAST_CSV.relative_to(config.PROJECT_DIR)}")

    print("Step 3/4  Plots")
    plot_collapse_index(ci_table, config.PLOTS_DIR / "ci_vs_generation.png")
    plot_lstm_forecast(ci_table, forecasts, config.PLOTS_DIR / "lstm_forecast.png")
    print("  saved results/plots/ci_vs_generation.png and results/plots/lstm_forecast.png")

    print("Step 4/4  Summary table and findings")
    summary = build_summary(results, ci_table, mae)
    summary.round(4).to_csv(config.SUMMARY_CSV, index=False)
    findings = write_findings(results, ci_table, summary, forecasts)
    config.FINDINGS_TXT.write_text(findings + "\n")
    print(f"  saved {config.SUMMARY_CSV.relative_to(config.PROJECT_DIR)} "
          f"and {config.FINDINGS_TXT.relative_to(config.PROJECT_DIR)}")

    print("\n" + summary.round(4).to_string(index=False))
    print("\n" + findings)


if __name__ == "__main__":
    main()
