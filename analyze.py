import pandas as pd

import config
from src.collapse import collapse_index_table
from src.plots import plot_collapse_index, plot_hc_vs_sa
from src.report import build_summary, write_findings

EXPECTED_RUNS = 2 * (1 + config.N_GENERATIONS * len(config.RATIOS))  

def main() -> None:
    if not config.RESULTS_CSV.exists():
        raise SystemExit("results/results.csv not found - run `python run_experiment.py` first.")
    results = pd.read_csv(config.RESULTS_CSV)
    if len(results) != EXPECTED_RUNS:
        raise SystemExit(f"results.csv has {len(results)} rows, expected {EXPECTED_RUNS}. "
                         "Finish `python run_experiment.py` first.")
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1/3  Collapse Index")
    ci_table = collapse_index_table(results)
    ci_table.to_csv(config.COLLAPSE_INDEX_CSV, index=False)
    print(f"  saved {config.COLLAPSE_INDEX_CSV.relative_to(config.PROJECT_DIR)}")

    print("Step 2/3  Plots")
    plot_collapse_index(ci_table, config.PLOTS_DIR / "ci_vs_generation.png")
    plot_hc_vs_sa(results, config.PLOTS_DIR / "hc_vs_sa.png")
    print("  saved results/plots/ci_vs_generation.png and results/plots/hc_vs_sa.png")

    print("Step 3/3  Summary table and findings")
    summary = build_summary(results, ci_table)
    summary.round(4).to_csv(config.SUMMARY_CSV, index=False)
    findings = write_findings(results, ci_table, summary)
    config.FINDINGS_TXT.write_text(findings + "\n")
    print(f"  saved {config.SUMMARY_CSV.relative_to(config.PROJECT_DIR)} "
          f"and {config.FINDINGS_TXT.relative_to(config.PROJECT_DIR)}")

    print("\n" + summary.round(4).to_string(index=False))
    print("\n" + findings)


if __name__ == "__main__":
    main()
