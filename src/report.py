"""
report.py - the summary table and the written answer (plan Phase 3.3).

    build_summary(results, ci_table, mae)       one row per ratio x algorithm -> summary.csv
    write_findings(results, summary, forecasts) the research question answered in plain English

The research question has two factors, and the findings answer each one:
    Ratio effect      fix the algorithm, compare 90/10 vs 70/30 vs 50/50
    Algorithm effect  fix the ratio, compare Hill Climbing vs Simulated Annealing
"""

import numpy as np
import pandas as pd

import config

CHECKPOINTS = (5, 10)   # the generations reported in the summary table


def build_summary(results: pd.DataFrame, ci_table: pd.DataFrame, mae: pd.DataFrame) -> pd.DataFrame:
    """For each ratio x algorithm: validation F1, test F1 and CI at G5/G10, and forecast errors."""
    rows = []
    for ratio in config.RATIOS:
        for algorithm in config.ALGORITHMS:
            runs = results[(results["algorithm"] == algorithm) & (results["ratio"] == ratio)]
            ci = ci_table[(ci_table["algorithm"] == algorithm) & (ci_table["ratio"] == ratio)]
            ci = ci.set_index("generation")

            row = {
                "ratio": ratio,
                "real/synthetic": config.ratio_label(ratio),
                "algorithm": algorithm,
                "mean_val_f1": runs["val_f1"].mean(),
            }
            for g in CHECKPOINTS:
                row[f"test_f1_G{g}"] = ci.loc[g, "test_f1"]
            for g in CHECKPOINTS:
                row[f"ci_G{g}"] = ci.loc[g, "collapse_index"]
            row["mean_ci_G1_G10"] = ci.loc[1:, "collapse_index"].mean()
            rows.append(row)

    # Not rounded here, so the findings use exact values; analyze.py rounds when saving.
    return pd.DataFrame(rows).merge(mae, on=["algorithm", "ratio"])


def _ratio_effect(summary: pd.DataFrame, ci_table: pd.DataFrame) -> list[str]:
    lines = ["1) RATIO EFFECT - mean Collapse Index over G1-G10 (positive = worse than baseline)"]
    always_worse_with_more_synthetic = True
    for algorithm in config.ALGORITHMS:
        part = summary[summary["algorithm"] == algorithm]          # rows ordered 90/10, 70/30, 50/50
        values = part["mean_ci_G1_G10"].to_numpy()
        labels = part["real/synthetic"].tolist()
        lines.append(f"   {algorithm}: " + ",  ".join(f"{label} {value:+.1%}" for label, value in zip(labels, values)))
        always_worse_with_more_synthetic &= bool(np.all(np.diff(values) > 0))

    if always_worse_with_more_synthetic:
        lines.append("   -> For both algorithms, a larger synthetic share led to more degradation.")
    else:
        lines.append("   -> The Collapse Index does NOT grow steadily with the synthetic share.")
    synthetic = ci_table[ci_table["generation"] > 0]["collapse_index"]
    lines.append(f"   -> Every Collapse Index value (all series, G1-G10) lies between "
                 f"{synthetic.min():+.1%} and {synthetic.max():+.1%}.")
    return lines


def _algorithm_effect(results: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    lines = ["2) ALGORITHM EFFECT - Hill Climbing vs Simulated Annealing at each ratio"]
    for ratio in config.RATIOS:
        part = summary[summary["ratio"] == ratio].set_index("algorithm")
        lines.append(f"   {config.ratio_label(ratio)}:  mean val F1  HC {part.loc['HC', 'mean_val_f1']:.3f}"
                     f" vs SA {part.loc['SA', 'mean_val_f1']:.3f};   "
                     f"mean CI  HC {part.loc['HC', 'mean_ci_G1_G10']:+.1%}"
                     f" vs SA {part.loc['SA', 'mean_ci_G1_G10']:+.1%}")

    # Compare the two algorithms run by run on the 30 synthetic mixtures.
    synthetic = results[results["generation"] > 0]
    paired = synthetic.pivot_table(index=["generation", "ratio"], columns="algorithm", values="val_f1")
    sa_better = int((paired["SA"] > paired["HC"]).sum())
    same = int((paired["SA"] == paired["HC"]).sum())
    hc_better = int((paired["SA"] < paired["HC"]).sum())
    lines.append(f"   -> Over the {len(paired)} synthetic runs, SA found a higher validation F1 than HC in "
                 f"{sa_better}, the same in {same}, and a lower one in {hc_better}.")

    evals = results.groupby("algorithm")["evals_used"].mean()
    lines.append(f"   -> Average evaluations used: HC {evals['HC']:.1f}, SA {evals['SA']:.1f} "
                 f"(budget {config.EVAL_BUDGET}).")
    return lines


def _forecast_check(summary: pd.DataFrame, forecasts: pd.DataFrame) -> list[str]:
    holdout = forecasts[forecasts["kind"] == "holdout"]
    lstm_mae = (holdout["lstm"] - holdout["actual"]).abs().mean()
    naive_mae = (holdout["naive"] - holdout["actual"]).abs().mean()
    series_won = int((summary["lstm_mae"] < summary["naive_mae"]).sum())
    verdict = "beats" if lstm_mae < naive_mae else "does NOT beat"

    future = forecasts[forecasts["kind"] == "forecast"]["lstm"]
    return [
        "3) FORECAST CHECK - LSTM vs naive 'repeat the last value', on held-out G9-G10",
        f"   LSTM MAE {lstm_mae:.4f}  vs  naive MAE {naive_mae:.4f}  "
        f"-> the LSTM {verdict} the naive baseline ({series_won} of {len(summary)} series).",
        f"   LSTM forecasts for G{config.N_GENERATIONS + 1}-G{config.N_GENERATIONS + config.N_FORECAST_STEPS}"
        f" range from {future.min():+.1%} to {future.max():+.1%}.",
        "   Note: the naive baseline is weak on a noisy series, so beating it does not by itself",
        "   prove the LSTM found a real trend - look at lstm_forecast.png before claiming one.",
    ]


def write_findings(results: pd.DataFrame, ci_table: pd.DataFrame,
                   summary: pd.DataFrame, forecasts: pd.DataFrame) -> str:
    """Return the findings as text (also printed and saved by analyze.py)."""
    baseline = results[results["generation"] == 0].set_index("algorithm")["test_f1"]
    lines = [
        "RESEARCH QUESTION: How do the synthetic-data ratio and the optimization algorithm",
        "affect predictive performance and degradation across generations?",
        "",
        "Baseline (G0, 100% real data): test F1 = "
        + ", ".join(f"{baseline[a]:.3f} ({a})" for a in config.ALGORITHMS),
        "",
        *_ratio_effect(summary, ci_table),
        "",
        *_algorithm_effect(results, summary),
        "",
        *_forecast_check(summary, forecasts),
    ]
    return "\n".join(lines)
