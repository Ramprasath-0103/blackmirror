
import numpy as np
import pandas as pd

import config

CHECKPOINTS = (5, 10)   


def build_summary(results: pd.DataFrame, ci_table: pd.DataFrame) -> pd.DataFrame:

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
                "mean_evals_used": runs["evals_used"].mean(),
            }
            for g in CHECKPOINTS:
                row[f"test_f1_G{g}"] = ci.loc[g, "test_f1"]
            for g in CHECKPOINTS:
                row[f"ci_G{g}"] = ci.loc[g, "collapse_index"]
            row["mean_ci_G1_G10"] = ci.loc[1:, "collapse_index"].mean()
            rows.append(row)

   
    return pd.DataFrame(rows)


def _ratio_effect(summary: pd.DataFrame, ci_table: pd.DataFrame) -> list[str]:
    lines = ["1) RATIO EFFECT - mean Collapse Index over G1-G10 (positive = worse than baseline)"]
    always_worse_with_more_synthetic = True
    for algorithm in config.ALGORITHMS:
        part = summary[summary["algorithm"] == algorithm]        
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


def _algorithm_effect(results: pd.DataFrame, ci_table: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    lines = ["2) ALGORITHM EFFECT - Hill Climbing vs Simulated Annealing at each ratio"]
    for ratio in config.RATIOS:
        part = summary[summary["ratio"] == ratio].set_index("algorithm")
        lines.append(f"   {config.ratio_label(ratio)}:  mean val F1  HC {part.loc['HC', 'mean_val_f1']:.3f}"
                     f" vs SA {part.loc['SA', 'mean_val_f1']:.3f};   "
                     f"mean CI  HC {part.loc['HC', 'mean_ci_G1_G10']:+.1%}"
                     f" vs SA {part.loc['SA', 'mean_ci_G1_G10']:+.1%}")


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


    hc = results[results["algorithm"] == "HC"]
    stuck = int((hc["evals_used"] < config.EVAL_BUDGET).sum())
    lines.append(f"   -> HC stopped at a local optimum (with budget left over) in {stuck} of {len(hc)} runs.")


    worst = ci_table[ci_table["generation"] > 0].nlargest(1, "collapse_index").iloc[0]
    other = "SA" if worst["algorithm"] == "HC" else "HC"
    other_ci = ci_table[(ci_table["algorithm"] == other) & (ci_table["ratio"] == worst["ratio"])
                        & (ci_table["generation"] == worst["generation"])]["collapse_index"].item()
    lines.append(f"   -> The largest Collapse Index ({worst['collapse_index']:+.1%}) is {worst['algorithm']} at "
                 f"G{int(worst['generation'])} {config.ratio_label(worst['ratio'])}; {other} on the same "
                 f"training data scored {other_ci:+.1%}.")
    if other_ci < worst["collapse_index"]:
        gap = worst["collapse_index"] - other_ci
        lines.append(f"      Same training data, so about {gap * 100:.1f} points of that spike come from the "
                     f"search, not the data.")
    return lines


def write_findings(results: pd.DataFrame, ci_table: pd.DataFrame, summary: pd.DataFrame) -> str:
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
        *_algorithm_effect(results, ci_table, summary),
    ]
    return "\n".join(lines)
