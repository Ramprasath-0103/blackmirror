

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_DIR / "results" / "results.csv"
PLOT_DIR = Path(__file__).resolve().parent / "plots"

RATIOS = (0.9, 0.7, 0.5)
ALGORITHMS = ("HC", "SA")
GENERATIONS = tuple(range(11))
RATIO_COLORS = {0.9: "#2a78d6", 0.7: "#eb6834", 0.5: "#1baf7a"}
ALGORITHM_NAMES = {"HC": "Hill Climbing", "SA": "Simulated Annealing"}
INK = "#202124"
SECONDARY = "#5f6368"
GRID = "#e3e6e8"
BACKGROUND = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "axes.edgecolor": "#c8cdd1",
    "axes.labelcolor": SECONDARY,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "text.color": INK,
    "font.size": 10,
    "legend.frameon": False,
    "savefig.facecolor": BACKGROUND,
})


def load_results() -> pd.DataFrame:
    """Load and check the complete experiment output before plotting."""
    if not RESULTS_FILE.is_file():
        raise FileNotFoundError(
            f"{RESULTS_FILE} not found. Run `python run_experiment.py` first."
        )

    results = pd.read_csv(RESULTS_FILE)
    required = {
        "generation", "ratio", "algorithm", "val_f1", "test_f1", "evals_used",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results.csv is missing required columns: {sorted(missing)}")

    results = results.copy()
    for column in ("generation", "ratio", "val_f1", "test_f1", "evals_used"):
        results[column] = pd.to_numeric(results[column], errors="raise")

    key_columns = ["generation", "ratio", "algorithm"]
    if results.duplicated(key_columns).any():
        raise ValueError("results.csv contains duplicate generation/ratio/algorithm runs.")
    if len(results) != 62:
        raise ValueError(
            f"Expected 62 completed runs in results.csv; found {len(results)}. "
            "Finish `python run_experiment.py` before plotting."
        )
    if set(results["algorithm"]) != set(ALGORITHMS):
        raise ValueError("Expected results for both HC and SA.")

    for algorithm in ALGORITHMS:
        runs = results[results["algorithm"] == algorithm]
        baseline = runs[(runs["generation"] == 0) & (runs["ratio"] == 1.0)]
        if len(baseline) != 1:
            raise ValueError(f"Expected one 100% real-data baseline for {algorithm}.")
        for generation in range(1, 11):
            for ratio in RATIOS:
                match = runs[
                    (runs["generation"] == generation) & (runs["ratio"] == ratio)
                ]
                if len(match) != 1:
                    raise ValueError(
                        f"Expected one {algorithm} run for G{generation}, "
                        f"real-data ratio {ratio}."
                    )

    if not results["test_f1"].between(0, 1).all():
        raise ValueError("test_f1 values must be between 0 and 1.")
    return results


def build_collapse_index(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate relative test-F1 loss against the same algorithm's baseline."""
    rows = []
    for algorithm in ALGORITHMS:
        runs = results[results["algorithm"] == algorithm]
        baseline = runs.loc[runs["generation"] == 0, "test_f1"].item()
        for ratio in RATIOS:
            series = runs[
                (runs["ratio"] == ratio) | (runs["generation"] == 0)
            ].sort_values("generation")
            for _, run in series.iterrows():
                rows.append({
                    "algorithm": algorithm,
                    "ratio": ratio,
                    "generation": int(run["generation"]),
                    "test_f1": run["test_f1"],
                    "collapse_index": (baseline - run["test_f1"]) / baseline,
                })
    return pd.DataFrame(rows)


def ratio_label(ratio: float) -> str:
    real = round(ratio * 100)
    return f"{real}/{100 - real} real/synthetic"


def finish_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(PROJECT_DIR)}")


def plot_collapse_by_generation(ci: pd.DataFrame) -> None:
    """README Plot 1: relative test-F1 loss by generation."""
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for algorithm in ALGORITHMS:
        for ratio in RATIOS:
            series = ci[
                (ci["algorithm"] == algorithm) & (ci["ratio"] == ratio)
            ].sort_values("generation")
            ax.plot(
                series["generation"], series["collapse_index"],
                color=RATIO_COLORS[ratio],
                linestyle="-" if algorithm == "HC" else "--",
                linewidth=2.1, marker="o", markersize=4.2,
                label=f"{ratio_label(ratio)}, {algorithm}",
            )

    ax.axhline(0, color=SECONDARY, linewidth=1)
    ax.set_xticks(GENERATIONS, [f"G{g}" for g in GENERATIONS])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Synthetic-data generation (G0 is the real-data baseline)")
    ax.set_ylabel("Collapse Index: relative loss in test F1")
    ax.set_title("Performance loss across generations", loc="left",
                 fontsize=15, fontweight="bold", pad=30)
    ax.text(
        0, 1.015,
        "Above 0% = lower test F1 than the same algorithm's 100%-real baseline. "
        "Solid = HC; dashed = SA.",
        transform=ax.transAxes, color=SECONDARY, fontsize=9,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.015))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    finish_figure(fig, PLOT_DIR / "01_collapse_index_by_generation.png")


def plot_search_comparison(results: pd.DataFrame) -> None:
    """README Plot 2: pair HC and SA for validation score and search effort."""
    paired = results.pivot(
        index=["generation", "ratio"], columns="algorithm",
        values=["val_f1", "evals_used"],
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    for ax, metric, title in (
        (axes[0], "val_f1", "Best validation F1"),
        (axes[1], "evals_used", "Search evaluations used"),
    ):
        hc = paired[(metric, "HC")]
        sa = paired[(metric, "SA")]
        low, high = min(hc.min(), sa.min()), max(hc.max(), sa.max())
        margin = max((high - low) * 0.08, 0.01)
        bounds = (low - margin, high + margin)
        ax.plot(bounds, bounds, linestyle="--", color=SECONDARY, linewidth=1)

        for ratio in (1.0,) + RATIOS:
            subset = paired[paired[("ratio", "")] == ratio]
            color = RATIO_COLORS.get(ratio, SECONDARY)
            ax.scatter(
                subset[(metric, "HC")], subset[(metric, "SA")],
                s=45, color=color, edgecolors="white", linewidths=0.8,
                label="100% real" if ratio == 1.0 else ratio_label(ratio),
                zorder=3,
            )

        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hill Climbing")
        ax.set_ylabel("Simulated Annealing")
        if metric == "evals_used":
            ax.set_xticks(np.arange(np.floor(low), np.ceil(high) + 1, 5))
            ax.set_yticks(np.arange(np.floor(low), np.ceil(high) + 1, 5))
        else:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    validation = results[results["generation"] > 0].pivot(
        index=["generation", "ratio"], columns="algorithm", values="val_f1"
    )
    wins = int((validation["SA"] > validation["HC"]).sum())
    ties = int((validation["SA"] == validation["HC"]).sum())
    losses = int((validation["SA"] < validation["HC"]).sum())
    fig.suptitle("Hill Climbing vs Simulated Annealing",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(
        0.01, 0.91,
        f"Each point is one matched run. Above the diagonal = higher value "
        f"(F1 in the left panel; evaluations in the right). "
        f"Validation-F1 outcomes across 30 synthetic runs: {wins} higher, "
        f"{ties} tied, {losses} lower for SA.",
        color=SECONDARY, fontsize=9,
    )
    axes[0].legend(loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    finish_figure(fig, PLOT_DIR / "02_hc_vs_sa.png")


def plot_mean_degradation(ci: pd.DataFrame) -> None:
    """Summarize average degradation by synthetic share, with generation ranges."""
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    x = np.arange(len(RATIOS))
    for algorithm in ALGORITHMS:
        means, lows, highs = [], [], []
        for ratio in RATIOS:
            values = ci.loc[
                (ci["algorithm"] == algorithm)
                & (ci["ratio"] == ratio)
                & (ci["generation"] > 0),
                "collapse_index",
            ]
            means.append(values.mean())
            lows.append(values.min())
            highs.append(values.max())
        means = np.asarray(means)
        color = "#4b5563" if algorithm == "HC" else "#7b4ab5"
        ax.plot(x, means, marker="o", markersize=7, linewidth=2.4,
                color=color, label=ALGORITHM_NAMES[algorithm])
        ax.vlines(x, lows, highs, color=color, linewidth=1.5, alpha=0.75)
        for xpos, mean in zip(x, means):
            ax.annotate(f"{mean:.1%}", (xpos, mean), xytext=(0, 10),
                        textcoords="offset points", ha="center", color=color,
                        fontsize=9, fontweight="bold")

    ax.set_xticks(x, [f"{round(r * 100)}% real\n{round((1-r) * 100)}% synthetic"
                      for r in RATIOS])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Training-data mix")
    ax.set_ylabel("Mean Collapse Index across G1–G10")
    ax.set_title("More synthetic data, greater average degradation",
                 loc="left", fontsize=14, fontweight="bold", pad=24)
    ax.text(
        0, 1.015,
        "Dots show the mean across generations; vertical ranges show the min–max "
        "across generations (not confidence intervals).",
        transform=ax.transAxes, color=SECONDARY, fontsize=9,
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    finish_figure(fig, PLOT_DIR / "03_mean_degradation_by_ratio.png")


def plot_test_f1_by_generation(results: pd.DataFrame) -> None:
    """Show absolute test F1 against each algorithm's 100%-real baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, algorithm in zip(axes, ALGORITHMS):
        runs = results[results["algorithm"] == algorithm]
        baseline = runs.loc[runs["generation"] == 0, "test_f1"].item()
        ax.axhline(
            baseline, color=INK, linestyle=":", linewidth=1.7,
            label=f"100% real baseline ({baseline:.3f})",
        )
        for ratio in RATIOS:
            series = runs[
                (runs["ratio"] == ratio) & (runs["generation"] > 0)
            ].sort_values("generation")
            ax.plot(
                series["generation"], series["test_f1"], color=RATIO_COLORS[ratio],
                marker="o", markersize=4, linewidth=2,
                label=ratio_label(ratio),
            )
        ax.set_xticks(range(1, 11), [f"G{g}" for g in range(1, 11)])
        ax.set_ylim(0.48, 0.68)
        ax.set_title(ALGORITHM_NAMES[algorithm], loc="left",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Synthetic-data generation")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    axes[0].set_ylabel("Test F1 (higher is better)")
    axes[1].legend(loc="lower right")
    fig.suptitle("Absolute test performance by generation",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.91,
             "Each point is one run; the dotted line is that algorithm's real-only baseline.",
             color=SECONDARY, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    finish_figure(fig, PLOT_DIR / "04_test_f1_by_generation.png")


def plot_paired_algorithm_difference(results: pd.DataFrame) -> None:
    """Show matched-run test-F1 difference; positive values favor SA."""
    synthetic = results[results["generation"] > 0]
    paired = synthetic.pivot(
        index=["generation", "ratio"], columns="algorithm", values="test_f1"
    ).reset_index()
    paired["difference"] = paired["SA"] - paired["HC"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for ratio in RATIOS:
        series = paired[paired["ratio"] == ratio].sort_values("generation")
        ax.plot(
            series["generation"], series["difference"] * 100,
            color=RATIO_COLORS[ratio], marker="o", markersize=4.5,
            linewidth=2, label=ratio_label(ratio),
        )
    ax.axhline(0, color=SECONDARY, linewidth=1.2)
    ax.set_xticks(range(1, 11), [f"G{g}" for g in range(1, 11)])
    ax.set_xlabel("Synthetic-data generation")
    ax.set_ylabel("Test-F1 difference, SA − HC (percentage points)")
    ax.set_title("Matched-run test-F1 difference by generation",
                 loc="left", fontsize=14, fontweight="bold", pad=24)
    ax.text(
        0, 1.015,
        "Above 0 = SA scored higher; below 0 = HC scored higher. "
        "Each point compares the same generation and data ratio.",
        transform=ax.transAxes, color=SECONDARY, fontsize=9,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.015))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    finish_figure(fig, PLOT_DIR / "05_paired_test_f1_difference.png")


def main() -> None:
    results = load_results()
    ci = build_collapse_index(results)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating five figures from results/results.csv:")
    plot_collapse_by_generation(ci)
    plot_search_comparison(results)
    plot_mean_degradation(ci)
    plot_test_f1_by_generation(results)
    plot_paired_algorithm_difference(results)
    print(f"Done. Figures are in {PLOT_DIR.relative_to(PROJECT_DIR)}/")


if __name__ == "__main__":
    main()
