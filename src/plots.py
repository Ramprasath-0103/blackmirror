

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

import config
from src.collapse import series_values

RATIO_COLORS = {0.9: "#2a78d6", 0.7: "#eb6834", 0.5: "#1baf7a"}
LINE_STYLES = {"HC": "-", "SA": "--"}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
BACKGROUND = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "font.size": 10,
})


def _style_axis(ax, last_generation: int) -> None:

    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8, zorder=1)
    ax.set_xticks(range(last_generation + 1))
    ax.set_xticklabels([f"G{g}" for g in range(last_generation + 1)])
    ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))   # whole-percent ticks only
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def plot_collapse_index(ci_table: pd.DataFrame, path) -> None:

    fig, ax = plt.subplots(figsize=(9, 5.2))

    for algorithm in config.ALGORITHMS:
        for ratio in config.RATIOS:
            values = series_values(ci_table, algorithm, ratio)
            ax.plot(range(len(values)), values,
                    color=RATIO_COLORS[ratio], linestyle=LINE_STYLES[algorithm],
                    linewidth=2, marker="o", markersize=4,
                    label=f"{config.ratio_label(ratio)} real/synthetic, {algorithm}")

    _style_axis(ax, config.N_GENERATIONS)
    ax.set_title("Collapse Index by generation", loc="left", fontsize=13, fontweight="bold", pad=24)
    ax.set_xlabel("Generation of synthetic data")
    ax.set_ylabel("Collapse Index (loss of test F1 vs. baseline)")
    ax.text(0.0, 1.01, "Above 0% = worse than training on real data only.  "
                       "Solid = Hill Climbing, dashed = Simulated Annealing.",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hc_vs_sa(results: pd.DataFrame, path) -> None:

    panels = [("val_f1", "Validation F1 found"),
              ("evals_used", f"Evaluations used (budget {config.EVAL_BUDGET})")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))

    for ax, (metric, title) in zip(axes, panels):
        paired = results.pivot_table(index=["generation", "ratio"], columns="algorithm", values=metric)
        paired = paired.reset_index()

        low, high = paired[["HC", "SA"]].min().min(), paired[["HC", "SA"]].max().max()
        margin = (high - low) * 0.08
        ax.plot([low - margin, high + margin], [low - margin, high + margin],
                color=INK_SECONDARY, linewidth=0.8, linestyle="--", zorder=1)

        for ratio in [1.0] + config.RATIOS:
            group = paired[paired["ratio"] == ratio]
            label = ("baseline (100% real)" if ratio == 1.0
                     else f"{config.ratio_label(ratio)} real/synthetic")
            ax.scatter(group["HC"], group["SA"], s=46, color=RATIO_COLORS.get(ratio, INK_SECONDARY),
                       edgecolors=BACKGROUND, linewidths=1, label=label, zorder=3)

        ax.set_xlim(low - margin, high + margin)
        ax.set_ylim(low - margin, high + margin)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Hill Climbing")
        ax.set_ylabel("Simulated Annealing")

        if metric == "evals_used":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            gap = (paired["SA"] - paired["HC"]).abs().idxmax()
            far = paired.loc[gap]
            ax.annotate(f"G{int(far['generation'])} {config.ratio_label(far['ratio'])}: largest gap",
                        (far["HC"], far["SA"]), xytext=(6, -60), textcoords="offset points",
                        color=INK_SECONDARY, fontsize=8,
                        arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.8))

    axes[0].legend(loc="lower right")
    fig.suptitle("Hill Climbing vs Simulated Annealing, run by run", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.905, "Each dot is one training mix. Above the dashed line = SA higher, "
                          "on the line = both found the same.", color=INK_SECONDARY, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)
