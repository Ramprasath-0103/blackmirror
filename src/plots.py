"""
plots.py - the two figures that answer the research question (plan Phase 3.3).

    plot_collapse_index(ci_table, path)                Plot 1: CI vs generation, 6 lines
    plot_lstm_forecast(ci_table, forecasts, path)      Plot 2: actual, holdout check, forecast

Colour = ratio (same colours in both plots). Line style = algorithm.
"""

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt  # imported after the line above so the setting takes effect
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

import config
from src.collapse import series_values

# One colour per ratio. These three were checked to stay distinguishable
# for colour-blind readers.
RATIO_COLORS = {0.9: "#2a78d6", 0.7: "#eb6834", 0.5: "#1baf7a"}
LINE_STYLES = {"HC": "-", "SA": "--"}
ALGORITHM_NAMES = {"HC": "Hill Climbing", "SA": "Simulated Annealing"}

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
    """Shared look: G0, G1, ... on the x-axis, percentages on the y-axis, a zero line."""
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8, zorder=1)
    ax.set_xticks(range(last_generation + 1))
    ax.set_xticklabels([f"G{g}" for g in range(last_generation + 1)])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def plot_collapse_index(ci_table: pd.DataFrame, path) -> None:
    """Plot 1: one line per (ratio, algorithm). Above 0 = worse than the baseline."""
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


def plot_lstm_forecast(ci_table: pd.DataFrame, forecasts: pd.DataFrame, path) -> None:
    """
    Plot 2: a grid with one small chart per series (rows = algorithm, columns = ratio).

    Each chart shows the real CI (G0-G10), what the LSTM predicted for the
    held-out G9-G10, and its forecast for G11-G13 (shaded area).
    """
    last_generation = config.N_GENERATIONS + config.N_FORECAST_STEPS
    fig, axes = plt.subplots(len(config.ALGORITHMS), len(config.RATIOS),
                             figsize=(13, 6.5), sharex=True, sharey=True)

    for row, algorithm in enumerate(config.ALGORITHMS):
        for col, ratio in enumerate(config.RATIOS):
            ax = axes[row, col]
            color = RATIO_COLORS[ratio]
            actual = series_values(ci_table, algorithm, ratio)
            series = forecasts[(forecasts["algorithm"] == algorithm) & (forecasts["ratio"] == ratio)]
            holdout = series[series["kind"] == "holdout"]
            future = series[series["kind"] == "forecast"]

            ax.axvspan(config.N_GENERATIONS + 0.5, last_generation + 0.5, color=GRID_COLOR, alpha=0.5, zorder=0)
            ax.plot(range(len(actual)), actual, color=color, linewidth=2, marker="o", markersize=4)
            ax.plot(holdout["generation"], holdout["lstm"], linestyle="none",
                    marker="D", markersize=7, markerfacecolor=BACKGROUND, markeredgecolor=color,
                    markeredgewidth=1.6)
            # Join the forecast to the last real point so the line is continuous
            # (markevery skips that first point so the real G10 marker stays visible).
            ax.plot([config.N_GENERATIONS] + future["generation"].tolist(),
                    [actual[-1]] + future["lstm"].tolist(),
                    color=color, linewidth=2, linestyle=":", marker="o", markersize=4,
                    markerfacecolor=BACKGROUND, markevery=slice(1, None))

            _style_axis(ax, last_generation)
            ax.tick_params(axis="x", labelsize=7)
            ax.set_title(f"{config.ratio_label(ratio)}  ·  {ALGORITHM_NAMES[algorithm]}",
                         loc="left", fontsize=10)

    legend_items = [
        Line2D([], [], color=INK_SECONDARY, linewidth=2, marker="o", markersize=4,
               label="Actual Collapse Index (G0-G10)"),
        Line2D([], [], color=INK_SECONDARY, linestyle="none", marker="D", markersize=7,
               markerfacecolor=BACKGROUND, label="LSTM check on held-out G9-G10"),
        Line2D([], [], color=INK_SECONDARY, linewidth=2, linestyle=":", marker="o",
               markersize=4, markerfacecolor=BACKGROUND, label="LSTM forecast G11-G13 (shaded)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("LSTM forecast of the Collapse Index", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.supylabel("Collapse Index", color=INK_SECONDARY, fontsize=10)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
