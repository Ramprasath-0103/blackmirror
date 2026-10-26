import { legend, lineChart, scatterChart } from "./charts.js";
import { ALGORITHM, RATIO_COLOR, emptyState, fmt, h, section, segmented, table, tableToggle } from "./ui.js";

const panel = document.getElementById("panel-results");

export function renderResults(data, { runPipeline }) {
  if (!data.ready) {
    panel.replaceChildren(emptyState(
      "No results yet",
      "Run the pipeline to fill results/results.csv. The charts and tables appear here as soon as all 62 runs are done.",
      h("button", { class: "btn btn-primary", type: "button", onclick: runPipeline }, "Run full pipeline"),
    ));
    return;
  }
  panel.replaceChildren(
    collapseSection(data.collapse),
    algorithmSection(data.runs),
    section("Summary table", "One row per ratio and algorithm. CI = Collapse Index (loss of test F1 against the baseline).",
      summaryTable(data.summary)),
    h("div", { class: "section" }, h("details", {},
      h("summary", {}, `All ${data.runs.length} runs (results/results.csv)`),
      h("div", { class: "details-body", style: "padding: 0" }, runsTable(data.runs)))),
  );
}

function collapseSection(collapse) {
  const ratios = [...new Set(collapse.map((r) => r.ratio))];
  const chartHost = h("div");
  const tableHost = h("div");
  let visible = "both";

  const draw = () => {
    const series = [];
    for (const algorithm of ["HC", "SA"]) {
      if (visible !== "both" && visible !== algorithm) continue;
      for (const ratio of ratios) {
        series.push({
          label: `${fmt.ratio(ratio)} · ${algorithm}`,
          color: RATIO_COLOR[ratio],
          dashed: ALGORITHM[algorithm].dash,
          points: collapse.filter((r) => r.algorithm === algorithm && r.ratio === ratio)
            .map((r) => ({ x: r.generation, y: r.collapse_index })),
        });
      }
    }
    lineChart(chartHost, series, {
      height: 340,
      xFormat: (g) => `G${g}`,
      yFormat: (v, precise) => (precise ? fmt.signedPct(v) : `${Math.round(v * 100)}%`),
      xTitle: "Generation of synthetic data",
      yTitle: "Collapse Index",
      includeZero: true,
      tooltipTitle: (g) => (g === 0 ? "G0 · baseline" : `Generation ${g}`),
      ariaLabel: "Line chart of the Collapse Index from generation 0 to 10 for each ratio and algorithm. Use the table view for exact values.",
    });
  };

  const filter = segmented("Algorithm", [
    { value: "both", label: "Both" },
    { value: "HC", label: "Hill Climbing" },
    { value: "SA", label: "Simulated Annealing" },
  ], visible, (value) => { visible = value; draw(); });

  const card = h("div", { class: "chart-card" },
    h("div", { class: "toolbar" }, filter),
    chartHost,
    legend([
      ...ratios.map((r) => ({ label: `${fmt.ratio(r)} real / synthetic`, color: RATIO_COLOR[r], dot: true })),
      { label: "Hill Climbing", color: "--text-2" },
      { label: "Simulated Annealing", color: "--text-2", dashed: true },
    ]),
    h("div", { class: "chart-foot" },
      h("span", { class: "hint", style: "margin: 0" }, "Above 0% = worse than training on real data only."),
      tableToggle(tableHost, () => collapseTable(collapse, ratios))),
    tableHost);

  queueMicrotask(draw);
  return section("Collapse Index by generation (ratio effect)",
    "How much test F1 each training mix lost against the 100% real baseline. Hover or focus the chart and use the arrow keys to read every value.",
    card);
}

function collapseTable(collapse, ratios) {
  const generations = [...new Set(collapse.map((r) => r.generation))];
  const columns = [{ key: "generation", label: "Generation", format: (g) => `G${g}` }];
  for (const algorithm of ["HC", "SA"]) {
    for (const ratio of ratios) {
      columns.push({ key: `${algorithm}-${ratio}`, label: `${fmt.ratio(ratio)} ${algorithm}`, num: true, format: fmt.signedPct });
    }
  }
  const rows = generations.map((g) => {
    const row = { generation: g };
    for (const r of collapse.filter((c) => c.generation === g)) row[`${r.algorithm}-${r.ratio}`] = r.collapse_index;
    return row;
  });
  return table(columns, rows, { caption: "Collapse Index by generation" });
}

function algorithmSection(runs) {
  const chartHost = h("div");
  const tableHost = h("div");
  let metric = "val_f1";
  const pairs = pairUp(runs);

  const draw = () => {
    const isF1 = metric === "val_f1";
    const format = isF1 ? fmt.f3 : (v) => String(Math.round(v));
    scatterChart(chartHost, pairs.map((p) => ({
      x: p.HC[metric],
      y: p.SA[metric],
      color: RATIO_COLOR[p.ratio],
      title: fmt.run(p.generation, p.ratio),
      rows: [
        ["Hill Climbing", format(p.HC[metric]), ALGORITHM.HC.color],
        ["Simulated Annealing", format(p.SA[metric]), ALGORITHM.SA.color, true],
      ],
    })), {
      format: isF1 ? (v) => v.toFixed(2) : (v) => String(Math.round(v)),
      integer: !isF1,
      xTitle: `Hill Climbing — ${isF1 ? "validation F1" : "evaluations used"}`,
      yTitle: `Simulated Annealing — ${isF1 ? "validation F1" : "evaluations used"}`,
      ariaLabel: `Scatter chart comparing ${isF1 ? "the validation F1 found" : "the evaluations used"} by Hill Climbing and Simulated Annealing on each of the 31 training mixes.`,
    });
  };

  const toggle = segmented("Measure", [
    { value: "val_f1", label: "Validation F1 found" },
    { value: "evals_used", label: "Evaluations used" },
  ], metric, (value) => { metric = value; draw(); });

  const card = h("div", { class: "chart-card" },
    h("div", { class: "toolbar" }, toggle),
    chartHost,
    legend([1, 0.9, 0.7, 0.5].map((r) => ({
      label: r === 1 ? "baseline (100% real)" : `${fmt.ratio(r)} real / synthetic`, color: RATIO_COLOR[r], dot: true,
    }))),
    h("div", { class: "chart-foot" },
      h("span", { class: "hint", style: "margin: 0" }, "Each dot is one training mix. Above the dashed line = Simulated Annealing higher, on it = a tie."),
      tableToggle(tableHost, () => table([
        { key: "run", label: "Training mix" },
        { key: "hcF1", label: "HC val F1", num: true, format: fmt.f3 },
        { key: "saF1", label: "SA val F1", num: true, format: fmt.f3 },
        { key: "hcEvals", label: "HC evaluations", num: true },
        { key: "saEvals", label: "SA evaluations", num: true },
      ], pairs.map((p) => ({
        run: fmt.run(p.generation, p.ratio), hcF1: p.HC.val_f1, saF1: p.SA.val_f1,
        hcEvals: p.HC.evals_used, saEvals: p.SA.evals_used,
      }))))),
    tableHost);

  queueMicrotask(draw);
  return section("Hill Climbing vs Simulated Annealing, run by run (algorithm effect)",
    "Both algorithms search the same 180 settings with the same start, neighbours and budget. Only the rule for accepting a move differs.",
    card);
}

function pairUp(runs) {
  const map = new Map();
  for (const run of runs) {
    const key = `${run.generation}-${run.ratio}`;
    if (!map.has(key)) map.set(key, { generation: run.generation, ratio: run.ratio });
    map.get(key)[run.algorithm] = run;
  }
  return [...map.values()];
}

function summaryTable(summary) {
  return table([
    { key: "real/synthetic", label: "Real / synthetic" },
    { key: "algorithm", label: "Algorithm", format: (a) => ALGORITHM[a].name },
    { key: "mean_val_f1", label: "Mean val F1", num: true, format: fmt.f3 },
    { key: "mean_evals_used", label: "Mean evaluations", num: true, format: (v) => v.toFixed(1) },
    { key: "test_f1_G5", label: "Test F1 G5", num: true, format: fmt.f3 },
    { key: "test_f1_G10", label: "Test F1 G10", num: true, format: fmt.f3 },
    { key: "ci_G5", label: "CI G5", num: true, format: fmt.signedPct },
    { key: "ci_G10", label: "CI G10", num: true, format: fmt.signedPct },
    { key: "mean_ci_G1_G10", label: "Mean CI G1–G10", num: true, format: fmt.signedPct },
  ], summary, { caption: "Summary per ratio and algorithm" });
}

function runsTable(runs) {
  return table([
    { key: "generation", label: "Gen", num: true },
    { key: "ratio", label: "Real / synthetic", format: fmt.ratio },
    { key: "algorithm", label: "Algo" },
    { key: "n_estimators", label: "Trees", num: true },
    { key: "max_depth", label: "Depth", num: true },
    { key: "min_samples_split", label: "Split", num: true },
    { key: "val_f1", label: "Val F1", num: true, format: fmt.f3 },
    { key: "test_accuracy", label: "Accuracy", num: true, format: fmt.f3 },
    { key: "test_precision", label: "Precision", num: true, format: fmt.f3 },
    { key: "test_recall", label: "Recall", num: true, format: fmt.f3 },
    { key: "test_f1", label: "Test F1", num: true, format: fmt.f3 },
    { key: "evals_used", label: "Evals", num: true },
  ], runs, { caption: "All runs from results.csv" });
}
