import { ALGORITHM, RATIO_COLOR, callout, emptyState, fmt, h, section, swatch } from "./ui.js";

const panel = document.getElementById("panel-overview");

const QUESTION = "How do the synthetic-data ratio and the optimization algorithm affect predictive " +
  "performance and degradation across generations?";

export function renderOverview(data, { runPipeline }) {
  if (!data.ready) {
    panel.replaceChildren(
      callout("warn", "No results yet", `${data.runs_done} of ${data.runs_expected} runs are in results/results.csv.`),
      emptyState(
        "Run the experiment to see the findings",
        "The full pipeline downloads the data, builds 10 synthetic generations, runs 62 searches and analyses them. It takes about five minutes.",
        h("button", { class: "btn btn-primary", type: "button", onclick: runPipeline }, "Run full pipeline"),
      ),
      h("p", { class: "hint", style: "text-align: center" }, QUESTION),
    );
    return;
  }

  const byRatio = data.test_f1_by_ratio;
  const algo = data.algorithm;
  const atLeast = algo.sa_higher + algo.same;

  panel.replaceChildren(
    callout("good", "Results ready",
      `${data.runs_done} of ${data.runs_expected} runs · baseline test F1 ${fmt.f3(data.baseline.HC)} with 100% real data`),

    section("Research question", null, h("p", { class: "muted", style: "max-width: 760px" }, QUESTION)),

    section("At a glance", null, h("div", { class: "findings" },
      finding("Ratio effect", `${fmt.f3(byRatio[0].f1)} → ${fmt.f3(byRatio[byRatio.length - 1].f1)}`,
        `Average test F1 falls as the synthetic share grows from 0% to 50%.${data.ratio_effect_monotone
          ? " For both algorithms, more synthetic data means more damage." : ""}`),
      finding("Algorithm effect", `${atLeast} / ${algo.total}`,
        `Runs where Simulated Annealing found an equal or better setting than Hill Climbing (strictly better in ${algo.sa_higher}).`),
      finding("Local optima", `${algo.hc_local_optima} / ${algo.hc_runs}`,
        "Hill Climbing stopped on a local optimum in every run, with evaluation budget still left over."),
    )),

    section("Test F1 by training mix",
      "Averaged over both algorithms and all generations. Higher is better.",
      h("div", { class: "tiles" }, byRatio.map((r) =>
        h("div", { class: "tile" },
          h("div", { class: "value" }, fmt.f3(r.f1)),
          h("div", { class: "label" }, swatch(RATIO_COLOR[r.ratio], { dot: true }), r.label))))),

    section("Average damage by ratio and algorithm",
      "Mean Collapse Index over generations 1–10: how much test F1 was lost compared with training on real data only.",
      damageBars(data.series)),

    section("Where the biggest spike came from", null,
      callout("warn", `${fmt.signedPct(data.largest.ci)} at G${data.largest.generation} ${data.largest.ratio_label}`,
        `That was ${ALGORITHM[data.largest.algorithm].name}. ${ALGORITHM[data.largest.other_algorithm].name} on the same training data lost ${fmt.signedPct(data.largest.other_ci)}, so about ${((data.largest.ci - data.largest.other_ci) * 100).toFixed(1)} points of the spike came from the search, not the data.`)),

    section("Search effort",
      `Average number of unique settings each algorithm evaluated (budget ${data.budget}). Simulated Annealing used more, which may explain part of its advantage.`,
      h("div", { class: "tiles" },
        ["HC", "SA"].map((a) => h("div", { class: "tile" },
          h("div", { class: "value" }, Number(algo.mean_evals[a]).toFixed(1)),
          h("div", { class: "label" }, swatch(ALGORITHM[a].color, { dot: true }), ALGORITHM[a].name))))),

    data.findings
      ? h("div", { class: "section" }, h("details", {},
          h("summary", {}, "Full written findings (results/findings.txt)"),
          h("div", { class: "details-body" }, h("pre", { class: "block" }, data.findings))))
      : null,
  );
}

function finding(kicker, headline, text) {
  return h("div", { class: "finding" },
    h("div", { class: "kicker" }, kicker),
    h("div", { class: "headline" }, headline),
    h("p", {}, text));
}

function damageBars(series) {
  const max = Math.max(...series.map((s) => s.mean_ci_G1_G10), 0.01) * 1.1;
  const ratios = [...new Set(series.map((s) => s.ratio))];
  return h("div", { class: "bars", role: "list" }, ratios.map((ratio) =>
    h("div", { class: "bar-group", role: "listitem" },
      h("h4", {}, `${fmt.ratio(ratio)} real / synthetic`),
      series.filter((s) => s.ratio === ratio).map((s) =>
        h("div", { class: "bar-row" },
          h("span", {}, swatch(ALGORITHM[s.algorithm].color, { dot: true }), s.algorithm === "HC" ? "Hill Climbing" : "Sim. Annealing"),
          h("div", { class: "track", "aria-hidden": "true" },
            h("div", { class: "fill", style: `width: ${Math.max(0, s.mean_ci_G1_G10 / max) * 100}%; background: var(${ALGORITHM[s.algorithm].color})` })),
          h("span", { class: "value" }, fmt.signedPct(s.mean_ci_G1_G10)))))));
}
