// "Search explorer": replay Hill Climbing and Simulated Annealing on one training mix.
import { getJSON } from "./api.js";
import { legend, lineChart } from "./charts.js";
import { ALGORITHM, callout, emptyState, fmt, h, section } from "./ui.js";

const panel = document.getElementById("panel-explorer");
const generationSelect = document.getElementById("ex-generation");
const ratioSelect = document.getElementById("ex-ratio");
const traceButton = document.getElementById("ex-trace");

let settings = null;
let openTab = () => {};
let replayTimer = null;

export function initExplorer(appSettings, { showTab }) {
  settings = appSettings;
  openTab = showTab;
  generationSelect.replaceChildren(...Array.from({ length: settings.generations + 1 }, (_, g) =>
    h("option", { value: g }, g === 0 ? "G0 (baseline)" : `G${g}`)));
  generationSelect.value = "1";
  generationSelect.addEventListener("change", fillRatios);
  fillRatios();
  traceButton.addEventListener("click", () => trace(Number(generationSelect.value), Number(ratioSelect.value)));
  renderIntro();
}

function fillRatios() {
  const baseline = generationSelect.value === "0";
  const options = baseline ? [1] : settings.ratios;
  const previous = Number(ratioSelect.value);
  ratioSelect.replaceChildren(...options.map((r) => h("option", { value: r }, fmt.ratio(r))));
  if (options.includes(previous)) ratioSelect.value = String(previous);
  else if (!baseline) ratioSelect.value = String(options[options.length - 1]);
  ratioSelect.disabled = baseline;
}

function suggestion(label, generation, ratio) {
  return h("button", {
    type: "button", class: "btn btn-secondary btn-small",
    onclick: () => {
      generationSelect.value = String(generation);
      fillRatios();
      ratioSelect.value = String(ratio);
      trace(generation, ratio);
    },
  }, label);
}

function renderIntro() {
  panel.replaceChildren(emptyState(
    "Watch the two searches think",
    "Pick a generation and a real/synthetic mix on the left and press Trace this run. Both algorithms run live on that mix, and every decision they make is shown here.",
    h("div", { class: "button-row", style: "justify-content: center" },
      suggestion("G1 · 50/50", 1, 0.5),
      suggestion("G7 · 50/50", 7, 0.5),
      suggestion("G0 · baseline", 0, 1)),
  ));
}

async function trace(generation, ratio) {
  openTab("explorer");
  document.getElementById("main").scrollIntoView({ block: "start", behavior: "smooth" });
  clearInterval(replayTimer);
  traceButton.disabled = true;
  panel.replaceChildren(
    callout("", `Tracing ${fmt.run(generation, ratio)}`, "Training and scoring up to 80 random forests. This takes about 5 to 20 seconds."),
    h("div", { class: "progress indeterminate", style: "margin-top: 16px" }, h("span")),
  );
  try {
    render(await getJSON(`/api/trace?generation=${generation}&ratio=${ratio}`));
  } catch (error) {
    panel.replaceChildren(callout("bad", "Could not trace this run", error.message));
  } finally {
    traceButton.disabled = false;
  }
}

function render(data) {
  const { HC, SA } = data;
  const winner = SA.val_f1 > HC.val_f1 ? "SA" : HC.val_f1 > SA.val_f1 ? "HC" : null;
  const matches = HC.matches_results_csv && SA.matches_results_csv;

  panel.replaceChildren(
    h("div", { class: "toolbar" },
      h("h3", { style: "font-size: 20px; font-weight: 400" }, fmt.run(data.generation, data.ratio)),
      h("span", { class: "spacer" }),
      matches
        ? h("span", { class: "chip chip-good" }, "✓ Same results as results.csv")
        : h("span", { class: "chip" }, "Not in results.csv yet")),
    h("p", { class: "muted", style: "font-size: 14px" },
      winner
        ? `${ALGORITHM[winner].name} found the better setting on this mix.`
        : "Both algorithms ended on the same setting on this mix.",
      " Settings are written as (trees, depth, split)."),

    h("div", { class: "grid-2", style: "margin-top: 18px" }, resultCard("HC", HC, winner), resultCard("SA", SA, winner)),

    section("Best F1 found so far",
      "Each search starts at (100, 4, 2) and spends up to 40 evaluations. A flat line means it found nothing better.",
      progressChart(HC, SA)),

    section("Hill Climbing, round by round",
      "Each round scores every neighbour (one setting changed by one step) and moves to the best one, but only if it is strictly better.",
      h("div", { class: "rounds" }, HC.rounds.map(roundCard))),

    section("Simulated Annealing, step by step",
      "Each step tries one random neighbour. Better moves are always taken; a worse move is taken if a random roll is below the chance exp(Δ / T).",
      annealingTable(SA)),
  );
}

function resultCard(name, result, winner) {
  const s = result.settings;
  return h("div", { class: "result-card", style: `--accent: var(${ALGORITHM[name].color})` },
    h("h4", {}, ALGORITHM[name].name, winner === name ? h("span", { class: "badge badge-good" }, "Better") : null),
    h("dl", {},
      h("dt", {}, "Best setting"), h("dd", {}, `${s.trees} trees · depth ${s.depth ?? "none"} · split ${s.split}`),
      h("dt", {}, "Validation F1"), h("dd", {}, fmt.f3(result.val_f1)),
      h("dt", {}, "Test F1"), h("dd", {}, fmt.f3(result.test_f1)),
      h("dt", {}, "Evaluations used"), h("dd", {}, `${result.evals_used} of ${settings.budget}`)));
}

function progressChart(HC, SA) {
  const host = h("div");
  const series = [
    { label: "Hill Climbing", color: ALGORITHM.HC.color, step: true, points: HC.progress.map((p) => ({ x: p.evals, y: p.best })) },
    { label: "Sim. Annealing", color: ALGORITHM.SA.color, dashed: true, step: true, points: SA.progress.map((p) => ({ x: p.evals, y: p.best })) },
  ];
  const last = Math.max(...series.flatMap((s) => s.points.map((p) => p.x)));
  series.forEach((s) => s.points.push({ x: last, y: s.points[s.points.length - 1].y }));
  queueMicrotask(() => lineChart(host, series, {
    height: 260,
    xTicks: [1, 5, 10, 15, 20, 25, 30, 35, 40].filter((t) => t <= last),
    xFormat: String,
    yFormat: (v) => v.toFixed(3),
    xTitle: "Evaluations used",
    yTitle: "Best validation F1",
    directLabels: true,
    tooltipTitle: (x) => `After ${x} evaluation${x === 1 ? "" : "s"}`,
    ariaLabel: `Step chart of the best validation F1 found so far. Hill Climbing reaches ${HC.val_f1.toFixed(3)} after ${HC.evals_used} evaluations; Simulated Annealing reaches ${SA.val_f1.toFixed(3)} after ${SA.evals_used}.`,
  }));
  return h("div", { class: "chart-card" }, host, legend([
    { label: "Hill Climbing", color: ALGORITHM.HC.color },
    { label: "Simulated Annealing", color: ALGORITHM.SA.color, dashed: true },
  ]));
}

function roundCard(round, index) {
  const decision = round.moved
    ? h("span", { class: "badge badge-good" }, "Move")
    : h("span", { class: "badge badge-warn" }, "Stop · local optimum");
  return h("article", { class: "round" },
    h("header", {},
      h("strong", {}, `Round ${index + 1}`),
      h("span", { class: "muted" }, `standing at ${fmt.settings(round.settings)} · F1 ${fmt.f3(round.f1)}`),
      h("span", { class: "spacer" }),
      decision),
    h("ul", {}, round.neighbours.map((n) =>
      h("li", { class: n.best ? "is-best" : null },
        h("span", { class: "mono" }, fmt.settings(n.settings)),
        h("span", { class: "num" }, fmt.f3(n.f1)),
        h("span", { class: "muted", style: "font-size: 12.5px; text-align: right" },
          n.best ? "best neighbour" : n.cached ? "already scored" : "")))));
}

function annealingTable(SA) {
  const rows = SA.steps.map((s) => {
    let decision;
    if (s.delta > 0) decision = h("span", { class: "badge badge-good" }, "Better · move");
    else if (s.accepted) decision = h("span", { class: "badge", style: "color: var(--sa); border-color: var(--sa)" }, s.delta === 0 ? "Sideways · move" : "Worse · move anyway");
    else decision = h("span", { class: "badge" }, "Stay");
    return h("tr", { class: `step-row${s.new_best ? " is-highlight" : ""}` },
      h("td", { class: "num" }, s.step),
      h("td", { class: "num" }, s.temperature.toFixed(4)),
      h("td", { class: "mono" }, fmt.settings(s.settings)),
      h("td", { class: "num" }, fmt.f3(s.f1)),
      h("td", { class: "num" }, `${s.delta >= 0 ? "+" : "−"}${Math.abs(s.delta).toFixed(3)}`),
      h("td", { class: "num" }, s.roll === null ? "—" : `${Math.round(s.probability * 100)}% / ${s.roll.toFixed(2)}`),
      h("td", {}, decision, s.new_best ? h("span", { class: "muted", style: "margin-left: 8px" }, "★ new best") : null));
  });

  const tbody = h("tbody", {}, rows);
  const wrap = h("div", { class: "table-wrap" }, h("table", {},
    h("caption", { class: "visually-hidden" }, "Simulated Annealing steps"),
    h("thead", {}, h("tr", {},
      h("th", { class: "num", scope: "col" }, "Step"),
      h("th", { class: "num", scope: "col" }, "Temperature"),
      h("th", { scope: "col" }, "Tried"),
      h("th", { class: "num", scope: "col" }, "F1"),
      h("th", { class: "num", scope: "col" }, "Δ"),
      h("th", { class: "num", scope: "col" }, "Chance / roll"),
      h("th", { scope: "col" }, "Decision"))),
    tbody));

  const replay = h("button", { type: "button", class: "btn btn-outline btn-small" }, "Replay step by step");
  replay.addEventListener("click", () => {
    clearInterval(replayTimer);
    rows.forEach((row) => { row.hidden = true; });
    let shown = 0;
    replayTimer = setInterval(() => {
      if (shown >= rows.length) return clearInterval(replayTimer);
      rows[shown].hidden = false;
      rows[shown].scrollIntoView({ block: "nearest", behavior: "smooth" });
      shown += 1;
    }, 450);
  });

  const note = SA.frozen_steps
    ? `Steps ${SA.steps.length + 1}–${SA.total_steps} are not shown: by then the temperature is almost zero and every neighbour has already been scored, so Simulated Annealing is frozen and the step cap ends the loop.`
    : "";
  return h("div", {},
    h("div", { class: "toolbar" }, replay,
      h("span", { class: "hint", style: "margin: 0" }, "Highlighted rows set a new best. “Chance / roll”: the move is taken if the roll is below the chance.")),
    wrap,
    note ? h("p", { class: "hint" }, note) : null);
}
