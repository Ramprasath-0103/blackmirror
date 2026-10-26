import { getJSON } from "./api.js";
import { lineChart } from "./charts.js";
import { callout, emptyState, fmt, h, section, table } from "./ui.js";

const panel = document.getElementById("panel-dataset");

export function renderDataset(data) {
  const intro = callout("", "Adult Income · 1994 US Census",
    `48,842 people. The task is to predict whether a person earns more than $50K a year. We use a stratified sample of ${fmt.int(data.subsample)}.`);

  const columns = section("Columns we use",
    `${data.columns.length - 1} features plus the answer, income.`,
    table([
      { key: "name", label: "Column", format: (v) => h("code", {}, v) },
      { key: "type", label: "Type" },
      { key: "description", label: "Meaning", wrap: true },
    ], data.columns, { caption: "Columns used" }));

  const dropped = section("Columns we drop, and why", null,
    table([
      { key: "name", label: "Column", format: (v) => h("code", {}, v) },
      { key: "reason", label: "Why it is dropped", wrap: true },
    ], data.dropped, { caption: "Dropped columns" }));

  if (!data.ready) {
    panel.replaceChildren(intro,
      emptyState("No data files yet", "Run phase 1 (Data & generation) to download, clean and split the data and create G1–G10."),
      columns, dropped);
    return;
  }

  panel.replaceChildren(
    intro,
    section("How the real data is split",
      "Stratified, so every set keeps the same share of people earning more than $50K.",
      h("div", { class: "tiles" }, data.splits.map((s) =>
        h("div", { class: "tile" },
          h("div", { class: "value" }, fmt.int(s.rows)),
          h("div", { class: "label" }, h("strong", { style: "color: var(--text-2); font-weight: 500" }, s.name), ` · ${fmt.pct(s.rate)} >50K`),
          h("div", { class: "label", style: "font-size: 12px" }, s.role))))),
    columns,
    dropped,
    rateSection(data.generations),
    sampleSection(data),
  );
}

function rateSection(generations) {
  const host = h("div");
  const base = generations[0].rate;
  queueMicrotask(() => lineChart(host, [{
    label: "Share earning >50K",
    color: "--text-2",
    points: generations.map((g) => ({ x: g.generation, y: g.rate })),
  }], {
    height: 220,
    xFormat: (g) => `G${g}`,
    yFormat: (v, precise) => (precise ? fmt.pct(v) : `${(v * 100).toFixed(0)}%`),
    xTitle: "Generation",
    reference: { y: base, label: `G0 ${fmt.pct(base)}` },
    tooltipTitle: (g) => (g === 0 ? "G0 · real data" : `Generation ${g}`),
    ariaLabel: `Line chart of the share of people earning more than $50K in each generation, from ${fmt.pct(base)} in G0 to ${fmt.pct(generations[generations.length - 1].rate)} in G${generations.length - 1}.`,
  }));
  return section("Share of high earners in each generation",
    "Each synthetic generation is learned from the one before it, so small errors can pile up.",
    h("div", { class: "chart-card" }, host));
}

function sampleSection(data) {
  const select = h("select", { id: "sample-generation", "aria-label": "Synthetic generation to compare" },
    data.generations.slice(1).map((g) => h("option", { value: g.generation }, `G${g.generation}`)));
  select.value = String(data.generation);
  const tables = h("div");

  const draw = (real, synthetic, generation) => {
    const columns = Object.keys(real[0]).map((key) => ({ key, label: key, num: typeof real[0][key] === "number" }));
    tables.replaceChildren(
      h("p", { class: "muted", style: "font-size: 13px; margin-bottom: 8px" }, "Real people (G0)"),
      table(columns, real, { caption: "Real people" }),
      h("p", { class: "muted", style: "font-size: 13px; margin: 18px 0 8px" }, `Synthetic people (G${generation})`),
      table(columns, synthetic, { caption: "Synthetic people" }),
    );
  };
  draw(data.sample_real, data.sample_synthetic, data.generation);

  select.addEventListener("change", async () => {
    const next = await getJSON(`/api/dataset?generation=${select.value}`);
    draw(next.sample_real, next.sample_synthetic, next.generation);
  });

  return section("Real vs synthetic rows",
    "The first rows of the real data next to a synthetic generation. Look for combinations no real person has.",
    h("div", { class: "toolbar" }, h("label", { class: "field", style: "width: 140px" }, h("span", {}, "Generation"), select)),
    tables);
}
