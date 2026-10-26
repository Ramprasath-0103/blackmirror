// Start-up: load the data, wire up the tabs, and refresh everything when a pipeline run finishes.
import { getJSON } from "./api.js";
import { renderDataset } from "./dataset.js";
import { initExplorer } from "./explorer.js";
import { renderOverview } from "./overview.js";
import { initPipeline, updateStatus } from "./pipeline.js";
import { renderResults } from "./results.js";
import { h } from "./ui.js";

const TABS = ["overview", "results", "explorer", "dataset", "log"];
const tabs = TABS.map((id) => document.getElementById(`tab-${id}`));

function showTab(id, { focus = false } = {}) {
  TABS.forEach((name, i) => {
    const selected = name === id;
    tabs[i].setAttribute("aria-selected", String(selected));
    tabs[i].tabIndex = selected ? 0 : -1;
    document.getElementById(`panel-${name}`).hidden = !selected;
  });
  if (focus) document.getElementById(`tab-${id}`).focus();
  history.replaceState(null, "", `#${id}`);
}

function initTabs() {
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => showTab(TABS[i]));
    tab.addEventListener("keydown", (event) => {
      const moves = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: TABS.length - 1 };
      if (event.key in moves) {
        event.preventDefault();
        showTab(TABS[(moves[event.key] + TABS.length) % TABS.length], { focus: true });
      }
    });
  });
  document.querySelectorAll("[data-tab-link]").forEach((link) =>
    link.addEventListener("click", (event) => {
      event.preventDefault();
      showTab(link.dataset.tabLink, { focus: true });
    }));
  const fromHash = location.hash.slice(1);
  showTab(TABS.includes(fromHash) ? fromHash : "overview");
}

function renderSettings(settings) {
  const items = [
    ["Seed", settings.seed],
    ["Features", settings.features],
    ["Generations", settings.generations],
    ["Ratios (real / synthetic)", settings.ratio_labels.join(", ")],
    ["Settings in the grid", settings.grid_size],
    ["Evaluation budget", settings.budget],
    ["SA start temperature", settings.start_temperature],
    ["SA cooling per step", `× ${settings.cooling_rate}`],
  ];
  document.getElementById("settings").replaceChildren(...items.flatMap(([label, value]) =>
    [h("dt", {}, label), h("dd", {}, String(value))]));
}

const runPipeline = () => document.getElementById("run-all").click();

async function refresh() {
  const [status, overview, results, dataset] = await Promise.all([
    getJSON("/api/status"), getJSON("/api/overview"), getJSON("/api/results"), getJSON("/api/dataset"),
  ]);
  updateStatus(status);
  renderOverview(overview, { runPipeline });
  renderResults(results, { runPipeline });
  renderDataset(dataset);
}

async function start() {
  initTabs();
  const settings = await getJSON("/api/settings");
  renderSettings(settings);
  initExplorer(settings, { showTab });
  initPipeline({ finished: refresh });
  await refresh();
}

start().catch((error) => {
  document.getElementById("panel-overview").replaceChildren(
    h("div", { class: "callout callout-bad" }, h("strong", {}, "Could not load the app"), h("p", {}, error.message)));
});
