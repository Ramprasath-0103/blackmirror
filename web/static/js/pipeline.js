// Left panel "Pipeline" + the "Run log" tab: start the command-line scripts and follow their output.
import { getJSON, postJSON } from "./api.js";
import { emptyState, h } from "./ui.js";

const PHASE_ORDER = ["prepare", "experiment", "analyze"];
const STATUS_TEXT = { running: "Running", done: "Finished", failed: "Failed", stopped: "Stopped" };

let phases = [];
let job = { status: "idle", lines: [] };
let lineCount = 0;
let pollTimer = null;
let onFinished = () => {};

const phaseList = document.getElementById("phases");
const runAllButton = document.getElementById("run-all");
const stopButton = document.getElementById("stop");
const freshBox = document.getElementById("fresh");
const jobBox = document.getElementById("job");
const logPanel = document.getElementById("panel-log");

export function initPipeline({ finished }) {
  onFinished = finished;
  runAllButton.addEventListener("click", () => start(PHASE_ORDER));
  stopButton.addEventListener("click", () => postJSON("/api/stop").catch(showError));
  renderLog();
  getJSON("/api/job").then((state) => {
    job = state;
    logLines.push(...state.lines);
    lineCount = state.line_count;
    renderJob();
    renderLog();
    if (state.status === "running") poll();
  });
}

export function updateStatus(status) {
  phases = status.phases;
  renderJob();
  phaseList.replaceChildren(...phases.map((phase, i) => {
    const running = job.status === "running" && job.current === phase.id;
    const state = running ? "is-running" : phase.done ? "is-done" : "";
    const chip = running
      ? h("span", { class: "chip" }, "Running")
      : phase.done ? h("span", { class: "chip chip-good" }, "✓ Done") : h("span", { class: "chip" }, "Not run");
    return h("li", { class: `phase ${state}` },
      h("span", { class: "step", "aria-hidden": "true" }, phase.done && !running ? "✓" : String(i + 1)),
      h("div", { class: "title-row" }, h("h3", {}, phase.title), chip),
      h("div", { class: "body" },
        h("code", {}, `python ${phase.script}`),
        h("div", { class: "actions" },
          h("p", { class: "detail" }, phase.detail),
          h("button", {
            type: "button", class: "btn btn-outline btn-small", disabled: job.status === "running",
            "aria-label": `Run phase ${i + 1}, ${phase.title}`,
            onclick: () => start([phase.id]),
          }, "Run"))));
  }));
}

async function start(phaseIds) {
  try {
    await postJSON("/api/run", { phases: phaseIds, fresh: freshBox.checked });
    job = { status: "running", phases: phaseIds, current: phaseIds[0], lines: [], progress: null, started: Date.now() / 1000 };
    lineCount = 0;
    logLines.length = 0;
    renderJob();
    renderLog();
    updateStatus({ phases });
    poll();
  } catch (error) {
    showError(error);
  }
}

function poll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    try {
      const state = await getJSON(`/api/job?since=${lineCount}`);
      logLines.push(...state.lines);
      lineCount = state.line_count;
      const wasRunning = job.status === "running";
      job = state;
      renderJob();
      appendLog(state.lines);
      if (state.status === "running") {
        updateStatus({ phases });
        poll();
      } else if (wasRunning) {
        onFinished();
      }
    } catch (error) {
      showError(error);
      poll();
    }
  }, 700);
}

function titleOf(id) {
  return phases.find((p) => p.id === id)?.title ?? id;
}

function renderJob() {
  const running = job.status === "running";
  runAllButton.disabled = running;
  stopButton.hidden = !running;
  freshBox.disabled = running;
  if (job.status === "idle") {
    jobBox.replaceChildren();
    return;
  }

  const position = job.phases ? job.phases.indexOf(job.current) + 1 : 0;
  const count = job.phases ? job.phases.length : 0;
  const progress = job.progress;
  const elapsed = Math.round(((job.finished || Date.now() / 1000) - job.started) || 0);
  const label = running
    ? `${titleOf(job.current)}${count > 1 ? ` (phase ${position} of ${count})` : ""}`
    : `${STATUS_TEXT[job.status]} · ${count > 1 ? "full pipeline" : titleOf(job.current)}`;
  const detail = progress ? `${progress.done} / ${progress.total}` : `${elapsed}s`;

  const bar = h("div", { class: `progress${running && !progress ? " indeterminate" : ""}` },
    h("span", { style: progress ? `width: ${(100 * progress.done) / progress.total}%` : running ? "" : "width: 100%" }));
  jobBox.replaceChildren(h("div", { class: "row" }, h("span", {}, label), h("span", { class: "num" }, detail)), bar);
}

const logLines = [];
let logPre = null;

function renderLog() {
  if (!logLines.length && job.status === "idle") {
    logPre = null;
    logPanel.replaceChildren(emptyState(
      "Nothing has run yet",
      "Press Run full pipeline on the left, or run a single phase. The output of each script appears here, exactly as it would in a terminal.",
    ), h("pre", { class: "block" }, "python prepare_data.py\npython run_experiment.py\npython analyze.py"));
    return;
  }
  logPre = h("pre", { class: "log", tabindex: "0", "aria-label": "Script output" });
  logPanel.replaceChildren(logPre);
  appendLog(logLines);
}

function appendLog(lines) {
  if (!logPre) return renderLog();
  const atBottom = logPre.scrollHeight - logPre.scrollTop - logPre.clientHeight < 40;
  for (const line of lines) {
    logPre.append(line.startsWith("$ ") ? h("span", { class: "cmd" }, line) : line, "\n");
  }
  if (atBottom) logPre.scrollTop = logPre.scrollHeight;
}

function showError(error) {
  jobBox.replaceChildren(h("div", { class: "callout callout-bad" }, h("p", {}, error.message)));
}
