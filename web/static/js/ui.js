// Small helpers shared by every view. Text always goes in through textContent,
// so nothing read from a file can be interpreted as HTML.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "style") el.style.cssText = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2).toLowerCase(), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export const RATIO_COLOR = { 1: "--text-2", 0.9: "--r90", 0.7: "--r70", 0.5: "--r50" };
export const ALGORITHM = {
  HC: { name: "Hill Climbing", color: "--hc", dash: false },
  SA: { name: "Simulated Annealing", color: "--sa", dash: true },
};

export const fmt = {
  f3: (v) => Number(v).toFixed(3),
  signedPct: (v, digits = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)}%`,
  pct: (v, digits = 1) => `${(v * 100).toFixed(digits)}%`,
  int: (v) => Number(v).toLocaleString("en-US"),
  ratio: (r) => (Number(r) === 1 ? "100% real" : `${Math.round(r * 100)}/${100 - Math.round(r * 100)}`),
  settings: (s) => `(${s.trees}, ${s.depth ?? "None"}, ${s.split})`,
  run: (generation, ratio) => (Number(generation) === 0 ? "G0 · 100% real" : `G${generation} · ${fmt.ratio(ratio)}`),
};

export function swatch(cssColorVar, { dashed = false, dot = false } = {}) {
  return h("span", {
    class: dot ? "dot" : `key${dashed ? " key-dashed" : ""}`,
    style: dot ? `background: var(${cssColorVar})` : `color: var(${cssColorVar})`,
    "aria-hidden": "true",
  });
}

export function table(columns, rows, { caption, rowClass } = {}) {
  const head = h("tr", {}, columns.map((c) => h("th", { scope: "col", class: c.num ? "num" : null }, c.label)));
  const body = rows.map((row) =>
    h("tr", { class: rowClass ? rowClass(row) : null },
      columns.map((c) => {
        const value = c.format ? c.format(row[c.key], row) : row[c.key];
        return h("td", { class: c.num ? "num" : c.wrap ? "wrap" : null }, value);
      })),
  );
  return h("div", { class: "table-wrap" },
    h("table", {}, caption ? h("caption", { class: "visually-hidden" }, caption) : null,
      h("thead", {}, head), h("tbody", {}, body)));
}

export function callout(kind, title, text, action) {
  return h("div", { class: `callout callout-${kind}` }, h("strong", {}, title), text ? h("p", {}, text) : null, action || null);
}

export function emptyState(title, text, ...actions) {
  return h("div", { class: "empty" }, h("h3", {}, title), h("p", {}, text), actions);
}

export function section(title, lede, ...children) {
  return h("div", { class: "section" }, h("h3", {}, title), lede ? h("p", { class: "lede" }, lede) : null, children);
}

export function segmented(label, options, selected, onChange) {
  const group = h("div", { class: "segmented", role: "group", "aria-label": label });
  for (const option of options) {
    group.append(h("button", {
      type: "button",
      "aria-pressed": String(option.value === selected),
      onclick: () => {
        group.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
        group.querySelector(`[data-value="${option.value}"]`).setAttribute("aria-pressed", "true");
        onChange(option.value);
      },
      "data-value": option.value,
    }, option.label));
  }
  return group;
}

export function tableToggle(container, buildTable) {
  let shown = false;
  let tableEl = null;
  const button = h("button", { type: "button", class: "link-button", "aria-expanded": "false" }, "Show as table");
  button.addEventListener("click", () => {
    shown = !shown;
    button.textContent = shown ? "Hide table" : "Show as table";
    button.setAttribute("aria-expanded", String(shown));
    if (shown) {
      tableEl = h("div", { style: "margin-top: 12px" }, buildTable());
      container.append(tableEl);
    } else if (tableEl) {
      tableEl.remove();
    }
  });
  return button;
}
