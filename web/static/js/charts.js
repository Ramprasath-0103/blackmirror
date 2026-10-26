// Interactive SVG charts: a line chart (with crosshair tooltip) and a scatter chart
// (with per-dot tooltip). Both redraw themselves when their container is resized.
import { cssVar, h } from "./ui.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) el.setAttribute(key, value);
  }
  return el;
}

function niceTicks(min, max, count = 5) {
  const span = max - min || Math.abs(max) || 1;
  const magnitude = 10 ** Math.floor(Math.log10(span / count));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => span / s <= count) || magnitude * 10;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Number(v.toFixed(10)));
  return { ticks, min: start, max: end };
}

const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const host = entry.target;
    const width = Math.round(entry.contentRect.width);
    if (host._chart && width && width !== host._width) {
      host._width = width;
      host._chart.draw();
    }
  }
});

function mount(host, draw) {
  host.classList.add("chart");
  host._chart = { draw };
  host._width = Math.round(host.clientWidth);
  resizeObserver.observe(host);
  draw();
}

function makeTooltip(host) {
  const tip = h("div", { class: "tooltip", hidden: true, role: "status" });
  host.append(tip);
  return tip;
}

function placeTooltip(host, tip, x, y) {
  tip.hidden = false;
  const width = tip.offsetWidth;
  const left = x + 16 + width > host.clientWidth ? x - 16 - width : x + 16;
  tip.style.left = `${Math.max(0, left)}px`;
  tip.style.top = `${Math.max(0, y - tip.offsetHeight / 2)}px`;
}

function tooltipLine(colorVar, dashed, label, value) {
  return h("div", { class: "line" },
    h("span", {}, h("span", { class: `key${dashed ? " key-dashed" : ""}`, style: `color: var(${colorVar})` }), label),
    h("strong", {}, value));
}

/**
 * series: [{ key, label, color: "--css-var", dashed, step, points: [{ x, y }] }]
 * options: xTicks, xFormat, yFormat, xTitle, yTitle, height, ariaLabel, includeZero,
 *          tooltipTitle(x), directLabels, reference: { y, label }
 */
export function lineChart(host, series, options) {
  mount(host, () => drawLine(host, series, options));
}

function drawLine(host, series, o) {
  host.replaceChildren();
  const width = Math.max(host.clientWidth, 280);
  const height = o.height || 300;
  const margin = { top: 14, right: o.directLabels ? 104 : 20, bottom: o.xTitle ? 48 : 30, left: 58 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b);
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  if (o.reference) ys.push(o.reference.y);
  if (o.includeZero) ys.push(0);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const pad = (yMax - yMin) * 0.08 || 0.01;
  const yt = niceTicks(o.includeZero && yMin >= 0 ? 0 : yMin - pad, yMax + pad, 5);

  const x = (v) => margin.left + ((v - xMin) / (xMax - xMin || 1)) * innerW;
  const y = (v) => margin.top + (1 - (v - yt.min) / (yt.max - yt.min || 1)) * innerH;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img", "aria-label": o.ariaLabel, tabindex: 0 });

  for (const t of yt.ticks) {
    svg.append(svgEl("line", { class: "gridline", x1: margin.left, x2: width - margin.right, y1: y(t), y2: y(t) }));
    const label = svgEl("text", { x: margin.left - 10, y: y(t) + 4, "text-anchor": "end" });
    label.textContent = o.yFormat(t);
    svg.append(label);
  }
  if (yt.min <= 0 && yt.max >= 0) {
    svg.append(svgEl("line", { class: "baseline", x1: margin.left, x2: width - margin.right, y1: y(0), y2: y(0) }));
  }
  for (const t of o.xTicks || xs) {
    const label = svgEl("text", { x: x(t), y: height - margin.bottom + 20, "text-anchor": "middle" });
    label.textContent = o.xFormat(t);
    svg.append(label);
  }
  if (o.xTitle) {
    const title = svgEl("text", { class: "axis-title", x: margin.left + innerW / 2, y: height - 6, "text-anchor": "middle" });
    title.textContent = o.xTitle;
    svg.append(title);
  }
  if (o.yTitle) {
    const title = svgEl("text", { class: "axis-title", transform: `translate(14 ${margin.top + innerH / 2}) rotate(-90)`, "text-anchor": "middle" });
    title.textContent = o.yTitle;
    svg.append(title);
  }
  if (o.reference) {
    svg.append(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y(o.reference.y), y2: y(o.reference.y),
      stroke: cssVar("--text-3"), "stroke-dasharray": "2 4" }));
    const label = svgEl("text", { x: width - margin.right, y: y(o.reference.y) - 6, "text-anchor": "end" });
    label.textContent = o.reference.label;
    svg.append(label);
  }

  const valueAt = (s, xv) => {
    if (!s.step) return s.points.find((p) => p.x === xv)?.y;
    let value;
    for (const p of s.points) if (p.x <= xv) value = p.y;
    return value;
  };

  for (const s of series) {
    const color = cssVar(s.color);
    let d = "";
    s.points.forEach((p, i) => {
      if (i === 0) d += `M${x(p.x)},${y(p.y)}`;
      else if (s.step) d += `H${x(p.x)}V${y(p.y)}`;
      else d += `L${x(p.x)},${y(p.y)}`;
    });
    if (s.step) d += `H${x(xMax)}`;
    svg.append(svgEl("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round",
      "stroke-linecap": "round", "stroke-dasharray": s.dashed ? "6 5" : null }));
    if (!s.step) {
      for (const p of s.points) {
        svg.append(svgEl("circle", { cx: x(p.x), cy: y(p.y), r: 3, fill: color, stroke: cssVar("--inset"), "stroke-width": 1.5 }));
      }
    }
    if (o.directLabels) {
      const last = s.step ? valueAt(s, xMax) : s.points[s.points.length - 1].y;
      const label = svgEl("text", { x: x(xMax) + 10, y: y(last) + 4 });
      label.textContent = s.label;
      label.style.fill = cssVar("--text-2");
      svg.append(label);
    }
  }

  const crosshair = svgEl("line", { class: "crosshair", y1: margin.top, y2: margin.top + innerH, visibility: "hidden" });
  const markers = svgEl("g");
  const overlay = svgEl("rect", { x: margin.left, y: margin.top, width: innerW, height: innerH, fill: "transparent" });
  svg.append(crosshair, markers, overlay);
  host.append(svg);
  const tip = makeTooltip(host);

  let index = -1;
  const show = (i) => {
    index = Math.max(0, Math.min(xs.length - 1, i));
    const xv = xs[index];
    crosshair.setAttribute("x1", x(xv));
    crosshair.setAttribute("x2", x(xv));
    crosshair.setAttribute("visibility", "visible");
    markers.replaceChildren();
    const rows = series
      .map((s) => ({ s, value: valueAt(s, xv) }))
      .filter((r) => r.value !== undefined)
      .sort((a, b) => b.value - a.value);
    for (const r of rows) {
      markers.append(svgEl("circle", { cx: x(xv), cy: y(r.value), r: 4.5, fill: cssVar(r.s.color), stroke: cssVar("--text"), "stroke-width": 1.5 }));
    }
    tip.replaceChildren(h("div", { class: "title" }, o.tooltipTitle(xv)),
      ...rows.map((r) => tooltipLine(r.s.color, r.s.dashed, r.s.label, o.yFormat(r.value, true))));
    const top = rows.length ? y(rows[0].value) : margin.top;
    placeTooltip(host, tip, x(xv), Math.min(top + 40, height - 40));
  };
  const hide = () => {
    crosshair.setAttribute("visibility", "hidden");
    markers.replaceChildren();
    tip.hidden = true;
  };

  overlay.addEventListener("pointermove", (event) => {
    const rect = svg.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * width;
    let nearest = 0;
    xs.forEach((v, i) => { if (Math.abs(x(v) - px) < Math.abs(x(xs[nearest]) - px)) nearest = i; });
    show(nearest);
  });
  overlay.addEventListener("pointerleave", hide);
  svg.addEventListener("focus", () => show(index < 0 ? 0 : index));
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight") { show(index + 1); event.preventDefault(); }
    if (event.key === "ArrowLeft") { show(index - 1); event.preventDefault(); }
    if (event.key === "Escape") hide();
  });
}

/**
 * points: [{ x, y, color: "--css-var", title, rows: [[label, value]] }]
 * options: xTitle, yTitle, format, integer, height, ariaLabel
 * Draws a dashed diagonal: dots above it have a higher y than x.
 */
export function scatterChart(host, points, options) {
  mount(host, () => drawScatter(host, points, options));
}

function drawScatter(host, points, o) {
  host.replaceChildren();
  const width = Math.max(host.clientWidth, 280);
  const height = o.height || Math.min(420, Math.max(300, width * 0.62));
  const margin = { top: 14, right: 20, bottom: 48, left: 58 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const values = points.flatMap((p) => [p.x, p.y]);
  const span = Math.max(...values) - Math.min(...values) || 1;
  const ticks = niceTicks(Math.min(...values) - span * 0.06, Math.max(...values) + span * 0.06, o.integer ? 6 : 5);
  const sx = (v) => margin.left + ((v - ticks.min) / (ticks.max - ticks.min)) * innerW;
  const sy = (v) => margin.top + (1 - (v - ticks.min) / (ticks.max - ticks.min)) * innerH;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img", "aria-label": o.ariaLabel });
  for (const t of ticks.ticks) {
    svg.append(svgEl("line", { class: "gridline", x1: margin.left, x2: width - margin.right, y1: sy(t), y2: sy(t) }));
    svg.append(svgEl("line", { class: "gridline", x1: sx(t), x2: sx(t), y1: margin.top, y2: margin.top + innerH }));
    const yl = svgEl("text", { x: margin.left - 10, y: sy(t) + 4, "text-anchor": "end" });
    yl.textContent = o.format(t);
    const xl = svgEl("text", { x: sx(t), y: height - margin.bottom + 20, "text-anchor": "middle" });
    xl.textContent = o.format(t);
    svg.append(yl, xl);
  }
  svg.append(svgEl("line", { x1: sx(ticks.min), y1: sy(ticks.min), x2: sx(ticks.max), y2: sy(ticks.max),
    stroke: cssVar("--text-3"), "stroke-dasharray": "4 5" }));

  const xTitle = svgEl("text", { class: "axis-title", x: margin.left + innerW / 2, y: height - 6, "text-anchor": "middle" });
  xTitle.textContent = o.xTitle;
  const yTitle = svgEl("text", { class: "axis-title", transform: `translate(14 ${margin.top + innerH / 2}) rotate(-90)`, "text-anchor": "middle" });
  yTitle.textContent = o.yTitle;
  svg.append(xTitle, yTitle);

  host.append(svg);
  const tip = makeTooltip(host);
  const surface = cssVar("--inset");

  for (const p of points) {
    const dot = svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 4.5, fill: cssVar(p.color), stroke: surface, "stroke-width": 2 });
    const hit = svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 12, fill: "transparent" });
    hit.addEventListener("pointerenter", () => {
      dot.setAttribute("stroke", cssVar("--text"));
      dot.setAttribute("r", 6);
      tip.replaceChildren(h("div", { class: "title" }, p.title),
        ...p.rows.map(([label, value, colorVar, dashed]) => tooltipLine(colorVar || "--text-3", dashed, label, value)));
      placeTooltip(host, tip, sx(p.x), sy(p.y));
    });
    hit.addEventListener("pointerleave", () => {
      dot.setAttribute("stroke", surface);
      dot.setAttribute("r", 4.5);
      tip.hidden = true;
    });
    svg.append(dot, hit);
  }
}

export function legend(items) {
  return h("div", { class: "legend" }, items.map((item) =>
    h("span", {}, h("span", {
      class: item.dot ? "dot" : `key${item.dashed ? " key-dashed" : ""}`,
      style: item.dot ? `background: var(${item.color})` : `color: var(${item.color})`,
      "aria-hidden": "true",
    }), item.label)));
}
