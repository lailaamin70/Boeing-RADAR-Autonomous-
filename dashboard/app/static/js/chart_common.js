// Shared Chart.js helpers for the Overview, Detectability and scene ego-motion charts.
window.TS = (function () {
  const COLORS = { radar: "#e2793d", lidar: "#4bb4c4", camera: "#7f9fd9", general: "#9aa4b5" };
  const AXIS = "#8b93a3";
  const GRID = "#2a3140";
  const TEXT = "#dde3ea";
  const registry = {};

  function axis(opts, title, extra) {
    return Object.assign(
      {
        stacked: !!opts.stacked,
        ticks: { color: AXIS },
        grid: { color: GRID },
        title: title ? { display: true, text: title, color: AXIS } : { display: false },
      },
      extra || {}
    );
  }

  // Draw the chart registered under `key` on `canvas`.
  function draw(key, canvas, type, labels, datasets, opts) {
    opts = opts || {};
    if (registry[key]) registry[key].destroy();
    const options = {
      responsive: true,
      indexAxis: opts.indexAxis || "x",
      plugins: { legend: { display: opts.legend !== false, labels: { color: TEXT } } },
      scales: {
        x: axis(opts, opts.xTitle, opts.x),
        y: axis(opts, opts.yTitle, Object.assign({ beginAtZero: true }, opts.y)),
      },
    };
    if (opts.aspectRatio) options.aspectRatio = opts.aspectRatio;
    if (opts.tooltip) options.plugins.tooltip = opts.tooltip;
    registry[key] = new Chart(canvas.getContext("2d"), {
      type: type,
      data: { labels: labels, datasets: datasets },
      options: options,
    });
    return registry[key];
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function fmt(v, digits) {
    if (v === null || v === undefined) return "\u2014";
    return typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: digits === undefined ? 2 : digits }) : v;
  }

  // Build a dark Bootstrap table. columns: [{label, get(row), align}]
  function table(columns, rows) {
    const head = columns.map((c) => `<th${c.align === "right" ? ' class="text-end"' : ""}>${esc(c.label)}</th>`).join("");
    const body = rows
      .map(
        (r) =>
          "<tr>" +
          columns
            .map((c) => {
              const v = c.get(r);
              return `<td${c.align === "right" ? ' class="text-end mono"' : ""}>${c.html ? v : esc(v === null || v === undefined ? "\u2014" : v)}</td>`;
            })
            .join("") +
          "</tr>"
      )
      .join("");
    return `<table class="table table-sm table-dark table-bordered" style="font-size:.82rem"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  return { COLORS: COLORS, AXIS: AXIS, GRID: GRID, TEXT: TEXT, draw: draw, esc: esc, fmt: fmt, table: table };
})();
