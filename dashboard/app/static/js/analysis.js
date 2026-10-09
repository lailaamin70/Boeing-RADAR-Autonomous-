const state = { charts: [] };

const MODALITY_COLORS = {
  radar: "#e2793d",
  lidar: "#4bb4c4",
  camera: "#7f9fd9",
  general: "#9aa4b5",
};

function selectedValues(selector) {
  return Array.from(document.querySelectorAll(selector))
    .filter((el) => el.checked)
    .map((el) => el.value);
}

function currentGroupBy() {
  return document.querySelector('input[name="group_by"]:checked').value;
}

async function runAnalysis() {
  const tags = selectedValues(".tag-check");
  const metrics = selectedValues(".metric-check");
  const groupBy = currentGroupBy();
  const expensive = document.getElementById("include-expensive").checked;
  const statusEl = document.getElementById("status");

  if (metrics.length === 0) {
    statusEl.textContent = "Select at least one metric.";
    document.getElementById("charts").innerHTML = "";
    document.getElementById("table-wrap").innerHTML = "";
    return;
  }

  const params = new URLSearchParams();
  tags.forEach((t) => params.append("tag", t));
  metrics.forEach((m) => params.append("metric", m));
  params.set("group_by", groupBy);
  if (expensive) params.set("expensive", "1");

  statusEl.textContent = expensive
    ? "Reading sensor files for the involved scenes - this can take a moment on first run..."
    : "Loading...";

  try {
    const res = await fetch(`${window.ANALYSIS_API}?${params.toString()}`);
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    const data = await res.json();
    render(data);
  } catch (err) {
    statusEl.textContent = `Couldn't load analysis: ${err.message}`;
  }
}

function render(data) {
  document.getElementById(
    "status"
  ).textContent = `${data.scene_count} scene(s) across ${data.labels.length} group(s)`;

  state.charts.forEach((c) => c.destroy());
  state.charts = [];
  const chartsEl = document.getElementById("charts");
  chartsEl.innerHTML = "";

  if (data.labels.length === 0) {
    chartsEl.innerHTML = '<div class="panel text-dim">No scenes match the current filters.</div>';
    document.getElementById("table-wrap").innerHTML = "";
    return;
  }

  Object.entries(data.metrics).forEach(([metricId, metric]) => {
    const wrap = document.createElement("div");
    wrap.className = "panel mb-3";
    wrap.innerHTML = `<div class="mb-2">${metric.label}${
      metric.unit ? " (" + metric.unit + ")" : ""
    }</div><canvas height="90"></canvas>`;
    chartsEl.appendChild(wrap);

    const ctx = wrap.querySelector("canvas").getContext("2d");
    const datasets = Object.entries(metric.series).map(([modality, values]) => ({
      label: modality,
      data: values,
      backgroundColor: MODALITY_COLORS[modality] || "#888",
    }));

    const chart = new Chart(ctx, {
      type: "bar",
      data: { labels: data.labels, datasets },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#dde3ea" } } },
        scales: {
          x: { ticks: { color: "#8b93a3" }, grid: { color: "#2a3140" } },
          y: {
            ticks: { color: "#8b93a3" },
            grid: { color: "#2a3140" },
            beginAtZero: true,
          },
        },
      },
    });
    state.charts.push(chart);
  });

  renderTable(data);
}

function renderTable(data) {
  const tableWrap = document.getElementById("table-wrap");
  const metricIds = Object.keys(data.metrics);
  if (metricIds.length === 0) {
    tableWrap.innerHTML = "";
    return;
  }

  const cols = [];
  metricIds.forEach((mid) => {
    Object.keys(data.metrics[mid].series).forEach((modality) => {
      cols.push({ mid, modality, label: `${data.metrics[mid].label} · ${modality}` });
    });
  });

  let html = '<table class="table table-sm table-dark table-bordered mono" style="font-size:.8rem">';
  html += `<thead><tr><th>Group</th>${cols
    .map((c) => `<th>${c.label}</th>`)
    .join("")}</tr></thead><tbody>`;
  data.labels.forEach((label, i) => {
    const cells = cols
      .map((c) => {
        const v = data.metrics[c.mid].series[c.modality][i];
        return `<td>${v === null || v === undefined ? "—" : v}</td>`;
      })
      .join("");
    html += `<tr><td>${label}</td>${cells}</tr>`;
  });
  html += "</tbody></table>";
  tableWrap.innerHTML = html;
}

document.getElementById("apply-btn").addEventListener("click", runAnalysis);
document
  .querySelectorAll('input[name="group_by"]')
  .forEach((el) => el.addEventListener("change", runAnalysis));

runAnalysis();
