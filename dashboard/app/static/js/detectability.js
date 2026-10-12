(function () {
  const $ = (id) => document.getElementById(id);
  const checked = (sel) => Array.from(document.querySelectorAll(sel)).filter((el) => el.checked).map((el) => el.value);

  const COMBO_COLORS = {
    "lidar + radar + camera": "#f2e5c9",
    "lidar + radar": "#c9b458",
    "lidar + camera": "#5fd0b0",
    "radar + camera": "#d98fb0",
    "lidar only": TS.COLORS.lidar,
    "radar only": TS.COLORS.radar,
    "camera only": TS.COLORS.camera,
    none: "#39414f",
  };

  let last = null;

  function binLabels(d) {
    // two-line tick labels: the distance bin and how many objects it holds
    return d.bins.map((b, i) => [b + " m", "n=" + d.n[i]]);
  }

  function lineSet(label, color, data) {
    return { label: label, data: data, borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 3, tension: 0.15, spanGaps: false };
  }

  function renderCombos() {
    if (!last) return;
    const d = last;
    const pct = $("combo-percent").checked;
    const names = Object.keys(d.combos);
    const datasets = names.map((name) => ({
      label: name,
      backgroundColor: COMBO_COLORS[name] || TS.COLORS.general,
      data: d.combos[name].map((count, i) => (pct ? (d.combo_n[i] ? (count / d.combo_n[i]) * 100 : null) : count)),
    }));
    const labels = d.bins.map((b, i) => [b + " m", "n=" + d.combo_n[i]]);
    TS.draw("combos", $("c-combos"), "bar", labels, datasets, {
      stacked: true,
      yTitle: pct ? "% of objects" : "objects",
      y: pct ? { max: 100 } : {},
    });
    $("combo-note").textContent = d.dropped_unknown_visibility
      ? `${d.dropped_unknown_visibility} object(s) with no visibility annotation are left out of this chart.`
      : "";
  }

  function renderSummary(d) {
    const cell = (name, s) =>
      `<div class="col-6 col-lg-3"><div class="panel"><div class="badge-${name}" style="font-size:.8rem">${name}</div>` +
      `<div class="mono">${TS.fmt(s.mean, 1)} <span class="text-dim">mean pts</span></div>` +
      `<div class="mono">${TS.fmt(s.zero_pct, 1)}% <span class="text-dim">zero points</span></div>` +
      `<div class="mono">${TS.fmt(s.seen_pct, 1)}% <span class="text-dim">seen</span></div></div></div>`;
    $("summary").innerHTML =
      `<div class="row g-3">` +
      `<div class="col-6 col-lg-3"><div class="panel"><div class="text-dim" style="font-size:.8rem">objects</div><div class="h4 mb-0 mono">${TS.fmt(d.total, 0)}</div></div></div>` +
      cell("lidar", d.summary.lidar) + cell("radar", d.summary.radar) + `</div>`;
  }

  function renderTable(d) {
    const rows = d.bins.map((b, i) => ({
      bin: b, n: d.n[i],
      lm: d.lidar.mean[i], lz: d.lidar.zero_pct[i], ls: d.lidar.seen_pct[i],
      rm: d.radar.mean[i], rz: d.radar.zero_pct[i], rs: d.radar.seen_pct[i],
    }));
    $("detail-table").innerHTML = TS.table(
      [
        { label: "Distance (m)", get: (r) => r.bin },
        { label: "Objects", align: "right", get: (r) => TS.fmt(r.n, 0) },
        { label: "Lidar mean", align: "right", get: (r) => TS.fmt(r.lm, 1) },
        { label: "Lidar 0 pts %", align: "right", get: (r) => TS.fmt(r.lz, 1) },
        { label: "Lidar seen %", align: "right", get: (r) => TS.fmt(r.ls, 1) },
        { label: "Radar mean", align: "right", get: (r) => TS.fmt(r.rm, 1) },
        { label: "Radar 0 pts %", align: "right", get: (r) => TS.fmt(r.rz, 1) },
        { label: "Radar seen %", align: "right", get: (r) => TS.fmt(r.rs, 1) },
      ],
      rows
    );
  }

  function render(d) {
    last = d;
    $("status").textContent = d.total
      ? `${d.total.toLocaleString()} annotated objects. Small bins are noisy - check n under each tick.`
      : "No annotated objects match these filters.";
    renderSummary(d);

    const labels = binLabels(d);
    TS.draw("mean", $("c-mean"), "line", labels, [lineSet("lidar", TS.COLORS.lidar, d.lidar.mean), lineSet("radar", TS.COLORS.radar, d.radar.mean)], {
      yTitle: "mean points on object", xTitle: "distance from truck",
    });
    TS.draw("seen", $("c-seen"), "line", labels, [
      lineSet(`lidar (\u2265 ${d.params.min_lidar})`, TS.COLORS.lidar, d.lidar.seen_pct),
      lineSet(`radar (\u2265 ${d.params.min_radar})`, TS.COLORS.radar, d.radar.seen_pct),
    ], { yTitle: "% of objects", xTitle: "distance from truck", y: { max: 100 } });
    renderCombos();
    renderTable(d);
  }

  async function run() {
    const params = new URLSearchParams();
    checked(".cat-check").forEach((c) => params.append("category", c));
    checked(".tag-check").forEach((t) => params.append("tag", t));
    params.set("min_lidar", $("min-lidar").value || "5");
    params.set("min_radar", $("min-radar").value || "1");
    if ($("include-camera").checked) {
      params.set("camera", "1");
      params.set("min_visibility", $("min-vis").value);
    }
    $("status").textContent = "Loading...";
    try {
      const res = await fetch(`${window.DETECTABILITY_API}?${params.toString()}`);
      if (!res.ok) throw new Error(`request failed: ${res.status}`);
      render(await res.json());
    } catch (err) {
      $("status").textContent = `Couldn't load detectability: ${err.message}`;
    }
  }

  $("apply-btn").addEventListener("click", run);
  $("include-camera").addEventListener("change", () => {
    $("min-vis").disabled = !$("include-camera").checked;
    run();
  });
  $("combo-percent").addEventListener("change", renderCombos);
  run();
})();
