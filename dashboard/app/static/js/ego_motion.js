(function () {
  const $ = (id) => document.getElementById(id);

  function series(label, color, data) {
    return { label: label, data: data, borderColor: color, backgroundColor: color, borderWidth: 1.5, pointRadius: 0, tension: 0.2 };
  }

  function line(key, canvasId, t, datasets, yTitle) {
    TS.draw(key, $(canvasId), "line", t, datasets, {
      yTitle: yTitle, xTitle: "time in scene (s)", legend: datasets.length > 1,
      y: { beginAtZero: false },
      x: { ticks: { color: TS.AXIS, maxTicksLimit: 10 } },
    });
  }

  function renderPath(path) {
    if (!path.x.length) return;
    const xs = path.x, ys = path.y;
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    // equal scale on both axes so the path's shape isn't distorted
    const span = Math.max(maxX - minX, maxY - minY, 1) * 0.6;
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const pts = xs.map((x, i) => ({ x: x, y: ys[i] }));
    TS.draw("path", $("c-path"), "scatter", [], [
      { label: "ego path", data: pts, showLine: true, borderColor: TS.COLORS.lidar, backgroundColor: TS.COLORS.lidar, pointRadius: 0, borderWidth: 2 },
      { label: "start", data: [pts[0]], borderColor: "#ffffff", backgroundColor: "#ffffff", pointRadius: 5 },
    ], {
      aspectRatio: 1.4, xTitle: "x (m)", yTitle: "y (m)",
      x: { min: cx - span, max: cx + span, ticks: { color: TS.AXIS } },
      y: { min: cy - span, max: cy + span, beginAtZero: false },
    });
  }

  function render(d) {
    if (!d.t || !d.t.length) {
      $("ego-status").textContent = d.note || "No ego-motion data for this scene.";
      return;
    }
    const st = d.stats;
    $("ego-status").textContent =
      `Mean speed ${TS.fmt(st.mean_speed_kmh, 1)} km/h, max ${TS.fmt(st.max_speed_kmh, 1)} km/h, over ${TS.fmt(st.duration_s, 1)} s.`;
    $("ego-charts").style.display = "";

    line("speed", "c-speed", d.t, [series("speed", TS.COLORS.lidar, d.speed_kmh)], "km/h");
    line("accel", "c-accel", d.t, [series("ax", TS.COLORS.radar, d.ax)], "m/s\u00b2");
    line("yaw", "c-yaw", d.t, [series("yaw rate", TS.COLORS.camera, d.yaw_rate_dps)], "deg/s");

    const tilt = [];
    if (d.pitch_diff_deg) tilt.push(series("pitch", TS.COLORS.radar, d.pitch_diff_deg));
    if (d.roll_diff_deg) tilt.push(series("roll", TS.COLORS.camera, d.roll_diff_deg));
    if (tilt.length) line("tilt", "c-tilt", d.t, tilt, "deg");
    renderPath(d.path);
  }

  fetch(window.EGO_API)
    .then((res) => {
      if (!res.ok) throw new Error("request failed: " + res.status);
      return res.json();
    })
    .then(render)
    .catch((err) => ($("ego-status").textContent = "Couldn't load ego motion: " + err.message));
})();
