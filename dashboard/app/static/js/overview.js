(function () {
  const $ = (id) => document.getElementById(id);

  function tiles(counts) {
    $("tiles").innerHTML = Object.entries(counts)
      .map(
        ([label, value]) =>
          `<div class="col-6 col-md-4 col-xl-2"><div class="panel"><div class="text-dim" style="font-size:.75rem">${TS.esc(label)}</div>` +
          `<div class="h4 mb-0 mono">${TS.fmt(value, 0)}</div></div></div>`
      )
      .join("");
  }

  function barH(key, canvasId, rows, labelKey, color) {
    TS.draw(
      key, $(canvasId), "bar", rows.map((r) => r[labelKey]),
      [{ label: "count", data: rows.map((r) => r.count !== undefined ? r.count : r.annotations), backgroundColor: color }],
      { indexAxis: "y", legend: false, aspectRatio: Math.max(1, 3.2 - rows.length * 0.06) }
    );
  }

  function renderOverview(d) {
    $("status").textContent = d.note || "";
    tiles(d.counts);

    TS.draw(
      "categories", $("c-categories"), "bar", d.categories.map((c) => c.name),
      [
        { label: "annotations", data: d.categories.map((c) => c.annotations), backgroundColor: TS.COLORS.lidar },
        { label: "instances", data: d.categories.map((c) => c.instances), backgroundColor: TS.COLORS.radar },
      ],
      { indexAxis: "y", aspectRatio: Math.max(0.9, 2.4 - d.categories.length * 0.05) }
    );

    TS.draw(
      "distance", $("c-distance"), "bar", d.distance.bins,
      [{ label: "annotations", data: d.distance.counts, backgroundColor: TS.COLORS.lidar }],
      { legend: false, xTitle: "distance from truck (m)", yTitle: "annotations" }
    );

    TS.draw(
      "visibility", $("c-visibility"), "bar", d.visibility.map((v) => v.label),
      [{ label: "annotations", data: d.visibility.map((v) => v.count), backgroundColor: TS.COLORS.camera }],
      { legend: false, yTitle: "annotations" }
    );

    if (d.attributes.length) barH("attributes", "c-attributes", d.attributes, "name", TS.COLORS.general);
    if (d.tags.length) barH("tags", "c-tags", d.tags, "name", TS.COLORS.general);

    $("scenes-table").innerHTML = TS.table(
      [
        { label: "Scene", html: true, get: (r) => `<a href="${window.SCENE_URL.replace("__TOKEN__", r.token)}">${TS.esc(r.name)}</a>` },
        { label: "Tags", html: true, get: (r) => r.tags.map((t) => `<span class="tag">${TS.esc(t)}</span>`).join("") },
        { label: "Samples", align: "right", get: (r) => TS.fmt(r.samples, 0) },
        { label: "Annotations", align: "right", get: (r) => TS.fmt(r.annotations, 0) },
        { label: "Mean speed (km/h)", align: "right", get: (r) => TS.fmt(r.mean_speed_kmh, 1) },
        { label: "Max speed (km/h)", align: "right", get: (r) => TS.fmt(r.max_speed_kmh, 1) },
      ],
      d.scenes
    );
  }

  function renderVolume(v) {
    $("volume-status").textContent =
      v.note + (v.missing_files ? ` ${v.missing_files} listed file(s) were not found on disk and are excluded.` : "");

    TS.draw(
      "volume", $("c-volume"), "bar", v.channels.map((c) => c.channel),
      [{ label: "MB/s", data: v.channels.map((c) => c.mb_per_s), backgroundColor: v.channels.map((c) => TS.COLORS[c.modality] || TS.COLORS.general) }],
      { indexAxis: "y", legend: false, aspectRatio: Math.max(0.8, 2.2 - v.channels.length * 0.07), xTitle: "MB per second" }
    );

    $("volume-table").innerHTML = TS.table(
      [
        { label: "Channel", get: (c) => c.channel },
        { label: "Modality", html: true, get: (c) => `<span class="tag badge-${TS.esc(c.modality)}">${TS.esc(c.modality)}</span>` },
        { label: "Files", align: "right", get: (c) => TS.fmt(c.files, 0) },
        { label: "Keyframes", align: "right", get: (c) => TS.fmt(c.keyframe_files, 0) },
        { label: "Total MB", align: "right", get: (c) => TS.fmt(c.total_mb, 1) },
        { label: "KB / file", align: "right", get: (c) => TS.fmt(c.mean_kb, 1) },
        { label: "Files / s", align: "right", get: (c) => TS.fmt(c.hz, 2) },
        { label: "MB / s", align: "right", get: (c) => TS.fmt(c.mb_per_s, 3) },
      ],
      v.channels
    );

    $("volume-modalities").innerHTML = v.modalities
      .map((m) => `<span class="tag badge-${TS.esc(m.modality)}" style="font-size:.8rem">${TS.esc(m.modality)}: ${TS.fmt(m.mb_per_s, 3)} MB/s, ${TS.fmt(m.total_mb, 0)} MB total</span>`)
      .join(" ");
  }

  async function getJson(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} failed: ${res.status}`);
    return res.json();
  }

  getJson(window.OVERVIEW_API).then(renderOverview).catch((e) => ($("status").textContent = `Couldn't load overview: ${e.message}`));
  getJson(window.VOLUME_API).then(renderVolume).catch((e) => ($("volume-status").textContent = `Couldn't scan file sizes: ${e.message}`));
})();
