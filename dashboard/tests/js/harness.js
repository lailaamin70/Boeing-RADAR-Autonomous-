"use strict";
// Runs the dashboard's real page scripts against real API output (written by
// tests/test_js_pages.py) inside a fake DOM, with a fake Chart.js and Plotly.
// No browser needed. Usage: node tests/js/harness.js <payload-dir>
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const assert = require("assert");

const DIR = process.argv[2];
if (!DIR) {
  console.error("usage: node harness.js <payload-dir>");
  process.exit(2);
}
const JS_DIR = path.join(__dirname, "..", "..", "app", "static", "js");
const load = (name) => JSON.parse(fs.readFileSync(path.join(DIR, name + ".json"), "utf8"));
const script = (name) => fs.readFileSync(path.join(JS_DIR, name), "utf8");
// arrays made inside the vm sandbox have another realm's prototype, so compare as JSON
const eq = (actual, expected, msg) => assert.strictEqual(JSON.stringify(actual), JSON.stringify(expected), msg);
const tick = (ms = 40) => new Promise((resolve) => setTimeout(resolve, ms));

function makeEnv(opts) {
  const o = Object.assign({ routes: () => null, values: {}, checked: {}, textContent: {}, selectors: {} }, opts);
  const charts = [], fetched = [], listeners = {}, els = {}, plotly = { calls: [], purged: 0 };

  function makeEl(id, init) {
    let html = "";
    const el = Object.assign(
      {
        id, value: "", checked: false, disabled: false, textContent: "", style: {}, children: [], attrs: {}, src: "",
        getContext: () => ({}),
        appendChild(child) { this.children.push(child); return child; },
        setAttribute(k, v) { this.attrs[k] = String(v); },
        getAttribute(k) { return this.attrs[k]; },
        querySelector: () => ({ getContext: () => ({}) }),
        addEventListener(ev, fn) { (listeners[id + ":" + ev] = listeners[id + ":" + ev] || []).push(fn); },
      },
      init || {}
    );

    Object.defineProperty(el, "innerHTML", {
      enumerable: true,
      get: () => html,
      set: (v) => { html = v; if (v === "") el.children = []; },
    });
    return el;
  }

  const el = (id) =>
    els[id] ||
    (els[id] = makeEl(id, {
      value: o.values[id] !== undefined ? o.values[id] : "",
      checked: !!o.checked[id],
      textContent: o.textContent[id] || "",
    }));

  class Chart {
    constructor(ctx, cfg) { this.cfg = cfg; charts.push(this); }
    destroy() { this.destroyed = true; }
  }

  const sandbox = {
    document: {
      getElementById: el,
      querySelectorAll: (sel) => o.selectors[sel] || [],
      querySelector: (sel) => {
        if (sel.endsWith(":checked")) return (o.selectors[sel.slice(0, -8)] || []).find((e) => e.checked) || null;
        return (o.selectors[sel] || [])[0] || null;
      },
      createElement: () => makeEl(undefined),
      createElementNS: (ns, tag) => makeEl(undefined, { tag }),
    },
    fetch: async (url) => {
      fetched.push(url);
      const key = o.routes(url);
      if (!key) return { ok: false, status: 404, json: async () => ({}) };
      return { ok: true, status: 200, json: async () => load(key) };
    },
    Chart,
    Plotly: {
      react: (target, traces, layout, cfg) => plotly.calls.push({ traces, layout, cfg }),
      purge: () => { plotly.purged += 1; },
    },
    console, URLSearchParams, setTimeout, clearTimeout,
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);

  return {
    sandbox, charts, fetched, listeners, els, plotly, makeEl, selectors: o.selectors,
    live: () => charts.filter((c) => !c.destroyed),
    run: (code, name) => vm.runInContext(code, sandbox, { filename: name }),
    fire: (id, ev, arg) => (listeners[id + ":" + ev] || []).forEach((fn) => fn(arg)),
  };
}

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

// overview
test("overview: draws every chart, table and tile from the real payloads", async () => {
  const e = makeEnv({ routes: (u) => (u.includes("volume") ? "volume" : "overview") });
  Object.assign(e.sandbox, { OVERVIEW_API: "/api/overview", VOLUME_API: "/api/data_volume", SCENE_URL: "/scene/__TOKEN__" });
  e.run(script("chart_common.js"), "chart_common.js");
  e.run(script("overview.js"), "overview.js");
  await tick();

  assert.strictEqual(e.charts.length, 6, "categories, distance, visibility, attributes, tags, volume");
  const errors = e.els["status"].textContent + e.els["volume-status"].textContent;
  assert(!/Couldn't/.test(errors), errors);
  const cats = e.charts[0].cfg;
  eq(cats.data.labels, ["vehicle.car", "vehicle.truck", "human.pedestrian.adult"], "sorted by annotation count");
  eq(cats.data.datasets.map((d) => d.data), [[2, 1, 1], [2, 1, 1]], "annotations and instances per category");
  assert.strictEqual(cats.options.indexAxis, "y");
  const table = e.els["scenes-table"].innerHTML;
  assert(table.includes("scene-0001") && table.includes('href="/scene/S"'), "scene row links to its page");
  assert(e.els["volume-table"].innerHTML.includes("CAMERA_FRONT"));
  assert(e.els["volume-modalities"].innerHTML.includes("camera"));
  assert(e.els["tiles"].innerHTML.includes("annotations"));
  const volume = e.charts[5].cfg.data.datasets[0];
  assert.strictEqual(volume.backgroundColor.length, 3, "one colour per channel");
});

test("overview: a failed request shows an error instead of throwing", async () => {
  const e = makeEnv({ routes: () => null });
  Object.assign(e.sandbox, { OVERVIEW_API: "/x", VOLUME_API: "/y", SCENE_URL: "/s/__TOKEN__" });
  e.run(script("chart_common.js"), "chart_common.js");
  e.run(script("overview.js"), "overview.js");
  await tick();
  assert(/Couldn't load overview/.test(e.els["status"].textContent));
  assert(/Couldn't scan/.test(e.els["volume-status"].textContent));
});

// detectability
function detectabilityEnv() {
  const e = makeEnv({
    routes: (u) => (u.includes("camera=1") ? "detectability_camera" : "detectability"),
    values: { "min-lidar": "5", "min-radar": "1", "min-vis": "2" },
    checked: { "combo-percent": true },
  });
  e.sandbox.DETECTABILITY_API = "/api/detectability";
  e.run(script("chart_common.js"), "chart_common.js");
  e.run(script("detectability.js"), "detectability.js");
  return e;
}

test("detectability: charts, null gaps and two-line tick labels", async () => {
  const e = detectabilityEnv();
  await tick();
  assert.strictEqual(e.live().length, 3);
  assert(/annotated objects/.test(e.els["status"].textContent), e.els["status"].textContent);
  const mean = e.live()[0].cfg;
  eq(mean.data.datasets.map((d) => d.label), ["lidar", "radar"]);
  eq(mean.data.labels[2], ["20-30 m", "n=1"], "tick = bin, then object count");
  assert.strictEqual(mean.data.datasets[0].data[2], 50, "a1 has 50 lidar points");
  assert.strictEqual(mean.data.datasets[0].data[0], null, "an empty bin is a gap, not 0");
  assert(e.els["detail-table"].innerHTML.includes("20-30"));
  assert(e.els["summary"].innerHTML.includes("lidar") && e.els["summary"].innerHTML.includes("radar"));
});

test("detectability: overlap chart is stacked 0-100% and toggles to counts without refetching", async () => {
  const e = detectabilityEnv();
  await tick();
  let combos = e.live()[2].cfg;
  assert.strictEqual(combos.options.scales.x.stacked, true);
  assert.strictEqual(combos.options.scales.y.max, 100);
  const both = (c) => c.data.datasets.find((d) => d.label === "lidar + radar").data;
  assert.strictEqual(both(combos)[2], 100, "the only 20-30 object is seen by both");
  assert.strictEqual(both(combos)[0], null, "empty bin stays a gap in percent mode");

  const before = e.fetched.length;
  e.els["combo-percent"].checked = false;
  e.fire("combo-percent", "change");
  combos = e.live()[2].cfg;
  assert.strictEqual(e.fetched.length, before, "toggling percent must not refetch");
  assert.strictEqual(both(combos)[2], 1);
  assert.strictEqual(combos.options.scales.y.title.text, "objects");
});

test("detectability: including the camera refetches and reports dropped objects", async () => {
  const e = detectabilityEnv();
  await tick();
  e.els["include-camera"].checked = true;
  e.fire("include-camera", "change");
  await tick();
  const url = e.fetched[e.fetched.length - 1];
  assert(url.includes("camera=1") && url.includes("min_visibility=2"), url);
  assert.strictEqual(e.els["min-vis"].disabled, false);
  assert(/left out/.test(e.els["combo-note"].textContent), e.els["combo-note"].textContent);
  assert(e.live()[2].cfg.data.datasets.some((d) => d.label === "lidar + radar + camera"));
});

test("detectability: thresholds and filters go into the request", async () => {
  const e = detectabilityEnv();
  e.selectors[".cat-check"] = [e.makeEl("cat-1", { value: "vehicle.car", checked: true }), e.makeEl("cat-2", { value: "vehicle.truck", checked: false })];
  e.selectors[".tag-check"] = [e.makeEl("tag-1", { value: "day", checked: true })];
  await tick();
  e.els["min-lidar"].value = "9";
  e.fire("apply-btn", "click");
  await tick();
  const url = e.fetched[e.fetched.length - 1];
  assert(url.includes("category=vehicle.car") && !url.includes("vehicle.truck"), url);
  assert(url.includes("tag=day") && url.includes("min_lidar=9") && url.includes("min_radar=1"), url);
});

// ego motion
function egoEnv(payload) {
  const e = makeEnv({ routes: () => payload });
  e.sandbox.EGO_API = "/api/scene/S/ego_motion";
  e.run(script("chart_common.js"), "chart_common.js");
  e.run(script("ego_motion.js"), "ego_motion.js");
  return e;
}

test("ego motion: five charts, status from the real stats, equal-scaled path", async () => {
  const e = egoEnv("ego");
  await tick();
  assert.strictEqual(e.charts.length, 5);
  assert(/Mean speed 36/.test(e.els["ego-status"].textContent), e.els["ego-status"].textContent);
  assert.strictEqual(e.els["ego-charts"].style.display, "");
  const path = e.charts.find((c) => c.cfg.type === "scatter").cfg;
  const xs = path.options.scales.x, ys = path.options.scales.y;
  assert(Math.abs(xs.max - xs.min - (ys.max - ys.min)) < 1e-9, "path axes must have equal span or the shape is distorted");
  assert.strictEqual(path.data.datasets[1].data.length, 1, "start marker");
  const tilt = e.charts.find((c) => c.cfg.data.datasets.length === 2);
  eq(tilt.cfg.data.datasets.map((d) => d.label), ["pitch", "roll"]);
});

test("ego motion: a scene without ego rows shows the note and no charts", async () => {
  const e = egoEnv("ego_empty");
  await tick();
  assert.strictEqual(e.charts.length, 0);
  assert(/No ego-motion rows/.test(e.els["ego-status"].textContent));
  // the page keeps #ego-charts hidden in its HTML; the script must not reveal it
  assert(!e.els["ego-charts"] || e.els["ego-charts"].style.display !== "", "charts must stay hidden");
});

// analysis
test("analysis: builds the request from the selections and draws one chart per metric", async () => {
  const e = makeEnv({ routes: () => "analysis", checked: { "include-expensive": false } });
  e.sandbox.ANALYSIS_API = "/api/analysis";
  const metric = (id, v, on) => e.makeEl(id, { value: v, checked: on });
  e.selectors[".metric-check"] = [metric("m1", "num_samples", true), metric("m2", "visibility", true), metric("m3", "mean_speed", false)];
  e.selectors[".tag-check"] = [];
  e.selectors['input[name="group_by"]'] = [e.makeEl("g1", { value: "tag", checked: true })];
  e.run(script("analysis.js"), "analysis.js");
  await tick();

  const url = e.fetched[0];
  assert(url.includes("metric=num_samples") && url.includes("metric=visibility") && url.includes("group_by=tag"), url);
  assert(!url.includes("mean_speed") && !url.includes("expensive"), url);
  assert.strictEqual(e.live().length, 2, "one chart per returned metric");
  assert(/1 scene\(s\) across 2 group\(s\)/.test(e.els["status"].textContent), e.els["status"].textContent);
  assert(e.els["table-wrap"].innerHTML.includes("Keyframes in scene"));
  eq(e.charts[0].cfg.data.labels, ["clear", "day"]);

  e.els["include-expensive"].checked = true;
  e.fire("apply-btn", "click");
  await tick();
  assert(e.fetched[e.fetched.length - 1].includes("expensive=1"));
  assert(/Reading sensor files/.test(e.els["status"].textContent) || /scene\(s\)/.test(e.els["status"].textContent));

  e.selectors[".metric-check"].forEach((m) => (m.checked = false));
  e.fire("apply-btn", "click");
  await tick();
  assert.strictEqual(e.els["status"].textContent, "Select at least one metric.");
  assert.strictEqual(e.els["table-wrap"].innerHTML, "");
});

// scene viewer
function viewerEnv() {
  const page = fs.readFileSync(path.join(DIR, "viewer.html"), "utf8");
  const sceneData = page.match(/<script id="scene-data" type="application\/json">([\s\S]*?)<\/script>/);
  assert(sceneData, "viewer page has no embedded scene-data");
  const inline = (page.match(/<script>[\s\S]*?<\/script>/g) || [])
    .map((s) => s.replace(/^<script>|<\/script>$/g, ""))
    .find((s) => s.includes("SCENE_ENDPOINTS"));
  assert(inline, "viewer page does not define SCENE_ENDPOINTS");

  const e = makeEnv({
    routes: (u) => (u.includes("/overlay.json") ? "overlay" : u.includes("/points.json") ? "points" : null),
    values: { "speed-select": "250", "score-threshold": "0.3", "camera-select": "CAMERA_FRONT", "frame-slider": "0" },
    checked: { "show-annotations": true, "show-ego": true, "show-detections": false },
    textContent: { "scene-data": sceneData[1] },
  });
  const sensor = (v) => e.makeEl("sensor-" + v, { value: v, checked: true });
  e.selectors[".sensor-check"] = [sensor("radar"), sensor("lidar")];
  e.topdown = e.makeEl("view-topdown", { value: "topdown", checked: true });
  e.threeD = e.makeEl("view-3d", { value: "3d", checked: false });
  e.selectors['input[name="view_mode"]'] = [e.topdown, e.threeD];
  e.run(inline, "SCENE_ENDPOINTS");
  e.run(script("scene_viewer.js"), "scene_viewer.js");
  e.query = () => new URLSearchParams(e.els["topdown-img"].src.split("?")[1]);
  e.overlayFetches = () => e.fetched.filter((u) => u.includes("/overlay.json"));
  e.switchTo3D = async () => {
    e.topdown.checked = false;
    e.threeD.checked = true;
    e.fire("view-3d", "change");
    await tick();
  };
  return e;
}

test("viewer: first frame requests the right top-down image, camera image and overlay", async () => {
  const e = viewerEnv();
  await tick();
  assert.strictEqual(e.els["frame-label"].textContent, "1 / 2");
  const src = e.els["topdown-img"].src;
  assert(src.startsWith("/sample/A/topdown.png?"), src);
  eq(e.query().getAll("sensor"), ["radar", "lidar"]);
  assert.strictEqual(e.query().get("annotations"), "1");
  assert.strictEqual(e.query().get("ego"), "1");
  assert(!e.query().has("detections") && !e.query().has("min_score"));
  assert.strictEqual(e.els["main-view-title"].textContent, "Top-down");
  assert.strictEqual(e.els["plot3d-wrap"].style.display, "none");
  assert.strictEqual(e.els["camera-col"].style.display, "");
  assert.strictEqual(e.els["camera-img"].src, "/media/samples/CAMERA_FRONT/cam_A.png");
  assert.strictEqual(e.els["score-threshold-wrap"].style.display, "none", "slider hidden until detections are on");
});

test("viewer: camera overlay draws 12 edges per projected box, solid for ground truth", async () => {
  const e = viewerEnv();
  await tick();
  const payload = load("overlay");
  const overlay = e.els["camera-overlay"];
  assert.strictEqual(e.overlayFetches().length, 1);
  assert(e.overlayFetches()[0].startsWith("/sample/A/camera/CAMERA_FRONT/overlay.json?"));
  assert.strictEqual(overlay.attrs["viewBox"], "0 0 1600 900");
  assert.strictEqual(overlay.children.length, 12 * (payload.annotations.length + payload.detections.length));
  const solid = overlay.children.filter((l) => l.attrs["stroke"] === "#f2e5c9" && !("stroke-dasharray" in l.attrs));
  assert.strictEqual(solid.length, 12 * payload.annotations.length);
  const line = solid[0].attrs;
  ["x1", "y1", "x2", "y2"].forEach((k) => assert(Number.isFinite(Number(line[k])), k + " must be a number"));
});

test("viewer: detections toggle shows the score slider, adds dashed boxes, and only sends min_score once touched", async () => {
  const e = viewerEnv();
  await tick();
  e.els["show-detections"].checked = true;
  e.fire("show-detections", "change");
  await tick();
  assert.strictEqual(e.query().get("detections"), "1");
  assert(!e.query().has("min_score"), "no override until the slider is moved");
  assert.strictEqual(e.els["score-threshold-wrap"].style.display, "");
  assert.strictEqual(e.els["score-threshold-label"].textContent, "default");

  e.els["score-threshold"].value = "0.55";
  e.fire("score-threshold", "input");
  await tick();
  assert.strictEqual(e.query().get("min_score"), "0.55");
  assert.strictEqual(e.els["score-threshold-label"].textContent, "0.55");

  const payload = load("overlay");
  const dashed = e.els["camera-overlay"].children.filter((l) => "stroke-dasharray" in l.attrs);
  assert.strictEqual(dashed.length, 12 * payload.detections.length);
  assert(dashed.every((l) => l.attrs["stroke"] === "#c6ff4d"), "lidar detections use the lidar detection colour");
  assert(e.overlayFetches().pop().includes("min_score=0.55"));
});

test("viewer: unticking a sensor or the ego toggle changes the request; no sensors clears the image", async () => {
  const e = viewerEnv();
  await tick();
  e.els["show-ego"].checked = false;
  e.fire("show-ego", "change");
  assert(!e.query().has("ego"));
  e.selectors[".sensor-check"][0].checked = false; // radar off
  e.fire("sensor-radar", "change");
  eq(e.query().getAll("sensor"), ["lidar"]);
  e.selectors[".sensor-check"][1].checked = false; // lidar off
  e.fire("sensor-lidar", "change");
  assert.strictEqual(e.els["topdown-img"].src, "", "nothing selected -> no image request");
});

test("viewer: the slider jumps frames; playing runs to the end then restarts from frame 1", async () => {
  const e = viewerEnv();
  await tick();
  e.fire("frame-slider", "input", { target: { value: "1" } });
  await tick();
  assert.strictEqual(e.els["frame-label"].textContent, "2 / 2");
  assert(e.els["topdown-img"].src.startsWith("/sample/B/topdown.png?"));
  assert.strictEqual(e.els["camera-img"].src, "/media/samples/CAMERA_FRONT/cam_B.png");

  e.fire("frame-slider", "input", { target: { value: "0" } });
  e.fire("play-btn", "click");
  assert.strictEqual(e.els["play-btn"].textContent, "Pause");
  await tick(400); // speed is 250 ms per frame
  assert.strictEqual(e.els["frame-label"].textContent, "2 / 2");
  assert.strictEqual(e.els["play-btn"].textContent, "Play", "playback stops at the last frame");

  e.fire("play-btn", "click"); // at the end: pressing play starts over
  assert.strictEqual(e.els["frame-label"].textContent, "1 / 2");
  assert.strictEqual(e.els["play-btn"].textContent, "Pause");
  e.fire("play-btn", "click"); // pause, so no timer is left running
  assert.strictEqual(e.els["play-btn"].textContent, "Play");
});

test("viewer: 3D view keeps the camera angle across frames and draws points, truck, sensors and boxes", async () => {
  const e = viewerEnv();
  e.els["show-detections"].checked = true;
  await tick();
  await e.switchTo3D();

  assert.strictEqual(e.els["main-view-title"].textContent, "3D");
  assert.strictEqual(e.els["plot3d-wrap"].style.display, "");
  assert.strictEqual(e.els["topdown-wrap"].style.display, "none");
  const pointsUrl = e.fetched.filter((u) => u.includes("/points.json")).pop();
  assert(pointsUrl.startsWith("/sample/A/points.json?") && pointsUrl.includes("ego=1") && pointsUrl.includes("annotations=1"), pointsUrl);

  const call = e.plotly.calls[e.plotly.calls.length - 1];
  const named = (n) => call.traces.filter((t) => t.name === n);
  assert.strictEqual(named("lidar")[0].marker.color, "#4bb4c4");
  assert.strictEqual(named("radar")[0].marker.color, "#e2793d");
  assert.strictEqual(named("ego truck")[0].type, "mesh3d");
  ["lidar sensors", "radar sensors", "camera sensors"].forEach((n) => assert.strictEqual(named(n).length, 1, n));
  assert.strictEqual(named("camera sensors")[0].marker.symbol, "circle");
  eq(named("lidar sensors")[0].text, ["LIDAR_TOP"], "hover text names the channel");

  const ann = named("annotations");
  assert(ann.length > 0 && ann.every((t) => t.line.dash === "solid" && t.line.color === "#f2e5c9"));
  assert.strictEqual(ann.filter((t) => t.showlegend).length, 1, "one legend entry for all annotation boxes");
  const det = named("lidar detections");
  assert(det.length > 0 && det.every((t) => t.line.dash === "dash" && t.line.color === "#c6ff4d"));
  assert.strictEqual(det.filter((t) => t.showlegend).length, 1);

  assert.strictEqual(call.layout.uirevision, "scene-viewer-3d", "without this the camera snaps back every frame");
  assert.strictEqual(call.layout.scene.uirevision, "scene-viewer-3d");

  e.fire("frame-slider", "input", { target: { value: "1" } });
  await tick();
  const next = e.plotly.calls[e.plotly.calls.length - 1];
  assert.strictEqual(next.layout.uirevision, call.layout.uirevision, "same revision across frames keeps the user's orbit");
});

test("viewer: 3D with no sensors selected purges the plot instead of fetching", async () => {
  const e = viewerEnv();
  await tick();
  e.selectors[".sensor-check"].forEach((s) => (s.checked = false));
  const before = e.fetched.filter((u) => u.includes("/points.json")).length;
  await e.switchTo3D();
  assert.strictEqual(e.plotly.purged, 1);
  assert.strictEqual(e.fetched.filter((u) => u.includes("/points.json")).length, before);
});

test("viewer: camera choice and overlay toggles control the overlay requests", async () => {
  const e = viewerEnv();
  await tick();
  const n = e.overlayFetches().length;

  e.els["show-annotations"].checked = false; // both overlays off -> clear, no request
  e.fire("show-annotations", "change");
  await tick();
  assert.strictEqual(e.els["camera-overlay"].children.length, 0);
  assert.strictEqual(e.overlayFetches().length, n);

  e.els["show-annotations"].checked = true;
  e.els["camera-select"].value = ""; // "None" hides the camera panel
  e.fire("camera-select", "change");
  await tick();
  assert.strictEqual(e.els["camera-col"].style.display, "none");

  e.els["camera-select"].value = "CAMERA_FRONT";
  e.fire("camera-select", "change");
  await tick();
  assert.strictEqual(e.els["camera-col"].style.display, "");
  assert(e.overlayFetches().length > n);
  assert(e.els["camera-overlay"].children.length > 0);
});

// runner
(async () => {
  let failed = 0;
  for (const [name, fn] of tests) {
    try {
      await fn();
      console.log("ok   - " + name);
    } catch (err) {
      failed += 1;
      console.log("FAIL - " + name + "\n       " + ((err && err.stack) || err).toString().split("\n").slice(0, 6).join("\n       "));
    }
  }
  console.log(`\n${tests.length - failed}/${tests.length} JS tests passed`);
  process.exit(failed ? 1 : 0);
})();
