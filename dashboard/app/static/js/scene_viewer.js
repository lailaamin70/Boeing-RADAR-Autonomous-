const samples = JSON.parse(document.getElementById("scene-data").textContent);

let frameIndex = 0;
let playing = false;
let playTimer = null;

const els = {
  slider: document.getElementById("frame-slider"),
  label: document.getElementById("frame-label"),
  timestamp: document.getElementById("timestamp-label"),
  playBtn: document.getElementById("play-btn"),
  speed: document.getElementById("speed-select"),
  topdownImg: document.getElementById("topdown-img"),
  topdownWrap: document.getElementById("topdown-wrap"),
  plot3dWrap: document.getElementById("plot3d-wrap"),
  cameraImg: document.getElementById("camera-img"),
  cameraCol: document.getElementById("camera-col"),
  cameraSelect: document.getElementById("camera-select"),
  mainTitle: document.getElementById("main-view-title"),
  showDetections: document.getElementById("show-detections"),
  scoreWrap: document.getElementById("score-threshold-wrap"),
  scoreSlider: document.getElementById("score-threshold"),
  scoreLabel: document.getElementById("score-threshold-label"),
};

els.slider.max = Math.max(samples.length - 1, 0);

let scoreTouched = false;

const MODALITY_COLOR = { radar: "#e2793d", lidar: "#4bb4c4" };
const ANNOTATION_COLOR = "#f2e5c9";
const DETECTION_COLOR = { radar: "#ff4d8d", lidar: "#c6ff4d" };

// Box edges: 0-3 front face, 4-7 rear face
const BOX_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

function selectedSensors() {
  return Array.from(document.querySelectorAll(".sensor-check"))
    .filter((el) => el.checked)
    .map((el) => el.value);
}

function viewMode() {
  return document.querySelector('input[name="view_mode"]:checked').value;
}

function showAnnotations() {
  return document.getElementById("show-annotations").checked;
}

function showDetections() {
  return els.showDetections.checked;
}

function scoreThreshold() {
  return scoreTouched ? els.scoreSlider.value : null;
}

function mediaUrl(filename) {
  return window.SCENE_ENDPOINTS.media.replace("__FILE__", filename);
}

function withParams(base, sensors, annotations, detections, minScore) {
  const params = new URLSearchParams();
  sensors.forEach((s) => params.append("sensor", s));
  if (annotations) params.set("annotations", "1");
  if (detections) {
    params.set("detections", "1");
    if (minScore !== null) params.set("min_score", minScore);
  }
  return `${base}?${params.toString()}`;
}

function topdownUrl(token, sensors, annotations, detections, minScore) {
  return withParams(
    window.SCENE_ENDPOINTS.topdown.replace("__TOKEN__", token),
    sensors, annotations, detections, minScore
  );
}

function pointsUrl(token, sensors, annotations, detections, minScore) {
  return withParams(
    window.SCENE_ENDPOINTS.points.replace("__TOKEN__", token),
    sensors, annotations, detections, minScore
  );
}

function addBoxTrace(traces, box, { color, dash, width, name, legendgroup, showlegend }) {
  const c = box.corners;
  const x = [], y = [], z = [];
  BOX_EDGES.forEach(([a, b]) => {
    x.push(c[a][0], c[b][0], null);
    y.push(c[a][1], c[b][1], null);
    z.push(c[a][2], c[b][2], null);
  });
  traces.push({
    type: "scatter3d",
    mode: "lines",
    name,
    legendgroup,
    showlegend,
    x, y, z,
    line: { color, width, dash },
    hoverinfo: "skip",
  });
}

async function render3D(sample, sensors, annotations, detections, minScore) {
  if (sensors.length === 0) {
    Plotly.purge(els.plot3dWrap);
    return;
  }

  let data;
  try {
    const res = await fetch(pointsUrl(sample.token, sensors, annotations, detections, minScore));
    data = await res.json();
  } catch (err) {
    els.plot3dWrap.innerHTML = `<div class="text-dim p-3">Couldn't load points: ${err.message}</div>`;
    return;
  }

  const traces = Object.entries(data.modalities).map(([modality, pts]) => ({
    type: "scatter3d",
    mode: "markers",
    name: modality,
    x: pts.x,
    y: pts.y,
    z: pts.z,
    marker: { size: 1.5, color: MODALITY_COLOR[modality] || "#888", opacity: 0.7 },
  }));

  (data.annotations || []).forEach((box, i) => {
    addBoxTrace(traces, box, {
      color: ANNOTATION_COLOR,
      dash: "solid",
      width: 3,
      name: "annotations",
      legendgroup: "annotations",
      showlegend: i === 0,
    });
  });

  const seenDetModalities = new Set();
  (data.detections || []).forEach((box) => {
    const first = !seenDetModalities.has(box.modality);
    seenDetModalities.add(box.modality);
    addBoxTrace(traces, box, {
      color: DETECTION_COLOR[box.modality] || "#ffffff",
      dash: "dash",
      width: 2,
      name: `${box.modality} detections`,
      legendgroup: `det-${box.modality}`,
      showlegend: first,
    });
  });

  const layout = {
    paper_bgcolor: "#1a2029",
    plot_bgcolor: "#1a2029",
    font: { color: "#dde3ea" },
    margin: { l: 0, r: 0, t: 10, b: 0 },
    scene: {
      xaxis: { title: "x (m)", color: "#8b93a3", gridcolor: "#2a3140" },
      yaxis: { title: "y (m)", color: "#8b93a3", gridcolor: "#2a3140" },
      zaxis: { title: "z (m)", color: "#8b93a3", gridcolor: "#2a3140" },
      aspectmode: "data",
    },
    legend: { font: { color: "#dde3ea" } },
    showlegend: true,
  };

  Plotly.react(els.plot3dWrap, traces, layout, { displayModeBar: false });
}

function updateCamera(sample) {
  if (!els.cameraSelect) {
    els.cameraCol.style.display = "none";
    return;
  }
  const channel = els.cameraSelect.value;
  const info = sample.channels[channel];
  if (!channel || !info || !info.filename) {
    els.cameraCol.style.display = "none";
    return;
  }
  els.cameraCol.style.display = "";
  els.cameraImg.src = mediaUrl(info.filename);
}

function updateFrame() {
  if (samples.length === 0) return;
  const sample = samples[frameIndex];
  els.slider.value = frameIndex;
  els.label.textContent = `${frameIndex + 1} / ${samples.length}`;
  els.timestamp.textContent = sample.timestamp;

  const sensors = selectedSensors();
  const annotations = showAnnotations();
  const detections = showDetections();
  const minScore = scoreThreshold();
  const mode = viewMode();

  els.scoreWrap.style.display = detections ? "" : "none";
  els.scoreLabel.textContent = scoreTouched ? Number(els.scoreSlider.value).toFixed(2) : "default";

  if (mode === "topdown") {
    els.mainTitle.textContent = "Top-down";
    els.topdownWrap.style.display = "";
    els.plot3dWrap.style.display = "none";
    els.topdownImg.src = sensors.length
      ? topdownUrl(sample.token, sensors, annotations, detections, minScore)
      : "";
  } else {
    els.mainTitle.textContent = "3D";
    els.topdownWrap.style.display = "none";
    els.plot3dWrap.style.display = "";
    render3D(sample, sensors, annotations, detections, minScore);
  }

  updateCamera(sample);
}

function stepFrame(delta) {
  frameIndex = Math.max(0, Math.min(samples.length - 1, frameIndex + delta));
  updateFrame();
  if (frameIndex === samples.length - 1 && playing) {
    stopPlayback();
  }
}

function startPlayback() {
  if (samples.length <= 1) return;

  if (frameIndex >= samples.length - 1) {
    frameIndex = 0;
    updateFrame();
  }

  playing = true;
  els.playBtn.textContent = "Pause";

  const tick = () => {
    if (!playing) return;
    if (frameIndex >= samples.length - 1) {
      stopPlayback();
      return;
    }

    stepFrame(1);
    playTimer = setTimeout(tick, Number(els.speed.value));
  };
  playTimer = setTimeout(tick, Number(els.speed.value));
}

function stopPlayback() {
  playing = false;
  els.playBtn.textContent = "Play";
  clearTimeout(playTimer);
}

els.playBtn.addEventListener("click", () => (playing ? stopPlayback() : startPlayback()));
els.slider.addEventListener("input", (e) => {
  stopPlayback();
  frameIndex = Number(e.target.value);
  updateFrame();
});
document.querySelectorAll('input[name="view_mode"]').forEach((el) => el.addEventListener("change", updateFrame));
document.querySelectorAll(".sensor-check").forEach((el) => el.addEventListener("change", updateFrame));
document.getElementById("show-annotations").addEventListener("change", updateFrame);
els.showDetections.addEventListener("change", updateFrame);
els.scoreSlider.addEventListener("input", () => {
  scoreTouched = true;
  updateFrame();
});
if (els.cameraSelect) {
  els.cameraSelect.addEventListener("change", () => updateCamera(samples[frameIndex]));
}

if (samples.length > 0) {
  updateFrame();
} else {
  els.label.textContent = "No samples in this scene";
}
