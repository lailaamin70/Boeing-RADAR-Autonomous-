"""
Cross-scene analysis: aggregate per-modality sensor stats for comparison
across scenes / conditions.

Two tiers of metrics, mirroring the cost tradeoff already hit in the
notebook analysis:

  - "cheap" metrics come straight out of the JSON tables already loaded in
    memory (sample_annotation counts, visibility, sample_data counts).
    Cheap enough to compute across every scene, every request.
  - "expensive" metrics open the actual sensor sweep binaries to compute
    point counts, range/azimuth coverage, velocity, RCS. To keep the page
    responsive these are computed from ONE representative sample per scene
    (the first keyframe) rather than every sample, and cached in-process
    per scene token so repeat requests (e.g. toggling metrics) are fast.

Both caches are plain in-memory dicts that live for the life of the Flask
process. Fine for a mini-split demo. For trainval-scale browsing, swap
`_expensive_cache` for a cache persisted to disk (or a small SQLite table)
built by an offline precompute pass, rather than computing on first request.
"""
from collections import defaultdict

import numpy as np

from app import truckscenes_loader as tsl

# metric_id -> (label, unit, applicable modalities, cheap?)
METRICS = {
    "num_samples": ("Keyframes in scene", "", ["general"], True),
    "visibility": ("Avg. annotation visibility", "level", ["general"], True),
    "annotated_lidar_pts": ("Avg. lidar pts / annotated object", "pts", ["lidar"], True),
    "annotated_radar_pts": ("Avg. radar pts / annotated object", "pts", ["radar"], True),
    "camera_frame_count": ("Camera frames in scene", "", ["camera"], True),
    "sweep_point_count": ("Points per sweep", "pts", ["lidar", "radar"], False),
    "range_coverage": ("Max range", "m", ["lidar", "radar"], False),
    "azimuth_coverage": ("Azimuth spread", "deg", ["lidar", "radar"], False),
    "mean_speed": ("Mean |velocity|", "m/s", ["radar"], False),
    "mean_rcs": ("Mean RCS", "dBsm", ["radar"], False),
}

_cheap_cache = {}
_expensive_cache = {}


def metric_catalog():
    """Metric metadata for building the filter UI."""
    return [
        {"id": mid, "label": label, "unit": unit, "modalities": mods, "cheap": cheap}
        for mid, (label, unit, mods, cheap) in METRICS.items()
    ]


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _get_sample_anns(trucksc, sample):
    """
    Sample annotation tokens for a sample. Assumes the devkit adds a
    reverse-lookup `sample['anns']` list at load time (standard nuScenes-
    style behaviour). Falls back to a linear scan if that assumption turns
    out to be wrong for this devkit fork - verify with a quick
    `print(trucksc.sample[0].keys())` if this path gets hit a lot, since
    the fallback is O(n) per call.
    """
    anns = sample.get("anns")
    if anns is not None:
        return anns
    return [a["token"] for a in trucksc.sample_annotation if a["sample_token"] == sample["token"]]


def _compute_cheap(scene_token):
    trucksc = tsl.get_trucksc()
    samples = tsl.list_samples_for_scene(scene_token)

    vis_levels, lidar_ann_pts, radar_ann_pts = [], [], []
    camera_frames = 0

    for sample in samples:
        for ann_token in _get_sample_anns(trucksc, sample):
            ann = trucksc.get("sample_annotation", ann_token)
            if ann.get("visibility_token"):
                vis = trucksc.get("visibility", ann["visibility_token"])
                level = vis.get("level")
                if isinstance(level, (int, float)):
                    vis_levels.append(level)
            lidar_ann_pts.append(ann["num_lidar_pts"])
            radar_ann_pts.append(ann["num_radar_pts"])
        for _, entry in tsl.get_sample_data_by_channel(sample["token"]).items():
            if entry["modality"] == "camera":
                camera_frames += 1

    return {
        "general": {"num_samples": len(samples), "visibility": _mean(vis_levels)},
        "lidar": {"annotated_lidar_pts": _mean(lidar_ann_pts)},
        "radar": {"annotated_radar_pts": _mean(radar_ann_pts)},
        "camera": {"camera_frame_count": camera_frames},
    }


def _compute_expensive(scene_token):
    samples = tsl.list_samples_for_scene(scene_token)
    if not samples:
        return {"lidar": {}, "radar": {}}

    first_sample = samples[0]
    lidar_stats, radar_stats = defaultdict(list), defaultdict(list)

    for _, entry in tsl.get_sample_data_by_channel(first_sample["token"]).items():
        sd_token = entry["sample_data"]["token"]

        if entry["modality"] == "lidar":
            pts = tsl.load_lidar_points(sd_token)
            if len(pts) == 0:
                continue
            r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
            az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
            lidar_stats["sweep_point_count"].append(len(pts))
            lidar_stats["range_coverage"].append(float(r.max()))
            lidar_stats["azimuth_coverage"].append(float(az.max() - az.min()))

        elif entry["modality"] == "radar":
            pts = tsl.load_radar_points(sd_token)
            if len(pts) == 0:
                continue
            x, y = pts[:, tsl.RADAR_ROW["x"]], pts[:, tsl.RADAR_ROW["y"]]
            vx, vy = pts[:, tsl.RADAR_ROW["vx"]], pts[:, tsl.RADAR_ROW["vy"]]
            rcs = pts[:, tsl.RADAR_ROW["rcs"]]
            r = np.sqrt(x ** 2 + y ** 2)
            az = np.degrees(np.arctan2(y, x))
            speed = np.sqrt(vx ** 2 + vy ** 2)
            radar_stats["sweep_point_count"].append(len(pts))
            radar_stats["range_coverage"].append(float(r.max()))
            radar_stats["azimuth_coverage"].append(float(az.max() - az.min()))
            radar_stats["mean_speed"].append(float(speed.mean()))
            radar_stats["mean_rcs"].append(float(rcs.mean()))

    return {
        "lidar": {k: _mean(v) for k, v in lidar_stats.items()},
        "radar": {k: _mean(v) for k, v in radar_stats.items()},
    }


def compute_scene_metrics(scene_token, include_expensive=False):
    if scene_token not in _cheap_cache:
        _cheap_cache[scene_token] = _compute_cheap(scene_token)
    result = {k: dict(v) for k, v in _cheap_cache[scene_token].items()}

    if include_expensive:
        if scene_token not in _expensive_cache:
            _expensive_cache[scene_token] = _compute_expensive(scene_token)
        for modality, vals in _expensive_cache[scene_token].items():
            result.setdefault(modality, {}).update(vals)

    return result


def aggregate_by_condition(
    selected_tags, selected_metrics, group_by="tag", include_expensive=False, max_scenes=200
):
    """
    Build chart-ready data: {labels, metrics: {id: {label, unit, series: {modality: [values]}}}, scene_count}

    group_by="tag": one group per selected condition tag (or every tag seen,
        if none selected), values averaged across scenes carrying that tag.
    group_by="scene": one group per scene (filtered by tag if any selected).
    """
    scenes = tsl.list_scenes()[:max_scenes]  # cap for responsiveness; see README

    if group_by == "scene":
        if selected_tags:
            scenes = [s for s in scenes if any(t in s["tags"] for t in selected_tags)]
        groups = [(s["name"], [s]) for s in scenes]
    else:
        tags_to_use = selected_tags or sorted({t for s in scenes for t in s["tags"]})
        groups = [(tag, [s for s in scenes if tag in s["tags"]]) for tag in tags_to_use]
        groups = [g for g in groups if g[1]]

    involved = {s["token"]: s for _, gs in groups for s in gs}
    stats_by_token = {
        token: compute_scene_metrics(token, include_expensive=include_expensive)
        for token in involved
    }

    labels = [g[0] for g in groups]
    metrics_out = {}
    for metric_id in selected_metrics:
        if metric_id not in METRICS:
            continue
        label, unit, modalities, _cheap = METRICS[metric_id]
        modality_series = {}
        for modality in modalities:
            series = []
            for _, group_scenes in groups:
                values = [
                    stats_by_token[s["token"]].get(modality, {}).get(metric_id)
                    for s in group_scenes
                ]
                m = _mean(values)
                series.append(round(m, 3) if m is not None else None)
            modality_series[modality] = series
        metrics_out[metric_id] = {"label": label, "unit": unit, "series": modality_series}

    return {"labels": labels, "metrics": metrics_out, "scene_count": len(involved)}
