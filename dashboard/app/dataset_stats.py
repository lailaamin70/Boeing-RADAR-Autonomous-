"""
Dataset-level statistics computed from the JSON tables. Used for the
Overview page, the Detectability page and the ego-motion section of the scene page.

Everything except data volume is cached in memory for the life of the
process. Data volume stats ~every file once, so its result is also cached
on disk (Config.DATA_VOLUME_CACHE_PATH, keyed on dataroot + version).
"""
import json
import math
import os
import re
import threading
from collections import Counter

import numpy as np

from app import truckscenes_loader as tsl
from config import Config

SCENE_SECONDS = 20.0
DIST_EDGES = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 300]
EXCLUDED_CATEGORY_MARKERS = ("ego_",)

COMBO_BY_CODE = {
    7: "lidar + radar + camera",
    3: "lidar + radar",
    5: "lidar + camera",
    6: "radar + camera",
    1: "lidar only",
    2: "radar only",
    4: "camera only",
    0: "none",
}

_table_lock = threading.Lock()
_ego_lock = threading.Lock()
_volume_lock = threading.Lock()
_table = None
_ego = None


# helpers
def bin_labels():
    labels = [f"{lo}-{hi}" for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:])]
    labels[-1] = f"{DIST_EDGES[-2]}+"
    return labels


def _bin_index(dist):
    return np.digitize(dist, DIST_EDGES[1:-1])


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def _level_key(v):
    """Sort key for visibility records: numeric level if it is one, otherwise the first number in it."""
    level = v.get("level")
    if isinstance(level, (int, float)):
        return (0, float(level))
    m = re.search(r"\d+", str(level))
    return (0, float(m.group())) if m else (1, str(v.get("token")))


def _seconds_per_tick(ts):
    ts = abs(float(ts))
    if ts > 1e14:
        return 1e-6
    if ts > 1e11:
        return 1e-3
    return 1.0


# annotation table (one row per non-ego annotation)
def _build_table():
    trucksc = tsl.get_trucksc()

    cat_name = {c["token"]: c["name"] for c in trucksc.category}
    inst_cat = {i["token"]: cat_name.get(i["category_token"], "unknown") for i in trucksc.instance}

    vis_sorted = sorted(trucksc.visibility, key=_level_key)
    vis_rank = {v["token"]: r for r, v in enumerate(vis_sorted, start=1)}
    vis_levels = [
        {"rank": r, "level": v.get("level"), "description": v.get("description", "")}
        for r, v in enumerate(vis_sorted, start=1)
    ]

    attr_name = {a["token"]: a["name"] for a in trucksc.attribute}

    sensor_modality = {s["token"]: s["modality"] for s in trucksc.sensor}
    cs_modality = {c["token"]: sensor_modality.get(c["sensor_token"]) for c in trucksc.calibrated_sensor}
    pose_xy = {e["token"]: (e["translation"][0], e["translation"][1]) for e in trucksc.ego_pose}

    scenes = list(trucksc.scene)
    scene_index = {sc["token"]: i for i, sc in enumerate(scenes)}

    sample_scene, sample_ego = {}, {}
    for s in trucksc.sample:
        sample_scene[s["token"]] = scene_index.get(s["scene_token"], -1)
        chosen = None
        for sd_token in s["data"].values():
            sd = trucksc.get("sample_data", sd_token)
            if cs_modality.get(sd["calibrated_sensor_token"]) == "lidar":
                chosen = sd
                break
            if chosen is None:
                chosen = sd
        if chosen is not None and chosen["ego_pose_token"] in pose_xy:
            sample_ego[s["token"]] = pose_xy[chosen["ego_pose_token"]]

    cat_idx, dist, cat, lid, rad, vis, scn = {}, [], [], [], [], [], []
    attr_counts = Counter()
    for ann in trucksc.sample_annotation:
        cname = inst_cat.get(ann["instance_token"], "unknown")
        if any(mark in cname for mark in EXCLUDED_CATEGORY_MARKERS):
            continue
        ego = sample_ego.get(ann["sample_token"])
        if ego is None:
            continue
        dist.append(math.hypot(ann["translation"][0] - ego[0], ann["translation"][1] - ego[1]))
        cat.append(cat_idx.setdefault(cname, len(cat_idx)))
        lid.append(ann["num_lidar_pts"])
        rad.append(ann["num_radar_pts"])
        vis.append(vis_rank.get(ann.get("visibility_token"), 0))
        scn.append(sample_scene.get(ann["sample_token"], -1))
        for t in ann.get("attribute_tokens") or []:
            attr_counts[attr_name.get(t, t)] += 1

    instance_counts = Counter(
        name for name in inst_cat.values() if not any(m in name for m in EXCLUDED_CATEGORY_MARKERS)
    )

    return {
        "dist": np.array(dist, dtype=float),
        "cat": np.array(cat, dtype=int),
        "lidar": np.array(lid, dtype=int),
        "radar": np.array(rad, dtype=int),
        "vis": np.array(vis, dtype=int),
        "scene": np.array(scn, dtype=int),
        "categories": list(cat_idx),
        "scenes": scenes,
        "scene_tags": [tsl.parse_scene_tags(sc["description"]) for sc in scenes],
        "vis_levels": vis_levels,
        "attr_counts": attr_counts,
        "instance_counts": instance_counts,
    }


def get_table():
    global _table
    if _table is None:
        with _table_lock:
            if _table is None:
                _table = _build_table()
    return _table


def filter_options():
    """Everything the Detectability page needs for the filter UI."""
    t = get_table()
    counts = np.bincount(t["cat"], minlength=len(t["categories"]))
    cats = sorted(
        ({"name": n, "count": int(c)} for n, c in zip(t["categories"], counts)),
        key=lambda x: -x["count"],
    )
    tags = sorted({tag for tags in t["scene_tags"] for tag in tags})
    return {"categories": cats, "tags": tags, "visibility_levels": t["vis_levels"]}


def _mask(t, categories, tags):
    m = np.ones(len(t["dist"]), dtype=bool)
    if categories:
        wanted = [i for i, c in enumerate(t["categories"]) if c in set(categories)]
        m &= np.isin(t["cat"], wanted)
    if tags:
        ok = np.array([any(tag in st for tag in tags) for st in t["scene_tags"]] + [False])
        m &= ok[t["scene"]]  # scene index -1 picks the trailing False
    return m


# detectability + complementarity
def detectability(categories=None, tags=None, min_lidar=5, min_radar=1,
                  include_camera=False, min_visibility=2):
    t = get_table()
    m = _mask(t, categories, tags)
    dist, lid, rad, vis = t["dist"][m], t["lidar"][m], t["radar"][m], t["vis"][m]

    nb = len(DIST_EDGES) - 1
    b = _bin_index(dist)

    def per_bin(values, fn):
        out = []
        for i in range(nb):
            v = values[b == i]
            out.append(None if len(v) == 0 else _r(fn(v)))
        return out

    def modality_block(values, threshold):
        return {
            "mean": per_bin(values, np.mean),
            "median": per_bin(values, np.median),
            "zero_pct": per_bin(values, lambda v: (v == 0).mean() * 100),
            "seen_pct": per_bin(values, lambda v: (v >= threshold).mean() * 100),
        }

    def summary(values, threshold):
        if len(values) == 0:
            return {"mean": None, "zero_pct": None, "seen_pct": None}
        return {
            "mean": _r(values.mean()),
            "zero_pct": _r((values == 0).mean() * 100),
            "seen_pct": _r((values >= threshold).mean() * 100),
        }

    # complementarity: which modalities see each object
    keep = (vis > 0) if include_camera else np.ones(len(vis), dtype=bool)
    cam = (vis >= min_visibility) if include_camera else np.zeros(len(vis), dtype=bool)
    code = (lid >= min_lidar).astype(int) + 2 * (rad >= min_radar).astype(int) + 4 * cam.astype(int)

    combos = {}
    for code_value, name in COMBO_BY_CODE.items():
        if not include_camera and code_value >= 4:
            continue
        combos[name] = [int(((b == i) & keep & (code == code_value)).sum()) for i in range(nb)]

    return {
        "bins": bin_labels(),
        "n": [int((b == i).sum()) for i in range(nb)],
        "lidar": modality_block(lid, min_lidar),
        "radar": modality_block(rad, min_radar),
        "combos": combos,
        "combo_n": [int(((b == i) & keep).sum()) for i in range(nb)],
        "dropped_unknown_visibility": int((~keep).sum()),
        "total": int(m.sum()),
        "summary": {"lidar": summary(lid, min_lidar), "radar": summary(rad, min_radar)},
        "params": {
            "min_lidar": min_lidar, "min_radar": min_radar,
            "include_camera": bool(include_camera), "min_visibility": min_visibility,
        },
    }


# ego motion
_EGO_KEYS = ("vx", "vy", "ax", "yaw_rate", "pitch", "roll")


def _series(rows, keys):
    rows = sorted(rows, key=lambda r: r["timestamp"])
    ts = np.array([r["timestamp"] for r in rows], dtype=np.float64)
    return ts, {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}


def get_ego():
    global _ego
    if _ego is None:
        with _ego_lock:
            if _ego is None:
                trucksc = tsl.get_trucksc()
                poses = sorted(getattr(trucksc, "ego_pose", []), key=lambda r: r["timestamp"])
                _ego = {
                    "chassis": _series(getattr(trucksc, "ego_motion_chassis", []), _EGO_KEYS),
                    "cabin": _series(getattr(trucksc, "ego_motion_cabin", []), _EGO_KEYS),
                    "pose_ts": np.array([p["timestamp"] for p in poses], dtype=np.float64),
                    "pose_xy": np.array(
                        [[p["translation"][0], p["translation"][1]] for p in poses], dtype=float
                    ).reshape(-1, 2),
                }
    return _ego


def _scene_window(scene):
    trucksc = tsl.get_trucksc()
    t0 = trucksc.get("sample", scene["first_sample_token"])["timestamp"]
    t1 = trucksc.get("sample", scene["last_sample_token"])["timestamp"]
    return t0, t1


def _speed_stats(scene):
    ts, cols = get_ego()["chassis"]
    if len(ts) == 0:
        return None, None
    t0, t1 = _scene_window(scene)
    lo, hi = np.searchsorted(ts, t0, side="left"), np.searchsorted(ts, t1, side="right")
    if hi - lo < 2:
        return None, None
    speed = np.hypot(cols["vx"][lo:hi], cols["vy"][lo:hi]) * 3.6
    return _r(speed.mean(), 1), _r(speed.max(), 1)


def ego_motion(scene_token, max_points=400):
    trucksc = tsl.get_trucksc()
    scene = trucksc.get("scene", scene_token)
    t0, t1 = _scene_window(scene)
    scale = _seconds_per_tick(t0)
    ego = get_ego()

    ts, ch = ego["chassis"]
    cb_ts, cb = ego["cabin"]
    lo, hi = np.searchsorted(ts, t0, side="left"), np.searchsorted(ts, t1, side="right")
    if len(ts) == 0 or hi - lo < 2:
        return {"t": [], "note": "No ego-motion rows fall inside this scene's time window."}

    stride = max(1, math.ceil((hi - lo) / max_points))
    sl = slice(lo, hi, stride)
    ts_sl = ts[sl]
    speed = np.hypot(ch["vx"][sl], ch["vy"][sl]) * 3.6

    pitch_diff = roll_diff = None
    if len(cb_ts) > 1:
        pitch_diff = np.degrees(np.interp(ts_sl, cb_ts, cb["pitch"]) - ch["pitch"][sl])
        roll_diff = np.degrees(np.interp(ts_sl, cb_ts, cb["roll"]) - ch["roll"][sl])

    path = {"x": [], "y": []}
    pts, pxy = ego["pose_ts"], ego["pose_xy"]
    plo, phi = np.searchsorted(pts, t0, side="left"), np.searchsorted(pts, t1, side="right")
    if phi - plo >= 2:
        pstride = max(1, math.ceil((phi - plo) / max_points))
        seg = pxy[plo:phi:pstride]
        seg = seg - seg[0]
        path = {"x": [_r(v) for v in seg[:, 0]], "y": [_r(v) for v in seg[:, 1]]}

    all_speed = np.hypot(ch["vx"][lo:hi], ch["vy"][lo:hi]) * 3.6
    return {
        "t": [_r((v - t0) * scale, 3) for v in ts_sl],
        "speed_kmh": [_r(v, 2) for v in speed],
        "ax": [_r(v, 3) for v in ch["ax"][sl]],
        "yaw_rate_dps": [_r(v, 3) for v in np.degrees(ch["yaw_rate"][sl])],
        "pitch_diff_deg": None if pitch_diff is None else [_r(v, 3) for v in pitch_diff],
        "roll_diff_deg": None if roll_diff is None else [_r(v, 3) for v in roll_diff],
        "path": path,
        "stats": {
            "mean_speed_kmh": _r(all_speed.mean(), 1),
            "max_speed_kmh": _r(all_speed.max(), 1),
            "duration_s": _r((ts[hi - 1] - ts[lo]) * scale, 1),
        },
    }


# overview
def overview():
    trucksc = tsl.get_trucksc()
    t = get_table()
    nb = len(DIST_EDGES) - 1

    cat_counts = np.bincount(t["cat"], minlength=len(t["categories"]))
    categories = sorted(
        (
            {"name": n, "annotations": int(c), "instances": int(t["instance_counts"].get(n, 0))}
            for n, c in zip(t["categories"], cat_counts)
        ),
        key=lambda x: -x["annotations"],
    )

    dist_counts = np.bincount(_bin_index(t["dist"]), minlength=nb) if len(t["dist"]) else np.zeros(nb, int)

    vis_counts = np.bincount(t["vis"], minlength=len(t["vis_levels"]) + 1)
    visibility = [{"label": "not annotated", "count": int(vis_counts[0])}] + [
        {"label": v["description"] or str(v["level"]), "count": int(vis_counts[v["rank"]])}
        for v in t["vis_levels"]
    ]

    tag_counts = Counter(tag for tags in t["scene_tags"] for tag in tags)

    valid_scene = t["scene"][t["scene"] >= 0]
    ann_per_scene = np.bincount(valid_scene, minlength=len(t["scenes"]))
    scene_rows = []
    for i, sc in enumerate(t["scenes"]):
        mean_kmh, max_kmh = _speed_stats(sc)
        scene_rows.append({
            "token": sc["token"], "name": sc["name"], "tags": t["scene_tags"][i],
            "samples": sc["nbr_samples"], "annotations": int(ann_per_scene[i]),
            "mean_speed_kmh": mean_kmh, "max_speed_kmh": max_kmh,
        })

    sample_data = list(trucksc.sample_data)
    return {
        "counts": {
            "scenes": len(t["scenes"]),
            "samples": len(trucksc.sample),
            "instances": sum(t["instance_counts"].values()),
            "annotations": int(len(t["dist"])),
            "sample_data files": len(sample_data),
            "keyframe files": sum(1 for sd in sample_data if sd.get("is_key_frame")),
        },
        "categories": categories,
        "distance": {"bins": bin_labels(), "counts": [int(c) for c in dist_counts]},
        "visibility": visibility,
        "attributes": [{"name": n, "count": c} for n, c in t["attr_counts"].most_common()],
        "tags": [{"name": n, "count": c} for n, c in sorted(tag_counts.items(), key=lambda x: -x[1])],
        "scenes": scene_rows,
        "note": "Annotations of the ego rig (vehicle.ego_trailer) are excluded.",
    }


# data volume (stats every file once -> cached on disk)
def _compute_volume():
    trucksc = tsl.get_trucksc()
    dataroot = Config.TRUCKSCENES_DATAROOT

    sensor = {s["token"]: s for s in trucksc.sensor}
    cs_sensor = {c["token"]: sensor.get(c["sensor_token"], {}) for c in trucksc.calibrated_sensor}

    agg = {}
    missing = 0
    for sd in trucksc.sample_data:
        info = cs_sensor.get(sd["calibrated_sensor_token"], {})
        channel = info.get("channel", "unknown")
        entry = agg.setdefault(
            channel, {"channel": channel, "modality": info.get("modality", "unknown"),
                      "files": 0, "keyframe_files": 0, "bytes": 0}
        )
        try:
            size = os.path.getsize(os.path.join(dataroot, sd["filename"]))
        except OSError:
            missing += 1
            continue
        entry["files"] += 1
        entry["bytes"] += size
        if sd.get("is_key_frame"):
            entry["keyframe_files"] += 1

    seconds = max(len(list(trucksc.scene)) * SCENE_SECONDS, 1.0)
    channels = []
    for e in sorted(agg.values(), key=lambda e: (e["modality"], e["channel"])):
        mb = e["bytes"] / 1e6
        channels.append({
            "channel": e["channel"], "modality": e["modality"], "files": e["files"],
            "keyframe_files": e["keyframe_files"], "total_mb": _r(mb, 1),
            "mean_kb": _r(e["bytes"] / 1e3 / e["files"], 1) if e["files"] else None,
            "hz": _r(e["files"] / seconds, 2), "mb_per_s": _r(mb / seconds, 3),
        })

    modalities = {}
    for c in channels:
        m = modalities.setdefault(c["modality"], {"modality": c["modality"], "total_mb": 0.0, "mb_per_s": 0.0})
        m["total_mb"] += c["total_mb"]
        m["mb_per_s"] += c["mb_per_s"]
    for m in modalities.values():
        m["total_mb"], m["mb_per_s"] = _r(m["total_mb"], 1), _r(m["mb_per_s"], 3)

    return {
        "channels": channels,
        "modalities": list(modalities.values()),
        "missing_files": missing,
        "seconds": seconds,
        "note": f"Rates assume {SCENE_SECONDS:.0f} s per scene (per the schema), over all sweeps as well as keyframes.",
    }


def data_volume(force=False):
    key = f"{Config.TRUCKSCENES_DATAROOT}|{Config.TRUCKSCENES_VERSION}"
    path = Config.DATA_VOLUME_CACHE_PATH
    with _volume_lock:
        if not force and os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    cached = json.load(f)
                if cached.get("key") == key:
                    return cached["data"]
            except (json.JSONDecodeError, OSError, KeyError):
                pass  # unreadable cache: just recompute

        data = _compute_volume()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump({"key": key, "data": data}, f)
        os.replace(tmp, path)
        return data
