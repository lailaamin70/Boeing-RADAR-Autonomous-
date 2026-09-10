"""
Wrapper around truckscenes-devkit for the dashboard.

This module intentionally keeps ALL devkit-specific quirks in one place so
routes.py never has to know about them. Two things learned the hard way
during notebook development are encoded here:

1. `RadarPointCloud.disable_filters()` does not exist in this fork of the
   devkit. Instead we inspect `RadarPointCloud.from_file`'s signature at
   runtime and only pass kwargs it actually accepts (`RADAR_KWARGS`).
2. The radar point array is a 7-row schema: [x, y, z, vx, vy, vz, rcs] -
   NOT the 18-row nuScenes layout. Row indices below are confirmed, not
   assumed. If you point this at a different devkit version, re-verify
   with `points.shape` before trusting these indices.
"""
import inspect
import io
import os
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud

from config import Config

# Radar row layout
RADAR_ROW = {"x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5, "rcs": 6}

_lock = threading.Lock()
_trucksc = None


def _build_radar_kwargs():
    """
    Build a passthrough kwargs dict for RadarPointCloud.from_file based on
    whatever filter-related parameters this devkit build actually exposes,
    instead of assuming disable_filters() is available.
    """
    try:
        sig = inspect.signature(RadarPointCloud.from_file)
    except (TypeError, ValueError):
        return {}

    candidates = {
        "invalid_states": list(range(18)),
        "dynprop_states": list(range(8)),
        "ambig_states": list(range(5)),
    }
    return {k: v for k, v in candidates.items() if k in sig.parameters}


RADAR_KWARGS = None


def get_trucksc():
    """Return a process-wide singleton TruckScenes instance."""
    global _trucksc, RADAR_KWARGS
    if _trucksc is None:
        with _lock:
            if _trucksc is None:
                _trucksc = TruckScenes(
                    version=Config.TRUCKSCENES_VERSION,
                    dataroot=Config.TRUCKSCENES_DATAROOT,
                    verbose=Config.TRUCKSCENES_VERBOSE,
                )
                RADAR_KWARGS = _build_radar_kwargs()
    return _trucksc


def parse_scene_tags(description):
    if not description:
        return []
    return [t.strip() for t in description.split(";") if t.strip()]


def list_scenes():
    trucksc = get_trucksc()
    scenes = []
    for scene in trucksc.scene:
        scenes.append(
            {
                "token": scene["token"],
                "name": scene["name"],
                "description": scene["description"],
                "tags": parse_scene_tags(scene["description"]),
                "nbr_samples": scene["nbr_samples"],
                "first_sample_token": scene["first_sample_token"],
            }
        )
    return scenes


def get_scene(scene_token):
    trucksc = get_trucksc()
    return trucksc.get("scene", scene_token)


def list_samples_for_scene(scene_token):
    trucksc = get_trucksc()
    scene = trucksc.get("scene", scene_token)
    samples = []
    token = scene["first_sample_token"]
    while token:
        sample = trucksc.get("sample", token)
        samples.append(sample)
        token = sample["next"]
    return samples


def get_sample(sample_token):
    trucksc = get_trucksc()
    return trucksc.get("sample", sample_token)


def get_sample_data_by_channel(sample_token):
    """
    Return {channel_name: sample_data record} for every sensor that has
    data attached to this sample.
    """
    trucksc = get_trucksc()
    sample = trucksc.get("sample", sample_token)
    out = {}
    for channel, sd_token in sample["data"].items():
        sd = trucksc.get("sample_data", sd_token)
        calibrated = trucksc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        sensor = trucksc.get("sensor", calibrated["sensor_token"])
        out[channel] = {"sample_data": sd, "modality": sensor["modality"]}
    return out


def get_full_path(filename):
    return os.path.join(Config.TRUCKSCENES_DATAROOT, filename)


def load_radar_points(sample_data_token):
    """Return an (N, 7) array: columns x,y,z,vx,vy,vz,rcs."""
    trucksc = get_trucksc()
    sd = trucksc.get("sample_data", sample_data_token)
    path = get_full_path(sd["filename"])
    kwargs = RADAR_KWARGS if RADAR_KWARGS is not None else _build_radar_kwargs()
    pc = RadarPointCloud.from_file(path, **kwargs)
    return pc.points.T  # devkit stores points as (rows, N); transpose to (N, rows)


def load_lidar_points(sample_data_token):
    """Return an (N, 4) array: x, y, z, intensity"""
    trucksc = get_trucksc()
    sd = trucksc.get("sample_data", sample_data_token)
    path = get_full_path(sd["filename"])
    pc = LidarPointCloud.from_file(path)
    return pc.points.T


def render_bev_png(sample_data_token, modality, point_size=2, xlim=100, ylim=100):
    """
    Render a top-down scatter of a radar or lidar sweep
    and return it as a PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)

    if modality == "radar":
        points = load_radar_points(sample_data_token)
        x, y = points[:, RADAR_ROW["x"]], points[:, RADAR_ROW["y"]]
        rcs = points[:, RADAR_ROW["rcs"]]
        sc = ax.scatter(x, y, c=rcs, cmap="viridis", s=point_size)
        fig.colorbar(sc, ax=ax, label="RCS (dBsm)", shrink=0.8)
        ax.set_title(f"Radar BEV ({len(points)} points)")
    elif modality == "lidar":
        points = load_lidar_points(sample_data_token)
        x, y = points[:, 0], points[:, 1]
        ax.scatter(x, y, c="steelblue", s=point_size * 0.5, alpha=0.6)
        ax.set_title(f"Lidar BEV ({len(points)} points)")
    else:
        raise ValueError(f"render_bev_png only supports radar/lidar, got {modality}")

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf
