"""
A tiny synthetic TruckScenes whose statistics precalculated, so
tests assert exact numbers instead of "something came back".
"""
import contextlib
import struct
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

T0 = 1_700_000_000_000_000  # microseconds
ID = [1.0, 0.0, 0.0, 0.0]
# camera frame (z forward, x right, y down) -> ego frame (x forward, y left, z up)
CAM_Q = [0.5, -0.5, 0.5, -0.5]
INTRINSIC = [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]
BOX_SIZE = [2.0, 4.5, 1.8]  # w, l, h

FILE_SIZES = {"lidar": 4_000, "radar": 1_000, "camera": 500_000}
CHANNEL_DIR = {"lidar": "LIDAR_TOP", "radar": "RADAR_FRONT", "camera": "CAMERA_FRONT"}
EXT = {"lidar": "bin", "radar": "pcd", "camera": "png"}


def _png(width=2, height=2):
    """A valid minimal RGB PNG, built with the stdlib."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\x80\x80\x80" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def build_tables():
    t = {}
    t["scene"] = [{
        "token": "S", "name": "scene-0001", "description": "day;clear", "log_token": "",
        "nbr_samples": 2, "first_sample_token": "A", "last_sample_token": "B",
    }]

    def data(suffix):
        return {"LIDAR_TOP": f"lidar_{suffix}", "RADAR_FRONT": f"radar_{suffix}", "CAMERA_FRONT": f"cam_{suffix}"}

    t["sample"] = [
        {"token": "A", "timestamp": T0, "scene_token": "S", "next": "B", "prev": "",
         "data": data("A"), "anns": ["a1", "a2", "a3", "a4", "a5"]},
        {"token": "B", "timestamp": T0 + 2_000_000, "scene_token": "S", "next": "", "prev": "A",
         "data": data("B"), "anns": []},
    ]

    cs = {"lidar": "cs_lidar", "radar": "cs_radar", "camera": "cs_cam"}
    sd = []
    for suffix, pose in (("A", "ep0"), ("B", "ep20")):
        for modality in ("lidar", "radar", "camera"):
            token = {"lidar": "lidar", "radar": "radar", "camera": "cam"}[modality] + f"_{suffix}"
            sd.append({
                "token": token, "calibrated_sensor_token": cs[modality], "ego_pose_token": pose,
                "is_key_frame": True, "fileformat": EXT[modality],
                "filename": f"samples/{CHANNEL_DIR[modality]}/{token}.{EXT[modality]}",
                "width": 1600 if modality == "camera" else None,
                "height": 900 if modality == "camera" else None,
                "timestamp": T0, "next": "", "prev": "", "sample_token": suffix,
            })
    t["sample_data"] = sd

    t["calibrated_sensor"] = [
        {"token": "cs_lidar", "sensor_token": "s_lidar", "translation": [4.0, 0.0, 3.8], "rotation": ID, "camera_intrinsic": []},
        {"token": "cs_radar", "sensor_token": "s_radar", "translation": [5.1, 0.0, 0.7], "rotation": ID, "camera_intrinsic": []},
        {"token": "cs_cam", "sensor_token": "s_cam", "translation": [4.6, 0.0, 3.0], "rotation": CAM_Q, "camera_intrinsic": INTRINSIC},
    ]
    t["sensor"] = [
        {"token": "s_lidar", "channel": "LIDAR_TOP", "modality": "lidar"},
        {"token": "s_radar", "channel": "RADAR_FRONT", "modality": "radar"},
        {"token": "s_cam", "channel": "CAMERA_FRONT", "modality": "camera"},
    ]

    # ego poses every 0.1 s for 2 s, moving +x at 10 m/s from (100, 200)
    t["ego_pose"] = [
        {"token": f"ep{i}", "timestamp": T0 + i * 100_000, "translation": [100.0 + i, 200.0, 0.0], "rotation": ID}
        for i in range(21)
    ]
    chassis = [
        {"token": f"c{i}", "timestamp": T0 + i * 100_000, "vx": 10.0, "vy": 0.0, "vz": 0.0,
         "ax": 0.5, "ay": 0.0, "az": 0.0, "yaw": 0.0, "pitch": 0.01, "roll": 0.0,
         "yaw_rate": 0.02, "pitch_rate": 0.0, "roll_rate": 0.0}
        for i in range(21)
    ]
    t["ego_motion_chassis"] = chassis
    t["ego_motion_cabin"] = [{**r, "token": "cb" + r["token"], "pitch": 0.03} for r in chassis]

    t["category"] = [
        {"token": "c_car", "name": "vehicle.car"},
        {"token": "c_truck", "name": "vehicle.truck"},
        {"token": "c_ped", "name": "human.pedestrian.adult"},
        {"token": "c_ego", "name": "vehicle.ego_trailer"},
    ]
    t["instance"] = [
        {"token": f"i{k}", "category_token": c}
        for k, c in [(1, "c_car"), (2, "c_car"), (3, "c_truck"), (4, "c_ped"), (5, "c_ego")]
    ]
    # deliberately out of order: ranks must come from the level, not the table order
    t["visibility"] = [
        {"token": "v3", "level": 3, "description": "60-80%"},
        {"token": "v1", "level": 1, "description": "0-40%"},
        {"token": "v4", "level": 4, "description": "80-100%"},
        {"token": "v2", "level": 2, "description": "40-60%"},
    ]
    t["attribute"] = [{"token": "at1", "name": "vehicle.moving"}, {"token": "at2", "name": "vehicle.parked"}]

    def ann(token, inst, xy, lidar, radar, vis, attrs):
        return {
            "token": token, "sample_token": "A", "instance_token": inst, "visibility_token": vis,
            "attribute_tokens": attrs, "translation": [xy[0], xy[1], 0.9], "size": list(BOX_SIZE),
            "rotation": ID, "num_lidar_pts": lidar, "num_radar_pts": radar,
        }

    t["sample_annotation"] = [
        ann("a1", "i1", (120.0, 200.0), 50, 3, "v4", ["at1"]),
        ann("a2", "i2", (135.0, 200.0), 8, 0, "v3", ["at1"]),
        ann("a3", "i3", (160.0, 200.0), 2, 1, "v1", ["at2"]),
        ann("a4", "i4", (100.0, 215.0), 20, 0, "", []),
        ann("a5", "i5", (96.0, 200.0), 999, 999, "v4", []),
    ]
    return t


class FakeTrucksc:
    """Just enough of the devkit object: tables as attributes, and get(table, token)."""

    def __init__(self, tables):
        self.tables = tables
        for name, rows in tables.items():
            setattr(self, name, rows)
        self._index = {name: {r["token"]: r for r in rows} for name, rows in tables.items()}

    def get(self, table, token):
        return self._index[table][token]


def fake_load_lidar_points(sd_token):
    n, far = (300, 70.0) if sd_token.endswith("_B") else (100, 50.0)
    x = np.linspace(1.0, far, n)
    return np.column_stack([x, np.zeros(n), np.zeros(n), np.ones(n)])


def fake_load_radar_points(sd_token):
    # columns: x, y, z, vx, vy, vz, rcs  (the confirmed 7-row radar schema)
    if sd_token.endswith("_B"):
        n, far, vx, vy, rcs = 20, 50.0, 0.0, 6.0, 20.0
    else:
        n, far, vx, vy, rcs = 10, 30.0, 3.0, 4.0, 10.0
    x = np.linspace(5.0, far, n)
    zeros = np.zeros(n)
    return np.column_stack([x, zeros, zeros, np.full(n, vx), np.full(n, vy), zeros, np.full(n, rcs)])


class FakeDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.tables = build_tables()
        self.trucksc = FakeTrucksc(self.tables)
        self.write_files()

    def write_files(self):
        for row in self.tables["sample_data"]:
            path = self.root / row["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            modality = {"lidar": "lidar", "radar": "radar", "cam": "camera"}[row["token"].split("_")[0]]
            payload = _png() if modality == "camera" else b""
            path.write_bytes(payload + b"\0" * (FILE_SIZES[modality] - len(payload)))


def install(root):
    """
    Point the whole app at the synthetic dataset and reset every module-level
    cache. Returns (dataset, stack); call stack.close() to undo everything.
    """
    from app import analytics, dataset_stats, detections_loader, truckscenes_loader as tsl
    from config import Config

    root = Path(root)
    ds = FakeDataset(root)
    stack = contextlib.ExitStack()
    enter = stack.enter_context

    enter(mock.patch.object(tsl, "get_trucksc", lambda: ds.trucksc))
    enter(mock.patch.object(tsl, "load_lidar_points", fake_load_lidar_points))
    enter(mock.patch.object(tsl, "load_radar_points", fake_load_radar_points))

    enter(mock.patch.object(Config, "TRUCKSCENES_DATAROOT", str(root)))
    enter(mock.patch.object(Config, "DATA_VOLUME_CACHE_PATH", str(root / "cache" / "volume.json")))
    enter(mock.patch.object(Config, "ANALYSIS_CACHE_PATH", str(root / "cache" / "analysis.json")))
    enter(mock.patch.object(
        Config, "DETECTIONS_PATHS",
        {"lidar": root / "det" / "lidar.pkl", "radar": root / "det" / "radar.pkl"},
    ))

    enter(mock.patch.object(dataset_stats, "_table", None))
    enter(mock.patch.object(dataset_stats, "_ego", None))
    enter(mock.patch.object(analytics, "_disk_cache_loaded", False))
    enter(mock.patch.dict(analytics._cheap_cache, clear=True))
    enter(mock.patch.dict(analytics._expensive_cache, clear=True))
    enter(mock.patch.dict(detections_loader._cache, clear=True))
    enter(mock.patch.dict(detections_loader._classes, clear=True))
    return ds, stack
