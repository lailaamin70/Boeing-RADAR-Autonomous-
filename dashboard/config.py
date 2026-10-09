import os
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # TruckScenes dataset location
    _DEFAULT_DATAROOT = os.path.normpath(
        os.path.join(BASE_DIR, "..", "data", "man-truckscenes")
    )
    TRUCKSCENES_DATAROOT = os.environ.get(
        "TRUCKSCENES_DATAROOT", _DEFAULT_DATAROOT
    )
    # Which split to load, e.g. v1.2-mini (must match a folder under the dataroot)
    TRUCKSCENES_VERSION = os.environ.get("TRUCKSCENES_VERSION", "v1.2-mini")

    TRUCKSCENES_VERBOSE = True

    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-change-me")
    JSON_SORT_KEYS = False

    # How many scenes / samples to show per page on listing views
    SCENES_PER_PAGE = 12
    SAMPLES_PER_PAGE = 20

    # Where per-scene expensive analysis metrics are cached on disk
    ANALYSIS_CACHE_PATH = os.environ.get(
        "ANALYSIS_CACHE_PATH",
        os.path.join(BASE_DIR, "instance", "analysis_expensive_cache.json"),
    )

    # Ego truck outline (scene viewer):
    EGO_TRUCK_BOXES = [
        {"name": "tractor", "center": [1.9, 0.0, 1.9], "size": [2.55, 6.5, 3.8]},
    ]

    # Object detection results:
    _DEFAULT_OMNIHD_DIR = os.path.normpath(
        os.path.join(BASE_DIR, "..", "results", "object_detection", "omnihd")
    )
    OMNIHD_RESULTS_DIR = os.environ.get("OMNIHD_RESULTS_DIR", _DEFAULT_OMNIHD_DIR)
    DETECTIONS_PATHS = {
        "lidar": Path(OMNIHD_RESULTS_DIR) / "lidar_mini_predictions.pkl",
        "radar": Path(OMNIHD_RESULTS_DIR) / "radar_mini_predictions.pkl",
    }
