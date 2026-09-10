import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # TruckScenes dataset location
    _DEFAULT_DATAROOT = os.path.normpath(
        os.path.join(BASE_DIR, "..", "data", "man-truckscenes")
    )
    TRUCKSCENES_DATAROOT = os.environ.get(
        "TRUCKSCENES_DATAROOT", _DEFAULT_DATAROOT
    )
    # Which split to load: v1.0-mini, v1.0-trainval, v1.0-test
    TRUCKSCENES_VERSION = os.environ.get("TRUCKSCENES_VERSION", "v1.2-mini")

    TRUCKSCENES_VERBOSE = True

    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-change-me")
    JSON_SORT_KEYS = False

    SCENES_PER_PAGE = 12
    SAMPLES_PER_PAGE = 20
