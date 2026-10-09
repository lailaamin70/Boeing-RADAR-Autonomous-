"""
Global configuration for TruckScenes velocity and efficiency analysis.
"""

from pathlib import Path
import os

# Velocity report root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset configuration
DATA_ROOT = Path(os.environ.get(
    "TRUCKSCENES_DATA_ROOT",
    PROJECT_ROOT / "data"
)).expanduser().resolve()

VERSION = os.environ.get("TRUCKSCENES_VERSION", "v1.2-trainval")

# Derived dataset paths
SENSOR_ROOT = DATA_ROOT
SAMPLES_ROOT = SENSOR_ROOT / "samples"
SWEEPS_ROOT = SENSOR_ROOT / "sweeps"
METADATA_ROOT = DATA_ROOT / VERSION

# Shared timestamp conversion
TIMESTAMP_UNITS_PER_SECOND = 1_000_000