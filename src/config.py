"""
Global configuration for the TruckScenes velocity and efficiency analysis.

This file stores dataset paths and shared project settings so that
dataset locations, versions, and common units are not hard-coded
inside individual notebooks or analysis modules.
"""

from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Dataset configuration
DATA_ROOT = PROJECT_ROOT / "data" / "man-truckscenes"

VERSION = "v1.2-mini"


# Derived dataset paths
SENSOR_ROOT = DATA_ROOT / "man-truckscenes"

SAMPLES_ROOT = SENSOR_ROOT / "samples"

SWEEPS_ROOT = SENSOR_ROOT / "sweeps"

METADATA_ROOT = DATA_ROOT / VERSION


# Shared timestamp conversion
# TruckScenes timestamps are stored in microseconds.
TIMESTAMP_UNITS_PER_SECOND = 1_000_000