"""
Global configuration for the TruckScenes analysis project.

This file stores dataset paths and shared project settings.

The purpose of this file is to avoid hard-coding dataset locations
and dataset versions inside individual notebooks or analysis modules.

During development, the project uses TruckScenes v1.2-mini.

To analyse the complete TruckScenes dataset later:

1. Update DATA_ROOT if the full dataset is stored in another location.
2. Change METADATA_DIR_NAME to the metadata directory used by the
   complete dataset.
3. Keep the analysis notebooks and reusable Python functions unchanged.

Additional project-wide settings can also be added to this file later.
"""

from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Dataset configuration
DATA_ROOT = PROJECT_ROOT / "data" / "man-truckscenes"

# Current metadata version.
# Change this when switching from the mini dataset to the full dataset.
METADATA_DIR_NAME = "v1.2-mini"


# Derived dataset paths
SENSOR_ROOT = DATA_ROOT / "man-truckscenes"

SAMPLES_ROOT = SENSOR_ROOT / "samples"

SWEEPS_ROOT = SENSOR_ROOT / "sweeps"

METADATA_ROOT = DATA_ROOT / METADATA_DIR_NAME