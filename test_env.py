"""
Simple environment check for the TruckScenes velocity and efficiency analysis.

This script verifies that the core third-party libraries and project modules
can be imported successfully. It does not run any dataset analysis.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud

from src import (
    annotation_velocity,
    comparison,
    config,
    data_io,
    dataset_inspection,
    lidar_velocity,
    radar_velocity,
    sensor_observation,
)


print(f"Python interpreter: {sys.executable}")
print("All core libraries and velocity/efficiency modules imported successfully!")