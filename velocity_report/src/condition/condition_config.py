"""
Configuration for condition-based RADAR-LiDAR velocity analysis.

Dataset-dependent condition values are discovered automatically from
the loaded TruckScenes data.
"""


CONDITIONS = [
    "area",
    "weather",
    "visibility",
    "object_type",
    "distance",
]


# Existing project definition:
# distance_3d is grouped into 25 m intervals.
DISTANCE_BIN_WIDTH = 25