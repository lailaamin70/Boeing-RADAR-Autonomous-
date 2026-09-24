"""
Helper functions for comparing RADAR and LiDAR velocity errors by area.

We already have the velocity results from 05_comparison.ipynb. 
Each row is one vehicle moving between two annotated frames, with:
    - the "true" velocity (from the annotations)
    - the velocity estimated by RADAR
    - the velocity estimated by LiDAR

This file adds two things so we can compare the sensors by area:
    1. The scene tags (area, weather, etc.) for each row
    2. A direction error, because velocity has both a speed AND a direction
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_io import load_json


# -------------------------------------------------------------------
# 1. Scene tags (area, weather, ...)
# -------------------------------------------------------------------

def parse_scene_tags(description):
    """
    Turn a scene description into a dictionary.

    Example:
        "weather.clear;area.highway;daytime.noon"
        -> {"weather": "clear", "area": "highway", "daytime": "noon"}
    """
    tags = {}
    for part in description.split(";"):
        if "." in part:
            key, value = part.strip().split(".", 1)
            tags[key] = value
    return tags


def get_scene_tags_for_samples(metadata_root):
    """
    Make a table that tells us which scene (and which area/weather)
    each sample belongs to.

    Uses scene.json and sample.json from the dataset metadata folder.
    """
    metadata_root = Path(metadata_root)
    scenes = load_json(metadata_root / "scene.json")
    samples = load_json(metadata_root / "sample.json")

    # One row per scene, with its tags as columns
    scene_rows = []
    for scene in scenes:
        row = {"scene_token": scene["token"], "scene_name": scene["name"]}
        row.update(parse_scene_tags(scene["description"]))
        scene_rows.append(row)
    scene_table = pd.DataFrame(scene_rows)

    # One row per sample, saying which scene it is in
    sample_table = pd.DataFrame(
        [{"sample_token": s["token"], "scene_token": s["scene_token"]} for s in samples]
    )

    return sample_table.merge(scene_table, on="scene_token", how="left")


def add_scene_tags(comparison_df, vehicle_reference_df, sample_tags):
    """
    Add the area/weather tags to every row of comparison_df.

    comparison_df doesn't store which sample a row came from, so we first
    get the start sample from vehicle_reference_df, then look up its scene.
    """
    start_samples = vehicle_reference_df[["start_annotation_token", "start_sample_token"]]
    df = comparison_df.merge(start_samples, on="start_annotation_token", how="left")

    sample_tags = sample_tags.rename(columns={"sample_token": "start_sample_token"})
    df = df.merge(sample_tags, on="start_sample_token", how="left")

    # Safety check: every row should have found its scene
    missing = df["scene_token"].isna().sum()
    if missing > 0:
        raise ValueError(
            f"{missing} rows could not be matched to a scene. "
            "Check that the metadata folder is the same dataset version (v1.2-mini)."
        )

    return df


# -------------------------------------------------------------------
# 2. Direction error
# -------------------------------------------------------------------

def angle_difference(true_x, true_y, sensor_x, sensor_y):
    """
    Angle (in degrees) between the true velocity and the sensor velocity.

    0 deg   = same direction (perfect)
    180 deg = completely opposite direction
    """
    cross = true_x * sensor_y - true_y * sensor_x
    dot = true_x * sensor_x + true_y * sensor_y
    return np.degrees(np.abs(np.arctan2(cross, dot)))


def add_direction_error(df, min_speed=2.0):
    """
    Add radar_dir_error_deg and lidar_dir_error_deg columns.

    A vehicle that is standing still doesn't really have a direction,
    so we only calculate this for vehicles moving at least `min_speed` m/s.
    Other rows are left empty (NaN).
    """
    df = df.copy()
    is_moving = df["speed_2d_mps"] >= min_speed

    for sensor in ["radar", "lidar"]:
        angle = angle_difference(
            df["vx_mps"], df["vy_mps"],
            df[f"{sensor}_vx_mps"], df[f"{sensor}_vy_mps"],
        )
        has_estimate = df[f"{sensor}_velocity_available"]
        df[f"{sensor}_dir_error_deg"] = angle.where(is_moving & has_estimate)

    return df


# -------------------------------------------------------------------
# 3. Summary tables
# -------------------------------------------------------------------

def count_scenes(df, row_tag="area", column_tag="weather"):
    """
    Count how many scenes we have for each area and weather.
    Useful to see if some groups only have 1 or 2 scenes.
    """
    one_row_per_scene = df.drop_duplicates("scene_token")
    return pd.crosstab(
        one_row_per_scene[row_tag], one_row_per_scene[column_tag], margins=True
    )


def availability_by_area(df, tag="area"):
    """
    For each area: how often did each sensor manage to estimate a velocity? (%)
    """
    table = df.groupby(tag).agg(
        scenes=("scene_token", "nunique"),
        vehicles=("start_annotation_token", "size"),
        radar_available_pct=("radar_velocity_available", "mean"),
        lidar_available_pct=("lidar_velocity_available", "mean"),
    )
    table[["radar_available_pct", "lidar_available_pct"]] *= 100
    return table.round(1).reset_index()


def error_summary(df, tag="area", both_sensors_only=False):
    """
    Summarise the speed error and direction error for each area and sensor.

    If both_sensors_only=True, we only keep vehicles where BOTH RADAR and
    LiDAR gave a velocity. This makes the comparison fair, because both
    sensors are tested on exactly the same vehicles.

    Columns:
        speed_err_median : typical speed error (m/s)
        speed_err_iqr    : spread of the middle 50% of errors (m/s)
        speed_err_std    : overall spread, affected by big outliers (m/s)
        dir_err_median   : typical direction error (degrees)
        dir_err_p90      : 90% of direction errors are below this (degrees)
    """
    if both_sensors_only:
        df = df[df["radar_velocity_available"] & df["lidar_velocity_available"]]

    rows = []
    for group_name, group in df.groupby(tag):
        for sensor, label in [("radar", "RADAR"), ("lidar", "LiDAR")]:
            has_estimate = group[group[f"{sensor}_velocity_available"]]
            speed_err = has_estimate[f"{sensor}_abs_error_2d_mps"].dropna()
            dir_err = has_estimate[f"{sensor}_dir_error_deg"].dropna()

            rows.append({
                tag: group_name,
                "sensor": label,
                "scenes": group["scene_token"].nunique(),
                "vehicles": len(has_estimate),
                "speed_err_median": speed_err.median(),
                "speed_err_iqr": speed_err.quantile(0.75) - speed_err.quantile(0.25),
                "speed_err_std": speed_err.std(),
                "moving_vehicles": len(dir_err),
                "dir_err_median": dir_err.median(),
                "dir_err_p90": dir_err.quantile(0.9),
            })

    return pd.DataFrame(rows).round(3)