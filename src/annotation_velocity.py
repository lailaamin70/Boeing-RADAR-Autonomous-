"""
Annotation-derived reference velocity utilities for the TruckScenes project.

This module contains reusable functions developed and validated in 02_Annotation_Reference_Velocity.ipynb.

The workflow derives object motion from consecutive TruckScenes annotations by linking object annotations to sample timestamps and constructing consecutive annotation pairs.

The module provides functions for:
- constructing annotation trajectory tables
- building consecutive annotation pairs
- calculating annotation time differences
- calculating 2D and 3D position changes
- calculating 2D and 3D annotation-derived reference velocity

The 2D ground-plane speed is used as the primary vehicle reference velocity, while the 3D speed is retained for diagnostic comparison.

RADAR measurements, LiDAR measurements, vehicle-speed grouping, and sensor comparison logic are handled separately.
"""

import pandas as pd


def build_annotation_trajectory_df(
    samples,
    annotations,
    instances,
    categories,
):
    """
    Build an annotation trajectory table from TruckScenes metadata.
    Each row represents one object annotation at one sample keyframe.

    Parameters
    ----------
    samples : list
        Records loaded from sample.json.

    annotations : list
        Records loaded from sample_annotation.json.

    instances : list
        Records loaded from instance.json.

    categories : list
        Records loaded from category.json.

    Returns
    -------
    pandas.DataFrame
        Annotation trajectory table containing object identity,
        timestamp, position, trajectory links, and category.
    """

    sample_timestamp = {
        sample["token"]: sample["timestamp"]
        for sample in samples
    }

    instance_lookup = {
        instance["token"]: instance
        for instance in instances
    }

    category_lookup = {
        category["token"]: category["name"]
        for category in categories
    }

    records = []

    for annotation in annotations:

        sample_token = annotation["sample_token"]
        instance_token = annotation["instance_token"]
        translation = annotation["translation"]

        if sample_token not in sample_timestamp:
            raise ValueError(
                f"Sample token not found: {sample_token}"
            )

        if instance_token not in instance_lookup:
            raise ValueError(
                f"Instance token not found: {instance_token}"
            )

        if len(translation) != 3:
            raise ValueError(
                f"Invalid translation for annotation "
                f"{annotation['token']}"
            )

        instance = instance_lookup[instance_token]

        category_token = instance["category_token"]

        if category_token not in category_lookup:
            raise ValueError(
                f"Category token not found: {category_token}"
            )

        x, y, z = translation

        records.append({
            "annotation_token": annotation["token"],
            "instance_token": instance_token,
            "sample_token": sample_token,
            "timestamp_us": sample_timestamp[sample_token],
            "x": x,
            "y": y,
            "z": z,
            "prev": annotation["prev"],
            "next": annotation["next"],
            "category_token": category_token,
            "category": category_lookup[category_token],
        })

    return pd.DataFrame(records)

def build_annotation_pairs(trajectory_df):
    """
    Build consecutive annotation pairs using annotation `next` links.
    Each row represents one motion interval between two consecutive annotations of the same object instance.

    Parameters
    ----------
    trajectory_df : pandas.DataFrame
        Annotation trajectory table produced by build_annotation_trajectory_df().

    Returns
    -------
    pandas.DataFrame
        Consecutive annotation pairs with start and end timestamps and positions.
    """

    required_columns = {
        "annotation_token",
        "instance_token",
        "sample_token",
        "timestamp_us",
        "x",
        "y",
        "z",
        "next",
        "category",
    }

    missing_columns = (
        required_columns - set(trajectory_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "trajectory_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    annotation_lookup = (
        trajectory_df
        .set_index("annotation_token")
        .to_dict("index")
    )

    records = []

    for _, start in trajectory_df.iterrows():

        next_token = start["next"]

        # The last annotation in a trajectory has no next annotation.
        if not next_token:
            continue

        if next_token not in annotation_lookup:
            raise ValueError(
                f"Next annotation not found: {next_token}"
            )

        end = annotation_lookup[next_token]

        if start["instance_token"] != end["instance_token"]:
            raise ValueError(
                "Consecutive annotations belong to different instances: "
                f"{start['annotation_token']} -> {next_token}"
            )

        records.append({
            "instance_token": start["instance_token"],
            "category": start["category"],

            "start_annotation_token": start["annotation_token"],
            "end_annotation_token": next_token,

            "start_sample_token": start["sample_token"],
            "end_sample_token": end["sample_token"],

            "start_timestamp_us": start["timestamp_us"],
            "end_timestamp_us": end["timestamp_us"],

            "start_x": start["x"],
            "start_y": start["y"],
            "start_z": start["z"],

            "end_x": end["x"],
            "end_y": end["y"],
            "end_z": end["z"],
        })

    return pd.DataFrame(records)

def add_time_differences(annotation_pairs):
    """
    Calculate time differences for consecutive annotation pairs.

    Parameters
    ----------
    annotation_pairs : pandas.DataFrame
        Consecutive annotation pairs produced by build_annotation_pairs().

    Returns
    -------
    pandas.DataFrame
        Annotation pairs with raw timestamp differences and time differences in seconds.
    """

    required_columns = {
        "start_timestamp_us",
        "end_timestamp_us",
    }

    missing_columns = (
        required_columns - set(annotation_pairs.columns)
    )

    if missing_columns:
        raise ValueError(
            "annotation_pairs is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    result = annotation_pairs.copy()

    result["delta_timestamp_us"] = (
        result["end_timestamp_us"]
        - result["start_timestamp_us"]
    )

    result["dt_s"] = (
        result["delta_timestamp_us"]
        / 1_000_000
    )

    return result

def add_position_changes(annotation_pairs):
    """
    Calculate position changes for consecutive annotation pairs.

    Parameters
    ----------
    annotation_pairs : pandas.DataFrame
        Consecutive annotation pairs containing start and end positions.

    Returns
    -------
    pandas.DataFrame
        Annotation pairs with coordinate changes and 2D and 3D displacement.
    """

    required_columns = {
        "start_x",
        "start_y",
        "start_z",
        "end_x",
        "end_y",
        "end_z",
    }

    missing_columns = (
        required_columns - set(annotation_pairs.columns)
    )

    if missing_columns:
        raise ValueError(
            "annotation_pairs is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    result = annotation_pairs.copy()

    result["dx"] = result["end_x"] - result["start_x"]
    result["dy"] = result["end_y"] - result["start_y"]
    result["dz"] = result["end_z"] - result["start_z"]

    result["displacement_2d"] = (
        result["dx"] ** 2
        + result["dy"] ** 2
    ) ** 0.5

    result["displacement_3d"] = (
        result["dx"] ** 2
        + result["dy"] ** 2
        + result["dz"] ** 2
    ) ** 0.5

    return result

def add_reference_velocity(annotation_pairs):
    """
    Calculate annotation-derived velocity components and 2D and 3D reference speeds.

    Parameters
    ----------
    annotation_pairs : pandas.DataFrame
        Consecutive annotation pairs containing position changes and time differences.

    Returns
    -------
    pandas.DataFrame
        Annotation pairs with velocity components and 2D and 3D reference speeds in m/s.
    """

    required_columns = {
        "dt_s",
        "dx",
        "dy",
        "dz",
    }

    missing_columns = (
        required_columns - set(annotation_pairs.columns)
    )

    if missing_columns:
        raise ValueError(
            "annotation_pairs is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if (annotation_pairs["dt_s"] <= 0).any():
        raise ValueError(
            "Reference velocity cannot be calculated "
            "because non-positive time differences are present."
        )

    result = annotation_pairs.copy()

    result["vx_mps"] = result["dx"] / result["dt_s"]
    result["vy_mps"] = result["dy"] / result["dt_s"]
    result["vz_mps"] = result["dz"] / result["dt_s"]

    result["speed_2d_mps"] = (
        result["vx_mps"] ** 2
        + result["vy_mps"] ** 2
    ) ** 0.5

    result["speed_3d_mps"] = (
        result["vx_mps"] ** 2
        + result["vy_mps"] ** 2
        + result["vz_mps"] ** 2
    ) ** 0.5

    return result