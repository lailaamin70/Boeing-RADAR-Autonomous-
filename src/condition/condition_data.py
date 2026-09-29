"""
Build standard condition data for condition-based RADAR-LiDAR velocity analysis.

Condition values are derived from TruckScenes metadata using the same
metadata relationships as the existing project analysis.
"""

import numpy as np
import pandas as pd

from .condition_config import DISTANCE_BIN_WIDTH


def parse_scene_description(description):
    """Parse scene-description tags into a dictionary."""
    conditions = {}

    for item in description.split(";"):
        if "." in item:
            key, value = item.split(".", 1)
            conditions[key] = value

    return conditions


def build_condition_table(trucksc, sample_tokens=None):
    """Build one annotation-level table containing all analysis conditions."""

    sample_lookup = {x["token"]: x for x in trucksc.sample}
    scene_lookup = {x["token"]: x for x in trucksc.scene}
    visibility_lookup = {x["token"]: x for x in trucksc.visibility}
    ego_pose_lookup = {x["token"]: x for x in trucksc.ego_pose}

    if sample_tokens is not None:
        sample_tokens = set(sample_tokens)

    rows = []

    for ann in trucksc.sample_annotation:
        if sample_tokens is not None and ann["sample_token"] not in sample_tokens:
            continue

        sample = sample_lookup[ann["sample_token"]]
        scene = scene_lookup[sample["scene_token"]]
        visibility = visibility_lookup[ann["visibility_token"]]
        ego_pose = ego_pose_lookup[sample["ego_pose_token"]]
        scene_conditions = parse_scene_description(scene["description"])

        object_position = np.asarray(ann["translation"], dtype=float)
        ego_position = np.asarray(ego_pose["translation"], dtype=float)
        distance_3d = np.linalg.norm(object_position - ego_position)

        rows.append({
            "annotation_token": ann["token"],
            "instance_token": ann["instance_token"],
            "sample_token": sample["token"],
            "scene_token": scene["token"],
            "area": scene_conditions.get("area"),
            "weather": scene_conditions.get("weather"),
            "visibility": visibility["level"],
            "object_type": ann["category_name"],
            "distance_3d": distance_3d,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No condition records were created.")

    df["distance"] = create_distance_bands(df["distance_3d"])

    return df


def create_distance_bands(distance, bin_width=DISTANCE_BIN_WIDTH):
    """Group continuous 3D distance into equal-width distance bands."""
    max_distance = distance.max()

    if pd.isna(max_distance):
        raise ValueError("No valid distance values found.")

    upper_bound = np.ceil(max_distance / bin_width) * bin_width
    bins = np.arange(0, upper_bound + bin_width, bin_width)

    return pd.cut(distance, bins=bins, right=False)


def build_condition_inventory(condition_df, conditions):
    """Summarise available values and sample sizes for each condition."""

    rows = []

    for condition in conditions:
        for value, group in condition_df.groupby(condition, observed=True):
            rows.append({
                "condition": condition,
                "value": value,
                "scenes": group["scene_token"].nunique(),
                "samples": group["sample_token"].nunique(),
                "objects": group["instance_token"].nunique(),
                "annotations": group["annotation_token"].nunique(),
            })

    return pd.DataFrame(rows)