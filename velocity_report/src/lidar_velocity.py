"""
Utilities for processing TruckScenes LiDAR data and estimating target velocities.

Target positions are estimated from LiDAR points, and velocities are
calculated from position changes between consecutive observations
rather than measured directly by the LiDAR sensor.
"""

from pathlib import Path
import pandas as pd
from collections import defaultdict
import numpy as np

from pyquaternion import Quaternion
from truckscenes.utils.data_classes import (
    Box,
    LidarPointCloud,
)
from truckscenes.utils.geometry_utils import points_in_box


def load_lidar_points(pcd_file):
    """Load LiDAR point coordinates, intensity, and timestamps from a PCD file."""

    pcd_file = Path(pcd_file)

    lidar_pc = LidarPointCloud.from_file(
        str(pcd_file)
    )

    points_df = pd.DataFrame(
        lidar_pc.points.T,
        columns=[
            "x",
            "y",
            "z",
            "intensity",
        ],
    )

    points_df["timestamp"] = (
        lidar_pc.timestamps.flatten()
    )

    return points_df


def build_target_lidar_information_df(
    annotations,
    instances,
    categories,
):
    """Build vehicle-level LiDAR information using annotation point counts."""

    instance_lookup = {
        record["token"]: record
        for record in instances
    }

    category_lookup = {
        record["token"]: record["name"]
        for record in categories
    }

    records = []

    for annotation in annotations:

        instance = instance_lookup[
            annotation["instance_token"]
        ]

        category = category_lookup[
            instance["category_token"]
        ]

        if (
            category.startswith("vehicle.")
            and category != "vehicle.ego_trailer"
        ):
            records.append({
                "annotation_token": annotation["token"],
                "instance_token": annotation["instance_token"],
                "sample_token": annotation["sample_token"],
                "category": category,
                "lidar_point_count": annotation["num_lidar_pts"],
            })

    return pd.DataFrame(records)


def build_lidar_target_states_df(
    target_lidar_df,
    annotations,
    lidar_observation_df,
    sample_data_lookup,
    ego_pose_lookup,
    calibrated_sensor_lookup,
    sensor_root,
):
    """
    Estimate target positions from LiDAR points at each keyframe.

    Points are transformed to global coordinates and associated with
    annotation boxes to calculate target positions.
    """

    annotation_lookup = {
        record["token"]: record
        for record in annotations
    }

    target_annotations_by_sample = defaultdict(list)

    for _, row in target_lidar_df.iterrows():
        target_annotations_by_sample[
            row["sample_token"]
        ].append(
            annotation_lookup[
                row["annotation_token"]
            ]
        )

    records = []

    for _, frame in lidar_observation_df.iterrows():

        sample_token = frame["Sample Token"]

        target_annotations = (
            target_annotations_by_sample[
                sample_token
            ]
        )

        if not target_annotations:
            continue

        sd = sample_data_lookup[
            frame["Token"]
        ]

        ego_pose = ego_pose_lookup[
            sd["ego_pose_token"]
        ]

        calibrated_sensor = calibrated_sensor_lookup[
            sd["calibrated_sensor_token"]
        ]

        lidar_points = load_lidar_points(
            Path(sensor_root) / sd["filename"]
        )

        xyz = lidar_points[
            ["x", "y", "z"]
        ].to_numpy().T

        timestamps = lidar_points[
            "timestamp"
        ].to_numpy()

        # Sensor -> ego
        xyz = (
            Quaternion(
                calibrated_sensor["rotation"]
            ).rotation_matrix
            @ xyz
        )

        xyz += np.asarray(
            calibrated_sensor["translation"],
            dtype=float,
        ).reshape(3, 1)

        # Ego -> global
        xyz = (
            Quaternion(
                ego_pose["rotation"]
            ).rotation_matrix
            @ xyz
        )

        xyz += np.asarray(
            ego_pose["translation"],
            dtype=float,
        ).reshape(3, 1)

        for annotation in target_annotations:

            box = Box(
                annotation["translation"],
                annotation["size"],
                Quaternion(
                    annotation["rotation"]
                ),
            )

            inside = points_in_box(
                box,
                xyz,
            )

            point_count = int(
                inside.sum()
            )

            if point_count == 0:
                continue

            target_xyz = xyz[:, inside]

            records.append({
                "annotation_token":
                    annotation["token"],

                "instance_token":
                    annotation["instance_token"],

                "sample_token":
                    sample_token,

                "sensor":
                    frame["Sensor"],

                "point_count":
                    point_count,

                "timestamp_us":
                    float(
                        np.median(
                            timestamps[inside]
                        )
                    ),

                "lidar_x_global":
                    float(
                        target_xyz[0].mean()
                    ),

                "lidar_y_global":
                    float(
                        target_xyz[1].mean()
                    ),

                "lidar_z_global":
                    float(
                        target_xyz[2].mean()
                    ),
            })

    return pd.DataFrame(records)


def build_lidar_velocity_df(
    lidar_target_states_df,
    annotations,
    min_points=20,
):
    """
    Estimate target velocities from consecutive LiDAR observations.

    Only same-sensor observations meeting the minimum point count are used.
    When multiple sensors are available, the estimate with the highest
    minimum point count is selected.
    """

    annotation_next = {
        annotation["token"]: annotation["next"]
        for annotation in annotations
    }

    start_states = (
        lidar_target_states_df
        .copy()
        .rename(columns={
            "annotation_token": "start_annotation_token",
            "sample_token": "start_sample_token",
            "point_count": "start_point_count",
            "timestamp_us": "start_timestamp_us",
            "lidar_x_global": "start_x",
            "lidar_y_global": "start_y",
            "lidar_z_global": "start_z",
        })
    )

    start_states["end_annotation_token"] = (
        start_states["start_annotation_token"]
        .map(annotation_next)
    )

    end_states = (
        lidar_target_states_df
        .copy()
        .rename(columns={
            "annotation_token": "end_annotation_token",
            "sample_token": "end_sample_token",
            "point_count": "end_point_count",
            "timestamp_us": "end_timestamp_us",
            "lidar_x_global": "end_x",
            "lidar_y_global": "end_y",
            "lidar_z_global": "end_z",
        })
    )

    velocity_candidates = start_states.merge(
        end_states[
            [
                "end_annotation_token",
                "instance_token",
                "sensor",
                "end_sample_token",
                "end_point_count",
                "end_timestamp_us",
                "end_x",
                "end_y",
                "end_z",
            ]
        ],
        on=[
            "end_annotation_token",
            "instance_token",
            "sensor",
        ],
        how="inner",
    )

    velocity_candidates["min_point_count"] = (
        velocity_candidates[
            ["start_point_count", "end_point_count"]
        ].min(axis=1)
    )

    velocity_candidates = velocity_candidates[
        velocity_candidates["min_point_count"] >= min_points
    ].copy()

    velocity_candidates["dt_s"] = (
        velocity_candidates["end_timestamp_us"]
        - velocity_candidates["start_timestamp_us"]
    ) / 1_000_000

    velocity_candidates = velocity_candidates[
        velocity_candidates["dt_s"] > 0
    ].copy()

    velocity_candidates["lidar_vx_mps"] = (
        velocity_candidates["end_x"]
        - velocity_candidates["start_x"]
    ) / velocity_candidates["dt_s"]

    velocity_candidates["lidar_vy_mps"] = (
        velocity_candidates["end_y"]
        - velocity_candidates["start_y"]
    ) / velocity_candidates["dt_s"]

    velocity_candidates["lidar_vz_mps"] = (
        velocity_candidates["end_z"]
        - velocity_candidates["start_z"]
    ) / velocity_candidates["dt_s"]

    velocity_candidates["lidar_speed_2d_mps"] = np.sqrt(
        velocity_candidates["lidar_vx_mps"] ** 2
        + velocity_candidates["lidar_vy_mps"] ** 2
    )

    velocity_candidates["lidar_speed_3d_mps"] = np.sqrt(
        velocity_candidates["lidar_vx_mps"] ** 2
        + velocity_candidates["lidar_vy_mps"] ** 2
        + velocity_candidates["lidar_vz_mps"] ** 2
    )

    lidar_velocity_df = (
        velocity_candidates
        .sort_values(
            [
                "start_annotation_token",
                "min_point_count",
            ],
            ascending=[True, False],
        )
        .drop_duplicates(
            subset=["start_annotation_token"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return lidar_velocity_df