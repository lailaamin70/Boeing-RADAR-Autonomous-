"""
Reusable RADAR velocity utilities for the TruckScenes project.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from truckscenes.utils.data_classes import Box
from truckscenes.utils.geometry_utils import points_in_box

from .config import METADATA_ROOT, SENSOR_ROOT
from .data_io import load_json, read_pcd_header


def _require_columns(df, required, name):
    """Raise a clear error when required DataFrame columns are missing."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: " + ", ".join(sorted(missing)))


def load_radar_points(pcd_file):
    """Load point-level data from a validated TruckScenes RADAR PCD file."""
    pcd_file = Path(pcd_file)
    header = read_pcd_header(pcd_file)

    fields = header["FIELDS"].split()
    sizes = [int(value) for value in header["SIZE"].split()]
    types = header["TYPE"].split()

    if header["DATA"] != "binary":
        raise ValueError(f"Unsupported PCD data format: {header['DATA']}")
    if not all(size == 4 for size in sizes):
        raise ValueError("Expected all RADAR PCD fields to use 4-byte values.")
    if not all(field_type == "F" for field_type in types):
        raise ValueError("Expected all RADAR PCD fields to be floating-point.")

    with open(pcd_file, "rb") as file:
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"PCD DATA header not found: {pcd_file}")
            if line.strip().startswith(b"DATA"):
                break
        values = np.fromfile(file, dtype="<f4")

    expected_values = header["POINTS"] * len(fields)
    if len(values) != expected_values:
        raise ValueError(
            f"Unexpected number of values in {pcd_file}: "
            f"expected {expected_values}, found {len(values)}."
        )

    points = values.reshape(header["POINTS"], len(fields))
    return pd.DataFrame(points, columns=fields)


def get_radar_points_in_annotation(
    radar_points_df,
    annotation,
    ego_pose,
    calibrated_sensor,
):
    """Return RADAR points inside one annotation box in the RADAR sensor frame."""
    _require_columns(radar_points_df, {"x", "y", "z"}, "radar_points_df")

    box = Box(
        annotation["translation"],
        annotation["size"],
        Quaternion(annotation["rotation"]),
    )

    # Global -> ego -> RADAR sensor frame.
    box.translate(-np.asarray(ego_pose["translation"], dtype=float))
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)
    box.translate(-np.asarray(calibrated_sensor["translation"], dtype=float))
    box.rotate(Quaternion(calibrated_sensor["rotation"]).inverse)

    points_xyz = radar_points_df[["x", "y", "z"]].to_numpy().T
    mask = points_in_box(box, points_xyz)
    return radar_points_df.loc[mask].copy()


def build_target_radar_points_df(
    radar_observation_df,
    annotations,
    instances,
    categories,
    sample_data_lookup,
    ego_pose_lookup,
    calibrated_sensor_lookup,
    sensor_root=None,
):
    """Associate keyframe RADAR returns with annotated target vehicles."""

    sensor_root = Path(SENSOR_ROOT if sensor_root is None else sensor_root)

    required = {"Token", "Sample Token", "Sensor", "Timestamp", "Is Key Frame"}
    _require_columns(radar_observation_df, required, "radar_observation_df")

    if not radar_observation_df["Is Key Frame"].eq(True).all():
        raise ValueError(
            "build_target_radar_points_df currently supports keyframe RADAR observations only."
        )

    missing_tokens = set(radar_observation_df["Token"]) - set(sample_data_lookup)

    if missing_tokens:
        raise ValueError(
            f"{len(missing_tokens)} RADAR observations could not be matched "
            "to sample_data metadata."
        )

    annotations_by_sample = {}
    for annotation in annotations:
        annotations_by_sample.setdefault(annotation["sample_token"], []).append(annotation)

    instance_lookup = {record["token"]: record for record in instances}
    category_lookup = {record["token"]: record["name"] for record in categories}

    point_tables = []

    for sample_token, radar_frames in radar_observation_df.groupby("Sample Token"):
        target_annotations = []

        for annotation in annotations_by_sample.get(sample_token, []):
            instance = instance_lookup[annotation["instance_token"]]
            category = category_lookup[instance["category_token"]]

            if category.startswith("vehicle.") and category != "vehicle.ego_trailer":
                target_annotations.append((annotation, category))

        for _, radar_frame in radar_frames.iterrows():
            sample_data_record = sample_data_lookup[radar_frame["Token"]]
            ego_pose = ego_pose_lookup[sample_data_record["ego_pose_token"]]
            calibrated_sensor = calibrated_sensor_lookup[
                sample_data_record["calibrated_sensor_token"]
            ]

            radar_points = load_radar_points(
                sensor_root / sample_data_record["filename"]
            )

            for annotation, category in target_annotations:
                target_points = get_radar_points_in_annotation(
                    radar_points,
                    annotation,
                    ego_pose,
                    calibrated_sensor,
                )

                if target_points.empty:
                    continue

                target_points["annotation_token"] = annotation["token"]
                target_points["instance_token"] = annotation["instance_token"]
                target_points["sample_token"] = sample_token
                target_points["category"] = category
                target_points["radar_sample_data_token"] = radar_frame["Token"]
                target_points["sensor"] = radar_frame["Sensor"]
                target_points["radar_timestamp_us"] = radar_frame["Timestamp"]

                point_tables.append(target_points)

    if not point_tables:
        return pd.DataFrame()

    return pd.concat(point_tables, ignore_index=True)


def build_target_radar_velocity_df(target_radar_points_df):
    """Build target-level summaries of original RADAR relative velocity."""

    required = {
        "annotation_token",
        "instance_token",
        "sample_token",
        "category",
        "sensor",
        "vrel_x",
        "vrel_y",
        "vrel_z",
    }
    _require_columns(target_radar_points_df, required, "target_radar_points_df")

    data = target_radar_points_df[list(required)].copy()

    data["vrel_magnitude"] = np.sqrt(
        data["vrel_x"] ** 2 + data["vrel_y"] ** 2 + data["vrel_z"] ** 2
    )

    return data.groupby(
        ["annotation_token", "instance_token", "sample_token", "category"],
        as_index=False,
    ).agg(
        radar_return_count=("vrel_magnitude", "size"),
        radar_sensor_count=("sensor", "nunique"),
        median_vrel_magnitude=("vrel_magnitude", "median"),
        mean_vrel_magnitude=("vrel_magnitude", "mean"),
        std_vrel_magnitude=("vrel_magnitude", "std"),
    )


def build_radar_radial_df(
    target_radar_points_df,
    sample_data_lookup,
    ego_pose_lookup,
    calibrated_sensor_lookup,
    ego_motion_cabin,
):
    """Build ego-compensated point-level RADAR radial-velocity constraints."""

    required = {
        "annotation_token",
        "radar_sample_data_token",
        "radar_timestamp_us",
        "sensor",
        "x",
        "y",
        "z",
        "vrel_x",
        "vrel_y",
        "vrel_z",
    }
    _require_columns(target_radar_points_df, required, "target_radar_points_df")

    radar_tokens = set(target_radar_points_df["radar_sample_data_token"])
    missing_sample_data = radar_tokens - set(sample_data_lookup)

    if missing_sample_data:
        raise ValueError(
            f"{len(missing_sample_data)} RADAR records could not be matched "
            "to sample_data metadata."
        )

    motion_df = (
        pd.DataFrame(ego_motion_cabin)
        .sort_values("timestamp")
        .rename(columns={"timestamp": "cabin_timestamp_us"})
    )

    _require_columns(
        motion_df,
        {"cabin_timestamp_us", "vx", "vy", "vz"},
        "ego_motion_cabin",
    )

    radar_observations = (
        target_radar_points_df[
            ["radar_sample_data_token", "radar_timestamp_us"]
        ]
        .drop_duplicates()
        .sort_values("radar_timestamp_us")
    )

    radar_observations = pd.merge_asof(
        radar_observations,
        motion_df,
        left_on="radar_timestamp_us",
        right_on="cabin_timestamp_us",
        direction="nearest",
    )

    if radar_observations[["vx", "vy", "vz"]].isna().any().any():
        raise ValueError(
            "Unable to match all RADAR observations with ego_motion_cabin records."
        )

    motion_lookup = radar_observations.set_index(
        "radar_sample_data_token"
    ).to_dict("index")

    radial_tables = []

    for sample_data_token, group in target_radar_points_df.groupby(
        "radar_sample_data_token"
    ):
        sample_data_record = sample_data_lookup[sample_data_token]

        ego_pose_token = sample_data_record["ego_pose_token"]
        calibrated_sensor_token = sample_data_record["calibrated_sensor_token"]

        if ego_pose_token not in ego_pose_lookup:
            raise ValueError(
                f"Ego pose not found for RADAR sample_data token: {sample_data_token}"
            )

        if calibrated_sensor_token not in calibrated_sensor_lookup:
            raise ValueError(
                f"Calibrated sensor not found for RADAR sample_data token: "
                f"{sample_data_token}"
            )

        ego_pose = ego_pose_lookup[ego_pose_token]
        calibrated_sensor = calibrated_sensor_lookup[calibrated_sensor_token]
        motion = motion_lookup[sample_data_token]

        group = group.copy()

        position_sensor = group[["x", "y", "z"]].to_numpy()
        position_norm = np.linalg.norm(position_sensor, axis=1)
        valid = position_norm > 0

        if not valid.all():
            group = group.loc[valid].copy()
            position_sensor = position_sensor[valid]
            position_norm = position_norm[valid]

        if group.empty:
            continue

        los_sensor = position_sensor / position_norm[:, None]
        vrel_sensor = group[["vrel_x", "vrel_y", "vrel_z"]].to_numpy()
        relative_radial = np.sum(vrel_sensor * los_sensor, axis=1)

        sensor_to_ego = Quaternion(
            calibrated_sensor["rotation"]
        ).rotation_matrix
        los_ego = los_sensor @ sensor_to_ego.T

        cabin_velocity_ego = np.array([
            motion["vx"],
            motion["vy"],
            motion["vz"],
        ])

        ego_radial = los_ego @ cabin_velocity_ego
        radar_radial = relative_radial + ego_radial

        ego_to_global = Quaternion(
            ego_pose["rotation"]
        ).rotation_matrix
        los_global = los_ego @ ego_to_global.T

        group["relative_radial_velocity_mps"] = relative_radial
        group["ego_radial_velocity_mps"] = ego_radial
        group["radar_radial_velocity_mps"] = radar_radial
        group["los_global_x"] = los_global[:, 0]
        group["los_global_y"] = los_global[:, 1]
        group["los_global_z"] = los_global[:, 2]

        radial_tables.append(group)

    if not radial_tables:
        return pd.DataFrame()

    return pd.concat(radial_tables, ignore_index=True)

def build_radar_radial_validation_df(radar_radial_df, vehicle_reference_df):
    """Validate RADAR radial velocity against projected reference velocity."""

    radial_required = {
        "annotation_token",
        "sensor",
        "radar_radial_velocity_mps",
        "los_global_x",
        "los_global_y",
    }
    reference_required = {
        "start_annotation_token",
        "end_annotation_token",
        "instance_token",
        "category",
        "vx_mps",
        "vy_mps",
    }

    _require_columns(radar_radial_df, radial_required, "radar_radial_df")
    _require_columns(vehicle_reference_df, reference_required, "vehicle_reference_df")

    radial_by_annotation = {
        token: group
        for token, group in radar_radial_df.groupby("annotation_token")
    }

    validation_tables = []

    for _, interval in vehicle_reference_df.iterrows():
        points = radial_by_annotation.get(interval["start_annotation_token"])

        if points is None or points.empty:
            continue

        data = points[
            [
                "annotation_token",
                "sensor",
                "radar_radial_velocity_mps",
                "los_global_x",
                "los_global_y",
            ]
        ].copy()

        data["reference_radial_velocity_mps"] = (
            interval["vx_mps"] * data["los_global_x"]
            + interval["vy_mps"] * data["los_global_y"]
        )

        data["radial_residual_mps"] = (
            data["radar_radial_velocity_mps"]
            - data["reference_radial_velocity_mps"]
        )

        data["radial_abs_error_mps"] = data["radial_residual_mps"].abs()

        data["start_annotation_token"] = interval["start_annotation_token"]
        data["end_annotation_token"] = interval["end_annotation_token"]
        data["instance_token"] = interval["instance_token"]
        data["category"] = interval["category"]

        validation_tables.append(data)

    if not validation_tables:
        return pd.DataFrame()

    return pd.concat(validation_tables, ignore_index=True)

def _huber_irls(U, b, max_iter=20, tol=1e-6):
    """Estimate a velocity vector using Huber iteratively reweighted least squares."""
    velocity, _, _, _ = np.linalg.lstsq(U, b, rcond=None)

    for _ in range(max_iter):
        residual = b - U @ velocity
        residual_median = np.median(residual)
        mad = np.median(np.abs(residual - residual_median))
        robust_scale = 1.4826 * mad

        if robust_scale < 1e-6:
            break

        huber_limit = 1.345 * robust_scale
        abs_residual = np.abs(residual)
        weights = np.ones_like(abs_residual)
        large = abs_residual > huber_limit
        weights[large] = huber_limit / abs_residual[large]

        sqrt_weights = np.sqrt(weights)
        weighted_U = U * sqrt_weights[:, None]
        weighted_b = b * sqrt_weights
        new_velocity, _, _, _ = np.linalg.lstsq(weighted_U, weighted_b, rcond=None)

        if np.linalg.norm(new_velocity - velocity) < tol:
            velocity = new_velocity
            break
        velocity = new_velocity

    residual = U @ velocity - b
    rmse = np.sqrt(np.mean(residual**2))
    return velocity, rmse


def build_radar_velocity_df(
    radar_radial_df,
    vehicle_reference_df,
    condition_limit=20.0,
):
    """
    Reconstruct interval-level RADAR-derived 2D target velocity.

    RADAR constraints from both endpoints of each annotation interval are
    combined using Huber robust least squares. The output retains all full-rank
    estimates and marks numerically stable estimates with `is_stable_2d`.
    """
    radar_required = {
        "annotation_token",
        "sensor",
        "radar_radial_velocity_mps",
        "los_global_x",
        "los_global_y",
    }
    reference_required = {
        "start_annotation_token",
        "end_annotation_token",
        "instance_token",
        "category",
    }
    _require_columns(radar_radial_df, radar_required, "radar_radial_df")
    _require_columns(vehicle_reference_df, reference_required, "vehicle_reference_df")

    if condition_limit <= 0:
        raise ValueError("condition_limit must be greater than zero.")

    radial_by_annotation = {
        token: group for token, group in radar_radial_df.groupby("annotation_token")
    }
    velocity_records = []

    for _, interval in vehicle_reference_df.iterrows():
        start_token = interval["start_annotation_token"]
        end_token = interval["end_annotation_token"]
        start_points = radial_by_annotation.get(start_token)
        end_points = radial_by_annotation.get(end_token)

        if start_points is None or end_points is None:
            continue

        U2 = np.vstack(
            [
                start_points[["los_global_x", "los_global_y"]].to_numpy(),
                end_points[["los_global_x", "los_global_y"]].to_numpy(),
            ]
        )
        b = np.concatenate(
            [
                start_points["radar_radial_velocity_mps"].to_numpy(),
                end_points["radar_radial_velocity_mps"].to_numpy(),
            ]
        )

        if np.linalg.matrix_rank(U2) < 2:
            continue

        condition_2d = np.linalg.cond(U2)
        velocity_2d, radial_rmse = _huber_irls(U2, b)

        velocity_records.append(
            {
                "start_annotation_token": start_token,
                "end_annotation_token": end_token,
                "instance_token": interval["instance_token"],
                "category": interval["category"],
                "radar_vx_mps": velocity_2d[0],
                "radar_vy_mps": velocity_2d[1],
                "radar_speed_2d_mps": np.linalg.norm(velocity_2d),
                "condition_2d": condition_2d,
                "radial_rmse_2d_mps": radial_rmse,
                "radar_return_count": len(start_points) + len(end_points),
                "radar_sensor_count": len(
                    set(start_points["sensor"]) | set(end_points["sensor"])
                ),
                "is_stable_2d": condition_2d <= condition_limit,
            }
        )

    return pd.DataFrame(velocity_records)