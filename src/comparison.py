"""
Reusable velocity-comparison utilities for the TruckScenes project.
Validated in 05_Velocity_Comparison.ipynb.
This module builds the common RADAR-LiDAR comparison table from already processed reference, RADAR, and LiDAR results.
"""

import pandas as pd


SPEED_BINS_KMH = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, float("inf")]
SPEED_LABELS_KMH = [
    "0–<10", "10–<20", "20–<30", "30–<40", "40–<50", "50–<60",
    "60–<70", "70–<80", "80–<90", "90–<100", "100–<110", "≥110",
]


def _require_columns(df, required, name):
    """Check that required DataFrame columns are present."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _require_unique(df, keys, name):
    """Check that the intended join key is unique."""
    if df.duplicated(subset=keys).any():
        raise ValueError(
            f"{name} contains duplicate rows for key: "
            + ", ".join(keys)
        )


def build_velocity_comparison_df(
    vehicle_reference_df,
    radar_velocity_df,
    lidar_velocity_df,
):
    """
    Build the common interval-level velocity comparison table.
    Each row represents one valid annotation-derived reference interval.
    RADAR and LiDAR velocity results are left-joined so unavailable estimates are retained.
    """

    interval_key = ["start_annotation_token", "end_annotation_token"]

    reference_columns = interval_key + [
        "instance_token", "category",
        "vx_mps", "vy_mps", "vz_mps",
        "speed_2d_mps", "speed_3d_mps",
    ]

    radar_columns = interval_key + [
        "radar_vx_mps", "radar_vy_mps", "radar_speed_2d_mps",
    ]

    lidar_columns = interval_key + [
        "lidar_vx_mps", "lidar_vy_mps", "lidar_vz_mps",
        "lidar_speed_2d_mps", "lidar_speed_3d_mps",
    ]

    _require_columns(vehicle_reference_df, reference_columns, "vehicle_reference_df")
    _require_columns(radar_velocity_df, radar_columns, "radar_velocity_df")
    _require_columns(lidar_velocity_df, lidar_columns, "lidar_velocity_df")

    _require_unique(vehicle_reference_df, interval_key, "vehicle_reference_df")
    _require_unique(radar_velocity_df, interval_key, "radar_velocity_df")
    _require_unique(lidar_velocity_df, interval_key, "lidar_velocity_df")

    result = (
        vehicle_reference_df[reference_columns]
        .merge(
            radar_velocity_df[radar_columns],
            on=interval_key,
            how="left",
            validate="one_to_one",
        )
        .merge(
            lidar_velocity_df[lidar_columns],
            on=interval_key,
            how="left",
            validate="one_to_one",
        )
    )

    result["radar_velocity_available"] = result["radar_speed_2d_mps"].notna()
    result["lidar_velocity_available"] = result["lidar_speed_2d_mps"].notna()

    return result


def add_sensor_information(
    comparison_df,
    target_radar_df,
    target_lidar_df,
):
    """
    Add interval-level RADAR and LiDAR target-information metrics.
    """

    _require_columns(
        comparison_df,
        {"start_annotation_token", "end_annotation_token"},
        "comparison_df",
    )
    _require_columns(
        target_radar_df,
        {"annotation_token", "radar_return_count", "radar_sensor_count"},
        "target_radar_df",
    )
    _require_columns(
        target_lidar_df,
        {"annotation_token", "lidar_point_count"},
        "target_lidar_df",
    )

    _require_unique(target_radar_df, ["annotation_token"], "target_radar_df")
    _require_unique(target_lidar_df, ["annotation_token"], "target_lidar_df")

    radar_info = target_radar_df[
        ["annotation_token", "radar_return_count", "radar_sensor_count"]
    ].copy()

    lidar_info = target_lidar_df[
        ["annotation_token", "lidar_point_count"]
    ].copy()

    result = comparison_df.copy()

    result = result.merge(
        radar_info.rename(columns={
            "annotation_token": "start_annotation_token",
            "radar_return_count": "radar_start_return_count",
            "radar_sensor_count": "radar_start_sensor_count",
        }),
        on="start_annotation_token",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        radar_info.rename(columns={
            "annotation_token": "end_annotation_token",
            "radar_return_count": "radar_end_return_count",
            "radar_sensor_count": "radar_end_sensor_count",
        }),
        on="end_annotation_token",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        lidar_info.rename(columns={
            "annotation_token": "start_annotation_token",
            "lidar_point_count": "lidar_start_point_count",
        }),
        on="start_annotation_token",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        lidar_info.rename(columns={
            "annotation_token": "end_annotation_token",
            "lidar_point_count": "lidar_end_point_count",
        }),
        on="end_annotation_token",
        how="left",
        validate="many_to_one",
    )

    radar_count_columns = [
        "radar_start_return_count", "radar_end_return_count",
        "radar_start_sensor_count", "radar_end_sensor_count",
    ]
    result[radar_count_columns] = result[radar_count_columns].fillna(0)

    result["radar_info_available"] = (
        (result["radar_start_return_count"] > 0)
        & (result["radar_end_return_count"] > 0)
    )
    result["lidar_info_available"] = (
        (result["lidar_start_point_count"] > 0)
        & (result["lidar_end_point_count"] > 0)
    )

    result["radar_return_count"] = (
        result["radar_start_return_count"] + result["radar_end_return_count"]
    )
    result["lidar_point_count"] = (
        result["lidar_start_point_count"] + result["lidar_end_point_count"]
    )

    return result


def add_speed_ranges(comparison_df):
    """Add reference speed in km/h and the common speed-range labels."""

    _require_columns(comparison_df, {"speed_2d_mps"}, "comparison_df")

    result = comparison_df.copy()
    result["reference_speed_kmh"] = result["speed_2d_mps"] * 3.6
    result["speed_range_kmh"] = pd.cut(
        result["reference_speed_kmh"],
        bins=SPEED_BINS_KMH,
        labels=SPEED_LABELS_KMH,
        right=False,
    )

    return result


def get_velocity_availability_summary(comparison_df):
    """Summarise RADAR/LiDAR velocity-estimation availability overlap."""

    _require_columns(
        comparison_df,
        {"radar_velocity_available", "lidar_velocity_available"},
        "comparison_df",
    )

    radar = comparison_df["radar_velocity_available"].astype(bool)
    lidar = comparison_df["lidar_velocity_available"].astype(bool)

    groups = pd.Series("Neither", index=comparison_df.index, dtype="object")
    groups.loc[radar & lidar] = "Both"
    groups.loc[radar & ~lidar] = "RADAR only"
    groups.loc[~radar & lidar] = "LiDAR only"

    order = ["Both", "RADAR only", "LiDAR only", "Neither"]

    summary = (
        groups.value_counts()
        .reindex(order, fill_value=0)
        .rename("Count")
        .to_frame()
    )
    summary["Percentage"] = (summary["Count"] / len(comparison_df) * 100).round(2)

    return summary