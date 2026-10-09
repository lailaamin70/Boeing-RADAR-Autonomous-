"""
Utilities for condition-based RADAR/LiDAR velocity analysis.

Calculates velocity availability, speed and direction errors,
and speed-error outliers across different operating conditions.
"""

import numpy as np
import pandas as pd

from pyod.models.mad import MAD


def angle_difference(true_x, true_y, sensor_x, sensor_y):
    """
    Calculate the absolute angle difference between two 2D velocity vectors.

    0 degrees   = same direction
    180 degrees = opposite direction
    """
    cross = true_x * sensor_y - true_y * sensor_x
    dot = true_x * sensor_x + true_y * sensor_y

    return np.degrees(np.abs(np.arctan2(cross, dot)))


def add_velocity_analysis_columns(df, min_speed=2.0):
    """
    Add common velocity-analysis columns.

    Speed error:
        Absolute difference between sensor speed and annotation-derived
        reference speed.

    Direction error:
        Calculated only when reference speed >= min_speed.

    RADAR accuracy uses numerically stable RADAR estimates only.
    """
    df = df.copy()

    # Absolute speed error
    for sensor in ["radar", "lidar"]:
        df[f"{sensor}_abs_error_2d_mps"] = (df[f"{sensor}_speed_2d_mps"] - df["speed_2d_mps"]
        ).abs()

    # Direction error
    is_moving = df["speed_2d_mps"] >= min_speed

    availability = {
        "radar": "radar_stable_available",
        "lidar": "lidar_velocity_available",
    }

    for sensor, availability_col in availability.items():
        angle = angle_difference(
            df["vx_mps"],
            df["vy_mps"],
            df[f"{sensor}_vx_mps"],
            df[f"{sensor}_vy_mps"],
        )

        df[f"{sensor}_dir_error_deg"] = angle.where(is_moving & df[availability_col])

    # Same intervals available for both sensors
    df["paired_available"] = (
        df["radar_stable_available"] & df["lidar_velocity_available"])

    return df


def availability_summary(df, tag):
    """
    Summarise sample size and velocity availability for each condition value.
    """
    table = (
        df.groupby(tag, observed=True)
        .agg(
            scenes=("scene_token", "nunique"),
            instances=("instance_token", "nunique"),
            intervals=("start_annotation_token", "size"),
            radar_reconstructable_n=(
                "radar_reconstructable_available",
                "sum",
            ),
            radar_stable_n=(
                "radar_stable_available",
                "sum",
            ),
            lidar_n=(
                "lidar_velocity_available",
                "sum",
            ),
            paired_n=(
                "paired_available",
                "sum",
            ),
        )
    )

    table["radar_reconstructable_pct"] = table["radar_reconstructable_n"] / table["intervals"] * 100
    table["radar_stable_pct"] = table["radar_stable_n"] / table["intervals"] * 100
    table["radar_stable_given_reconstructable_pct"] = table["radar_stable_n"] / table["radar_reconstructable_n"].replace(0, np.nan) * 100
    table["lidar_pct"] = table["lidar_n"] / table["intervals"] * 100
    table["paired_pct"] = table["paired_n"] / table["intervals"] * 100
    count_columns = [
        "scenes",
        "instances",
        "intervals",
        "radar_reconstructable_n",
        "radar_stable_n",
        "lidar_n",
        "paired_n",
    ]
    table[count_columns] = table[count_columns].fillna(0).astype(int)

    return table.reset_index()


def detect_speed_error_outliers(speed_error, threshold=3.5):
    """
    Detect speed-error outliers using PyOD MAD.
    Outliers are flagged for descriptive analysis only and are not removed.
    """
    speed_error = speed_error.dropna()

    if len(speed_error) < 2:
        return pd.Series(False, index=speed_error.index)

    median = speed_error.median()
    mad = (speed_error - median).abs().median()

    if mad == 0 or pd.isna(mad):
        return pd.Series(False, index=speed_error.index)

    detector = MAD(threshold=threshold)
    detector.fit(speed_error.to_numpy().reshape(-1, 1))

    return pd.Series(
        detector.labels_.astype(bool),
        index=speed_error.index,
    )


def error_summary(
    df,
    tag,
    both_sensors_only=False,
    outlier_threshold=3.5,
):
    """
    Summarise speed and direction errors for each condition.

    Supports both paired-sensor and individual-sensor comparisons,
    including speed-error outlier statistics.
    """
    if both_sensors_only:
        df = df[df["paired_available"]].copy()

    availability = {
        "radar": ("RADAR", "radar_stable_available"),
        "lidar": ("LiDAR", "lidar_velocity_available"),
    }

    rows = []

    for group_name, group in df.groupby(tag, observed=True):
        for sensor, (label, availability_col) in availability.items():

            available = group[group[availability_col]]

            speed_error = available[
                f"{sensor}_abs_error_2d_mps"
            ].dropna()

            direction_error = available[
                f"{sensor}_dir_error_deg"
            ].dropna()

            if speed_error.empty:
                speed_median = np.nan
                speed_iqr = np.nan
                speed_std = np.nan
                speed_p90 = np.nan
                outlier_n = 0
                outlier_pct = np.nan

            else:
                q1 = speed_error.quantile(0.25)
                q3 = speed_error.quantile(0.75)

                speed_median = speed_error.median()
                speed_iqr = q3 - q1
                speed_std = speed_error.std()
                speed_p90 = speed_error.quantile(0.90)

                outliers = detect_speed_error_outliers(
                    speed_error,
                    threshold=outlier_threshold,
                )

                outlier_n = int(outliers.sum())
                outlier_pct = outliers.mean() * 100

            rows.append({
                tag: group_name,
                "sensor": label,

                "scenes": group["scene_token"].nunique(),
                "instances": available["instance_token"].nunique(),
                "intervals": len(available),

                "speed_err_median": speed_median,
                "speed_err_iqr": speed_iqr,
                "speed_err_std": speed_std,
                "speed_err_p90": speed_p90,

                "speed_err_outlier_n": outlier_n,
                "speed_err_outlier_pct": outlier_pct,

                "moving_intervals": len(direction_error),
                "dir_err_median": direction_error.median(),
                "dir_err_p90": direction_error.quantile(0.90),
            })

    return pd.DataFrame(rows)