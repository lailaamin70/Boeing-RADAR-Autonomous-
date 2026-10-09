import numpy as np
import pandas as pd

from .condition_velocity import angle_difference


def add_fusion_columns(df, min_speed=2.0):
    """Add sensor complementarity and equal-weight vector fusion results."""

    df = df.copy()
    radar = df["radar_stable_available"].fillna(False)
    lidar = df["lidar_velocity_available"].fillna(False)

    both = radar & lidar
    radar_only = radar & ~lidar
    lidar_only = ~radar & lidar

    df["fusion_state"] = np.select(
        [both, radar_only, lidar_only],
        ["Both", "RADAR only", "LiDAR only"],
        default="Neither",
    )
    df["combined_available"] = radar | lidar

    # Equal-weight vector fusion when both sensors are available.
    df["fused_vx_mps"] = ((df["radar_vx_mps"] + df["lidar_vx_mps"]) / 2).where(both)
    df["fused_vy_mps"] = ((df["radar_vy_mps"] + df["lidar_vy_mps"]) / 2).where(both)
    df["fused_speed_2d_mps"] = np.hypot(df["fused_vx_mps"], df["fused_vy_mps"])
    df["fused_abs_error_2d_mps"] = (df["fused_speed_2d_mps"] - df["speed_2d_mps"]).abs()

    # Combined system: fuse when both are available, otherwise use the available sensor.
    df["combined_vx_mps"] = np.select(
        [both, radar_only, lidar_only],
        [(df["radar_vx_mps"] + df["lidar_vx_mps"]) / 2, df["radar_vx_mps"], df["lidar_vx_mps"]],
        default=np.nan,
    )
    df["combined_vy_mps"] = np.select(
        [both, radar_only, lidar_only],
        [(df["radar_vy_mps"] + df["lidar_vy_mps"]) / 2, df["radar_vy_mps"], df["lidar_vy_mps"]],
        default=np.nan,
    )
    df["combined_speed_2d_mps"] = np.hypot(df["combined_vx_mps"], df["combined_vy_mps"])
    df["combined_abs_error_2d_mps"] = (df["combined_speed_2d_mps"] - df["speed_2d_mps"]).abs()

    moving = df["speed_2d_mps"] >= min_speed
    df["fused_dir_error_deg"] = angle_difference(
        df["vx_mps"], df["vy_mps"], df["fused_vx_mps"], df["fused_vy_mps"]
    ).where(moving & both)

    df["combined_dir_error_deg"] = angle_difference(
        df["vx_mps"], df["vy_mps"], df["combined_vx_mps"], df["combined_vy_mps"]
    ).where(moving & df["combined_available"])

    return df


def complementarity_summary(df):
    """Summarise individual and combined velocity availability."""

    total = len(df)
    counts = df["fusion_state"].value_counts().reindex(
        ["Both", "RADAR only", "LiDAR only", "Neither"],
        fill_value=0,
    )

    radar_n = int(df["radar_stable_available"].sum())
    lidar_n = int(df["lidar_velocity_available"].sum())
    combined_n = int(df["combined_available"].sum())

    return pd.DataFrame({
        "metric": [
            "RADAR stable", "LiDAR available", "Both", "RADAR only", "LiDAR only",
            "Neither", "Combined available", "Incremental vs RADAR", "Incremental vs LiDAR",
        ],
        "intervals": [
            radar_n, lidar_n, counts["Both"], counts["RADAR only"], counts["LiDAR only"],
            counts["Neither"], combined_n, combined_n - radar_n, combined_n - lidar_n,
        ],
        "pct_reference": [
            radar_n / total * 100,
            lidar_n / total * 100,
            counts["Both"] / total * 100,
            counts["RADAR only"] / total * 100,
            counts["LiDAR only"] / total * 100,
            counts["Neither"] / total * 100,
            combined_n / total * 100,
            (combined_n - radar_n) / total * 100,
            (combined_n - lidar_n) / total * 100,
        ],
    }).round(1)


def fusion_accuracy_summary(df):
    """Compare RADAR, LiDAR and fusion on the same paired intervals."""

    paired = df[df["fusion_state"] == "Both"]
    rows = []

    for sensor, error_col, dir_col in [
        ("RADAR", "radar_abs_error_2d_mps", "radar_dir_error_deg"),
        ("LiDAR", "lidar_abs_error_2d_mps", "lidar_dir_error_deg"),
        ("Fused", "fused_abs_error_2d_mps", "fused_dir_error_deg"),
    ]:
        error = paired[error_col].dropna()
        direction = paired[dir_col].dropna()

        rows.append({
            "sensor": sensor,
            "intervals": len(error),
            "speed_err_median": error.median(),
            "speed_err_iqr": error.quantile(0.75) - error.quantile(0.25),
            "speed_err_p90": error.quantile(0.90),
            "dir_err_median": direction.median(),
            "dir_err_p90": direction.quantile(0.90),
        })

    return pd.DataFrame(rows).round(3)


def fusion_condition_summary(df, condition):
    """Compare RADAR, LiDAR and fusion accuracy by condition on paired intervals."""

    paired = df[df["fusion_state"] == "Both"]
    rows = []

    for category, group in paired.groupby(condition, observed=True):
        for sensor, error_col in [
            ("RADAR", "radar_abs_error_2d_mps"),
            ("LiDAR", "lidar_abs_error_2d_mps"),
            ("Fused", "fused_abs_error_2d_mps"),
        ]:
            error = group[error_col].dropna()

            rows.append({
                condition: category,
                "sensor": sensor,
                "intervals": len(error),
                "speed_err_median": error.median(),
                "speed_err_iqr": error.quantile(0.75) - error.quantile(0.25),
                "speed_err_p90": error.quantile(0.90),
            })

    return pd.DataFrame(rows).round(3)


def fusion_gain_summary(fusion_summary, condition):
    """Calculate the median-error change produced by fusion."""

    table = fusion_summary.pivot(
        index=condition,
        columns="sensor",
        values="speed_err_median",
    ).reset_index()

    table["fusion_minus_lidar"] = table["Fused"] - table["LiDAR"]
    table["fusion_minus_radar"] = table["Fused"] - table["RADAR"]

    return table.round(3)