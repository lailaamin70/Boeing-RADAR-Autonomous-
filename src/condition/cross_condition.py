import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .condition_data import create_distance_bands
from .condition_velocity import add_velocity_analysis_columns


def build_cross_condition_df(comparison_df, condition_df):
    """Build one interval-level table containing all analysis conditions."""

    condition_lookup = condition_df.set_index("annotation_token")
    df = comparison_df.copy()

    df["start_scene_token"] = df["start_annotation_token"].map(condition_lookup["scene_token"])
    df["end_scene_token"] = df["end_annotation_token"].map(condition_lookup["scene_token"])
    df["start_instance_token"] = df["start_annotation_token"].map(condition_lookup["instance_token"])
    df["end_instance_token"] = df["end_annotation_token"].map(condition_lookup["instance_token"])

    if not (df["start_scene_token"] == df["end_scene_token"]).all():
        raise ValueError("Some velocity intervals cross scene boundaries.")
    if not (df["start_instance_token"] == df["end_instance_token"]).all():
        raise ValueError("Some velocity intervals cross object instances.")

    df["scene_token"] = df["start_scene_token"]
    df["instance_token"] = df["start_instance_token"]
    df["area"] = df["start_annotation_token"].map(condition_lookup["area"])
    df["weather"] = df["start_annotation_token"].map(condition_lookup["weather"])
    df["object_type"] = df["start_annotation_token"].map(condition_lookup["object_type"])

    start_visibility = df["start_annotation_token"].map(condition_lookup["visibility"])
    end_visibility = df["end_annotation_token"].map(condition_lookup["visibility"])
    df["visibility"] = pd.concat([start_visibility, end_visibility], axis=1).min(axis=1)

    start_distance = df["start_annotation_token"].map(condition_lookup["distance_3d"])
    end_distance = df["end_annotation_token"].map(condition_lookup["distance_3d"])
    df["distance_3d"] = (start_distance + end_distance) / 2
    df["distance"] = create_distance_bands(df["distance_3d"])

    required = ["scene_token", "instance_token", "area", "weather", "visibility", "object_type", "distance_3d"]
    if df[required].isna().any().any():
        raise ValueError("Missing values found in cross-condition variables.")

    return add_velocity_analysis_columns(df)


def build_paired_long_df(cross_condition_df):
    """Convert paired RADAR-LiDAR intervals into long format."""

    paired = cross_condition_df[cross_condition_df["paired_available"]].copy()

    common_cols = [
        "scene_token", "instance_token", "start_annotation_token", "end_annotation_token",
        "area", "weather", "visibility", "object_type", "distance_3d", "distance", "speed_2d_mps",
    ]

    radar = paired[common_cols].copy()
    radar["sensor"] = "RADAR"
    radar["abs_error_2d_mps"] = paired["radar_abs_error_2d_mps"].values
    radar["dir_error_deg"] = paired["radar_dir_error_deg"].values

    lidar = paired[common_cols].copy()
    lidar["sensor"] = "LiDAR"
    lidar["abs_error_2d_mps"] = paired["lidar_abs_error_2d_mps"].values
    lidar["dir_error_deg"] = paired["lidar_dir_error_deg"].values

    return pd.concat([radar, lidar], ignore_index=True)


def prepare_interaction_model_df(paired_long_df):
    """Prepare paired observations for multivariable interaction modelling."""

    df = paired_long_df.copy()
    df["log_abs_error"] = np.log1p(df["abs_error_2d_mps"])
    df["distance_10m"] = df["distance_3d"] / 10.0

    model_columns = [
        "scene_token", "instance_token", "start_annotation_token", "end_annotation_token",
        "sensor", "area", "weather", "visibility", "object_type", "distance_10m",
        "abs_error_2d_mps", "log_abs_error",
    ]

    return df[model_columns].dropna().copy()


def fit_sensor_condition_model(model_df, clustered=False):
    """Fit the multivariable sensor-condition interaction model."""

    formula = (
        'log_abs_error ~ C(sensor, Treatment(reference="LiDAR")) * ('
        'C(area, Treatment(reference="city")) + '
        'C(weather, Treatment(reference="clear")) + '
        'C(visibility, Treatment(reference=4)) + '
        'C(object_type, Treatment(reference="vehicle.car")) + '
        'distance_10m)'
    )

    model = smf.ols(formula, data=model_df)

    if clustered:
        return model.fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["scene_token"]},
        )

    return model.fit()


def fit_condition_interaction_model(model_df, clustered=False):
    """Fit sensor-condition and pairwise condition-condition interactions."""

    sensor = 'C(sensor, Treatment(reference="LiDAR"))'
    conditions = (
        'C(area, Treatment(reference="city")) + '
        'C(weather, Treatment(reference="clear")) + '
        'C(visibility, Treatment(reference=4)) + '
        'C(object_type, Treatment(reference="vehicle.car")) + '
        'distance_10m'
    )

    formula = (f'log_abs_error ~ {sensor} * ({conditions}) + ' f'({conditions}) ** 2')

    model = smf.ols(formula, data=model_df)

    if clustered:
        return model.fit(cov_type="cluster", cov_kwds={"groups": model_df["scene_token"]})

    return model.fit()