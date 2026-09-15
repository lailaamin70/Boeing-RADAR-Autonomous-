"""
Sensor observation utilities for TruckScenes.

Provides functions for:
- building RADAR, LiDAR, and Camera observation tables
- summarising observation availability
- calculating sampling intervals and frequencies

TruckScenes timestamps are stored in microseconds (µs).
This module uses metadata only and does not load sensor data.
"""

from pathlib import Path
import pandas as pd
from .config import METADATA_ROOT
from .data_io import load_json

TIMESTAMP_UNITS_PER_SECOND = 1_000_000


def build_sensor_data_df(metadata_root=None):
    """
    Build a sensor observation table from TruckScenes metadata.
    Includes samples and sweeps for RADAR, LiDAR, and Camera.

    Parameters
    ----------
    metadata_root : str or Path, optional
        Metadata directory. Uses METADATA_ROOT by default.

    Returns
    -------
    pandas.DataFrame
        Sensor observation metadata.

    Notes
    -----
    Timestamps are preserved in microseconds (µs).
    """

    metadata_root = Path(metadata_root or METADATA_ROOT)

    if not metadata_root.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_root}")

    required_files = [
        "sample.json",
        "sample_data.json",
        "calibrated_sensor.json",
        "sensor.json",
    ]

    for filename in required_files:
        file_path = metadata_root / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required metadata file not found: {file_path}")

    samples = load_json(metadata_root / "sample.json")
    sample_data = load_json(metadata_root / "sample_data.json")
    calibrated_sensors = load_json(metadata_root / "calibrated_sensor.json")
    sensors = load_json(metadata_root / "sensor.json")

    for name, data in {
        "sample.json": samples,
        "sample_data.json": sample_data,
        "calibrated_sensor.json": calibrated_sensors,
        "sensor.json": sensors,
    }.items():
        if not isinstance(data, list):
            raise ValueError(f"{name} is expected to contain a list of records.")
        if not all(isinstance(record, dict) for record in data):
            raise ValueError(f"{name} contains one or more non-dictionary records.")

    sample_to_scene = {
        record["token"]: record.get("scene_token")
        for record in samples if "token" in record
    }
    calibrated_sensor_lookup = {
        record["token"]: record
        for record in calibrated_sensors if "token" in record
    }
    sensor_lookup = {
        record["token"]: record
        for record in sensors if "token" in record
    }

    records = []
    skipped_records = 0

    for record in sample_data:
        sample_token = record.get("sample_token")
        calibrated_sensor_token = record.get("calibrated_sensor_token")

        if sample_token is None or calibrated_sensor_token is None:
            skipped_records += 1
            continue

        calibrated_sensor = calibrated_sensor_lookup.get(calibrated_sensor_token)
        if calibrated_sensor is None:
            skipped_records += 1
            continue

        sensor_token = calibrated_sensor.get("sensor_token")
        if sensor_token is None:
            skipped_records += 1
            continue

        sensor = sensor_lookup.get(sensor_token)
        if sensor is None:
            skipped_records += 1
            continue

        modality_raw = sensor.get("modality")
        if not isinstance(modality_raw, str):
            skipped_records += 1
            continue

        modality_map = {
            "lidar": "LiDAR",
            "radar": "RADAR",
            "camera": "Camera",
        }
        modality = modality_map.get(modality_raw.lower())

        if modality is None:
            continue

        filename = record.get("filename")

        if isinstance(filename, str):
            if filename.startswith("samples/"):
                data_group = "samples"
            elif filename.startswith("sweeps/"):
                data_group = "sweeps"
            else:
                data_group = "other"
        else:
            filename = None
            data_group = "unknown"

        records.append({
            "Token": record.get("token"),
            "Sample Token": sample_token,
            "Scene Token": sample_to_scene.get(sample_token),
            "Modality": modality,
            "Sensor": sensor.get("channel"),
            "Timestamp": record.get("timestamp"),
            "Is Key Frame": record.get("is_key_frame"),
            "Data Group": data_group,
            "Filename": filename,
        })

    if not records:
        raise ValueError("No supported sensor observations could be constructed from the metadata.")

    sensor_data_df = pd.DataFrame(records)
    sensor_data_df.attrs["Skipped Records"] = skipped_records

    return sensor_data_df

def get_observation_summary(sensor_data_df):
    """
    Summarise observations by modality, sensor, and data group.
    """

    required_columns = {"Modality", "Sensor", "Data Group", "Timestamp"}
    missing_columns = required_columns - set(sensor_data_df.columns)

    if missing_columns:
        raise ValueError(
            "sensor_data_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if sensor_data_df.empty:
        raise ValueError("sensor_data_df is empty. No sensor observations are available.")

    return (
        sensor_data_df
        .groupby(["Modality", "Sensor", "Data Group"], dropna=False)
        .agg(
            Number_of_Observations=("Timestamp", "size"),
            Unique_Timestamps=("Timestamp", "nunique"),
            Missing_Timestamps=("Timestamp", lambda x: x.isna().sum()),
        )
    )


def get_sampling_intervals(sensor_data_df):
    """
    Calculate intervals between consecutive observations within each scene.
    """

    required_columns = {"Modality", "Sensor", "Scene Token", "Timestamp"}
    missing_columns = required_columns - set(sensor_data_df.columns)

    if missing_columns:
        raise ValueError(
            "sensor_data_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if sensor_data_df.empty:
        raise ValueError("sensor_data_df is empty. No sensor observations are available.")

    data = sensor_data_df.copy()

    missing_scene_count = data["Scene Token"].isna().sum()
    if missing_scene_count > 0:
        raise ValueError(
            f"{missing_scene_count} sensor observations do not have a Scene Token."
        )

    data["Timestamp"] = pd.to_numeric(data["Timestamp"], errors="coerce")
    missing_timestamp_count = data["Timestamp"].isna().sum()

    interval_records = []

    for (modality, sensor, scene_token), group in data.groupby(
        ["Modality", "Sensor", "Scene Token"]
    ):
        timestamps = (
            group["Timestamp"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .to_numpy()
        )

        if len(timestamps) < 2:
            continue

        for interval in timestamps[1:] - timestamps[:-1]:
            interval_records.append({
                "Modality": modality,
                "Sensor": sensor,
                "Scene Token": scene_token,
                "Timestamp Interval": interval,
            })

    interval_df = pd.DataFrame(interval_records)
    interval_df.attrs["Missing Timestamps"] = missing_timestamp_count

    return interval_df

def get_sampling_interval_summary(interval_df):
    """
    Summarise sampling intervals by sensor.
    """

    required_columns = {"Modality", "Sensor", "Timestamp Interval"}
    missing_columns = required_columns - set(interval_df.columns)

    if missing_columns:
        raise ValueError(
            "interval_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if interval_df.empty:
        raise ValueError("interval_df is empty. No timestamp intervals are available.")

    data = interval_df.copy()
    data["Timestamp Interval"] = pd.to_numeric(
        data["Timestamp Interval"], errors="coerce"
    )

    return (
        data
        .groupby(["Modality", "Sensor"])
        .agg(
            Number_of_Intervals=("Timestamp Interval", "count"),
            Missing_Intervals=("Timestamp Interval", lambda x: x.isna().sum()),
            Mean_Timestamp_Interval=("Timestamp Interval", "mean"),
            Median_Timestamp_Interval=("Timestamp Interval", "median"),
            Min_Timestamp_Interval=("Timestamp Interval", "min"),
            Max_Timestamp_Interval=("Timestamp Interval", "max"),
        )
        .round(2)
    )


def get_frequency_summary(sampling_interval_summary):
    """
    Convert median timestamp intervals into observation frequencies.
    """

    required_columns = {"Median_Timestamp_Interval"}
    missing_columns = required_columns - set(sampling_interval_summary.columns)

    if missing_columns:
        raise ValueError(
            "sampling_interval_summary is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if sampling_interval_summary.empty:
        raise ValueError("sampling_interval_summary is empty.")

    sensor_frequency = sampling_interval_summary.copy()

    sensor_frequency["Median_Timestamp_Interval"] = pd.to_numeric(
        sensor_frequency["Median_Timestamp_Interval"], errors="coerce"
    )

    valid_intervals = sensor_frequency["Median_Timestamp_Interval"] > 0

    sensor_frequency["Median_Interval_ms"] = pd.NA
    sensor_frequency["Approx_Frequency_Hz"] = pd.NA

    sensor_frequency.loc[valid_intervals, "Median_Interval_ms"] = (
        sensor_frequency.loc[valid_intervals, "Median_Timestamp_Interval"]
        / TIMESTAMP_UNITS_PER_SECOND * 1000
    )

    sensor_frequency.loc[valid_intervals, "Approx_Frequency_Hz"] = (
        TIMESTAMP_UNITS_PER_SECOND
        / sensor_frequency.loc[valid_intervals, "Median_Timestamp_Interval"]
    )

    columns = ["Median_Interval_ms", "Approx_Frequency_Hz"]
    sensor_frequency[columns] = sensor_frequency[columns].astype("float64").round(2)

    modality_frequency = (
        sensor_frequency
        .reset_index()
        .groupby("Modality")
        .agg(
            Median_Interval_ms=("Median_Interval_ms", "median"),
            Median_Frequency_Hz=("Approx_Frequency_Hz", "median"),
        )
        .round(2)
    )

    return sensor_frequency, modality_frequency