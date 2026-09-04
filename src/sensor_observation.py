"""
Sensor observation utilities for the TruckScenes analysis project.

This module provides functions for:
- building RADAR and LiDAR observation tables
- summarising sample and sweep availability
- calculating within-scene sampling intervals
- estimating sensor observation frequencies

TruckScenes metadata timestamps are stored in microseconds (µs).
Raw timestamp values are preserved until conversion is required.

This module does not load full point-cloud data or calculate vehicle velocity.
"""

from pathlib import Path

import pandas as pd

from .config import METADATA_ROOT
from .data_io import load_json

TIMESTAMP_UNITS_PER_SECOND = 1_000_000

# Sensor observation metadata
def build_sensor_data_df(metadata_root=None):
    """
    Build a RADAR and LiDAR observation table from TruckScenes metadata.

    Links sample_data, calibrated_sensor, sensor, and sample metadata.
    Both key samples and intermediate sweeps are included when available.

    Parameters
    ----------
    metadata_root : str or Path, optional
        Path to the TruckScenes metadata directory.
        If None, METADATA_ROOT from config.py is used.

    Returns
    -------
    pandas.DataFrame
        Sensor observation metadata including sample, scene, modality,
        sensor channel, timestamp, data group, and filename.

    Notes
    -----
    TruckScenes timestamps are stored in microseconds (µs).
    Original timestamp values are preserved without conversion.
    """

    if metadata_root is None:
        metadata_root = METADATA_ROOT

    metadata_root = Path(metadata_root)

    if not metadata_root.exists():
        raise FileNotFoundError(
            f"Metadata directory not found: {metadata_root}"
        )

    required_files = [
        "sample.json",
        "sample_data.json",
        "calibrated_sensor.json",
        "sensor.json",
    ]

    for filename in required_files:
        file_path = metadata_root / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required metadata file not found: {file_path}"
            )

    samples = load_json(metadata_root / "sample.json")
    sample_data = load_json(metadata_root / "sample_data.json")
    calibrated_sensors = load_json(
        metadata_root / "calibrated_sensor.json"
    )
    sensors = load_json(metadata_root / "sensor.json")

    for name, data in {
        "sample.json": samples,
        "sample_data.json": sample_data,
        "calibrated_sensor.json": calibrated_sensors,
        "sensor.json": sensors,
    }.items():

        if not isinstance(data, list):
            raise ValueError(
                f"{name} is expected to contain a list of records."
            )

        if not all(isinstance(record, dict) for record in data):
            raise ValueError(
                f"{name} contains one or more non-dictionary records."
            )

    sample_to_scene = {
        record["token"]: record.get("scene_token")
        for record in samples
        if "token" in record
    }

    calibrated_sensor_lookup = {
        record["token"]: record
        for record in calibrated_sensors
        if "token" in record
    }

    sensor_lookup = {
        record["token"]: record
        for record in sensors
        if "token" in record
    }

    records = []
    skipped_records = 0

    for record in sample_data:

        sample_token = record.get("sample_token")
        calibrated_sensor_token = record.get(
            "calibrated_sensor_token"
        )

        if sample_token is None or calibrated_sensor_token is None:
            skipped_records += 1
            continue

        calibrated_sensor = calibrated_sensor_lookup.get(
            calibrated_sensor_token
        )

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

        modality_raw = modality_raw.lower()

        if modality_raw == "lidar":
            modality = "LiDAR"

        elif modality_raw == "radar":
            modality = "RADAR"

        else:
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
            "Data Group": data_group,
            "Filename": filename,
        })

    if not records:
        raise ValueError(
            "No RADAR or LiDAR observations could be constructed "
            "from the metadata."
        )

    sensor_data_df = pd.DataFrame(records)

    sensor_data_df.attrs["Skipped Records"] = skipped_records

    return sensor_data_df

def get_observation_summary(sensor_data_df):
    """
    Summarise sensor-observation availability by modality, sensor channel, and data group.

    Parameters
    ----------
    sensor_data_df : pandas.DataFrame
        Sensor observation table produced by build_sensor_data_df().

    Returns
    -------
    pandas.DataFrame
        Number of observations, unique timestamps, and missing timestamps for each modality, sensor, and data group.
    """

    required_columns = {
        "Modality",
        "Sensor",
        "Data Group",
        "Timestamp",
    }

    missing_columns = required_columns - set(sensor_data_df.columns)

    if missing_columns:
        raise ValueError(
            "sensor_data_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if sensor_data_df.empty:
        raise ValueError(
            "sensor_data_df is empty. No sensor observations are available."
        )

    summary = (
        sensor_data_df
        .groupby(
            ["Modality", "Sensor", "Data Group"],
            dropna=False,
        )
        .agg(
            Number_of_Observations=("Timestamp", "size"),
            Unique_Timestamps=("Timestamp", "nunique"),
            Missing_Timestamps=(
                "Timestamp",
                lambda x: x.isna().sum()
            ),
        )
    )

    return summary


# Sampling intervals
def get_sampling_intervals(sensor_data_df):
    """
    Calculate timestamp intervals between consecutive sensor observationswithin each scene.

    Parameters
    ----------
    sensor_data_df : pandas.DataFrame
        Sensor observation table produced by build_sensor_data_df().

    Returns
    -------
    pandas.DataFrame
        Consecutive timestamp intervals in microseconds (µs) for each sensor within each scene.
    """

    required_columns = {
        "Modality",
        "Sensor",
        "Scene Token",
        "Timestamp",
    }

    missing_columns = required_columns - set(sensor_data_df.columns)

    if missing_columns:
        raise ValueError(
            "sensor_data_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if sensor_data_df.empty:
        raise ValueError(
            "sensor_data_df is empty. No sensor observations are available."
        )

    data = sensor_data_df.copy()

    missing_scene_count = data["Scene Token"].isna().sum()

    if missing_scene_count > 0:
        raise ValueError(
            f"{missing_scene_count} sensor observations do not have "
            "a Scene Token. Sampling intervals cannot be calculated "
            "safely for these records."
        )

    data["Timestamp"] = pd.to_numeric(
        data["Timestamp"],
        errors="coerce"
    )

    missing_timestamp_count = data["Timestamp"].isna().sum()

    interval_records = []

    grouped = data.groupby([
        "Modality",
        "Sensor",
        "Scene Token",
    ])

    for (modality, sensor, scene_token), group in grouped:

        timestamps = (
            group["Timestamp"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .to_numpy()
        )

        if len(timestamps) < 2:
            continue

        intervals = timestamps[1:] - timestamps[:-1]

        for interval in intervals:
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
    Summarise within-scene timestamp intervals by sensor channel.

    Parameters
    ----------
    interval_df : pandas.DataFrame
        Interval table produced by get_sampling_intervals().

    Returns
    -------
    pandas.DataFrame
        Number, mean, median, minimum, and maximum timestamp intervals in microseconds (µs) for each sensor channel.
    """

    required_columns = {
        "Modality",
        "Sensor",
        "Timestamp Interval",
    }

    missing_columns = required_columns - set(interval_df.columns)

    if missing_columns:
        raise ValueError(
            "interval_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if interval_df.empty:
        raise ValueError(
            "interval_df is empty. No timestamp intervals are available."
        )

    data = interval_df.copy()

    data["Timestamp Interval"] = pd.to_numeric(
        data["Timestamp Interval"],
        errors="coerce"
    )

    summary = (
        data
        .groupby(["Modality", "Sensor"])
        .agg(
            Number_of_Intervals=("Timestamp Interval", "count"),
            Missing_Intervals=(
                "Timestamp Interval",
                lambda x: x.isna().sum()
            ),
            Mean_Timestamp_Interval=("Timestamp Interval", "mean"),
            Median_Timestamp_Interval=("Timestamp Interval", "median"),
            Min_Timestamp_Interval=("Timestamp Interval", "min"),
            Max_Timestamp_Interval=("Timestamp Interval", "max"),
        )
        .round(2)
    )

    return summary


# Observation frequency
def get_frequency_summary(sampling_interval_summary):
    """
    Convert median timestamp intervals into approximate observation frequencies.

    Parameters
    ----------
    sampling_interval_summary : pandas.DataFrame
        Output from get_sampling_interval_summary().
        Timestamp intervals are in microseconds (µs).

    Returns
    -------
    sensor_frequency : pandas.DataFrame
        Frequency estimate for each sensor channel.

    modality_frequency : pandas.DataFrame
        Median frequency for each sensor modality.
    """

    required_columns = {
        "Median_Timestamp_Interval",
    }

    missing_columns = (
        required_columns
        - set(sampling_interval_summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "sampling_interval_summary is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if sampling_interval_summary.empty:
        raise ValueError(
            "sampling_interval_summary is empty."
        )

    sensor_frequency = sampling_interval_summary.copy()

    sensor_frequency["Median_Timestamp_Interval"] = pd.to_numeric(
        sensor_frequency["Median_Timestamp_Interval"],
        errors="coerce",
    )

    valid_intervals = (
        sensor_frequency["Median_Timestamp_Interval"] > 0
    )

    sensor_frequency["Median_Interval_ms"] = pd.NA
    sensor_frequency["Approx_Frequency_Hz"] = pd.NA

    sensor_frequency.loc[
        valid_intervals,
        "Median_Interval_ms"
    ] = (
        sensor_frequency.loc[
            valid_intervals,
            "Median_Timestamp_Interval"
        ]
        / TIMESTAMP_UNITS_PER_SECOND
        * 1000
    )

    sensor_frequency.loc[
        valid_intervals,
        "Approx_Frequency_Hz"
    ] = (
        TIMESTAMP_UNITS_PER_SECOND
        / sensor_frequency.loc[
            valid_intervals,
            "Median_Timestamp_Interval"
        ]
    )

    sensor_frequency[
        ["Median_Interval_ms", "Approx_Frequency_Hz"]
    ] = (
        sensor_frequency[
            ["Median_Interval_ms", "Approx_Frequency_Hz"]
        ]
        .astype("float64")
        .round(2)
    )

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