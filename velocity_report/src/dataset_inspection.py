"""
Utilities for inspecting TruckScenes sensor files and metadata.

Includes sensor channel inspection, PCD header analysis,
point-count statistics, and metadata structure checks.
"""

from pathlib import Path

import pandas as pd

from .config import SAMPLES_ROOT, METADATA_ROOT
from .data_io import load_json, read_pcd_header


def get_sensor_structure(samples_root=None):
    """List RADAR and LiDAR sensor channels and count their PCD files."""

    if samples_root is None:
        samples_root = SAMPLES_ROOT

    samples_root = Path(samples_root)

    if not samples_root.exists():
        raise FileNotFoundError(
            f"Samples directory not found: {samples_root}"
        )

    if not samples_root.is_dir():
        raise NotADirectoryError(
            f"Samples path is not a directory: {samples_root}"
        )

    records = []

    for sensor_folder in sorted(samples_root.iterdir()):

        if not sensor_folder.is_dir():
            continue

        sensor_name = sensor_folder.name

        if sensor_name.startswith("LIDAR"):
            modality = "LiDAR"

        elif sensor_name.startswith("RADAR"):
            modality = "RADAR"

        else:
            continue

        pcd_files = list(sensor_folder.glob("*.pcd"))

        records.append({
            "Modality": modality,
            "Sensor": sensor_name,
            "Number_of_PCD_Files": len(pcd_files),
        })

    if not records:
        raise ValueError(
            f"No RADAR or LiDAR sensor folders were found in: {samples_root}"
        )

    return pd.DataFrame(records)


def inspect_pcd_files(samples_root=None):
    """Read RADAR and LiDAR PCD headers to inspect file structure and point counts."""

    if samples_root is None:
        samples_root = SAMPLES_ROOT

    samples_root = Path(samples_root)

    if not samples_root.exists():
        raise FileNotFoundError(
            f"Samples directory not found: {samples_root}"
        )

    if not samples_root.is_dir():
        raise NotADirectoryError(
            f"Samples path is not a directory: {samples_root}"
        )

    records = []

    for sensor_folder in sorted(samples_root.iterdir()):

        if not sensor_folder.is_dir():
            continue

        sensor_name = sensor_folder.name

        if sensor_name.startswith("LIDAR"):
            modality = "LiDAR"

        elif sensor_name.startswith("RADAR"):
            modality = "RADAR"

        else:
            continue

        pcd_files = sorted(sensor_folder.glob("*.pcd"))

        for pcd_file in pcd_files:

            try:
                header = read_pcd_header(pcd_file)

            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"Failed to read PCD header: {pcd_file}"
                ) from error

            records.append({
                "Modality": modality,
                "Sensor": sensor_name,
                "Filename": pcd_file.name,
                "Fields": header.get("FIELDS"),
                "Points": header.get("POINTS"),
                "Data_Format": header.get("DATA"),
            })

    if not records:
        raise ValueError(
            f"No RADAR or LiDAR PCD files were found in: {samples_root}"
        )

    return pd.DataFrame(records)


def get_field_consistency(pcd_df):
    """Summarise PCD field structures and data formats for each sensor."""

    required_columns = {
        "Modality",
        "Sensor",
        "Filename",
        "Fields",
        "Data_Format",
    }

    missing_columns = required_columns - set(pcd_df.columns)

    if missing_columns:
        raise ValueError(
            "pcd_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if pcd_df.empty:
        raise ValueError(
            "pcd_df is empty. No PCD files are available for inspection."
        )

    summary = (
        pcd_df
        .groupby(["Modality", "Sensor"])
        .agg(
            Number_of_Files=("Filename", "size"),

            Unique_Field_Structures=(
                "Fields",
                lambda x: x.nunique(dropna=True)
            ),

            Missing_Field_Headers=(
                "Fields",
                lambda x: x.isna().sum()
            ),

            Unique_Data_Formats=(
                "Data_Format",
                lambda x: x.nunique(dropna=True)
            ),

            Missing_Data_Formats=(
                "Data_Format",
                lambda x: x.isna().sum()
            ),
        )
    )

    return summary


def get_point_summary(pcd_df):
    """Calculate point-count statistics by sensor type and channel."""

    required_columns = {
        "Modality",
        "Sensor",
        "Filename",
        "Points",
    }

    missing_columns = required_columns - set(pcd_df.columns)

    if missing_columns:
        raise ValueError(
            "pcd_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if pcd_df.empty:
        raise ValueError(
            "pcd_df is empty. No PCD files are available for analysis."
        )

    # Ensure point counts are numeric. Invalid or missing values become NaN.
    data = pcd_df.copy()

    data["Points"] = pd.to_numeric(
        data["Points"],
        errors="coerce"
    )

    modality_summary = (
        data
        .groupby("Modality")
        .agg(
            Number_of_PCD_Files=("Filename", "size"),
            Missing_Point_Counts=("Points", lambda x: x.isna().sum()),
            Mean_Points_Per_PCD=("Points", "mean"),
            Median_Points_Per_PCD=("Points", "median"),
            Min_Points=("Points", "min"),
            Max_Points=("Points", "max"),
        )
        .round(2)
    )

    sensor_summary = (
        data
        .groupby(["Modality", "Sensor"])
        .agg(
            Number_of_PCD_Files=("Filename", "size"),
            Missing_Point_Counts=("Points", lambda x: x.isna().sum()),
            Mean_Points_Per_PCD=("Points", "mean"),
            Median_Points_Per_PCD=("Points", "median"),
            Min_Points=("Points", "min"),
            Max_Points=("Points", "max"),
        )
        .round(2)
    )

    return modality_summary, sensor_summary


def inspect_metadata_structure(metadata_root=None):
    """
    Inspect JSON metadata structure and field availability.
    Checks field consistency across records in each metadata file.
    """

    if metadata_root is None:
        metadata_root = METADATA_ROOT

    metadata_root = Path(metadata_root)

    if not metadata_root.exists():
        raise FileNotFoundError(
            f"Metadata directory not found: {metadata_root}"
        )

    if not metadata_root.is_dir():
        raise NotADirectoryError(
            f"Metadata path is not a directory: {metadata_root}"
        )

    json_files = sorted(metadata_root.glob("*.json"))

    if not json_files:
        raise ValueError(
            f"No JSON metadata files found in: {metadata_root}"
        )

    metadata_results = []
    field_results = []

    for file_path in json_files:

        try:
            data = load_json(file_path)

        except (OSError, ValueError) as error:
            metadata_results.append({
                "Metadata File": file_path.name,
                "JSON Type": "Unreadable",
                "Number of Records": 0,
                "Number of Fields": 0,
                "Fields": "",
                "Fields Present in All Records": "",
                "Consistent Structure": False,
                "Error": str(error),
            })
            continue

        # Convert JSON content into a list of records
        if isinstance(data, list):
            records = data
            json_type = "list"

        elif isinstance(data, dict):
            records = [data]
            json_type = "dict"

        else:
            metadata_results.append({
                "Metadata File": file_path.name,
                "JSON Type": type(data).__name__,
                "Number of Records": 0,
                "Number of Fields": 0,
                "Fields": "",
                "Fields Present in All Records": "",
                "Consistent Structure": False,
                "Error": "Unsupported JSON structure",
            })
            continue

        if not records:
            metadata_results.append({
                "Metadata File": file_path.name,
                "JSON Type": json_type,
                "Number of Records": 0,
                "Number of Fields": 0,
                "Fields": "",
                "Fields Present in All Records": "",
                "Consistent Structure": True,
                "Error": "",
            })
            continue

        dict_records = [
            record
            for record in records
            if isinstance(record, dict)
        ]

        if not dict_records:
            metadata_results.append({
                "Metadata File": file_path.name,
                "JSON Type": json_type,
                "Number of Records": len(records),
                "Number of Fields": 0,
                "Fields": "",
                "Fields Present in All Records": "",
                "Consistent Structure": False,
                "Error": "Records are not dictionaries",
            })
            continue

        all_fields = sorted(
            set().union(
                *(record.keys() for record in dict_records)
            )
        )

        fields_present_in_all = [
            field
            for field in all_fields
            if all(
                field in record
                for record in dict_records
            )
        ]

        consistent_structure = (
            len(fields_present_in_all)
            == len(all_fields)
        )

        metadata_results.append({
            "Metadata File": file_path.name,
            "JSON Type": json_type,
            "Number of Records": len(records),
            "Number of Fields": len(all_fields),
            "Fields": ", ".join(all_fields),
            "Fields Present in All Records": ", ".join(
                fields_present_in_all
            ),
            "Consistent Structure": consistent_structure,
            "Error": "",
        })

        for field in all_fields:

            present_count = sum(
                field in record
                for record in dict_records
            )

            field_results.append({
                "Metadata File": file_path.name,
                "Field": field,
                "Present in Records": present_count,
                "Total Records": len(dict_records),
                "Presence Ratio": (
                    present_count / len(dict_records)
                ),
            })

    metadata_summary = pd.DataFrame(metadata_results)

    field_summary = pd.DataFrame(field_results)

    if not field_summary.empty:
        field_summary["Presence Ratio"] = (
            field_summary["Presence Ratio"]
            .round(4)
        )

    return metadata_summary, field_summary
