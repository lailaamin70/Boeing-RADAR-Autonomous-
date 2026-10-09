from pathlib import Path

import pandas as pd

from .config import METADATA_ROOT, SENSOR_ROOT
from .data_io import load_json


def prepare_available_sample_data(
    metadata_root=None,
    data_root=None,
):
    """
    Filter sample_data metadata to files available in the local dataset.
    """

    if metadata_root is None:
        metadata_root = METADATA_ROOT

    if data_root is None:
        data_root = SENSOR_ROOT

    metadata_root = Path(metadata_root)
    data_root = Path(data_root)

    sample_data = pd.DataFrame(
        load_json(
            metadata_root / "sample_data.json"
        )
    )

    required_columns = {
        "token",
        "sample_token",
        "filename",
    }

    missing_columns = (
        required_columns
        - set(sample_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if sample_data["token"].duplicated().any():
        raise ValueError(
            "Duplicate sample_data tokens detected."
        )

    local_files = set()

    for folder_name in ("samples", "sweeps"):
        folder = data_root / folder_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        for file_path in folder.rglob("*"):
            if file_path.is_file():
                local_files.add(
                    file_path
                    .relative_to(data_root)
                    .as_posix()
                )

    sample_data["file_available"] = (
        sample_data["filename"]
        .str.replace("\\", "/", regex=False)
        .isin(local_files)
    )

    available_sample_data = (
        sample_data[
            sample_data["file_available"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    metadata_files = set(
        sample_data["filename"]
        .str.replace("\\", "/", regex=False)
    )

    extra_files = local_files - metadata_files

    summary = pd.Series(
        {
            "Metadata records":
                len(sample_data),

            "Available local files":
                len(available_sample_data),

            "Unavailable locally":
                int(
                    (~sample_data["file_available"]).sum()
                ),

            "Extra / unreferenced local files":
                len(extra_files),
        },
        name="Dataset Filtering Summary",
    )

    return (
        available_sample_data,
        summary,
        extra_files,
    )