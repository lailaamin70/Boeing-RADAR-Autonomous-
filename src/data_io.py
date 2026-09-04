"""
Basic data-loading utilities for the TruckScenes analysis project.
This module contains reusable functions for reading TruckScenes files.

The functions in this file should only handle data input and basic parsing.
Dataset analysis, velocity calculation, sensor comparison, and other research logic should be implemented in separate modules.

These functions are designed to work with both the mini dataset and the complete TruckScenes dataset because file locations are passed to the functions rather than hard-coded here.
"""

from pathlib import Path
import json


# JSON loading
def load_json(file_path):
    """
    Load a JSON file and return its contents.

    Parameters
    ----------
    file_path : str or Path
        Path to the JSON file.

    Returns
    -------
    dict or list
        Parsed JSON content.
    """

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# PCD header loading
def read_pcd_header(pcd_file):
    """
    Read the header of a PCD file without loading the point-cloud data.

    Parameters
    ----------
    pcd_file : str or Path
        Path to the PCD file.

    Returns
    -------
    dict
        PCD header information including fields, size, type,
        dimensions, point count, and data format.
    """

    pcd_file = Path(pcd_file)

    header = {}

    with open(pcd_file, "rb") as f:

        while True:

            line = f.readline()

            if not line:
                break

            text = line.decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if text.startswith("FIELDS"):
                header["FIELDS"] = text.replace("FIELDS ", "")

            elif text.startswith("SIZE"):
                header["SIZE"] = text.replace("SIZE ", "")

            elif text.startswith("TYPE"):
                header["TYPE"] = text.replace("TYPE ", "")

            elif text.startswith("WIDTH"):
                header["WIDTH"] = int(text.split()[1])

            elif text.startswith("HEIGHT"):
                header["HEIGHT"] = int(text.split()[1])

            elif text.startswith("POINTS"):
                header["POINTS"] = int(text.split()[1])

            elif text.startswith("DATA"):
                header["DATA"] = text.replace("DATA ", "")
                break

    return header