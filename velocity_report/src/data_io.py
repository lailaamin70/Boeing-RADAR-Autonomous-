"""
Utilities for reading TruckScenes data files.

Provides functions for loading JSON metadata and reading PCD file
headers without loading the point-cloud data.
"""

from pathlib import Path
import json


def load_json(file_path):
    """Load and parse a JSON file."""

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_pcd_header(pcd_file):
    """
    Read PCD header information without loading point-cloud data.
    Extracts fields, data types, dimensions, point count, and data format.
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