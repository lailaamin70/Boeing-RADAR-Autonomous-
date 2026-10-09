# Velocity and Efficiency Analysis

This module contains the TruckScenes RADAR–LiDAR velocity and sensor analysis, including data processing scripts, Jupyter notebooks, and saved DataFrames.

The analysis environment is independent of the dashboard application.

## Table of Contents

- [1. Overview](#1-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Installation Guide](#3-installation-guide)
  - [3.1 Python Requirements](#31-python-requirements)
  - [3.2 Create a Virtual Environment](#32-create-a-virtual-environment)
  - [3.3 Activate the Virtual Environment](#33-activate-the-virtual-environment)
  - [3.4 Install Dependencies](#34-install-dependencies)
  - [3.5 Dataset Configuration](#35-dataset-configuration)
  - [3.6 Jupyter Notebook Setup](#36-jupyter-notebook-setup)
- [4. User Guide](#4-user-guide)
  - [4.1 Main Reports](#41-main-reports)
  - [4.2 Velocity and Efficiency Report](#42-velocity-and-efficiency-report)
  - [4.3 Condition-Based Analysis](#43-condition-based-analysis)
  - [4.4 Saved DataFrames](#44-saved-dataframes)
- [5. Notes and Limitations](#5-notes-and-limitations)

---

## 1. Overview

The velocity analysis compares RADAR and LiDAR velocity estimation and sensor observations using the TruckScenes dataset.

The module includes two main analysis reports:

- `velocity_efficiency_report.ipynb` – Velocity estimation and cross-sensor efficiency analysis.
- `condition_velocity.ipynb` – Condition-based velocity analysis, including comparisons across different operating conditions.

Supporting scripts and notebooks are provided for data processing, development, and reproducing the analysis.

## 2. Repository Structure

```text
velocity_report/
├── src/                            # Analysis functions
│   ├── condition/                  # Condition-based analysis functions
│   └── config.py                   # Dataset configuration
├── notebook/                       # Development and split notebooks
│   └── condition/                  # Split condition analysis workflow
├── df/                             # Saved DataFrames
│   └── condition/
├── condition_velocity.ipynb        # Condition-based analysis report
├── velocity_efficiency_report.ipynb # Velocity and efficiency report
├── analysis_workflow.png           # Condition analysis pipeline
├── requirements.txt
└── README.md
```

Dataset configuration is managed in `src/config.py`.

## 3. Installation Guide

All commands in this section are intended to be run from the repository root unless otherwise specified.

### 3.1 Python Requirements

Use Python 3.11.x.

`truckscenes-devkit==1.2.0` requires Python >= 3.8 and < 3.12.

### 3.2 Create a Virtual Environment

The examples below use `.venv` as the virtual environment name. You may use a different name if preferred.

**Windows**

```powershell
py -3.11 -m venv .venv
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
```

### 3.3 Activate the Virtual Environment

**Windows – PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3.4 Install Dependencies

Install the analysis dependencies using the module's requirements file:

```bash
python -m pip install -r velocity_report/requirements.txt
```

The dashboard application uses a separate environment and dependency configuration.

### 3.5 Dataset Configuration

Dataset paths and versions are managed through `velocity_report/src/config.py`.

By default, the dataset is expected under `velocity_report/data/`.

The required dataset version depends on the report:

| Report | Metadata | Sensor Data |
|---|---|---|
| `velocity_efficiency_report.ipynb` | `v1.2-mini` | Complete mini dataset |
| `condition_velocity.ipynb` | `v1.2-trainval` | Sensor data packages 01, 03 and 05 |

The dataset directory should contain the corresponding metadata folder and the extracted `samples/` and `sweeps/` directories.

The dataset location and version can be configured using environment variables.

**Windows – PowerShell**

```powershell
$env:TRUCKSCENES_DATA_ROOT = "YOUR_DATASET_PATH"
$env:TRUCKSCENES_VERSION = "v1.2-trainval"
```

**macOS / Linux**

```bash
export TRUCKSCENES_DATA_ROOT="/path/to/dataset"
export TRUCKSCENES_VERSION="v1.2-trainval"
```

Replace the dataset path with the local TruckScenes directory containing the sensor data and versioned metadata.

Set `TRUCKSCENES_VERSION` to `v1.2-mini` when running the Velocity and Efficiency Report, or `v1.2-trainval` when running the Condition Report.

The current default version in `src/config.py` is `v1.2-trainval`.

Environment variables must be available to the Python process running the notebooks before the configuration module is imported. Restart the notebook kernel if necessary.

The local `data/` directory should not be committed to GitHub.

### 3.6 Jupyter Notebook Setup

Open the notebooks in VS Code or Jupyter and select the Python kernel from the virtual environment.

For reliable relative path resolution, use the notebook's own directory as the working directory.

- Main reports are located directly under `velocity_report/`.
- Split condition notebooks are located under `velocity_report/notebook/condition/`.

## 4. User Guide

### 4.1 Main Reports

Two main reports are available in the `velocity_report/` directory.

| Notebook | Dataset | Execution |
|---|---|---|
| `velocity_efficiency_report.ipynb` | TruckScenes v1.2-mini (complete) | Run All |
| `condition_velocity.ipynb` | TruckScenes v1.2-trainval (sensor data 01, 03 and 05) | Full or Split Workflow |

Both reports use the shared analysis functions in `src/`, but their current results are based on different dataset configurations.

The Condition Report uses trainval metadata with selected sensor data packages rather than the complete trainval dataset.

### 4.2 Velocity and Efficiency Report

Open `velocity_efficiency_report.ipynb` and select the configured Python kernel.

Ensure that the complete `v1.2-mini` dataset is available and that `TRUCKSCENES_VERSION` is set to `v1.2-mini`.

Use **Run All** to execute the analysis.

This notebook performs the velocity and sensor efficiency analysis using the TruckScenes data and the functions provided in `src/`.

### 4.3 Condition-Based Analysis

The condition-based analysis supports two execution workflows.

#### Full Workflow

`condition_velocity.ipynb` provides the integrated condition analysis in a single notebook.

It supports two data-processing modes using the following configuration:

```python
USE_CACHED_DATA = True
```

- `True` – Load previously saved DataFrames from `df/condition/`.
- `False` – Regenerate the required analysis data from the TruckScenes dataset.

Using cached data avoids repeating computationally expensive processing stages.

After selecting the mode, use **Run All** to execute the notebook.

When using cached data, ensure that the required saved DataFrames are available and compatible with the analysis.

When regenerating data, ensure that the appropriate TruckScenes metadata and sensor files are available.

#### Split Workflow

For large datasets or systems with limited memory, the condition analysis is divided into smaller notebooks under:

```text
notebook/condition/
├── 01_dataset_preparation.ipynb
├── 02_radar_velocity.ipynb
├── 03_lidar_velocity.ipynb
├── 04_comparison_conditions.ipynb
└── 05_radar_doppler.ipynb
```

These notebooks divide the processing into separate stages and save reusable DataFrames to `df/condition/`.

Run the required stages in dependency order, ensuring that the necessary inputs from previous stages are available.

Individual stages can be rerun without restarting the entire workflow when their required intermediate DataFrames have already been generated.

The saved DataFrames can also be loaded directly by the Full Workflow using `USE_CACHED_DATA = True`.

The split notebooks use the following import path setup when executed from their own directory:

```python
import sys
sys.path.append("../..")
```

### 4.4 Saved DataFrames

The `df/` directory stores processed DataFrames generated during the analysis.

```text
df/
└── condition/
    ├── available_sample_data.csv
    ├── vehicle_reference_df.csv
    ├── radar_velocity_df.csv
    ├── lidar_velocity_df.csv
    ├── comparison_df.csv
    ├── condition_df.csv
    └── ...
```

These tables serve two purposes:

- Supporting the Split Workflow by providing inputs for subsequent stages.
- Allowing the Full Workflow to reuse saved data without repeating computationally expensive processing.

The saved DataFrames are processed analysis data, not the original TruckScenes dataset.

## 5. Notes and Limitations

- The raw TruckScenes dataset is not included in this module.
- Full data regeneration requires access to the relevant TruckScenes sensor files and metadata.
- The two reports use different dataset configurations: the Velocity and Efficiency Report uses the complete `v1.2-mini` dataset, while the Condition Report currently uses `v1.2-trainval` metadata with sensor data packages 01, 03 and 05.
- Saved DataFrames depend on the dataset version and subset used during processing and should not be assumed to represent the complete trainval dataset.
- All analysis workflows are designed to be reproducible. However, the interpretations and conclusions presented in the reports are based on the datasets used during the analysis and may need to be revised when different datasets or subsets are used.
- The dashboard application has its own setup and dependencies.
