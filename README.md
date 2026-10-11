# Boeing RADAR for Autonomous Driving

This CITS3200 project compares RADAR and LiDAR for autonomous-driving perception using the MAN TruckScenes dataset.

The project includes sensor and point-cloud analysis, velocity estimation, pretrained object-detection evaluation, and an interactive dashboard.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Modules and Usage](#modules-and-usage)
  - [Dashboard](#dashboard)
  - [Velocity and Efficiency Analysis](#velocity-and-efficiency-analysis)
  - [Object Detection](#object-detection)
- [Reports and Results](#reports-and-results)
- [Dataset and Saved Data](#dataset-and-saved-data)

## Repository Structure

| Path | Description |
|---|---|
| [dashboard/](dashboard/) | Interactive dashboard for browsing scenes, viewing sensor data and detection overlays, and comparing sensor statistics. |
| [velocity_report/](velocity_report/) | Velocity and sensor-efficiency reports, analysis functions, split notebooks, dependencies, and saved DataFrames. |
| [object_detection/](object_detection/) | Dataset adapters, pretrained model inference and evaluation scripts, and saved detection results. |
| [radar_lidar_pts.ipynb](radar_lidar_pts.ipynb) | Point-cloud and sensor-condition analysis. |
| [recommendation_and_conclusion.ipynb](recommendation_and_conclusion.ipynb) | Overall recommendations and conclusions. |
| [setup.sh](setup.sh) | Dashboard setup and launch script. |
| [requirements.txt](requirements.txt) | Python dependencies for the dashboard. |

## Modules and Usage

### Dashboard

The dashboard provides an interactive interface for exploring TruckScenes data and the project's sensor comparisons.

See the [Dashboard README](dashboard/README.md) for installation, dataset configuration, launch instructions, usage, and troubleshooting.

The dashboard dependencies are listed in the root [requirements.txt](requirements.txt). The [setup script](setup.sh) provides automatic setup and launch in a Linux/macOS Bash environment.

The dashboard requires local TruckScenes data. With the default configuration, the mini dataset should be arranged as follows:

```text
data/
└── man-truckscenes/
    ├── samples/
    ├── sweeps/
    └── v1.2-mini/
        ├── scene.json
        ├── sample.json
        ├── sample_data.json
        └── ...
```

For another dataset location, configure `TRUCKSCENES_DATAROOT`.

### Velocity and Efficiency Analysis

This module compares RADAR and LiDAR velocity estimation and sensor observations, including analysis under different operating conditions.

See the [Velocity README](velocity_report/README.md) for:

- Python environment and dependency installation.
- Dataset paths and version configuration.
- Main report and split notebook execution.
- Saved DataFrames and Git LFS requirements.

The dependencies for this module are listed in [velocity_report/requirements.txt](velocity_report/requirements.txt).

The velocity module uses `TRUCKSCENES_DATA_ROOT` for its dataset path. This differs from the dashboard variable, `TRUCKSCENES_DATAROOT`.

### Object Detection

This module evaluates pretrained OmniHD PointPillars models on TruckScenes and compares LiDAR and 4D RADAR detection results.
See the [Object Detection README](object_detection/README.md) for the module structure, saved results, external requirements, analysis workflow, and inference and evaluation command templates.

Module contents:

- [Dataset adapters](object_detection/src/omnihd_adapter/) – Convert TruckScenes data into the format required by the model workflow.
- [Inference and evaluation scripts](object_detection/scripts/omnihd/) – Dataset preparation, model inference, detection evaluation, and complementarity analysis.
- [Saved results](object_detection/results/omnihd/) – Detection metrics, mini predictions, and the evaluation report.

To review the findings, open the [Object-Detection Evaluation Report](object_detection/results/omnihd/lidar_radar_evaluation_analysis.ipynb).

Running inference requires a separate OmniHD-Scenes repository and its dependencies, compatible model configuration files, pretrained checkpoints, and a CUDA-capable GPU. The external model resources are not included in this repository.

## Reports and Results

| Report | Scope |
|---|---|
| [RADAR and LiDAR Point Analysis](radar_lidar_pts.ipynb) | Point-cloud characteristics and sensor evidence under different conditions. |
| [Velocity and Efficiency Report](velocity_report/velocity_efficiency_report.ipynb) | Velocity estimation and sensor-efficiency analysis using the complete mini dataset. |
| [Condition Velocity Report](velocity_report/condition_velocity.ipynb) | Condition-based analysis using trainval sensor data packages 01, 03 and 05. |
| [Object-Detection Evaluation Report](object_detection/results/omnihd/lidar_radar_evaluation_analysis.ipynb) | Pretrained LiDAR and RADAR detection evaluation and comparison. |
| [Recommendations and Conclusion](recommendation_and_conclusion.ipynb) | Overall findings, limitations, and recommendations. |

Notebooks with saved outputs can be inspected without rerunning the analysis. Reproducing results requires the appropriate environment, dataset files, and any external model resources.

## Dataset and Saved Data

The mini dataset contains 10 scenes. The trainval metadata contains 598 scenes, but the condition analysis uses sensor data packages 01, 03 and 05 rather than the complete trainval dataset.

Raw TruckScenes sensor data must be obtained separately and should not be committed to this repository.

Saved condition-analysis CSV files are tracked using Git LFS. After installing Git LFS, run the following commands from the repository root:

```bash
git lfs install
git lfs pull
```

See the [Saved DataFrames guide](velocity_report/README.md#44-saved-dataframes) for cached-analysis requirements.

The dashboard, velocity analysis, and object-detection inference have different dependency requirements. Follow the instructions for the selected module.

Results should be interpreted within the dataset coverage and methods described in each report. Point-cloud, velocity, and object-detection metrics measure different aspects of sensor performance.

