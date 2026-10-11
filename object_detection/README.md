# Object Detection Analysis

This module evaluates pretrained OmniHD PointPillars models on the MAN TruckScenes dataset and compares LiDAR and 4D RADAR detection results.

The workflow uses pretrained checkpoints for inference rather than training new models.

## Table of Contents

- [1. Overview](#1-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Reports and Results](#3-reports-and-results)
- [4. Environment and Data Requirements](#4-environment-and-data-requirements)
- [5. Analysis Workflow](#5-analysis-workflow)
- [6. Inference and Evaluation](#6-inference-and-evaluation)
- [7. Notes and Limitations](#7-notes-and-limitations)

## 1. Overview

The analysis compares predictions from pretrained LiDAR and RADAR models transferred from OmniHD to TruckScenes.

The module supports:

- TruckScenes dataset conversion.
- Annotation metadata enrichment.
- LiDAR and RADAR inference.
- Detection evaluation.
- Sensor complementarity analysis.

## 2. Repository Structure

```text
object_detection/
├── src/
│   └── omnihd_adapter/              # Dataset conversion and sensor adapters
├── scripts/
│   └── omnihd/
│       ├── build_truckscenes_dataset.py
│       ├── enrich_truckscenes_metadata.py
│       ├── run_lidar_inference.py
│       ├── run_radar_inference.py
│       ├── evaluate_detection.py
│       └── evaluate_complementarity.py
├── results/
│   └── omnihd/                      # Saved predictions, metrics and report
└── README.md
```

## 3. Reports and Results

Open the [LiDAR and RADAR Evaluation Report](results/omnihd/lidar_radar_evaluation_analysis.ipynb) to review the methods, comparisons and findings.

| Results | Files |
|---|---|
| Mini LiDAR | [Metrics](results/omnihd/lidar_mini_metrics.json), [predictions](results/omnihd/lidar_mini_predictions.pkl) |
| Mini RADAR | [Metrics](results/omnihd/radar_mini_metrics.json), [predictions](results/omnihd/radar_mini_predictions.pkl) |
| Trainval LiDAR | [Metrics](results/omnihd/lidar_trainval_135_metrics.json) |
| Trainval RADAR | [Metrics](results/omnihd/radar_trainval_135_metrics.json) |
| Trainval complementarity | [LiDAR–RADAR complementarity results](results/omnihd/lidar_radar_trainval_135_complementarity.json) |

The trainval results use sensor data packages 01, 03 and 05 rather than the complete trainval dataset.

Saved notebook outputs and JSON metrics can be inspected without rerunning inference. Prediction PKLs are binary files for programmatic use.

Converted annotation PKLs and trainval prediction PKLs are not included in this module and must be generated or obtained separately to reproduce the evaluation.

## 4. Environment and Data Requirements

The complete workflow requires:

- Raw TruckScenes sensor data and corresponding metadata.
- A separate OmniHD-Scenes repository.
- A compatible PyTorch, MMCV and MMDetection3D environment.
- LiDAR and RADAR model configuration files.
- The corresponding pretrained checkpoints.
- A CUDA-capable GPU for inference.

Follow the environment instructions for the OmniHD-Scenes version used with the selected configurations and checkpoints.

The external model resources and raw dataset are not included in this repository.

The raw dataset root should contain:

```text
TRUCKSCENES_DATA_ROOT/
├── samples/
├── sweeps/
└── v1.2-mini/       # or v1.2-trainval
    ├── scene.json
    ├── sample.json
    ├── sample_data.json
    └── ...
```

The raw dataset root and converted dataset root are separate inputs. Inference uses the converted dataset.

## 5. Analysis Workflow

| Stage | Script | Purpose |
|---|---|---|
| 1 | [build_truckscenes_dataset.py](scripts/omnihd/build_truckscenes_dataset.py) | Convert selected TruckScenes samples into the OmniHD-compatible dataset and annotation format. |
| 2 | [enrich_truckscenes_metadata.py](scripts/omnihd/enrich_truckscenes_metadata.py) | Add visibility, motion, weather and area metadata to the annotation PKL. |
| 3 | [run_lidar_inference.py](scripts/omnihd/run_lidar_inference.py) | Generate LiDAR predictions. |
| 3 | [run_radar_inference.py](scripts/omnihd/run_radar_inference.py) | Generate RADAR predictions. |
| 4 | [evaluate_detection.py](scripts/omnihd/evaluate_detection.py) | Evaluate each sensor against the converted ground truth. |
| 5 | [evaluate_complementarity.py](scripts/omnihd/evaluate_complementarity.py) | Compare detections using enriched annotations and predictions from both sensors. |

LiDAR and RADAR inference can run separately after conversion. Complementarity analysis requires matching sample coverage across annotations and both prediction files.

The dataset builder requires one selection method:

- `--all-samples`
- One or more `--sample-token` arguments.
- One or more `--scene-token` arguments.

Select samples with the required sensor files available. Having trainval metadata does not mean all trainval sensor files are present.

For metadata enrichment, `--metadata-root` points directly to the versioned metadata directory containing the JSON tables.

## 6. Inference and Evaluation

Run the following command templates from the repository root in the configured model environment.

Replace every placeholder with the appropriate local path. Absolute paths are recommended to avoid ambiguity.

### 6.1 LiDAR Inference

```text
python object_detection/scripts/omnihd/run_lidar_inference.py --omnihd-root "OMNIHD_ROOT" --config "LIDAR_CONFIG" --checkpoint "LIDAR_CHECKPOINT" --data-root "CONVERTED_DATA_ROOT" --ann-file "ANNOTATION_PKL" --output "LIDAR_PREDICTIONS_PKL"
```

### 6.2 RADAR Inference

```text
python object_detection/scripts/omnihd/run_radar_inference.py --omnihd-root "OMNIHD_ROOT" --config "RADAR_CONFIG" --checkpoint "RADAR_CHECKPOINT" --data-root "CONVERTED_DATA_ROOT" --ann-file "ANNOTATION_PKL" --output "RADAR_PREDICTIONS_PKL"
```

For both inference scripts:

- Relative configuration and checkpoint paths resolve from `--omnihd-root`.
- Other relative paths resolve from the launch directory.
- `--max-samples` limits the number of samples processed.
- `--start-index` sets the starting dataset index.

Use matching sample selections for sensor comparisons.

### 6.3 Detection Evaluation

Run this command separately for LiDAR and RADAR predictions:

```text
python object_detection/scripts/omnihd/evaluate_detection.py --omnihd-root "OMNIHD_ROOT" --ann-file "ANNOTATION_PKL" --predictions "PREDICTIONS_PKL" --output "METRICS_JSON"
```

The default evaluation configuration is `detection_newsc_config_final`.

Optional fixed-operating-point metrics require both `--score-threshold` and `--match-distance`. Use the same settings for both sensors.

### 6.4 Complementarity Evaluation

```text
python object_detection/scripts/omnihd/evaluate_complementarity.py --ann-file "ENRICHED_ANNOTATION_PKL" --lidar-predictions "LIDAR_PREDICTIONS_PKL" --radar-predictions "RADAR_PREDICTIONS_PKL" --output "COMPLEMENTARITY_JSON"
```

The default score threshold is 0.5 and the default XY matching distance is 2 metres.

The output groups ground-truth objects into those detected by both sensors, LiDAR only, RADAR only, or neither sensor.

## 7. Notes and Limitations

- Results describe pretrained OmniHD models transferred to TruckScenes.
- Performance depends on the models, dataset conversion, class mapping, evaluation protocol and sample coverage.
- Trainval results cover packages 01, 03 and 05 rather than the complete trainval dataset.
- Sensor comparisons require matching samples, ground truth and evaluation settings.
- Complementarity analysis measures differences between saved detections; it does not demonstrate a trained sensor-fusion model.
- Paths stored in result JSON files describe the original execution environment and may differ from the current repository layout.
- Viewing saved results does not require the same resources as reproducing model inference.
