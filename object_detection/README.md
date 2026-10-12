
# Object Detection Analysis

This module contains the LiDAR and 4D RADAR object-detection analysis conducted using the MAN TruckScenes dataset and OmniHD-pretrained PointPillars models.

The analysis compares the detection performance of both modalities under identical evaluation settings and investigates their complementary contributions across different object classes and operating conditions.

The workflow uses pretrained models for inference rather than training new models. Both models are transferred from OmniHD to TruckScenes without retraining or fine-tuning.

The object-detection environment is separate from the dashboard and velocity-analysis environments.

## Table of Contents

- [1. Overview](#1-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Installation Guide](#3-installation-guide)
  - [3.1 Python and GPU Requirements](#31-python-and-gpu-requirements)
  - [3.2 Environment Setup](#32-environment-setup)
  - [3.3 TruckScenes Dataset Configuration](#33-truckscenes-dataset-configuration)
  - [3.4 OmniHD Models and Checkpoints](#34-omnihd-models-and-checkpoints)
  - [3.5 Jupyter Notebook Setup](#35-jupyter-notebook-setup)
- [4. User Guide](#4-user-guide)
  - [4.1 Main Analysis Report](#41-main-analysis-report)
  - [4.2 Saved Results](#42-saved-results)
  - [4.3 Understanding the Evaluation](#43-understanding-the-evaluation)
- [5. Analysis Workflow](#5-analysis-workflow)
  - [5.1 Dataset Conversion](#51-dataset-conversion)
  - [5.2 Metadata Enrichment](#52-metadata-enrichment)
  - [5.3 LiDAR and RADAR Inference](#53-lidar-and-radar-inference)
  - [5.4 Detection Evaluation](#54-detection-evaluation)
  - [5.5 Complementarity Evaluation](#55-complementarity-evaluation)
- [6. Reproducing the Evaluation](#6-reproducing-the-evaluation)
- [7. Notes and Limitations](#7-notes-and-limitations)

---

## 1. Overview

The object-detection analysis investigates how LiDAR and 4D RADAR perform when two OmniHD-pretrained PointPillars models are evaluated on the MAN TruckScenes dataset.

The two models use the same PointPillars architecture family but separate modality-specific configurations and pretrained checkpoints.

The main objectives are to:

1. Evaluate the standalone object-detection performance of LiDAR and 4D RADAR.
2. Compare detection accuracy across object classes.
3. Investigate performance under different operating conditions, including distance, object size, visibility, motion, weather and driving environment.
4. Identify objects successfully detected by RADAR but missed by LiDAR.
5. Assess the potential complementary role of RADAR in autonomous-driving perception.

The analysis evaluates four mapped object classes:

| Class | Description |
|---|---|
| `car` | Passenger cars |
| `pedestrian` | Pedestrians |
| `rider` | Motorcycles and bicycles |
| `large_vehicle` | Trucks, buses, trailers and other mapped large vehicles |

The class mapping converts selected TruckScenes object categories into the four classes used by the OmniHD models.

### Dataset Coverage

Two dataset configurations are supported:

| Dataset | Purpose |
|---|---|
| TruckScenes `v1.2-mini` | Initial pipeline development, testing and validation |
| TruckScenes `v1.2-trainval` | Main object-detection comparison using available sensor data packages 01, 03 and 05 |

The main trainval evaluation contains:

- **252 scenes**
- **10,063 evaluated samples**
- **87,781 evaluated ground-truth objects**

These results are based on the locally available trainval sensor data, not the complete TruckScenes trainval dataset.

---

## 2. Repository Structure

```text
object_detection/
├── src/
│   └── omnihd_adapter/
│       ├── build_dataset.py              # Dataset construction
│       ├── annotation_adapter.py         # Ground-truth annotation conversion
│       ├── class_mapping.py              # TruckScenes-to-OmniHD class mapping
│       ├── lidar_adapter.py              # LiDAR data adaptation
│       ├── radar_adapter.py              # RADAR data adaptation
│       └── transforms.py                 # Coordinate transformations
│
├── scripts/
│   └── omnihd/
│       ├── build_truckscenes_dataset.py   # Convert TruckScenes dataset
│       ├── enrich_truckscenes_metadata.py # Add condition metadata
│       ├── run_lidar_inference.py         # LiDAR model inference
│       ├── run_radar_inference.py         # RADAR model inference
│       ├── evaluate_detection.py         # Standalone detection metrics
│       └── evaluate_complementarity.py   # Cross-sensor complementarity
│
├── results/
│   └── omnihd/
│       ├── lidar_radar_evaluation_analysis.ipynb
│       ├── lidar_mini_metrics.json
│       ├── radar_mini_metrics.json
│       ├── lidar_mini_predictions.pkl
│       ├── radar_mini_predictions.pkl
│       ├── lidar_trainval_135_metrics.json
│       ├── radar_trainval_135_metrics.json
│       └── lidar_radar_trainval_135_complementarity.json
│
└── README.md
```

The `src/omnihd_adapter/` directory contains the conversion logic required to adapt TruckScenes data to the input format expected by the OmniHD models.

The `scripts/omnihd/` directory contains the executable workflow for conversion, inference and evaluation.

The `results/omnihd/` directory contains the main evaluation notebook and saved metrics.

Large intermediate datasets, converted annotation files and full trainval prediction files may be stored outside Git rather than committed to the repository.

---

## 3. Installation Guide

The complete object-detection workflow requires two environments:

1. A local Python environment for dataset preparation, evaluation and result analysis.
2. A compatible OmniHD model environment with CUDA support for pretrained-model inference.

The saved evaluation notebook can be inspected without recreating the full inference environment.

### 3.1 Python and GPU Requirements

The TruckScenes dataset tools support Python 3.11 through the project's existing environment.

For pretrained PointPillars inference, the following components are required:

- Python compatible with the selected OmniHD implementation.
- PyTorch with CUDA support.
- MMCV.
- MMDetection.
- MMDetection3D.
- OmniHD-Scenes source code.
- Pretrained LiDAR and 4D RADAR PointPillars checkpoints.
- NVIDIA CUDA-capable GPU.

The exact PyTorch, CUDA, MMCV and MMDetection3D versions must be compatible with the OmniHD-Scenes implementation and model configurations.

Do not assume that the latest versions of these libraries are compatible with the pretrained checkpoints.

### 3.2 Environment Setup

For local data processing and analysis, create a Python virtual environment from the repository root.

**Windows**

```powershell
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

For TruckScenes data access, install the compatible TruckScenes devkit:

```bash
python -m pip install truckscenes-devkit==1.2.0
```

Install the remaining dependencies required by the analysis scripts in the selected environment.

For pretrained-model inference, follow the installation and environment requirements of the external OmniHD-Scenes repository.

The complete inference environment may need to be configured separately on a GPU workstation or cloud instance.

### 3.3 TruckScenes Dataset Configuration

The workflow requires TruckScenes metadata and corresponding raw sensor files.

The dataset should follow a structure similar to:

```text
TRUCKSCENES_DATA_ROOT/
├── samples/
├── sweeps/
├── v1.2-mini/
│   ├── scene.json
│   ├── sample.json
│   ├── sample_data.json
│   └── ...
└── v1.2-trainval/
    ├── scene.json
    ├── sample.json
    ├── sample_data.json
    └── ...
```

For mini-dataset evaluation, use `v1.2-mini` metadata and the complete mini sensor dataset.

For the main trainval evaluation, use `v1.2-trainval` metadata and the available sensor data packages 01, 03 and 05.

**Important:** Having the complete trainval metadata does not mean that all sensor files are available.

The dataset-conversion stage must select samples whose required LiDAR and RADAR files are present locally.

Raw TruckScenes data is not included in this repository.

### 3.4 OmniHD Models and Checkpoints

The pretrained models are obtained from the external OmniHD-Scenes project.

Two modality-specific models are required:

| Model | Configuration | Purpose |
|---|---|---|
| LiDAR PointPillars | `pointpillars_LiDAR.py` | LiDAR object detection |
| 4D RADAR PointPillars | `pointpillars_4DRadar.py` | RADAR object detection |

Each model requires its corresponding pretrained checkpoint.

The two models share the PointPillars architecture family but use different input modalities and trained weights.

Both checkpoints were pretrained using OmniHD rather than TruckScenes.

No additional training or fine-tuning was performed before the reported TruckScenes evaluation.

The model configurations and checkpoints are external resources and are not included in this repository.

### 3.5 Jupyter Notebook Setup

Open the main notebook in VS Code or Jupyter:

[LiDAR and RADAR Evaluation Analysis](results/omnihd/lidar_radar_evaluation_analysis.ipynb)

Select a Python kernel containing the required notebook-analysis dependencies.

The notebook loads saved JSON metrics rather than rerunning the pretrained models.

For reliable relative-path resolution, use the notebook's directory as the working directory:

```text
object_detection/results/omnihd/
```

The notebook's saved outputs may also be inspected directly without rerunning its cells.

---

## 4. User Guide

### 4.1 Main Analysis Report

The main report is located at:

`results/omnihd/lidar_radar_evaluation_analysis.ipynb`

This notebook provides the final LiDAR and RADAR detection comparison using the selected TruckScenes trainval subset.

It covers the following analyses:

| Section | Analysis | Purpose |
|---|---|---|
| 1 | Overall Detection Performance | Compare overall detection accuracy |
| 2 | Performance by Class | Compare detection across object categories |
| 3 | Performance by Object Distance | Examine how detection changes with distance |
| 4 | Performance by Object Size | Compare performance across object sizes |
| 5 | Performance by Visibility | Examine performance under different visibility levels |
| 6 | Performance by Motion | Compare moving and stationary objects |
| 7 | Performance by Weather | Examine weather-related differences |
| 8 | Performance by Area | Compare different driving environments |
| 9 | LiDAR–RADAR Complementarity | Identify objects detected by one or both sensors |
| 10 | Key Findings and Recommendations | Summarise findings and interpretation |

To inspect the report:

1. Open `lidar_radar_evaluation_analysis.ipynb`.
2. Select the configured Python kernel if execution is required.
3. Ensure that the relevant JSON result files are available in `results/omnihd/`.
4. Run the notebook cells, or inspect their saved outputs.

The report includes tables, charts and interpretation of the results.

Reproducing the report from existing JSON metrics does not require rerunning the PointPillars inference models.

### 4.2 Saved Results

The `results/omnihd/` directory stores evaluation metrics and selected prediction outputs.

| Dataset | Modality | Result files |
|---|---|---|
| Mini | LiDAR | `lidar_mini_metrics.json`, `lidar_mini_predictions.pkl` |
| Mini | RADAR | `radar_mini_metrics.json`, `radar_mini_predictions.pkl` |
| Trainval 01/03/05 | LiDAR | `lidar_trainval_135_metrics.json` |
| Trainval 01/03/05 | RADAR | `radar_trainval_135_metrics.json` |
| Trainval 01/03/05 | Both | `lidar_radar_trainval_135_complementarity.json` |

**JSON metrics**

The JSON files contain the evaluation outputs used to generate the report, including overall results and condition-based summaries.

**Prediction PKLs**

Prediction PKLs contain saved model predictions for programmatic evaluation and comparison.

Mini prediction files are included for reference and development.

The full trainval prediction files are not included because of their size. They must be generated or obtained separately if the evaluation needs to be reproduced.

**Annotation PKLs**

The converted annotation PKLs contain TruckScenes ground truth adapted to the OmniHD evaluation format.

Enriched annotation files additionally contain metadata used for condition-based analysis.

These intermediate annotation files are not included in the saved results and must be regenerated or obtained separately for full reproducibility.

### 4.3 Understanding the Evaluation

The object-detection analysis uses two types of evaluation metrics.

#### Official Detection Metrics

The official OmniHD/NewScenes evaluation reports Average Precision using centre-distance thresholds of:

- 1 metre
- 2 metres
- 3 metres
- 4 metres

Mean Average Precision (mAP) summarises detection performance across the evaluated object classes.

The report also includes:

| Metric | Description | Direction |
|---|---|---|
| Precision | Proportion of predicted objects that are correctly matched | Higher is better |
| Recall | Proportion of ground-truth objects successfully detected | Higher is better |
| F1-score | Harmonic mean of precision and recall | Higher is better |
| ATE | Average Translation Error, measuring object-centre localisation error | Lower is better |
| ASE | Average Scale Error, measuring size or scale error | Lower is better |
| AOE | Average Orientation Error, measuring bounding-box orientation error | Lower is better |

ATE, ASE and AOE are calculated from successfully matched detections.

#### Fixed-Operating-Point Evaluation

For condition-based and complementarity analysis, the main report uses:

- **Confidence threshold:** score ≥ 0.5
- **XY centre-distance matching:** less than 2 metres

The same evaluation settings are applied to both modalities.

A prediction must satisfy the relevant class and matching criteria to count as a true positive.

The fixed-operating-point metrics should not be confused with official AP, which evaluates predictions across confidence thresholds.

#### Complementarity Analysis

The complementarity evaluation groups ground-truth objects into four outcomes:

| Outcome | Meaning |
|---|---|
| Both | Successfully detected by both LiDAR and RADAR |
| LiDAR only | Detected by LiDAR but missed by RADAR |
| RADAR only | Detected by RADAR but missed by LiDAR |
| Neither | Missed by both modalities |

These outcomes support the analysis of potential combined recall and the proportion of LiDAR misses recovered by RADAR.

Complementarity analysis does not represent an implemented sensor-fusion model.

---

## 5. Analysis Workflow

The complete workflow consists of five main stages.

| Stage | Script | Purpose |
|---|---|---|
| 1 | `build_truckscenes_dataset.py` | Convert TruckScenes samples into OmniHD-compatible data |
| 2 | `enrich_truckscenes_metadata.py` | Add condition metadata to annotations |
| 3 | `run_lidar_inference.py`, `run_radar_inference.py` | Generate pretrained-model predictions |
| 4 | `evaluate_detection.py` | Calculate standalone detection metrics |
| 5 | `evaluate_complementarity.py` | Evaluate the complementary detections of LiDAR and RADAR |

All scripts are located under:

```text
object_detection/scripts/omnihd/
```

The evaluation results can then be inspected and visualised using the main report notebook.

### 5.1 Dataset Conversion

**Script:** `build_truckscenes_dataset.py`

This stage prepares TruckScenes samples for the pretrained OmniHD PointPillars models.

The conversion includes:

- Selecting samples with available sensor data.
- Loading the corresponding TruckScenes metadata.
- Adapting LiDAR and RADAR point-cloud inputs.
- Transforming sensor coordinates into the expected reference coordinate system.
- Mapping TruckScenes object classes to the OmniHD target classes.
- Converting ground-truth annotations into the required format.
- Generating the converted dataset and annotation PKL.

The conversion implementation is located in:

```text
object_detection/src/omnihd_adapter/
```

The dataset builder supports the following sample-selection options:

| Option | Purpose |
|---|---|
| `--all-samples` | Process all selected samples |
| `--sample-token` | Process specified sample tokens |
| `--scene-token` | Process samples from specified scenes |

Use one sample-selection method per build.

For the main analysis, only samples with the necessary locally available sensor data should be selected.

### 5.2 Metadata Enrichment

**Script:** `enrich_truckscenes_metadata.py`

This stage enriches the converted ground-truth annotations with additional TruckScenes metadata.

The added information supports analysis by:

- Visibility.
- Motion state.
- Weather.
- Driving area.

The `--metadata-root` argument should point directly to the versioned TruckScenes metadata directory containing the required JSON files.

The enriched annotation PKL is used during condition-based and complementarity evaluation.

### 5.3 LiDAR and RADAR Inference

**Scripts:**

- `run_lidar_inference.py`
- `run_radar_inference.py`

The two scripts run their respective pretrained PointPillars models on the converted TruckScenes dataset.

Inference requires:

- OmniHD-Scenes source code.
- The matching model configuration.
- The corresponding pretrained checkpoint.
- Converted TruckScenes sensor data.
- Converted annotation PKL.
- Compatible CUDA inference environment.

LiDAR and RADAR inference can be executed independently after dataset conversion.

To ensure a valid comparison, both models should be evaluated on the same sample selection.

The output of each inference script is a prediction PKL containing model detections.

### 5.4 Detection Evaluation

**Script:** `evaluate_detection.py`

This stage compares each model's saved predictions against the converted TruckScenes ground truth.

The script produces standalone metrics for LiDAR or RADAR.

The default evaluation configuration is:

`detection_newsc_config_final`

Optional fixed-operating-point metrics require the following parameters:

- `--score-threshold`
- `--match-distance`

Use identical thresholds and sample coverage when comparing LiDAR and RADAR.

The evaluation outputs are saved as JSON files.

### 5.5 Complementarity Evaluation

**Script:** `evaluate_complementarity.py`

This stage compares saved LiDAR and RADAR predictions against the same enriched ground-truth annotations.

The default matching configuration is:

- Confidence score threshold: 0.5.
- XY centre-distance threshold: 2 metres.

The evaluation identifies:

- Objects detected by both sensors.
- Objects detected only by LiDAR.
- Objects detected only by RADAR.
- Objects missed by both sensors.

It also supports complementarity analysis by object class and operating conditions.

Both prediction files must correspond to the same evaluated sample set.

---

## 6. Reproducing the Evaluation

All commands in this section should be run from the repository root in the configured OmniHD-compatible environment.

Replace placeholder paths with the appropriate paths on the local machine or GPU instance.

Absolute paths are recommended where possible.

### 6.1 LiDAR Inference

```bash
python object_detection/scripts/omnihd/run_lidar_inference.py \
  --omnihd-root "OMNIHD_ROOT" \
  --config "LIDAR_CONFIG" \
  --checkpoint "LIDAR_CHECKPOINT" \
  --data-root "CONVERTED_DATA_ROOT" \
  --ann-file "ANNOTATION_PKL" \
  --output "LIDAR_PREDICTIONS_PKL"
```

### 6.2 RADAR Inference

```bash
python object_detection/scripts/omnihd/run_radar_inference.py \
  --omnihd-root "OMNIHD_ROOT" \
  --config "RADAR_CONFIG" \
  --checkpoint "RADAR_CHECKPOINT" \
  --data-root "CONVERTED_DATA_ROOT" \
  --ann-file "ANNOTATION_PKL" \
  --output "RADAR_PREDICTIONS_PKL"
```

For both inference scripts:

- Relative configuration and checkpoint paths resolve from `--omnihd-root`.
- Other relative paths resolve from the command launch directory.
- `--max-samples` can limit the number of processed samples.
- `--start-index` can specify the starting dataset index.

The optional sample-index arguments can be useful for processing subsets or dividing inference work into batches.

When combining outputs from multiple runs, ensure that predictions remain aligned with the correct sample tokens.

### 6.3 Standalone Detection Evaluation

Run the evaluator separately for LiDAR and RADAR.

**LiDAR**

```bash
python object_detection/scripts/omnihd/evaluate_detection.py \
  --omnihd-root "OMNIHD_ROOT" \
  --ann-file "ANNOTATION_PKL" \
  --predictions "LIDAR_PREDICTIONS_PKL" \
  --output "LIDAR_METRICS_JSON"
```

**RADAR**

```bash
python object_detection/scripts/omnihd/evaluate_detection.py \
  --omnihd-root "OMNIHD_ROOT" \
  --ann-file "ANNOTATION_PKL" \
  --predictions "RADAR_PREDICTIONS_PKL" \
  --output "RADAR_METRICS_JSON"
```

For the fixed-operating-point comparison, configure the same score threshold and matching distance for both modalities.

### 6.4 Complementarity Evaluation

```bash
python object_detection/scripts/omnihd/evaluate_complementarity.py \
  --ann-file "ENRICHED_ANNOTATION_PKL" \
  --lidar-predictions "LIDAR_PREDICTIONS_PKL" \
  --radar-predictions "RADAR_PREDICTIONS_PKL" \
  --output "COMPLEMENTARITY_JSON"
```

This script evaluates the detection overlap between LiDAR and RADAR.

It requires matching annotation and prediction coverage across both modalities.

### 6.5 Reviewing the Results

After completing the evaluation:

1. Place the generated metric JSON files in the expected results directory, or configure the notebook to read their actual locations.
2. Open `object_detection/results/omnihd/lidar_radar_evaluation_analysis.ipynb`.
3. Select the relevant Python kernel.
4. Run the analysis cells to generate comparison tables, charts and findings.

The notebook can also be inspected using its saved results without rerunning the complete pipeline.

---

## 7. Notes and Limitations

### Dataset Coverage

- The main object-detection analysis uses selected TruckScenes `v1.2-trainval` sensor packages 01, 03 and 05.
- The reported results do not represent the complete trainval dataset.
- The complete trainval metadata may contain samples whose corresponding sensor files are not locally available.
- Sample selection must be consistent across LiDAR inference, RADAR inference and evaluation.

### Pretrained Model Transfer

- Both PointPillars models were pretrained on OmniHD.
- No TruckScenes-specific retraining or fine-tuning was performed.
- Differences in sensor hardware, placement, calibration and data distributions may influence performance.
- The observed detection results should not be interpreted as pure intrinsic sensor-performance comparisons.

### RADAR Feature Adaptation

The OmniHD RADAR model expects input features that are not directly available in the TruckScenes data.

In the current adapter:

- TruckScenes Radar Cross Section (RCS) is used as a proxy for the required Power feature.
- Missing SNR values are replaced with a constant value of 0.

These substitutions allow the TruckScenes data to be processed by the pretrained model, but they introduce potential feature-distribution mismatches.

RCS is not physically equivalent to received signal power, and a constant SNR value does not capture actual point-to-point signal quality.

These assumptions may affect RADAR detection performance.

### Evaluation Limitations

- Official AP and fixed-operating-point metrics use different evaluation procedures and should not be interpreted interchangeably.
- ATE, ASE and AOE are calculated from successfully matched predictions only.
- RADAR detects substantially fewer objects than LiDAR, so its matched-detection errors may reflect a more selective set of objects.
- Condition-based results may be influenced by sample imbalance and correlations between operating conditions.
- Complementarity results describe potential ground-truth coverage, not an implemented detection-fusion system.
- Additional RADAR-only detections do not automatically imply improved fused precision or mAP.
- The analysis does not directly evaluate downstream tracking, trajectory prediction or collision avoidance.

### Reproducibility and External Resources

- Raw TruckScenes data is not included in this repository.
- OmniHD-Scenes source code, pretrained checkpoints and model configurations must be obtained separately.
- Full trainval prediction PKLs and converted annotation PKLs may need to be regenerated or obtained separately.
- Saved JSON results can be inspected without rerunning the pretrained models.
- Paths stored in JSON result files may reflect the original execution environment and differ from the current repository layout.
- The complete inference pipeline requires a compatible CUDA-capable GPU and model environment.
- The dashboard and velocity-analysis modules have their own setup and dependency requirements.

The analysis is intended to provide a reproducible comparison of the selected pretrained models under the documented TruckScenes subset and evaluation protocol. Results and recommendations may need to be revised when different datasets, checkpoints, adaptation methods or evaluation settings are used.
