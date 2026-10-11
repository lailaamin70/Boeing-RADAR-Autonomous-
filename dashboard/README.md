# TruckScenes Explorer Dashboard

The dashboard provides a browser interface for exploring TruckScenes Camera, LiDAR and RADAR data and viewing saved object-detection results.

## Table of Contents

- [Setup](#setup)
- [Data Configuration](#data-configuration)
- [Running the Dashboard](#running-the-dashboard)
- [Usage](#usage)
- [Object Detection Results](#object-detection-results)
- [Troubleshooting](#troubleshooting)

## Setup

The setup commands below assume that your terminal starts in the `dashboard/` directory. The `cd ..` command moves to the repository root.

If you are already at the repository root, skip `cd ..`.

### Automatic (Recommended)

Run the setup script, add permissions if needed:

```bash
cd ..
chmod +x setup.sh
./setup.sh
```

The script downloads and extracts the mini dataset, checks for Python 3.11, creates `.venv`, installs dependencies and starts the dashboard. It asks for confirmation before attempting to install system software.

To complete setup without starting the dashboard:

```bash
./setup.sh --no-run
```

For Windows PowerShell, use the manual setup instructions below.

### Manual

Download data:

```bash
cd ..
aws s3 sync --no-sign-request s3://man-truckscenes/release/mini/ data/
```

If the AWS cli is not installed, please install using your preferred method.

Then unzip the two zip files downloaded by the aws command in the data folder.

Creating the virtual environment (Python 3.11 required):

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install truckscenes-devkit[all]
```

For Windows PowerShell, replace the environment creation and activation commands with:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

Then install the dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install "truckscenes-devkit[all]"
```

If the dataset is already available locally, skip the download and configure its location below.

## Data Configuration

Dataset settings are defined in [config.py](config.py).

| Environment Variable | Default |
|---|---|
| `TRUCKSCENES_DATAROOT` | `<repository-root>/data/man-truckscenes` |
| `TRUCKSCENES_VERSION` | `v1.2-mini` |

The expected default structure is:

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

The automatic setup script extracts the dataset into this location.

To use another dataset location, set the environment variables in the terminal before starting the dashboard.

### Windows PowerShell

```powershell
$env:TRUCKSCENES_DATAROOT = "D:\MAN_TruckScenes"
$env:TRUCKSCENES_VERSION = "v1.2-mini"
```

### Linux/macOS Bash

```bash
export TRUCKSCENES_DATAROOT="/path/to/man-truckscenes"
export TRUCKSCENES_VERSION="v1.2-mini"
```

Replace the example path with the directory containing `samples/`, `sweeps/` and the versioned metadata directory.

These settings apply to the current terminal session. Restart the dashboard after changing them.

The dashboard uses `TRUCKSCENES_DATAROOT`; the velocity module uses `TRUCKSCENES_DATA_ROOT`.

## Running the Dashboard

After manual setup, your terminal should be at the repository root with the virtual environment activated.

Run it:

```bash
cd dashboard
python run.py
```

Then open http://127.0.0.1:5000.

Keep the terminal running while using the dashboard. Press `Ctrl+C` to stop it.

### Subsequent Launches

From the repository root, activate the environment and start the application.

Manual setup creates `venv/`; automatic setup creates `.venv/`. Use the activation path corresponding to your setup.

**Linux/macOS — manual setup**

```bash
source venv/bin/activate
cd dashboard
python run.py
```

**Windows PowerShell — manual setup**

```powershell
.\venv\Scripts\Activate.ps1
cd dashboard
python run.py
```

**Linux/macOS — automatic setup**

```bash
source .venv/bin/activate
cd dashboard
python run.py
```

## Usage

### Browse Scenes and Samples

1. Open the home page to browse the available scenes.
2. Filter scenes by the available tags and use pagination to browse the results.
3. Open a scene to view its description, tags and sample list.
4. Select a sample to inspect its available sensor channels, including camera images and LiDAR/RADAR bird's-eye-view plots.

### Scene Viewer

Open the scene viewer to explore samples through scene playback.

The viewer supports:

- Top-down and 3D point-cloud views.
- LiDAR and RADAR sensor selection.
- Camera views for available channels.
- Ground-truth annotation and saved detection overlays.
- A minimum prediction-score threshold for filtering detections.

Detection overlays require prediction files covering the selected samples. They display saved model predictions; they do not run model inference.

### Cross-Scene Analysis

Open `/analysis` to access the cross-scene comparison page.

Select the available tags and metrics, and group results by tag or scene. Additional computationally expensive metrics can be included when needed.

The available views and results depend on the dataset, sensor files and prediction files available locally.

## Object Detection Results

The default prediction files are:

- `object_detection/results/omnihd/lidar_mini_predictions.pkl`
- `object_detection/results/omnihd/radar_mini_predictions.pkl`

The paths are configured through `DETECTIONS_PATHS` in [config.py](config.py).

To use another prediction directory, set `OMNIHD_RESULTS_DIR` before starting the dashboard.

**Windows PowerShell**

```powershell
$env:OMNIHD_RESULTS_DIR = "D:\analysis\omnihd"
```

**Linux/macOS Bash**

```bash
export OMNIHD_RESULTS_DIR="/path/to/omnihd"
```

The configured filenames remain the same. To use different filenames, update `DETECTIONS_PATHS` in `config.py`.

Predictions must correspond to the loaded dataset and sample tokens. Changing the dataset version does not automatically select different prediction files.

For inference, evaluation and saved reports, see the [Object Detection README](../object_detection/README.md).

## Troubleshooting

| Problem | Checks |
|---|---|
| Dataset files cannot be found | Check `TRUCKSCENES_DATAROOT`, the versioned metadata directory and extracted sensor folders. |
| Python dependency cannot be imported | Activate the intended environment and install the root `requirements.txt` and `truckscenes-devkit[all]`. |
| Detection results are unavailable | Check the prediction paths, filenames and whether the files cover the selected sample. |
| Browser cannot connect | Check that `python run.py` is still running without an error, then open `http://127.0.0.1:5000`. |
| Port 5000 is already in use | Stop the conflicting application or change the port in `run.py` and use the corresponding browser address. |

Restart the dashboard after changing environment variables or configuration.

The current `run.py` enables Flask debugging. These instructions describe local dashboard use.
