# TruckScenes Dashboard: Setup and User Guide

A local web dashboard for exploring the MAN TruckScenes dataset and comparing radar, lidar and camera where you can browse scenes, replay them as a top-down or 3D view with annotations and object detections, and compare the sensors across conditions.

Throughout this guide, **project root** means the folder that contains `setup.sh`, `data/` and `dashboard/`.

## Contents

1. [Requirements](#1-requirements)
2. [Folder layout](#2-folder-layout)
3. [Setup, option A: the setup script](#3-setup-option-a-the-setup-script)
4. [Setup, option B: manual](#4-setup-option-b-manual)
5. [Configuration](#5-configuration)
6. [Using the dashboard](#6-using-the-dashboard)
7. [Caches](#7-caches)
8. [Reading the numbers: caveats](#8-reading-the-numbers-caveats)
9. [Troubleshooting](#9-troubleshooting)
10. [Tests and further reading](#10-tests-and-further-reading)

---

## 1. Requirements

| Requirement                          | Description                                                                                                |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **macOS or Linux** to use `setup.sh` | The script is bash. On Windows use WSL, or follow the [manual steps](#4-setup-option-b-manual).            |
| **Python 3.11**                      | Newer Pythons (3.12+) are not supported because of `open3d`.                                               |
| **AWS CLI**                          | Only used to download the dataset. No AWS account is needed (the bucket is read anonymously).              |
| **`unzip`**                          | The dataset arrives as zip archives.                                                                       |
| **Internet access**                  | For the dataset download, and at run time: the pages load Bootstrap, Chart.js, Plotly and fonts from CDNs. |
| **Disk space**                       | The mini dataset takes roughly 10GB, plus more space might be needed temporarily when unzipping            |
| **A modern browser with WebGL**      | The 3D view needs WebGL.                                                                                   |

---

## 2. Folder layout

```
project root/
├── setup.sh                      setup + launch script
├── requirements.txt              Python dependencies
├── .venv/                        virtual environment (created by setup)
├── data/
│   ├── man-truckscenes_*.zip     downloaded archives
│   └── man-truckscenes/
│       └── v1.2-mini/            dataset tables (JSON); samples/ and sweeps/ sit alongside
├── object_detection/                      object-detection output
│   └── results/omnihd/
│       ├── lidar_mini_predictions.pkl
│       └── radar_mini_predictions.pkl
└── dashboard/
    ├── run.py                    starts the web server
    ├── config.py                 settings
    ├── requirements.txt          what the app itself imports
    ├── app/                      the flask application
    ├── tests/                    test suite
    └── instance/                 caches, created on first use
```

The dashboard finds the dataset at `<project root>/data/man-truckscenes` by default, so as long as you keep this layout nothing needs configuring.

---

## 3. Setup, option A: the setup script

From the project root:

```bash
chmod +x setup.sh      # only if it isn't already executable
./setup.sh
```

(`bash setup.sh` also works.)

### What the scrupts

Each step is skipped if it is already done, so re-running is safe.

1. **AWS CLI**: checks it is installed; if not, offers to install it (package manager first, then AWS's official installer).
2. **`unzip`**: same check, with an offer to install.
3. **Dataset**: syncs `s3://man-truckscenes/release/mini/` into `data/`, checks every archive is intact, and extracts them to `data/man-truckscenes/v1.2-mini`.
4. **Python 3.11**: checks it is installed; if not, offers to install it.
5. **Virtual environment**: creates `.venv/` (recreating it if it looks broken).
6. **Dependencies**: installs `requirements.txt` and `truckscenes-devkit[all]` into the venv.
7. **Launch**: starts the dashboard at <http://127.0.0.1:5000>.

Anything that installs software on your machine asks for `y/N` confirmation first (you may be asked for your `sudo` password). Press `Ctrl+C` in the terminal to stop the dashboard.

### Options

| Flag | Effect |
|---|---|
| `--no-run` | Set everything up but don't start the dashboard. |
| `--force-data` | Delete the extracted dataset and downloaded zips, then download again. Use this if the dataset is corrupt or incomplete. |
| `--yes`, `-y` | Answer yes to every install prompt (for non-interactive use). Without a terminal and without `--yes`, install prompts default to "no". |
| `--help`, `-h` | Print the script's usage. |

### Starting it again later

Either re-run `./setup.sh` (every step done already should get skipped), or start it directly:

```bash
source .venv/bin/activate
cd dashboard
python run.py
```

---

## 4. Setup, option B: manual

These steps do the same thing as what the script does. Run from the project root.

### 4.1 Install the AWS CLI

Check first: `aws --version`. If it's missing:

- **macOS:** `brew install awscli`
- **Debian/Ubuntu:** `sudo apt-get install awscli`
- **Other Linux / any OS:** use AWS's installer: <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html> (on Windows, the MSI installer)

### 4.2 Download and extract the dataset

```bash
mkdir -p data
aws s3 sync --no-sign-request s3://man-truckscenes/release/mini/ data/
```

Extract every archive into `data/`:

```bash
for z in data/man-truckscenes_*.zip; do unzip -o "$z" -d data; done
```

Check it worked: `data/man-truckscenes/v1.2-mini` should exist and contain JSON tables.

```bash
ls data/man-truckscenes
```

Optional: test an archive before extracting with `unzip -tq data/<file>.zip`. If the download was interrupted, delete the bad zip and run the `aws s3 sync` command again; it only fetches what's missing.

### 4.3 Install Python 3.11

Check first: `python3.11 --version`.

- **macOS:** `brew install python@3.11`
- **Debian/Ubuntu:** `sudo apt-get install python3.11 python3.11-venv python3.11-dev`
- **Fedora/RHEL:** `sudo dnf install python3.11`
- **Windows:** install 3.11 from python.org and use `py -3.11` below.

### 4.4 Create the environment and install dependencies

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "truckscenes-devkit[all]"
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "truckscenes-devkit[all]"
```

If you later see `ModuleNotFoundError` for something the dashboard imports (for example `pyquaternion`), also run `pip install -r dashboard/requirements.txt`.

### 4.5 Run the dashboard

```bash
cd dashboard
python run.py
```

Open <http://127.0.0.1:5000>.

### 4.6 Check it worked

- The sidebar shows `v1.2-mini` under the app name.
- **Scenes** lists the scenes (the mini split has 10, with 400 keyframes in total).
- Opening a scene, then **Open scene viewer**, shows a top-down view of points.

---

## 5. Configuration

Settings are read from environment variables (set them before starting) or can be edited in `dashboard/config.py`. With `setup.sh`, export the variable in the same terminal first: it is inherited by the dashboard.

```bash
export TRUCKSCENES_VERSION=v1.2-mini
./setup.sh
```

| Variable                 | Default                                            | Description                                                     |
| ------------------------ | -------------------------------------------------- | --------------------------------------------------------------- |
| `TRUCKSCENES_DATAROOT`   | `<project root>/data/man-truckscenes`              | Folder containing `samples/`, `sweeps/` and the version folder. |
| `TRUCKSCENES_VERSION`    | `v1.2-mini`                                        | Must exactly match a folder name inside the dataroot.           |
| `OMNIHD_RESULTS_DIR`     | `<project root>/object_detection/results/omnihd`   | Where the detection `.pkl` files are looked for.                |
| `ANALYSIS_CACHE_PATH`    | `dashboard/instance/analysis_expensive_cache.json` | Cache for the Analysis page's point-cloud metrics.              |
| `DATA_VOLUME_CACHE_PATH` | `dashboard/instance/data_volume_cache.json`        | Cache for the Overview page's data-volume scan.                 |
| `SECRET_KEY`             | a development value                                | Flask secret key.                                               |

Other miscellaneous options:

| Location                                              | Description                                                                           |
|---|---|
| `config.py`, `EGO_TRUCK_BOXES`                        | The truck outline drawn in the scene viewer (see [6.5](#65-scene-viewer)).            |
| `config.py`, `SCENES_PER_PAGE`                        | Scenes per page on the Scenes list (default 12).                                      |
| `app/detections_loader.py`, `DEFAULT_SCORE_THRESHOLD` | Default detection score cut-offs (0.3 lidar, 0.6 radar).                              |
| `run.py`                                              | The port (5000). If you change it, the URL printed by `setup.sh` will be out of date. |

`run.py` starts Flask in debug mode, which is fine on your own machine but should not be exposed to a network.

---

## 6. Using the dashboard

The sidebar has four pages: **Scenes**, **Analysis**, **Overview**, **Detectability**. The dataset version is shown under the app name.

There is a consistent colour scheme: **radar=orange, lidar=teal, camera=blue**.

### 6.1 Scenes

The home page lists every scene as a card (name, number of keyframes, up to four condition tags).

- Click a **tag** at the top to show only scenes with that condition; click **all** to clear the filter.
- Click a card to open the scene.
- Long lists are paginated.

### 6.2 Scene page

Shows the scene's tags and description, then:

- **Open scene viewer**: replay the whole scene (see [6.5](#65-scene-viewer)).
- **Samples (2 Hz keyframes)**: a numbered strip, one square per keyframe. Click one to open its sample page.
- **Ego motion**: how the truck moved during the scene.
  - speed (km/h), longitudinal acceleration, yaw rate
  - **cabin minus chassis tilt** (pitch and roll): how much the cab moves relative to the chassis, which can affect sensor alignment
  - the **path** driven, relative to the start, on equal axes so the shape isn't distorted
  - a summary line with mean speed, max speed and duration

### 6.3 Sample page

One card per sensor channel for a single keyframe, with **Prev** / **Next** buttons to step through the scene.

- **Camera** channels show the image.
- **Lidar and radar** channels show a bird's-eye plot in *that sensor's own frame* (±100 m; radar points are coloured by RCS).

### 6.4 Combining sensors: the ego frame

The scene viewer draws everything in the **ego (truck) frame**: x forward, y left, z up. Each sensor's points are moved into that frame using its calibration so radar and lidar line up in one picture.

### 6.5 Scene viewer

Open it from a scene page. The left panel holds the controls; the right side is the player.

**View**

| Option       | Description                                                                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Top-down** | A bird's-eye image of 80 m around the truck.                                                                                                           |
| **3D**       | An interactive 3D plot where you can drag to rotate, scroll to zoom, right-drag to pan. Your viewing angle is kept as frames change and while playing. |

**Sensors**: tick **Radar**, **Lidar**, or both. Both ticked is the combined view.

**Overlays**

| Overlay               | Description                                                                                                                                                                                                                                     | Default |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| **Ego truck**         | A translucent box for the truck, a dark front edge in top-down, and a marker for every sensor at its mounting position (radar triangle/square, lidar diamond, camera square/circle). You can hover over a sensor in 3D to see its channel name. | on      |
| **Annotations**       | Ground-truth boxes, **solid cream** outlines.                                                                                                                                                                                                   | on      |
| **Object detections** | Model predictions, **dashed** outlines coloured by the sensor that produced them (radar magenta, lidar lime). Ticking it reveals a **Min score** slider.                                                                                        | off     |

The truck box is an **approximation** (see [section 8](#8-reading-the-numbers-caveats)) while the sensor markers are real positions.

**Min score**: until you move the slider it says "default" and each sensor uses its own cut-off (lidar 0.3, radar 0.6). Move it to apply one threshold to both. Radar detections are mostly noise, so raise it for a cleaner view.

**Camera**: pick a camera channel (or **None**). The image follows the current frame, and when annotations and/or detections are on their boxes are **projected onto the image**. Boxes with any corner behind the camera are skipped rather than drawn wrongly, so objects very close to a camera may not appear.

**Playback**

- **Play/Pause** and a **frame slider** with a counter (`3 / 40`) and the frame's timestamp.
- **Speed** (0.5x, 1x, 2x) sets the delay between frames (500, 250, 125 ms). Each frame is rendered on request, so the speed might be affected by your machine.
- Pressing **Play** at the last frame starts again from frame 1.
- This is a frame-by-frame player, not an encoded video file.
- In 3D, large point clouds are randomly thinned to 20,000 points per sensor to keep the browser responsive.

### 6.6 Analysis

Compare radar, lidar and camera across conditions or individual scenes. The charts update from the controls on the left.

1. **Group by**: *Condition / tag* (one bar group per tag, averaged over the scenes that have the tag) or *Individual scene*.
2. **Conditions**: tick tags to restrict the comparison. Nothing ticked means every condition.
3. **Metrics**: tick what to chart (each is labelled with its sensor).
4. **Compute point-cloud metrics**: opt-in for the metrics marked *reads sensor files*.
5. **Apply**.

You get one bar chart per metric (bars coloured by sensor) plus a table of the exact values, which is handy to copy into a report.

| Metric                                      | Cost        | Description                                                                       |
| ------------------------------------------- | ----------- | --------------------------------------------------------------------------------- |
| Keyframes in scene                          | cheap       | Number of keyframes.                                                              |
| Avg. annotation visibility                  | cheap       | Average visibility level of annotated objects at an ordinal level.                |
| Avg. lidar / radar pts per annotated object | cheap       | Mean point count stored on each annotation.                                       |
| Camera frames in scene                      | cheap       | Camera images across the keyframes.                                               |
| Points per sweep                            | reads files | Mean number of points per lidar/radar sweep.                                      |
| Max range                                   | reads files | Farthest point per sweep, averaged (m).                                           |
| Azimuth spread                              | reads files | Max minus min azimuth per sweep, averaged (degrees). Not wrap-aware.              |
| Mean \|velocity\|                           | reads files | Magnitude of the points' (vx, vy) as stored in the data (m/s). This is Radar only |
| Mean RCS                                    | reads files | Radar only (dBsm).                                                                |

The "reads files" metrics open the sensor files for **every keyframe** of every scene involved, so the first run for a set of scenes takes a while. Results are cached to disk ([section 7](#7-caches)), so later runs are much faster. If nothing is ticked you'll be asked to select a metric.

### 6.7 Overview

This page features an overview of the full dataset. 

- **Tiles**: scenes, samples, instances, annotations, `sample_data` files, keyframe files.
- **Annotations per category** (with instance counts), **annotations by distance** from the truck, **annotation visibility**, **attributes**, **scenes per condition tag**.
- **Scenes table**: each scene (linked), its tags, keyframes, annotations, and mean/max speed.
- **Data volume per sensor**: files, total MB, KB per file, files per second and **MB per second** per channel, plus totals per modality. It counts every sweep and assumes 20-second scenes. It may be useful for the data-rate and cost side of a sensor comparison.

### 6.8 Detectability

This page shows many lidar and radar points land on annotated objects at each distance, and which sensors reach each object. It uses the point counts stored on the annotations.

**Controls**

- **"Seen" threshold**: the minimum points for an object to count as seen by lidar (default 5) and radar (default 1).
- **Include camera in overlap chart**: adds the camera, you can also adjust it's visibility.
- **Object category** and **Scene conditions**: The tag filters, if none are ticked they are all included.
- **Apply** to refresh.

**Output**

- **Summary tiles**: number of objects, and for each of lidar and radar the mean points, the % with zero points, and the % seen.
- **Mean points per object** by distance bin (0-10 m up to 150+ m).
- **Objects with at least the threshold number of points (%)** by distance.
- **Which sensors see each object**: a stacked chart per distance bin (lidar only, radar only, lidar + radar, none, plus camera combinations if included). A **percent** switch toggles between shares and counts. There is also a  none class for objects no sensor can see.
- **Detail table** with the exact figures per bin.

Every distance bin is labelled with its object count (`n=`). It is best to check it when examining data as small bins can be noisy.

### 6.9 Object detections: setup

The detection overlay reads OmniHD-PointPillars predictions from `OMNIHD_RESULTS_DIR` (default `object_detection/results/omnihd/` in the project root):

- `lidar_mini_predictions.pkl`
- `radar_mini_predictions.pkl`

The dashboard could theoretically read other object detection results from different models if they follow the same format (boxes as `[x, y, z, w, l, h, yaw, vx, vy]` in the ego frame, classes `car`, `pedestrian`, `rider`, `large_vehicle`). 

### 6.10 JSON endpoints

| Endpoint                                               | Returns                                                                                                                            |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `/api/overview`                                        | Dataset counts and distributions                                                                                                   |
| `/api/data_volume`                                     | Per-sensor file volume (`?refresh=1` rescans the data).                                                                            |
| `/api/detectability`                                   | Detectability data. The parameters are: `category` and `tag` (repeatable), `min_lidar`, `min_radar`, `camera=1`, `min_visibility`. |
| `/api/analysis`                                        | Analysis data. Parameters: `metric` and `tag` (repeatable), `group_by` (`tag` or `scene`), `expensive=1`.                          |
| `/api/scene/<scene token>/ego_motion`                  | Ego-motion series for a scene.                                                                                                     |
| `/sample/<sample token>/topdown.png`                   | Top-down image. Parameters: `sensor` (repeatable), `annotations=1`, `detections=1`, `min_score`, `ego=1`.                          |
| `/sample/<sample token>/points.json`                   | Points and overlay boxes for the 3D view with the same parameters.                                                                 |
| `/sample/<sample token>/camera/<channel>/overlay.json` | Projected boxes for a camera image.                                                                                                |

---

## 7. Troubleshooting

| Symptom                                                             | Likely cause and fix                                                                                                                                                                   |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError` on startup mentioning a version folder          | The dataset isn't where the dashboard looks, or the version string is wrong. Check `ls data/man-truckscenes`; the version folder name must match `TRUCKSCENES_VERSION`.                |
| `Address already in use` / port 5000 busy                           | On macOS the AirPlay Receiver uses port 5000. Turn it off in System Settings, or change the port in `run.py`.                                                                          |
| Charts or the 3D view are empty or the page looks unstyled          | The page libraries load from CDNs, so you need internet access.                                                                                                                        |
| 3D view is blank or errors                                          | Your browser has WebGL disabled or unavailable. The top-down view will still work.                                                                                                     |
| `ModuleNotFoundError: pyquaternion` (or another import)             | `pip install -r dashboard/requirements.txt` inside the virtual environment.                                                                                                            |
| Errors about `matplotlib.cm.get_cmap`                               | matplotlib is too new for the devkit. Use `pip install "matplotlib<3.9"`.                                                                                                              |
| Install problems with `open3d` or a very new Python                 | Use Python 3.11.                                                                                                                                                                       |
| Overview data-volume table or Analysis point-cloud metrics are slow | This is expected the first time as they read many files. These results will be cached for future runs.                                                                                 |
| Numbers look the same after changing data or settings               | Delete the files in `dashboard/instance/` and reload.                                                                                                                                  |
| Dataset download fails, or an archive is reported corrupt           | Re-run `./setup.sh --force-data`. Manually: delete the bad `data/man-truckscenes_*.zip` and repeat the `aws s3 sync` and `unzip` steps.                                                |
| `pip install` fails inside `setup.sh`                               | Read the error above the `[fail]` line, fix it, and re-run `./setup.sh`, the completed steps will be skipped.                                                                          |
| The detections checkbox shows nothing                               | The `.pkl` files aren't in `OMNIHD_RESULTS_DIR` ([6.9](#69-object-detections-setup)), or no detection passes the score threshold. You can fix the latter issue by lowering the slider. |

---

## 10. Tests

The tests were used in development, but you can still run them if you are curious from `dashboard/` with the virtual environment active:

```bash
python -m unittest discover -s tests -t .
```

It runs against a small synthetic dataset. The python libraries `selenium` and `pytest` are required.
