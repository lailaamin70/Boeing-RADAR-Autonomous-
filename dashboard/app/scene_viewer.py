"""
Whole-scene viewer: a top-down "video" (frame-by-frame PNG the client pages
through) and an interactive 3D point view, both combining whichever sensors
the user picks (radar/lidar/both) in a shared ego-vehicle frame so multiple
sensors line up in one plot, with an optional annotation overlay.
"""
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon

from app import detections_loader, geometry, truckscenes_loader as tsl
from config import Config

MODALITY_COLOR = {"radar": "#e2793d", "lidar": "#4bb4c4"}
ANNOTATION_COLOR = "#f2e5c9"
DETECTION_COLOR = {"radar": "#ff4d8d", "lidar": "#c6ff4d"}
EGO_COLOR = "#8b93a3"
# matplotlib marker + colour per sensor modality for the ego-truck sensor mounts
SENSOR_MARKER = {"radar": ("^", "#e2793d"), "lidar": ("D", "#4bb4c4"), "camera": ("s", "#7f9fd9")}

MAX_3D_POINTS_PER_MODALITY = 20000


def get_detections_for_sample(sample_token, modalities, score_threshold=None):
    """
    OmniHD-PointPillars detections for whichever modalities are selected,
    converted to the same ego-frame corner format as _collect_annotation_boxes.
    """
    boxes = []
    for modality in modalities:
        for pred in detections_loader.get_predictions(modality, sample_token, score_threshold):
            quat = geometry.yaw_to_quaternion(pred["yaw"])
            corners = geometry.box_corners(pred["center"], pred["size"], quat)
            boxes.append(
                {
                    "corners": corners,
                    "category": pred["label"],
                    "score": pred["score"],
                    "modality": modality,
                }
            )
    return boxes


def _collect_points(sample_token, sensors):
    """{modality: {"xyz": (N,3) ego-frame array, "color": (N,) array}}, added across every channel of that modality."""
    channels = tsl.get_sample_data_by_channel(sample_token)
    out = {}
    for modality in sensors:
        xyz_parts, color_parts = [], []
        for _, entry in channels.items():
            if entry["modality"] != modality:
                continue
            xyz, color = geometry.load_points_ego_frame(entry["sample_data"]["token"], modality)
            if len(xyz) > 0:
                xyz_parts.append(xyz)
                color_parts.append(color)
        if xyz_parts:
            out[modality] = {
                "xyz": np.concatenate(xyz_parts, axis=0),
                "color": np.concatenate(color_parts, axis=0),
            }
    return out


def _reference_sample_data(channels):
    """One sample_data token to anchor the shared ego frame - prefer lidar, else radar."""
    for _, entry in channels.items():
        if entry["modality"] == "lidar":
            return entry["sample_data"]["token"]
    for _, entry in channels.items():
        if entry["modality"] == "radar":
            return entry["sample_data"]["token"]
    return None


def _category_name(trucksc, ann):
    instance = trucksc.get("instance", ann["instance_token"])
    category = trucksc.get("category", instance["category_token"])
    return category["name"]


def _collect_annotation_boxes(sample_token, channels):
    reference = _reference_sample_data(channels)
    if reference is None:
        return []

    trucksc = tsl.get_trucksc()
    boxes = []
    for ann_token in tsl.get_sample_annotations(sample_token):
        ann = trucksc.get("sample_annotation", ann_token)
        corners = geometry.annotation_to_ego_frame(ann, reference)
        boxes.append({"corners": corners, "category": _category_name(trucksc, ann)})
    return boxes


def get_ego_vehicle(sample_token):
    """
    Static truck representation for the scene viewer, in the ego frame.
    """
    trucksc = tsl.get_trucksc()

    boxes = []
    for spec in Config.EGO_TRUCK_BOXES:
        corners = geometry.box_corners(spec["center"], spec["size"], [1, 0, 0, 0])
        boxes.append({"name": spec["name"], "corners": corners})

    sensors = []
    for channel, entry in tsl.get_sample_data_by_channel(sample_token).items():
        calib = trucksc.get("calibrated_sensor", entry["sample_data"]["calibrated_sensor_token"])
        sensors.append(
            {
                "channel": channel,
                "modality": entry["modality"],
                "xyz": [float(v) for v in calib["translation"]],
            }
        )
    return {"boxes": boxes, "sensors": sensors}


def _draw_ego_topdown(ax, sample_token):
    ego = get_ego_vehicle(sample_token)

    for i, box in enumerate(ego["boxes"]):
        footprint = box["corners"][[0, 1, 5, 4], :2]
        ax.add_patch(
            Polygon(
                footprint,
                closed=True,
                facecolor=to_rgba(EGO_COLOR, 0.25),
                edgecolor=EGO_COLOR,
                linewidth=1.2,
                zorder=0.5,
                label="ego truck" if i == 0 else None,
            )
        )
        ax.plot(footprint[[0, 1], 0], footprint[[0, 1], 1], c="#2b3140", linewidth=3, zorder=3)

    by_modality = {}
    for sensor in ego["sensors"]:
        by_modality.setdefault(sensor["modality"], []).append(sensor["xyz"])
    for modality, xyzs in by_modality.items():
        marker, color = SENSOR_MARKER.get(modality, ("o", "#ffffff"))
        arr = np.array(xyzs)
        ax.scatter(
            arr[:, 0], arr[:, 1], marker=marker, s=26, c=color,
            edgecolors="white", linewidths=0.6, zorder=6, label=f"{modality} sensors",
        )


def render_topdown_png(
    sample_token, sensors, show_annotations, show_detections=False, min_score=None,
    show_ego=False, xlim=80, ylim=80,
):
    channels = tsl.get_sample_data_by_channel(sample_token)
    points = _collect_points(sample_token, sensors)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=110)

    if "lidar" in points:
        xyz = points["lidar"]["xyz"]
        ax.scatter(xyz[:, 0], xyz[:, 1], s=1, c=MODALITY_COLOR["lidar"], alpha=0.5, label="lidar")
    if "radar" in points:
        xyz = points["radar"]["xyz"]
        ax.scatter(xyz[:, 0], xyz[:, 1], s=5, c=MODALITY_COLOR["radar"], alpha=0.85, label="radar")

    if show_ego:
        _draw_ego_topdown(ax, sample_token)

    if show_annotations:
        for box in _collect_annotation_boxes(sample_token, channels):
            # top-face corners
            footprint = box["corners"][[0, 1, 5, 4], :2]
            closed = np.vstack([footprint, footprint[0]])
            ax.plot(closed[:, 0], closed[:, 1], c=ANNOTATION_COLOR, linewidth=1)

    if show_detections:
        labeled_modalities = set()
        for det in get_detections_for_sample(sample_token, sensors, min_score):
            footprint = det["corners"][[0, 1, 5, 4], :2]
            closed = np.vstack([footprint, footprint[0]])
            label = None
            if det["modality"] not in labeled_modalities:
                label = f'{det["modality"]} detections'
                labeled_modalities.add(det["modality"])
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                c=DETECTION_COLOR.get(det["modality"], "#ffffff"),
                linewidth=1.3,
                linestyle="--",
                label=label,
            )

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    if points:
        ax.legend(loc="upper right", fontsize=8)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def _downsample(xyz, color, cap):
    if len(xyz) <= cap:
        return xyz, color
    idx = np.random.choice(len(xyz), cap, replace=False)
    return xyz[idx], color[idx]


def get_points_payload(
    sample_token, sensors, show_annotations, show_detections=False, min_score=None, show_ego=False
):
    channels = tsl.get_sample_data_by_channel(sample_token)
    points = _collect_points(sample_token, sensors)

    payload = {"modalities": {}}
    for modality, data in points.items():
        xyz, color = _downsample(data["xyz"], data["color"], MAX_3D_POINTS_PER_MODALITY)
        payload["modalities"][modality] = {
            "x": xyz[:, 0].tolist(),
            "y": xyz[:, 1].tolist(),
            "z": xyz[:, 2].tolist(),
            "color": color.tolist(),
        }

    if show_annotations:
        payload["annotations"] = [
            {"corners": box["corners"].tolist(), "category": box["category"]}
            for box in _collect_annotation_boxes(sample_token, channels)
        ]
    else:
        payload["annotations"] = []

    if show_detections:
        payload["detections"] = [
            {
                "corners": det["corners"].tolist(),
                "category": det["category"],
                "score": det["score"],
                "modality": det["modality"],
            }
            for det in get_detections_for_sample(sample_token, sensors, min_score)
        ]
    else:
        payload["detections"] = []

    if show_ego:
        ego = get_ego_vehicle(sample_token)
        payload["ego"] = {
            "boxes": [{"name": b["name"], "corners": b["corners"].tolist()} for b in ego["boxes"]],
            "sensors": ego["sensors"],
        }
    else:
        payload["ego"] = None

    return payload


def get_camera_overlay_payload(
    sample_token, channel, show_annotations=False, show_detections=False, detection_modalities=None, min_score=None
):
    """
    Annotation/detection boxes for one camera channel, so the
    client can draw an SVG overlay on top of the raw image.
    """
    channels = tsl.get_sample_data_by_channel(sample_token)
    entry = channels.get(channel)
    if entry is None or entry["modality"] != "camera":
        raise KeyError(f"no camera channel {channel!r} for sample {sample_token}")

    camera_sd_token = entry["sample_data"]["token"]
    sd = tsl.get_trucksc().get("sample_data", camera_sd_token)
    payload = {"width": sd["width"], "height": sd["height"], "annotations": [], "detections": []}

    if show_annotations:
        for box in _collect_annotation_boxes(sample_token, channels):
            uv = geometry.project_box_corners(box["corners"], camera_sd_token)
            if uv is None:
                continue
            payload["annotations"].append({"uv": uv.tolist(), "category": box["category"]})

    if show_detections and detection_modalities:
        for det in get_detections_for_sample(sample_token, detection_modalities, min_score):
            uv = geometry.project_box_corners(det["corners"], camera_sd_token)
            if uv is None:
                continue
            payload["detections"].append(
                {"uv": uv.tolist(), "category": det["category"], "modality": det["modality"], "score": det["score"]}
            )

    return payload
