"""Convert TruckScenes global annotations into OmniHD reference-sensor boxes."""

from collections.abc import Mapping, Sequence

import numpy as np

from .class_mapping import OMNIHD_CLASSES, map_truckscenes_category
from .transforms import (
    global_to_sensor_transform,
    quaternion_to_rotation_matrix,
    transform_points,
)


def _vector(record: Mapping, key: str, length: int) -> np.ndarray:
    """Read one finite, fixed-length numeric annotation field."""
    if key not in record:
        raise KeyError(f"annotation is missing required field: {key}")
    try:
        values = np.asarray(record[key], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"annotation {key} must contain numeric values") from exc
    if values.shape != (length,) or not np.all(np.isfinite(values)):
        raise ValueError(f"annotation {key} must contain {length} finite numbers")
    return values


def _geometry(annotation: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate global centre, width/length/height, and quaternion rotation."""
    if not isinstance(annotation, Mapping):
        raise ValueError("annotation must be a mapping")
    center_global = _vector(annotation, "translation", 3)
    size_wlh = _vector(annotation, "size", 3)
    if np.any(size_wlh <= 0):
        raise ValueError("annotation size values must be strictly positive")
    rotation = _vector(annotation, "rotation", 4)
    R_global_from_box = quaternion_to_rotation_matrix(rotation)
    return center_global, size_wlh, R_global_from_box


def _annotation_identifiers(annotation: Mapping) -> None:
    """Require the identifying tokens retained in converted annotations."""
    for key in ("token", "sample_token", "instance_token"):
        if key not in annotation:
            raise KeyError(f"annotation is missing required field: {key}")
        if not isinstance(annotation[key], str) or not annotation[key]:
            raise ValueError(f"annotation {key} must be a non-empty string")


def convert_box_to_reference(
    annotation: Mapping,
    reference_calibrated_sensor: Mapping,
    reference_ego_pose: Mapping,
) -> np.ndarray:
    """Return a centred OmniHD metadata box ``[x,y,z,w,l,h,yaw]``.

    TruckScenes box centres and quaternions are global. OmniHD's converter
    stores geometric centres and width/length/height in that order. Its
    MMDetection3D LiDAR boxes use a clockwise yaw for the local width axis:
    yaw 0 points that axis along +x, and -pi/2 points it along +y. The
    dataset loader later shifts the centre to bottom centre using
    ``origin=(0.5, 0.5, 0.5)``. No z shift belongs in this metadata box.

    Full orientation is transformed first; yaw is the horizontal projection
    of the transformed width axis. Pitch and roll cannot be encoded in 7D.
    """
    center_global, size_wlh, R_global_from_box = _geometry(annotation)
    T_reference_from_global = global_to_sensor_transform(
        reference_calibrated_sensor, reference_ego_pose
    )
    center_reference = transform_points(
        center_global.reshape(1, 3), T_reference_from_global
    )[0]
    R_reference_from_box = T_reference_from_global[:3, :3] @ R_global_from_box
    width_axis_xy = R_reference_from_box[:2, 1]
    if np.linalg.norm(width_axis_xy) <= np.finfo(float).eps:
        raise ValueError("box width axis has no horizontal projection in reference frame")
    # MMDetection3D v0.17.1 rotates row-vector box corners clockwise for
    # positive LiDAR yaw; the width axis is TruckScenes local +y.
    yaw = np.arctan2(-width_axis_xy[1], width_axis_xy[0])
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    box = np.concatenate((center_reference, size_wlh, [yaw]))
    if not np.all(np.isfinite(box)):
        raise ValueError("converted box contains non-finite values")
    return box


def convert_annotation(
    annotation: Mapping,
    category_name: str,
    reference_calibrated_sensor: Mapping,
    reference_ego_pose: Mapping,
) -> dict | None:
    """Convert one validated annotation, or exclude a mapped-out category."""
    if not isinstance(annotation, Mapping):
        raise ValueError("annotation must be a mapping")
    _annotation_identifiers(annotation)
    _geometry(annotation)
    if not isinstance(category_name, str):
        raise ValueError("category_name must be a string")
    name = map_truckscenes_category(category_name)
    if name is None:
        return None

    box = convert_box_to_reference(
        annotation, reference_calibrated_sensor, reference_ego_pose
    )
    return {
        "box": box,
        "name": name,
        "label": OMNIHD_CLASSES.index(name),
        "annotation_token": annotation["token"],
        "instance_token": annotation["instance_token"],
        "sample_token": annotation["sample_token"],
        "original_category": category_name,
        "visibility_token": annotation.get("visibility_token"),
        "attribute_tokens": list(annotation.get("attribute_tokens") or []),
        "num_lidar_pts": annotation.get("num_lidar_pts"),
        "num_radar_pts": annotation.get("num_radar_pts"),
    }


def convert_annotations(
    annotations: Sequence[Mapping],
    category_name_by_instance_token: Mapping[str, str],
    reference_calibrated_sensor: Mapping,
    reference_ego_pose: Mapping,
) -> dict:
    """Convert a sample's annotations using the caller's instance-category map.

    Returns ``gt_boxes`` as centred Nx7 OmniHD metadata boxes, plus names,
    int64 labels, per-annotation metadata, and inclusion/exclusion counts.
    Empty included sets retain the unambiguous shapes (0, 7) and (0,).
    """
    converted = []
    num_input = 0
    for annotation in annotations:
        num_input += 1
        if not isinstance(annotation, Mapping):
            raise ValueError(f"annotation {num_input - 1} must be a mapping")
        if "instance_token" not in annotation:
            raise KeyError("annotation is missing required field: instance_token")
        instance_token = annotation["instance_token"]
        if instance_token not in category_name_by_instance_token:
            raise KeyError(f"Unknown TruckScenes instance token: {instance_token!r}")
        result = convert_annotation(
            annotation,
            category_name_by_instance_token[instance_token],
            reference_calibrated_sensor,
            reference_ego_pose,
        )
        if result is not None:
            converted.append(result)

    num_included = len(converted)
    gt_boxes = (
        np.stack([item["box"] for item in converted])
        if converted
        else np.empty((0, 7), dtype=float)
    )
    return {
        "gt_boxes": gt_boxes,
        "gt_names": [item["name"] for item in converted],
        "gt_labels": np.asarray([item["label"] for item in converted], dtype=np.int64),
        "annotations": converted,
        "num_input_annotations": num_input,
        "num_included_annotations": num_included,
        "num_excluded_annotations": num_input - num_included,
    }
