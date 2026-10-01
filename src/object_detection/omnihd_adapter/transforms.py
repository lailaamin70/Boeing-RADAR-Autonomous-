"""Reusable coordinate-transformation utilities for the OmniHD adapter."""

import numpy as np


def _as_float_array(values, expected_size: int, name: str) -> np.ndarray:
    """Convert a fixed-size sequence to a finite one-dimensional float array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if array.shape != (expected_size,):
        raise ValueError(f"{name} must contain exactly {expected_size} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_transform(transform) -> np.ndarray:
    """Convert and validate a homogeneous 4x4 transform."""
    try:
        array = np.asarray(transform, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("transform must be a numeric 4x4 array") from exc

    if array.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    if not np.all(np.isfinite(array)):
        raise ValueError("transform must contain only finite values")
    return array


def quaternion_to_rotation_matrix(quaternion) -> np.ndarray:
    """Return a 3x3 rotation matrix from a TruckScenes ``[w, x, y, z]`` quaternion."""
    w, x, y, z = _as_float_array(quaternion, 4, "quaternion")
    norm = np.linalg.norm([w, x, y, z])
    if norm <= np.finfo(float).eps:
        raise ValueError("quaternion norm is effectively zero")

    w, x, y, z = np.array([w, x, y, z]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def make_transform_matrix(translation, quaternion) -> np.ndarray:
    """Build ``T_target_from_source`` from translation and ``[w, x, y, z]`` rotation."""
    translation_array = _as_float_array(translation, 3, "translation")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    transform[:3, 3] = translation_array
    return transform


def invert_transform(transform) -> np.ndarray:
    """Return the rigid-transform inverse of a 4x4 homogeneous transform."""
    transform_array = _validate_transform(transform)
    rotation = transform_array[:3, :3]
    translation = transform_array[:3, 3]

    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _validate_points(points, name: str) -> np.ndarray:
    """Convert and validate an Nx3 array of points or vectors."""
    try:
        array = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric Nx3 array") from exc

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def transform_points(points, transform) -> np.ndarray:
    """Transform Nx3 row-vector points with ``T_target_from_source``."""
    points_array = _validate_points(points, "points")
    transform_array = _validate_transform(transform)
    return points_array @ transform_array[:3, :3].T + transform_array[:3, 3]


def transform_vectors(vectors, transform) -> np.ndarray:
    """Transform Nx3 row-vector directions with rotation only."""
    vectors_array = _validate_points(vectors, "vectors")
    transform_array = _validate_transform(transform)
    return vectors_array @ transform_array[:3, :3].T


def transform_from_record(record: dict) -> np.ndarray:
    """Build ``T_target_from_source`` from a TruckScenes pose record."""
    missing_keys = [key for key in ("translation", "rotation") if key not in record]
    if missing_keys:
        raise KeyError(f"record is missing required key(s): {', '.join(missing_keys)}")
    return make_transform_matrix(record["translation"], record["rotation"])


def sensor_to_global_transform(
    calibrated_sensor_record: dict,
    ego_pose_record: dict,
) -> np.ndarray:
    """Return ``T_global_from_sensor`` for a calibrated sensor and ego pose."""
    T_ego_from_sensor = transform_from_record(calibrated_sensor_record)
    T_global_from_ego = transform_from_record(ego_pose_record)
    return T_global_from_ego @ T_ego_from_sensor


def global_to_sensor_transform(
    calibrated_sensor_record: dict,
    ego_pose_record: dict,
) -> np.ndarray:
    """Return ``T_sensor_from_global`` for a calibrated sensor and ego pose."""
    return invert_transform(
        sensor_to_global_transform(calibrated_sensor_record, ego_pose_record)
    )


def sensor_to_sensor_transform(
    source_calibrated_sensor: dict,
    source_ego_pose: dict,
    reference_calibrated_sensor: dict,
    reference_ego_pose: dict,
) -> np.ndarray:
    """Return ``T_reference_sensor_from_source_sensor`` across any timestamps."""
    T_global_from_source = sensor_to_global_transform(
        source_calibrated_sensor, source_ego_pose
    )
    T_global_from_reference = sensor_to_global_transform(
        reference_calibrated_sensor, reference_ego_pose
    )
    return invert_transform(T_global_from_reference) @ T_global_from_source


def sensor_to_ego_reference_transform(
    source_calibrated_sensor: dict,
    source_ego_pose: dict,
    reference_ego_pose: dict,
) -> np.ndarray:
    """Return ``T_reference_ego_from_source_sensor`` for a chosen ego reference."""
    T_global_from_source = sensor_to_global_transform(
        source_calibrated_sensor, source_ego_pose
    )
    T_global_from_reference_ego = transform_from_record(reference_ego_pose)
    return invert_transform(T_global_from_reference_ego) @ T_global_from_source


def global_to_ego_transform(ego_pose_record: dict) -> np.ndarray:
    """Return ``T_ego_from_global`` from an ego-pose record."""
    return invert_transform(transform_from_record(ego_pose_record))
