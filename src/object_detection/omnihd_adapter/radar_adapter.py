"""TruckScenes to OmniHD raw RADAR sweep adapter.

OmniHD's pretrained RADAR PointPillars model expects power and SNR features,
but TruckScenes does not expose direct equivalents for both. For cross-dataset
transfer, TruckScenes RCS can be used as a proxy for OmniHD power. RCS is not
assumed to be physically or statistically identical to OmniHD power.

TruckScenes does not expose the required SNR feature, so the caller must supply
a constant placeholder. These substitutions are explicit transfer
approximations and must be considered when interpreting RADAR performance.

This module writes OmniHD's raw Nx8 sweep representation. The official
``LoadRadarPointsMultiSweeps`` pipeline remains responsible for ego-motion
compensation, transformation into the current ego frame, time differences,
and multi-sweep aggregation.
"""

from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from pypcd4 import PointCloud

from .transforms import (
    quaternion_to_rotation_matrix,
    sensor_to_ego_reference_transform,
)


TRUCKSCENES_RADAR_CHANNELS = (
    "RADAR_LEFT_BACK",
    "RADAR_LEFT_FRONT",
    "RADAR_LEFT_SIDE",
    "RADAR_RIGHT_BACK",
    "RADAR_RIGHT_FRONT",
    "RADAR_RIGHT_SIDE",
)

_RADAR_FIELDS = ("x", "y", "z", "vrel_x", "vrel_y", "vrel_z", "rcs")
_MIN_RADAR_RANGE_METRES = 1e-8


def _point_array(values: ArrayLike, columns: int, name: str) -> np.ndarray:
    """Return a finite float64 array with shape (N, columns)."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric (N, {columns}) array") from exc
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_scalar(value: Real, name: str) -> float:
    """Return one finite real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric scalar")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return scalar


def _metadata_vector(values: ArrayLike, length: int, name: str) -> np.ndarray:
    """Return one finite fixed-length metadata vector."""
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain {length} numeric values") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {length} finite numeric values")
    return vector


def load_truckscenes_radar(path: str | Path) -> np.ndarray:
    """Load Nx7 ``[x,y,z,vrel_x,vrel_y,vrel_z,rcs]`` from a RADAR PCD.

    Fields are selected by name through pypcd4. RCS and sensor-frame relative
    velocity components are preserved without scaling or normalization.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RADAR PCD file not found: {path}")
    if path.suffix.lower() != ".pcd":
        raise ValueError(f"Expected a RADAR .pcd file: {path}")

    try:
        cloud = PointCloud.from_path(path)
    except Exception as exc:
        raise ValueError(f"Cannot read RADAR PCD {path}: {exc}") from exc

    data = cloud.pc_data
    names = data.dtype.names or ()
    missing = [field for field in _RADAR_FIELDS if field not in names]
    if missing:
        raise ValueError(f"RADAR PCD {path} is missing fields: {', '.join(missing)}")
    if data.size != cloud.metadata.points:
        raise ValueError(f"RADAR PCD {path} point count does not match its header")

    data = data.reshape(-1)
    points = np.column_stack([data[field] for field in _RADAR_FIELDS])
    return _point_array(points, 7, f"RADAR PCD {path}")


def extract_radar_features(
    points: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split Nx7 native points into XYZ, relative velocity, and unscaled RCS."""
    points = _point_array(points, 7, "TruckScenes RADAR points")
    return points[:, :3].copy(), points[:, 3:6].copy(), points[:, 6:7].copy()


def derive_radial_velocity(xyz: ArrayLike, vrel: ArrayLike) -> np.ndarray:
    """Project sensor-frame relative velocity onto the outward line of sight.

    The result is ``dot(vrel, xyz / ||xyz||)`` using all three TruckScenes
    velocity components. Positive values point away from the sensor; no sign
    inversion is applied. Zero and near-zero ranges are rejected.
    """
    xyz = _point_array(xyz, 3, "xyz")
    vrel = _point_array(vrel, 3, "vrel")
    if xyz.shape[0] != vrel.shape[0]:
        raise ValueError("xyz and vrel must have the same number of points")

    ranges = np.linalg.norm(xyz, axis=1)
    invalid = ranges <= _MIN_RADAR_RANGE_METRES
    if np.any(invalid):
        indices = np.flatnonzero(invalid)
        preview = ", ".join(str(index) for index in indices[:5])
        raise ValueError(
            "Cannot derive radial velocity for zero/near-zero range points "
            f"at indices: {preview}"
        )
    radial_velocity = np.sum(vrel * (xyz / ranges[:, None]), axis=1, keepdims=True)
    return _point_array(radial_velocity, 1, "radial velocity")


def adapt_signal_features(
    rcs: ArrayLike,
    power_mode: str,
    snr_fill_value: Real,
    power_fill_value: Real | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create explicit OmniHD power and SNR transfer approximations.

    ``rcs_proxy`` copies TruckScenes RCS without claiming equivalence to
    OmniHD power and rejects a redundant power fill value. ``constant``
    requires ``power_fill_value``. SNR is always a caller-supplied constant.
    """
    rcs = _point_array(rcs, 1, "rcs")
    snr_value = _finite_scalar(snr_fill_value, "snr_fill_value")

    if power_mode == "rcs_proxy":
        if power_fill_value is not None:
            raise ValueError("power_fill_value must be omitted for power_mode='rcs_proxy'")
        power = rcs.copy()
    elif power_mode == "constant":
        if power_fill_value is None:
            raise ValueError("power_fill_value is required for power_mode='constant'")
        power_value = _finite_scalar(power_fill_value, "power_fill_value")
        power = np.full(rcs.shape, power_value, dtype=np.float64)
    else:
        raise ValueError("power_mode must be 'rcs_proxy' or 'constant'")

    snr = np.full(rcs.shape, snr_value, dtype=np.float64)
    return power, snr


def pack_omnihd_radar_raw(
    points: ArrayLike,
    power_mode: str,
    snr_fill_value: Real,
    power_fill_value: Real | None = None,
) -> np.ndarray:
    """Pack native points as OmniHD raw Nx8 float32 RADAR sweeps.

    Output columns are ``[x, y, z, radial_velocity, power, 0, SNR, 0]``.
    The zero columns are motion-state and valid-flag placeholders discarded by
    the official loader. They are not assigned fabricated semantic meaning.
    This is raw sweep data, not the loader's final model-feature layout.
    """
    xyz, vrel, rcs = extract_radar_features(points)
    radial_velocity = derive_radial_velocity(xyz, vrel)
    power, snr = adapt_signal_features(
        rcs, power_mode, snr_fill_value, power_fill_value
    )
    raw = np.zeros((xyz.shape[0], 8), dtype=np.float64)
    raw[:, :3] = xyz
    raw[:, 3:4] = radial_velocity
    raw[:, 4:5] = power
    raw[:, 6:7] = snr
    with np.errstate(over="ignore", invalid="ignore"):
        raw_float32 = raw.astype(np.float32)
    if not np.all(np.isfinite(raw_float32)):
        raise ValueError("packed RADAR values must be representable as float32")
    return raw_float32


def save_omnihd_radar_raw(points: ArrayLike, output_path: str | Path) -> None:
    """Write validated Nx8 points as headerless float32 binary."""
    points = _point_array(points, 8, "OmniHD raw RADAR points")
    with np.errstate(over="ignore", invalid="ignore"):
        packed = points.astype(np.float32)
    if not np.all(np.isfinite(packed)):
        raise ValueError("raw RADAR values must be representable as float32")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packed.tofile(output_path)


def convert_radar_source(
    radar_path: str | Path,
    power_mode: str,
    snr_fill_value: Real,
    power_fill_value: Real | None = None,
) -> np.ndarray:
    """Load one TruckScenes PCD and return an untransformed OmniHD raw sweep.

    XYZ remains in the source sensor frame. Ego-motion compensation and the
    source-sensor to reference-ego transform are deferred to OmniHD's loader.
    """
    return pack_omnihd_radar_raw(
        load_truckscenes_radar(radar_path),
        power_mode,
        snr_fill_value,
        power_fill_value,
    )


def build_radar_sweep_metadata(
    converted_data_path: str | Path,
    timestamp: Real,
    ego_velocity: ArrayLike,
    source_calibrated_sensor: Mapping,
    source_ego_pose: Mapping,
    reference_ego_pose: Mapping,
) -> dict:
    """Build metadata consumed by OmniHD ``LoadRadarPointsMultiSweeps``.

    ``sensor2lidar_*`` is the official compatibility key, but its target here
    is the caller's current ego frame. ``ego_velocity`` must be supplied in the
    source ego frame. The source sensor-to-ego quaternion is retained for the
    loader's radial ego-motion compensation. Timestamp remains microseconds.
    """
    if not isinstance(converted_data_path, (str, Path)):
        raise ValueError("converted_data_path must be a string or Path")
    if isinstance(converted_data_path, str) and not converted_data_path.strip():
        raise ValueError("converted_data_path must not be empty")
    path = Path(converted_data_path)
    if isinstance(timestamp, (bool, np.bool_)) or not isinstance(timestamp, Real):
        raise ValueError("timestamp must be an integer number of microseconds")
    timestamp_float = float(timestamp)
    if not np.isfinite(timestamp_float) or not timestamp_float.is_integer():
        raise ValueError("timestamp must be an integer number of microseconds")
    timestamp_us = int(timestamp_float)

    ego_velocity_array = _metadata_vector(ego_velocity, 3, "ego_velocity")
    if "rotation" not in source_calibrated_sensor:
        raise KeyError("source_calibrated_sensor is missing required key: rotation")
    sensor2ego_rotation = _metadata_vector(
        source_calibrated_sensor["rotation"], 4, "sensor2ego_rotation"
    )
    quaternion_to_rotation_matrix(sensor2ego_rotation)

    T_reference_ego_from_source_sensor = sensor_to_ego_reference_transform(
        source_calibrated_sensor, source_ego_pose, reference_ego_pose
    )
    if not np.all(np.isfinite(T_reference_ego_from_source_sensor)):
        raise ValueError("source-sensor to reference-ego transform is not finite")

    return {
        "data_path": str(path),
        "timestamp": timestamp_us,
        "ego_velocity": ego_velocity_array.tolist(),
        "sensor2ego_rotation": sensor2ego_rotation.tolist(),
        "sensor2lidar_rotation": T_reference_ego_from_source_sensor[:3, :3].copy(),
        "sensor2lidar_translation": T_reference_ego_from_source_sensor[:3, 3].copy(),
    }


def collect_radar_sweep_records(
    current_sample_data: Mapping,
    sample_data_by_token: Mapping[str, Mapping],
    sweeps_num: int = 3,
) -> list[Mapping]:
    """Return up to ``sweeps_num`` distinct records as current, prev1, prev2.

    TruckScenes ``prev`` chains terminate with an empty string at scene
    boundaries. Available distinct records are returned without padding; the
    OmniHD loader explicitly accepts fewer than ``sweeps_num`` entries.
    Cyclic chains and missing referenced records are errors.
    """
    if isinstance(sweeps_num, (bool, np.bool_)) or not isinstance(sweeps_num, Integral):
        raise ValueError("sweeps_num must be an integer greater than or equal to 1")
    if sweeps_num < 1:
        raise ValueError("sweeps_num must be an integer greater than or equal to 1")
    if not isinstance(current_sample_data, Mapping):
        raise ValueError("current_sample_data must be a mapping")

    sweeps = []
    seen_tokens = set()
    record = current_sample_data
    while len(sweeps) < sweeps_num:
        if "token" not in record or "prev" not in record:
            raise KeyError("sample_data record requires token and prev fields")
        token = record["token"]
        if not isinstance(token, str) or not token:
            raise ValueError("sample_data token must be a non-empty string")
        if token in seen_tokens:
            raise ValueError(f"Cyclic RADAR prev chain detected at token: {token}")
        seen_tokens.add(token)
        sweeps.append(record)

        previous_token = record["prev"]
        if previous_token == "":
            break
        if not isinstance(previous_token, str) or not previous_token:
            raise ValueError("sample_data prev must be a token or an empty string")
        if previous_token not in sample_data_by_token:
            raise KeyError(f"Unknown previous sample_data token: {previous_token!r}")
        record = sample_data_by_token[previous_token]
        if not isinstance(record, Mapping):
            raise ValueError(f"sample_data record {previous_token!r} must be a mapping")

    return sweeps
