"""Convert caller-selected TruckScenes LiDARs for OmniHD LiDAR PointPillars.

Intensity is preserved without normalization; cross-dataset intensity calibration
and point-density changes from combining sensors remain experiment limitations.
XYZ filtering is left to OmniHD. No channels or reference frame are selected here.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from pypcd4 import PointCloud

from .transforms import sensor_to_sensor_transform, transform_points


TRUCKSCENES_LIDAR_CHANNELS = (
    "LIDAR_LEFT",
    "LIDAR_RIGHT",
    "LIDAR_TOP_FRONT",
    "LIDAR_TOP_LEFT",
    "LIDAR_TOP_RIGHT",
    "LIDAR_REAR",
)

_LIDAR_FEATURE_FIELDS = ("x", "y", "z", "intensity")


def _point_array(
    values: ArrayLike, columns: int, name: str, dtype: type = np.float64
) -> np.ndarray:
    """Validate a finite numeric point array, including after dtype conversion."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric (N, {columns}) array") from exc
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values representable as {array.dtype}")
    return array


def load_truckscenes_lidar(path: str | Path) -> np.ndarray:
    """Load finite Nx4 float64 ``[x, y, z, intensity]`` from a LiDAR PCD.

    Reuses pypcd4, the TruckScenes devkit's reader for binary and
    binary_compressed PCD (also supports ASCII). Required fields are selected by
    name; timestamps and other fields are excluded. Malformed clouds raise
    ValueError rather than dropping points or fabricating intensity.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LiDAR PCD file not found: {path}")
    if path.suffix.lower() != ".pcd":
        raise ValueError(f"Expected a LiDAR .pcd file: {path}")

    try:
        cloud = PointCloud.from_path(path)
    except Exception as exc:
        raise ValueError(f"Cannot read LiDAR PCD {path}: {exc}") from exc

    data = cloud.pc_data
    missing = [name for name in _LIDAR_FEATURE_FIELDS if name not in (data.dtype.names or ())]
    if missing:
        raise ValueError(f"LiDAR PCD {path} is missing required fields: {', '.join(missing)}")
    if data.size != cloud.metadata.points:
        raise ValueError(f"LiDAR PCD {path} point count does not match its header")
    # pypcd4 can return a scalar for one ASCII point or a 2D empty array.
    data = data.reshape(-1)
    features = np.column_stack([data[name] for name in _LIDAR_FEATURE_FIELDS])
    return _point_array(features, 4, f"LiDAR PCD {path}")


def extract_lidar_features(points: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Split loaded Nx4 points into Nx3 XYZ and Nx1 unscaled intensity."""
    points = _point_array(points, 4, "LiDAR features")
    return points[:, :3].copy(), points[:, 3:4].copy()


def convert_lidar_points(
    xyz: ArrayLike,
    intensity: ArrayLike,
    source_calibrated_sensor: dict,
    source_ego_pose: dict,
    reference_calibrated_sensor: dict,
    reference_ego_pose: dict,
) -> np.ndarray:
    """Return Nx4 reference-frame XYZ and unchanged Nx1 source intensity.

    Source and reference poses may have different timestamps. Inputs are not
    modified. No spatial filtering or per-point motion compensation is applied.
    """
    xyz = _point_array(xyz, 3, "xyz")
    intensity = _point_array(intensity, 1, "intensity")
    if xyz.shape[0] != intensity.shape[0]:
        raise ValueError("xyz and intensity must have the same number of points")

    T_reference_from_source = sensor_to_sensor_transform(
        source_calibrated_sensor,
        source_ego_pose,
        reference_calibrated_sensor,
        reference_ego_pose,
    )
    xyz_reference = transform_points(xyz, T_reference_from_source)
    return _point_array(
        np.concatenate((xyz_reference, intensity), axis=1), 4, "converted points"
    )


def convert_lidar_source(
    lidar_path: str | Path,
    source_calibrated_sensor: dict,
    source_ego_pose: dict,
    reference_calibrated_sensor: dict,
    reference_ego_pose: dict,
) -> np.ndarray:
    """Load one PCD and return Nx4 features in the explicit reference frame."""
    xyz, intensity = extract_lidar_features(load_truckscenes_lidar(lidar_path))
    return convert_lidar_points(
        xyz,
        intensity,
        source_calibrated_sensor,
        source_ego_pose,
        reference_calibrated_sensor,
        reference_ego_pose,
    )


def combine_lidar_sources(
    sources: Sequence[dict],
    reference_calibrated_sensor: dict,
    reference_ego_pose: dict,
) -> np.ndarray:
    """Combine exactly the supplied sources into reference-frame Mx4 features.

    Each source mapping requires ``channel``, ``lidar_path``,
    ``calibrated_sensor``, and ``ego_pose``. Supply a non-empty sequence with
    unique known LiDAR channels. Source order and point order are preserved.
    The reference calibration and ego pose are mandatory caller choices.
    """
    if len(sources) == 0:
        raise ValueError("At least one explicit LiDAR source is required")
    required = ("channel", "lidar_path", "calibrated_sensor", "ego_pose")
    seen_channels = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"LiDAR source {index} must be a mapping")
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"LiDAR source {index} is missing: {', '.join(missing)}")
        channel = source["channel"]
        if not isinstance(channel, str) or channel not in TRUCKSCENES_LIDAR_CHANNELS:
            raise ValueError(f"Unknown TruckScenes LiDAR channel: {channel!r}")
        if channel in seen_channels:
            raise ValueError(f"Duplicate LiDAR channel: {channel}")
        seen_channels.add(channel)
        if not isinstance(source["lidar_path"], (str, Path)):
            raise ValueError(f"{channel}: lidar_path must be a string or Path")
        for key in ("calibrated_sensor", "ego_pose"):
            if not isinstance(source[key], Mapping):
                raise ValueError(f"{channel}: {key} must be a metadata mapping")

    clouds = [
        convert_lidar_source(
            source["lidar_path"],
            source["calibrated_sensor"],
            source["ego_pose"],
            reference_calibrated_sensor,
            reference_ego_pose,
        )
        for source in sources
    ]
    return np.concatenate(clouds, axis=0)


def pack_omnihd_lidar(points: ArrayLike) -> np.ndarray:
    """Pack Nx4 features as Nx6 float32 ``[x, y, z, intensity, 0, 0]``.

    OmniHD's pointpillars_LiDAR.py uses LoadPointsFromFile with load_dim=6
    and use_dim=4. MMDetection3D v0.17.1 selects columns [0, 1, 2, 3];
    columns 4 and 5 are padding discarded before the model, with no assigned
    semantics. No intensity normalization is configured in this pipeline.
    """
    features = _point_array(points, 4, "converted points", dtype=np.float32)
    packed = np.zeros((features.shape[0], 6), dtype=np.float32)
    packed[:, :4] = features
    return packed


def save_omnihd_lidar(points: ArrayLike, output_path: str | Path) -> None:
    """Write Nx6 points as headerless float32 binary, creating parent folders.

    Compatible with ``np.fromfile(path, dtype=np.float32).reshape(-1, 6)``.
    An existing output file is overwritten after input validation succeeds.
    """
    packed = _point_array(points, 6, "packed points", dtype=np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packed.tofile(output_path)
