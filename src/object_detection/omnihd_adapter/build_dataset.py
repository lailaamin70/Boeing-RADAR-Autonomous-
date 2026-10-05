"""Build an OmniHD-style dataset from explicitly configured TruckScenes data.

The reference sensor record selects the current timestamp and ego pose. LiDAR
points and annotations are written in that reference ego frame, while RADAR
sweeps stay in their source sensor frames for OmniHD's official multi-sweep
loader to compensate and transform at load time.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
import pickle
from typing import Any

import numpy as np

from .annotation_adapter import convert_annotations
from .class_mapping import OMNIHD_CLASSES
from .lidar_adapter import (
    TRUCKSCENES_LIDAR_CHANNELS,
    combine_lidar_sources,
    pack_omnihd_lidar,
    save_omnihd_lidar,
)
from .radar_adapter import (
    TRUCKSCENES_RADAR_CHANNELS,
    adapt_signal_features,
    build_radar_sweep_metadata,
    collect_radar_sweep_records,
    convert_radar_source,
    save_omnihd_radar_raw,
)


RADAR_SLOT_NAMES = (
    "radar_front",
    "radar_left_front",
    "radar_right_front",
    "radar_back",
    "radar_left_back",
    "radar_right_back",
)

_TABLE_NAMES = (
    "sample",
    "sample_data",
    "calibrated_sensor",
    "sensor",
    "ego_pose",
    "ego_motion_cabin",
    "ego_motion_chassis",
    "sample_annotation",
    "instance",
    "category",
    "scene",
)


@dataclass(frozen=True)
class BuildConfig:
    """All experiment choices needed for a reproducible adapter build."""

    data_root: str | Path
    output_root: str | Path
    metadata_version: str
    lidar_channels: tuple[str, ...]
    radar_channels: tuple[str, ...]
    reference_channel: str
    radar_channel_aliases: Mapping[str, str]
    radar_power_mode: str
    radar_snr_fill_value: float
    ego_velocity_source: str
    radar_power_fill_value: float | None = None
    radar_sweeps_num: int = 3
    sample_tokens: tuple[str, ...] | None = None
    scene_tokens: tuple[str, ...] | None = None
    build_all_samples: bool = False
    filter_missing_sensor_files: bool = True
    metadata_path_prefix: str | Path | None = None
    info_filename: str = "truckscenes_omnihd_infos.pkl"
    manifest_filename: str = "build_manifest.json"


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_token_list(values: Sequence[str] | None, name: str) -> None:
    if values is None:
        return
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of tokens or None")
    if not values:
        raise ValueError(f"{name} must not be empty when supplied")
    for value in values:
        _require_non_empty_string(value, f"{name} entry")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate tokens")


def _validate_config(config: BuildConfig) -> None:
    if not isinstance(config, BuildConfig):
        raise TypeError("config must be a BuildConfig")

    _require_non_empty_string(config.metadata_version, "metadata_version")
    _require_non_empty_string(config.reference_channel, "reference_channel")
    if not config.lidar_channels:
        raise ValueError("At least one LiDAR channel must be selected")
    if not config.radar_channels:
        raise ValueError("At least one RADAR channel must be selected")
    if len(set(config.lidar_channels)) != len(config.lidar_channels):
        raise ValueError("lidar_channels must not contain duplicates")
    if len(set(config.radar_channels)) != len(config.radar_channels):
        raise ValueError("radar_channels must not contain duplicates")

    unknown_lidars = set(config.lidar_channels) - set(TRUCKSCENES_LIDAR_CHANNELS)
    if unknown_lidars:
        raise ValueError(f"Unknown TruckScenes LiDAR channels: {sorted(unknown_lidars)}")
    unknown_radars = set(config.radar_channels) - set(TRUCKSCENES_RADAR_CHANNELS)
    if unknown_radars:
        raise ValueError(f"Unknown TruckScenes RADAR channels: {sorted(unknown_radars)}")

    if not isinstance(config.radar_channel_aliases, Mapping):
        raise ValueError("radar_channel_aliases must be a mapping")
    alias_keys = set(config.radar_channel_aliases)
    selected_radars = set(config.radar_channels)
    if alias_keys != selected_radars:
        missing = sorted(selected_radars - alias_keys)
        extra = sorted(alias_keys - selected_radars)
        raise ValueError(
            "radar_channel_aliases must have exactly the selected RADAR channels; "
            f"missing={missing}, extra={extra}"
        )
    aliases = list(config.radar_channel_aliases.values())
    if any(alias not in RADAR_SLOT_NAMES for alias in aliases):
        raise ValueError(f"RADAR aliases must be chosen from {RADAR_SLOT_NAMES}")
    if len(set(aliases)) != len(aliases):
        raise ValueError("RADAR aliases must be unique")

    if config.ego_velocity_source not in ("cabin", "chassis"):
        raise ValueError("ego_velocity_source must be 'cabin' or 'chassis'")
    if (
        isinstance(config.radar_sweeps_num, (bool, np.bool_))
        or not isinstance(config.radar_sweeps_num, Integral)
        or config.radar_sweeps_num < 1
    ):
        raise ValueError("radar_sweeps_num must be an integer greater than or equal to 1")

    # The RADAR adapter owns signal-policy validation. An empty feature array
    # checks the configuration without fabricating or converting data.
    adapt_signal_features(
        np.empty((0, 1), dtype=float),
        config.radar_power_mode,
        config.radar_snr_fill_value,
        config.radar_power_fill_value,
    )

    _validate_token_list(config.sample_tokens, "sample_tokens")
    _validate_token_list(config.scene_tokens, "scene_tokens")
    selection_modes = sum(
        (
            config.sample_tokens is not None,
            config.scene_tokens is not None,
            config.build_all_samples is True,
        )
    )
    if selection_modes != 1:
        raise ValueError(
            "Select exactly one of sample_tokens, scene_tokens, or "
            "build_all_samples=True"
        )
    if not isinstance(config.build_all_samples, bool):
        raise ValueError("build_all_samples must be a bool")
    if not isinstance(config.filter_missing_sensor_files, bool):
        raise ValueError("filter_missing_sensor_files must be a bool")

    for filename, name, suffixes in (
        (config.info_filename, "info_filename", (".pkl", ".pickle")),
        (config.manifest_filename, "manifest_filename", (".json",)),
    ):
        value = _require_non_empty_string(filename, name)
        if Path(value).name != value or Path(value).suffix.lower() not in suffixes:
            raise ValueError(f"{name} must be a plain filename ending in {suffixes}")


def load_truckscenes_tables(
    data_root: str | Path, metadata_version: str
) -> dict[str, list[dict[str, Any]]]:
    """Load each required TruckScenes JSON table exactly once."""
    root = Path(data_root)
    version = _require_non_empty_string(metadata_version, "metadata_version")
    metadata_root = root / version
    if not root.is_dir():
        raise FileNotFoundError(f"TruckScenes data root not found: {root}")
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"TruckScenes metadata directory not found: {metadata_root}")

    tables: dict[str, list[dict[str, Any]]] = {}
    for table_name in _TABLE_NAMES:
        path = metadata_root / f"{table_name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Required TruckScenes table not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as stream:
                records = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read TruckScenes table {path}: {exc}") from exc
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise ValueError(f"TruckScenes table {path} must contain a JSON list of objects")
        tables[table_name] = records
    return tables


def _index_by_token(records: Sequence[Mapping], table_name: str) -> dict[str, Mapping]:
    indexed: dict[str, Mapping] = {}
    for index, record in enumerate(records):
        token = record.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError(f"{table_name}[{index}] has an invalid token")
        if token in indexed:
            raise ValueError(f"Duplicate token in {table_name}: {token}")
        indexed[token] = record
    return indexed


def _linked_record(
    record: Mapping, field: str, index: Mapping[str, Mapping], table_name: str
) -> Mapping:
    if field not in record:
        raise KeyError(f"record is missing required field: {field}")
    token = record[field]
    if not isinstance(token, str) or not token:
        raise ValueError(f"{field} must contain a non-empty {table_name} token")
    if token not in index:
        raise KeyError(f"Unknown {table_name} token in {field}: {token!r}")
    return index[token]


def _optional_link(record: Mapping, field: str, index: Mapping[str, Mapping]) -> None:
    if field not in record:
        raise KeyError(f"record is missing required field: {field}")
    token = record[field]
    if token == "":
        return
    if not isinstance(token, str) or token not in index:
        raise KeyError(f"Unknown token in {field}: {token!r}")


def build_table_indexes(tables: Mapping[str, Sequence[Mapping]]) -> dict[str, Any]:
    """Index tables and fail on broken links or ambiguous key-frame records."""
    missing = [name for name in _TABLE_NAMES if name not in tables]
    if missing:
        raise KeyError(f"Missing TruckScenes tables: {', '.join(missing)}")

    indexes: dict[str, Any] = {
        f"{name}_by_token": _index_by_token(tables[name], name)
        for name in _TABLE_NAMES
    }
    samples = indexes["sample_by_token"]
    sample_data = indexes["sample_data_by_token"]
    calibrated = indexes["calibrated_sensor_by_token"]
    sensors = indexes["sensor_by_token"]
    ego_poses = indexes["ego_pose_by_token"]
    cabin_motion = indexes["ego_motion_cabin_by_token"]
    chassis_motion = indexes["ego_motion_chassis_by_token"]
    annotations = indexes["sample_annotation_by_token"]
    instances = indexes["instance_by_token"]
    categories = indexes["category_by_token"]
    scenes = indexes["scene_by_token"]

    current_by_sample_channel: dict[str, dict[str, Mapping]] = defaultdict(dict)
    channel_by_sample_data_token: dict[str, str] = {}
    for record in sample_data.values():
        _linked_record(record, "sample_token", samples, "sample")
        _linked_record(record, "ego_pose_token", ego_poses, "ego_pose")
        calibration = _linked_record(
            record, "calibrated_sensor_token", calibrated, "calibrated_sensor"
        )
        sensor = _linked_record(calibration, "sensor_token", sensors, "sensor")
        channel = _require_non_empty_string(sensor.get("channel"), "sensor channel")
        channel_by_sample_data_token[record["token"]] = channel
        _optional_link(record, "prev", sample_data)
        _optional_link(record, "next", sample_data)
        _linked_record(
            record, "ego_motion_cabin_token", cabin_motion, "ego_motion_cabin"
        )
        _linked_record(
            record, "ego_motion_chassis_token", chassis_motion, "ego_motion_chassis"
        )
        is_key_frame = record.get("is_key_frame")
        if not isinstance(is_key_frame, bool):
            raise ValueError("sample_data is_key_frame must be a bool")
        if is_key_frame:
            sample_token = record["sample_token"]
            if channel in current_by_sample_channel[sample_token]:
                raise ValueError(
                    f"Multiple key-frame sample_data records for sample {sample_token} "
                    f"and channel {channel}"
                )
            current_by_sample_channel[sample_token][channel] = record

    annotations_by_sample: dict[str, list[Mapping]] = defaultdict(list)
    for annotation in annotations.values():
        sample = _linked_record(annotation, "sample_token", samples, "sample")
        _linked_record(annotation, "instance_token", instances, "instance")
        annotations_by_sample[sample["token"]].append(annotation)

    category_name_by_instance: dict[str, str] = {}
    for instance in instances.values():
        category = _linked_record(instance, "category_token", categories, "category")
        category_name_by_instance[instance["token"]] = _require_non_empty_string(
            category.get("name"), "category name"
        )

    for sample in samples.values():
        _linked_record(sample, "scene_token", scenes, "scene")
        _linked_record(sample, "ego_pose_token", ego_poses, "ego_pose")
        _optional_link(sample, "prev", samples)
        _optional_link(sample, "next", samples)
        _linked_record(sample, "ego_motion_cabin_token", cabin_motion, "ego_motion_cabin")
        _linked_record(
            sample, "ego_motion_chassis_token", chassis_motion, "ego_motion_chassis"
        )

    indexes["current_sample_data_by_sample_channel"] = dict(current_by_sample_channel)
    indexes["channel_by_sample_data_token"] = channel_by_sample_data_token
    indexes["annotations_by_sample_token"] = dict(annotations_by_sample)
    indexes["category_name_by_instance_token"] = category_name_by_instance
    return indexes


def _sample_token(sample: Mapping) -> str:
    return _require_non_empty_string(sample.get("token"), "sample token")


def _current_sample_data(
    sample: Mapping, channel: str, indexes: Mapping[str, Any]
) -> Mapping:
    sample_token = _sample_token(sample)
    records = indexes["current_sample_data_by_sample_channel"].get(sample_token, {})
    if channel not in records:
        raise KeyError(
            f"No key-frame sample_data for sample {sample_token!r} and channel {channel!r}"
        )
    return records[channel]


def resolve_reference_record(
    sample: Mapping, indexes: Mapping[str, Any], reference_channel: str
) -> tuple[Mapping, Mapping]:
    """Return the reference channel's current record and its ego pose."""
    record = _current_sample_data(sample, reference_channel, indexes)
    pose = _linked_record(
        record, "ego_pose_token", indexes["ego_pose_by_token"], "ego_pose"
    )
    return record, pose


def resolve_ego_velocity(
    sample_data_record: Mapping, indexes: Mapping[str, Any], source: str
) -> np.ndarray:
    """Resolve source-ego velocity as finite ``[vx, vy, vz]``."""
    if source not in ("cabin", "chassis"):
        raise ValueError("source must be 'cabin' or 'chassis'")
    field = f"ego_motion_{source}_token"
    index = indexes[f"ego_motion_{source}_by_token"]
    motion = _linked_record(sample_data_record, field, index, f"ego_motion_{source}")
    try:
        velocity = np.asarray([motion[axis] for axis in ("vx", "vy", "vz")], dtype=float)
    except KeyError as exc:
        raise KeyError(f"ego_motion_{source} record is missing {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ego_motion_{source} velocity must be numeric") from exc
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError(f"ego_motion_{source} velocity must contain three finite values")
    return velocity


def _resolve_source_data_path(data_root: str | Path, filename: object) -> Path:
    relative = Path(_require_non_empty_string(filename, "sample_data filename"))
    if relative.is_absolute():
        raise ValueError(f"sample_data filename must be relative: {relative}")
    root = Path(data_root).resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"sample_data filename escapes data_root: {relative}")
    return path


def _source_data_path(data_root: str | Path, filename: object) -> Path:
    path = _resolve_source_data_path(data_root, filename)
    if not path.is_file():
        raise FileNotFoundError(f"TruckScenes sensor file not found: {path}")
    return path


def _safe_path_component(value: object, name: str) -> str:
    component = _require_non_empty_string(value, name)
    if Path(component).name != component:
        raise ValueError(f"{name} must not contain path separators: {component!r}")
    return component


def _stored_output_path(config: BuildConfig, relative_path: Path) -> str:
    if config.metadata_path_prefix is None:
        return (Path(config.output_root) / relative_path).resolve().as_posix()
    prefix = PurePosixPath(str(config.metadata_path_prefix).replace("\\", "/"))
    relative = PurePosixPath(*relative_path.parts)
    return (prefix / relative).as_posix()


def _calibration_and_pose(
    sample_data_record: Mapping, indexes: Mapping[str, Any]
) -> tuple[Mapping, Mapping]:
    calibration = _linked_record(
        sample_data_record,
        "calibrated_sensor_token",
        indexes["calibrated_sensor_by_token"],
        "calibrated_sensor",
    )
    pose = _linked_record(
        sample_data_record, "ego_pose_token", indexes["ego_pose_by_token"], "ego_pose"
    )
    return calibration, pose


def build_lidar_for_sample(
    sample: Mapping,
    config: BuildConfig,
    indexes: Mapping[str, Any],
    reference_ego_pose: Mapping,
) -> tuple[str, int]:
    """Convert selected current LiDARs and return stored path and point count."""
    sources = []
    for channel in config.lidar_channels:
        record = _current_sample_data(sample, channel, indexes)
        calibration, pose = _calibration_and_pose(record, indexes)
        sources.append(
            {
                "channel": channel,
                "lidar_path": _source_data_path(config.data_root, record.get("filename")),
                "calibrated_sensor": calibration,
                "ego_pose": pose,
            }
        )

    converted = combine_lidar_sources(sources, reference_ego_pose)
    packed = pack_omnihd_lidar(converted)
    token = _safe_path_component(_sample_token(sample), "sample token")
    relative_path = Path("lidar") / f"{token}.bin"
    save_omnihd_lidar(packed, Path(config.output_root) / relative_path)
    return _stored_output_path(config, relative_path), int(packed.shape[0])


def build_radar_for_sample(
    sample: Mapping,
    config: BuildConfig,
    indexes: Mapping[str, Any],
    reference_ego_pose: Mapping,
    converted_radar_tokens: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Write unique raw sweeps and rebuild metadata for this reference frame."""
    radars: dict[str, list[dict[str, Any]]] = {}
    num_new_files = 0
    sample_data_by_token = indexes["sample_data_by_token"]
    channels_by_token = indexes["channel_by_sample_data_token"]

    for source_channel in config.radar_channels:
        current = _current_sample_data(sample, source_channel, indexes)
        sweeps = collect_radar_sweep_records(
            current, sample_data_by_token, config.radar_sweeps_num
        )
        metadata_records = []
        for sweep in sweeps:
            sweep_token = _safe_path_component(sweep.get("token"), "sample_data token")
            actual_channel = channels_by_token.get(sweep_token)
            if actual_channel != source_channel:
                raise ValueError(
                    f"RADAR prev chain changed channel at {sweep_token}: "
                    f"expected {source_channel}, found {actual_channel}"
                )
            relative_path = Path("radar") / source_channel / f"{sweep_token}.bin"
            output_path = Path(config.output_root) / relative_path
            if sweep_token not in converted_radar_tokens:
                converted = convert_radar_source(
                    _source_data_path(config.data_root, sweep.get("filename")),
                    config.radar_power_mode,
                    config.radar_snr_fill_value,
                    config.radar_power_fill_value,
                )
                save_omnihd_radar_raw(converted, output_path)
                converted_radar_tokens.add(sweep_token)
                num_new_files += 1

            calibration, pose = _calibration_and_pose(sweep, indexes)
            metadata = build_radar_sweep_metadata(
                _stored_output_path(config, relative_path),
                sweep.get("timestamp"),
                resolve_ego_velocity(sweep, indexes, config.ego_velocity_source),
                calibration,
                pose,
                reference_ego_pose,
            )
            metadata["source_channel"] = source_channel
            metadata["source_sample_data_token"] = sweep_token
            metadata_records.append(metadata)

        alias = config.radar_channel_aliases[source_channel]
        radars[alias] = metadata_records
    return radars, num_new_files


def build_annotations_for_sample(
    sample: Mapping, indexes: Mapping[str, Any], reference_ego_pose: Mapping
) -> dict[str, Any]:
    """Convert a sample's GT and use NaN for unavailable compatible velocity."""
    sample_token = _sample_token(sample)
    converted = convert_annotations(
        indexes["annotations_by_sample_token"].get(sample_token, []),
        indexes["category_name_by_instance_token"],
        reference_ego_pose,
    )
    count = int(converted["gt_boxes"].shape[0])
    converted["gt_names"] = np.asarray(converted["gt_names"], dtype=str)
    converted["gt_velocity"] = np.full((count, 2), np.nan, dtype=np.float32)
    converted["valid_flag"] = np.ones(count, dtype=bool)
    return converted


def _integer_timestamp(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an integer number of microseconds")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer number of microseconds")
    return int(numeric)


def _pose_field(pose: Mapping, name: str, length: int) -> list[float]:
    if name not in pose:
        raise KeyError(f"ego_pose is missing required field: {name}")
    try:
        values = np.asarray(pose[name], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ego_pose {name} must be numeric") from exc
    if values.shape != (length,) or not np.all(np.isfinite(values)):
        raise ValueError(f"ego_pose {name} must contain {length} finite values")
    return values.tolist()


def build_sample_info(
    sample: Mapping,
    config: BuildConfig,
    indexes: Mapping[str, Any],
    converted_radar_tokens: set[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build all files and one OmniHD info record for a TruckScenes sample."""
    reference_record, reference_pose = resolve_reference_record(
        sample, indexes, config.reference_channel
    )
    lidar_path, lidar_points = build_lidar_for_sample(
        sample, config, indexes, reference_pose
    )
    radars, new_radar_files = build_radar_for_sample(
        sample, config, indexes, reference_pose, converted_radar_tokens
    )
    annotations = build_annotations_for_sample(sample, indexes, reference_pose)

    info = {
        "token": _sample_token(sample),
        "scene_token": sample["scene_token"],
        "prev": sample.get("prev", ""),
        "next": sample.get("next", ""),
        "lidar_path": lidar_path,
        "sweeps": [],
        "timestamp": _integer_timestamp(reference_record.get("timestamp"), "timestamp"),
        "radars": radars,
        "lidar2ego_translation": [0.0, 0.0, 0.0],
        "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
        "ego2global_translation": _pose_field(reference_pose, "translation", 3),
        "ego2global_rotation": _pose_field(reference_pose, "rotation", 4),
        "reference_channel": config.reference_channel,
        "reference_sample_data_token": reference_record["token"],
        "reference_ego_pose_token": reference_record["ego_pose_token"],
        "gt_boxes": annotations["gt_boxes"],
        "gt_names": annotations["gt_names"],
        "gt_labels": annotations["gt_labels"],
        "gt_velocity": annotations["gt_velocity"],
        "valid_flag": annotations["valid_flag"],
        "truckscenes_annotation_metadata": annotations["annotations"],
    }
    counts = {
        "lidar_points": lidar_points,
        "new_radar_files": new_radar_files,
        "input_annotations": annotations["num_input_annotations"],
        "included_annotations": annotations["num_included_annotations"],
        "excluded_annotations": annotations["num_excluded_annotations"],
    }
    return info, counts


def _select_samples(config: BuildConfig, indexes: Mapping[str, Any]) -> list[Mapping]:
    samples_by_token = indexes["sample_by_token"]
    scenes_by_token = indexes["scene_by_token"]

    if config.sample_tokens is not None:
        unknown = [token for token in config.sample_tokens if token not in samples_by_token]
        if unknown:
            raise KeyError(f"Unknown TruckScenes sample tokens: {unknown}")
        selected = [samples_by_token[token] for token in config.sample_tokens]
    elif config.scene_tokens is not None:
        unknown = [token for token in config.scene_tokens if token not in scenes_by_token]
        if unknown:
            raise KeyError(f"Unknown TruckScenes scene tokens: {unknown}")
        requested = set(config.scene_tokens)
        selected = [
            sample for sample in samples_by_token.values() if sample["scene_token"] in requested
        ]
    else:
        selected = list(samples_by_token.values())

    if not selected:
        raise ValueError("The configured sample selection produced no samples")
    return sorted(
        selected,
        key=lambda sample: (
            _integer_timestamp(sample.get("timestamp"), "sample timestamp"),
            _sample_token(sample),
        ),
    )


def _required_sensor_records(
    sample: Mapping, config: BuildConfig, indexes: Mapping[str, Any]
) -> list[Mapping]:
    """Return every LiDAR/RADAR file record consumed for one output sample."""
    records = [
        _current_sample_data(sample, channel, indexes)
        for channel in config.lidar_channels
    ]
    sample_data_by_token = indexes["sample_data_by_token"]
    for channel in config.radar_channels:
        current = _current_sample_data(sample, channel, indexes)
        records.extend(
            collect_radar_sweep_records(
                current, sample_data_by_token, config.radar_sweeps_num
            )
        )
    return records


def _filter_samples_by_local_sensor_files(
    samples: Sequence[Mapping], config: BuildConfig, indexes: Mapping[str, Any]
) -> tuple[list[Mapping], dict[str, Any]]:
    """Keep only samples whose complete configured sensor inputs are local."""
    if not config.filter_missing_sensor_files:
        summary = {
            "enabled": False,
            "metadata_scenes": len(indexes["scene_by_token"]),
            "metadata_samples": len(indexes["sample_by_token"]),
            "selected_samples": len(samples),
            "skipped_missing_sensor_files": 0,
            "usable_samples": len(samples),
            "usable_scenes": len({sample["scene_token"] for sample in samples}),
            "missing_file_references": 0,
            "unique_missing_files": 0,
            "missing_file_examples": [],
        }
        return list(samples), summary

    availability_cache: dict[str, bool] = {}
    usable_samples = []
    usable_scene_tokens = set()
    unique_missing_files = set()
    missing_file_examples = []
    missing_file_references = 0

    for sample in samples:
        sample_has_missing_file = False
        checked_tokens = set()
        for record in _required_sensor_records(sample, config, indexes):
            record_token = _require_non_empty_string(
                record.get("token"), "sample_data token"
            )
            if record_token in checked_tokens:
                continue
            checked_tokens.add(record_token)

            filename = _require_non_empty_string(
                record.get("filename"), "sample_data filename"
            )
            if filename not in availability_cache:
                path = _resolve_source_data_path(config.data_root, filename)
                availability_cache[filename] = path.is_file()
            if availability_cache[filename]:
                continue

            sample_has_missing_file = True
            missing_file_references += 1
            if filename not in unique_missing_files:
                unique_missing_files.add(filename)
                if len(missing_file_examples) < 5:
                    missing_file_examples.append(filename)

        if not sample_has_missing_file:
            usable_samples.append(sample)
            usable_scene_tokens.add(sample["scene_token"])

    summary = {
        "enabled": True,
        "metadata_scenes": len(indexes["scene_by_token"]),
        "metadata_samples": len(indexes["sample_by_token"]),
        "selected_samples": len(samples),
        "skipped_missing_sensor_files": len(samples) - len(usable_samples),
        "usable_samples": len(usable_samples),
        "usable_scenes": len(usable_scene_tokens),
        "missing_file_references": missing_file_references,
        "unique_missing_files": len(unique_missing_files),
        "missing_file_examples": missing_file_examples,
    }
    return usable_samples, summary


def _print_availability_summary(summary: Mapping[str, Any]) -> None:
    """Print the trainval metadata and local sensor-file selection counts."""
    print("TruckScenes metadata and local-file availability")
    print("------------------------------------------------")
    print("Metadata scenes: {}".format(summary["metadata_scenes"]))
    print("Metadata samples: {}".format(summary["metadata_samples"]))
    print("Samples requested: {}".format(summary["selected_samples"]))
    print(
        "Samples skipped for missing required sensor files: {}".format(
            summary["skipped_missing_sensor_files"]
        )
    )
    print("Usable local samples selected: {}".format(summary["usable_samples"]))
    print("Usable scenes represented: {}".format(summary["usable_scenes"]))
    if summary["missing_file_examples"]:
        print("Missing sensor-file examples:")
        for filename in summary["missing_file_examples"]:
            print("  {}".format(filename))
    print(flush=True)


def _selection_manifest(config: BuildConfig) -> dict[str, Any]:
    if config.sample_tokens is not None:
        return {"mode": "sample_tokens", "requested_count": len(config.sample_tokens)}
    if config.scene_tokens is not None:
        return {"mode": "scene_tokens", "scene_tokens": list(config.scene_tokens)}
    return {"mode": "all_samples"}


def build_dataset(config: BuildConfig) -> dict[str, Any]:
    """Build files, info pickle, and manifest; return their paths and counts.

    ``gt_velocity`` is intentionally all-NaN because the available trajectory
    utility does not provide a clean per-sample, reference-ego-frame Nx2 value.
    Consume this pickle with OmniHD ``with_velocity=False``.
    """
    _validate_config(config)
    tables = load_truckscenes_tables(config.data_root, config.metadata_version)
    indexes = build_table_indexes(tables)
    selected_samples = _select_samples(config, indexes)
    selected_samples, availability = _filter_samples_by_local_sensor_files(
        selected_samples, config, indexes
    )
    _print_availability_summary(availability)
    if not selected_samples:
        raise ValueError(
            "No selected TruckScenes samples have all required local LiDAR and "
            "RADAR files"
        )

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    converted_radar_tokens: set[str] = set()
    infos = []
    totals = {
        "samples": 0,
        "lidar_files": 0,
        "lidar_points": 0,
        "radar_files": 0,
        "gt_boxes": 0,
        "input_annotations": 0,
        "included_annotations": 0,
        "excluded_annotations": 0,
    }
    for sample in selected_samples:
        info, counts = build_sample_info(
            sample, config, indexes, converted_radar_tokens
        )
        infos.append(info)
        totals["samples"] += 1
        totals["lidar_files"] += 1
        totals["lidar_points"] += counts["lidar_points"]
        totals["radar_files"] += counts["new_radar_files"]
        totals["gt_boxes"] += counts["included_annotations"]
        totals["input_annotations"] += counts["input_annotations"]
        totals["included_annotations"] += counts["included_annotations"]
        totals["excluded_annotations"] += counts["excluded_annotations"]

    metadata = {
        "version": "truckscenes-omnihd-transfer",
        "source_dataset": "TruckScenes",
        "source_metadata_version": config.metadata_version,
        "target_format": "OmniHD/NewScenes",
        "classes": OMNIHD_CLASSES,
        "reference_channel": config.reference_channel,
        "lidar_channels": tuple(config.lidar_channels),
        "radar_channels": tuple(config.radar_channels),
        "radar_channel_aliases": dict(config.radar_channel_aliases),
        "radar_sweeps_num": int(config.radar_sweeps_num),
        "radar_power_mode": config.radar_power_mode,
        "radar_snr_fill_value": float(config.radar_snr_fill_value),
        "radar_power_fill_value": config.radar_power_fill_value,
        "ego_velocity_source": config.ego_velocity_source,
        "source_availability": availability,
        "with_velocity": False,
    }
    payload = {"infos": infos, "metadata": metadata}
    info_path = output_root / config.info_filename
    with info_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    limitations = [
        "gt_velocity is NaN; configure NewScenesDataset with with_velocity=False.",
        "RADAR aliases name OmniHD loader slots only and do not claim "
        "sensor-position equivalence.",
        "TruckScenes RCS is only an OmniHD power proxy when radar_power_mode is rcs_proxy.",
        "TruckScenes has no compatible SNR field, so the configured constant is used.",
        "LiDAR intensity is preserved without cross-dataset calibration.",
    ]
    manifest = {
        "format_version": "truckscenes-omnihd-transfer",
        "source_dataset": "TruckScenes",
        "source_metadata_version": config.metadata_version,
        "target_format": "OmniHD/NewScenes",
        "info_file": info_path.resolve().as_posix(),
        "path_policy": (
            "resolved_local_paths"
            if config.metadata_path_prefix is None
            else "metadata_path_prefix_plus_output_relative_path"
        ),
        "metadata_path_prefix": (
            None
            if config.metadata_path_prefix is None
            else str(config.metadata_path_prefix).replace("\\", "/")
        ),
        "selection": _selection_manifest(config),
        "source_availability": availability,
        "reference_channel": config.reference_channel,
        "lidar_channels": list(config.lidar_channels),
        "radar_channels": list(config.radar_channels),
        "radar_channel_aliases": dict(config.radar_channel_aliases),
        "radar_sweeps_num": int(config.radar_sweeps_num),
        "radar_signal_policy": {
            "power_mode": config.radar_power_mode,
            "power_fill_value": config.radar_power_fill_value,
            "snr_fill_value": float(config.radar_snr_fill_value),
        },
        "ego_velocity_source": config.ego_velocity_source,
        "gt_velocity_policy": {
            "stored_value": "NaN",
            "required_dataset_setting": "with_velocity=False",
        },
        "valid_flag_policy": (
            "True for annotations retained by the transfer class mapping; not a "
            "camera-visibility or sensor-detection claim."
        ),
        "counts": totals,
        "limitations": limitations,
    }
    manifest_path = output_root / config.manifest_filename
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    return {
        "info_path": info_path,
        "manifest_path": manifest_path,
        "counts": totals,
        "availability": availability,
    }
