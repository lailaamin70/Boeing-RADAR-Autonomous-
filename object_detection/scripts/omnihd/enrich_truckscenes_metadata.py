#!/usr/bin/env python3
"""Enrich an existing TruckScenes-to-OmniHD annotation PKL with metadata."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np


MOVING_ATTRIBUTES = {"vehicle.moving", "pedestrian.moving"}
STATIONARY_ATTRIBUTES = {
    "vehicle.stopped",
    "vehicle.parked",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
}
WEATHER_DETAIL_FIELDS = (
    "precipitation",
    "temperature",
    "wind_speed",
    "wind_direction",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich an existing converted TruckScenes OmniHD annotation PKL with "
            "visibility, motion, weather, and area metadata."
        )
    )
    parser.add_argument(
        "--ann-file",
        required=True,
        help="Existing converted TruckScenes OmniHD annotation PKL.",
    )
    parser.add_argument(
        "--metadata-root",
        required=True,
        help="TruckScenes metadata directory containing the trainval JSON tables.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Destination enriched PKL. Defaults to <ann-file stem>_enriched.pkl "
            "beside the source file."
        ),
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    ann_file = Path(args.ann_file).expanduser().resolve()
    metadata_root = Path(args.metadata_root).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else ann_file.with_name("{}_enriched.pkl".format(ann_file.stem))
    )
    if not ann_file.is_file():
        raise FileNotFoundError("Annotation PKL not found: {}".format(ann_file))
    if not metadata_root.is_dir():
        raise FileNotFoundError(
            "TruckScenes metadata directory not found: {}".format(metadata_root)
        )
    if output == ann_file:
        raise ValueError("--output must not overwrite --ann-file")
    if output.exists() and output.is_dir():
        raise IsADirectoryError("Output path is a directory: {}".format(output))
    return {
        "ann_file": ann_file,
        "metadata_root": metadata_root,
        "output": output,
    }


def _load_json_table(metadata_root: Path, name: str) -> List[Dict[str, Any]]:
    path = metadata_root / "{}.json".format(name)
    if not path.is_file():
        raise FileNotFoundError("Required TruckScenes table not found: {}".format(path))
    try:
        with path.open("r", encoding="utf-8") as stream:
            records = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read TruckScenes table {}: {}".format(path, exc)) from exc
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("{} must contain a JSON list of objects".format(path))
    return records


def _index_by_token(records: Iterable[Mapping[str, Any]], name: str) -> Dict[str, Mapping]:
    result: Dict[str, Mapping] = {}
    for index, record in enumerate(records):
        token = record.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError("{} record {} has an invalid token".format(name, index))
        if token in result:
            raise ValueError("Duplicate {} token: {}".format(name, token))
        result[token] = record
    return result


def _iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Mapping]:
    """Stream objects from a top-level JSON array without loading the full table."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    eof = False
    with path.open("r", encoding="utf-8") as stream:
        while True:
            if position >= chunk_size:
                buffer = buffer[position:]
                position = 0
            if not eof and len(buffer) - position < chunk_size:
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError("JSON array is empty or incomplete: {}".format(path))
                    continue
                if buffer[position] != "[":
                    raise ValueError("Expected a top-level JSON array in {}".format(path))
                position += 1
                started = True

            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer):
                if eof:
                    raise ValueError("Incomplete JSON array: {}".format(path))
                continue

            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError("Invalid JSON in {}: {}".format(path, exc)) from exc
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                    continue
                eof = True
                continue
            if not isinstance(value, Mapping):
                raise ValueError("{} must contain only JSON objects".format(path))
            position = end
            yield value


def _load_selected_annotations(
    path: Path, required_tokens: Mapping[str, str]
) -> Dict[str, Mapping]:
    """Load exact annotation records needed by the converted PKL."""
    if not path.is_file():
        raise FileNotFoundError(
            "Required TruckScenes table not found: {}".format(path)
        )
    selected: Dict[str, Mapping] = {}
    for record in _iter_json_array(path):
        token = record.get("token")
        if token not in required_tokens:
            continue
        if token in selected:
            raise ValueError("Duplicate sample_annotation token: {}".format(token))
        selected[token] = record
        if len(selected) == len(required_tokens):
            break
    missing = sorted(set(required_tokens) - set(selected))
    if missing:
        raise KeyError(
            "Converted GT references {} annotation tokens absent from {}: {}{}".format(
                len(missing),
                path,
                missing[:5],
                "..." if len(missing) > 5 else "",
            )
        )
    return selected


def _load_pickle(path: Path) -> Dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise ValueError("Cannot load annotation PKL {}: {}".format(path, exc)) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("infos"), list):
        raise TypeError("Annotation PKL must contain a top-level infos list")
    if not payload["infos"]:
        raise ValueError("Annotation PKL contains no sample infos")
    return payload


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _preservation_snapshot(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    snapshot = []
    for index, info in enumerate(payload["infos"]):
        if not isinstance(info, Mapping):
            raise TypeError("GT info {} must be a mapping".format(index))
        token = info.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError("GT info {} has an invalid token".format(index))
        boxes = np.asarray(info.get("gt_boxes"))
        names = np.asarray(info.get("gt_names"))
        if boxes.ndim != 2 or names.ndim != 1 or boxes.shape[0] != names.shape[0]:
            raise ValueError("GT boxes/names are misaligned for sample {}".format(token))
        snapshot.append(
            {
                "token": token,
                "count": int(boxes.shape[0]),
                "boxes": _array_digest(boxes),
                "names": tuple(str(value) for value in names),
                "labels": _array_digest(info.get("gt_labels", [])),
                "valid_flag": _array_digest(info.get("valid_flag", [])),
            }
        )
    if len({item["token"] for item in snapshot}) != len(snapshot):
        raise ValueError("Converted annotation PKL contains duplicate sample tokens")
    return snapshot


def _required_annotation_tokens(payload: Mapping[str, Any]) -> Dict[str, str]:
    required: Dict[str, str] = {}
    for info in payload["infos"]:
        token = info["token"]
        boxes = np.asarray(info["gt_boxes"])
        metadata = info.get("truckscenes_annotation_metadata")
        if not isinstance(metadata, list) or len(metadata) != boxes.shape[0]:
            raise ValueError(
                "Sample {} requires one truckscenes_annotation_metadata record per "
                "GT box".format(token)
            )
        for object_index, item in enumerate(metadata):
            if not isinstance(item, Mapping):
                raise TypeError(
                    "Sample {} object metadata {} must be a mapping".format(
                        token, object_index
                    )
                )
            annotation_token = item.get("annotation_token")
            if not isinstance(annotation_token, str) or not annotation_token:
                raise ValueError(
                    "Sample {} object {} has no annotation_token".format(
                        token, object_index
                    )
                )
            if annotation_token in required:
                raise ValueError(
                    "Converted annotation token appears more than once: {}".format(
                        annotation_token
                    )
                )
            required[annotation_token] = token
    return required


def _parse_scene_tag(description: Any, prefix: str, scene_token: str) -> str:
    if not isinstance(description, str) or not description:
        raise ValueError("Scene {} has no description".format(scene_token))
    values = [
        item[len(prefix) :]
        for item in description.split(";")
        if item.startswith(prefix) and item[len(prefix) :]
    ]
    if len(values) != 1:
        raise ValueError(
            "Scene {} must contain exactly one {} tag; found {}".format(
                scene_token, prefix, values
            )
        )
    return values[0]


def _motion_state(attribute_names: List[str]) -> str:
    if len(attribute_names) != 1:
        return "unknown"
    if attribute_names[0] in MOVING_ATTRIBUTES:
        return "moving"
    if attribute_names[0] in STATIONARY_ATTRIBUTES:
        return "stationary"
    return "unknown"


def enrich_payload(
    payload: Dict[str, Any], metadata_root: Path
) -> Dict[str, Any]:
    """Attach aligned semantic metadata while preserving all existing content."""
    before = _preservation_snapshot(payload)
    required_annotations = _required_annotation_tokens(payload)

    samples = _index_by_token(_load_json_table(metadata_root, "sample"), "sample")
    scenes = _index_by_token(_load_json_table(metadata_root, "scene"), "scene")
    visibility = _index_by_token(
        _load_json_table(metadata_root, "visibility"), "visibility"
    )
    attributes = _index_by_token(
        _load_json_table(metadata_root, "attribute"), "attribute"
    )
    weather_annotations = _index_by_token(
        _load_json_table(metadata_root, "weather_annotation"),
        "weather_annotation",
    )
    annotations = _load_selected_annotations(
        metadata_root / "sample_annotation.json", required_annotations
    )

    visibility_counts: Counter = Counter()
    motion_counts: Counter = Counter()
    weather_sample_counts: Counter = Counter()
    area_sample_counts: Counter = Counter()
    used_scenes = set()

    for info in payload["infos"]:
        sample_token = info["token"]
        if sample_token not in samples:
            raise KeyError("Converted sample token absent from sample.json: {}".format(sample_token))
        sample = samples[sample_token]
        scene_token = sample.get("scene_token")
        if scene_token not in scenes:
            raise KeyError(
                "Sample {} references unknown scene {}".format(sample_token, scene_token)
            )
        scene = scenes[scene_token]
        used_scenes.add(scene_token)
        description = scene.get("description")
        weather = _parse_scene_tag(description, "weather.", scene_token)
        area = _parse_scene_tag(description, "area.", scene_token)
        weather_token = scene.get("weather_annotation_token")
        if weather_token not in weather_annotations:
            raise KeyError(
                "Scene {} references unknown weather annotation {}".format(
                    scene_token, weather_token
                )
            )
        weather_record = weather_annotations[weather_token]
        weather_details = {}
        for field in WEATHER_DETAIL_FIELDS:
            value = weather_record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "Weather annotation {} field {} must be numeric".format(
                        weather_token, field
                    )
                )
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(
                    "Weather annotation {} field {} must be finite".format(
                        weather_token, field
                    )
                )
            weather_details[field] = numeric

        visibility_levels: List[int] = []
        motion_states: List[str] = []
        attribute_names_by_object: List[List[str]] = []
        for object_index, retained in enumerate(
            info["truckscenes_annotation_metadata"]
        ):
            annotation_token = retained["annotation_token"]
            annotation = annotations[annotation_token]
            if annotation.get("sample_token") != sample_token:
                raise ValueError(
                    "Annotation {} belongs to sample {}, expected {}".format(
                        annotation_token,
                        annotation.get("sample_token"),
                        sample_token,
                    )
                )
            visibility_token = annotation.get("visibility_token")
            if retained.get("visibility_token") != visibility_token:
                raise ValueError(
                    "Retained visibility token differs from sample_annotation for {}".format(
                        annotation_token
                    )
                )
            if visibility_token not in visibility:
                raise KeyError(
                    "Annotation {} references unknown visibility {}".format(
                        annotation_token, visibility_token
                    )
                )
            level = visibility[visibility_token].get("level")
            if isinstance(level, bool) or not isinstance(level, int) or level not in (1, 2, 3, 4):
                raise ValueError(
                    "Visibility {} has invalid semantic level {}".format(
                        visibility_token, level
                    )
                )

            attribute_tokens = annotation.get("attribute_tokens") or []
            if retained.get("attribute_tokens") != attribute_tokens:
                raise ValueError(
                    "Retained attribute tokens differ from sample_annotation for {}".format(
                        annotation_token
                    )
                )
            if not isinstance(attribute_tokens, list):
                raise TypeError(
                    "Annotation {} attribute_tokens must be a list".format(
                        annotation_token
                    )
                )
            attribute_names = []
            for attribute_token in attribute_tokens:
                if attribute_token not in attributes:
                    raise KeyError(
                        "Annotation {} references unknown attribute {}".format(
                            annotation_token, attribute_token
                        )
                    )
                name = attributes[attribute_token].get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        "Attribute {} has no semantic name".format(attribute_token)
                    )
                attribute_names.append(name)

            state = _motion_state(attribute_names)
            visibility_levels.append(level)
            motion_states.append(state)
            attribute_names_by_object.append(attribute_names)
            visibility_counts[level] += 1
            motion_counts[state] += 1

        object_count = int(np.asarray(info["gt_boxes"]).shape[0])
        if not (
            len(visibility_levels)
            == len(motion_states)
            == len(attribute_names_by_object)
            == object_count
        ):
            raise RuntimeError(
                "Enriched object metadata is misaligned for sample {}".format(
                    sample_token
                )
            )

        info["gt_visibility_levels"] = visibility_levels
        info["gt_motion_states"] = motion_states
        info["gt_attribute_names"] = attribute_names_by_object
        info["weather"] = weather
        info["area"] = area
        info["scene_description"] = description
        info["weather_details"] = weather_details
        weather_sample_counts[weather] += 1
        area_sample_counts[area] += 1

    after = _preservation_snapshot(payload)
    if before != after:
        raise RuntimeError(
            "Metadata enrichment changed sample order or existing GT content"
        )

    summary = {
        "format_version": 1,
        "metadata_root": str(metadata_root),
        "samples": len(payload["infos"]),
        "scenes": len(used_scenes),
        "objects": len(required_annotations),
        "visibility_level_counts": {
            str(level): int(visibility_counts[level]) for level in (1, 2, 3, 4)
        },
        "motion_state_counts": {
            state: int(motion_counts[state])
            for state in ("moving", "stationary", "unknown")
        },
        "weather_sample_counts": {
            name: int(count) for name, count in sorted(weather_sample_counts.items())
        },
        "area_sample_counts": {
            name: int(count) for name, count in sorted(area_sample_counts.items())
        },
        "joins": {
            "sample": "converted info token -> sample.token",
            "scene": "sample.scene_token -> scene.token",
            "weather_annotation": (
                "scene.weather_annotation_token -> weather_annotation.token"
            ),
            "object": (
                "truckscenes_annotation_metadata.annotation_token -> "
                "sample_annotation.token"
            ),
            "visibility": "sample_annotation.visibility_token -> visibility.token",
            "attributes": "sample_annotation.attribute_tokens -> attribute.token",
        },
    }
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("Annotation PKL metadata must be a dictionary when present")
    metadata["metadata_enrichment"] = summary
    return summary


def _atomic_pickle_dump(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("{}.tmp".format(output.name))
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def print_summary(summary: Mapping[str, Any], output: Path) -> None:
    print("TruckScenes metadata enrichment complete")
    print("----------------------------------------")
    print("Samples: {}".format(summary["samples"]))
    print("Scenes: {}".format(summary["scenes"]))
    print("GT objects: {}".format(summary["objects"]))
    print("Visibility levels: {}".format(summary["visibility_level_counts"]))
    print("Motion states: {}".format(summary["motion_state_counts"]))
    print("Weather categories: {}".format(summary["weather_sample_counts"]))
    print("Area categories: {}".format(summary["area_sample_counts"]))
    print("Enriched PKL: {}".format(output))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    paths = resolve_paths(args)
    payload = _load_pickle(paths["ann_file"])
    summary = enrich_payload(payload, paths["metadata_root"])
    _atomic_pickle_dump(payload, paths["output"])
    print_summary(summary, paths["output"])
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
