#!/usr/bin/env python3
"""Evaluate LiDAR-RADAR complementarity from saved prediction PKLs."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.omnihd.evaluate_detection import (  # noqa: E402
    CLASSES,
    _match_one_class,
    _numeric_array,
    _prediction_classes,
    _sample_token,
    _validate_boxes,
    _validated_gt_arrays,
    _validated_prediction_arrays,
    load_pickle,
)


MAX_BOXES_PER_SAMPLE = 500
CLASS_RANGE = {class_name: [60.0, 40.0] for class_name in CLASSES}
DISTANCE_BINS = (
    ("0-25", 0.0, 25.0),
    ("25-50", 25.0, 50.0),
    ("50-75", 50.0, 75.0),
)


class _ProtocolConfig:
    class_range = CLASS_RANGE


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate GT-object-level LiDAR-RADAR complementarity from saved "
            "TruckScenes prediction PKLs without rerunning inference."
        )
    )
    parser.add_argument(
        "--ann-file",
        required=True,
        help="Enriched TruckScenes OmniHD-style annotation PKL.",
    )
    parser.add_argument(
        "--lidar-predictions",
        required=True,
        help="LiDAR prediction PKL produced by run_lidar_inference.py.",
    )
    parser.add_argument(
        "--radar-predictions",
        required=True,
        help="RADAR prediction PKL produced by run_radar_inference.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination complementarity JSON.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Prediction score threshold (default: 0.5).",
    )
    parser.add_argument(
        "--match-distance",
        type=float,
        default=2.0,
        help="Strict XY match distance in metres (default: 2.0).",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    paths = {
        "ann_file": Path(args.ann_file).expanduser().resolve(),
        "lidar_predictions": Path(args.lidar_predictions).expanduser().resolve(),
        "radar_predictions": Path(args.radar_predictions).expanduser().resolve(),
        "output": Path(args.output).expanduser().resolve(),
    }
    for key, description in (
        ("ann_file", "enriched annotation PKL"),
        ("lidar_predictions", "LiDAR prediction PKL"),
        ("radar_predictions", "RADAR prediction PKL"),
    ):
        if not paths[key].is_file():
            raise FileNotFoundError("{} not found: {}".format(description, paths[key]))
    if len({paths["ann_file"], paths["lidar_predictions"], paths["radar_predictions"]}) != 3:
        raise ValueError("Annotation, LiDAR, and RADAR inputs must be distinct files")
    if paths["output"] in {
        paths["ann_file"],
        paths["lidar_predictions"],
        paths["radar_predictions"],
    }:
        raise ValueError("Output must not overwrite an input PKL")
    if paths["output"].exists() and paths["output"].is_dir():
        raise IsADirectoryError("Output path is a directory: {}".format(paths["output"]))
    return paths


def validate_thresholds(score_threshold: float, match_distance: float) -> None:
    if not np.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("--score-threshold must be finite and in [0, 1]")
    if not np.isfinite(match_distance) or match_distance <= 0.0:
        raise ValueError("--match-distance must be finite and greater than zero")


def _index_gt_infos(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    infos = payload["infos"]
    result: Dict[str, Mapping[str, Any]] = {}
    required_fields = (
        "gt_visibility_levels",
        "gt_motion_states",
        "weather",
        "area",
    )
    for index, info in enumerate(infos):
        if not isinstance(info, Mapping):
            raise TypeError("GT info {} must be a mapping".format(index))
        token = _sample_token(info, "GT", index)
        if token in result:
            raise ValueError("Duplicate GT sample token: {}".format(token))
        missing = [field for field in required_fields if field not in info]
        if missing:
            raise KeyError(
                "GT sample {} is not fully enriched; missing {}".format(token, missing)
            )
        object_count = int(np.asarray(info["gt_boxes"]).shape[0])
        if len(info["gt_visibility_levels"]) != object_count:
            raise ValueError("Visibility metadata is misaligned for {}".format(token))
        if len(info["gt_motion_states"]) != object_count:
            raise ValueError("Motion metadata is misaligned for {}".format(token))
        if any(int(value) not in (1, 2, 3, 4) for value in info["gt_visibility_levels"]):
            raise ValueError("Invalid visibility level in sample {}".format(token))
        if set(info["gt_motion_states"]) - {"moving", "stationary", "unknown"}:
            raise ValueError("Invalid motion state in sample {}".format(token))
        for field in ("weather", "area"):
            if not isinstance(info[field], str) or not info[field]:
                raise ValueError("{} must be non-empty for sample {}".format(field, token))
        result[token] = info
    return result


def _validate_prediction_record(record: Mapping[str, Any], token: str) -> None:
    missing = [
        name
        for name in ("boxes_3d", "scores_3d", "labels_3d")
        if name not in record
    ]
    if missing:
        raise KeyError("Prediction sample {} is missing {}".format(token, missing))
    boxes = _validate_boxes(record["boxes_3d"], "prediction boxes for {}".format(token))
    scores = _numeric_array(
        record["scores_3d"], "prediction scores for {}".format(token)
    )
    labels = _numeric_array(
        record["labels_3d"], "prediction labels for {}".format(token)
    )
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("Prediction scores and labels for {} must be 1D".format(token))
    if boxes.shape[0] != scores.shape[0] or scores.shape[0] != labels.shape[0]:
        raise ValueError("Prediction array counts differ for {}".format(token))
    if boxes.shape[0] > MAX_BOXES_PER_SAMPLE:
        raise ValueError(
            "Prediction sample {} has {} boxes; maximum is {}".format(
                token, boxes.shape[0], MAX_BOXES_PER_SAMPLE
            )
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("Prediction scores contain NaN/Inf for {}".format(token))
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Prediction labels must have integer dtype for {}".format(token))
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= len(CLASSES)):
        raise ValueError("Prediction labels must be in [0, 3] for {}".format(token))


def _index_predictions(
    payload: Mapping[str, Any], expected_modality: str
) -> Dict[str, Mapping[str, Any]]:
    _, modality = _prediction_classes(payload)
    if modality.lower() != expected_modality:
        raise ValueError(
            "Expected {} predictions, metadata reports {!r}".format(
                expected_modality, modality
            )
        )
    records = payload["predictions"]
    result: Dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError("Prediction record {} must be a mapping".format(index))
        token = _sample_token(record, "prediction", index)
        if token in result:
            raise ValueError(
                "Duplicate {} prediction token: {}".format(expected_modality, token)
            )
        _validate_prediction_record(record, token)
        result[token] = record
    return result


def _new_bucket() -> Dict[str, int]:
    return {
        "total_gt": 0,
        "both": 0,
        "lidar_only": 0,
        "radar_only": 0,
        "neither": 0,
    }


def _record_flags(bucket: Dict[str, int], lidar_detected: bool, radar_detected: bool) -> None:
    bucket["total_gt"] += 1
    if lidar_detected and radar_detected:
        bucket["both"] += 1
    elif lidar_detected:
        bucket["lidar_only"] += 1
    elif radar_detected:
        bucket["radar_only"] += 1
    else:
        bucket["neither"] += 1


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def _finalize_bucket(bucket: Mapping[str, int]) -> Dict[str, Any]:
    total = int(bucket["total_gt"])
    both = int(bucket["both"])
    lidar_only = int(bucket["lidar_only"])
    radar_only = int(bucket["radar_only"])
    neither = int(bucket["neither"])
    if both + lidar_only + radar_only + neither != total:
        raise RuntimeError("Complementarity categories do not sum to total_gt")
    union_detected = both + lidar_only + radar_only
    return {
        "total_gt": total,
        "both": both,
        "lidar_only": lidar_only,
        "radar_only": radar_only,
        "neither": neither,
        "lidar_misses": radar_only + neither,
        "radar_misses": lidar_only + neither,
        "union_detected": union_detected,
        "lidar_recall": _safe_ratio(both + lidar_only, total),
        "radar_recall": _safe_ratio(both + radar_only, total),
        "union_recall": _safe_ratio(union_detected, total),
        "radar_recovery_rate": _safe_ratio(radar_only, radar_only + neither),
        "lidar_recovery_rate": _safe_ratio(lidar_only, lidar_only + neither),
    }


def _distance_bin(box: np.ndarray) -> str:
    distance = float(np.linalg.norm(box[:2]))
    for name, lower, upper in DISTANCE_BINS:
        if lower <= distance < upper:
            return name
    raise ValueError(
        "Range-filtered GT distance {:.6f} m is outside 0-75 m bins".format(distance)
    )


def _detection_flags(
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    prediction: Mapping[str, Any],
    token: str,
    config: Any,
    score_threshold: float,
    match_distance: float,
) -> np.ndarray:
    prediction_boxes, scores, labels, source_indices = _validated_prediction_arrays(
        prediction, token, config, score_threshold
    )
    detected = np.zeros(gt_boxes.shape[0], dtype=bool)
    for class_index, class_name in enumerate(CLASSES):
        gt_class_indices = np.flatnonzero(gt_names == class_name)
        prediction_class_indices = np.flatnonzero(labels == class_index)
        matches, _, _ = _match_one_class(
            gt_boxes[gt_class_indices],
            prediction_boxes[prediction_class_indices],
            scores[prediction_class_indices],
            source_indices[prediction_class_indices],
            match_distance,
        )
        for _, class_gt_index, _ in matches:
            gt_index = int(gt_class_indices[class_gt_index])
            if detected[gt_index]:
                raise RuntimeError(
                    "GT object matched more than once for {} in {}".format(
                        class_name, token
                    )
                )
            detected[gt_index] = True
    return detected


def calculate_complementarity(
    gt_by_token: Mapping[str, Mapping[str, Any]],
    lidar_by_token: Mapping[str, Mapping[str, Any]],
    radar_by_token: Mapping[str, Mapping[str, Any]],
    score_threshold: float,
    match_distance: float,
) -> Dict[str, Any]:
    config = _ProtocolConfig()
    overall = _new_bucket()
    by_class = {name: _new_bucket() for name in CLASSES}
    by_distance = {name: _new_bucket() for name, _, _ in DISTANCE_BINS}
    by_visibility = {str(level): _new_bucket() for level in (1, 2, 3, 4)}
    by_motion = {
        state: _new_bucket() for state in ("moving", "stationary", "unknown")
    }
    by_weather: Dict[str, Dict[str, int]] = {}
    by_area: Dict[str, Dict[str, int]] = {}
    processed_identities = set()

    for token, info in gt_by_token.items():
        gt_boxes, gt_names, source_indices = _validated_gt_arrays(info, token, config)
        lidar_flags = _detection_flags(
            gt_boxes,
            gt_names,
            lidar_by_token[token],
            token,
            config,
            score_threshold,
            match_distance,
        )
        radar_flags = _detection_flags(
            gt_boxes,
            gt_names,
            radar_by_token[token],
            token,
            config,
            score_threshold,
            match_distance,
        )
        if lidar_flags.shape != radar_flags.shape or lidar_flags.shape[0] != gt_boxes.shape[0]:
            raise RuntimeError("Detection flag arrays are misaligned for {}".format(token))

        weather = info["weather"]
        area = info["area"]
        by_weather.setdefault(weather, _new_bucket())
        by_area.setdefault(area, _new_bucket())
        for filtered_index, source_index in enumerate(source_indices):
            identity = (token, int(source_index))
            if identity in processed_identities:
                raise RuntimeError("GT object processed more than once: {}".format(identity))
            processed_identities.add(identity)
            lidar_detected = bool(lidar_flags[filtered_index])
            radar_detected = bool(radar_flags[filtered_index])
            visibility = str(int(info["gt_visibility_levels"][source_index]))
            motion = info["gt_motion_states"][source_index]
            buckets = (
                overall,
                by_class[str(gt_names[filtered_index])],
                by_distance[_distance_bin(gt_boxes[filtered_index])],
                by_visibility[visibility],
                by_motion[motion],
                by_weather[weather],
                by_area[area],
            )
            for bucket in buckets:
                _record_flags(bucket, lidar_detected, radar_detected)

    finalized_overall = _finalize_bucket(overall)
    if len(processed_identities) != finalized_overall["total_gt"]:
        raise RuntimeError("Not every range-filtered GT object was categorized exactly once")
    return {
        "overall": finalized_overall,
        "class_breakdown": {
            name: _finalize_bucket(bucket) for name, bucket in by_class.items()
        },
        "distance_breakdown": {
            name: _finalize_bucket(bucket) for name, bucket in by_distance.items()
        },
        "visibility_breakdown": {
            name: _finalize_bucket(bucket) for name, bucket in by_visibility.items()
        },
        "motion_breakdown": {
            name: _finalize_bucket(bucket) for name, bucket in by_motion.items()
        },
        "weather_breakdown": {
            name: _finalize_bucket(bucket)
            for name, bucket in sorted(by_weather.items())
        },
        "area_breakdown": {
            name: _finalize_bucket(bucket)
            for name, bucket in sorted(by_area.items())
        },
        "processed_gt_identities": len(processed_identities),
    }


def _validate_token_sets(
    gt_tokens: set, lidar_tokens: set, radar_tokens: set
) -> Dict[str, Any]:
    if gt_tokens != lidar_tokens or gt_tokens != radar_tokens:
        raise ValueError(
            "GT/LiDAR/RADAR sample-token sets differ: missing LiDAR={} extra "
            "LiDAR={} missing RADAR={} extra RADAR={}".format(
                sorted(gt_tokens - lidar_tokens)[:5],
                sorted(lidar_tokens - gt_tokens)[:5],
                sorted(gt_tokens - radar_tokens)[:5],
                sorted(radar_tokens - gt_tokens)[:5],
            )
        )
    return {
        "sample_token_sets_equal": True,
        "sample_count": len(gt_tokens),
    }


def _validate_overall_metrics(
    overall: Mapping[str, Any], processed_gt_identities: int
) -> Dict[str, bool]:
    """Validate complementarity counts and recalls from the current run."""
    total = int(overall["total_gt"])
    both = int(overall["both"])
    lidar_only = int(overall["lidar_only"])
    radar_only = int(overall["radar_only"])
    neither = int(overall["neither"])

    def ratio_matches(field: str, numerator: int, denominator: int) -> bool:
        expected = _safe_ratio(numerator, denominator)
        actual = overall[field]
        if expected is None:
            return actual is None
        return actual is not None and bool(np.isclose(float(actual), expected, atol=1e-12, rtol=0.0))

    checks = {
        "every_gt_categorized_once": processed_gt_identities == total,
        "category_sum_matches_total": both + lidar_only + radar_only + neither == total,
        "lidar_tp_consistent": both + lidar_only == total - int(overall["lidar_misses"]),
        "radar_tp_consistent": both + radar_only == total - int(overall["radar_misses"]),
        "lidar_misses_consistent": int(overall["lidar_misses"]) == radar_only + neither,
        "radar_misses_consistent": int(overall["radar_misses"]) == lidar_only + neither,
        "union_detected_consistent": int(overall["union_detected"])
        == both + lidar_only + radar_only,
        "lidar_recall_consistent": ratio_matches("lidar_recall", both + lidar_only, total),
        "radar_recall_consistent": ratio_matches("radar_recall", both + radar_only, total),
        "union_recall_consistent": ratio_matches(
            "union_recall", both + lidar_only + radar_only, total
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Complementarity validation failed: {}".format(", ".join(failed))
        )
    return checks


def build_result(
    paths: Mapping[str, Path],
    args: argparse.Namespace,
    metrics: Mapping[str, Any],
    token_validation: Mapping[str, Any],
) -> Dict[str, Any]:
    validation = _validate_overall_metrics(
        metrics["overall"], int(metrics["processed_gt_identities"])
    )
    return {
        "protocol": {
            "classes": list(CLASSES),
            "score_threshold": float(args.score_threshold),
            "score_comparison": "prediction score >= score_threshold",
            "match_distance_m": float(args.match_distance),
            "match_comparison": "XY center distance < match_distance_m",
            "class_range": CLASS_RANGE,
            "matching": (
                "LiDAR and RADAR are matched independently within each sample and "
                "class. Predictions are sorted by descending score with original "
                "prediction index as tie-break, then matched to the nearest unmatched "
                "GT with GT index as equal-distance tie-break."
            ),
            "gt_identity": "(sample_token, original GT object index)",
            "distance_bins_m": [name for name, _, _ in DISTANCE_BINS],
        },
        "inputs": {
            "ann_file": str(paths["ann_file"]),
            "lidar_predictions": str(paths["lidar_predictions"]),
            "radar_predictions": str(paths["radar_predictions"]),
        },
        "overall": metrics["overall"],
        "class_breakdown": metrics["class_breakdown"],
        "distance_breakdown": metrics["distance_breakdown"],
        "visibility_breakdown": metrics["visibility_breakdown"],
        "motion_breakdown": metrics["motion_breakdown"],
        "weather_breakdown": metrics["weather_breakdown"],
        "area_breakdown": metrics["area_breakdown"],
        "validation": {
            **dict(token_validation),
            "processed_gt_identities": int(metrics["processed_gt_identities"]),
            **validation,
        },
    }


def save_json(result: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("{}.tmp".format(output.name))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def print_summary(overall: Mapping[str, Any], output: Path) -> None:
    print("Total GT: {}".format(overall["total_gt"]))
    print("Both: {}".format(overall["both"]))
    print("LiDAR only: {}".format(overall["lidar_only"]))
    print("RADAR only: {}".format(overall["radar_only"]))
    print("Neither: {}".format(overall["neither"]))
    print("LiDAR recall: {}".format(overall["lidar_recall"]))
    print("RADAR recall: {}".format(overall["radar_recall"]))
    print("Union recall: {}".format(overall["union_recall"]))
    print("RADAR recovery rate: {}".format(overall["radar_recovery_rate"]))
    print("LiDAR recovery rate: {}".format(overall["lidar_recovery_rate"]))
    print("Output JSON: {}".format(output))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    validate_thresholds(args.score_threshold, args.match_distance)
    paths = resolve_paths(args)
    gt_payload = load_pickle(paths["ann_file"], "infos", "annotation pickle")
    lidar_payload = load_pickle(
        paths["lidar_predictions"], "predictions", "LiDAR prediction pickle"
    )
    radar_payload = load_pickle(
        paths["radar_predictions"], "predictions", "RADAR prediction pickle"
    )
    gt_by_token = _index_gt_infos(gt_payload)
    lidar_by_token = _index_predictions(lidar_payload, "lidar")
    radar_by_token = _index_predictions(radar_payload, "radar")
    token_validation = _validate_token_sets(
        set(gt_by_token), set(lidar_by_token), set(radar_by_token)
    )
    metrics = calculate_complementarity(
        gt_by_token,
        lidar_by_token,
        radar_by_token,
        float(args.score_threshold),
        float(args.match_distance),
    )
    result = build_result(paths, args, metrics, token_validation)
    save_json(result, paths["output"])
    print_summary(result["overall"], paths["output"])
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
