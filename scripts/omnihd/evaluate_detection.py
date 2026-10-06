#!/usr/bin/env python3
"""Evaluate transferred OmniHD detections on converted TruckScenes ground truth.

The official AP/mAP path remains separate from optional fixed-operating-point
counts, localization/box errors, and supported condition breakdowns.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CONFIG_NAME = "detection_newsc_config_final"
CLASSES = ("car", "pedestrian", "rider", "large_vehicle")
DISTANCE_BINS = (
    ("0-25", 0.0, 25.0),
    ("25-50", 25.0, 50.0),
    ("50-75", 50.0, 75.0),
    ("75-100", 75.0, 100.0),
    ("100+", 100.0, None),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse evaluator arguments without importing the external OmniHD code."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OmniHD-pretrained PointPillars predictions transferred to "
            "TruckScenes, preserving official center-distance AP/mAP and optionally "
            "adding fixed-operating-point metrics."
        )
    )
    parser.add_argument(
        "--omnihd-root",
        required=True,
        help="Path to the separate OmniHD-Scenes repository.",
    )
    parser.add_argument(
        "--ann-file",
        required=True,
        help="Converted TruckScenes OmniHD-style ground-truth pickle.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction pickle produced by an OmniHD inference script.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSON file for all evaluation metrics.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=(
            "Prediction score threshold for fixed-operating-point metrics. "
            "Supply together with --match-distance."
        ),
    )
    parser.add_argument(
        "--match-distance",
        type=float,
        default=None,
        help=(
            "Strict XY matching distance in metres for fixed-operating-point "
            "metrics. Supply together with --score-threshold."
        ),
    )
    return parser.parse_args(argv)


def validate_operating_point_args(args: argparse.Namespace) -> bool:
    """Validate optional fixed-operating-point arguments as an all-or-none pair."""
    supplied = (args.score_threshold is not None, args.match_distance is not None)
    if supplied == (False, False):
        return False
    if supplied[0] != supplied[1]:
        raise ValueError(
            "--score-threshold and --match-distance must be supplied together"
        )
    if not np.isfinite(args.score_threshold) or not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be finite and in [0, 1]")
    if not np.isfinite(args.match_distance) or args.match_distance <= 0.0:
        raise ValueError("--match-distance must be a finite value greater than zero")
    return True


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """Resolve and validate inputs without changing the working directory."""
    launch_directory = Path.cwd().resolve()

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = launch_directory / path
        return path.resolve()

    paths = {
        "omnihd_root": resolve(args.omnihd_root),
        "ann_file": resolve(args.ann_file),
        "predictions": resolve(args.predictions),
        "output": resolve(args.output),
    }
    if not paths["omnihd_root"].is_dir():
        raise FileNotFoundError(
            "OmniHD root does not exist or is not a directory: {}".format(
                paths["omnihd_root"]
            )
        )
    for key, description in (
        ("ann_file", "annotation pickle"),
        ("predictions", "prediction pickle"),
    ):
        if not paths[key].is_file():
            raise FileNotFoundError(
                "{} does not exist or is not a file: {}".format(
                    description, paths[key]
                )
            )
    if paths["output"].exists() and paths["output"].is_dir():
        raise IsADirectoryError("Output path is a directory: {}".format(paths["output"]))
    if paths["output"] in {paths["ann_file"], paths["predictions"]}:
        raise ValueError("Output must not overwrite an input pickle")
    return paths


def import_official_evaluation(omnihd_root: Path) -> Dict[str, Any]:
    """Import the official evaluator after adding the external repository."""
    root_string = str(omnihd_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        from newscenes_devkit.eval.common.data_classes import EvalBoxes
        from newscenes_devkit.eval.detection.algo import accumulate, calc_ap
        from newscenes_devkit.eval.detection.config import config_factory
        from newscenes_devkit.eval.detection.data_classes import DetectionBox
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the official OmniHD/NewScenes evaluation code from "
            "{}: {}".format(omnihd_root, exc)
        ) from exc

    return {
        "EvalBoxes": EvalBoxes,
        "DetectionBox": DetectionBox,
        "accumulate": accumulate,
        "calc_ap": calc_ap,
        "config_factory": config_factory,
    }


def validate_official_config(cfg: Any) -> None:
    """Ensure the loaded official config is the requested baseline protocol."""
    if list(cfg.class_names) != list(CLASSES):
        raise ValueError(
            "Official config classes are {}, expected {}".format(
                list(cfg.class_names), list(CLASSES)
            )
        )
    if cfg.dist_fcn != "center_distance":
        raise ValueError(
            "Official config distance function is {!r}, expected center_distance".format(
                cfg.dist_fcn
            )
        )
    thresholds = [float(value) for value in cfg.dist_ths]
    if thresholds != [1.0, 2.0, 3.0, 4.0]:
        raise ValueError(
            "Official config thresholds are {}, expected [1.0, 2.0, 3.0, 4.0]".format(
                thresholds
            )
        )


def load_pickle(path: Path, required_key: str, description: str) -> Dict[str, Any]:
    """Load and validate the top-level structure of an input pickle."""
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise ValueError("Could not load {} {}: {}".format(description, path, exc)) from exc
    if not isinstance(payload, dict):
        raise TypeError("{} must contain a top-level dictionary".format(description))
    if required_key not in payload:
        raise KeyError(
            "{} is missing required top-level key: {}".format(description, required_key)
        )
    if not isinstance(payload[required_key], list):
        raise TypeError("{}[{}] must be a list".format(description, required_key))
    return payload


def _sample_token(record: Mapping[str, Any], source: str, index: int) -> str:
    token = record.get("sample_token" if source == "prediction" else "token")
    if not isinstance(token, str) or not token:
        raise ValueError("{} record {} has an invalid sample token".format(source, index))
    return token


def _numeric_array(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a numeric array".format(name)) from exc
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError("{} must be a real numeric array".format(name))
    return array


def _validate_boxes(boxes: Any, name: str) -> np.ndarray:
    array = _numeric_array(boxes, name)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("{} must be a two-dimensional array with at least 3 columns".format(name))
    if not np.all(np.isfinite(array[:, :3])):
        raise ValueError("{} x/y/z centers must contain only finite values".format(name))
    return array


def _empty_class_counts() -> Dict[str, int]:
    return {class_name: 0 for class_name in CLASSES}


def _in_class_range(box: np.ndarray, class_name: str, cfg: Any) -> bool:
    limits = cfg.class_range[class_name]
    if not isinstance(limits, (list, tuple)) or len(limits) != 2:
        raise ValueError("Official class_range for {} must contain x/y limits".format(class_name))
    return abs(float(box[0])) <= float(limits[0]) and abs(float(box[1])) <= float(
        limits[1]
    )


def _make_detection_box(
    DetectionBox: Any,
    sample_token: str,
    box: np.ndarray,
    class_name: str,
    score: float,
) -> Any:
    """Create an official box with neutral fields unused by this AP report."""
    translation = tuple(float(value) for value in box[:3])
    return DetectionBox(
        sample_token=sample_token,
        translation=translation,
        ego_translation=translation,
        size=(1.0, 1.0, 1.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        velocity=(0.0, 0.0),
        num_pts=-1,
        detection_name=class_name,
        detection_score=float(score),
        visibility=1,
    )


def build_gt_eval_boxes(
    payload: Mapping[str, Any], cfg: Any, EvalBoxes: Any, DetectionBox: Any
) -> Tuple[Any, set, Dict[str, Any]]:
    """Validate converted GT and construct range-filtered official boxes."""
    eval_boxes = EvalBoxes()
    tokens = set()
    before_by_class = _empty_class_counts()
    after_by_class = _empty_class_counts()
    total_before = 0
    total_after = 0

    if not payload["infos"]:
        raise ValueError("Annotation pickle contains no sample infos")

    metadata = payload.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("Annotation metadata must be a mapping when present")
        metadata_classes = metadata.get("classes")
        if metadata_classes is not None and list(metadata_classes) != list(CLASSES):
            raise ValueError(
                "Annotation metadata classes must equal {}".format(list(CLASSES))
            )

    for index, info in enumerate(payload["infos"]):
        if not isinstance(info, Mapping):
            raise TypeError("GT info {} must be a mapping".format(index))
        token = _sample_token(info, "GT", index)
        if token in tokens:
            raise ValueError("Duplicate GT sample token: {}".format(token))
        tokens.add(token)

        if "gt_boxes" not in info or "gt_names" not in info:
            raise KeyError("GT sample {} is missing gt_boxes or gt_names".format(token))
        boxes = _validate_boxes(info["gt_boxes"], "GT boxes for {}".format(token))
        names = np.asarray(info["gt_names"])
        if names.ndim != 1 or names.shape[0] != boxes.shape[0]:
            raise ValueError(
                "GT names for {} must be one-dimensional and match gt_boxes".format(token)
            )

        if "valid_flag" in info:
            valid_flag = np.asarray(info["valid_flag"])
            if valid_flag.ndim != 1 or valid_flag.shape[0] != boxes.shape[0]:
                raise ValueError("valid_flag for {} must match gt_boxes".format(token))
            if not np.issubdtype(valid_flag.dtype, np.bool_):
                raise ValueError("valid_flag for {} must contain booleans".format(token))
        else:
            valid_flag = np.ones(boxes.shape[0], dtype=bool)

        sample_boxes: List[Any] = []
        for box, name_value in zip(boxes[valid_flag], names[valid_flag]):
            class_name = str(name_value)
            if class_name not in CLASSES:
                raise ValueError(
                    "Unsupported mapped GT class {!r} in sample {}".format(
                        class_name, token
                    )
                )
            total_before += 1
            before_by_class[class_name] += 1
            if _in_class_range(box, class_name, cfg):
                sample_boxes.append(
                    _make_detection_box(
                        DetectionBox, token, box, class_name, score=-1.0
                    )
                )
                total_after += 1
                after_by_class[class_name] += 1
        eval_boxes.add_boxes(token, sample_boxes)

    return eval_boxes, tokens, {
        "before": total_before,
        "after": total_after,
        "before_by_class": before_by_class,
        "after_by_class": after_by_class,
    }


def _prediction_classes(payload: Mapping[str, Any]) -> Tuple[List[str], str]:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("Prediction metadata must be a mapping when present")
    metadata_classes = metadata.get("classes")
    if metadata_classes is not None and list(metadata_classes) != list(CLASSES):
        raise ValueError(
            "Prediction metadata classes must equal {} in that order".format(
                list(CLASSES)
            )
        )
    modality = metadata.get("modality", "unknown")
    if not isinstance(modality, str) or not modality:
        raise ValueError("Prediction metadata modality must be a non-empty string")
    return list(CLASSES), modality


def build_prediction_eval_boxes(
    payload: Mapping[str, Any], cfg: Any, EvalBoxes: Any, DetectionBox: Any
) -> Tuple[Any, set, Dict[str, Any], str]:
    """Validate predictions and construct range-filtered official boxes."""
    _, modality = _prediction_classes(payload)
    eval_boxes = EvalBoxes()
    tokens = set()
    before_by_class = _empty_class_counts()
    after_by_class = _empty_class_counts()
    total_before = 0
    total_after = 0

    if not payload["predictions"]:
        raise ValueError("Prediction pickle contains no prediction records")

    for index, prediction in enumerate(payload["predictions"]):
        if not isinstance(prediction, Mapping):
            raise TypeError("Prediction record {} must be a mapping".format(index))
        token = _sample_token(prediction, "prediction", index)
        if token in tokens:
            raise ValueError("Duplicate prediction sample token: {}".format(token))
        tokens.add(token)

        required = ("boxes_3d", "scores_3d", "labels_3d")
        missing = [name for name in required if name not in prediction]
        if missing:
            raise KeyError(
                "Prediction sample {} is missing: {}".format(token, ", ".join(missing))
            )
        boxes = _validate_boxes(
            prediction["boxes_3d"], "prediction boxes for {}".format(token)
        )
        scores = _numeric_array(
            prediction["scores_3d"], "prediction scores for {}".format(token)
        )
        labels = _numeric_array(
            prediction["labels_3d"], "prediction labels for {}".format(token)
        )
        if scores.ndim != 1 or labels.ndim != 1:
            raise ValueError("Prediction scores and labels for {} must be 1D".format(token))
        if boxes.shape[0] != scores.shape[0] or scores.shape[0] != labels.shape[0]:
            raise ValueError(
                "Prediction box, score, and label counts differ for {}".format(token)
            )
        if boxes.shape[0] > int(cfg.max_boxes_per_sample):
            raise ValueError(
                "Prediction sample {} has {} boxes; official maximum is {}".format(
                    token, boxes.shape[0], cfg.max_boxes_per_sample
                )
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("Prediction scores for {} contain NaN or Inf".format(token))
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Prediction labels for {} must have integer dtype".format(token))
        if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= len(CLASSES)):
            raise ValueError("Prediction labels for {} must be in [0, 3]".format(token))

        sample_boxes: List[Any] = []
        for box, score_value, label_value in zip(boxes, scores, labels):
            class_name = CLASSES[int(label_value)]
            total_before += 1
            before_by_class[class_name] += 1
            if _in_class_range(box, class_name, cfg):
                sample_boxes.append(
                    _make_detection_box(
                        DetectionBox,
                        token,
                        box,
                        class_name,
                        score=float(score_value),
                    )
                )
                total_after += 1
                after_by_class[class_name] += 1
        eval_boxes.add_boxes(token, sample_boxes)

    return eval_boxes, tokens, {
        "before": total_before,
        "after": total_after,
        "before_by_class": before_by_class,
        "after_by_class": after_by_class,
    }, modality


def validate_matching_tokens(gt_tokens: set, prediction_tokens: set) -> None:
    """Require exactly one prediction record for every GT sample and no extras."""
    if gt_tokens == prediction_tokens:
        return
    missing = sorted(gt_tokens - prediction_tokens)
    extra = sorted(prediction_tokens - gt_tokens)
    raise ValueError(
        "GT and prediction sample-token sets differ; missing predictions={}{}; "
        "unexpected predictions={}{}".format(
            missing[:5],
            "..." if len(missing) > 5 else "",
            extra[:5],
            "..." if len(extra) > 5 else "",
        )
    )


def calculate_ap(
    gt_eval_boxes: Any,
    prediction_eval_boxes: Any,
    cfg: Any,
    accumulate: Any,
    calc_ap: Any,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float]:
    """Calculate official AP at each configured threshold and aggregate means."""
    ap_by_threshold: Dict[str, Dict[str, float]] = {}
    class_ap: Dict[str, float] = {}
    for class_name in CLASSES:
        threshold_results: Dict[str, float] = {}
        for threshold_value in cfg.dist_ths:
            threshold = float(threshold_value)
            metric_data = accumulate(
                gt_eval_boxes,
                prediction_eval_boxes,
                class_name,
                cfg.dist_fcn_callable,
                threshold,
                verbose=False,
            )
            threshold_results[str(threshold)] = float(
                calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
            )
        ap_by_threshold[class_name] = threshold_results
        class_ap[class_name] = float(
            sum(threshold_results.values()) / len(threshold_results)
        )
    mean_ap = float(sum(class_ap.values()) / len(class_ap))
    return ap_by_threshold, class_ap, mean_ap


def _validated_gt_arrays(
    info: Mapping[str, Any], token: str, cfg: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid, class-range-filtered GT arrays for operating metrics."""
    boxes = _validate_boxes(info["gt_boxes"], "GT boxes for {}".format(token))
    names = np.asarray(info["gt_names"])
    if names.ndim != 1 or names.shape[0] != boxes.shape[0]:
        raise ValueError("GT names for {} must match gt_boxes".format(token))
    if "valid_flag" in info:
        valid_flag = np.asarray(info["valid_flag"])
        if valid_flag.ndim != 1 or valid_flag.shape[0] != boxes.shape[0]:
            raise ValueError("valid_flag for {} must match gt_boxes".format(token))
        if not np.issubdtype(valid_flag.dtype, np.bool_):
            raise ValueError("valid_flag for {} must contain booleans".format(token))
    else:
        valid_flag = np.ones(boxes.shape[0], dtype=bool)

    keep = []
    for is_valid, box, name_value in zip(valid_flag, boxes, names):
        class_name = str(name_value)
        if is_valid and class_name not in CLASSES:
            raise ValueError(
                "Unsupported mapped GT class {!r} in sample {}".format(
                    class_name, token
                )
            )
        keep.append(bool(is_valid) and _in_class_range(box, class_name, cfg))
    mask = np.asarray(keep, dtype=bool)
    source_indices = np.arange(boxes.shape[0], dtype=np.int64)
    return boxes[mask], names[mask].astype(str), source_indices[mask]


def _validated_prediction_arrays(
    prediction: Mapping[str, Any], token: str, cfg: Any, score_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return score- and class-range-filtered prediction arrays and source indices."""
    boxes = _validate_boxes(
        prediction["boxes_3d"], "prediction boxes for {}".format(token)
    )
    scores = _numeric_array(
        prediction["scores_3d"], "prediction scores for {}".format(token)
    )
    labels = _numeric_array(
        prediction["labels_3d"], "prediction labels for {}".format(token)
    )
    source_indices = np.arange(boxes.shape[0], dtype=np.int64)
    keep = np.asarray(
        [
            float(score) >= score_threshold
            and _in_class_range(box, CLASSES[int(label)], cfg)
            for box, score, label in zip(boxes, scores, labels)
        ],
        dtype=bool,
    )
    return boxes[keep], scores[keep], labels[keep], source_indices[keep]


def _geometry_support(
    gt_payload: Mapping[str, Any], prediction_payload: Mapping[str, Any]
) -> Tuple[bool, Optional[str]]:
    """Check whether real dimensions and yaw are available for every raw box."""
    sources = (
        (gt_payload["infos"], "gt_boxes", "GT"),
        (prediction_payload["predictions"], "boxes_3d", "prediction"),
    )
    for records, key, description in sources:
        for index, record in enumerate(records):
            boxes = np.asarray(record[key])
            if boxes.ndim != 2 or boxes.shape[1] < 7:
                return False, (
                    "{} record {} does not provide [x, y, z, w, l, h, yaw]"
                ).format(description, index)
            if not np.all(np.isfinite(boxes[:, 3:7])):
                return False, "{} dimensions or yaw contain NaN/Inf".format(
                    description
                )
            if np.any(boxes[:, 3:6] <= 0.0):
                return False, "{} dimensions are not strictly positive".format(
                    description
                )
    return True, None


def _size_thresholds(
    gt_payload: Mapping[str, Any], cfg: Any
) -> Optional[Tuple[float, float]]:
    """Derive reproducible GT-volume tertiles for small/medium/large bins."""
    volumes: List[float] = []
    for index, info in enumerate(gt_payload["infos"]):
        token = _sample_token(info, "GT", index)
        boxes, _, _ = _validated_gt_arrays(info, token, cfg)
        if boxes.shape[1] < 7:
            return None
        volumes.extend(np.prod(boxes[:, 3:6], axis=1).astype(float).tolist())
    if not volumes:
        return None
    lower, upper = np.percentile(np.asarray(volumes), [100.0 / 3.0, 200.0 / 3.0])
    return float(lower), float(upper)


def _new_metric_bucket() -> Dict[str, Any]:
    return {
        "gt_count": 0,
        "prediction_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "translation_error_sum": 0.0,
        "scale_error_sum": 0.0,
        "orientation_error_sum": 0.0,
    }


def _scale_error(gt_box: np.ndarray, prediction_box: np.ndarray) -> float:
    """Return OmniHD/NewScenes-compatible 1 - aligned dimension IoU."""
    gt_size = gt_box[3:6].astype(float)
    prediction_size = prediction_box[3:6].astype(float)
    intersection = float(np.prod(np.minimum(gt_size, prediction_size)))
    union = float(np.prod(gt_size) + np.prod(prediction_size) - intersection)
    return 1.0 - intersection / union


def _orientation_error(gt_box: np.ndarray, prediction_box: np.ndarray) -> float:
    """Return absolute smallest yaw difference in radians with period 2*pi."""
    difference = float(prediction_box[6]) - float(gt_box[6])
    return abs((difference + np.pi) % (2.0 * np.pi) - np.pi)


def _record_match(
    bucket: Dict[str, Any],
    translation_error: float,
    scale_error: Optional[float],
    orientation_error: Optional[float],
) -> None:
    bucket["gt_count"] += 1
    bucket["prediction_count"] += 1
    bucket["tp"] += 1
    bucket["translation_error_sum"] += translation_error
    if scale_error is not None:
        bucket["scale_error_sum"] += scale_error
    if orientation_error is not None:
        bucket["orientation_error_sum"] += orientation_error


def _record_false_negative(bucket: Dict[str, Any]) -> None:
    bucket["gt_count"] += 1
    bucket["fn"] += 1


def _record_false_positive(bucket: Dict[str, Any]) -> None:
    bucket["prediction_count"] += 1
    bucket["fp"] += 1


def _finalize_metric_bucket(
    bucket: Mapping[str, Any], geometry_supported: bool
) -> Dict[str, Any]:
    tp = int(bucket["tp"])
    fp = int(bucket["fp"])
    fn = int(bucket["fn"])
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    if precision is None or recall is None:
        f1_score = None
    elif precision + recall == 0.0:
        f1_score = 0.0
    else:
        f1_score = float(2.0 * precision * recall / (precision + recall))
    return {
        "gt_count": int(bucket["gt_count"]),
        "prediction_count": int(bucket["prediction_count"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "ate_m": float(bucket["translation_error_sum"] / tp) if tp else None,
        "ase": (
            float(bucket["scale_error_sum"] / tp)
            if geometry_supported and tp
            else None
        ),
        "aoe_rad": (
            float(bucket["orientation_error_sum"] / tp)
            if geometry_supported and tp
            else None
        ),
    }


def _finalize_object_condition_bucket(
    bucket: Mapping[str, Any], geometry_supported: bool
) -> Dict[str, Any]:
    """Finalize a GT-defined condition without inventing unmatched-FP labels."""
    tp = int(bucket["tp"])
    fn = int(bucket["fn"])
    return {
        "gt_count": int(bucket["gt_count"]),
        "tp": tp,
        "fn": fn,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "ate_m": float(bucket["translation_error_sum"] / tp) if tp else None,
        "ase": (
            float(bucket["scale_error_sum"] / tp)
            if geometry_supported and tp
            else None
        ),
        "aoe_rad": (
            float(bucket["orientation_error_sum"] / tp)
            if geometry_supported and tp
            else None
        ),
    }


def _condition_metadata_support(
    gt_payload: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Validate enriched condition fields or identify an unenriched PKL."""
    definitions = {
        "visibility": "gt_visibility_levels",
        "motion": "gt_motion_states",
        "weather": "weather",
        "area": "area",
    }
    result: Dict[str, Dict[str, Any]] = {}
    infos = gt_payload["infos"]
    for condition, field in definitions.items():
        present = [field in info for info in infos]
        if not any(present):
            result[condition] = {
                "supported": False,
                "reason": "Enriched field {!r} is absent from the annotation PKL.".format(
                    field
                ),
            }
            continue
        if not all(present):
            raise ValueError(
                "Enriched field {!r} is present for only some samples".format(field)
            )

        categories = set()
        for index, info in enumerate(infos):
            token = _sample_token(info, "GT", index)
            object_count = int(np.asarray(info["gt_boxes"]).shape[0])
            value = info[field]
            if condition == "visibility":
                if not isinstance(value, (list, tuple)) or len(value) != object_count:
                    raise ValueError(
                        "{} must align with GT boxes for sample {}".format(field, token)
                    )
                if any(
                    isinstance(level, bool)
                    or not isinstance(level, (int, np.integer))
                    or int(level) not in (1, 2, 3, 4)
                    for level in value
                ):
                    raise ValueError(
                        "{} contains an invalid visibility level for sample {}".format(
                            field, token
                        )
                    )
                categories.update(str(int(level)) for level in value)
            elif condition == "motion":
                if not isinstance(value, (list, tuple)) or len(value) != object_count:
                    raise ValueError(
                        "{} must align with GT boxes for sample {}".format(field, token)
                    )
                invalid = sorted(
                    set(value) - {"moving", "stationary", "unknown"}
                )
                if invalid:
                    raise ValueError(
                        "{} contains invalid motion states for sample {}: {}".format(
                            field, token, invalid
                        )
                    )
                categories.update(value)
            else:
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        "{} must be a non-empty string for sample {}".format(
                            field, token
                        )
                    )
                categories.add(value)
        result[condition] = {
            "supported": True,
            "reason": None,
            "categories": sorted(categories),
        }
    return result


def _distance_bin(box: np.ndarray) -> str:
    distance = float(np.linalg.norm(box[:2]))
    for name, lower, upper in DISTANCE_BINS:
        if distance >= lower and (upper is None or distance < upper):
            return name
    raise RuntimeError("Could not assign distance bin")


def _size_bin(box: np.ndarray, thresholds: Tuple[float, float]) -> str:
    volume = float(np.prod(box[3:6]))
    if volume < thresholds[0]:
        return "small"
    if volume < thresholds[1]:
        return "medium"
    return "large"


def _match_one_class(
    gt_boxes: np.ndarray,
    prediction_boxes: np.ndarray,
    prediction_scores: np.ndarray,
    prediction_source_indices: np.ndarray,
    match_distance: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedily match score-ordered predictions to nearest unmatched GT."""
    prediction_order = sorted(
        range(prediction_boxes.shape[0]),
        key=lambda index: (
            -float(prediction_scores[index]),
            int(prediction_source_indices[index]),
        ),
    )
    unmatched_gt = set(range(gt_boxes.shape[0]))
    matches: List[Tuple[int, int, float]] = []
    unmatched_predictions: List[int] = []
    for prediction_index in prediction_order:
        if not unmatched_gt:
            unmatched_predictions.append(prediction_index)
            continue
        candidate_indices = np.asarray(sorted(unmatched_gt), dtype=np.int64)
        distances = np.linalg.norm(
            gt_boxes[candidate_indices, :2]
            - prediction_boxes[prediction_index, :2],
            axis=1,
        )
        nearest_position = int(np.argmin(distances))
        gt_index = int(candidate_indices[nearest_position])
        distance = float(distances[nearest_position])
        if distance < match_distance:
            unmatched_gt.remove(gt_index)
            matches.append((prediction_index, gt_index, distance))
        else:
            unmatched_predictions.append(prediction_index)
    return matches, unmatched_predictions, sorted(unmatched_gt)


def calculate_operating_point_metrics(
    gt_payload: Mapping[str, Any],
    prediction_payload: Mapping[str, Any],
    cfg: Any,
    score_threshold: float,
    match_distance: float,
) -> Dict[str, Any]:
    """Calculate deterministic fixed-threshold detection and error metrics."""
    geometry_supported, geometry_reason = _geometry_support(
        gt_payload, prediction_payload
    )
    size_thresholds = _size_thresholds(gt_payload, cfg) if geometry_supported else None
    size_supported = size_thresholds is not None
    condition_support = _condition_metadata_support(gt_payload)

    overall = _new_metric_bucket()
    by_class = {class_name: _new_metric_bucket() for class_name in CLASSES}
    by_distance = {name: _new_metric_bucket() for name, _, _ in DISTANCE_BINS}
    by_size = {
        name: _new_metric_bucket() for name in ("small", "medium", "large")
    }
    by_visibility = {
        level: _new_metric_bucket() for level in ("1", "2", "3", "4")
    }
    by_motion = {
        state: _new_metric_bucket()
        for state in ("moving", "stationary", "unknown")
    }
    by_weather = {
        name: _new_metric_bucket()
        for name in condition_support["weather"].get("categories", [])
    }
    by_area = {
        name: _new_metric_bucket()
        for name in condition_support["area"].get("categories", [])
    }
    weather_sample_counts = {name: 0 for name in by_weather}
    area_sample_counts = {name: 0 for name in by_area}
    predictions_by_token = {
        _sample_token(record, "prediction", index): record
        for index, record in enumerate(prediction_payload["predictions"])
    }

    for index, info in enumerate(gt_payload["infos"]):
        token = _sample_token(info, "GT", index)
        gt_boxes, gt_names, gt_source_indices = _validated_gt_arrays(
            info, token, cfg
        )
        weather = info["weather"] if condition_support["weather"]["supported"] else None
        area = info["area"] if condition_support["area"]["supported"] else None
        if weather is not None:
            weather_sample_counts[weather] += 1
        if area is not None:
            area_sample_counts[area] += 1
        prediction = predictions_by_token[token]
        prediction_boxes, prediction_scores, prediction_labels, source_indices = (
            _validated_prediction_arrays(
                prediction, token, cfg, score_threshold
            )
        )

        for class_index, class_name in enumerate(CLASSES):
            gt_class_indices = np.flatnonzero(gt_names == class_name)
            prediction_class_indices = np.flatnonzero(
                prediction_labels == class_index
            )
            class_gt_boxes = gt_boxes[gt_class_indices]
            class_prediction_boxes = prediction_boxes[prediction_class_indices]
            class_prediction_scores = prediction_scores[prediction_class_indices]
            class_source_indices = source_indices[prediction_class_indices]
            matches, unmatched_predictions, unmatched_gt = _match_one_class(
                class_gt_boxes,
                class_prediction_boxes,
                class_prediction_scores,
                class_source_indices,
                match_distance,
            )

            for prediction_index, gt_index, translation_error in matches:
                gt_box = class_gt_boxes[gt_index]
                prediction_box = class_prediction_boxes[prediction_index]
                original_gt_index = int(
                    gt_source_indices[gt_class_indices[gt_index]]
                )
                scale_error = (
                    _scale_error(gt_box, prediction_box)
                    if geometry_supported
                    else None
                )
                orientation_error = (
                    _orientation_error(gt_box, prediction_box)
                    if geometry_supported
                    else None
                )
                buckets = [
                    overall,
                    by_class[class_name],
                    by_distance[_distance_bin(gt_box)],
                ]
                if size_supported:
                    buckets.append(by_size[_size_bin(gt_box, size_thresholds)])
                if weather is not None:
                    buckets.append(by_weather[weather])
                if area is not None:
                    buckets.append(by_area[area])
                for bucket in buckets:
                    _record_match(
                        bucket,
                        translation_error,
                        scale_error,
                        orientation_error,
                    )
                if condition_support["visibility"]["supported"]:
                    visibility_level = str(
                        int(info["gt_visibility_levels"][original_gt_index])
                    )
                    _record_match(
                        by_visibility[visibility_level],
                        translation_error,
                        scale_error,
                        orientation_error,
                    )
                if condition_support["motion"]["supported"]:
                    motion_state = info["gt_motion_states"][original_gt_index]
                    _record_match(
                        by_motion[motion_state],
                        translation_error,
                        scale_error,
                        orientation_error,
                    )

            for gt_index in unmatched_gt:
                gt_box = class_gt_boxes[gt_index]
                original_gt_index = int(
                    gt_source_indices[gt_class_indices[gt_index]]
                )
                buckets = [
                    overall,
                    by_class[class_name],
                    by_distance[_distance_bin(gt_box)],
                ]
                if size_supported:
                    buckets.append(by_size[_size_bin(gt_box, size_thresholds)])
                if weather is not None:
                    buckets.append(by_weather[weather])
                if area is not None:
                    buckets.append(by_area[area])
                for bucket in buckets:
                    _record_false_negative(bucket)
                if condition_support["visibility"]["supported"]:
                    visibility_level = str(
                        int(info["gt_visibility_levels"][original_gt_index])
                    )
                    _record_false_negative(by_visibility[visibility_level])
                if condition_support["motion"]["supported"]:
                    motion_state = info["gt_motion_states"][original_gt_index]
                    _record_false_negative(by_motion[motion_state])

            for prediction_index in unmatched_predictions:
                prediction_box = class_prediction_boxes[prediction_index]
                buckets = [
                    overall,
                    by_class[class_name],
                    by_distance[_distance_bin(prediction_box)],
                ]
                if size_supported:
                    buckets.append(
                        by_size[_size_bin(prediction_box, size_thresholds)]
                    )
                if weather is not None:
                    buckets.append(by_weather[weather])
                if area is not None:
                    buckets.append(by_area[area])
                for bucket in buckets:
                    _record_false_positive(bucket)

    finalized_class = {
        name: _finalize_metric_bucket(bucket, geometry_supported)
        for name, bucket in by_class.items()
    }
    finalized_distance = {
        name: _finalize_metric_bucket(bucket, geometry_supported)
        for name, bucket in by_distance.items()
    }
    finalized_size = (
        {
            name: _finalize_metric_bucket(bucket, geometry_supported)
            for name, bucket in by_size.items()
        }
        if size_supported
        else None
    )
    finalized_visibility = (
        {
            level: _finalize_object_condition_bucket(bucket, geometry_supported)
            for level, bucket in by_visibility.items()
        }
        if condition_support["visibility"]["supported"]
        else None
    )
    finalized_motion = (
        {
            state: _finalize_object_condition_bucket(bucket, geometry_supported)
            for state, bucket in by_motion.items()
        }
        if condition_support["motion"]["supported"]
        else None
    )
    finalized_weather = (
        {
            name: {
                "sample_count": int(weather_sample_counts[name]),
                **_finalize_metric_bucket(bucket, geometry_supported),
            }
            for name, bucket in by_weather.items()
        }
        if condition_support["weather"]["supported"]
        else None
    )
    finalized_area = (
        {
            name: {
                "sample_count": int(area_sample_counts[name]),
                **_finalize_metric_bucket(bucket, geometry_supported),
            }
            for name, bucket in by_area.items()
        }
        if condition_support["area"]["supported"]
        else None
    )
    return {
        "score_threshold": float(score_threshold),
        "match_distance_m": float(match_distance),
        "overall": _finalize_metric_bucket(overall, geometry_supported),
        "by_class": finalized_class,
        "by_distance": finalized_distance,
        "by_size": finalized_size,
        "by_visibility": finalized_visibility,
        "by_weather": finalized_weather,
        "by_motion": finalized_motion,
        "by_area": finalized_area,
        "condition_support": condition_support,
        "geometry_supported": geometry_supported,
        "geometry_reason": geometry_reason,
        "size_thresholds_m3": (
            {
                "small_medium": float(size_thresholds[0]),
                "medium_large": float(size_thresholds[1]),
            }
            if size_supported
            else None
        ),
    }


def _protocol_class_range(cfg: Any) -> Dict[str, List[float]]:
    return {
        class_name: [float(value) for value in cfg.class_range[class_name]]
        for class_name in CLASSES
    }


def build_result(
    paths: Mapping[str, Path],
    cfg: Any,
    modality: str,
    sample_count: int,
    gt_counts: Mapping[str, Any],
    prediction_counts: Mapping[str, Any],
    ap_by_threshold: Mapping[str, Mapping[str, float]],
    class_ap: Mapping[str, float],
    mean_ap: float,
    operating_metrics: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build a JSON-safe report while preserving the original AP fields."""
    operating_enabled = operating_metrics is not None
    result = {
        "experiment": {
            "model": "OmniHD-pretrained PointPillars",
            "source_training_dataset": "OmniHD",
            "evaluation_dataset": "TruckScenes",
            "prediction_modality": modality,
            "interpretation": (
                "OmniHD-pretrained PointPillars transferred to TruckScenes"
            ),
        },
        "inputs": {
            "omnihd_root": str(paths["omnihd_root"]),
            "ann_file": str(paths["ann_file"]),
            "predictions": str(paths["predictions"]),
        },
        "protocol": {
            "name": "OmniHD/NewScenes adapted center-distance AP",
            "config": CONFIG_NAME,
            "classes": list(CLASSES),
            "distance_thresholds_m": [float(value) for value in cfg.dist_ths],
            "min_recall": float(cfg.min_recall),
            "min_precision": float(cfg.min_precision),
            "class_range": _protocol_class_range(cfg),
            "max_boxes_per_sample": int(cfg.max_boxes_per_sample),
            "matching": "XY center distance with strict distance < threshold",
            "operating_point": {
                "enabled": operating_enabled,
                "score_threshold": (
                    operating_metrics["score_threshold"]
                    if operating_enabled
                    else None
                ),
                "score_comparison": "prediction score >= score threshold",
                "match_distance_m": (
                    operating_metrics["match_distance_m"]
                    if operating_enabled
                    else None
                ),
                "matching_algorithm": (
                    "Within each sample and class, predictions are ordered by "
                    "descending score with original prediction index as the tie-break. "
                    "Each prediction is matched to the nearest still-unmatched GT; "
                    "GT index breaks equal-distance ties; distance must be strictly "
                    "less than match_distance_m."
                ),
            },
            "visibility_policy": (
                "No native OmniHD camera-visibility filtering. TruckScenes "
                "valid_flag represents adapter-retained mapped annotations only."
            ),
            "reported_metrics": (
                "Official AP/mAP plus fixed-operating-point detection and box errors"
                if operating_enabled
                else "Official AP and mAP only"
            ),
            "neutral_box_fields": (
                "Size, rotation, velocity, and point count are neutral placeholders "
                "inside the official DetectionBox AP path. ATE/ASE/AOE use the real "
                "raw GT and prediction arrays, never these placeholders."
            ),
            "box_error_definitions": {
                "ate_m": "Mean matched-TP XY center distance in metres.",
                "ase": (
                    "Mean matched-TP 1 - scale IoU, using aligned real w/l/h "
                    "dimensions as in OmniHD/NewScenes."
                ),
                "aoe_rad": (
                    "Mean matched-TP absolute smallest yaw difference in radians, "
                    "with 2*pi periodicity for all four evaluated classes."
                ),
            },
        },
        "dataset": {
            "samples": int(sample_count),
            "gt_boxes_before_range_filter": int(gt_counts["before"]),
            "gt_boxes_after_range_filter": int(gt_counts["after"]),
            "predictions_before_range_filter": int(prediction_counts["before"]),
            "predictions_after_range_filter": int(prediction_counts["after"]),
            "gt_boxes_by_class_before_range_filter": {
                name: int(value)
                for name, value in gt_counts["before_by_class"].items()
            },
            "gt_boxes_by_class_after_range_filter": {
                name: int(value)
                for name, value in gt_counts["after_by_class"].items()
            },
            "predictions_by_class_before_range_filter": {
                name: int(value)
                for name, value in prediction_counts["before_by_class"].items()
            },
            "predictions_by_class_after_range_filter": {
                name: int(value)
                for name, value in prediction_counts["after_by_class"].items()
            },
        },
        "ap_by_class_and_threshold": {
            name: {threshold: float(value) for threshold, value in values.items()}
            for name, values in ap_by_threshold.items()
        },
        "class_ap": {name: float(value) for name, value in class_ap.items()},
        "mAP": float(mean_ap),
    }

    result["overall_metrics"] = {
        "mAP": float(mean_ap),
        "operating_point": (
            dict(operating_metrics["overall"]) if operating_enabled else None
        ),
    }
    result["class_metrics"] = {
        name: {
            "ap_by_distance_threshold": {
                threshold: float(value)
                for threshold, value in ap_by_threshold[name].items()
            },
            "mean_ap": float(class_ap[name]),
            "operating_point": (
                dict(operating_metrics["by_class"][name])
                if operating_enabled
                else None
            ),
        }
        for name in CLASSES
    }
    result["distance_metrics"] = (
        {
            "supported": True,
            "distance_definition": "sqrt(x^2 + y^2) in metres",
            "bin_boundaries": "lower-inclusive, upper-exclusive",
            "attribution": (
                "Matched predictions use matched GT distance; unmatched predictions "
                "use prediction distance; unmatched GT uses GT distance."
            ),
            "bins": {
                name: dict(operating_metrics["by_distance"][name])
                for name, _, _ in DISTANCE_BINS
            },
        }
        if operating_enabled
        else {
            "supported": False,
            "reason": "No fixed operating point was supplied.",
        }
    )
    size_supported = bool(
        operating_enabled and operating_metrics["by_size"] is not None
    )
    result["size_metrics"] = (
        {
            "supported": True,
            "size_definition": "3D box volume w*l*h in cubic metres",
            "bin_definition": (
                "Small/medium/large are GT-volume tertiles after valid_flag and "
                "official class-range filtering. Thresholds are recorded below."
            ),
            "thresholds_m3": dict(operating_metrics["size_thresholds_m3"]),
            "attribution": (
                "Matched predictions use matched GT size; unmatched predictions use "
                "predicted size; unmatched GT uses GT size."
            ),
            "bins": {
                name: dict(operating_metrics["by_size"][name])
                for name in ("small", "medium", "large")
            },
        }
        if size_supported
        else {
            "supported": False,
            "reason": (
                operating_metrics["geometry_reason"]
                if operating_enabled and operating_metrics["geometry_reason"]
                else "No fixed operating point was supplied."
            ),
        }
    )
    no_operating_reason = "No fixed operating point was supplied."
    object_condition_attribution = (
        "Matched predictions inherit the matched GT condition; unmatched GT uses "
        "its own condition. Unmatched false positives have no defensible GT condition, "
        "so FP, prediction count, precision, and F1 are intentionally omitted."
    )
    if operating_enabled and operating_metrics["by_visibility"] is not None:
        result["visibility_metrics"] = {
            "supported": True,
            "levels": {
                level: dict(operating_metrics["by_visibility"][level])
                for level in ("1", "2", "3", "4")
            },
            "level_meaning": {
                "1": "0-40% visible",
                "2": "41-60% visible",
                "3": "61-80% visible",
                "4": "81-100% visible",
            },
            "false_positive_attribution": object_condition_attribution,
        }
    else:
        reason = (
            operating_metrics["condition_support"]["visibility"]["reason"]
            if operating_enabled
            else no_operating_reason
        )
        result["visibility_metrics"] = {"supported": False, "reason": reason}

    if operating_enabled and operating_metrics["by_motion"] is not None:
        result["motion_metrics"] = {
            "supported": True,
            "states": {
                state: dict(operating_metrics["by_motion"][state])
                for state in ("moving", "stationary", "unknown")
            },
            "false_positive_attribution": object_condition_attribution,
        }
    else:
        reason = (
            operating_metrics["condition_support"]["motion"]["reason"]
            if operating_enabled
            else no_operating_reason
        )
        result["motion_metrics"] = {"supported": False, "reason": reason}

    for condition, output_key in (
        ("weather", "weather_metrics"),
        ("area", "area_metrics"),
    ):
        values = operating_metrics["by_{}".format(condition)] if operating_enabled else None
        if values is not None:
            result[output_key] = {
                "supported": True,
                "attribution": (
                    "The condition is sample-level, so all GT and predictions in a "
                    "sample inherit the sample condition."
                ),
                "categories": {
                    name: dict(metrics) for name, metrics in values.items()
                },
            }
        else:
            reason = (
                operating_metrics["condition_support"][condition]["reason"]
                if operating_enabled
                else no_operating_reason
            )
            result[output_key] = {"supported": False, "reason": reason}

    supported_metrics = ["official_ap", "class_mean_ap", "overall_map"]
    unsupported_metrics = {
        "ave": "Compatible GT object velocity is unavailable and is not fabricated.",
        "lidar_radar_complementarity": (
            "A second-modality trainval prediction PKL is not part of this run."
        ),
    }
    if operating_enabled:
        supported_metrics.extend(
            [
                "tp_fp_fn",
                "precision_recall_f1",
                "ate",
                "class_breakdown",
                "distance_breakdown",
            ]
        )
        if operating_metrics["geometry_supported"]:
            supported_metrics.extend(["ase", "aoe", "size_breakdown"])
        else:
            reason = operating_metrics["geometry_reason"]
            unsupported_metrics["ase"] = reason
            unsupported_metrics["aoe"] = reason
            unsupported_metrics["size_breakdown"] = reason
        for condition in ("visibility", "weather", "motion", "area"):
            support = operating_metrics["condition_support"][condition]
            metric_name = "{}_breakdown".format(condition)
            if support["supported"]:
                supported_metrics.append(metric_name)
            else:
                unsupported_metrics[metric_name] = support["reason"]
    else:
        unsupported_metrics["fixed_operating_point_metrics"] = (
            "Supply both --score-threshold and --match-distance to calculate them."
        )
    result["supported_metrics"] = supported_metrics
    result["unsupported_metrics"] = unsupported_metrics
    return result


def save_json(result: Mapping[str, Any], output_path: Path) -> None:
    """Save metrics as human-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=False, allow_nan=False)
        stream.write("\n")


def print_result(result: Mapping[str, Any], output_path: Path) -> None:
    """Print the compact AP table, optional operating point, and totals."""
    thresholds = result["protocol"]["distance_thresholds_m"]
    print("Detection evaluation")
    print("-" * 78)
    print(
        "{:<18} {:>9} {:>9} {:>9} {:>9} {:>10}".format(
            "Class", "AP@1m", "AP@2m", "AP@3m", "AP@4m", "Mean AP"
        )
    )
    for class_name in CLASSES:
        values = result["ap_by_class_and_threshold"][class_name]
        row = [values[str(float(threshold))] for threshold in thresholds]
        print(
            "{:<18} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>10.4f}".format(
                class_name, row[0], row[1], row[2], row[3], result["class_ap"][class_name]
            )
        )
    print("-" * 78)
    print("Overall mAP: {:.4f}".format(result["mAP"]))
    operating = result["overall_metrics"]["operating_point"]
    if operating is not None:
        print(
            "Operating point: score >= {:.4f}, match distance < {:.4f} m".format(
                result["protocol"]["operating_point"]["score_threshold"],
                result["protocol"]["operating_point"]["match_distance_m"],
            )
        )
        print(
            "TP={} FP={} FN={} Precision={} Recall={} F1={} ATE={} m".format(
                operating["tp"],
                operating["fp"],
                operating["fn"],
                operating["precision"],
                operating["recall"],
                operating["f1_score"],
                operating["ate_m"],
            )
        )
    print()
    print("Samples evaluated: {}".format(result["dataset"]["samples"]))
    print(
        "GT boxes after filtering: {}".format(
            result["dataset"]["gt_boxes_after_range_filter"]
        )
    )
    print(
        "Predictions after filtering: {}".format(
            result["dataset"]["predictions_after_range_filter"]
        )
    )
    print()
    print("Metrics saved: {}".format(output_path))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate inputs, calculate official AP and optional richer metrics."""
    operating_enabled = validate_operating_point_args(args)
    paths = resolve_paths(args)
    official = import_official_evaluation(paths["omnihd_root"])
    cfg = official["config_factory"](CONFIG_NAME)
    validate_official_config(cfg)

    gt_payload = load_pickle(paths["ann_file"], "infos", "annotation pickle")
    prediction_payload = load_pickle(
        paths["predictions"], "predictions", "prediction pickle"
    )
    gt_eval_boxes, gt_tokens, gt_counts = build_gt_eval_boxes(
        gt_payload, cfg, official["EvalBoxes"], official["DetectionBox"]
    )
    prediction_eval_boxes, prediction_tokens, prediction_counts, modality = (
        build_prediction_eval_boxes(
            prediction_payload,
            cfg,
            official["EvalBoxes"],
            official["DetectionBox"],
        )
    )
    validate_matching_tokens(gt_tokens, prediction_tokens)

    ap_by_threshold, class_ap, mean_ap = calculate_ap(
        gt_eval_boxes,
        prediction_eval_boxes,
        cfg,
        official["accumulate"],
        official["calc_ap"],
    )
    operating_metrics = None
    if operating_enabled:
        operating_metrics = calculate_operating_point_metrics(
            gt_payload,
            prediction_payload,
            cfg,
            float(args.score_threshold),
            float(args.match_distance),
        )
    result = build_result(
        paths,
        cfg,
        modality,
        len(gt_tokens),
        gt_counts,
        prediction_counts,
        ap_by_threshold,
        class_ap,
        mean_ap,
        operating_metrics,
    )
    save_json(result, paths["output"])
    print_result(result, paths["output"])
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
