#!/usr/bin/env python3
"""Evaluate transferred OmniHD detections on converted TruckScenes ground truth."""

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CONFIG_NAME = "detection_newsc_config_final"
CLASSES = ("car", "pedestrian", "rider", "large_vehicle")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse evaluator arguments without importing the external OmniHD code."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate official OmniHD/NewScenes center-distance AP for "
            "OmniHD-pretrained PointPillars predictions transferred to TruckScenes."
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
        help="Destination JSON file for AP and mAP metrics.",
    )
    return parser.parse_args(argv)


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
) -> Dict[str, Any]:
    """Build a JSON-safe, AP-only experiment report."""
    return {
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
            "visibility_policy": (
                "No native OmniHD camera-visibility filtering. TruckScenes "
                "valid_flag represents adapter-retained mapped annotations only."
            ),
            "reported_metrics": "AP and mAP only",
            "neutral_box_fields": (
                "Size, rotation, velocity, and point count are neutral placeholders "
                "required by DetectionBox and are not reported as TP metrics."
            ),
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


def save_json(result: Mapping[str, Any], output_path: Path) -> None:
    """Save metrics as human-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=False, allow_nan=False)
        stream.write("\n")


def print_result(result: Mapping[str, Any], output_path: Path) -> None:
    """Print the compact AP table and dataset totals."""
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
    """Validate inputs, call official AP utilities, and save the report."""
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
