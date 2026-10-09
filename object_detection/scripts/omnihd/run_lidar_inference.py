#!/usr/bin/env python3
"""Run OmniHD PointPillars LiDAR inference on a converted dataset."""

import argparse
import importlib
import os
from pathlib import Path
import pickle
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


MODEL_NAME = "OmniHD PointPillars LiDAR"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the official OmniHD LiDAR PointPillars model over a converted "
            "TruckScenes dataset."
        )
    )
    parser.add_argument(
        "--omnihd-root",
        required=True,
        help="Path to the separate OmniHD-Scenes repository.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="OmniHD LiDAR config; relative paths are resolved from --omnihd-root.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="LiDAR checkpoint; relative paths are resolved from --omnihd-root.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help=(
            "Root of the converted TruckScenes dataset; relative paths resolve "
            "from the launch directory."
        ),
    )
    parser.add_argument(
        "--ann-file",
        required=True,
        help=(
            "Converted TruckScenes OmniHD-style annotation pickle; relative paths "
            "resolve from the launch directory."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Destination prediction pickle; relative paths resolve from the "
            "launch directory."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process; default processes the remaining dataset.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based dataset index at which to start (default: 0).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Atomically save progress after this many completed samples (default: 50).",
    )
    return parser.parse_args(argv)


def _resolve_path(value: str, base: Path) -> Path:
    """Resolve a user path after expanding ``~``."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_and_validate_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """Resolve config/checkpoint from OmniHD and other paths from launch cwd."""
    launch_directory = Path.cwd().resolve()
    omnihd_root = _resolve_path(args.omnihd_root, launch_directory)
    config = _resolve_path(args.config, omnihd_root)
    checkpoint = _resolve_path(args.checkpoint, omnihd_root)
    data_root = _resolve_path(args.data_root, launch_directory)
    ann_file = _resolve_path(args.ann_file, launch_directory)
    output = _resolve_path(args.output, launch_directory)

    for path, description in (
        (omnihd_root, "OmniHD root"),
        (data_root, "converted data root"),
    ):
        if not path.is_dir():
            raise FileNotFoundError("{} does not exist or is not a directory: {}".format(
                description, path
            ))
    for path, description in (
        (config, "OmniHD config"),
        (checkpoint, "checkpoint"),
        (ann_file, "annotation pickle"),
    ):
        if not path.is_file():
            raise FileNotFoundError("{} does not exist or is not a file: {}".format(
                description, path
            ))
    if output.exists() and output.is_dir():
        raise IsADirectoryError("Output path is a directory: {}".format(output))
    if output in {config, checkpoint, ann_file}:
        raise ValueError("Output path must not overwrite an input file: {}".format(output))

    return {
        "omnihd_root": omnihd_root,
        "config": config,
        "checkpoint": checkpoint,
        "data_root": data_root,
        "ann_file": ann_file,
        "output": output,
    }


def _validate_range_arguments(args: argparse.Namespace) -> None:
    if args.start_index < 0:
        raise ValueError("--start-index must be greater than or equal to 0")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be greater than or equal to 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be greater than or equal to 1")


def _import_omnihd_dependencies(omnihd_root: Path) -> Dict[str, Any]:
    """Import legacy OmniHD dependencies after adding its root to ``sys.path``."""
    root_string = str(omnihd_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    try:
        import torch
        from mmcv import Config
        from mmcv.parallel import MMDataParallel
        from mmcv.runner import load_checkpoint
        from mmdet3d.datasets import build_dataloader, build_dataset
        from mmdet3d.models import build_model

        importlib.import_module("projects.mmdet3d_plugin")
        # Explicitly import the dataset package so NewScenesDataset is
        # registered even if a plugin __init__ changes its transitive imports.
        importlib.import_module("projects.mmdet3d_plugin.datasets")
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the OmniHD/MMDetection3D environment after "
            "activating {!s}: {}".format(omnihd_root, exc)
        ) from exc

    return {
        "torch": torch,
        "Config": Config,
        "MMDataParallel": MMDataParallel,
        "load_checkpoint": load_checkpoint,
        "build_dataloader": build_dataloader,
        "build_dataset": build_dataset,
        "build_model": build_model,
    }


def _portable_metadata_path(path: Path, base: Optional[Path] = None) -> str:
    """Serialize a path without exposing machine-specific absolute locations."""
    if base is not None:
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            pass
    return path.name


def _prepare_config(Config: Any, paths: Dict[str, Path]) -> Any:
    """Load the official config and apply the proven single-GPU overrides."""
    cfg = Config.fromfile(str(paths["config"]))
    try:
        cfg.data.test.with_velocity = False
        cfg.data.test.test_mode = True
        cfg.data.test.data_root = str(paths["data_root"])
        cfg.data.test.ann_file = str(paths["ann_file"])
        cfg.data.test.pop("samples_per_gpu", None)
        cfg.data.workers_per_gpu = 0

        cfg.model.pts_voxel_encoder.norm_cfg = dict(
            type="BN1d", eps=1e-3, momentum=0.01
        )
        cfg.model.pts_backbone.norm_cfg = dict(
            type="BN2d", eps=1e-3, momentum=0.01
        )
        cfg.model.pts_neck.norm_cfg = dict(
            type="BN2d", eps=1e-3, momentum=0.01
        )
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "The supplied config does not have the expected OmniHD LiDAR "
            "PointPillars structure: {}".format(exc)
        ) from exc
    return cfg


def _to_numpy(value: Any, name: str) -> np.ndarray:
    """Detach a tensor-like prediction and copy it into CPU NumPy memory."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError("{} could not be converted to a numeric NumPy array".format(name))
    return np.array(array, copy=True)


def _extract_prediction(result: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract serializable arrays from either supported OmniHD result layout."""
    if isinstance(result, (list, tuple)):
        if not result:
            raise ValueError("Model returned an empty result sequence")
        prediction = result[0]
    elif isinstance(result, dict):
        prediction = result
    else:
        raise TypeError("Unexpected model result type: {}".format(type(result).__name__))

    if not isinstance(prediction, dict):
        raise TypeError("Per-sample model result must be a dictionary")
    if "pts_bbox" in prediction:
        prediction = prediction["pts_bbox"]
    if not isinstance(prediction, dict):
        raise TypeError("pts_bbox result must be a dictionary")

    required = ("boxes_3d", "scores_3d", "labels_3d")
    missing = [key for key in required if key not in prediction]
    if missing:
        raise KeyError("Prediction is missing fields: {}".format(", ".join(missing)))

    boxes_value = prediction["boxes_3d"]
    if hasattr(boxes_value, "tensor"):
        boxes_value = boxes_value.tensor
    boxes = _to_numpy(boxes_value, "boxes_3d")
    scores = _to_numpy(prediction["scores_3d"], "scores_3d").reshape(-1)
    labels = _to_numpy(prediction["labels_3d"], "labels_3d").reshape(-1)

    if boxes.ndim != 2:
        raise ValueError("boxes_3d must be a two-dimensional array")
    if boxes.shape[0] != scores.shape[0] or scores.shape[0] != labels.shape[0]:
        raise ValueError(
            "Prediction box, score, and label counts do not match: {}, {}, {}".format(
                boxes.shape[0], scores.shape[0], labels.shape[0]
            )
        )
    return boxes, scores, labels


def _timing_summary(inference_times: List[float]) -> Dict[str, float]:
    total = float(sum(inference_times))
    count = len(inference_times)
    return {
        "total_inference_sec": total,
        "mean_inference_sec": total / count if count else 0.0,
        "median_inference_sec": (
            float(statistics.median(inference_times)) if count else 0.0
        ),
        "samples_per_second": count / total if total > 0.0 else 0.0,
    }


def _build_output(
    paths: Dict[str, Path],
    args: argparse.Namespace,
    classes: Sequence[str],
    gpu_name: str,
    torch_version: str,
    cuda_runtime: Optional[str],
    dataset_samples: int,
    predictions: List[Dict[str, Any]],
    inference_times: List[float],
    interrupted: bool,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "model": MODEL_NAME,
            "modality": "lidar",
            "classes": list(classes),
            "omnihd_root": "OmniHD-Scenes",
            "config": _portable_metadata_path(
                paths["config"], paths["omnihd_root"]
            ),
            "checkpoint": _portable_metadata_path(
                paths["checkpoint"], paths["omnihd_root"]
            ),
            "data_root": _portable_metadata_path(paths["data_root"]),
            "ann_file": _portable_metadata_path(paths["ann_file"]),
            "gpu": gpu_name,
            "torch_version": torch_version,
            "cuda_runtime": cuda_runtime,
            "dataset_samples": dataset_samples,
            "processed_samples": len(predictions),
            "start_index": args.start_index,
            "max_samples": args.max_samples,
            "interrupted": interrupted,
        },
        "predictions": predictions,
        "timing": _timing_summary(inference_times),
    }


def _atomic_pickle_dump(payload: Dict[str, Any], destination: Path) -> None:
    """Write a pickle beside its destination and atomically replace the target."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(destination))
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def run(args: argparse.Namespace) -> int:
    """Configure OmniHD, run the selected dataset range, and save predictions."""
    _validate_range_arguments(args)
    paths = _resolve_and_validate_paths(args)
    dependencies = _import_omnihd_dependencies(paths["omnihd_root"])
    torch = dependencies["torch"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; OmniHD LiDAR inference requires a GPU")
    torch.cuda.set_device(0)

    cfg = _prepare_config(dependencies["Config"], paths)
    dataset = dependencies["build_dataset"](cfg.data.test)
    dataset_samples = len(dataset)
    if dataset_samples == 0:
        raise ValueError("The supplied annotation pickle produced an empty dataset")
    if args.start_index >= dataset_samples:
        raise ValueError(
            "--start-index {} is outside a dataset with {} samples".format(
                args.start_index, dataset_samples
            )
        )

    stop_index = dataset_samples
    if args.max_samples is not None:
        stop_index = min(stop_index, args.start_index + args.max_samples)
    samples_to_process = stop_index - args.start_index
    print("Dataset samples: {}".format(dataset_samples), flush=True)
    print(
        "Processing indices [{}:{}) ({} samples)".format(
            args.start_index, stop_index, samples_to_process
        ),
        flush=True,
    )

    loader = dependencies["build_dataloader"](
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    model = dependencies["build_model"](
        cfg.model, test_cfg=cfg.get("test_cfg")
    )
    checkpoint = dependencies["load_checkpoint"](
        model, str(paths["checkpoint"]), map_location="cpu"
    )
    checkpoint_meta = checkpoint.get("meta", {}) or {}
    if "CLASSES" in checkpoint_meta:
        model.CLASSES = checkpoint_meta["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    model = dependencies["MMDataParallel"](
        model.cuda(), device_ids=[0]
    )
    model.eval()

    classes = [str(name) for name in model.module.CLASSES]
    gpu_name = str(torch.cuda.get_device_name(0))
    torch_version = str(torch.__version__)
    cuda_runtime = None if torch.version.cuda is None else str(torch.version.cuda)
    predictions: List[Dict[str, Any]] = []
    inference_times: List[float] = []

    def save_progress(interrupted: bool) -> None:
        payload = _build_output(
            paths,
            args,
            classes,
            gpu_name,
            torch_version,
            cuda_runtime,
            dataset_samples,
            predictions,
            inference_times,
            interrupted,
        )
        _atomic_pickle_dump(payload, paths["output"])

    try:
        for sample_index, data in enumerate(loader):
            if sample_index < args.start_index:
                continue
            if sample_index >= stop_index:
                break

            info = dataset.data_infos[sample_index]
            if "token" not in info:
                raise KeyError(
                    "dataset.data_infos[{}] has no token".format(sample_index)
                )
            sample_token = str(info["token"])

            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            inference_time = float(time.perf_counter() - started)

            boxes, scores, labels = _extract_prediction(result)
            number_of_detections = int(scores.shape[0])
            predictions.append(
                {
                    "sample_index": int(sample_index),
                    "sample_token": sample_token,
                    "boxes_3d": boxes,
                    "scores_3d": scores,
                    "labels_3d": labels,
                    "inference_time_sec": inference_time,
                    "number_of_detections": number_of_detections,
                }
            )
            inference_times.append(inference_time)

            completed = len(predictions)
            print(
                "[{}/{}] token={} detections={} time={:.3f}s".format(
                    completed,
                    samples_to_process,
                    sample_token,
                    number_of_detections,
                    inference_time,
                ),
                flush=True,
            )
            if completed % args.save_every == 0:
                save_progress(interrupted=False)
                print("Saved progress: {}".format(paths["output"]), flush=True)
    except KeyboardInterrupt:
        save_progress(interrupted=True)
        print(
            "\nInterrupted; saved {} completed predictions to {}".format(
                len(predictions), paths["output"]
            ),
            file=sys.stderr,
            flush=True,
        )
        return 130

    save_progress(interrupted=False)
    timing = _timing_summary(inference_times)
    print("Saved predictions: {}".format(paths["output"]), flush=True)
    print(
        "Completed {} samples; mean={:.3f}s, median={:.3f}s, throughput={:.3f}/s".format(
            len(predictions),
            timing["mean_inference_sec"],
            timing["median_inference_sec"],
            timing["samples_per_second"],
        ),
        flush=True,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, IsADirectoryError, ValueError, RuntimeError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
