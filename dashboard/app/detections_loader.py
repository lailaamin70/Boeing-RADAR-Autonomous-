"""
Loads OmniHD-PointPillars detection output
(results/object_detection/omnihd/{lidar,radar}_mini_predictions.pkl) and reads
"""
import pickle
import threading

from config import Config

_lock = threading.Lock()
_cache = {}  # {sample_token: {"boxes": arr, "scores": arr, "labels": arr}}
_classes = {}  # list[str]

DEFAULT_SCORE_THRESHOLD = {"lidar": 0.3, "radar": 0.6}


def _load(modality):
    if modality in _cache:
        return
    with _lock:
        if modality in _cache:
            return
        path = Config.DETECTIONS_PATHS.get(modality)
        if path is None or not path.is_file():
            _cache[modality] = {}
            _classes[modality] = []
            return

        with open(path, "rb") as f:
            raw = pickle.load(f)

        _classes[modality] = raw.get("metadata", {}).get("classes", [])
        by_token = {}
        for frame in raw.get("predictions", []):
            by_token[frame["sample_token"]] = {
                "boxes": frame["boxes_3d"],
                "scores": frame["scores_3d"],
                "labels": frame["labels_3d"],
            }
        _cache[modality] = by_token


def get_predictions(modality, sample_token, score_threshold=None):
    """
    Detections for sample filtered by score. Format:
    {"center": [x,y,z], "size": [w,l,h], "yaw": float, "label": str, "score": float}
    """
    _load(modality)
    entry = _cache.get(modality, {}).get(sample_token)
    if entry is None:
        return []

    threshold = (
        DEFAULT_SCORE_THRESHOLD.get(modality, 0.3) if score_threshold is None else score_threshold
    )
    classes = _classes.get(modality, [])

    out = []
    for box, score, label in zip(entry["boxes"], entry["scores"], entry["labels"]):
        if score < threshold:
            continue
        out.append(
            {
                "center": [float(box[0]), float(box[1]), float(box[2])],
                "size": [float(box[3]), float(box[4]), float(box[5])],
                "yaw": float(box[6]),
                "label": classes[label] if label < len(classes) else str(label),
                "score": float(score),
            }
        )
    return out
