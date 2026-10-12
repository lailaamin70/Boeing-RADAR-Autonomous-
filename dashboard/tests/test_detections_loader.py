import json
import pickle
import unittest
from pathlib import Path

import numpy as np

from app import detections_loader as dl
from config import Config
from tests.base import DatasetTestCase

CLASSES = ["car", "pedestrian", "rider", "large_vehicle"]


def write_predictions(path, frames, classes=CLASSES):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"metadata": {"classes": classes}, "predictions": frames, "timing": {}}, f)


def frame(token, boxes, scores, labels):
    return {
        "sample_index": 0, "sample_token": token,
        "boxes_3d": np.array(boxes, dtype=np.float32),
        "scores_3d": np.array(scores, dtype=np.float32),
        "labels_3d": np.array(labels),
    }


# [x, y, z, w, l, h, yaw, vx, vy]  - the confirmed OmniHD-PointPillars layout
CAR = [10.0, 2.0, 0.1, 2.1, 4.5, 1.8, 1.57, 5.0, 0.1]
PED = [-5.0, 1.0, 0.2, 0.6, 0.6, 1.7, 0.0, 0.0, 0.0]


class GetPredictionsTest(DatasetTestCase):
    def setUp(self):
        super().setUp()
        write_predictions(
            Config.DETECTIONS_PATHS["lidar"],
            [frame("A", [CAR, PED], [0.9, 0.1], [0, 1]), frame("B", [], [], [])],
        )

    def test_box_columns_are_unpacked_in_the_confirmed_order(self):
        (det,) = dl.get_predictions("lidar", "A")
        self.assertEqual(det["label"], "car")
        self.assertAlmostEqual(det["center"][0], 10.0, places=4)
        self.assertEqual([round(v, 4) for v in det["size"]], [2.1, 4.5, 1.8])  # w, l, h
        self.assertAlmostEqual(det["yaw"], 1.57, places=4)
        self.assertAlmostEqual(det["score"], 0.9, places=4)

    def test_default_threshold_for_lidar_drops_the_low_score_box(self):
        self.assertEqual([d["label"] for d in dl.get_predictions("lidar", "A")], ["car"])

    def test_explicit_threshold_overrides_the_default(self):
        labels = [d["label"] for d in dl.get_predictions("lidar", "A", score_threshold=0.0)]
        self.assertEqual(labels, ["car", "pedestrian"])
        self.assertEqual(dl.get_predictions("lidar", "A", score_threshold=0.95), [])

    def test_radar_default_threshold_is_stricter_than_lidar(self):
        self.assertGreater(dl.DEFAULT_SCORE_THRESHOLD["radar"], dl.DEFAULT_SCORE_THRESHOLD["lidar"])
        write_predictions(Config.DETECTIONS_PATHS["radar"], [frame("A", [CAR], [0.45], [0])])
        self.assertEqual(dl.get_predictions("radar", "A"), [])  # 0.45 passes lidar's 0.3 but not radar's 0.6
        self.assertEqual(len(dl.get_predictions("radar", "A", score_threshold=0.3)), 1)

    def test_unknown_sample_and_empty_sample_give_empty_lists(self):
        self.assertEqual(dl.get_predictions("lidar", "no-such-sample"), [])
        self.assertEqual(dl.get_predictions("lidar", "B"), [])

    def test_missing_pickle_degrades_to_empty_instead_of_crashing(self):
        self.assertEqual(dl.get_predictions("radar", "A"), [])  # no radar file was written

    def test_unknown_modality_is_empty(self):
        self.assertEqual(dl.get_predictions("camera", "A"), [])

    def test_label_index_outside_the_class_list_falls_back_to_the_number(self):
        write_predictions(Config.DETECTIONS_PATHS["lidar"], [frame("A", [CAR], [0.9], [9])])
        dl._cache.clear()
        dl._classes.clear()
        self.assertEqual(dl.get_predictions("lidar", "A")[0]["label"], "9")

    def test_output_is_plain_python_and_json_serialisable(self):
        det = dl.get_predictions("lidar", "A")[0]  # numpy float32 in the pickle must not leak out
        json.dumps(det)
        self.assertIsInstance(det["score"], float)

    def test_file_is_read_once_then_served_from_memory(self):
        dl.get_predictions("lidar", "A")
        Path(Config.DETECTIONS_PATHS["lidar"]).unlink()
        self.assertEqual(len(dl.get_predictions("lidar", "A")), 1)


if __name__ == "__main__":
    unittest.main()
