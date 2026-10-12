import io
import json
import math
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

from app import detections_loader, scene_viewer as sv
from config import Config
from tests.base import DatasetTestCase

# one lidar detection 30 m straight ahead
DETECTION = {"center": [30.0, 0.0, 0.9], "size": [2.0, 4.5, 1.8], "yaw": 0.0, "label": "car", "score": 0.9}


def with_detections(preds_by_modality):
    return mock.patch.object(
        detections_loader, "get_predictions",
        side_effect=lambda modality, token, threshold=None: list(preds_by_modality.get(modality, [])),
    )


class EgoVehicleTest(DatasetTestCase):
    def test_sensor_markers_sit_at_the_calibrated_mounts(self):
        ego = sv.get_ego_vehicle("A")
        xyz = {s["channel"]: s["xyz"] for s in ego["sensors"]}
        self.assertEqual(xyz, {"LIDAR_TOP": [4.0, 0.0, 3.8], "RADAR_FRONT": [5.1, 0.0, 0.7], "CAMERA_FRONT": [4.6, 0.0, 3.0]})
        self.assertEqual({s["channel"]: s["modality"] for s in ego["sensors"]}["CAMERA_FRONT"], "camera")

    def test_truck_boxes_come_from_config(self):
        spec = {"name": "cab", "center": [3.0, 0.0, 1.5], "size": [2.5, 2.0, 3.0]}
        with mock.patch.object(Config, "EGO_TRUCK_BOXES", [spec]):
            (box,) = sv.get_ego_vehicle("A")["boxes"]
        self.assertEqual(box["name"], "cab")
        np.testing.assert_allclose(box["corners"].mean(axis=0), [3.0, 0.0, 1.5])
        np.testing.assert_allclose(box["corners"].max(axis=0) - box["corners"].min(axis=0), [2.0, 2.5, 3.0])  # l, w, h

    def test_default_truck_box_encloses_the_origin_region_and_has_eight_corners(self):
        (box,) = sv.get_ego_vehicle("A")["boxes"]
        self.assertEqual(box["corners"].shape, (8, 3))
        self.assertLess(box["corners"][:, 0].min(), 0.0 + 1e-9)   # reaches behind the ego origin
        self.assertGreater(box["corners"][:, 0].max(), 4.0)       # reaches ahead past the sensors


class DetectionsForSampleTest(DatasetTestCase):
    def test_no_modalities_means_no_detections(self):
        with with_detections({"lidar": [DETECTION]}):
            self.assertEqual(sv.get_detections_for_sample("A", []), [])

    def test_boxes_carry_label_score_and_source_modality(self):
        with with_detections({"lidar": [DETECTION]}):
            (det,) = sv.get_detections_for_sample("A", ["lidar", "radar"])
        self.assertEqual((det["category"], det["modality"], det["score"]), ("car", "lidar", 0.9))
        np.testing.assert_allclose(det["corners"].mean(axis=0), [30.0, 0.0, 0.9])

    def test_yaw_is_applied_to_the_box(self):
        turned = dict(DETECTION, yaw=math.pi / 2)
        with with_detections({"lidar": [turned]}):
            (det,) = sv.get_detections_for_sample("A", ["lidar"])
        extent = det["corners"].max(axis=0) - det["corners"].min(axis=0)
        np.testing.assert_allclose(extent, [2.0, 4.5, 1.8], atol=1e-9)  # length now runs along y

    def test_score_threshold_is_passed_through_per_modality(self):
        with with_detections({}) as get:
            sv.get_detections_for_sample("A", ["lidar", "radar"], 0.42)
        self.assertEqual([c.args for c in get.call_args_list], [("lidar", "A", 0.42), ("radar", "A", 0.42)])


class PointsPayloadTest(DatasetTestCase):
    def test_points_are_in_the_ego_frame_and_only_for_selected_sensors(self):
        p = sv.get_points_payload("A", ["lidar"], False)
        self.assertEqual(set(p["modalities"]), {"lidar"})
        self.assertEqual(len(p["modalities"]["lidar"]["x"]), 100)
        self.assertAlmostEqual(min(p["modalities"]["lidar"]["x"]), 1.0 + 4.0)  # sensor x + mount x
        self.assertAlmostEqual(min(p["modalities"]["lidar"]["z"]), 3.8)

    def test_overlays_are_empty_unless_requested(self):
        p = sv.get_points_payload("A", ["lidar", "radar"], False)
        self.assertEqual((p["annotations"], p["detections"], p["ego"]), ([], [], None))

    def test_annotations_are_ego_frame_boxes_with_categories(self):
        p = sv.get_points_payload("A", ["lidar"], True)
        self.assertEqual(len(p["annotations"]), 5)  # the viewer still draws the ego trailer
        a1 = next(a for a in p["annotations"] if a["category"] == "vehicle.car" and abs(np.mean([c[0] for c in a["corners"]]) - 20) < 1e-6)
        self.assertEqual(len(a1["corners"]), 8)
        self.assertEqual({a["category"] for a in p["annotations"]},
                         {"vehicle.car", "vehicle.truck", "human.pedestrian.adult", "vehicle.ego_trailer"})

    def test_ego_overlay_has_the_truck_and_every_sensor(self):
        p = sv.get_points_payload("A", ["lidar"], False, show_ego=True)
        self.assertEqual([b["name"] for b in p["ego"]["boxes"]], ["tractor"])
        self.assertEqual(len(p["ego"]["sensors"]), 3)

    def test_detections_overlay(self):
        with with_detections({"lidar": [DETECTION]}):
            p = sv.get_points_payload("A", ["lidar"], False, show_detections=True)
        self.assertEqual([d["modality"] for d in p["detections"]], ["lidar"])

    def test_large_clouds_are_downsampled_to_the_cap(self):
        with mock.patch.object(sv, "MAX_3D_POINTS_PER_MODALITY", 30):
            p = sv.get_points_payload("A", ["lidar"], False)
        self.assertEqual(len(p["modalities"]["lidar"]["x"]), 30)

    def test_payload_is_json_serialisable(self):
        with with_detections({"lidar": [DETECTION]}):
            json.dumps(sv.get_points_payload("A", ["lidar", "radar"], True, True, 0.3, True))

    def test_unknown_sample_raises_keyerror(self):
        with self.assertRaises(KeyError):
            sv.get_points_payload("nope", ["lidar"], False)


class TopdownPngTest(DatasetTestCase):
    def render(self, *args, **kwargs):
        return sv.render_topdown_png("A", *args, **kwargs).read()

    def test_returns_a_decodable_png(self):
        png = self.render(["lidar", "radar"], False)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        img = plt.imread(io.BytesIO(png), format="png")
        self.assertEqual(img.ndim, 3)
        self.assertGreater(img.shape[0], 100)

    def test_every_overlay_combination_renders(self):
        with with_detections({"lidar": [DETECTION], "radar": [DETECTION]}):
            for sensors in (["lidar"], ["radar"], ["lidar", "radar"]):
                for ann in (False, True):
                    for det in (False, True):
                        for ego in (False, True):
                            png = self.render(sensors, ann, det, None, show_ego=ego)
                            self.assertEqual(png[:4], b"\x89PNG", (sensors, ann, det, ego))

    def test_each_overlay_actually_changes_the_image(self):
        base = self.render(["lidar"], False)
        with with_detections({"lidar": [DETECTION]}):
            self.assertNotEqual(base, self.render(["lidar"], False, show_ego=True))
            self.assertNotEqual(base, self.render(["lidar"], True))
            self.assertNotEqual(base, self.render(["lidar"], False, show_detections=True))

    def test_open_figures_are_released(self):
        before = len(plt.get_fignums())
        self.render(["lidar"], True, show_ego=True)
        self.assertEqual(len(plt.get_fignums()), before)  # memory leak

    def test_sensor_choice_changes_the_image(self):
        self.assertNotEqual(self.render(["lidar"], False), self.render(["radar"], False))


class CameraOverlayTest(DatasetTestCase):
    def overlay(self, **kw):
        args = dict(show_annotations=True, show_detections=False, detection_modalities=["lidar"], min_score=None)
        args.update(kw)
        return sv.get_camera_overlay_payload("A", "CAMERA_FRONT", **args)

    def test_image_size_comes_from_the_sample_data(self):
        o = self.overlay()
        self.assertEqual((o["width"], o["height"]), (1600, 900))

    def test_only_boxes_in_front_of_the_camera_are_projected(self):
        o = self.overlay()
        # a1/a2/a3 are ahead; the pedestrian is beside the cab (behind the camera); the ego trailer is behind it
        self.assertEqual(len(o["annotations"]), 3)
        self.assertEqual({a["category"] for a in o["annotations"]}, {"vehicle.car", "vehicle.truck"})

    def test_projected_boxes_have_eight_pixel_corners_inside_the_frame(self):
        for a in self.overlay()["annotations"]:
            uv = np.array(a["uv"])
            self.assertEqual(uv.shape, (8, 2))
            self.assertTrue((uv[:, 0] > 0).all() and (uv[:, 0] < 1600).all())
            self.assertTrue((uv[:, 1] > 0).all() and (uv[:, 1] < 900).all())

    def test_nearest_box_is_the_biggest_on_screen(self):
        widths = sorted(np.ptp(np.array(a["uv"])[:, 0]) for a in self.overlay()["annotations"])
        self.assertEqual(len(widths), 3)
        self.assertGreater(widths[-1], widths[0] * 2)  # 20 m vs 60 m away

    def test_overlays_follow_the_flags(self):
        self.assertEqual(self.overlay(show_annotations=False)["annotations"], [])
        self.assertEqual(self.overlay()["detections"], [])  # detections off
        with with_detections({"lidar": [DETECTION]}):
            o = self.overlay(show_annotations=False, show_detections=True)
            self.assertEqual([d["modality"] for d in o["detections"]], ["lidar"])
            self.assertEqual(o["detections"][0]["score"], 0.9)
            self.assertEqual(self.overlay(show_detections=True, detection_modalities=[])["detections"], [])

    def test_non_camera_and_unknown_channels_raise_keyerror(self):
        for channel in ("LIDAR_TOP", "NO_SUCH_CHANNEL"):
            with self.assertRaises(KeyError):
                sv.get_camera_overlay_payload("A", channel)

    def test_payload_is_json_serialisable(self):
        with with_detections({"lidar": [DETECTION]}):
            json.dumps(self.overlay(show_detections=True))


if __name__ == "__main__":
    unittest.main()
