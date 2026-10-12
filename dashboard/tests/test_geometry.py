import math
import re
import unittest
from pathlib import Path

import numpy as np

from app import geometry
from tests.base import DatasetTestCase
from tests.fake_dataset import CAM_Q

JS = Path(__file__).resolve().parents[1] / "app" / "static" / "js" / "scene_viewer.js"
QUAT_90_YAW = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]


def extent(corners):
    return corners.max(axis=0) - corners.min(axis=0)


class BoxCornersTest(unittest.TestCase):
    def test_axis_aligned_box_has_the_right_size_and_centre(self):
        c = geometry.box_corners([10.0, 2.0, 1.0], [2.0, 4.0, 1.5], [1, 0, 0, 0])  # w, l, h
        self.assertEqual(c.shape, (8, 3))
        np.testing.assert_allclose(extent(c), [4.0, 2.0, 1.5])  # x=length, y=width, z=height
        np.testing.assert_allclose(c.mean(axis=0), [10.0, 2.0, 1.0])

    def test_top_face_footprint_order_traces_a_rectangle(self):
        # scene_viewer draws the top-down outline from corners [0, 1, 5, 4]
        c = geometry.box_corners([0, 0, 0], [2.0, 6.0, 1.5], [1, 0, 0, 0])
        x, y = c[[0, 1, 5, 4], 0], c[[0, 1, 5, 4], 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))  # shoelace
        self.assertAlmostEqual(area, 2.0 * 6.0)

    def test_yaw_rotation_swaps_the_horizontal_extents(self):
        c = geometry.box_corners([0, 0, 0], [2.0, 6.0, 1.5], geometry.yaw_to_quaternion(math.pi / 2))
        np.testing.assert_allclose(extent(c), [2.0, 6.0, 1.5], atol=1e-9)

    def test_rotation_leaves_the_centre_alone(self):
        c = geometry.box_corners([7.0, -3.0, 0.9], [2.0, 4.5, 1.8], geometry.yaw_to_quaternion(0.7))
        np.testing.assert_allclose(c.mean(axis=0), [7.0, -3.0, 0.9], atol=1e-9)


class YawToQuaternionTest(unittest.TestCase):
    def test_zero_yaw_is_identity(self):
        np.testing.assert_allclose(geometry.yaw_to_quaternion(0.0), [1, 0, 0, 0])

    def test_is_a_unit_quaternion_about_z(self):
        for yaw in (-3.0, -0.5, 0.3, 2.0, 4.7):  # OmniHD yaws are not wrapped to [-pi, pi]
            q = geometry.yaw_to_quaternion(yaw)
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0)
            self.assertEqual((q[1], q[2]), (0.0, 0.0))


class JsCornerOrderTest(unittest.TestCase):
    """scene_viewer.js hard-codes the box edges and mesh triangles; they must match box_corners' order."""

    @classmethod
    def setUpClass(cls):
        cls.src = JS.read_text()
        cls.corners = geometry.box_corners([0, 0, 0], [2.0, 4.0, 1.5], [1, 0, 0, 0])

    def _js_pairs(self, name):
        body = re.search(rf"const {name} = \[(.*?)\n\];", self.src, re.S).group(1)
        return [tuple(map(int, m)) for m in re.findall(r"\[(\d+), (\d+)\]", body)]

    def test_box_edges_are_the_twelve_edges_of_a_box(self):
        edges = self._js_pairs("BOX_EDGES")
        self.assertEqual(len(edges), 12)
        self.assertEqual(len({frozenset(e) for e in edges}), 12, "duplicate edge")
        lengths = sorted(round(float(np.linalg.norm(self.corners[a] - self.corners[b])), 6) for a, b in edges)
        self.assertEqual(lengths, sorted([2.0] * 4 + [1.5] * 4 + [4.0] * 4))  # 4 each of w, h, l

    def test_mesh_triangles_cover_all_six_faces(self):
        tri = {}
        for axis in "ijk":
            body = re.search(rf"{axis}: \[([\d, ]+)\]", self.src).group(1)
            tri[axis] = [int(v) for v in body.split(",")]
        triangles = list(zip(tri["i"], tri["j"], tri["k"]))
        self.assertEqual(len(triangles), 12)
        faces = set()
        for t in triangles:
            pts = self.corners[list(t)]
            flat = [(ax, float(pts[0, ax])) for ax in range(3) if np.ptp(pts[:, ax]) < 1e-9]
            self.assertEqual(len(flat), 1, f"triangle {t} is not on a single box face")
            faces.add(flat[0])
        self.assertEqual(len(faces), 6, "triangles must cover each of the six faces")


class FrameTransformTest(unittest.TestCase):
    def setUp(self):
        self.calib = {"translation": [1.0, -2.0, 0.5], "rotation": [0.9239, 0.0, 0.2, 0.3827]}

    def test_sensor_to_ego_and_back_is_the_identity(self):
        pts = np.random.default_rng(0).uniform(-30, 30, (50, 3))
        back = geometry.points_ego_to_sensor(geometry.points_sensor_to_ego(pts, self.calib), self.calib)
        np.testing.assert_allclose(back, pts, atol=1e-9)

    def test_sensor_origin_lands_on_the_calibrated_translation(self):
        out = geometry.points_sensor_to_ego(np.zeros((1, 3)), self.calib)
        np.testing.assert_allclose(out[0], self.calib["translation"])

    def test_empty_inputs_pass_straight_through(self):
        empty = np.zeros((0, 3))
        self.assertEqual(geometry.points_sensor_to_ego(empty, self.calib).shape, (0, 3))
        self.assertEqual(geometry.points_ego_to_sensor(empty, self.calib).shape, (0, 3))


class CameraProjectionTest(DatasetTestCase):
    """project_box_corners uses the fixture camera: at (4.6, 0, 3.0), fx=fy=1000, pp=(800, 450)."""

    def corners(self, center_ego, yaw=0.0):
        return geometry.box_corners(center_ego, [2.0, 4.5, 1.8], geometry.yaw_to_quaternion(yaw))

    def test_quaternion_maps_camera_forward_to_ego_forward(self):
        r = geometry.Quaternion(CAM_Q).rotation_matrix
        np.testing.assert_allclose(r @ [0, 0, 1], [1, 0, 0], atol=1e-9)  # camera z -> ego x
        np.testing.assert_allclose(r @ [1, 0, 0], [0, -1, 0], atol=1e-9)  # camera right -> ego -y
        np.testing.assert_allclose(r @ [0, 1, 0], [0, 0, -1], atol=1e-9)  # camera down -> ego -z

    def test_box_dead_ahead_projects_centred_and_inside_the_image(self):
        uv = geometry.project_box_corners(self.corners([20.0, 0.0, 0.9]), "cam_A")
        self.assertEqual(uv.shape, (8, 2))
        self.assertLess(abs(uv[:, 0].mean() - 800), 1.0)
        self.assertTrue((uv[:, 0] > 0).all() and (uv[:, 0] < 1600).all())
        self.assertTrue((uv[:, 1] > 450).all() and (uv[:, 1] < 900).all())  # below the horizon

    def test_object_to_the_left_projects_left_of_centre(self):
        uv = geometry.project_box_corners(self.corners([30.0, 5.0, 0.9]), "cam_A")  # +y is ego-left
        self.assertLess(uv[:, 0].mean(), 800)
        uv = geometry.project_box_corners(self.corners([30.0, -5.0, 0.9]), "cam_A")
        self.assertGreater(uv[:, 0].mean(), 800)

    def test_farther_objects_appear_smaller(self):
        near = geometry.project_box_corners(self.corners([15.0, 0.0, 0.9]), "cam_A")
        far = geometry.project_box_corners(self.corners([60.0, 0.0, 0.9]), "cam_A")
        self.assertGreater(np.ptp(near[:, 0]), np.ptp(far[:, 0]) * 2)

    def test_box_behind_the_camera_is_rejected(self):
        self.assertIsNone(geometry.project_box_corners(self.corners([-10.0, 0.0, 0.9]), "cam_A"))

    def test_box_straddling_the_camera_plane_is_rejected_not_flipped(self):
        # centred 0.4 m in front of the camera, but 4.5 m long -> half of it is behind
        self.assertIsNone(geometry.project_box_corners(self.corners([5.0, 0.0, 0.9]), "cam_A"))


class AnnotationFrameTest(DatasetTestCase):
    def ann(self, token="a1"):
        return self.ds.trucksc.get("sample_annotation", token)

    def test_global_to_ego_is_a_plain_offset_when_the_truck_faces_forward(self):
        c = geometry.annotation_to_ego_frame(self.ann("a1"), "lidar_A").mean(axis=0)
        np.testing.assert_allclose(c, [20.0, 0.0, 0.9], atol=1e-9)  # ego at (100,200), a1 at (120,200)

    def test_global_to_ego_accounts_for_the_trucks_heading(self):
        # truck turned 90 deg left: something that was dead ahead in global +x is now on its RIGHT
        self.ds.trucksc.get("ego_pose", "ep0")["rotation"] = QUAT_90_YAW
        c = geometry.annotation_to_ego_frame(self.ann("a1"), "lidar_A").mean(axis=0)
        np.testing.assert_allclose(c, [0.0, -20.0, 0.9], atol=1e-6)

    def test_sensor_frame_is_the_ego_frame_shifted_by_the_mount(self):
        ego = geometry.annotation_to_ego_frame(self.ann(), "lidar_A").mean(axis=0)
        sensor = geometry.annotation_to_sensor_frame(self.ann(), "lidar_A").mean(axis=0)
        np.testing.assert_allclose(sensor, ego - [4.0, 0.0, 3.8], atol=1e-9)  # lidar mount, identity rotation


class PointsEgoFrameTest(DatasetTestCase):
    def test_lidar_points_are_shifted_by_the_sensor_mount(self):
        xyz, intensity = geometry.load_points_ego_frame("lidar_A", "lidar")
        self.assertEqual(xyz.shape, (100, 3))
        np.testing.assert_allclose(xyz[0], [1.0 + 4.0, 0.0, 3.8])
        np.testing.assert_allclose(intensity, 1.0)

    def test_radar_points_use_the_confirmed_seven_row_schema(self):
        xyz, rcs = geometry.load_points_ego_frame("radar_A", "radar")
        self.assertEqual(xyz.shape, (10, 3))
        np.testing.assert_allclose(xyz[0], [5.0 + 5.1, 0.0, 0.7])
        np.testing.assert_allclose(rcs, 10.0)  # rcs comes from column 6, not a velocity column

    def test_camera_is_not_a_point_modality(self):
        with self.assertRaises(ValueError):
            geometry.load_points_ego_frame("cam_A", "camera")


if __name__ == "__main__":
    unittest.main()
