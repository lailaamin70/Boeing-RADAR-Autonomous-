import json
import unittest
from unittest import mock

from app import create_app, detections_loader
from tests.base import DatasetTestCase
from tests.fake_dataset import FILE_SIZES

DETECTION = {"center": [30.0, 0.0, 0.9], "size": [2.0, 4.5, 1.8], "yaw": 0.0, "label": "car", "score": 0.9}


class RouteTestCase(DatasetTestCase):
    def setUp(self):
        super().setUp()
        app = create_app()
        app.testing = True
        self.client = app.test_client()

    def ok(self, url):
        r = self.client.get(url)
        self.addCleanup(r.close)  # send_file keeps the file handle open until the response is closed
        self.assertEqual(r.status_code, 200, f"{url} -> {r.status_code}")
        return r

    def json(self, url):
        return self.ok(url).get_json()


class PageTest(RouteTestCase):
    PAGES = ["/", "/scene/S", "/scene/S/viewer", "/sample/A", "/analysis", "/overview", "/detectability"]

    def test_every_page_renders(self):
        for url in self.PAGES:
            with self.subTest(url=url):
                html = self.ok(url).get_data(as_text=True)
                self.assertIn("TruckScenes Explorer", html)

    def test_sidebar_links_to_every_top_level_page(self):
        html = self.ok("/").get_data(as_text=True)
        for href in ("/analysis", "/overview", "/detectability"):
            self.assertIn(f'href="{href}"', html)

    def test_scene_list_shows_the_scene_and_filters_by_tag(self):
        self.assertIn("scene-0001", self.ok("/").get_data(as_text=True))
        self.assertIn("scene-0001", self.ok("/?tag=day").get_data(as_text=True))
        self.assertIn("No scenes match", self.ok("/?tag=night").get_data(as_text=True))

    def test_scene_page_has_the_ego_motion_section_and_viewer_link(self):
        html = self.ok("/scene/S").get_data(as_text=True)
        self.assertIn("/api/scene/S/ego_motion", html)
        self.assertIn("/scene/S/viewer", html)

    def test_viewer_page_embeds_one_entry_per_sample_with_camera_files(self):
        html = self.ok("/scene/S/viewer").get_data(as_text=True)
        start = html.index('type="application/json">') + len('type="application/json">')
        samples = json.loads(html[start:html.index("</script>", start)])
        self.assertEqual([s["token"] for s in samples], ["A", "B"])
        self.assertEqual(samples[0]["channels"]["CAMERA_FRONT"]["filename"], "samples/CAMERA_FRONT/cam_A.png")
        self.assertIsNone(samples[0]["channels"]["LIDAR_TOP"]["filename"])

    def test_viewer_page_has_the_controls_the_script_needs(self):
        html = self.ok("/scene/S/viewer").get_data(as_text=True)
        for element_id in ("show-ego", "show-annotations", "show-detections", "score-threshold", "camera-overlay",
                           "frame-slider", "play-btn", "plot3d-wrap", "topdown-img", "camera-select"):
            self.assertIn(f'id="{element_id}"', html, element_id)

    def test_viewer_page_exposes_all_four_endpoints_to_the_script(self):
        html = self.ok("/scene/S/viewer").get_data(as_text=True)
        for key in ("media:", "topdown:", "points:", "cameraOverlay:"):
            self.assertIn(key, html)

    def test_unknown_scene_or_sample_is_a_friendly_404(self):
        for url in ("/scene/NOPE", "/scene/NOPE/viewer", "/sample/NOPE"):
            with self.subTest(url=url):
                r = self.client.get(url)
                self.assertEqual(r.status_code, 404)
                self.assertIn("Not found", r.get_data(as_text=True))


class MediaRouteTest(RouteTestCase):
    def test_serves_a_file_from_the_dataroot(self):
        r = self.ok("/media/samples/CAMERA_FRONT/cam_A.png")
        self.assertEqual(len(r.data), FILE_SIZES["camera"])
        self.assertEqual(r.data[:8], b"\x89PNG\r\n\x1a\n")

    def test_missing_file_is_404(self):
        self.assertEqual(self.client.get("/media/samples/nothing.png").status_code, 404)

    def test_paths_cannot_escape_the_dataroot(self):
        (self.root.parent / "outside-secret.txt").write_text("secret")
        self.addCleanup((self.root.parent / "outside-secret.txt").unlink)
        for url in ("/media/../outside-secret.txt", "/media/samples/../../outside-secret.txt",
                    "/media/%2e%2e/outside-secret.txt", "/media//etc/passwd", "/media/%2fetc/passwd"):
            with self.subTest(url=url):
                r = self.client.get(url)
                self.assertNotEqual(r.status_code, 200, url)
                self.assertNotIn(b"secret", r.data)

    def test_directories_are_not_served(self):
        self.assertEqual(self.client.get("/media/samples").status_code, 404)


class SceneViewerEndpointTest(RouteTestCase):
    def test_topdown_png(self):
        r = self.ok("/sample/A/topdown.png?sensor=lidar&sensor=radar&annotations=1&ego=1")
        self.assertEqual(r.mimetype, "image/png")
        self.assertEqual(r.data[:4], b"\x89PNG")

    def test_overlay_flags_change_the_image(self):
        plain = self.ok("/sample/A/topdown.png?sensor=lidar").data
        self.assertNotEqual(plain, self.ok("/sample/A/topdown.png?sensor=lidar&ego=1").data)
        self.assertNotEqual(plain, self.ok("/sample/A/topdown.png?sensor=lidar&annotations=1").data)

    def test_missing_or_invalid_sensors_fall_back_to_both(self):
        both = self.ok("/sample/A/points.json?sensor=lidar&sensor=radar").get_json()
        self.assertEqual(set(self.json("/sample/A/points.json")["modalities"]), {"lidar", "radar"})
        self.assertEqual(set(self.json("/sample/A/points.json?sensor=bogus")["modalities"]), {"lidar", "radar"})
        self.assertEqual(set(both["modalities"]), {"lidar", "radar"})
        self.assertEqual(set(self.json("/sample/A/points.json?sensor=radar&sensor=bogus")["modalities"]), {"radar"})

    def test_points_json_flags(self):
        p = self.json("/sample/A/points.json?sensor=lidar")
        self.assertEqual((p["annotations"], p["detections"], p["ego"]), ([], [], None))
        p = self.json("/sample/A/points.json?sensor=lidar&annotations=1&ego=1")
        self.assertEqual(len(p["annotations"]), 5)
        self.assertEqual(len(p["ego"]["sensors"]), 3)

    def test_score_threshold_reaches_the_detector_and_junk_is_ignored(self):
        with mock.patch.object(detections_loader, "get_predictions", return_value=[DETECTION]) as get:
            self.json("/sample/A/points.json?sensor=lidar&detections=1&min_score=0.55")
            self.assertEqual(get.call_args.args, ("lidar", "A", 0.55))
            self.json("/sample/A/points.json?sensor=lidar&detections=1&min_score=banana")
            self.assertEqual(get.call_args.args, ("lidar", "A", None))  # junk = use the per-modality default
            self.json("/sample/A/points.json?sensor=lidar&detections=1")
            self.assertEqual(get.call_args.args, ("lidar", "A", None))

    def test_detections_are_only_computed_when_asked_for(self):
        with mock.patch.object(detections_loader, "get_predictions", return_value=[DETECTION]) as get:
            self.json("/sample/A/points.json?sensor=lidar")
            get.assert_not_called()

    def test_camera_overlay(self):
        o = self.json("/sample/A/camera/CAMERA_FRONT/overlay.json?annotations=1")
        self.assertEqual((o["width"], o["height"]), (1600, 900))
        self.assertEqual(len(o["annotations"]), 3)
        self.assertEqual(self.json("/sample/A/camera/CAMERA_FRONT/overlay.json")["annotations"], [])

    def test_camera_overlay_with_detections(self):
        with mock.patch.object(detections_loader, "get_predictions", return_value=[DETECTION]):
            o = self.json("/sample/A/camera/CAMERA_FRONT/overlay.json?detections=1&sensor=lidar")
        self.assertEqual([d["modality"] for d in o["detections"]], ["lidar"])

    def test_unknown_sample_or_channel_is_404(self):
        for url in ("/sample/NOPE/topdown.png", "/sample/NOPE/points.json",
                    "/sample/A/camera/LIDAR_TOP/overlay.json", "/sample/A/camera/NOPE/overlay.json"):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 404)


class SensorPlotRouteTest(RouteTestCase):
    """/sample/<token>/plot/<channel>: the single-sensor bird's-eye plot on the sample page."""

    def test_radar_and_lidar_channels_render_a_png(self):
        for channel in ("LIDAR_TOP", "RADAR_FRONT"):
            with self.subTest(channel=channel):
                r = self.ok(f"/sample/A/plot/{channel}")
                self.assertEqual((r.mimetype, r.data[:4]), ("image/png", b"\x89PNG"))

    def test_lidar_and_radar_plots_differ(self):
        self.assertNotEqual(self.ok("/sample/A/plot/LIDAR_TOP").data, self.ok("/sample/A/plot/RADAR_FRONT").data)

    def test_camera_channels_are_rejected_and_unknown_ones_are_404(self):
        self.assertEqual(self.client.get("/sample/A/plot/CAMERA_FRONT").status_code, 400)
        self.assertEqual(self.client.get("/sample/A/plot/NOPE").status_code, 404)

    def test_sample_page_references_a_plot_per_point_sensor_and_the_camera_image(self):
        html = self.ok("/sample/A").get_data(as_text=True)
        self.assertIn("/sample/A/plot/LIDAR_TOP", html)
        self.assertIn("/sample/A/plot/RADAR_FRONT", html)
        self.assertIn("/media/samples/CAMERA_FRONT/cam_A.png", html)
        self.assertIn("/sample/B", html)  # link to the next sample


class DatasetApiTest(RouteTestCase):
    def test_overview_and_volume(self):
        self.assertEqual(self.json("/api/overview")["counts"]["annotations"], 4)
        vol = self.json("/api/data_volume")
        self.assertEqual({c["channel"] for c in vol["channels"]}, {"LIDAR_TOP", "RADAR_FRONT", "CAMERA_FRONT"})

    def test_volume_refresh_flag_forces_a_rescan(self):
        self.json("/api/data_volume")
        (self.root / "samples/CAMERA_FRONT/cam_A.png").unlink()
        self.assertEqual(self.json("/api/data_volume")["missing_files"], 0)
        self.assertEqual(self.json("/api/data_volume?refresh=1")["missing_files"], 1)

    def test_detectability_filters(self):
        self.assertEqual(self.json("/api/detectability")["total"], 4)
        self.assertEqual(self.json("/api/detectability?category=vehicle.car")["total"], 2)
        self.assertEqual(self.json("/api/detectability?category=vehicle.car&category=vehicle.truck")["total"], 3)
        self.assertEqual(self.json("/api/detectability?tag=night")["total"], 0)

    def test_detectability_thresholds_and_camera_flag(self):
        d = self.json("/api/detectability?min_lidar=9&min_radar=2&camera=1&min_visibility=4")
        self.assertEqual(d["params"], {"min_lidar": 9, "min_radar": 2, "include_camera": True, "min_visibility": 4})
        self.assertIn("camera only", d["combos"])
        self.assertNotIn("camera only", self.json("/api/detectability")["combos"])

    def test_junk_and_out_of_range_numbers_fall_back_instead_of_erroring(self):
        d = self.json("/api/detectability?min_lidar=abc&min_radar=-5&min_visibility=zzz")
        self.assertEqual((d["params"]["min_lidar"], d["params"]["min_radar"], d["params"]["min_visibility"]), (5, 1, 2))
        self.assertEqual(self.json("/api/detectability?min_lidar=0")["params"]["min_lidar"], 1)  # clamped to >= 1

    def test_ego_motion(self):
        em = self.json("/api/scene/S/ego_motion")
        self.assertEqual(em["stats"]["mean_speed_kmh"], 36.0)
        self.assertEqual(self.client.get("/api/scene/NOPE/ego_motion").status_code, 404)

    def test_analysis_api(self):
        d = self.json("/api/analysis?metric=num_samples&metric=visibility&group_by=tag")
        self.assertEqual(d["labels"], ["clear", "day"])
        self.assertEqual(set(d["metrics"]), {"num_samples", "visibility"})

    def test_analysis_api_defaults_and_bad_group_by(self):
        self.assertIn("sweep_point_count", self.json("/api/analysis")["metrics"])  # default metric
        self.assertEqual(self.json("/api/analysis?metric=num_samples&group_by=banana")["labels"], ["clear", "day"])

    def test_analysis_expensive_flag(self):
        off = self.json("/api/analysis?metric=sweep_point_count&tag=day")
        on = self.json("/api/analysis?metric=sweep_point_count&tag=day&expensive=1")
        self.assertEqual(off["metrics"]["sweep_point_count"]["series"]["lidar"], [None])
        self.assertEqual(on["metrics"]["sweep_point_count"]["series"]["lidar"], [200.0])


if __name__ == "__main__":
    unittest.main()
