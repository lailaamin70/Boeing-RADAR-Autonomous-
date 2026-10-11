import json
import os
import unittest
from unittest import mock

from app import analytics
from config import Config
from tests.base import DatasetTestCase


class CheapMetricsTest(DatasetTestCase):
    def setUp(self):
        super().setUp()
        self.m = analytics.compute_scene_metrics("S")

    def test_scene_and_camera_counts(self):
        self.assertEqual(self.m["general"]["num_samples"], 2)
        self.assertEqual(self.m["camera"]["camera_frame_count"], 2)

    def test_annotated_point_counts_exclude_the_ego_trailer(self):
        # a5 (the ego trailer, 999 pts) must not skew these: lidar (50+8+2+20)/4, radar (3+0+1+0)/4
        self.assertEqual(self.m["lidar"]["annotated_lidar_pts"], 20.0)
        self.assertEqual(self.m["radar"]["annotated_radar_pts"], 1.0)

    def test_visibility_averages_the_annotated_levels_only(self):
        # a1=4, a2=3, a3=1; a4 has none and the ego trailer is excluded
        self.assertAlmostEqual(self.m["general"]["visibility"], (4 + 3 + 1) / 3)

    def test_expensive_metrics_are_absent_unless_requested(self):
        self.assertNotIn("sweep_point_count", self.m["lidar"])

    def test_cheap_results_are_cached_in_memory(self):
        self.assertIn("S", analytics._cheap_cache)
        with mock.patch.object(analytics, "_compute_cheap", side_effect=AssertionError("recomputed")):
            analytics.compute_scene_metrics("S")


class ExpensiveMetricsTest(DatasetTestCase):
    def setUp(self):
        super().setUp()
        self.m = analytics.compute_scene_metrics("S", include_expensive=True)

    def test_point_metrics_average_across_every_keyframe_not_just_the_first(self):
        # lidar: 100 pts in frame A, 300 in frame B -> 200 (a first-frame-only version would say 100)
        self.assertEqual(self.m["lidar"]["sweep_point_count"], 200.0)
        self.assertEqual(self.m["radar"]["sweep_point_count"], 15.0)  # (10 + 20) / 2
        self.assertEqual(self.m["lidar"]["range_coverage"], 60.0)     # (50 + 70) / 2
        self.assertEqual(self.m["radar"]["range_coverage"], 40.0)     # (30 + 50) / 2

    def test_azimuth_spread_is_zero_for_points_on_the_axis(self):
        self.assertEqual(self.m["lidar"]["azimuth_coverage"], 0.0)

    def test_radar_speed_and_rcs_use_the_right_columns(self):
        self.assertEqual(self.m["radar"]["mean_speed"], 5.5)  # |(3,4)| = 5 and |(0,6)| = 6
        self.assertEqual(self.m["radar"]["mean_rcs"], 15.0)

    def test_cheap_metrics_are_still_present(self):
        self.assertEqual(self.m["general"]["num_samples"], 2)


class DiskCacheTest(DatasetTestCase):
    def test_expensive_results_are_written_to_disk(self):
        analytics.compute_scene_metrics("S", include_expensive=True)
        with open(Config.ANALYSIS_CACHE_PATH) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["S"]["lidar"]["sweep_point_count"], 200.0)

    def test_a_fresh_process_reads_the_cache_instead_of_recomputing(self):
        analytics.compute_scene_metrics("S", include_expensive=True)
        analytics._expensive_cache.clear()          # simulate a restart
        analytics._disk_cache_loaded = False
        with mock.patch.object(analytics, "_compute_expensive", side_effect=AssertionError("recomputed")):
            m = analytics.compute_scene_metrics("S", include_expensive=True)
        self.assertEqual(m["lidar"]["sweep_point_count"], 200.0)

    def test_cache_write_leaves_no_temp_file(self):
        analytics.compute_scene_metrics("S", include_expensive=True)
        self.assertEqual(os.listdir(os.path.dirname(Config.ANALYSIS_CACHE_PATH)), ["analysis.json"])

    def test_corrupt_cache_file_is_ignored_and_recomputed(self):
        os.makedirs(os.path.dirname(Config.ANALYSIS_CACHE_PATH), exist_ok=True)
        with open(Config.ANALYSIS_CACHE_PATH, "w") as f:
            f.write("{truncated")
        m = analytics.compute_scene_metrics("S", include_expensive=True)
        self.assertEqual(m["lidar"]["sweep_point_count"], 200.0)

    def test_expensive_work_happens_once_per_scene(self):
        analytics.compute_scene_metrics("S", include_expensive=True)
        with mock.patch.object(analytics, "_compute_expensive", side_effect=AssertionError("recomputed")):
            analytics.compute_scene_metrics("S", include_expensive=True)


class AggregateTest(DatasetTestCase):
    def test_group_by_tag_uses_every_tag_when_none_selected(self):
        out = analytics.aggregate_by_condition([], ["num_samples"], group_by="tag")
        self.assertEqual(out["labels"], ["clear", "day"])
        self.assertEqual(out["metrics"]["num_samples"]["series"]["general"], [2.0, 2.0])
        self.assertEqual(out["scene_count"], 1)

    def test_group_by_scene(self):
        out = analytics.aggregate_by_condition([], ["num_samples"], group_by="scene")
        self.assertEqual(out["labels"], ["scene-0001"])

    def test_tag_filter_with_no_match_gives_no_groups(self):
        out = analytics.aggregate_by_condition(["night"], ["num_samples"], group_by="scene")
        self.assertEqual((out["labels"], out["scene_count"]), ([], 0))

    def test_metric_applies_to_each_of_its_modalities(self):
        out = analytics.aggregate_by_condition(["day"], ["sweep_point_count"], include_expensive=True)
        series = out["metrics"]["sweep_point_count"]["series"]
        self.assertEqual(series, {"lidar": [200.0], "radar": [15.0]})

    def test_expensive_metric_is_null_when_not_computed(self):
        out = analytics.aggregate_by_condition(["day"], ["sweep_point_count"], include_expensive=False)
        self.assertEqual(out["metrics"]["sweep_point_count"]["series"], {"lidar": [None], "radar": [None]})

    def test_unknown_metric_ids_are_ignored(self):
        out = analytics.aggregate_by_condition([], ["num_samples", "bogus"])
        self.assertEqual(list(out["metrics"]), ["num_samples"])

    def test_catalog_lists_every_metric_with_its_cost_tier(self):
        catalog = {m["id"]: m for m in analytics.metric_catalog()}
        self.assertEqual(set(catalog), set(analytics.METRICS))
        self.assertTrue(catalog["visibility"]["cheap"])
        self.assertFalse(catalog["mean_speed"]["cheap"])
        self.assertEqual(catalog["sweep_point_count"]["modalities"], ["lidar", "radar"])

    def test_result_is_json_serialisable(self):
        json.dumps(analytics.aggregate_by_condition([], list(analytics.METRICS), include_expensive=True))


if __name__ == "__main__":
    unittest.main()
