import json
import os
import unittest

import numpy as np

from app import dataset_stats as ds
from config import Config
from tests.base import DatasetTestCase
from tests.fake_dataset import FILE_SIZES, T0


def idx():
    return {label: i for i, label in enumerate(ds.bin_labels())}


class HelpersTest(unittest.TestCase):
    def test_bin_labels(self):
        labels = ds.bin_labels()
        self.assertEqual(labels[0], "0-10")
        self.assertEqual(labels[-1], "150+")  # open-ended last bin
        self.assertEqual(len(labels), len(ds.DIST_EDGES) - 1)

    def test_bin_edges_belong_to_the_upper_bin_and_the_last_bin_is_open(self):
        b = ds._bin_index(np.array([0.0, 9.99, 10.0, 20.0, 60.0, 149.9, 150.0, 5000.0]))
        labels = ds.bin_labels()
        self.assertEqual([labels[i] for i in b], ["0-10", "0-10", "10-20", "20-30", "60-80", "100-150", "150+", "150+"])

    def test_timestamp_unit_is_inferred_from_magnitude(self):
        self.assertEqual(ds._seconds_per_tick(1.7e15), 1e-6)  # microseconds
        self.assertEqual(ds._seconds_per_tick(1.7e12), 1e-3)  # milliseconds
        self.assertEqual(ds._seconds_per_tick(1.7e9), 1.0)    # seconds

    def test_visibility_levels_sort_numerically_whether_ints_or_strings(self):
        ints = [{"token": "a", "level": 3}, {"token": "b", "level": 1}, {"token": "c", "level": 2}]
        self.assertEqual([v["token"] for v in sorted(ints, key=ds._level_key)], ["b", "c", "a"])
        strings = [{"token": "x", "level": "v60-80"}, {"token": "y", "level": "v0-40"}, {"token": "z", "level": "v40-60"}]
        self.assertEqual([v["token"] for v in sorted(strings, key=ds._level_key)], ["y", "z", "x"])


class AnnotationTableTest(DatasetTestCase):
    def test_ego_trailer_is_excluded(self):
        t = ds.get_table()
        self.assertEqual(len(t["dist"]), 4)
        self.assertNotIn("vehicle.ego_trailer", t["categories"])
        self.assertNotIn(999, t["lidar"])  # the ego trailer's sentinel point count

    def test_distances_are_xy_distance_to_the_ego_pose(self):
        self.assertEqual(sorted(ds.get_table()["dist"].round(3)), [15.0, 20.0, 35.0, 60.0])

    def test_visibility_ranks_follow_level_not_table_order(self):
        levels = ds.get_table()["vis_levels"]
        self.assertEqual([v["description"] for v in levels], ["0-40%", "40-60%", "60-80%", "80-100%"])

    def test_missing_visibility_becomes_rank_zero(self):
        t = ds.get_table()
        self.assertEqual(sorted(t["vis"].tolist()), [0, 1, 3, 4])

    def test_table_is_built_once(self):
        self.assertIs(ds.get_table(), ds.get_table())

    def test_filter_options_list_categories_by_count_without_the_ego_rig(self):
        opts = ds.filter_options()
        self.assertEqual(opts["categories"][0], {"name": "vehicle.car", "count": 2})
        self.assertNotIn("vehicle.ego_trailer", [c["name"] for c in opts["categories"]])
        self.assertEqual(opts["tags"], ["clear", "day"])


class DetectabilityTest(DatasetTestCase):
    def test_per_bin_counts_and_point_statistics(self):
        d, i = ds.detectability(min_lidar=5, min_radar=1), idx()
        self.assertEqual(sum(d["n"]), 4)
        self.assertEqual([d["n"][i[b]] for b in ("10-20", "20-30", "30-40", "60-80")], [1, 1, 1, 1])
        self.assertEqual(d["lidar"]["mean"][i["20-30"]], 50)   # a1
        self.assertEqual(d["radar"]["mean"][i["20-30"]], 3)
        self.assertEqual(d["radar"]["zero_pct"][i["30-40"]], 100)  # a2 has no radar points
        self.assertEqual(d["lidar"]["seen_pct"][i["60-80"]], 0)    # a3: 2 pts < 5
        self.assertEqual(d["radar"]["seen_pct"][i["60-80"]], 100)  # a3: 1 pt >= 1

    def test_empty_bins_are_null_not_zero(self):
        d, i = ds.detectability(), idx()
        for key in ("mean", "median", "zero_pct", "seen_pct"):
            self.assertIsNone(d["lidar"][key][i["0-10"]])
            self.assertIsNone(d["radar"][key][i["100-150"]])

    def test_threshold_changes_what_counts_as_seen(self):
        i = idx()
        self.assertEqual(ds.detectability(min_lidar=5)["lidar"]["seen_pct"][i["30-40"]], 100)  # a2 has 8
        self.assertEqual(ds.detectability(min_lidar=9)["lidar"]["seen_pct"][i["30-40"]], 0)

    def test_two_way_overlap(self):
        d, i = ds.detectability(min_lidar=5, min_radar=1), idx()
        c = d["combos"]
        self.assertEqual(set(c), {"lidar + radar", "lidar only", "radar only", "none"})  # no camera combos
        self.assertEqual(c["lidar + radar"][i["20-30"]], 1)
        self.assertEqual(c["lidar only"][i["30-40"]], 1)
        self.assertEqual(c["lidar only"][i["10-20"]], 1)
        self.assertEqual(c["radar only"][i["60-80"]], 1)
        self.assertEqual(d["dropped_unknown_visibility"], 0)

    def test_three_way_overlap_drops_objects_without_visibility(self):
        d, i = ds.detectability(min_lidar=5, min_radar=1, include_camera=True, min_visibility=2), idx()
        c = d["combos"]
        self.assertEqual(d["dropped_unknown_visibility"], 1)  # a4, the pedestrian
        self.assertEqual(c["lidar + radar + camera"][i["20-30"]], 1)  # a1: visibility 4 >= 2
        self.assertEqual(c["lidar + camera"][i["30-40"]], 1)          # a2: visibility 3 >= 2
        self.assertEqual(c["radar only"][i["60-80"]], 1)              # a3: visibility 1 < 2
        self.assertEqual(sum(d["combo_n"]), 3)

    def test_every_kept_object_lands_in_exactly_one_overlap_class(self):
        for camera in (False, True):
            d = ds.detectability(include_camera=camera)
            for b in range(len(d["bins"])):
                self.assertEqual(sum(v[b] for v in d["combos"].values()), d["combo_n"][b])

    def test_stricter_camera_visibility_removes_camera_coverage(self):
        d, i = ds.detectability(include_camera=True, min_visibility=4), idx()
        self.assertEqual(d["combos"]["lidar + camera"][i["30-40"]], 0)  # a2 has rank 3 < 4
        self.assertEqual(d["combos"]["lidar only"][i["30-40"]], 1)

    def test_category_and_tag_filters(self):
        self.assertEqual(ds.detectability(categories=["vehicle.car"])["total"], 2)
        self.assertEqual(ds.detectability(categories=["vehicle.car", "vehicle.truck"])["total"], 3)
        self.assertEqual(ds.detectability(tags=["day"])["total"], 4)
        self.assertEqual(ds.detectability(tags=["night"])["total"], 0)
        self.assertEqual(ds.detectability(categories=["vehicle.car"], tags=["night"])["total"], 0)

    def test_empty_selection_does_not_crash(self):
        d = ds.detectability(tags=["night"])
        self.assertEqual(d["total"], 0)
        self.assertEqual(d["summary"]["lidar"], {"mean": None, "zero_pct": None, "seen_pct": None})
        self.assertTrue(all(v is None for v in d["lidar"]["mean"]))

    def test_summary_is_over_all_selected_objects(self):
        s = ds.detectability(min_lidar=5, min_radar=1)["summary"]
        self.assertEqual(s["lidar"]["mean"], 20.0)       # (50 + 8 + 2 + 20) / 4
        self.assertEqual(s["lidar"]["seen_pct"], 75.0)   # all but a3
        self.assertEqual(s["radar"]["zero_pct"], 50.0)   # a2 and a4

    def test_result_is_json_serialisable(self):
        json.dumps(ds.detectability(include_camera=True))


class OverviewTest(DatasetTestCase):
    def setUp(self):
        super().setUp()
        self.ov = ds.overview()

    def test_counts(self):
        c = self.ov["counts"]
        self.assertEqual((c["scenes"], c["samples"], c["annotations"], c["instances"]), (1, 2, 4, 4))
        self.assertEqual((c["sample_data files"], c["keyframe files"]), (6, 6))

    def test_categories_sorted_with_annotation_and_instance_counts(self):
        cats = self.ov["categories"]
        self.assertEqual(cats[0], {"name": "vehicle.car", "annotations": 2, "instances": 2})
        self.assertEqual({c["name"] for c in cats}, {"vehicle.car", "vehicle.truck", "human.pedestrian.adult"})

    def test_distance_histogram(self):
        h = self.ov["distance"]
        self.assertEqual(sum(h["counts"]), 4)
        self.assertEqual(h["counts"][idx()["60-80"]], 1)

    def test_visibility_includes_a_not_annotated_bucket(self):
        v = {row["label"]: row["count"] for row in self.ov["visibility"]}
        self.assertEqual(v["not annotated"], 1)
        self.assertEqual(v["80-100%"], 1)
        self.assertEqual(sum(v.values()), 4)

    def test_attributes_and_tags(self):
        self.assertEqual({a["name"]: a["count"] for a in self.ov["attributes"]}, {"vehicle.moving": 2, "vehicle.parked": 1})
        self.assertEqual({t["name"]: t["count"] for t in self.ov["tags"]}, {"day": 1, "clear": 1})

    def test_scene_row(self):
        (row,) = self.ov["scenes"]
        self.assertEqual((row["name"], row["samples"], row["annotations"]), ("scene-0001", 2, 4))
        self.assertEqual((row["mean_speed_kmh"], row["max_speed_kmh"]), (36.0, 36.0))

    def test_result_is_json_serialisable(self):
        json.dumps(self.ov)


class EgoMotionTest(DatasetTestCase):
    def test_window_speed_and_duration(self):
        em = ds.ego_motion("S")
        self.assertEqual(len(em["t"]), 21)
        self.assertEqual(em["t"][0], 0.0)
        self.assertAlmostEqual(em["t"][-1], 2.0)  # microsecond ticks converted to seconds
        self.assertEqual(set(em["speed_kmh"]), {36.0})  # 10 m/s
        self.assertEqual(em["stats"], {"mean_speed_kmh": 36.0, "max_speed_kmh": 36.0, "duration_s": 2.0})

    def test_unit_conversions(self):
        em = ds.ego_motion("S")
        self.assertEqual(set(em["ax"]), {0.5})
        self.assertAlmostEqual(em["yaw_rate_dps"][0], np.degrees(0.02), places=3)

    def test_cabin_minus_chassis_tilt_in_degrees(self):
        em = ds.ego_motion("S")
        self.assertAlmostEqual(em["pitch_diff_deg"][0], np.degrees(0.03 - 0.01), places=3)
        self.assertAlmostEqual(em["roll_diff_deg"][0], 0.0, places=6)

    def test_path_starts_at_the_origin_and_follows_the_truck(self):
        path = ds.ego_motion("S")["path"]
        self.assertEqual((path["x"][0], path["y"][0]), (0.0, 0.0))
        self.assertAlmostEqual(path["x"][-1], 20.0)  # 10 m/s for 2 s
        self.assertEqual(set(path["y"]), {0.0})

    def test_long_scenes_are_downsampled(self):
        em = ds.ego_motion("S", max_points=5)
        self.assertLessEqual(len(em["t"]), 6)
        self.assertEqual(em["t"][0], 0.0)

    def test_rows_outside_the_scene_window_are_ignored(self):
        self.ds.tables["ego_motion_chassis"].append(
            {**self.ds.tables["ego_motion_chassis"][0], "token": "late", "timestamp": T0 + 60_000_000, "vx": 99.0}
        )
        self.assertEqual(ds.ego_motion("S")["stats"]["max_speed_kmh"], 36.0)

    def test_scene_with_no_ego_rows_explains_itself(self):
        self.ds.tables["ego_motion_chassis"].clear()
        em = ds.ego_motion("S")
        self.assertEqual(em["t"], [])
        self.assertIn("No ego-motion rows", em["note"])

    def test_unknown_scene_raises_keyerror(self):
        with self.assertRaises(KeyError):
            ds.ego_motion("nope")


class DataVolumeTest(DatasetTestCase):
    def channels(self, vol=None):
        return {c["channel"]: c for c in (vol or ds.data_volume())["channels"]}

    def test_sizes_counts_and_rates(self):
        ch = self.channels()
        self.assertEqual(ch["CAMERA_FRONT"]["files"], 2)
        self.assertEqual(ch["CAMERA_FRONT"]["total_mb"], 1.0)  # 2 x 500 kB
        self.assertEqual(ch["LIDAR_TOP"]["mean_kb"], 4.0)
        self.assertEqual(ch["RADAR_FRONT"]["mean_kb"], 1.0)
        self.assertEqual(ch["CAMERA_FRONT"]["hz"], 0.1)        # 2 files / (1 scene x 20 s)
        self.assertEqual(ch["CAMERA_FRONT"]["mb_per_s"], 0.05)
        self.assertEqual(ch["CAMERA_FRONT"]["keyframe_files"], 2)

    def test_modality_totals(self):
        mods = {m["modality"]: m for m in ds.data_volume()["modalities"]}
        self.assertEqual(mods["camera"]["total_mb"], 1.0)
        self.assertEqual(set(mods), {"lidar", "radar", "camera"})

    def test_result_is_cached_on_disk_and_reused(self):
        ds.data_volume()
        self.assertTrue(os.path.isfile(Config.DATA_VOLUME_CACHE_PATH))
        os.remove(self.root / "samples/CAMERA_FRONT/cam_A.png")
        self.assertEqual(ds.data_volume()["missing_files"], 0)  # served from the cache, not rescanned

    def test_refresh_rescans_and_reports_missing_files(self):
        ds.data_volume()
        os.remove(self.root / "samples/CAMERA_FRONT/cam_A.png")
        vol = ds.data_volume(force=True)
        self.assertEqual(vol["missing_files"], 1)
        self.assertEqual(self.channels(vol)["CAMERA_FRONT"]["files"], 1)

    def test_cache_for_a_different_dataset_is_ignored(self):
        ds.data_volume()
        os.remove(self.root / "samples/CAMERA_FRONT/cam_A.png")
        from unittest import mock
        with mock.patch.object(Config, "TRUCKSCENES_VERSION", "another-version"):
            self.assertEqual(ds.data_volume()["missing_files"], 1)

    def test_corrupt_cache_file_is_recomputed(self):
        os.makedirs(os.path.dirname(Config.DATA_VOLUME_CACHE_PATH), exist_ok=True)
        with open(Config.DATA_VOLUME_CACHE_PATH, "w") as f:
            f.write("{not json")
        self.assertEqual(self.channels()["CAMERA_FRONT"]["files"], 2)


if __name__ == "__main__":
    unittest.main()
