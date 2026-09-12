import os
import numpy as np
import pandas as pd
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud


def parse_scene_tags(description):
    """
    Parse a TruckScenes scene description string (e.g.
    "weather.clear;area.terminal;daytime.noon;...") into a dict of tags.

    Parameters
    ----------
    description : str
        Raw scene['description'] string.

    Returns
    -------
    dict mapping tag key -> tag value (e.g. {'weather': 'clear', 'area': 'terminal', ...})
    """
    tags = {}
    for pair in description.split(';'):
        key, value = pair.split('.', 1)
        tags[key] = value
    return tags


def get_scene_metrics(nusc, dataroot, scene_token):
    """
    Compute total point count and range statistics (max, median, 95th percentile)
    for LiDAR and RADAR across an entire scene, summed/pooled across ALL LiDAR and
    RADAR sensor channels on the vehicle (not just one).

    Parameters
    ----------
    nusc : TruckScenes
        Loaded TruckScenes dataset object.
    dataroot : str
        Path to the TruckScenes dataset root.
    scene_token : str
        Token of the scene to process.

    Returns
    -------
    pandas.Series with keys:
        lidar_total_points, lidar_max_range, lidar_median_range, lidar_p95_range,
        radar_total_points, radar_max_range, radar_median_range, radar_p95_range
    """
    scene = nusc.get('scene', scene_token)
    current_sample_token = scene['first_sample_token']

    lidar_total_points = 0
    radar_total_points = 0
    lidar_ranges_all = []
    radar_ranges_all = []

    while current_sample_token:
        sample = nusc.get('sample', current_sample_token)

        lidar_channels = [k for k in sample['data'].keys() if 'LIDAR' in k]
        radar_channels = [k for k in sample['data'].keys() if 'RADAR' in k]

        for ch in lidar_channels:
            sd = nusc.get('sample_data', sample['data'][ch])
            pc = LidarPointCloud.from_file(os.path.join(dataroot, sd['filename']))
            lidar_total_points += pc.points.shape[1]
            if pc.points.shape[1] > 0:
                ranges = np.linalg.norm(pc.points[:3, :], axis=0)
                lidar_ranges_all.append(ranges)

        for ch in radar_channels:
            sd = nusc.get('sample_data', sample['data'][ch])
            # RadarPointCloud.from_file applies the devkit's default validity
            # filters (invalid_states=[0], dynprop_states, ambig_states) automatically.
            pc = RadarPointCloud.from_file(os.path.join(dataroot, sd['filename']))
            radar_total_points += pc.points.shape[1]
            if pc.points.shape[1] > 0:
                ranges = np.linalg.norm(pc.points[:2, :], axis=0)
                radar_ranges_all.append(ranges)

        current_sample_token = sample['next']

    lidar_ranges_all = np.concatenate(lidar_ranges_all) if lidar_ranges_all else np.array([])
    radar_ranges_all = np.concatenate(radar_ranges_all) if radar_ranges_all else np.array([])

    return pd.Series({
        'lidar_total_points': lidar_total_points,
        'lidar_max_range': lidar_ranges_all.max() if lidar_ranges_all.size else np.nan,
        'lidar_median_range': np.median(lidar_ranges_all) if lidar_ranges_all.size else np.nan,
        'lidar_p95_range': np.percentile(lidar_ranges_all, 95) if lidar_ranges_all.size else np.nan,
        'radar_total_points': radar_total_points,
        'radar_max_range': radar_ranges_all.max() if radar_ranges_all.size else np.nan,
        'radar_median_range': np.median(radar_ranges_all) if radar_ranges_all.size else np.nan,
        'radar_p95_range': np.percentile(radar_ranges_all, 95) if radar_ranges_all.size else np.nan,
    })


def get_zero_radar_summary(nusc, clear_scenes):
    """
    For every annotated object across the given scenes, use the dataset's
    precomputed 'num_radar_pts' field (no manual point-in-box overlap needed)
    to work out what percentage of objects received zero RADAR points, grouped
    by area.

    Parameters
    ----------
    nusc : TruckScenes
        Loaded TruckScenes dataset object.
    clear_scenes : pandas.DataFrame
        Must contain columns 'scene_token' and 'area' (e.g. output of filtering
        the scenes_df built from parse_scene_tags to weather == 'clear').

    Returns
    -------
    pandas.DataFrame indexed by area, with columns:
        n_objects, pct_zero_radar
    """
    annotation_rows = []

    for _, row in clear_scenes.iterrows():
        scene = nusc.get('scene', row['scene_token'])
        current_sample_token = scene['first_sample_token']

        while current_sample_token:
            sample = nusc.get('sample', current_sample_token)
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                annotation_rows.append({
                    'area': row['area'],
                    'num_radar_pts': ann['num_radar_pts'],
                    'num_lidar_pts': ann['num_lidar_pts'],
                })
            current_sample_token = sample['next']

    ann_df = pd.DataFrame(annotation_rows)

    return ann_df.groupby('area').agg(
        n_objects=('num_radar_pts', 'count'),
        pct_zero_radar=('num_radar_pts', lambda x: (x == 0).mean() * 100)
    ).round(1)


def analyze_areas(dataroot, version='v1.2-mini', verbose=False, output_csv=None):
    """
    End-to-end "Different Areas" analysis: filters the dataset to clear-weather
    scenes (the only weather condition common to all area types in the mini
    split), then computes point-count/range metrics and zero-RADAR-detection
    rates per area.

    Parameters
    ----------
    dataroot : str
        Path to the TruckScenes dataset root.
    version : str
        Dataset version to load.
    verbose : bool
        Passed through to TruckScenes loader.
    output_csv : str or None
        If given, used as a prefix to save two CSVs: "<output_csv>_scenes.csv"
        and "<output_csv>_zero_radar.csv".

    Returns
    -------
    (clear_scenes_full, zero_radar_summary) : tuple of pandas.DataFrame
    """
    nusc = TruckScenes(version=version, dataroot=dataroot, verbose=verbose)

    print("Parsing scene tags and filtering to clear-weather scenes...")
    scene_rows = []
    for scene in nusc.scene:
        tags = parse_scene_tags(scene['description'])
        scene_rows.append({
            'scene_token': scene['token'],
            'scene_name': scene['name'],
            'area': tags.get('area'),
            'weather': tags.get('weather'),
            'structure': tags.get('structure'),
        })
    scenes_df = pd.DataFrame(scene_rows)
    clear_scenes = scenes_df[scenes_df['weather'] == 'clear'].reset_index(drop=True)

    print(f"Found {len(clear_scenes)} clear-weather scenes. Computing point count and range metrics...")
    metrics = clear_scenes['scene_token'].apply(lambda t: get_scene_metrics(nusc, dataroot, t))
    clear_scenes_full = pd.concat([clear_scenes, metrics], axis=1)

    print("Computing per-area zero-RADAR-detection rates...")
    zero_radar_summary = get_zero_radar_summary(nusc, clear_scenes)

    print("\n=======================================================")
    print("  DIFFERENT AREAS: POINT COUNT & RANGE (clear weather)")
    print("=======================================================")
    print(clear_scenes_full[[
        'scene_name', 'area', 'lidar_total_points', 'lidar_median_range', 'lidar_p95_range',
        'radar_total_points', 'radar_max_range', 'radar_median_range', 'radar_p95_range'
    ]].to_string(index=False))

    print("\n=======================================================")
    print("  % OBJECTS WITH ZERO RADAR POINTS, BY AREA")
    print("=======================================================")
    print(zero_radar_summary)

    if output_csv:
        clear_scenes_full.to_csv(f"{output_csv}_scenes.csv", index=False)
        zero_radar_summary.to_csv(f"{output_csv}_zero_radar.csv")
        print(f"\nSaved results to {output_csv}_scenes.csv and {output_csv}_zero_radar.csv")

    return clear_scenes_full, zero_radar_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze TruckScenes RADAR/LiDAR metrics by area.")
    parser.add_argument("dataroot", help="Path to the TruckScenes dataset root")
    parser.add_argument("--version", default="v1.2-mini", help="Dataset version")
    parser.add_argument("--output-csv", default="different_areas_metrics", help="CSV output prefix")
    args = parser.parse_args()

    analyze_areas(args.dataroot, version=args.version, output_csv=args.output_csv)