import os
import cv2
import numpy as np
import pandas as pd
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud


def analyze_visibility(dataroot, version='v1.2-mini', verbose=False, output_csv=None):
    """
    Iterate through every scene/sample in the dataset and compute
    camera brightness, LiDAR point count, and radar point count,
    grouped by a coarse weather condition parsed from the scene
    description.

    Parameters
    ----------
    dataroot : str
        Path to the TruckScenes dataset root.
    version : str
        Dataset version to load.
    verbose : bool
        Passed through to TruckScenes loader.
    output_csv : str or None
        If given, also save the resulting DataFrame to this CSV path.

    Returns
    -------
    pandas.DataFrame with columns:
        Scene, Condition, Camera_Brightness, LiDAR_Points, Radar_Points
    """
    nusc = TruckScenes(version=version, dataroot=dataroot, verbose=verbose)

    records = []
    print("Processing scenes and computing sensor metrics across visibility conditions...")

    # Iterate through each scene and categorize visibility condition
    for scene in nusc.scene:
        desc = scene['description'].lower()

        if "rain" in desc:
            condition = "Rainy"
        elif "snow" in desc:
            condition = "Snowy"
        else:
            condition = "Clear"

        current_sample_token = scene['first_sample_token']

        while current_sample_token:
            sample = nusc.get('sample', current_sample_token)

            # Camera: Mean pixel intensity (0 to 255)
            cam_token = sample['data'].get('CAMERA_LEFT_FRONT')
            cam_path = os.path.join(dataroot, nusc.get('sample_data', cam_token)['filename'])
            img = cv2.imread(cam_path)
            brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) if img is not None else 0.0

            # LiDAR: Total valid return points count
            lidar_token = sample['data'].get('LIDAR_TOP_FRONT', sample['data'].get('LIDAR_LEFT'))
            lidar_path = os.path.join(dataroot, nusc.get('sample_data', lidar_token)['filename'])
            lidar_pc = LidarPointCloud.from_file(lidar_path)
            lidar_points_count = lidar_pc.points.shape[1]

            # Radar: Total detection returns count
            radar_token = sample['data'].get('RADAR_LEFT_FRONT')
            radar_path = os.path.join(dataroot, nusc.get('sample_data', radar_token)['filename'])
            radar_pc = RadarPointCloud.from_file(radar_path)
            radar_points_count = radar_pc.points.shape[1]

            records.append({
                "Scene": scene['name'],
                "Condition": condition,
                "Camera_Brightness": brightness,
                "LiDAR_Points": lidar_points_count,
                "Radar_Points": radar_points_count
            })

            current_sample_token = sample['next']

    df = pd.DataFrame(records)

    print("\n=======================================================")
    print("  SENSOR METRICS COMPARISON ACROSS VISIBILITY CONDITIONS")
    print("=======================================================")
    summary = df.groupby("Condition")[["Camera_Brightness", "LiDAR_Points", "Radar_Points"]].agg(["mean", "std"]).round(2)
    print(summary)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nExtracted dataset successfully saved to: {output_csv}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze TruckScenes sensor visibility metrics.")
    parser.add_argument("dataroot", help="Path to the TruckScenes dataset root")
    parser.add_argument("--version", default="v1.2-mini", help="Dataset version")
    parser.add_argument("--output-csv", default="visibility_sensor_metrics.csv", help="Where to save results")
    args = parser.parse_args()

    analyze_visibility(args.dataroot, version=args.version, output_csv=args.output_csv)
