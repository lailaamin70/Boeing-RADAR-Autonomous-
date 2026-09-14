import os
import cv2
import numpy as np
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud


def inspect_visibility(dataroot, version='v1.2-mini', verbose=False):
    """
    Inspect available sensor channels and print/return basic
    camera brightness and LiDAR point count for the first sample.

    Parameters
    ----------
    dataroot : str
        Path to the TruckScenes dataset root.
    version : str
        Dataset version to load.
    verbose : bool
        Passed through to TruckScenes loader.

    Returns
    -------
    dict with keys: cam_channel, lidar_channel, brightness, lidar_points
    """
    print("Loading dataset metadata...")
    nusc = TruckScenes(version=version, dataroot=dataroot, verbose=verbose)

    # Inspect available sensor channels on the vehicle
    sample = nusc.sample[0]
    print("\n=== Available Sensor Channels ===")
    for sensor_name in sample['data'].keys():
        print(f"- {sensor_name}")

    # Select primary test channels
    cam_channel = next((k for k in sample['data'].keys() if 'CAM' in k), None)
    lidar_channel = next((k for k in sample['data'].keys() if 'LIDAR' in k), None)

    print(f"\nSelected Test Channels: Camera [{cam_channel}], LiDAR [{lidar_channel}]")

    # Extract Camera sample and calculate mean brightness
    cam_token = sample['data'][cam_channel]
    cam_meta = nusc.get('sample_data', cam_token)
    cam_path = os.path.join(dataroot, cam_meta['filename'])

    brightness = None
    img = cv2.imread(cam_path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        print(f"Camera Mean Brightness (0=Dark, 255=Bright): {brightness:.2f}")
    else:
        print("Warning: Image file not found. Check the samples directory.")

    # Extract LiDAR sample and count return points
    lidar_token = sample['data'][lidar_channel]
    lidar_meta = nusc.get('sample_data', lidar_token)
    lidar_path = os.path.join(dataroot, lidar_meta['filename'])

    pc = LidarPointCloud.from_file(lidar_path)
    lidar_points = pc.points.shape[1]
    print(f"LiDAR Total Return Points: {lidar_points}")

    return {
        "cam_channel": cam_channel,
        "lidar_channel": lidar_channel,
        "brightness": brightness,
        "lidar_points": lidar_points,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect TruckScenes sensor channels.")
    parser.add_argument("dataroot", help="Path to the TruckScenes dataset root")
    parser.add_argument("--version", default="v1.2-mini", help="Dataset version")
    args = parser.parse_args()

    inspect_visibility(args.dataroot, version=args.version)
