import os
import cv2
import numpy as np
import pandas as pd
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud

# 1. Dataset root directory
DATAROOT = r"C:/Users/mindfulness/Desktop/CITS3200/man-truckscenes_sensordata_v1.2-mini/man-truckscenes"
nusc = TruckScenes(version='v1.2-mini', dataroot=DATAROOT, verbose=False)

records = []
print("Processing scenes and computing sensor metrics across visibility conditions...")

# 2. Iterate through each scene and categorize visibility condition
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
        
        # 2.1 Camera: Mean pixel intensity (0 to 255)
        cam_token = sample['data'].get('CAMERA_LEFT_FRONT')
        cam_path = os.path.join(DATAROOT, nusc.get('sample_data', cam_token)['filename'])
        img = cv2.imread(cam_path)
        brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) if img is not None else 0.0

        # 2.2 LiDAR: Total valid return points count
        lidar_token = sample['data'].get('LIDAR_TOP_FRONT', sample['data'].get('LIDAR_LEFT'))
        lidar_path = os.path.join(DATAROOT, nusc.get('sample_data', lidar_token)['filename'])
        lidar_pc = LidarPointCloud.from_file(lidar_path)
        lidar_points_count = lidar_pc.points.shape[1]

        # 2.3 Radar: Total detection returns count
        radar_token = sample['data'].get('RADAR_LEFT_FRONT')
        radar_path = os.path.join(DATAROOT, nusc.get('sample_data', radar_token)['filename'])
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

# 3. Aggregate results using pandas
df = pd.DataFrame(records)

print("\n=======================================================")
print("  SENSOR METRICS COMPARISON ACROSS VISIBILITY CONDITIONS")
print("=======================================================")
summary = df.groupby("Condition")[["Camera_Brightness", "LiDAR_Points", "Radar_Points"]].agg(["mean", "std"]).round(2)
print(summary)

# 4. Export results to CSV for technical reporting and dashboard integration
output_csv = "visibility_sensor_metrics.csv"
df.to_csv(output_csv, index=False)
print(f"\nExtracted dataset successfully saved to: {output_csv}")