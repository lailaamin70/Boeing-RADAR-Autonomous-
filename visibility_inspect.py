import os
import cv2
import numpy as np
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud

# 1. Dataset root directory
DATAROOT = r"C:/Users/mindfulness/Desktop/CITS3200/man-truckscenes_sensordata_v1.2-mini/man-truckscenes"

print("Loading dataset metadata...")
nusc = TruckScenes(version='v1.2-mini', dataroot=DATAROOT, verbose=False)

# 2. Inspect available sensor channels on the vehicle
sample = nusc.sample[0]
print("\n=== Available Sensor Channels ===")
for sensor_name in sample['data'].keys():
    print(f"- {sensor_name}")

# 3. Select primary test channels
cam_channel = next((k for k in sample['data'].keys() if 'CAM' in k), None)
lidar_channel = next((k for k in sample['data'].keys() if 'LIDAR' in k), None)

print(f"\nSelected Test Channels: Camera [{cam_channel}], LiDAR [{lidar_channel}]")

# 4. Extract Camera sample and calculate mean brightness
cam_token = sample['data'][cam_channel]
cam_meta = nusc.get('sample_data', cam_token)
cam_path = os.path.join(DATAROOT, cam_meta['filename'])

img = cv2.imread(cam_path)
if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    print(f"Camera Mean Brightness (0=Dark, 255=Bright): {brightness:.2f}")
else:
    print("Warning: Image file not found. Check the samples directory.")

# 5. Extract LiDAR sample and count return points
lidar_token = sample['data'][lidar_channel]
lidar_meta = nusc.get('sample_data', lidar_token)
lidar_path = os.path.join(DATAROOT, lidar_meta['filename'])

pc = LidarPointCloud.from_file(lidar_path)
print(f"LiDAR Total Return Points: {pc.points.shape[1]}")