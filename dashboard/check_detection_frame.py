"""
Misc script for looking at format of frame
"""
import pickle
import numpy as np

np.set_printoptions(precision=2, suppress=True)

from app.truckscenes_loader import get_trucksc, get_sample_annotations, get_sample_data_by_channel
from app.geometry import annotation_to_ego_frame, annotation_to_sensor_frame

SAMPLE_TOKEN = "454fa18d64d94009914137ce901c8e7c"

for modality, fname in [("lidar", "lidar_mini_predictions.pkl"), ("radar", "radar_mini_predictions.pkl")]:
    with open(f"../results/object_detection/omnihd/{fname}", "rb") as f:
        preds = pickle.load(f)
    classes = preds["metadata"]["classes"]
    frame = next(p for p in preds["predictions"] if p["sample_token"] == SAMPLE_TOKEN)
    boxes, scores, labels = frame["boxes_3d"], frame["scores_3d"], frame["labels_3d"]
    order = np.argsort(-scores)[:5]
    print(f"\n=== top 5 {modality} predictions, sample {SAMPLE_TOKEN} ===")
    for i in order:
        print(f"  center={boxes[i][:3]}  yaw={boxes[i][6]:.2f}  class={classes[labels[i]]}  score={scores[i]:.2f}")

trucksc = get_trucksc()
channels = get_sample_data_by_channel(SAMPLE_TOKEN)
lidar_ref = next(e["sample_data"]["token"] for e in channels.values() if e["modality"] == "lidar")

print(f"\n=== ground-truth annotations, sample {SAMPLE_TOKEN} ===")
for ann_token in get_sample_annotations(SAMPLE_TOKEN):
    ann = trucksc.get("sample_annotation", ann_token)
    instance = trucksc.get("instance", ann["instance_token"])
    category = trucksc.get("category", instance["category_token"])
    ego_center = annotation_to_ego_frame(ann, lidar_ref).mean(axis=0)
    sensor_center = annotation_to_sensor_frame(ann, lidar_ref).mean(axis=0)
    print(f"  ego={ego_center}  lidar-sensor={sensor_center}  category={category['name']}")
