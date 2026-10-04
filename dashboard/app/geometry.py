"""
Coordinate transforms shared by the scene viewer: moving sensor points and
annotation boxes between frames.

    sensor frame --(calibrated_sensor translation/rotation)--> ego frame
    ego frame    --(ego_pose translation/rotation)-->          global frame

sample_annotation boxes are stored in the GLOBAL frame.
load_radar_points()/load_lidar_points() return points in SENSOR frame. To
draw everything in one shared frame for the scene viewer, move
everything to the EGO frame:
  - sensor points:   sensor -> ego           (points_sensor_to_ego)
  - annotation boxes: global -> ego           (annotation_to_ego_frame)

"""
import numpy as np
from pyquaternion import Quaternion

from app import truckscenes_loader as tsl


def quat_rotate(quat_wxyz, vectors):
    """Rotate an (N, 3) array of vectors by a [w, x, y, z] quaternion."""
    q = Quaternion(quat_wxyz)
    return (q.rotation_matrix @ vectors.T).T


def box_corners(center, size, quat_wxyz):
    """
    8 corners of an oriented box, in whatever frame `center`/`quat_wxyz`
    are given in. size = [w, l, h].

    Corner order:
      0: front-left-top   1: front-right-top   2: front-right-bottom  3: front-left-bottom
      4: rear-left-top    5: rear-right-top    6: rear-right-bottom   7: rear-left-bottom
    """
    w, l, h = size
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack([x_corners, y_corners, z_corners]).T  # (8, 3)
    corners = quat_rotate(quat_wxyz, corners)
    return corners + np.array(center)


def yaw_to_quaternion(yaw):
    """
    [w, x, y, z] quaternion for a rotation of `yaw` radians about the z
    axis.
    """
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def points_sensor_to_ego(points_xyz, calibrated_sensor):
    """(N, 3) points in sensor frame -> (N, 3) points in ego frame."""
    if len(points_xyz) == 0:
        return points_xyz
    rotated = quat_rotate(calibrated_sensor["rotation"], points_xyz)
    return rotated + np.array(calibrated_sensor["translation"])


def load_points_ego_frame(sample_data_token, modality):
    """
    Raw sensor points, transformed into the ego vehicle frame. Returns (xyz, color)
    where color is RCS for radar, intensity for lidar (zeros if the lidar
    array has no 4th column).
    """
    trucksc = tsl.get_trucksc()
    sd = trucksc.get("sample_data", sample_data_token)
    calib = trucksc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    if modality == "radar":
        pts = tsl.load_radar_points(sample_data_token)
        xyz = pts[:, [tsl.RADAR_ROW["x"], tsl.RADAR_ROW["y"], tsl.RADAR_ROW["z"]]]
        color = pts[:, tsl.RADAR_ROW["rcs"]]
    elif modality == "lidar":
        pts = tsl.load_lidar_points(sample_data_token)
        xyz = pts[:, :3]
        color = pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts))
    else:
        raise ValueError(f"load_points_ego_frame only supports radar/lidar, got {modality}")

    return points_sensor_to_ego(xyz, calib), color


def _global_to_ego(corners, ego_pose):
    corners = corners - np.array(ego_pose["translation"])
    return quat_rotate(Quaternion(ego_pose["rotation"]).inverse.elements, corners)


def annotation_to_ego_frame(ann, sample_data_token):
    """Box corners for one sample_annotation, global frame -> ego frame."""
    trucksc = tsl.get_trucksc()
    sd = trucksc.get("sample_data", sample_data_token)
    ego_pose = trucksc.get("ego_pose", sd["ego_pose_token"])

    corners = box_corners(ann["translation"], ann["size"], ann["rotation"])
    return _global_to_ego(corners, ego_pose)


def annotation_to_sensor_frame(ann, sample_data_token):
    """
    Box corners for one sample_annotation, global frame -> ego -> sensor
    frame.
    """
    trucksc = tsl.get_trucksc()
    sd = trucksc.get("sample_data", sample_data_token)
    ego_pose = trucksc.get("ego_pose", sd["ego_pose_token"])
    calib = trucksc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    corners = box_corners(ann["translation"], ann["size"], ann["rotation"])
    corners = _global_to_ego(corners, ego_pose)
    corners = corners - np.array(calib["translation"])
    return quat_rotate(Quaternion(calib["rotation"]).inverse.elements, corners)
