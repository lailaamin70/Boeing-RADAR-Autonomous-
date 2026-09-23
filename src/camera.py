import numpy as np
import pandas as pd

from truckscenes.utils.geometry_utils import BoxVisibility, view_points


def build_camera_target_df(truckscenes, vehicle_annotation_df, camera_keyframes):
    """Build target-level Camera visual-support information."""

    targets_by_sample = (
        vehicle_annotation_df
        .groupby("sample_token")["annotation_token"]
        .apply(list)
        .to_dict()
    )

    records = []

    for _, observation in camera_keyframes.iterrows():
        annotation_tokens = targets_by_sample.get(observation["Sample Token"], [])

        if not annotation_tokens:
            continue

        sample_data = truckscenes.get("sample_data", observation["Token"])
        image_width = sample_data["width"]
        image_height = sample_data["height"]

        _, boxes, camera_intrinsic = truckscenes.get_sample_data(
            observation["Token"],
            box_vis_level=BoxVisibility.ANY,
            selected_anntokens=annotation_tokens,
        )

        for box in boxes:
            corners = view_points(
                box.corners(),
                camera_intrinsic,
                normalize=True,
            )[:2]

            x_min = max(0, corners[0].min())
            x_max = min(image_width, corners[0].max())
            y_min = max(0, corners[1].min())
            y_max = min(image_height, corners[1].max())

            projected_area = (
                max(0, x_max - x_min)
                * max(0, y_max - y_min)
            )

            records.append({
                "annotation_token": box.token,
                "camera_sensor": observation["Sensor"],
                "projected_area_px": projected_area,
            })

    camera_view_df = pd.DataFrame(records)

    if camera_view_df.empty:
        support_df = pd.DataFrame(columns=[
            "annotation_token",
            "camera_view_count",
            "camera_projected_area_px",
        ])
    else:
        support_df = (
            camera_view_df
            .groupby("annotation_token")
            .agg(
                camera_view_count=("camera_sensor", "nunique"),
                camera_projected_area_px=("projected_area_px", "max"),
            )
            .reset_index()
        )

    target_camera_df = (
        vehicle_annotation_df[
            ["annotation_token", "instance_token", "sample_token", "category"]
        ]
        .merge(support_df, on="annotation_token", how="left")
    )

    target_camera_df["camera_view_count"] = (
        target_camera_df["camera_view_count"].fillna(0).astype(int)
    )

    target_camera_df["camera_projected_area_px"] = (
        target_camera_df["camera_projected_area_px"].fillna(0)
    )

    target_camera_df["has_camera_information"] = (
        target_camera_df["camera_view_count"] > 0
    )

    return camera_view_df, target_camera_df