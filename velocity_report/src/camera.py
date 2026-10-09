import pandas as pd
from truckscenes.utils.geometry_utils import BoxVisibility


def build_camera_target_df(truckscenes, vehicle_annotation_df, camera_keyframes):
    """
    Build target-level Camera visual-support information from keyframe observations.
    """

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

        _, boxes, _ = truckscenes.get_sample_data(
            observation["Token"],
            box_vis_level=BoxVisibility.ANY,
            selected_anntokens=annotation_tokens,
        )

        for box in boxes:
            records.append({
                "annotation_token": box.token,
                "camera_sensor": observation["Sensor"],
            })

    camera_view_df = pd.DataFrame(records)

    if camera_view_df.empty:
        support_df = pd.DataFrame(
            columns=["annotation_token", "camera_view_count"]
        )
    else:
        support_df = (
            camera_view_df
            .groupby("annotation_token")
            .agg(camera_view_count=("camera_sensor", "nunique"))
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

    target_camera_df["has_camera_information"] = (
        target_camera_df["camera_view_count"] > 0
    )

    return camera_view_df, target_camera_df