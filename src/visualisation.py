from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from .config import SENSOR_ROOT
from .lidar_velocity import load_lidar_points


def load_sample_lidar_points(truckscenes, sample_sensor_data, sensor_root=None):
    """Load and align all keyframe LiDAR points to one sample ego frame."""

    sensor_root = Path(sensor_root or SENSOR_ROOT)

    lidar_frames = sample_sensor_data[
        sample_sensor_data["Modality"] == "LiDAR"
    ]

    sample_tokens = lidar_frames["Sample Token"].unique()

    if len(sample_tokens) != 1:
        raise ValueError("LiDAR observations must belong to one sample.")

    sample = truckscenes.get("sample", sample_tokens[0])

    reference_pose = truckscenes.getclosest(
        "ego_pose",
        sample["timestamp"],
    )

    reference_rotation = Quaternion(
        reference_pose["rotation"]
    ).rotation_matrix

    reference_translation = np.asarray(
        reference_pose["translation"],
        dtype=float,
    ).reshape(3, 1)

    point_tables = []

    for _, frame in lidar_frames.iterrows():
        sample_data = truckscenes.get(
            "sample_data",
            frame["Token"],
        )

        calibrated_sensor = truckscenes.get(
            "calibrated_sensor",
            sample_data["calibrated_sensor_token"],
        )

        ego_pose = truckscenes.get(
            "ego_pose",
            sample_data["ego_pose_token"],
        )

        points = load_lidar_points(
            sensor_root / sample_data["filename"]
        ).copy()

        xyz = points[["x", "y", "z"]].to_numpy().T

        # Sensor -> ego at LiDAR acquisition time
        xyz = (
            Quaternion(calibrated_sensor["rotation"]).rotation_matrix
            @ xyz
        )

        xyz += np.asarray(
            calibrated_sensor["translation"],
            dtype=float,
        ).reshape(3, 1)

        # Ego at LiDAR acquisition time -> global
        xyz = (
            Quaternion(ego_pose["rotation"]).rotation_matrix
            @ xyz
        )

        xyz += np.asarray(
            ego_pose["translation"],
            dtype=float,
        ).reshape(3, 1)

        # Global -> common sample ego frame
        xyz -= reference_translation
        xyz = reference_rotation.T @ xyz

        points[["x", "y", "z"]] = xyz.T
        points["sensor"] = frame["Sensor"]

        point_tables.append(points)

    return pd.concat(point_tables, ignore_index=True)


def plot_lidar_bev(
    lidar_points,
    x_range=(-50, 100),
    y_range=(-50, 50),
):
    """Plot LiDAR points in ego-frame Bird's-Eye View."""

    visible = lidar_points[
        lidar_points["x"].between(*x_range)
        & lidar_points["y"].between(*y_range)
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        visible["x"],
        visible["y"],
        s=0.5,
    )

    ax.scatter(
        0,
        0,
        marker="x",
        s=60,
        label="Ego vehicle",
    )

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("Ego x (m)")
    ax.set_ylabel("Ego y (m)")
    ax.set_title("LiDAR Bird's-Eye View")
    ax.grid(alpha=0.2)
    ax.legend()

    plt.show()


def plot_lidar_3d(
    lidar_points,
    x_range=(-50, 100),
    y_range=(-50, 50),
    z_range=(-5, 10),
    max_points=100000,
):
    """Plot an interactive 3D LiDAR point cloud."""

    import plotly.express as px

    visible = lidar_points[
        lidar_points["x"].between(*x_range)
        & lidar_points["y"].between(*y_range)
        & lidar_points["z"].between(*z_range)
    ].copy()

    if len(visible) > max_points:
        visible = visible.sample(max_points, random_state=42)

    fig = px.scatter_3d(
        visible,
        x="x",
        y="y",
        z="z",
        color="z",
        color_continuous_scale="Turbo",
        opacity=0.8,
    )

    fig.update_traces(
        marker=dict(
            size=1.2,
        )
    )

    fig.update_layout(
        title="3D LiDAR Point Cloud",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            bgcolor="black",
            xaxis=dict(
                title="Ego x (m)",
                range=list(x_range),
                backgroundcolor="black",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="white",
                color="white",
            ),
            yaxis=dict(
                title="Ego y (m)",
                range=list(y_range),
                backgroundcolor="black",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="white",
                color="white",
            ),
            zaxis=dict(
                title="Height (m)",
                range=list(z_range),
                backgroundcolor="black",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="white",
                color="white",
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=0.8)
            ),
        ),
        coloraxis_colorbar=dict(
            title="Height (z)",
            tickfont=dict(color="white"),
            title_font=dict(color="white"),
        ),
    )

    fig.show()



def get_sample_boxes_ego(truckscenes, sample_objects):
    """Transform Ground Truth boxes from global coordinates to ego frame."""

    boxes = []

    for annotation_token in sample_objects["annotation_token"]:
        box = truckscenes.get_box(annotation_token)

        annotation = truckscenes.get(
            "sample_annotation",
            annotation_token,
        )

        sample = truckscenes.get(
            "sample",
            annotation["sample_token"],
        )

        ego_pose = truckscenes.getclosest(
            "ego_pose",
            sample["timestamp"],
        )

        box.translate(
            -np.asarray(ego_pose["translation"], dtype=float)
        )

        box.rotate(
            Quaternion(ego_pose["rotation"]).inverse
        )

        boxes.append(box)

    return boxes


def plot_lidar_bev_with_boxes(
    lidar_points,
    boxes,
    highlight_category=None,
    highlight_distance_range=None,
    x_range=(-50, 100),
    y_range=(-50, 50),
):
    """Plot LiDAR BEV with highlighted Ground Truth objects."""

    import matplotlib.pyplot as plt
    import numpy as np

    visible = lidar_points[
        lidar_points["x"].between(*x_range)
        & lidar_points["y"].between(*y_range)
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(
        visible["x"],
        visible["y"],
        s=0.5,
        alpha=0.8,
    )

    has_highlight_filter = (
        highlight_category is not None
        or highlight_distance_range is not None
    )

    for box in boxes:
        distance = np.hypot(
            box.center[0],
            box.center[1],
        )

        highlighted = has_highlight_filter

        if highlight_category is not None:
            highlighted &= box.name == highlight_category

        if highlight_distance_range is not None:
            min_distance, max_distance = highlight_distance_range

            if min_distance is not None:
                highlighted &= distance >= min_distance

            if max_distance is not None:
                highlighted &= distance < max_distance

        corners = box.bottom_corners()
        x = corners[0, [0, 1, 2, 3, 0]]
        y = corners[1, [0, 1, 2, 3, 0]]

        ax.plot(
            x,
            y,
            color="red" if highlighted else "gray",
            linewidth=2.5 if highlighted else 1.0,
            alpha=1.0 if highlighted else 0.6,
        )

        if highlighted:
            ax.text(
                box.center[0],
                box.center[1],
                box.name.split(".")[-1],
                color="red",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

    ax.scatter(
        0,
        0,
        marker="x",
        s=80,
        color="tab:orange",
        linewidths=2.5,
        label="Ego vehicle",
        zorder=10,
    )

    ax.annotate(
        "",
        xy=(5, 0),
        xytext=(1, 0),
        arrowprops=dict(
            arrowstyle="->",
            color="tab:orange",
            lw=2,
        ),
    )

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("Ego x (m)")
    ax.set_ylabel("Ego y (m)")
    ax.set_title("LiDAR BEV with Ground Truth Objects")
    ax.grid(alpha=0.2)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_lidar_3d_with_boxes(
    lidar_points,
    boxes,
    highlight_category=None,
    highlight_distance_range=None,
    x_range=(-50, 100),
    y_range=(-50, 50),
    z_range=(-5, 10),
    max_points=100000,
):
    """Plot interactive 3D LiDAR point cloud with highlighted GT boxes."""

    import numpy as np
    import plotly.graph_objects as go

    visible = lidar_points[
        lidar_points["x"].between(*x_range)
        & lidar_points["y"].between(*y_range)
        & lidar_points["z"].between(*z_range)
    ].copy()

    if len(visible) > max_points:
        visible = visible.sample(
            max_points,
            random_state=42,
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=visible["x"],
            y=visible["y"],
            z=visible["z"],
            mode="markers",
            marker=dict(
                size=1,
                color=visible["z"],
                colorscale="Turbo",
                opacity=0.8,
                colorbar=dict(
                    title="Height (z)",
                    tickfont=dict(color="white"),
                    title_font=dict(color="white"),
                ),
            ),
            name="LiDAR points",
        )
    )

    # Schematic 3D ego vehicle marker.
    ego_length = 5.0
    ego_width = 2.4
    ego_height = 2.2

    x0, x1 = -ego_length / 2, ego_length / 2
    y0, y1 = -ego_width / 2, ego_width / 2
    z0, z1 = 0.0, ego_height

    ego_vertices = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])

    fig.add_trace(
        go.Mesh3d(
            x=ego_vertices[:, 0],
            y=ego_vertices[:, 1],
            z=ego_vertices[:, 2],
            i=[0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6, 3],
            j=[1, 2, 3, 2, 5, 3, 5, 6, 6, 1, 7, 7],
            k=[2, 3, 4, 5, 4, 7, 6, 7, 1, 2, 3, 4],
            color="white",
            opacity=0.45,
            name="Ego vehicle",
            hovertext="Ego vehicle",
            hoverinfo="text",
        )
    )

    # Forward direction.
    fig.add_trace(
        go.Scatter3d(
            x=[ego_length / 2, ego_length / 2 + 3],
            y=[0, 0],
            z=[ego_height / 2, ego_height / 2],
            mode="lines+markers",
            line=dict(
                color="white",
                width=6,
            ),
            marker=dict(
                color="white",
                size=[2, 6],
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[ego_height + 0.8],
            mode="text",
            text=["Ego vehicle"],
            textfont=dict(
                color="white",
                size=12,
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    has_highlight_filter = (
        highlight_category is not None
        or highlight_distance_range is not None
    )

    for box in boxes:
        distance = np.hypot(
            box.center[0],
            box.center[1],
        )

        highlighted = has_highlight_filter

        if highlight_category is not None:
            highlighted &= box.name == highlight_category

        if highlight_distance_range is not None:
            min_distance, max_distance = highlight_distance_range

            if min_distance is not None:
                highlighted &= distance >= min_distance

            if max_distance is not None:
                highlighted &= distance < max_distance

        box_color = "red" if highlighted else "gray"
        box_width = 7 if highlighted else 2

        corners = box.corners()

        for i, j in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[
                        corners[0, i],
                        corners[0, j],
                    ],
                    y=[
                        corners[1, i],
                        corners[1, j],
                    ],
                    z=[
                        corners[2, i],
                        corners[2, j],
                    ],
                    mode="lines",
                    line=dict(
                        width=box_width,
                        color=box_color,
                    ),
                    opacity=1.0 if highlighted else 0.4,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        if highlighted:
            fig.add_trace(
                go.Scatter3d(
                    x=[box.center[0]],
                    y=[box.center[1]],
                    z=[
                        box.center[2]
                        + box.wlh[2] / 2
                        + 0.8
                    ],
                    mode="text",
                    text=[box.name.split(".")[-1]],
                    textfont=dict(
                        color="red",
                        size=12,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title="3D LiDAR Point Cloud with Ground Truth Objects",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            bgcolor="black",
            xaxis=dict(
                title="Ego x (m)",
                range=list(x_range),
                backgroundcolor="black",
                gridcolor="gray",
                zerolinecolor="white",
                color="white",
            ),
            yaxis=dict(
                title="Ego y (m)",
                range=list(y_range),
                backgroundcolor="black",
                gridcolor="gray",
                zerolinecolor="white",
                color="white",
            ),
            zaxis=dict(
                title="Height (m)",
                range=list(z_range),
                backgroundcolor="black",
                gridcolor="gray",
                zerolinecolor="white",
                color="white",
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=1.8,
                    y=-1.8,
                    z=1.2,
                )
            ),
        ),
    )

    fig.show()