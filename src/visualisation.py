from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyquaternion import Quaternion
import plotly.express as px
from PIL import Image
from truckscenes.utils.geometry_utils import BoxVisibility

from .config import SENSOR_ROOT
from .lidar_velocity import load_lidar_points
from .radar_velocity import load_radar_points


def load_sample_lidar_points(truckscenes, sample_sensor_data, sensor_root=None):
    """Load and align all keyframe LiDAR points to one sample ego frame."""

    sensor_root = Path(sensor_root or SENSOR_ROOT)

    lidar_frames = sample_sensor_data[sample_sensor_data["Modality"] == "LiDAR"]
    sample_tokens = lidar_frames["Sample Token"].unique()

    if len(sample_tokens) != 1:
        raise ValueError("LiDAR observations must belong to one sample.")

    sample = truckscenes.get("sample", sample_tokens[0])

    reference_pose = truckscenes.getclosest("ego_pose", sample["timestamp"])
    reference_rotation = Quaternion(reference_pose["rotation"]).rotation_matrix
    reference_translation = np.asarray(reference_pose["translation"], dtype=float).reshape(3, 1)

    point_tables = []

    for _, frame in lidar_frames.iterrows():
        sample_data = truckscenes.get("sample_data", frame["Token"])
        calibrated_sensor = truckscenes.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        ego_pose = truckscenes.get("ego_pose", sample_data["ego_pose_token"])
        points = load_lidar_points(sensor_root / sample_data["filename"]).copy()

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
    save_path=None,
):
    """Plot LiDAR BEV with highlighted Ground Truth objects."""

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

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

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
    save_path=None,
):
    """Plot interactive 3D LiDAR point cloud with highlighted GT boxes."""

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

    if save_path is not None:
        fig.write_html(
            save_path,
            include_plotlyjs=True,
        )

    fig.show()


def load_sample_radar_points(
    truckscenes,
    sample_sensor_data,
    sensor_root=None,
):
    """Load and align all keyframe RADAR returns to one sample ego frame."""

    sensor_root = Path(sensor_root or SENSOR_ROOT)

    radar_frames = sample_sensor_data[
        sample_sensor_data["Modality"] == "RADAR"
    ]

    sample_tokens = radar_frames["Sample Token"].unique()

    if len(sample_tokens) != 1:
        raise ValueError("RADAR observations must belong to one sample.")

    sample = truckscenes.get(
        "sample",
        sample_tokens[0],
    )

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

    for _, frame in radar_frames.iterrows():
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

        points = load_radar_points(
            sensor_root / sample_data["filename"]
        ).copy()

        xyz = points[
            ["x", "y", "z"]
        ].to_numpy().T

        # RADAR sensor -> ego at acquisition time
        xyz = (
            Quaternion(
                calibrated_sensor["rotation"]
            ).rotation_matrix
            @ xyz
        )

        xyz += np.asarray(
            calibrated_sensor["translation"],
            dtype=float,
        ).reshape(3, 1)

        # Ego at acquisition time -> global
        xyz = (
            Quaternion(
                ego_pose["rotation"]
            ).rotation_matrix
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

    return pd.concat(
        point_tables,
        ignore_index=True,
    )


def plot_radar_bev_with_boxes(
    radar_points,
    boxes,
    highlight_category=None,
    highlight_distance_range=None,
    x_range=(-50, 100),
    y_range=(-50, 50),
    save_path=None,
):
    """Plot RADAR BEV with highlighted Ground Truth objects."""

    visible = radar_points[
        radar_points["x"].between(*x_range)
        & radar_points["y"].between(*y_range)
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        visible["x"],
        visible["y"],
        s=18,
        c=visible["rcs"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="none",
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
    ax.set_title("RADAR BEV with Ground Truth Objects")
    ax.grid(alpha=0.2)
    ax.legend()

    cbar = plt.colorbar(
        scatter,
        ax=ax,
        pad=0.01,
    )
    cbar.set_label("Radar Cross Section (RCS)")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )
    plt.show()


def plot_radar_3d_with_boxes(
    radar_points,
    boxes,
    highlight_category=None,
    highlight_distance_range=None,
    x_range=(-50, 100),
    y_range=(-50, 50),
    z_range=(-5, 10),
    save_path=None,
):
    """Plot interactive 3D RADAR returns with highlighted GT boxes."""

    visible = radar_points[
        radar_points["x"].between(*x_range)
        & radar_points["y"].between(*y_range)
        & radar_points["z"].between(*z_range)
    ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=visible["x"],
            y=visible["y"],
            z=visible["z"],
            mode="markers",
            marker=dict(
                size=3,
                color=visible["rcs"],
                colorscale="Viridis",
                opacity=0.85,
                colorbar=dict(
                    title="RCS",
                    tickfont=dict(color="white"),
                    title_font=dict(color="white"),
                ),
            ),
            name="RADAR returns",
        )
    )

    # Schematic ego vehicle marker.
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

    # Ego forward direction.
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
                    x=[corners[0, i], corners[0, j]],
                    y=[corners[1, i], corners[1, j]],
                    z=[corners[2, i], corners[2, j]],
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
        title="3D RADAR Point Cloud with Ground Truth Objects",
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

    if save_path is not None:
        fig.write_html(
            save_path,
            include_plotlyjs=True,
        )

    fig.show()


def select_camera_view(
    truckscenes,
    sample_sensor_data,
    annotation_tokens,
):
    """Select the camera containing the most selected GT objects."""

    camera_frames = sample_sensor_data[
        sample_sensor_data["Modality"] == "Camera"
    ]

    results = []

    for _, frame in camera_frames.iterrows():
        _, boxes, _ = truckscenes.get_sample_data(
            frame["Token"],
            box_vis_level=BoxVisibility.ANY,
            selected_anntokens=annotation_tokens,
        )

        results.append({
            "frame": frame,
            "visible_objects": len(boxes),
        })

    if not results:
        raise ValueError("No Camera observations found for this sample.")

    return max(
        results,
        key=lambda item: item["visible_objects"],
    )


def plot_camera_with_boxes(
    truckscenes,
    camera_frame,
    sample_objects,
    sensor_root=None,
    highlight_category=None,
    highlight_distance_range=None,
):
    """Plot a Camera image with projected Ground Truth boxes."""

    sensor_root = Path(sensor_root or SENSOR_ROOT)

    annotation_tokens = sample_objects[
        "annotation_token"
    ].tolist()

    _, boxes, camera_intrinsic = truckscenes.get_sample_data(
        camera_frame["Token"],
        box_vis_level=BoxVisibility.ANY,
        selected_anntokens=annotation_tokens,
    )

    image = Image.open(
        sensor_root / camera_frame["Filename"]
    )

    object_lookup = (
        sample_objects
        .set_index("annotation_token")
        .to_dict("index")
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image)

    has_highlight_filter = (
        highlight_category is not None
        or highlight_distance_range is not None
    )

    highlighted_count = 0

    for box in boxes:
        obj = object_lookup.get(box.token)

        if obj is None:
            continue

        highlighted = has_highlight_filter

        if highlight_category is not None:
            highlighted &= obj["category"] == highlight_category

        if highlight_distance_range is not None:
            min_distance, max_distance = highlight_distance_range
            distance = obj["distance_m"]

            if min_distance is not None:
                highlighted &= distance >= min_distance

            if max_distance is not None:
                highlighted &= distance < max_distance

        color = "red" if highlighted else "gray"

        box.render(
            ax,
            view=np.asarray(camera_intrinsic),
            normalize=True,
            colors=(color, color, color),
            linewidth=2.5 if highlighted else 1.0,
        )

        if highlighted:
            highlighted_count += 1

            corners = box.corners()
            projected = np.asarray(camera_intrinsic) @ corners
            projected /= projected[2:3, :]

            ax.text(
                projected[0].mean(),
                projected[1].min(),
                obj["category"].split(".")[-1],
                color="red",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.axis("off")

    ax.set_title(
        f'{camera_frame["Sensor"]} | '
        f'Highlighted objects: {highlighted_count}'
    )

    plt.tight_layout()
    plt.show()


def plot_all_camera_views(
    truckscenes,
    sample_sensor_data,
    sample_objects,
    sensor_root=None,
    highlight_category=None,
    highlight_distance_range=None,
    save_path=None,
):
    """Plot all four Camera views for one sample with projected GT boxes."""

    sensor_root = Path(sensor_root or SENSOR_ROOT)

    camera_frames = (
        sample_sensor_data[
            (sample_sensor_data["Modality"] == "Camera")
            & (sample_sensor_data["Is Key Frame"] == True)
        ]
        .sort_values("Sensor")
        .reset_index(drop=True)
    )

    if len(camera_frames) != 4:
        raise ValueError(
            f"Expected 4 Camera keyframes, found {len(camera_frames)}."
        )

    annotation_tokens = sample_objects[
        "annotation_token"
    ].tolist()

    object_lookup = (
        sample_objects
        .set_index("annotation_token")
        .to_dict("index")
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 10),
    )

    axes = axes.flatten()

    has_highlight_filter = (
        highlight_category is not None
        or highlight_distance_range is not None
    )

    for ax, (_, frame) in zip(
        axes,
        camera_frames.iterrows(),
    ):
        _, boxes, camera_intrinsic = (
            truckscenes.get_sample_data(
                frame["Token"],
                box_vis_level=BoxVisibility.ANY,
                selected_anntokens=annotation_tokens,
            )
        )

        image = Image.open(
            sensor_root / frame["Filename"]
        )

        ax.imshow(image)

        highlighted_count = 0

        for box in boxes:
            obj = object_lookup.get(box.token)

            if obj is None:
                continue

            highlighted = has_highlight_filter

            if highlight_category is not None:
                highlighted &= (
                    obj["category"]
                    == highlight_category
                )

            if highlight_distance_range is not None:
                min_distance, max_distance = (
                    highlight_distance_range
                )

                distance = obj["distance_m"]

                if min_distance is not None:
                    highlighted &= (
                        distance >= min_distance
                    )

                if max_distance is not None:
                    highlighted &= (
                        distance < max_distance
                    )

            color = (
                "red"
                if highlighted
                else "gray"
            )

            box.render(
                ax,
                view=np.asarray(
                    camera_intrinsic
                ),
                normalize=True,
                colors=(
                    color,
                    color,
                    color,
                ),
                linewidth=(
                    2.5
                    if highlighted
                    else 0.8
                ),
            )

            if highlighted:
                highlighted_count += 1

                corners = box.corners()

                projected = (
                    np.asarray(
                        camera_intrinsic
                    )
                    @ corners
                )

                projected /= projected[
                    2:3,
                    :
                ]

                ax.text(
                    projected[0].mean(),
                    projected[1].min(),
                    obj["category"]
                    .split(".")[-1],
                    color="red",
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )

        ax.set_xlim(
            0,
            image.width,
        )

        ax.set_ylim(
            image.height,
            0,
        )

        ax.set_title(
            f'{frame["Sensor"]} | '
            f'Targets: {highlighted_count}'
        )

        ax.axis("off")

    fig.suptitle(
        "Camera Views with Ground Truth Objects",
        fontsize=16,
    )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )
    plt.show()


def plot_sensor_triptych(
    truckscenes,
    camera_frame,
    lidar_points,
    radar_points,
    boxes,
    sample_objects,
    sensor_root=None,
    highlight_category=None,
    highlight_distance_range=None,
    x_range=(-50, 100),
    y_range=(-50, 50),
):
    """Plot Camera, LiDAR BEV and RADAR BEV for the same sample."""

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    sensor_root = Path(sensor_root or SENSOR_ROOT)

    annotation_tokens = sample_objects[
        "annotation_token"
    ].tolist()

    object_lookup = (
        sample_objects
        .set_index("annotation_token")
        .to_dict("index")
    )

    has_highlight_filter = (
        highlight_category is not None
        or highlight_distance_range is not None
    )

    def is_highlighted(obj):
        highlighted = has_highlight_filter

        if highlight_category is not None:
            highlighted &= (
                obj["category"] == highlight_category
            )

        if highlight_distance_range is not None:
            min_distance, max_distance = (
                highlight_distance_range
            )

            distance = obj["distance_m"]

            if min_distance is not None:
                highlighted &= distance >= min_distance

            if max_distance is not None:
                highlighted &= distance < max_distance

        return highlighted

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 7),
    )

    # --------------------------------------------------
    # Camera
    # --------------------------------------------------

    _, camera_boxes, camera_intrinsic = (
        truckscenes.get_sample_data(
            camera_frame["Token"],
            box_vis_level=BoxVisibility.ANY,
            selected_anntokens=annotation_tokens,
        )
    )

    image = Image.open(
        sensor_root / camera_frame["Filename"]
    )

    axes[0].imshow(image)

    for box in camera_boxes:
        obj = object_lookup.get(box.token)

        if obj is None:
            continue

        highlighted = is_highlighted(obj)

        color = (
            "red"
            if highlighted
            else "gray"
        )

        box.render(
            axes[0],
            view=np.asarray(camera_intrinsic),
            normalize=True,
            colors=(color, color, color),
            linewidth=(
                2.5
                if highlighted
                else 0.8
            ),
        )

        if highlighted:
            corners = box.corners()

            projected = (
                np.asarray(camera_intrinsic)
                @ corners
            )

            projected /= projected[2:3, :]

            axes[0].text(
                projected[0].mean(),
                projected[1].min(),
                obj["category"].split(".")[-1],
                color="red",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

    axes[0].set_xlim(0, image.width)
    axes[0].set_ylim(image.height, 0)
    axes[0].set_title(camera_frame["Sensor"])
    axes[0].axis("off")

    # --------------------------------------------------
    # LiDAR BEV
    # --------------------------------------------------

    lidar_visible = lidar_points[
        lidar_points["x"].between(*x_range)
        & lidar_points["y"].between(*y_range)
    ]

    axes[1].scatter(
        lidar_visible["x"],
        lidar_visible["y"],
        s=0.4,
        alpha=0.8,
    )

    # --------------------------------------------------
    # RADAR BEV
    # --------------------------------------------------

    radar_visible = radar_points[
        radar_points["x"].between(*x_range)
        & radar_points["y"].between(*y_range)
    ]

    axes[2].scatter(
        radar_visible["x"],
        radar_visible["y"],
        s=12,
        c=radar_visible["rcs"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="none",
    )

    # --------------------------------------------------
    # Same GT boxes on LiDAR and RADAR
    # --------------------------------------------------

    for box in boxes:
        obj = object_lookup.get(box.token)

        if obj is None:
            continue

        highlighted = is_highlighted(obj)

        color = (
            "red"
            if highlighted
            else "gray"
        )

        linewidth = (
            2.5
            if highlighted
            else 0.8
        )

        corners = box.bottom_corners()

        x = corners[0, [0, 1, 2, 3, 0]]
        y = corners[1, [0, 1, 2, 3, 0]]

        axes[1].plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            alpha=1.0 if highlighted else 0.55,
        )

        axes[2].plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            alpha=1.0 if highlighted else 0.55,
        )

        if highlighted:
            label = obj["category"].split(".")[-1]

            axes[1].text(
                box.center[0],
                box.center[1],
                label,
                color="red",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

            axes[2].text(
                box.center[0],
                box.center[1],
                label,
                color="red",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

    # --------------------------------------------------
    # Ego vehicle
    # --------------------------------------------------

    for ax in axes[1:]:
        ax.scatter(
            0,
            0,
            marker="x",
            s=70,
            color="tab:orange",
            linewidths=2,
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
        ax.grid(alpha=0.2)

    axes[1].set_ylabel("Ego y (m)")
    axes[1].set_title("LiDAR BEV")
    axes[2].set_title("RADAR BEV")

    fig.suptitle(
        "Cross-Sensor Ground Truth Comparison",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()