import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SENSOR_ORDER = ["RADAR", "LiDAR", "Fused"]


def _ordered_categories(df, condition):
    values = df[condition].dropna().unique().tolist()

    if condition == "visibility":
        return sorted(values)

    if condition == "distance":
        return sorted(values, key=lambda x: x.left if hasattr(x, "left") else str(x))

    return values


def plot_fusion_complementarity(complementarity):
    """Plot complementary coverage and individual/combined availability."""

    table = complementarity.set_index("metric")
    states = ["Both", "RADAR only", "LiDAR only", "Neither"]
    availability = ["RADAR stable", "LiDAR available", "Combined available"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    left = 0
    for state in states:
        value = table.loc[state, "pct_reference"]
        axes[0].barh(["Reference intervals"], value, left=left, label=state)
        axes[0].text(left + value / 2, 0, f"{value:.1f}%", ha="center", va="center", fontsize=8)
        left += value

    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Reference intervals (%)")
    axes[0].set_title("Sensor Complementarity")
    axes[0].legend(fontsize=8)

    values = table.loc[availability, "pct_reference"]
    bars = axes[1].bar(availability, values)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Reference intervals (%)")
    axes[1].set_title("Velocity Availability")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].bar_label(bars, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    plt.show()


def plot_overall_fusion_accuracy(fusion_accuracy):
    """Compare overall RADAR, LiDAR and fused accuracy."""

    metrics = [
        ("speed_err_median", "Median Speed Error", "m/s"),
        ("speed_err_iqr", "Speed Error IQR", "m/s"),
        ("speed_err_p90", "P90 Speed Error", "m/s"),
        ("dir_err_median", "Median Direction Error", "degrees"),
    ]

    data = fusion_accuracy.set_index("sensor").reindex(SENSOR_ORDER)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (column, title, ylabel) in zip(axes.flat, metrics):
        bars = ax.bar(data.index, data[column])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    fig.suptitle("Overall RADAR–LiDAR Fusion Performance", fontsize=14)
    plt.tight_layout()
    plt.show()


def _plot_condition_error(ax, summary, condition, title):
    order = _ordered_categories(summary, condition)
    pivot = summary.pivot(index=condition, columns="sensor", values="speed_err_median")
    pivot = pivot.reindex(index=order, columns=SENSOR_ORDER)

    pivot.plot(kind="bar", ax=ax, rot=45 if condition == "object_type" else 0)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Median absolute speed error (m/s)")
    ax.grid(axis="y", alpha=0.3)


def plot_fusion_condition_grid(area, weather, visibility, object_type):
    """Compare median RADAR, LiDAR and fused error across categorical conditions."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    _plot_condition_error(axes[0, 0], area, "area", "Area")
    _plot_condition_error(axes[0, 1], weather, "weather", "Weather")
    _plot_condition_error(axes[1, 0], visibility, "visibility", "Visibility")
    _plot_condition_error(axes[1, 1], object_type, "object_type", "Object Type")

    fig.suptitle("Velocity Error Across Operating Conditions", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_fusion_distance(distance_summary):
    """Compare median RADAR, LiDAR and fused error across ordered distance bands."""

    order = _ordered_categories(distance_summary, "distance")
    pivot = distance_summary.pivot(index="distance", columns="sensor", values="speed_err_median")
    pivot = pivot.reindex(index=order, columns=SENSOR_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))

    for sensor in SENSOR_ORDER:
        ax.plot(range(len(pivot)), pivot[sensor], marker="o", linewidth=2, label=sensor)

    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels([str(x) for x in pivot.index], rotation=45)
    ax.set_xlabel("Distance Band (m)")
    ax.set_ylabel("Median absolute speed error (m/s)")
    ax.set_title("Velocity Error by Distance")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def _plot_gain(ax, table, condition, title):
    order = _ordered_categories(table, condition)
    plot_df = table.set_index(condition).reindex(order)
    x = np.arange(len(plot_df))
    width = 0.36

    ax.bar(x - width / 2, plot_df["fusion_minus_lidar"], width, label="Fusion − LiDAR")
    ax.bar(x + width / 2, plot_df["fusion_minus_radar"], width, label="Fusion − RADAR")
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    labels = [
        str(v).replace("vehicle.", "")
        for v in plot_df.index
    ]
    ax.set_xticklabels(labels, rotation=45 if condition == "object_type" else 0)
    ax.set_title(title)
    ax.set_ylabel("Change in median error (m/s)")
    ax.grid(axis="y", alpha=0.3)


def plot_fusion_gain_grid(area, weather, visibility, object_type):
    """Plot fusion median-error change across categorical conditions."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    _plot_gain(axes[0, 0], area, "area", "Area")
    _plot_gain(axes[0, 1], weather, "weather", "Weather")
    _plot_gain(axes[1, 0], visibility, "visibility", "Visibility")
    _plot_gain(axes[1, 1], object_type, "object_type", "Object Type")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(
        "Fusion Change in Median Speed Error Across Conditions\n"
        "Negative = lower error after fusion",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def plot_fusion_gain_distance(distance_gain):
    """Plot fusion median-error change across ordered distance bands."""

    order = _ordered_categories(distance_gain, "distance")
    data = distance_gain.set_index("distance").reindex(order)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(data))

    ax.plot(x, data["fusion_minus_lidar"], marker="o", linewidth=2, label="Fusion − LiDAR")
    ax.plot(x, data["fusion_minus_radar"], marker="o", linewidth=2, label="Fusion − RADAR")
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in data.index], rotation=45)
    ax.set_xlabel("Distance Band (m)")
    ax.set_ylabel("Change in median error (m/s)")
    ax.set_title("Fusion Gain/Loss by Distance")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()