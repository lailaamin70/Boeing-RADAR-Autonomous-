import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _coef_table(model):
    ci = model.conf_int()
    return pd.DataFrame({
        "coefficient": model.params,
        "ci_low": ci[0],
        "ci_high": ci[1],
    })


def _is_full_rank(model):
    X = model.model.exog
    return np.linalg.matrix_rank(X) == X.shape[1]


def _sensor_term_label(term):
    values = re.findall(r"\[T\.([^\]]+)\]", term)
    if "distance_10m" in term:
        return "Distance (+10 m)"
    value = values[-1].replace("vehicle.", "")
    if "C(area" in term:
        return f"Area: {value}"
    if "C(weather" in term:
        return f"Weather: {value}"
    if "C(visibility" in term:
        return f"Visibility: {value}"
    if "C(object_type" in term:
        return f"Object: {value}"
    return term


def plot_sensor_interaction_coefficients(model):
    """Plot Sensor × Condition interaction coefficients with 95% confidence intervals."""

    table = _coef_table(model)
    table = table[
        table.index.str.contains("C\\(sensor") &
        table.index.str.contains(":")
    ].copy()
    table["label"] = [_sensor_term_label(term) for term in table.index]

    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(
        table["coefficient"], y,
        xerr=[
            table["coefficient"] - table["ci_low"],
            table["ci_high"] - table["coefficient"],
        ],
        fmt="o", capsize=3,
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Interaction coefficient on log(1 + absolute speed error)")
    ax.set_title("Sensor–Condition Interaction Effects")
    ax.grid(axis="x", alpha=0.3)

    for i, value in enumerate(table["coefficient"]):
        ax.annotate(
            f"{value:.3f}", (value, i),
            xytext=(6 if value >= 0 else -6, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            va="center", fontsize=8,
        )

    plt.tight_layout()
    plt.show()


def plot_sensor_distance_interaction(model, model_df):
    """Plot adjusted RADAR and LiDAR predictions across distance."""

    distance_range = np.linspace(
        model_df["distance_10m"].min(),
        model_df["distance_10m"].max(),
        100,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for sensor in ["RADAR", "LiDAR"]:
        pred_df = pd.DataFrame({
            "sensor": sensor,
            "area": "city",
            "weather": "clear",
            "visibility": 4,
            "object_type": "vehicle.car",
            "distance_10m": distance_range,
        })

        predicted_error = np.expm1(model.predict(pred_df))
        ax.plot(distance_range * 10, predicted_error, linewidth=2, label=sensor)

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Back-transformed predicted error (m/s)")
    ax.set_title("Sensor × Distance Interaction")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def _condition_predictions(model, model_df, condition):
    reference = {
        "area": "city",
        "weather": "clear",
        "visibility": 4,
        "object_type": "vehicle.car",
        "distance_10m": model_df["distance_10m"].median(),
    }

    if condition == "visibility":
        levels = sorted(model_df[condition].unique())
    elif condition == "object_type":
        levels = model_df[condition].value_counts().index.tolist()
    else:
        levels = model_df[condition].drop_duplicates().tolist()

    rows = []
    for level in levels:
        for sensor in ["RADAR", "LiDAR"]:
            pred_df = pd.DataFrame([{**reference, condition: level, "sensor": sensor}])
            rows.append({
                condition: level,
                "sensor": sensor,
                "predicted_error": np.expm1(model.predict(pred_df).iloc[0]),
            })

    result = pd.DataFrame(rows)
    if condition == "object_type":
        result[condition] = result[condition].str.replace("vehicle.", "", regex=False)

    return result


def plot_sensor_condition_grid(model, model_df):
    """Plot adjusted RADAR-LiDAR predictions across categorical conditions."""

    settings = [
        ("area", "Area"),
        ("weather", "Weather"),
        ("visibility", "Visibility"),
        ("object_type", "Object Type"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for ax, (condition, title) in zip(axes.flat, settings):
        result = _condition_predictions(model, model_df, condition)
        pivot = result.pivot(index=condition, columns="sensor", values="predicted_error")
        pivot = pivot.reindex(columns=["RADAR", "LiDAR"])

        pivot.plot(
            kind="bar",
            ax=ax,
            rot=45 if condition == "object_type" else 0,
            color=["C0", "C1"],
        )

        ax.set_title(f"Sensor × {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Back-transformed predicted error (m/s)")
        ax.grid(axis="y", alpha=0.3)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)

    fig.suptitle("Adjusted RADAR–LiDAR Error Across Operating Conditions", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_condition_interactions(model, top_n=20):
    """Plot the largest Condition × Condition coefficients when the model is full rank."""

    if not _is_full_rank(model):
        print(
            "Condition–condition interaction plot skipped: "
            "the current model is rank-deficient."
        )
        return

    table = _coef_table(model)
    table = table[
        table.index.str.contains(":") &
        ~table.index.str.contains("C\\(sensor")
    ].copy()

    table["abs_coefficient"] = table["coefficient"].abs()
    table = table.sort_values("abs_coefficient", ascending=False).head(top_n)

    labels = [
        term
        .replace('C(area, Treatment(reference="city"))', "Area")
        .replace('C(weather, Treatment(reference="clear"))', "Weather")
        .replace("C(visibility, Treatment(reference=4))", "Visibility")
        .replace('C(object_type, Treatment(reference="vehicle.car"))', "Object")
        .replace("distance_10m", "Distance")
        .replace("[T.", ": ")
        .replace("]", "")
        .replace("vehicle.", "")
        for term in table.index
    ]

    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.errorbar(
        table["coefficient"], y,
        xerr=[
            table["coefficient"] - table["ci_low"],
            table["ci_high"] - table["coefficient"],
        ],
        fmt="o", capsize=3,
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Interaction coefficient on log(1 + absolute speed error)")
    ax.set_title(f"Largest Condition–Condition Interaction Effects (Top {top_n})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()