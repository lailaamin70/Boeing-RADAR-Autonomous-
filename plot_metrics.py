import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the extracted metrics
csv_path = "visibility_sensor_metrics.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Cannot find {csv_path}. Please run analyze_visibility.py first.")

df = pd.read_csv(csv_path)

# Set global figure style for technical report
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

metrics = [
    ("Camera_Brightness", "Camera Brightness (Grayscale 0-255)", "#2b5c8f"),
    ("LiDAR_Points", "LiDAR Valid Return Points", "#2a9d8f"),
    ("Radar_Points", "Radar Detection Returns", "#e76f51")
]

# Order conditions logically
condition_order = ["Clear", "Rainy", "Snowy"]
df["Condition"] = pd.Categorical(df["Condition"], categories=condition_order, ordered=True)

# 2. Generate subplots
for idx, (col_name, label, color) in enumerate(metrics):
    ax = axes[idx]
    
    # Calculate group means and standard deviations
    summary = df.groupby("Condition", observed=False)[col_name].agg(["mean", "std"])
    x_positions = range(len(condition_order))
    
    # Plot bar chart with error bars
    bars = ax.bar(
        x_positions, 
        summary["mean"], 
        yerr=summary["std"], 
        capsize=6, 
        color=color, 
        alpha=0.85, 
        edgecolor='black', 
        linewidth=1.2
    )
    
    # Add numerical labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title(f"{label}\nvs. Weather Condition", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Visibility / Weather Condition", fontsize=11, labelpad=8)
    ax.set_ylabel(label, fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(condition_order, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# 3. Save plot for the report
output_fig = "sensor_degradation_comparison.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"Academic figure successfully generated and saved to: {output_fig}")
plt.show()