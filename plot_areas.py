import pandas as pd
import matplotlib.pyplot as plt


def plot_area_metrics(zero_radar_summary, clear_scenes_full, output_fig=None):
    """
    Plot 3 bar charts: % objects with zero RADAR points by area, and RADAR
    median / 95th-percentile range by scene.

    Parameters
    ----------
    zero_radar_summary : pandas.DataFrame
        Indexed by area, with column 'pct_zero_radar'. Typically the output of
        analyze_areas.get_zero_radar_summary().
    clear_scenes_full : pandas.DataFrame
        Must contain columns 'area', 'radar_median_range', 'radar_p95_range'.
        Typically the first element returned by analyze_areas.analyze_areas().
    output_fig : str or None
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    area_order = ['terminal', 'highway', 'city']

    # Build unique per-scene labels (city (1), city (2), etc.) since some
    # areas have more than one scene in the clear-weather subset.
    plot_df = clear_scenes_full.copy()
    plot_df['area_cat'] = pd.Categorical(plot_df['area'], categories=area_order, ordered=True)
    plot_df = plot_df.sort_values('area_cat').reset_index(drop=True)

    counts = {}
    labels = []
    area_list = list(plot_df['area'])
    for a in area_list:
        counts[a] = counts.get(a, 0) + 1
        labels.append(a if area_list.count(a) == 1 else f"{a} ({counts[a]})")
    plot_df['bar_label'] = labels

    # =======================================================
    # Chart 1: % objects with zero RADAR points, by area
    ax = axes[0]
    zero_radar_plot = zero_radar_summary.reindex(area_order)
    bars = ax.bar(area_order, zero_radar_plot['pct_zero_radar'], color='#e76f51', alpha=0.85, edgecolor='black')
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 5),
                    textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    ax.set_title('% Objects with Zero RADAR Points\nby Area', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('% of objects', fontsize=11)
    ax.set_xlabel('Area', fontsize=11)

    # =======================================================
    # Chart 2: RADAR median range, by scene
    ax = axes[1]
    bars = ax.bar(plot_df['bar_label'], plot_df['radar_median_range'], color='#2b5c8f', alpha=0.85, edgecolor='black')
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 5),
                    textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    ax.set_title('RADAR Median Range (m)\nby Scene', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Median range (m)', fontsize=11)
    ax.set_xlabel('Area / Scene', fontsize=11)

    # =======================================================
    # Chart 3: RADAR 95th-percentile range, by scene
    ax = axes[2]
    bars = ax.bar(plot_df['bar_label'], plot_df['radar_p95_range'], color='#2a9d8f', alpha=0.85, edgecolor='black')
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 5),
                    textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    ax.set_title('RADAR 95th-Percentile Range (m)\nby Scene', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('P95 range (m)', fontsize=11)
    ax.set_xlabel('Area / Scene', fontsize=11)

    plt.tight_layout()

    if output_fig:
        plt.savefig(output_fig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_fig}")

    return fig


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Plot TruckScenes Different Areas metrics.")
    parser.add_argument("scenes_csv", nargs="?", default="different_areas_metrics_scenes.csv")
    parser.add_argument("zero_radar_csv", nargs="?", default="different_areas_metrics_zero_radar.csv")
    parser.add_argument("--output-fig", default="different_areas_radar_metrics.png")
    args = parser.parse_args()

    if not os.path.exists(args.scenes_csv) or not os.path.exists(args.zero_radar_csv):
        raise FileNotFoundError("Run analyze_areas.py first to generate the required CSVs.")

    clear_scenes_full = pd.read_csv(args.scenes_csv)
    zero_radar_summary = pd.read_csv(args.zero_radar_csv, index_col='area')
    plot_area_metrics(zero_radar_summary, clear_scenes_full, output_fig=args.output_fig)
    plt.show()