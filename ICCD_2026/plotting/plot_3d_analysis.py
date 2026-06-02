import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
SUMMARY_FILE = "results/summary.txt"
OUTPUT_PATH = "results/plots/3d_tradeoff_analysis.png"
# ==========================================

def parse_summary(file_path):
    """Parses the summary.txt file into a DataFrame."""
    data = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip() or "=" not in line:
                continue
            entry = dict(re.findall(r'(\w+)=([\d\.\w\-]+)', line))
            for key in ["average_temp_c", "elapsed_s", "total_energy_j"]:
                if key in entry:
                    entry[key] = float(entry[key])
            data.append(entry)
    return pd.DataFrame(data)

def plot_3d_tradeoff(df):
    if df.empty:
        print("No data to plot.")
        return

    # Filter out sequential for better 3D scaling (it's an extreme outlier)
    df_plot = df[df['scheme'] != 'sequential'].copy()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Styles
    styles = {
        "baseline": {"color": "#e31a1c", "marker": "X", "label": "Baseline"},
        "concurrent": {"color": "#1f78b4", "marker": "o", "label": "Concurrent"},
        "dvfs": {"color": "#33a02c", "marker": "^", "label": "DVFS"},
        "loople": {"color": "#ff7f00", "marker": "s", "label": "Loople"},
        "predictive_headroom": {"color": "#6a3d9a", "marker": "D", "label": "Pred. Headroom"}
    }

    # Plot points
    for _, row in df_plot.iterrows():
        scheme = row['scheme']
        style = styles.get(scheme, {"color": "black", "marker": "*", "label": scheme})
        
        # X: Time, Y: Temp, Z: Energy
        ax.scatter(row['elapsed_s'], row['average_temp_c'], row['total_energy_j'], 
                   s=300, c=style["color"], marker=style["marker"], 
                   edgecolors='black', linewidth=1.5, label=style["label"], alpha=0.9)
        
        # Add text labels
        ax.text(row['elapsed_s'], row['average_temp_c'], row['total_energy_j'] + 5, 
                f" {scheme}", fontsize=10, fontweight='bold')

    # Labels
    ax.set_xlabel('Execution Time (s)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Avg Temperature (°C)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel('Total Energy (J)', fontsize=14, fontweight='bold', labelpad=10)
    
    plt.title('3D System Trade-off Analysis\nTime vs. Heat vs. Energy Efficiency', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Customize the viewing angle to see the trade-offs best
    ax.view_init(elev=20, azim=45)
    
    # Improved Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), 
              title="Legend", title_fontsize='14', fontsize='11', 
              frameon=True, shadow=True, borderpad=1, labelspacing=1.2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"3D Trade-off plot saved to: {OUTPUT_PATH}")
    plt.close()

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))
    df = parse_summary(SUMMARY_FILE)
    plot_3d_tradeoff(df)

if __name__ == "__main__":
    main()
