import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
SUMMARY_FILE = "results/summary.txt"
OUTPUT_PATH = "results/plots/thermal_energy_correlation.png"
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

def plot_correlation(df):
    if df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 8))
    
    styles = {
        "baseline": {"color": "#e31a1c", "marker": "X", "label": "Baseline"},
        "concurrent": {"color": "#1f78b4", "marker": "o", "label": "Concurrent"},
        "dvfs": {"color": "#33a02c", "marker": "^", "label": "DVFS"},
        "loople": {"color": "#ff7f00", "marker": "s", "label": "Loople"},
        "predictive_headroom": {"color": "#6a3d9a", "marker": "D", "label": "Pred. Headroom"},
        "sequential": {"color": "#777777", "marker": "P", "label": "Sequential"}
    }

    # Plot points
    for _, row in df.iterrows():
        scheme = row['scheme']
        style = styles.get(scheme, {"color": "black", "marker": "*", "label": scheme})
        
        plt.scatter(row['average_temp_c'], row['total_energy_j'], 
                    s=350, alpha=0.85, edgecolors='black', linewidth=1.5,
                    c=style["color"], marker=style["marker"], label=style["label"])
        
        plt.text(row['average_temp_c'] + 0.1, row['total_energy_j'], f" {scheme}", 
                 verticalalignment='center', fontsize=11, fontweight='bold')

    # Add a "Thermal Efficiency Curve" (Visual guide showing the sweet spot)
    # This is a conceptual curve showing how energy drops as we use more of the thermal budget,
    # then rises again if we overheat.
    sorted_df = df.sort_values('average_temp_c')
    plt.plot(sorted_df['average_temp_c'], sorted_df['total_energy_j'], 
             color='gray', linestyle=':', alpha=0.3, zorder=0)

    plt.xlabel("Average CPU Temperature (°C)", fontsize=13, fontweight='bold')
    plt.ylabel("Total Energy Consumption (Joules)", fontsize=13, fontweight='bold')
    plt.title("The 'Sweet Spot' Analysis: Thermal vs. Energy Efficiency\nProving the Energy Penalty of Low-Temperature (Slow) Execution", 
              fontsize=16, fontweight='bold', pad=25)

    # Annotate the "Energy Penalty" area
    if "sequential" in df['scheme'].values:
        seq_data = df[df['scheme'] == 'sequential'].iloc[0]
        plt.annotate('ENERGY PENALTY\n(System stays ON too long)', 
                     xy=(seq_data['average_temp_c'], seq_data['total_energy_j']),
                     xytext=(seq_data['average_temp_c'] + 0.5, seq_data['total_energy_j'] - 200),
                     arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.6),
                     fontsize=12, fontweight='bold', color='red', ha='left')

    # Highlight the "Optimal Efficiency Band"
    if "concurrent" in df['scheme'].values:
        opt_data = df[df['scheme'] == 'concurrent'].iloc[0]
        plt.axvspan(opt_data['average_temp_c'] - 0.5, opt_data['average_temp_c'] + 1.0, 
                    color='green', alpha=0.1, label='Optimal Thermal Band')

    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='best', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Thermal-Energy correlation plot saved to: {OUTPUT_PATH}")
    plt.close()

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))
    df = parse_summary(SUMMARY_FILE)
    plot_correlation(df)

if __name__ == "__main__":
    main()
