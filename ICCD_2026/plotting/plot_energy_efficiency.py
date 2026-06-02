import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
SUMMARY_FILE = "results/summary.txt"
OUTPUT_PATH = "results/plots/energy_efficiency_comparison.png"
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
            # Convert numeric strings
            for key in ["average_temp_c", "elapsed_s", "total_energy_j"]:
                if key in entry:
                    entry[key] = float(entry[key])
            data.append(entry)
    return pd.DataFrame(data)

def plot_energy_metrics(df):
    if df.empty:
        print("No data to plot.")
        return

    # Filter out 'sequential' if it's too large and skews the graph, 
    # but for now let's keep it and see.
    # df = df[df['scheme'] != 'sequential']

    # Calculate EDP (Energy-Delay Product) = Energy * Time
    # Lower is better. This metric penalizes being slow and being power-hungry.
    df['edp'] = (df['total_energy_j'] * df['elapsed_s']) / 1000.0 # Normalized for scale

    df = df.sort_values(by='total_energy_j')

    schemes = df['scheme'].tolist()
    energy = df['total_energy_j'].tolist()
    edp = df['edp'].tolist()

    x = np.arange(len(schemes))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Bar 1: Total Energy
    rects1 = ax1.bar(x - width/2, energy, width, label='Total Energy (Joules)', color='#1f77b4', alpha=0.8)
    ax1.set_ylabel('Total Energy Consumption (J)', fontsize=12, fontweight='bold', color='#1f77b4')
    ax1.set_xlabel('Thermal Management Scheme', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    # Bar 2: EDP (Second Axis)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, edp, width, label='Energy-Delay Product (EDP)', color='#d62728', alpha=0.8)
    ax2.set_ylabel('Energy-Delay Product (Normalized)', fontsize=12, fontweight='bold', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    # Add labels, title and custom x-axis tick labels, etc.
    plt.title('Energy Efficiency & EDP Analysis\nQuantifying Total System Impact', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(schemes, rotation=15, ha='right')
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, shadow=True)

    # Grid
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects, ax, color):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

    autolabel(rects1, ax1, '#1f77b4')
    autolabel(rects2, ax2, '#d62728')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Energy efficiency plot saved to: {OUTPUT_PATH}")
    plt.close()

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))
        
    df = parse_summary(SUMMARY_FILE)
    plot_energy_metrics(df)

if __name__ == "__main__":
    main()
