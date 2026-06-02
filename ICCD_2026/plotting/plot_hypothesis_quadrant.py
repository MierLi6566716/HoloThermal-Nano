import matplotlib.pyplot as plt
import os
import re
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
SUMMARY_FILE = "results/summary.txt"
OUTPUT_PATH = "results/plots/hypothesis_validation_quadrant.png"
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

def plot_quadrant_comparison(df):
    if df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(11, 7))
    
    # Professional color palette and markers
    styles = {
        "baseline": {"color": "#e31a1c", "marker": "X", "label": "Baseline (Reactive)"},
        "concurrent": {"color": "#1f78b4", "marker": "o", "label": "Concurrent (Thread Alloc)"},
        "dvfs": {"color": "#33a02c", "marker": "^", "label": "DVFS (Linux Governor)"},
        "loople": {"color": "#ff7f00", "marker": "s", "label": "Loople (Bandit)"},
        "predictive_headroom": {"color": "#6a3d9a", "marker": "D", "label": "Pred. Headroom (Lookahead)"},
        "sequential": {"color": "#777777", "marker": "P", "label": "Sequential (Single Core)"}
    }

    # Plot each scheme
    for _, row in df.iterrows():
        scheme = row['scheme']
        style = styles.get(scheme, {"color": "black", "marker": "*", "label": scheme})
        
        plt.scatter(row['elapsed_s'], row['average_temp_c'], 
                    s=250, alpha=0.8, edgecolors='black', linewidth=1.5,
                    c=style["color"], marker=style["marker"], label=style["label"])
        
        # Add label next to point
        plt.text(row['elapsed_s'] + 2, row['average_temp_c'], f" {scheme}", 
                 verticalalignment='center', fontsize=10, fontweight='bold')

    # Draw Quadrant Dividers (relative to the primary 'Baseline' or the median)
    # Using baseline as the "Current Industry Standard" reference
    if "baseline" in df['scheme'].values:
        ref_time = df.loc[df['scheme'] == 'baseline', 'elapsed_s'].values[0]
        ref_temp = df.loc[df['scheme'] == 'baseline', 'average_temp_c'].values[0]
    else:
        ref_time = df['elapsed_s'].median()
        ref_temp = df['average_temp_c'].median()

    plt.axvline(x=ref_time, color='black', linestyle='--', alpha=0.4, linewidth=1)
    plt.axhline(y=ref_temp, color='black', linestyle='--', alpha=0.4, linewidth=1)

    # Label the Quadrants
    # Get current plot limits to position text
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    plt.text(xlim[0] + (ref_time - xlim[0])/2, ylim[0] + (ref_temp - ylim[0])/2, 
             "OPTIMAL\n(Fast & Cool)", fontsize=14, fontweight='bold', color='green', 
             alpha=0.4, ha='center', va='center')
    
    plt.text(xlim[1] - (xlim[1] - ref_time)/2, ylim[1] - (ylim[1] - ref_temp)/2, 
             "FAILED\n(Slow & Hot)", fontsize=14, fontweight='bold', color='red', 
             alpha=0.4, ha='center', va='center')

    plt.xlabel("Execution Time (Seconds) → [Lower is Better]", fontsize=13, fontweight='bold')
    plt.ylabel("Average CPU Temperature (°C) → [Lower is Better]", fontsize=13, fontweight='bold')
    plt.title("Performance-Thermal Pareto Analysis\nTesting the Hypothesis: Thread Allocation Efficiency", 
              fontsize=16, pad=20, fontweight='bold')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Hypothesis validation plot saved to: {OUTPUT_PATH}")
    plt.close()

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))
        
    df = parse_summary(SUMMARY_FILE)
    plot_quadrant_comparison(df)

if __name__ == "__main__":
    main()
