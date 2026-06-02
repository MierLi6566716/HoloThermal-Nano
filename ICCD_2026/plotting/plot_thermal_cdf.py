import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
CSV_DIR = "results/csv"
OUTPUT_PATH = "results/plots/thermal_reliability_cdf.png"
# ==========================================

def plot_thermal_cdf():
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {CSV_DIR}")
        return

    plt.figure(figsize=(10, 7))
    
    # Professional color palette
    styles = {
        "baseline": {"color": "#e31a1c", "linestyle": "-"},
        "concurrent": {"color": "#1f78b4", "linestyle": "-"},
        "dvfs": {"color": "#33a02c", "linestyle": "--"},
        "loople": {"color": "#ff7f00", "linestyle": "-."},
        "predictive_headroom": {"color": "#6a3d9a", "linestyle": ":"},
        "sequential": {"color": "#777777", "linestyle": "--"}
    }

    for csv_path in sorted(csv_files):
        scheme = os.path.basename(csv_path).replace(".csv", "")
        if "__" in scheme:
            scheme = scheme.split("__")[0]
            
        try:
            df = pd.read_csv(csv_path)
            if "temp_c" not in df.columns:
                continue
                
            # Calculate CDF
            sorted_temps = np.sort(df["temp_c"])
            y_values = np.arange(len(sorted_temps)) / float(len(sorted_temps) - 1)
            
            style = styles.get(scheme, {"color": None, "linestyle": "-"})
            
            plt.plot(sorted_temps, y_values * 100, 
                     label=scheme, 
                     color=style["color"], 
                     linestyle=style["linestyle"],
                     linewidth=2.5)
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    plt.axvline(x=45, color='red', linestyle=':', alpha=0.5, label='Thermal Threshold (45°C)')
    
    plt.xlabel("CPU Temperature (°C)", fontsize=13, fontweight='bold')
    plt.ylabel("Cumulative Execution Time (%)", fontsize=13, fontweight='bold')
    plt.title("Thermal Reliability Analysis (CDF)\nHow much time is spent at high temperatures?", 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    
    # Add annotation explaining the plot
    plt.annotate('Shifted Left = Better Reliability\n(More time at lower temps)', 
                 xy=(40, 80), xytext=(35, 90),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Thermal CDF plot saved to: {OUTPUT_PATH}")
    plt.close()

if __name__ == "__main__":
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))
    plot_thermal_cdf()
