
import matplotlib.pyplot as plt
import os
import re

def parse_summary(file_path):
    """
    Parses the summary.txt file into a list of dictionaries.
    """
    data = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return data

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip() or "=" not in line:
                continue
            # Parse line using regex to handle key=value pairs
            entry = dict(re.findall(r'(\w+)=([\d\.\w\-]+)', line))
            # Convert numeric strings to floats
            for key in ["start_temp_c", "baseline_temp_c", "peak_temp_c", "average_temp_c", "average_p_mw", "total_energy_j", "elapsed_s"]:
                if key in entry:
                    entry[key] = float(entry[key])
            data.append(entry)
    return data

def plot_quadrant_analysis(summary_path, output_path):
    all_data = parse_summary(summary_path)
    if not all_data:
        return

    # Group data by scheme
    schemes = {}
    for entry in all_data:
        s = entry.get("scheme", "unknown")
        if s not in schemes:
            schemes[s] = {"time": [], "temp": []}
        schemes[s]["time"].append(entry["elapsed_s"])
        schemes[s]["temp"].append(entry["average_temp_c"])

    plt.figure(figsize=(12, 8))
    
    # Define colors and markers for common schemes
    styles = {
        "baseline": {"color": "tab:red", "marker": "X", "label": "Baseline (Reactive Throttling)"},
        "concurrent": {"color": "tab:blue", "marker": "o", "label": "Concurrent (Data Parallel)"},
        "loople": {"color": "tab:green", "marker": "s", "label": "Loople (CRUX Bandit)"},
        "predictive_headroom": {"color": "tab:purple", "marker": "D", "label": "Predictive Headroom (Lookahead)"}
    }

    # Plot each scheme
    for name, vals in schemes.items():
        style = styles.get(name, {"color": None, "marker": "*", "label": name})
        plt.scatter(vals["time"], vals["temp"], 
                    s=150, alpha=0.8, edgecolors='black',
                    c=style["color"], marker=style["marker"], label=style["label"])

    # Calculate Quadrant Dividers (using medians for robust center)
    all_times = [e["elapsed_s"] for e in all_data]
    all_temps = [e["average_temp_c"] for e in all_data]
    mid_time = (max(all_times) + min(all_times)) / 2
    mid_temp = (max(all_temps) + min(all_temps)) / 2

    plt.axvline(x=mid_time, color='black', linestyle='--', alpha=0.3)
    plt.axhline(y=mid_temp, color='black', linestyle='--', alpha=0.3)

    # Label the Quadrants
    plt.text(plt.xlim()[0] + 0.1*(plt.xlim()[1]-plt.xlim()[0]), plt.ylim()[0] + 0.1*(plt.ylim()[1]-plt.ylim()[0]), 
             "OPTIMAL\n(Fast & Cool)", fontsize=12, fontweight='bold', color='green', alpha=0.5)
    
    plt.text(plt.xlim()[1] - 0.2*(plt.xlim()[1]-plt.xlim()[0]), plt.ylim()[1] - 0.1*(plt.ylim()[1]-plt.ylim()[0]), 
             "FAILED\n(Slow & Hot)", fontsize=12, fontweight='bold', color='red', alpha=0.5)

    plt.xlabel("Elapsed Execution Time (seconds)", fontsize=14, fontweight='bold')
    plt.ylabel("Average CPU Temperature (°C)", fontsize=14, fontweight='bold')
    plt.title("Performance-Thermal Quadrant Analysis\n(ICCD 2026 Thermal Benchmarking)", fontsize=16, pad=20)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Quadrant plot saved to: {output_path}")

if __name__ == "__main__":
    # Point this to your scp'ed summary file
    summary_file = "results/summary.txt"
    output_image = "results/plots/quadrant_analysis.png"
    
    if not os.path.exists("results/plots"):
        os.makedirs("results/plots")
        
    plot_quadrant_analysis(summary_file, output_image)
