
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

def plot_three_dimensions(csv_path, output_path):
    """
    Standalone plotter for ICCD 2026 Thermal Benchmark results.
    Plots: Temperature, Power, and Cumulative Energy.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    # Load Data
    df = pd.read_csv(csv_path)
    stem = os.path.basename(csv_path).replace(".csv", "")

    # Check Columns
    has_power = "power_mw" in df.columns
    num_plots = 3 if has_power else 1
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes]

    # Panel 1: Temperature
    axes[0].plot(df["elapsed_s"], df["temp_c"], color='#d62728', linewidth=2, label='Temperature')
    axes[0].set_ylabel("Temp (°C)", fontsize=12, fontweight='bold')
    axes[0].set_title(f"Multi-Dimensional Analysis: {stem}", fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper right')
    
    peak_temp = df["temp_c"].max()
    axes[0].axhline(y=peak_temp, color='red', linestyle=':', alpha=0.5)
    axes[0].text(df["elapsed_s"].iloc[0], peak_temp + 0.5, f"Peak: {peak_temp:.1f}°C", color='red', fontweight='bold')

    if has_power:
        # Panel 2: Power Consumption
        axes[1].plot(df["elapsed_s"], df["power_mw"], color='#1f77b4', linewidth=2, label='Power')
        axes[1].set_ylabel("Power (mW)", fontsize=12, fontweight='bold')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend(loc='upper right')
        
        avg_p = df["power_mw"].mean()
        axes[1].axhline(y=avg_p, color='blue', linestyle=':', alpha=0.5)
        axes[1].text(df["elapsed_s"].iloc[0], avg_p + 100, f"Avg: {avg_p:.0f} mW", color='blue', fontweight='bold')

        # Panel 3: Cumulative Energy (Work Efficiency)
        # Energy (Joules) = Sum of (Power_W * delta_t)
        dt = df["elapsed_s"].diff().fillna(0)
        energy_j = (df["power_mw"] / 1000.0 * dt).cumsum()
        
        axes[2].plot(df["elapsed_s"], energy_j, color='#2ca02c', linewidth=2, label='Energy Consumption')
        axes[2].set_ylabel("Energy (Joules)", fontsize=12, fontweight='bold')
        axes[2].set_xlabel("Elapsed Time (seconds)", fontsize=12)
        axes[2].grid(True, linestyle='--', alpha=0.6)
        axes[2].legend(loc='upper right')
        
        total_j = energy_j.iloc[-1]
        axes[2].text(df["elapsed_s"].iloc[-1], total_j, f" Total: {total_j:.1f} J ", 
                     ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Analysis saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_standalone.py <input_csv> [output_png]")
    else:
        csv_in = sys.argv[1]
        png_out = sys.argv[2] if len(sys.argv) > 2 else csv_in.replace(".csv", "_analysis.png")
        plot_three_dimensions(csv_in, png_out)
