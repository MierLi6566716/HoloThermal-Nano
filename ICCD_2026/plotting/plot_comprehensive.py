
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_comprehensive(csv_path, output_path, title_stem):
    # Load the data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check available data
    has_power = "power_mw" in df.columns
    has_freq = "freq_mhz" in df.columns
    
    # Setup subplots
    plot_list = ["temp"]
    if has_power: plot_list.append("power")
    if has_freq: plot_list.append("freq")
    if has_power: plot_list.append("energy")
    
    num_plots = len(plot_list)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3.5 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes]

    ax_idx = 0

    # 1. Temperature Plot
    axes[ax_idx].plot(df["elapsed_s"], df["temp_c"], color='tab:red', linewidth=2)
    axes[ax_idx].set_ylabel("Temp (°C)", fontsize=12, fontweight='bold')
    axes[ax_idx].set_title(f"Thermal & Performance Analysis: {title_stem}", fontsize=14)
    axes[ax_idx].grid(True, linestyle='--', alpha=0.7)
    peak_temp = df["temp_c"].max()
    axes[ax_idx].axhline(y=peak_temp, color='red', linestyle=':', alpha=0.5)
    axes[ax_idx].text(df["elapsed_s"].iloc[0], peak_temp + 0.5, f"Peak: {peak_temp:.1f}°C", color='red')
    ax_idx += 1

    # 2. Frequency Plot
    if has_freq:
        axes[ax_idx].plot(df["elapsed_s"], df["freq_mhz"], color='tab:orange', linewidth=2)
        axes[ax_idx].set_ylabel("Freq (MHz)", fontsize=12, fontweight='bold')
        axes[ax_idx].grid(True, linestyle='--', alpha=0.7)
        avg_freq = df["freq_mhz"].mean()
        axes[ax_idx].axhline(y=avg_freq, color='orange', linestyle=':', alpha=0.5)
        axes[ax_idx].text(df["elapsed_s"].iloc[0], avg_freq + 50, f"Avg: {avg_freq:.0f}MHz", color='orange')
        ax_idx += 1

    # 3. Power Plot
    if has_power:
        axes[ax_idx].plot(df["elapsed_s"], df["power_mw"], color='tab:blue', linewidth=2)
        axes[ax_idx].set_ylabel("Power (mW)", fontsize=12, fontweight='bold')
        axes[ax_idx].grid(True, linestyle='--', alpha=0.7)
        avg_p = df["power_mw"].mean()
        axes[ax_idx].axhline(y=avg_p, color='blue', linestyle=':', alpha=0.5)
        axes[ax_idx].text(df["elapsed_s"].iloc[0], avg_p + 100, f"Avg: {avg_p:.0f}mW", color='blue')
        ax_idx += 1

    # 4. Energy Accumulation
    if has_power:
        energy_j = (df["power_mw"] / 1000.0 * (df["elapsed_s"].diff().fillna(0))).cumsum()
        axes[ax_idx].plot(df["elapsed_s"], energy_j, color='tab:green', linewidth=2)
        axes[ax_idx].set_ylabel("Energy (J)", fontsize=12, fontweight='bold')
        axes[ax_idx].set_xlabel("Time (seconds)", fontsize=12)
        axes[ax_idx].grid(True, linestyle='--', alpha=0.7)
        ax_idx += 1
    else:
        axes[ax_idx-1].set_xlabel("Time (seconds)", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Comprehensive plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 plot_comprehensive.py <csv_path> <output_path> <title>")
    else:
        plot_comprehensive(sys.argv[1], sys.argv[2], sys.argv[3])
