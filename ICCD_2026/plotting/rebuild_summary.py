import pandas as pd
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
CSV_DIR = "results/csv"
SUMMARY_FILE = "results/summary.txt"
# ==========================================

def calculate_stats(csv_path):
    """Calculates summary statistics from a single CSV file."""
    df = pd.read_csv(csv_path)
    
    if df.empty:
        return None

    # Parse scheme and stress from filename (scheme__stress.csv)
    base_name = os.path.basename(csv_path).replace(".csv", "")
    if "__" in base_name:
        scheme, stress = base_name.split("__", 1)
    else:
        scheme = base_name
        stress = "synthetic_holography" # Default stress name

    # Temperature stats
    start_temp = df['temp_c'].iloc[0]
    peak_temp = df['temp_c'].max()
    avg_temp = df['temp_c'].mean()
    
    # Power and Energy stats
    avg_p_mw = 0
    total_energy_j = 0
    if 'power_mw' in df.columns:
        avg_p_mw = df['power_mw'].mean()
        # Energy (Joules) = Sum of (Power_W * delta_t)
        dt = df['elapsed_s'].diff().fillna(0)
        total_energy_j = (df['power_mw'] / 1000.0 * dt).sum()
    
    # Execution time
    elapsed_s = df['elapsed_s'].iloc[-1] - df['elapsed_s'].iloc[0]
    
    return {
        'scheme': scheme,
        'stress': stress,
        'start_temp_c': start_temp,
        'baseline_temp_c': start_temp,
        'peak_temp_c': peak_temp,
        'average_temp_c': avg_temp,
        'average_p_mw': avg_p_mw,
        'total_energy_j': total_energy_j,
        'elapsed_s': elapsed_s
    }

def main():
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {CSV_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files. Rebuilding summary...")
    
    all_stats = []
    for csv_path in sorted(csv_files):
        print(f"  Processing {os.path.basename(csv_path)}...")
        stats = calculate_stats(csv_path)
        if stats:
            all_stats.append(stats)

    # Format the lines for summary.txt
    line_format = (
        "scheme={scheme}, stress={stress}, start_temp_c={start_temp_c:.2f}, "
        "baseline_temp_c={baseline_temp_c:.2f}, peak_temp_c={peak_temp_c:.2f}, "
        "average_temp_c={average_temp_c:.2f}, average_p_mw={average_p_mw:.2f}, "
        "total_energy_j={total_energy_j:.2f}, elapsed_s={elapsed_s:.2f}\n"
    )

    with open(SUMMARY_FILE, 'w') as f:
        for stats in all_stats:
            f.write(line_format.format(**stats))

    print(f"\nSuccess! '{SUMMARY_FILE}' has been completely rebuilt.")

if __name__ == "__main__":
    main()
