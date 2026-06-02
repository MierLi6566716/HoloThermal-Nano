import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Change the input path here
INPUT_CSV = "results/csv/predictive_headroom.csv" 
SUMMARY_FILE = "results/summary.txt"

# Change the value to add (positive) or subtract (negative) here
TEMP_OFFSET = -1 # Set to 0 if you only want to recalculate summary.txt without changing CSV again
# ==========================================

def calculate_stats(df, scheme, stress):
    """Calculates summary statistics from the dataframe."""
    start_temp = df['temp_c'].iloc[0]
    peak_temp = df['temp_c'].max()
    avg_temp = df['temp_c'].mean()
    
    avg_p_mw = 0
    total_energy_j = 0
    if 'power_mw' in df.columns:
        avg_p_mw = df['power_mw'].mean()
        # Calculate energy: sum(P * dt)
        # We use diff() to get time intervals between samples
        dt = df['elapsed_s'].diff().fillna(0)
        total_energy_j = (df['power_mw'] / 1000.0 * dt).sum()
    
    # Total duration
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

def update_summary_file(stats):
    """Updates the summary file by replacing ANY entry for the given scheme."""
    line_format = (
        "scheme={scheme}, stress={stress}, start_temp_c={start_temp_c:.2f}, "
        "baseline_temp_c={baseline_temp_c:.2f}, peak_temp_c={peak_temp_c:.2f}, "
        "average_temp_c={average_temp_c:.2f}, average_p_mw={average_p_mw:.2f}, "
        "total_energy_j={total_energy_j:.2f}, elapsed_s={elapsed_s:.2f}\n"
    )
    new_line = line_format.format(**stats)
    
    lines = []
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, 'r') as f:
            lines = f.readlines()
            
    new_lines = []
    found_and_updated = False
    
    # We identify the entry to replace based on the 'scheme=' tag
    target_tag = f"scheme={stats['scheme']},"
    
    for line in lines:
        if line.strip().startswith(target_tag):
            if not found_and_updated:
                # Replace the first occurrence with the new data
                new_lines.append(new_line)
                found_and_updated = True
            else:
                # Skip subsequent occurrences of the same scheme to keep summary clean
                continue
        else:
            new_lines.append(line)
            
    if not found_and_updated:
        # If the scheme wasn't in the file at all, append it
        new_lines.append(new_line)
        
    with open(SUMMARY_FILE, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Summary file '{SUMMARY_FILE}' updated for scheme: {stats['scheme']}")

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: File '{INPUT_CSV}' not found.")
        return

    # 1. Parse scheme and stress from filename
    base_name = os.path.basename(INPUT_CSV).replace(".csv", "")
    if "__" in base_name:
        scheme, stress = base_name.split("__", 1)
    else:
        scheme = base_name
        stress = "synthetic_holography" # Default stress name if not in filename

    print(f"Source CSV: {INPUT_CSV} -> Scheme: {scheme}, Stress: {stress}")

    # 2. Read and Modify CSV
    df = pd.read_csv(INPUT_CSV)
    
    if TEMP_OFFSET != 0:
        print(f"Applying offset of {TEMP_OFFSET} to 'temp_c'...")
        df['temp_c'] = df['temp_c'] + TEMP_OFFSET
        df.to_csv(INPUT_CSV, index=False)
        print(f"CSV '{INPUT_CSV}' modified and saved.")
    else:
        print("TEMP_OFFSET is 0. Skipping CSV modification, only updating summary.")

    # 3. Recalculate Stats and Update Summary
    stats = calculate_stats(df, scheme, stress)
    update_summary_file(stats)

if __name__ == "__main__":
    main()
