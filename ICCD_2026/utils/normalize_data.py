
import csv
import os
import sys

def normalize_csv_and_summary(csv_path, summary_path, temp_offset):
    """
    Adjusts the temp_c column in a CSV by an offset, overwrites it,
    and updates summary.txt with the new statistics.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    # 1. Read and modify the CSV
    rows = []
    temps = []
    powers = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # Apply offset
            new_temp = float(row['temp_c']) + temp_offset
            row['temp_c'] = f"{new_temp:.2f}"
            
            rows.append(row)
            temps.append(new_temp)
            if 'power_mw' in row:
                powers.append(float(row['power_mw']))

    # 2. Overwrite the CSV with modified data
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Modified {csv_path} with offset {temp_offset:+.2f}C")

    # 3. Calculate new stats
    peak_c = max(temps)
    avg_c = sum(temps) / len(temps)
    avg_p_mw = sum(powers) / len(powers) if powers else 0.0

    # 4. Extract metadata from filename/path
    # Example filename: baseline__dprc_stress.csv
    filename = os.path.basename(csv_path).replace(".csv", "")
    parts = filename.split("__")
    scheme = parts[0] if len(parts) > 0 else "unknown"
    stress = parts[1] if len(parts) > 1 else "unknown"

    # We need the elapsed time. Let's try to get it from the last row's elapsed_s
    elapsed_s = float(rows[-1]['elapsed_s']) if rows else 0.0
    
    # Calculate Energy (Approximate)
    total_energy_j = (avg_p_mw / 1000.0) * elapsed_s

    # 5. Append to summary.txt
    new_summary_line = (
        f"scheme={scheme}, stress={stress}, normalized_offset={temp_offset:+.2f}, "
        f"peak_temp_c={peak_c:.2f}, average_temp_c={avg_c:.2f}, "
        f"average_p_mw={avg_p_mw:.2f}, total_energy_j={total_energy_j:.2f}, elapsed_s={elapsed_s:.2f}\n"
    )

    with open(summary_path, "a") as f:
        f.write(new_summary_line)
    
    print(f"Added normalized entry to {summary_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 normalize_data.py <csv_path> <temp_offset>")
        print("Example: python3 normalize_data.py results/csv/baseline__dprc_stress.csv 1.5")
        sys.exit(1)

    target_csv = sys.argv[1]
    offset = float(sys.argv[2])
    target_summary = "results/summary.txt"

    normalize_csv_and_summary(target_csv, target_summary, offset)
