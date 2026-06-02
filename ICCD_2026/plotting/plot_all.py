from __future__ import print_function
import csv
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATTERN = "results/csv/*.csv"
OUTPUT_PATH = "results/plots/all_schemes.png"
TITLE = "All scheme runs"

def load_csv(csv_path, temp_offset=0):
    x_values = []
    y_values = []
    with open(csv_path, "r") as handle:
        reader = csv.DictReader(line.replace("\x00", "") for line in handle)
        for row in reader:
            x_values.append(float(row["elapsed_s"]))
            y_values.append(float(row["temp_c"]) + temp_offset)
    return x_values, y_values

def main():
    csv_files = sorted(glob.glob(CSV_PATTERN))
    if not csv_files:
        print("No CSV files found.")
        return

    print("Found files:", csv_files)  # <-- will confirm loople.csv is found

    plt.figure(figsize=(10, 6))
    for csv_path in csv_files:
        label = os.path.splitext(os.path.basename(csv_path))[0]
        x_values, y_values = load_csv(csv_path, temp_offset=10)
        print(f"  {label}: first temp = {y_values[0]}")  # <-- confirms +10 is applied
        plt.plot(x_values, y_values, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (C)")
    plt.title(TITLE)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close()
    print("Saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()