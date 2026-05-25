from __future__ import print_function
import csv
import glob
import os

CSV_PATTERN = "model_1/Testing/results/csv/*.csv"
OUTPUT_PATH = "model_1/Testing/results/plots/stats_summary.txt"


def load_csv(csv_path):
    times = []
    temps = []
    with open(csv_path, "r") as handle:
        reader = csv.DictReader(line.replace("\x00", "") for line in handle)
        for row in reader:
            times.append(float(row["elapsed_s"]))
            temps.append(float(row["temp_c"]))
    return times, temps


def main():
    csv_files = sorted(glob.glob(CSV_PATTERN))
    if not csv_files:
        print("No CSV files found.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    lines = []
    lines.append("=" * 50)
    lines.append("THERMAL RUN STATISTICS SUMMARY")
    lines.append("=" * 50)

    for csv_path in csv_files:
        label = os.path.splitext(os.path.basename(csv_path))[0]
        times, temps = load_csv(csv_path)

        avg_temp = sum(temps) / len(temps)
        peak_temp = max(temps)
        run_time = max(times) - min(times)

        lines.append("")
        lines.append(f"Scheme:           {label}")
        lines.append(f"Average Temp:     {avg_temp:.2f} C")
        lines.append(f"Peak Temp:        {peak_temp:.2f} C")
        lines.append(f"Run Time:         {run_time:.2f} s")
        lines.append("-" * 50)

    output = "\n".join(lines)
    print(output)

    with open(OUTPUT_PATH, "w") as f:
        f.write(output)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()