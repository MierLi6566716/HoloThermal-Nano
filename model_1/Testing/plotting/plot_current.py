
from __future__ import print_function

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_csv(csv_path, output_path, title_text):
    x_values = []
    y_values = []

    with open(csv_path, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_values.append(float(row["elapsed_s"]))
            y_values.append(float(row["temp_c"]))

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (C)")
    plt.title(title_text)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
