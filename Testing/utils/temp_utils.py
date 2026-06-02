
from __future__ import print_function

import csv
import os
import time


def read_temp_c(temp_path):
    with open(temp_path, "r") as handle:
        return float(handle.read().strip()) / 1000.0


def save_baseline_start_temp(path, temp_c):
    with open(path, "w") as handle:
        handle.write(str(float(temp_c)))


def load_baseline_start_temp(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as handle:
        return float(handle.read().strip())


def warm_up_briefly(core_id, seconds=0.25):
    try:
        old_affinity = os.sched_getaffinity(0)
    except Exception:
        old_affinity = None

    if old_affinity is not None:
        try:
            os.sched_setaffinity(0, set([int(core_id)]))
        except Exception:
            pass

    end_time = time.time() + float(seconds)
    value = 1
    while time.time() < end_time:
        value = (value * 1103515245 + 12345) & 2147483647

    if old_affinity is not None:
        try:
            os.sched_setaffinity(0, old_affinity)
        except Exception:
            pass


def match_target_temp(temp_path, target_c, tolerance_c, warmup_core):
    while True:
        current_c = read_temp_c(temp_path)

        if abs(current_c - target_c) <= tolerance_c:
            return current_c

        if current_c > (target_c + tolerance_c):
            print("Cooling down: current=%.2f C target=%.2f C" % (current_c, target_c))
            time.sleep(1.0)
        else:
            print("Warming up: current=%.2f C target=%.2f C" % (current_c, target_c))
            warm_up_briefly(warmup_core)


def compute_csv_stats(csv_path):
    temps = []
    with open(csv_path, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            temps.append(float(row["temp_c"]))

    peak_c = max(temps)
    average_c = sum(temps) / float(len(temps))
    return peak_c, average_c
