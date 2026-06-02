"""
Instead of a hard binary switch (all cores vs 1 core), this scheme uses
THREE temperature bands to cascade down gradually:

  temp < guard - 2*margin  -> use ALL cores  (high_cores)
  temp < guard - margin    -> use MID cores  (mid_cores)
  temp < guard             -> use LOW cores  (hybrid_low_core only)
  temp >= guard            -> pause, wait for cooldown before resuming

Why it might beat DVFS:
  DVFS reduces frequency uniformly across all cores when hot.
  This scheme instead reduces the NUMBER of active cores while keeping
  the remaining cores at FULL frequency. On the Jetson Nano (in-order
  cores, shared L2), fewer cores at full speed generates less heat per
  unit of work than all cores at reduced speed, and avoids the latency
  penalty of frequency ramp-up when the CPU cools.
"""

from utils.worker import make_even_assignments, run_assignments
from utils.temp_utils import read_temp_c
import time

COOLDOWN_POLL_S = 0.5


def run(stress_test_name, config):
    remaining   = int(config["workload_units"])
    chunk_units = int(config["chunk_units"])
    high_cores  = list(config["hybrid_high_cores"])
    mid_cores   = list(config["hybrid_mid_cores"])
    low_core    = int(config["hybrid_low_core"])
    guard_c     = float(config["temp_guard_c"])
    margin_c    = float(config["temp_margin_c"])
    stagger_s   = float(config.get("stagger_s", 0.0))
    temp_path   = config["thermal_path"]

    while remaining > 0:
        temp = read_temp_c(temp_path)

        if temp >= guard_c:
            # Too hot — pause entirely until we drop one margin below guard
            print("[cascading_cores] Pausing (%.1f C >= %.1f C)..." % (temp, guard_c))
            while read_temp_c(temp_path) >= guard_c - margin_c:
                time.sleep(COOLDOWN_POLL_S)
            continue

        if temp < guard_c - 2 * margin_c:
            selected = high_cores
        elif temp < guard_c - margin_c:
            selected = mid_cores
        else:
            selected = [low_core]

        wave = min(chunk_units * len(selected), remaining)
        assignments = make_even_assignments(wave, selected)
        run_assignments(assignments, stress_test_name, config, stagger_s=stagger_s)
        remaining -= wave