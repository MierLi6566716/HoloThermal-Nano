"""
rotating_burst.py

Runs a fixed burst of work on one core at a time, then rotates to the
next core. No temperature sensing needed — the rotation itself acts as
a passive thermal spreader.

  core 0 does BURST_UNITS of work
  core 1 does BURST_UNITS of work
  core 2 does BURST_UNITS of work
  core 3 does BURST_UNITS of work
  repeat until done

Why it might beat DVFS:
  The Jetson Nano's 4 cores share a heatsink and thermal mass. Sequential
  single-core runs are cold but very slow. Fully concurrent runs are fast
  but hot. Rotating bursts hit a middle ground: each core gets time to
  radiate heat while idle, and the thermal load never concentrates long
  enough on any single core to trigger kernel throttling. Unlike DVFS,
  clocks stay fixed at max — we trade spatial heat spreading for time.

Tune BURST_UNITS (= chunk_units in config) smaller for smoother rotation,
larger for more throughput per switch.
"""

from utils.worker import make_even_assignments, run_assignments
from utils.temp_utils import read_temp_c


def run(stress_test_name, config):
    remaining   = int(config["workload_units"])
    burst_units = int(config["chunk_units"])
    cores       = list(config["active_cores"])
    stagger_s   = float(config.get("stagger_s", 0.0))

    core_idx = 0
    while remaining > 0:
        core = cores[core_idx % len(cores)]
        wave = min(burst_units, remaining)
        assignments = make_even_assignments(wave, [core])
        run_assignments(assignments, stress_test_name, config, stagger_s=stagger_s)
        remaining -= wave
        core_idx += 1