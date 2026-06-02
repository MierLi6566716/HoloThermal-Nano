"""
predictive_cooldown.py

Instead of reacting AFTER the temperature crosses tmax, this scheme
measures the rate of temperature rise (dT/dt) and preemptively reduces
cores BEFORE the threshold is hit.

Logic:
  - Sample temperature every SAMPLE_INTERVAL_S seconds.
  - Compute dT/dt (degrees per second) over a short rolling window.
  - If projected temperature in LOOKAHEAD_S seconds exceeds guard_c,
    drop to mid or low cores now rather than waiting for the crossing.
  - If temperature is stable and well below guard, use all cores.

Why it might beat DVFS:
  DVFS is reactive — it only kicks in after the CPU is already hot, and
  frequency ramp-down/ramp-up adds latency. This scheme acts before the
  thermal budget is exceeded, keeping temperature inside the safe band
  more smoothly. On the Jetson Nano where the thermal mass is small and
  temperature rises fast under matrix workloads, a 2-3 second lookahead
  can prevent the spike that triggers throttling entirely.
"""

import time
import collections
from utils.worker import make_even_assignments, run_assignments
from utils.temp_utils import read_temp_c

SAMPLE_INTERVAL_S = 0.5
LOOKAHEAD_S       = 3.0   # project temperature this many seconds ahead
WINDOW_SIZE       = 6     # number of samples for dT/dt estimate


def _estimate_rate(history):
    """Estimate dT/dt in °C/s from a deque of (time, temp) pairs."""
    if len(history) < 2:
        return 0.0
    t0, T0 = history[0]
    t1, T1 = history[-1]
    dt = t1 - t0
    if dt <= 0:
        return 0.0
    return (T1 - T0) / dt


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

    history = collections.deque(maxlen=WINDOW_SIZE)
    last_sample = time.perf_counter() - SAMPLE_INTERVAL_S

    while remaining > 0:
        now = time.perf_counter()

        # Refresh temperature sample on schedule
        if now - last_sample >= SAMPLE_INTERVAL_S:
            history.append((now, read_temp_c(temp_path)))
            last_sample = now

        current_temp = history[-1][1] if history else read_temp_c(temp_path)
        rate         = _estimate_rate(history)           # °C/s
        projected    = current_temp + rate * LOOKAHEAD_S # where we'll be in LOOKAHEAD_S

        if projected >= guard_c:
            selected = [low_core]
        elif projected >= guard_c - margin_c:
            selected = mid_cores
        else:
            selected = high_cores

        wave = min(chunk_units * len(selected), remaining)
        assignments = make_even_assignments(wave, selected)
        run_assignments(assignments, stress_test_name, config, stagger_s=stagger_s)
        remaining -= wave