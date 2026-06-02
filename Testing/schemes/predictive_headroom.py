"""
predictive_headroom.py

A temperature + temperature-slope scheduler for pinned multiprocessing runs.

Idea:
- Stay on all cores while there is thermal headroom.
- Step down to fewer xactive cores as temperature rises or begins rising quickly.
- Step back up only after the board has cooled AND the temperature slope is flat.

This is meant to be more responsive than a pure threshold-only scheduler because
it reacts not just to "how hot am I now?" but also to "how fast am I heating up?"

Mode meanings:
    4 -> use 4 pinned cores
    3 -> use 3 pinned cores
    2 -> use 2 pinned cores
    1 -> use 1 pinned core

The preferred single-core fallback is config["hybrid_low_core"] if provided.

Optional config keys:
    ph_t4_to_t3       temp to drop from 4 cores to 3 cores
    ph_t3_to_t2       temp to drop from 3 cores to 2 cores
    ph_t2_to_t1       temp to drop from 2 cores to 1 core

    ph_t1_to_t2       temp to allow rising from 1 core back to 2 cores
    ph_t2_to_t3       temp to allow rising from 2 cores back to 3 cores
    ph_t3_to_t4       temp to allow rising from 3 cores back to 4 cores

    ph_slope_down_one   if slope exceeds this, step down one level
    ph_slope_down_fast  if slope exceeds this, step down two levels
    ph_slope_up_allow   only step up when slope is at or below this

    ph_slope_window_s   rolling window used to estimate slope
    ph_min_up_dwell_s   minimum seconds to wait before stepping upward again
    ph_wave_scale       scales wave size down for faster control response

Suggested starting values for your current temperature range:
    ph_t4_to_t3 = 37.5
    ph_t3_to_t2 = 38.8
    ph_t2_to_t1 = 40.0

    ph_t1_to_t2 = 39.0
    ph_t2_to_t3 = 38.0
    ph_t3_to_t4 = 37.2

    ph_slope_down_one = 0.12
    ph_slope_down_fast = 0.20
    ph_slope_up_allow = 0.02

    ph_slope_window_s = 3.0
    ph_min_up_dwell_s = 2.5
    ph_wave_scale = 0.5
"""

import time
from collections import deque

from utils.worker import make_even_assignments, run_assignments
from utils.temp_utils import read_temp_c


DEFAULTS = {
    "ph_t4_to_t3": 37.5,
    "ph_t3_to_t2": 38.8,
    "ph_t2_to_t1": 40.0,
    "ph_t1_to_t2": 39.0,
    "ph_t2_to_t3": 38.0,
    "ph_t3_to_t4": 37.2,
    "ph_slope_down_one": 0.12,
    "ph_slope_down_fast": 0.20,
    "ph_slope_up_allow": 0.02,
    "ph_slope_window_s": 3.0,
    "ph_min_up_dwell_s": 2.5,
    "ph_wave_scale": 0.5,
}


def _cfg_float(config, key):
    return float(config.get(key, DEFAULTS[key]))


def _ordered_cores(all_cores, preferred_single_core):
    """
    Put the preferred fallback core first, then preserve the rest.
    Example:
        all_cores=[0,1,2,3], preferred_single_core=2
        -> [2,0,1,3]
    """
    ordered = []
    seen = set()

    if preferred_single_core in all_cores:
        ordered.append(preferred_single_core)
        seen.add(preferred_single_core)

    for core in all_cores:
        if core not in seen:
            ordered.append(core)
            seen.add(core)

    return ordered


def _pick_cores(ordered_cores, count):
    count = max(1, min(int(count), len(ordered_cores)))
    return ordered_cores[:count]


def _record_temp(history, now_s, temp_c, window_s):
    history.append((now_s, temp_c))
    cutoff = now_s - window_s
    while len(history) > 1 and history[0][0] < cutoff:
        history.popleft()


def _temp_slope_c_per_s(history):
    """
    Estimate slope over the rolling window:
        slope = (newest_temp - oldest_temp) / (newest_time - oldest_time)
    """
    if len(history) < 2:
        return 0.0

    old_t, old_temp = history[0]
    new_t, new_temp = history[-1]
    dt = new_t - old_t
    if dt <= 1e-9:
        return 0.0
    return (new_temp - old_temp) / dt


def _decide_mode(current_mode, max_mode, temp_c, slope_cps, thresholds):
    """
    Decide desired mode before dwell filtering.

    Lower mode number = more conservative:
        4 = hottest / fastest
        1 = coolest / slowest
    """
    target = current_mode

    # Absolute temperature thresholds take first priority.
    if temp_c >= thresholds["t2_to_t1"]:
        target = 1
    elif max_mode >= 2 and temp_c >= thresholds["t3_to_t2"]:
        target = min(target, 2)
    elif max_mode >= 3 and temp_c >= thresholds["t4_to_t3"]:
        target = min(target, 3)

    # Predictive downshifts based on slope.
    if slope_cps >= thresholds["slope_down_fast"]:
        target = min(target, max(1, current_mode - 2))
    elif slope_cps >= thresholds["slope_down_one"]:
        target = min(target, max(1, current_mode - 1))

    # Conservative step-up logic:
    # only one level at a time, only if slope is flat enough.
    if slope_cps <= thresholds["slope_up_allow"]:
        if current_mode == 1 and max_mode >= 2 and temp_c <= thresholds["t1_to_t2"]:
            target = max(target, 2)
        elif current_mode == 2 and max_mode >= 3 and temp_c <= thresholds["t2_to_t3"]:
            target = max(target, 3)
        elif current_mode == 3 and max_mode >= 4 and temp_c <= thresholds["t3_to_t4"]:
            target = max(target, 4)

    return max(1, min(target, max_mode))


def run(stress_test_name, config):
    remaining = int(config["workload_units"])
    chunk_units = max(1, int(config["chunk_units"]))
    all_cores = list(config["active_cores"])
    temp_path = config["thermal_path"]
    stagger_s = float(config.get("stagger_s", 0.0))

    preferred_single_core = int(config.get("hybrid_low_core", all_cores[0]))
    ordered_cores = _ordered_cores(all_cores, preferred_single_core)
    max_mode = len(ordered_cores)

    thresholds = {
        "t4_to_t3": _cfg_float(config, "ph_t4_to_t3"),
        "t3_to_t2": _cfg_float(config, "ph_t3_to_t2"),
        "t2_to_t1": _cfg_float(config, "ph_t2_to_t1"),
        "t1_to_t2": _cfg_float(config, "ph_t1_to_t2"),
        "t2_to_t3": _cfg_float(config, "ph_t2_to_t3"),
        "t3_to_t4": _cfg_float(config, "ph_t3_to_t4"),
        "slope_down_one": _cfg_float(config, "ph_slope_down_one"),
        "slope_down_fast": _cfg_float(config, "ph_slope_down_fast"),
        "slope_up_allow": _cfg_float(config, "ph_slope_up_allow"),
    }

    slope_window_s = _cfg_float(config, "ph_slope_window_s")
    min_up_dwell_s = _cfg_float(config, "ph_min_up_dwell_s")
    wave_scale = max(0.1, _cfg_float(config, "ph_wave_scale"))

    current_mode = max_mode
    last_mode_change_s = time.perf_counter()
    history = deque()

    # Prime the history with the initial reading.
    now_s = time.perf_counter()
    temp_c = read_temp_c(temp_path)
    _record_temp(history, now_s, temp_c, slope_window_s)

    print(
        "[predictive_headroom] start: temp=%.2fC mode=%d cores=%s preferred_single=%d"
        % (temp_c, current_mode, ordered_cores, preferred_single_core)
    )
    print(
        "[predictive_headroom] thresholds: "
        "4->3=%.2f 3->2=%.2f 2->1=%.2f | "
        "1->2=%.2f 2->3=%.2f 3->4=%.2f | "
        "slope_down_one=%.3f slope_down_fast=%.3f slope_up_allow=%.3f"
        % (
            thresholds["t4_to_t3"],
            thresholds["t3_to_t2"],
            thresholds["t2_to_t1"],
            thresholds["t1_to_t2"],
            thresholds["t2_to_t3"],
            thresholds["t3_to_t4"],
            thresholds["slope_down_one"],
            thresholds["slope_down_fast"],
            thresholds["slope_up_allow"],
        )
    )

    while remaining > 0:
        # Read current temp before dispatching the next wave.
        now_s = time.perf_counter()
        temp_c = read_temp_c(temp_path)
        _record_temp(history, now_s, temp_c, slope_window_s)
        slope_cps = _temp_slope_c_per_s(history)

        desired_mode = _decide_mode(
            current_mode=current_mode,
            max_mode=max_mode,
            temp_c=temp_c,
            slope_cps=slope_cps,
            thresholds=thresholds,
        )

        # Allow immediate downshifts.
        # Gate only upward transitions with a minimum dwell time.
        if desired_mode > current_mode:
            if (now_s - last_mode_change_s) < min_up_dwell_s:
                desired_mode = current_mode

        if desired_mode != current_mode:
            print(
                "[predictive_headroom] mode %d -> %d | temp=%.2fC slope=%.3fC/s remaining=%d"
                % (current_mode, desired_mode, temp_c, slope_cps, remaining)
            )
            current_mode = desired_mode
            last_mode_change_s = now_s

        active_cores = _pick_cores(ordered_cores, current_mode)

        # Smaller waves = more responsive control.
        # Keep at least one chunk per active core.
        wave_units = max(
            len(active_cores),
            int(round(chunk_units * len(active_cores) * wave_scale)),
        )
        wave = min(wave_units, remaining)

        assignments = make_even_assignments(wave, active_cores)
        run_assignments(assignments, stress_test_name, config, stagger_s=stagger_s)
        remaining -= wave

        # Post-wave sample helps the slope estimator stay fresh.
        after_s = time.perf_counter()
        after_temp_c = read_temp_c(temp_path)
        _record_temp(history, after_s, after_temp_c, slope_window_s)

    final_temp_c = read_temp_c(temp_path)
    print(
        "[predictive_headroom] done: final_temp=%.2fC remaining=%d"
        % (final_temp_c, remaining)
    )