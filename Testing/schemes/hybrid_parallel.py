
from utils.worker import make_even_assignments, run_assignments
from utils.temp_utils import read_temp_c


def run(stress_test_name, config):
    remaining = int(config["workload_units"])
    chunk_units = int(config["chunk_units"])
    high_cores = list(config["hybrid_high_cores"])
    mid_cores = list(config["hybrid_mid_cores"])
    low_core = int(config["hybrid_low_core"])
    guard_c = float(config["temp_guard_c"])
    margin_c = float(config["temp_margin_c"])
    temp_path = config["thermal_path"]

    while remaining > 0:
        current_temp_c = read_temp_c(temp_path)

        if current_temp_c < (guard_c - margin_c):
            selected_cores = high_cores
        elif current_temp_c < guard_c:
            selected_cores = mid_cores
        else:
            selected_cores = [low_core]

        wave_units = chunk_units * len(selected_cores)
        if wave_units > remaining:
            wave_units = remaining

        assignments = make_even_assignments(int(wave_units), list(selected_cores))
        run_assignments(assignments, stress_test_name, config, stagger_s=float(config.get("stagger_s", 0.0)))
        remaining -= wave_units
