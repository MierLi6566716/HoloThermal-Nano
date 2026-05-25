
from utils.worker import run_assignments


def run(stress_test_name, config):
    remaining = int(config["workload_units"])
    chunk_units = int(config["chunk_units"])
    cores = list(config["active_cores"])
    core_index = 0

    while remaining > 0:
        units = chunk_units
        if units > remaining:
            units = remaining
        assignments = [(int(cores[core_index]), int(units))]
        run_assignments(assignments, stress_test_name, config, stagger_s=0.0)
        remaining -= units
        core_index = (core_index + 1) % len(cores)
