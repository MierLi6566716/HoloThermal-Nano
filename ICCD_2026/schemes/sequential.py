
from utils.worker import run_assignments


def run(stress_test_name, config):
    assignments = [(int(config["sequential_core"]), int(config["workload_units"]))]
    run_assignments(assignments, stress_test_name, config, stagger_s=0.0)
