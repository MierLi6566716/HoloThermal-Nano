
from utils.worker import make_even_assignments, run_assignments


def run(stress_test_name, config):
    slots = [None] * int(config["baseline_workers"])
    assignments = make_even_assignments(int(config["workload_units"]), slots)
    run_assignments(assignments, stress_test_name, config, stagger_s=0.0)
