
from utils.worker import make_even_assignments, run_assignments


def run(stress_test_name, config):
    assignments = make_even_assignments(int(config["workload_units"]), list(config["active_cores"]))
    run_assignments(assignments, stress_test_name, config, stagger_s=float(config.get("stagger_s", 0.0)))
