
from utils.worker import make_even_assignments, run_assignments

def run(stress_test_name, config):
    """
    Concurrent Scheme: Multi-Process Data Parallelism.
    Spawns 1 process per core to maximize CPU utilization and thermal stress.
    This is the "strongest concurrent" version.
    """
    total_units = int(config["workload_units"])
    active_cores = list(config["active_cores"])
    
    print(f"\n[Concurrent] Dispatching {total_units} units across cores {active_cores}")
    
    # Divide units among cores and spawn separate processes
    assignments = make_even_assignments(total_units, active_cores)
    run_assignments(assignments, stress_test_name, config, stagger_s=0.0)

    print(f"[Concurrent] Workload completed.")
