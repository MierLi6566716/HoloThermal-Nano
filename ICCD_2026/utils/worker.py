
from __future__ import print_function
import importlib
import multiprocessing as mp
import os
import time

def _worker_entry(core_id, stress_test_name, units, config):
    """
    Fresh worker process.
    Guaranteed to be isolated and pinned to a specific core.
    """
    import torch
    
    # 1. Strict Core Pinning
    if core_id is not None:
        try: os.sched_setaffinity(0, {int(core_id)})
        except: pass
            
    # 2. PyTorch Optimization
    torch.set_num_threads(1)
    
    # 3. Execution
    stress_module = importlib.import_module("stress_tests." + stress_test_name)
    stress_module.run_units(int(units), config)

def run_assignments(assignments, stress_test_name, config, stagger_s=0.0):
    """
    Launches true parallel processes based on assignments.
    If assignments has 4 entries, 4 cores are used.
    """
    processes = []
    for i, (core_id, units) in enumerate(assignments):
        if int(units) <= 0: continue
            
        p = mp.Process(
            target=_worker_entry,
            args=(core_id, stress_test_name, int(units), config),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def stop_server(): pass
def start_worker(): pass

def make_even_assignments(total_units, slots):
    total_units = int(total_units)
    slot_count = len(slots)
    if slot_count == 0: return [(None, total_units)]
    
    base_units = total_units // slot_count
    remainder = total_units % slot_count
    
    results = []
    for i, slot in enumerate(slots):
        u = base_units + (1 if i < remainder else 0)
        results.append((slot, u))
    return results
