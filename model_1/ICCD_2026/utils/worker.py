
from __future__ import print_function

import importlib
import multiprocessing as mp
import os
import time


def make_even_assignments(total_units, slots):
    total_units = int(total_units)
    slot_count = len(slots)
    base_units = total_units // slot_count
    remainder = total_units % slot_count

    assignments = []
    for index, slot in enumerate(slots):
        units = base_units
        if index < remainder:
            units += 1
        assignments.append((slot, units))
    return assignments


def _worker_entry(core_id, stress_test_name, units, config, delay_s):
    if delay_s > 0.0:
        time.sleep(delay_s)

    if core_id is not None:
        try:
            os.sched_setaffinity(0, set([int(core_id)]))
        except Exception:
            pass

    stress_module = importlib.import_module("stress_tests." + stress_test_name)
    stress_module.run_units(int(units), config)


def run_assignments(assignments, stress_test_name, config, stagger_s=0.0):
    processes = []

    for index, assignment in enumerate(assignments):
        core_id, units = assignment
        if int(units) <= 0:
            continue

        process = mp.Process(
            target=_worker_entry,
            args=(core_id, stress_test_name, int(units), config, float(index) * float(stagger_s)),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
