
from __future__ import print_function

import glob
import os


def run_command(command):
    print(command)
    return os.system(command)


def set_governor(governor_name):
    governor_files = glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor")
    for path in governor_files:
        try:
            with open(path, "w") as handle:
                handle.write(governor_name)
        except Exception:
            pass


def prepare_nonbaseline_mode(use_maxn=True, use_jetson_clocks=True, set_performance_governor=True):
    if use_maxn:
        run_command("sudo nvpmodel -m 0")
    if use_jetson_clocks:
        run_command("sudo jetson_clocks")
    if set_performance_governor:
        set_governor("performance")
