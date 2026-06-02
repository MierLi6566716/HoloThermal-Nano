
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


def set_thermal_trip_points(temp_c, indices=[0, 1, 4]):
    """
    Sets multiple thermal trip point temperatures for CPU-therm (Zone 1).
    Indices 0, 1 are active (fans), Index 4 is passive (throttling).
    """
    for i in indices:
        path = "/sys/class/thermal/thermal_zone1/trip_point_%d_temp" % i
        try:
            # Temperature is in millidegrees Celsius
            cmd = 'sudo sh -c "echo %d > %s"' % (int(temp_c * 1000), path)
            run_command(cmd)
        except Exception:
            pass

def prepare_nonbaseline_mode(use_maxn=True, use_jetson_clocks=True, set_performance_governor=True):
    if use_maxn:
        run_command("sudo nvpmodel -m 0")
    if use_jetson_clocks:
        run_command("sudo jetson_clocks")
    if set_performance_governor:
        set_governor("performance")
    
    # Move trip points 0, 1, 4 to 80C to allow headroom
    set_thermal_trip_points(80.0, indices=[0, 1, 4])

def enable_thermal_throttling():
    # Force throttling by moving the PASSIVE trip point (4) to 44C
    # Indices 0 and 1 are kept at 44C to keep fans on
    set_thermal_trip_points(44.0, indices=[0, 1, 4])
