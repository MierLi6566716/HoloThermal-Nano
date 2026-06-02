
from __future__ import print_function

import csv
import threading
import time


from .power_utils import read_power_mw


def get_cpu_freq_mhz():
    """Reads the current frequency of CPU0 in MHz."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r") as f:
            return float(f.read().strip()) / 1000.0
    except:
        return 0.0

def _logger_loop(csv_path, temp_path, interval_s, stop_event):
    with open(csv_path, "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["elapsed_s", "temp_c", "power_mw", "freq_mhz"])

        start_time = time.time()

        while not stop_event.is_set():
            now = time.time()
            try:
                with open(temp_path, "r") as temp_file:
                    temp_c = float(temp_file.read().strip()) / 1000.0
            except:
                temp_c = 0.0
            
            power_mw = read_power_mw()
            freq_mhz = get_cpu_freq_mhz()
            writer.writerow([now - start_time, temp_c, power_mw, freq_mhz])
            handle.flush()
            stop_event.wait(interval_s)

        now = time.time()
        try:
            with open(temp_path, "r") as temp_file:
                temp_c = float(temp_file.read().strip()) / 1000.0
        except:
            temp_c = 0.0
        
        power_mw = read_power_mw()
        freq_mhz = get_cpu_freq_mhz()
        writer.writerow([now - start_time, temp_c, power_mw, freq_mhz])
        handle.flush()


def start_temperature_logger(csv_path, temp_path, interval_s):
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_logger_loop,
        args=(csv_path, temp_path, interval_s, stop_event),
    )
    thread.daemon = True
    thread.start()
    return stop_event, thread
