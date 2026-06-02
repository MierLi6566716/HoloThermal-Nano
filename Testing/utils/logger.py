
from __future__ import print_function

import csv
import threading
import time


def _logger_loop(csv_path, temp_path, interval_s, stop_event):
    with open(csv_path, "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["elapsed_s", "temp_c"])

        start_time = time.time()

        while not stop_event.is_set():
            now = time.time()
            with open(temp_path, "r") as temp_file:
                temp_c = float(temp_file.read().strip()) / 1000.0
            writer.writerow([now - start_time, temp_c])
            handle.flush()
            stop_event.wait(interval_s)

        now = time.time()
        with open(temp_path, "r") as temp_file:
            temp_c = float(temp_file.read().strip()) / 1000.0
        writer.writerow([now - start_time, temp_c])
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
