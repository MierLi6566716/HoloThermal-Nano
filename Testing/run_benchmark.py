
from __future__ import print_function

import os
import time
import importlib

from utils import dvfs
from utils import temp_utils
from utils import logger as temp_logger
from plotting.plot_current import plot_csv

# =========================================================
# EDIT THIS BLOCK ONLY
# =========================================================
SCHEME_NAME = "baseline"
# baseline, sequential, concurrent, rotating_single, hybrid_parallel, cascading, rotating_bursts, predictive_cooldown, predictive_cooldown_v2
# predictive_headroom

STRESS_TEST_NAME = "iterative_physics"
# iterative_physics, physics_stress, dprc_stress, matrix_multiply, mlp_numpy

WORKLOAD_UNITS = 128
CYCLES = 1

# stress-test size knobs
MATRIX_SIZE = 1024
MLP_SAMPLES = 512
MLP_FEATURES = 128
MLP_HIDDEN = 128
MLP_OUTPUTS = 64

# pinning / core choices
ACTIVE_CORES = [0, 1, 2, 3]
SEQUENTIAL_CORE = 0
BASELINE_WORKERS = 1

# fine-grained allocation knobs
CHUNK_UNITS = 2
HYBRID_HIGH_CORES = [0, 1, 2, 3]
HYBRID_MID_CORES = [0, 1]
HYBRID_LOW_CORE = 0
TEMP_GUARD_C = 78.0
TEMP_MARGIN_C = 3.0
STAGGER_S = 0.0

# temperature / logging
THERMAL_PATH = os.environ.get("THERMAL_PATH", "/sys/class/thermal/thermal_zone1/temp")
LOG_INTERVAL_S = 0.25
BASELINE_TEMP_FILE = "baseline_start_temp.txt"
TEMP_MATCH_TOLERANCE_C = 0.5
WARMUP_CORE = 0

# stock-vs-custom policy
NON_BASELINE_USE_MAXN = True
NON_BASELINE_USE_JETSON_CLOCKS = True
NON_BASELINE_SET_PERFORMANCE_GOVERNOR = True
# =========================================================


def make_run_config():
    return {
        "scheme_name": SCHEME_NAME,
        "stress_test_name": STRESS_TEST_NAME,
        "workload_units": int(WORKLOAD_UNITS),
        "cycles": int(CYCLES),
        "matrix_size": int(MATRIX_SIZE),
        "mlp_samples": int(MLP_SAMPLES),
        "mlp_features": int(MLP_FEATURES),
        "mlp_hidden": int(MLP_HIDDEN),
        "mlp_outputs": int(MLP_OUTPUTS),
        "active_cores": list(ACTIVE_CORES),
        "sequential_core": int(SEQUENTIAL_CORE),
        "baseline_workers": int(BASELINE_WORKERS),
        "chunk_units": int(CHUNK_UNITS),
        "hybrid_high_cores": list(HYBRID_HIGH_CORES),
        "hybrid_mid_cores": list(HYBRID_MID_CORES),
        "hybrid_low_core": int(HYBRID_LOW_CORE),
        "temp_guard_c": float(TEMP_GUARD_C),
        "temp_margin_c": float(TEMP_MARGIN_C),
        "stagger_s": float(STAGGER_S),
        "thermal_path": THERMAL_PATH,
        "log_interval_s": float(LOG_INTERVAL_S),
        "baseline_temp_file": BASELINE_TEMP_FILE,
        "temp_match_tolerance_c": float(TEMP_MATCH_TOLERANCE_C),
        "warmup_core": int(WARMUP_CORE),
        "nonbaseline_use_maxn": bool(NON_BASELINE_USE_MAXN),
        "nonbaseline_use_jetson_clocks": bool(NON_BASELINE_USE_JETSON_CLOCKS),
        "nonbaseline_set_performance_governor": bool(NON_BASELINE_SET_PERFORMANCE_GOVERNOR),
    }


def ensure_results_dirs():
    for path in ["results_long", "results_long/csv", "results_long/plots"]:
        if not os.path.isdir(path):
            os.makedirs(path)


def prepare_start_temperature(config):
    initial_temp = temp_utils.read_temp_c(config["thermal_path"])

    if config["scheme_name"] == "baseline":
        temp_utils.save_baseline_start_temp(config["baseline_temp_file"], initial_temp)
        return initial_temp, initial_temp

    baseline_temp = temp_utils.load_baseline_start_temp(config["baseline_temp_file"])
    if baseline_temp is None:
        raise RuntimeError("Run baseline first so non-baseline schemes can match its starting temperature.")

    print("Matching baseline start temperature: %.2f C" % baseline_temp)
    matched_temp = temp_utils.match_target_temp(
        config["thermal_path"],
        baseline_temp,
        config["temp_match_tolerance_c"],
        config["warmup_core"],
    )
    return matched_temp, baseline_temp


def prepare_nonbaseline_mode(config):
    if config["scheme_name"] == "baseline":
        print("Baseline run: leaving Jetson defaults alone.")
        return

    print("Preparing non-baseline mode.")
    dvfs.prepare_nonbaseline_mode(
        use_maxn=config["nonbaseline_use_maxn"],
        use_jetson_clocks=config["nonbaseline_use_jetson_clocks"],
        set_performance_governor=config["nonbaseline_set_performance_governor"],
    )


def main():
    ensure_results_dirs()
    config = make_run_config()

    print("Scheme:", config["scheme_name"])
    print("Stress test:", config["stress_test_name"])

    start_temp_c, baseline_temp_c = prepare_start_temperature(config)
    prepare_nonbaseline_mode(config)

    scheme_module = importlib.import_module("schemes." + config["scheme_name"])

    stem = "%s__%s" % (config["scheme_name"], config["stress_test_name"])
    csv_path = os.path.join("results_long", "csv", stem + ".csv")
    plot_path = os.path.join("results_long", "plots", stem + ".png")

    stop_event, log_thread = temp_logger.start_temperature_logger(
        csv_path,
        config["thermal_path"],
        config["log_interval_s"],
    )

    wall_start = time.time()
    for cycle_index in range(config["cycles"]):
        print("Cycle %d / %d" % (cycle_index + 1, config["cycles"]))
        scheme_module.run(config["stress_test_name"], config)
    wall_end = time.time()

    stop_event.set()
    log_thread.join()

    peak_c, average_c = temp_utils.compute_csv_stats(csv_path)
    elapsed_s = wall_end - wall_start

    plot_csv(csv_path, plot_path, stem)

    with open(os.path.join("results_long", "summary.txt"), "a") as summary_file:
        summary_file.write(
            "scheme=%s, stress=%s, start_temp_c=%.2f, baseline_temp_c=%.2f, peak_temp_c=%.2f, average_temp_c=%.2f, elapsed_s=%.2f\n"
            % (
                config["scheme_name"],
                config["stress_test_name"],
                start_temp_c,
                baseline_temp_c,
                peak_c,
                average_c,
                elapsed_s,
            )
        )

    print("CSV:", csv_path)
    print("Plot:", plot_path)
    print("Peak temp: %.2f C" % peak_c)
    print("Average temp: %.2f C" % average_c)
    print("Elapsed: %.2f s" % elapsed_s)


if __name__ == "__main__":
    main()
