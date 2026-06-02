
# Jetson Nano thermal scheduling benchmark

This project is written to stay compatible with Python 3.6.9 on Jetson Nano.

## Workflow

1. Edit the config block at the top of `run_benchmark.py`.
2. Run **baseline first** from a fresh boot:
   ```bash
   python3 run_benchmark.py
   ```
3. Change `SCHEME_NAME` and rerun for the next scheme.
4. The non-baseline schemes will wait until the board temperature matches the saved baseline start temperature.
5. `run_benchmark.py` automatically:
   - logs temperature from `THERMAL_PATH`
   - runs the scheme on the stress test
   - saves a CSV
   - saves a plot for that run
   - appends peak/average/elapsed info to `results/summary.txt`

## Important notes

- This folder uses `THERMAL_PATH = /sys/class/thermal/thermal_zone1/temp` by default because that is what you asked for.
- On some Jetson images, `thermal_zone1` may not be the CPU sensor. Check with:
  ```bash
  cat /sys/class/thermal/thermal_zone1/type
  ```
  If that is not the CPU-related zone you want, change `THERMAL_PATH` in `run_benchmark.py`.
- Baseline is meant to be the stock Jetson run.
- The other schemes can optionally run:
  - `sudo nvpmodel -m 0`
  - `sudo jetson_clocks`
  - CPU governor -> `performance`
- This code does **not** try to disable Jetson thermal safety shutdown/throttling.

## Schemes

- `baseline`: stock OS scheduling, unpinned worker processes
- `sequential`: all work on one pinned core
- `concurrent`: even split across pinned cores
- `rotating_single`: one chunk at a time, rotating across cores
- `hybrid_parallel`: temperature-aware chunked parallelism; uses 4, then 2, then 1 core(s) as temperature rises

## Stress tests

- `matrix_multiply`
- `mlp_numpy`

## Plotting

- `run_benchmark.py` automatically makes a single-run plot.
- `python3 plotting/plot_all.py` makes one plot with every CSV in `results/csv`.
