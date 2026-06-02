
# Jetson Nano Thermal Stress Test Setup Guide

This guide explains how to set up your Jetson Nano to run the DPRC Holography stress test within the `model_1` thermal management framework.

## 1. Physical & Power Setup
The Jetson Nano must be in high-performance mode to generate measurable heat.

```bash
# Set power mode to 10W (MAXN)
sudo nvpmodel -m 0

# Lock CPU/GPU clocks to maximum
sudo jetson_clocks
```

## 2. Memory Management (Critical)
PyTorch + DPRC will exceed the 4GB RAM on the Jetson Nano. **You must enable swap space** or the OS will kill the process (OOM).

```bash
# Check existing swap
free -h

# If no swap or < 4G, create a 4GB swap file:
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# To make it permanent, add this to /etc/fstab:
# /swapfile swap swap defaults 0 0
```

## 3. Verify Thermal Zones
The thermal management framework reads from `/sys/class/thermal/thermal_zoneX/temp`. 
Verify which zone corresponds to the CPU:

```bash
for zone in /sys/class/thermal/thermal_zone*; do 
  echo "$(cat $zone/type): $zone"; 
done
```
*If your CPU zone is NOT `thermal_zone1`, open `Testing/run_benchmark.py` and change `THERMAL_PATH`.*

## 4. Running the Stress Test

1. **Go to the Testing directory**:
   ```bash
   cd Testing
   ```

2. **Run a Baseline**:
   Open `run_benchmark.py` and set:
   - `SCHEME_NAME = "baseline"`
   - `STRESS_TEST_NAME = "dprc_stress"`
   - `WORKLOAD_UNITS = 20` (Start small, then increase)

   Run:
   ```bash
   sudo swapoff /swapfile
   sudo fallocate -l 12G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   export OPENBLAS_CORETYPE=ARMV8
   python3 run_benchmark.py

   ```

3. **Run Thermal Schemes**:
   Change `SCHEME_NAME` to `hybrid_parallel` or `predictive_headroom` and rerun.

## 5. Viewing Outputs
Reconstructions generated during the stress test are saved to:
`Testing/holography/results/outputs/`

Logs and plots for each scheme are saved to:
`Testing/results_long/`


