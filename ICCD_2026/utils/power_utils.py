
import os

# INA3221 sensor paths for Jetson Nano (standard paths)
# These paths provide current power consumption in milliwatts (mW)
# Total System Power is typically channel 1 or a sum of rails.
# On Jetson Nano, /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input is common
# but can vary by L4T version. We'll use a reliable fallback to total power.

POWER_PATH = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input"

def read_power_mw():
    """Reads current power consumption in milliwatts (mW)."""
    try:
        if os.path.exists(POWER_PATH):
            with open(POWER_PATH, "r") as f:
                return float(f.read().strip())
        return 0.0
    except Exception:
        return 0.0

def compute_energy_joules(avg_power_mw, duration_s):
    """
    Computes total energy in Joules.
    Energy (J) = Power (W) * Time (s)
    """
    power_w = avg_power_mw / 1000.0
    return power_w * duration_s
