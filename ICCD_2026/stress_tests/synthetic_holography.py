
import torch
import time
import os
import gc
import numpy as np

# Cache synthetic data to avoid allocation overhead in the inner loop
_TARGET_CACHE = {}

def run_units(units, config_thermal):
    """
    Ultra-Aggressive Synthetic Holography Stress Test.
    Designed to hit 100% CPU utilization and generate maximum heat.
    """
    # 1. Threading
    torch.set_num_threads(1)
    
    scheme = config_thermal.get("scheme_name", "unknown")
    deadline_s = float(config_thermal.get("deadline_s", 0))
    
    # --- AGGRESSIVE CALIBRATION ---
    # Increased Resolution to hit memory bandwidth harder
    # Increased iterations to stay in the math kernel longer
    RES = (1024, 1024) 
    GS_ITERATIONS = 40 
    
    # 2. Lazy Initialization of large tensors
    if 'amp' not in _TARGET_CACHE:
        print(f"[{os.getpid()}] Initializing heavy math tensors (1024x1024)...")
        _TARGET_CACHE['amp'] = torch.rand(RES, dtype=torch.float32)
        _TARGET_CACHE['phase'] = torch.rand(RES, dtype=torch.float32) * 6.28
        _TARGET_CACHE['ones'] = torch.ones(RES, dtype=torch.float32)

    target_amp = _TARGET_CACHE['amp']
    current_phase = _TARGET_CACHE['phase']
    ones = _TARGET_CACHE['ones']

    for i in range(int(units)):
        unit_start = time.perf_counter()
        
        # --- TIGHT MATH LOOP (ZERO GAPS) ---
        # We use purely in-place operations or direct assignments to keep
        # the CPU SIMD units and cache constantly saturated.
        wavefront = torch.polar(ones, current_phase)
        
        with torch.no_grad():
            for _ in range(GS_ITERATIONS):
                # Forward
                field = torch.fft.fft2(wavefront)
                # Constrain Image Plane
                field = torch.polar(target_amp, torch.angle(field))
                # Backward
                field = torch.fft.ifft2(field)
                # Constrain SLM Plane
                wavefront = torch.polar(ones, torch.angle(field))

            # Final check to force completion
            _ = torch.abs(field).mean().item()

        unit_duration = time.perf_counter() - unit_start
        fps = 1.0 / unit_duration if unit_duration > 0 else 0
        
        # Status
        status = "PASS"
        if deadline_s > 0 and unit_duration > deadline_s:
            status = f"FAIL (+{unit_duration-deadline_s:.2f}s)"

        # MINIMAL LOGGING: Only log every 10 images to avoid printing overhead
        if i % 10 == 0:
            print(f"[{os.getpid()}] Unit {i}: {unit_duration:.2f}s ({fps:.2f} FPS) | {status}")
            
    # CRITICAL: No gc.collect() here. We let Python handle it naturally
    # to avoid the "70% usage" gaps caused by stop-the-world GC.
