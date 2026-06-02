
import torch
import time
import os
import gc

def run_units(units, config_thermal):
    """
    Heavy Math Stress Test (Memory Efficient).
    Uses iterative FFTs and Large Matrix Ops to generate the same heat as DPRC
    without the 1GB RAM overhead of Transformer weights.
    """
    # 1. Threading Scaling
    num_threads = int(config_thermal.get('current_threads', 1))
    torch.set_num_threads(num_threads)
    
    # 2. Pinning
    active_cores = config_thermal.get('active_cores', [0, 1, 2, 3])
    try: os.sched_setaffinity(0, set(active_cores))
    except: pass

    scheme = config_thermal.get("scheme_name", "unknown")
    deadline_s = float(config_thermal.get("deadline_s", 0))
    
    # --- CONFIGURATION ---
    # Calibrated to take ~1.3s on 1 core at max freq
    ITERATIONS = 450 
    RES = (512, 512)
    
    # 3. Create synthetic workload in RAM (Uses ~2MB)
    workload = torch.randn(RES, dtype=torch.float32)
    
    print(f"[{os.getpid()}] Running {units} units ({scheme}) [HEAVY-MATH]... Threads: {num_threads}")

    for i in range(int(units)):
        unit_start = time.perf_counter()
        
        # --- CORE WORKLOAD ---
        # We perform a mix of FFTs (ASM-like) and Matrix multiplications (ViT-like)
        # to generate intense thermal stress.
        x = workload.clone()
        with torch.no_grad():
            for _ in range(ITERATIONS):
                # 1. Spectral Math
                x = torch.fft.fft2(x).real
                # 2. Non-linear heat generation
                x = torch.sin(x) + torch.cos(x)
                # 3. Floating point pressure
                x = x * 1.0001
            
            # Final reduction to force computation
            _ = x.mean().item()

        unit_duration = time.perf_counter() - unit_start
        fps = 1.0 / unit_duration if unit_duration > 0 else 0
        
        # Deadline Tracking (for terminal feedback)
        status = "PASS"
        if deadline_s > 0 and unit_duration > deadline_s:
            status = f"FAIL (+{unit_duration-deadline_s:.2f}s)"

        # Throttle terminal logging to keep math clean
        if i % 5 == 0:
            print(f"[{os.getpid()}] Unit {i}: {unit_duration:.2f}s ({fps:.2f} FPS) | {status} | Threads: {num_threads}")
            
    gc.collect()
