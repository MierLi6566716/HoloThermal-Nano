import time
from utils.worker import make_even_assignments, run_assignments
from utils.temp_utils import read_temp_c

# Define a Jetson Nano specific Loople trajectory based on the requested rotation:
# Spike -> Conservative -> Mild -> Deep.
# Tuple format: (t_min, t_max, d_cool, d_hot, cool_cores)

L_NANO_OPT = [
    # 1. SPIKE: High throughput, high threshold, minimal cooling (3 cores)
    (43.0, 48.5, 3.0, 8.0, 3), 
    
    # 2. CONSERVATIVE: Aggressive cooling start, low threshold, long cooling dwell
    (40.0, 45.0, 8.0, 3.0, 1),
    
    # 3. MILD: Balanced middle ground (2 cores)
    (41.5, 46.5, 5.0, 5.0, 2),
    
    # 4. DEEP: Very low return temp, aggressive single-core dump
    (39.0, 46.0, 7.0, 4.0, 1)
]

EPOCH_TIME_S = 25.0  # Time to spend in each tuple before moving to the next

def run(stress_test_name, config):
    remaining = int(config["workload_units"])
    
    # Force small chunk units so the scheduler evaluates temperatures frequently
    # 1 unit per active core ensures a fast feedback loop for the scheduler.
    chunk_units = 1 
    
    all_cores = list(config["active_cores"]) # e.g. [0, 1, 2, 3]
    temp_path = config["thermal_path"]
    
    # Initialize State
    loople_idx = 0
    epoch_start_time = time.perf_counter()
    
    mode = "hot" # 'hot' (4 cores) or 'cool' (cool_cores)
    mode_start_time = time.perf_counter()
    
    print(f"\n[Loople] Starting Execution. Trajectory length: {len(L_NANO_OPT)}")
    print(f"[Loople] Initial Epoch {loople_idx}: {L_NANO_OPT[loople_idx]}")
    
    while remaining > 0:
        now_s = time.perf_counter()
        
        # 1. Advance the Loople Epoch if time has expired
        if (now_s - epoch_start_time) > EPOCH_TIME_S:
            loople_idx = (loople_idx + 1) % len(L_NANO_OPT)
            epoch_start_time = now_s
            print(f"\n[Loople] >>> Advancing to Epoch {loople_idx}: {L_NANO_OPT[loople_idx]}")
            
        # Unpack current tuple
        t_min, t_max, d_cool, d_hot, cool_cores = L_NANO_OPT[loople_idx]
        current_temp_c = read_temp_c(temp_path)
        
        time_in_mode = now_s - mode_start_time
        
        # 2. Evaluate Mode Transitions (Timed Thermal Logic)
        if mode == "hot":
            # Switch to cooling if dwell time passed AND temp is too high
            if time_in_mode >= d_hot and current_temp_c >= t_max:
                print(f"[Loople] Temp {current_temp_c:.1f}C >= {t_max}C. Switching to COOL mode ({cool_cores} cores).")
                mode = "cool"
                mode_start_time = time.perf_counter()
        elif mode == "cool":
            # Switch to hot if dwell time passed AND temp is low enough
            if time_in_mode >= d_cool and current_temp_c <= t_min:
                print(f"[Loople] Temp {current_temp_c:.1f}C <= {t_min}C. Switching to HOT mode (4 cores).")
                mode = "hot"
                mode_start_time = time.perf_counter()
                
        # 3. Dispatch the Workload Chunk
        if mode == "hot":
            active_cores = all_cores
        else:
            # Drop down to the number of cooling cores specified by the current epoch tuple
            active_cores = all_cores[:int(cool_cores)] 
            
        wave_units = min(remaining, chunk_units * len(active_cores))
        wave_units = max(1, wave_units) # Ensure at least 1 unit is dispatched
        
        assignments = make_even_assignments(wave_units, active_cores)
        
        # run_assignments is synchronous, it waits for the chunk to finish
        run_assignments(assignments, stress_test_name, config, stagger_s=0.0)
        
        remaining -= wave_units

    print(f"\n[Loople] Completed workload.")
