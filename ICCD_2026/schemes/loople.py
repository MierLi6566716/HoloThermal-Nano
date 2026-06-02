
import time
import os
import multiprocessing as mp
import random
import copy
from statistics import mean
from utils.worker import _worker_entry
from utils.temp_utils import read_temp_c
from utils.logger import get_cpu_freq_mhz

# --- MULTI-ARM BANDIT UTILS ---

class EpsilonGreedyBandit:
    """
    Throughput-Oriented Learning engine.
    """
    def __init__(self, strategies, epsilon=0.2):
        self.strategies = strategies
        self.epsilon = epsilon
        self.rewards = {i: [] for i in range(len(strategies))}
        
    def select_strategy(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.strategies) - 1), "EXPLORE"
        avg_rewards = {i: (mean(r) if r else 0) for i, r in self.rewards.items()}
        best_idx = max(avg_rewards, key=avg_rewards.get)
        return best_idx, "EXPLOIT"
    
    def update(self, idx, reward):
        self.rewards[idx].append(reward)

def pt_generator(pt): 
    """
    Neighborhood generator for thermal strategy exploration.
    """
    nbhd = []
    # pt is [t_min, t_max, t_s, t_c]
    for i in [-2.0, 2.0]: # Wider t_min variance
        for j in [-1.0, 1.0]: # Wider t_max variance
            for k in [-2, 2]: # Wider serial duration variance
                for l in [-2, 2]: # Wider concurrent duration variance
                    new_pt = copy.deepcopy(pt)
                    new_pt[0] += i
                    new_pt[1] += j
                    new_pt[2] += k
                    new_pt[3] += l
                    
                    # Hard bounds for Jetson Nano 45C limit
                    new_pt[0] = max(37.0, min(43.0, new_pt[0])) # t_min
                    new_pt[1] = max(44.0, min(48.0, new_pt[1])) # t_max
                    new_pt[2] = max(1.0, min(20.0, new_pt[2]))  # t_s
                    new_pt[3] = max(1.0, min(20.0, new_pt[3]))  # t_c
                    nbhd.append(new_pt)
    return nbhd

def run(stress_test_name, config):
    total_units = int(config["workload_units"])
    all_cores = list(config["active_cores"]) # [0, 1, 2, 3]
    temp_path = config["thermal_path"]
    
    # Anchor point optimized for throughput (Fast bursts, short cooling)
    anchor = [42.0, 44.5, 4.0, 10.0] 
    strategies = pt_generator(anchor)
    bandit = EpsilonGreedyBandit(strategies, epsilon=0.25) # Slightly higher exploration
    
    epoch_duration = 15.0 # Seconds per bandit trial
    
    print(f"\n[Loople] Starting THROUGHPUT-PRIORITY Bandit")
    print(f"[Loople] Arms: {len(strategies)} | Epsilon: {bandit.epsilon}")

    # 1. Spawn worker processes
    units_per_worker = total_units // len(all_cores)
    remainder = total_units % len(all_cores)
    processes = []
    for i, core_id in enumerate(all_cores):
        u = units_per_worker + (1 if i < remainder else 0)
        p = mp.Process(target=_worker_entry, args=(core_id, stress_test_name, u, config))
        p.start()
        processes.append(p)

    overall_start = time.perf_counter()
    
    # 2. Learning Loop
    while any(p.is_alive() for p in processes):
        strat_idx, mode_name = bandit.select_strategy()
        t_min, t_max, t_s, t_c = bandit.strategies[strat_idx]
        
        print(f"[Loople] [{mode_name}] Arm {strat_idx}: min={t_min} max={t_max} s={t_s} c={t_c}")
        
        epoch_start_time = time.perf_counter()
        samples = [] # List of (freq * cores) tuples
        
        mode = "CONCURRENT"
        for i, p in enumerate(processes):
            try: os.sched_setaffinity(p.pid, {all_cores[i]})
            except: pass

        while (time.perf_counter() - epoch_start_time) < epoch_duration:
            if not any(p.is_alive() for p in processes): break
            
            # Sampling: Measure throughput delivered at this exact millisecond
            cur_freq = get_cpu_freq_mhz()
            cur_cores = 4 if mode == "CONCURRENT" else 1
            samples.append(cur_freq * cur_cores)

            current_temp = read_temp_c(temp_path)
            
            # --- CRUX STATE MACHINE ---
            if current_temp >= t_max and mode == "CONCURRENT":
                print(f"  -> Panic ({current_temp:.1f}C). Switching to SERIAL for {t_s}s")
                for p in processes:
                    try: os.sched_setaffinity(p.pid, {all_cores[0]})
                    except: pass
                mode = "SERIAL"
                hold_end = time.perf_counter() + t_s
                while time.perf_counter() < hold_end:
                    if not any(p.is_alive() for p in processes): break
                    samples.append(get_cpu_freq_mhz() * 1) # Recording serial throughput
                    time.sleep(0.1)

            elif current_temp <= t_min and mode == "SERIAL":
                print(f"  -> Recovery ({current_temp:.1f}C). Switching to CONCURRENT for {t_c}s")
                for i, p in enumerate(processes):
                    try: os.sched_setaffinity(p.pid, {all_cores[i]})
                    except: pass
                mode = "CONCURRENT"
                hold_end = time.perf_counter() + t_c
                while time.perf_counter() < hold_end:
                    if not any(p.is_alive() for p in processes): break
                    samples.append(get_cpu_freq_mhz() * 4) # Recording concurrent throughput
                    time.sleep(0.1)
            
            time.sleep(0.1)

        # 3. Reward = Average Throughput (Compute Units / s)
        if samples:
            avg_throughput = mean(samples)
            bandit.update(strat_idx, avg_throughput)

    duration = time.perf_counter() - overall_start
    print(f"\n[Loople] Execution completed in {duration:.2f}s")
    
    # WINNING STRATEGY REPORT
    avg_rewards = {i: (mean(r) if r else 0) for i, r in bandit.rewards.items()}
    best_idx = max(avg_rewards, key=avg_rewards.get)
    print(f"[Loople] HIGHEST THROUGHPUT STRATEGY: Arm {best_idx} -> {strategies[best_idx]}")
