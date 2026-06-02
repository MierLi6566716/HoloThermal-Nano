
import time
import os
import multiprocessing as mp
import random
import copy
from statistics import mean
from utils.worker import _worker_entry
from utils.temp_utils import read_temp_c
from utils.logger import get_cpu_freq_mhz

# --- PREDICTIVE MULTI-ARM BANDIT UTILS ---

class EpsilonGreedyBandit:
    def __init__(self, strategies, epsilon=0.25):
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
    Neighborhood generator for THROUGHPUT-PRIORITY Predictive parameters.
    pt: [threshold_temp, lookahead_s, t_s, t_c]
    """
    nbhd = []
    # pt is [threshold_temp, lookahead_s, t_s, t_c]
    for i in [-1.0, 1.0]: # Wider threshold variance
        for j in [-3.0, 3.0]: # Wider lookahead variance
            for k in [-2, 2]: # Serial hold variance
                for l in [-3, 3]: # Concurrent hold variance
                    new_pt = copy.deepcopy(pt)
                    new_pt[0] += i
                    new_pt[1] += j
                    new_pt[2] += k
                    new_pt[3] += l
                    
                    # Hard bounds for Jetson Nano 45C limit (Pushing the limits)
                    new_pt[0] = max(42.0, min(47.5, new_pt[0])) # Threshold
                    new_pt[1] = max(0.5, min(15.0, new_pt[1]))  # Lookahead
                    new_pt[2] = max(1.0, min(15.0, new_pt[2]))  # t_s
                    new_pt[3] = max(1.0, min(25.0, new_pt[3]))  # t_c
                    nbhd.append(new_pt)
    return nbhd

class ThermalPredictor:
    def __init__(self, history_size=15):
        self.history = []
        self.history_size = history_size

    def add_sample(self, temp):
        self.history.append((time.perf_counter(), temp))
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def get_slope(self):
        if len(self.history) < 2: return 0.0
        dt = self.history[-1][0] - self.history[0][0]
        dy = self.history[-1][1] - self.history[0][1]
        return dy / dt if dt > 0 else 0.0

    def predict_future(self, lookahead_s):
        if not self.history: return 0.0
        slope = self.get_slope()
        return self.history[-1][1] + (slope * lookahead_s)

# --- MAIN SCHEME ---

def run(stress_test_name, config):
    total_units = int(config["workload_units"])
    all_cores = list(config["active_cores"]) # [0, 1, 2, 3]
    temp_path = config["thermal_path"]
    
    # 1. Initialize Bandit with Aggressive Predictive Anchor
    # [Threshold, Lookahead, Serial-Duration, Concurrent-Duration]
    anchor = [44.8, 4.0, 3.0, 15.0] 
    strategies = pt_generator(anchor)
    bandit = EpsilonGreedyBandit(strategies, epsilon=0.25)
    predictor = ThermalPredictor(history_size=20)
    
    epoch_duration = 15.0 
    
    print(f"\n[Predictive] Starting THROUGHPUT-PRIORITY MAB-Headroom")
    print(f"[Predictive] Learning optimal lookahead and core-duty cycle...")

    # 2. Spawn worker processes
    units_per_worker = total_units // len(all_cores)
    remainder = total_units % len(all_cores)
    processes = []
    for i, core_id in enumerate(all_cores):
        u = units_per_worker + (1 if i < remainder else 0)
        p = mp.Process(target=_worker_entry, args=(core_id, stress_test_name, u, config))
        p.start()
        processes.append(p)

    overall_start = time.perf_counter()
    
    # 3. Learning loop
    while any(p.is_alive() for p in processes):
        strat_idx, mode_name = bandit.select_strategy()
        t_threshold, lookahead, t_s, t_c = bandit.strategies[strat_idx]
        
        print(f"[Predictive] [{mode_name}] Arm {strat_idx}: Threshold={t_threshold}C | Lookahead={lookahead}s | c_hold={t_c}s")
        
        epoch_start_time = time.perf_counter()
        samples = [] # Throughput samples (Freq * Cores)
        
        mode = "CONCURRENT"
        # Immediate Concurrent state
        for i, p in enumerate(processes):
            try: os.sched_setaffinity(p.pid, {all_cores[i]})
            except: pass

        while (time.perf_counter() - epoch_start_time) < epoch_duration:
            if not any(p.is_alive() for p in processes): break
            
            cur_temp = read_temp_c(temp_path)
            predictor.add_sample(cur_temp)
            
            # Record current throughput
            cur_freq = get_cpu_freq_mhz()
            cur_cores = 4 if mode == "CONCURRENT" else 1
            samples.append(cur_freq * cur_cores)

            # PREDICTION
            predicted_temp = predictor.predict_future(lookahead)

            # --- PREDICTIVE STATE MACHINE ---
            if predicted_temp >= t_threshold and mode == "CONCURRENT":
                print(f"  -> Headroom Trigger! Pred={predicted_temp:.1f}C. Switching to SERIAL.")
                for p in processes:
                    try: os.sched_setaffinity(p.pid, {all_cores[0]})
                    except: pass
                mode = "SERIAL"
                hold_end = time.perf_counter() + t_s
                while time.perf_counter() < hold_end:
                    if not any(p.is_alive() for p in processes): break
                    samples.append(get_cpu_freq_mhz() * 1)
                    time.sleep(0.1)

            elif cur_temp <= 41.5 and mode == "SERIAL":
                print(f"  -> Recovery ({cur_temp:.1f}C). Switching to CONCURRENT.")
                for i, p in enumerate(processes):
                    try: os.sched_setaffinity(p.pid, {all_cores[i]})
                    except: pass
                mode = "CONCURRENT"
                hold_end = time.perf_counter() + t_c
                while time.perf_counter() < hold_end:
                    if not any(p.is_alive() for p in processes): break
                    samples.append(get_cpu_freq_mhz() * 4)
                    time.sleep(0.1)
            
            time.sleep(0.1)

        # 4. Update Bandit Reward (Throughput delivered)
        if samples:
            bandit.update(strat_idx, mean(samples))

    duration = time.perf_counter() - overall_start
    print(f"\n[Predictive] Workload finished in {duration:.2f}s")
    
    avg_rewards = {i: (mean(r) if r else 0) for i, r in bandit.rewards.items()}
    best_idx = max(avg_rewards, key=avg_rewards.get)
    print(f"[Predictive] HIGHEST THROUGHPUT STRATEGY: Arm {best_idx} -> {strategies[best_idx]}")
