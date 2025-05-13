import csv
import os
from datetime import datetime # Added for unique filenames

class SymbolicEntropyLogger:
    def __init__(self, log_dir="zArena_logs", file_prefix="entropy_log"):
        os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{file_prefix}_{timestamp}.csv")
        
        self.header = ["episode", "step_in_episode", "global_step", "time_env", "H_t_value", "temperature", "policy_entropy"]
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
        print(f"SymbolicEntropyLogger initialized. Logging to: {self.log_path}")

    def log(self, episode, step_in_episode, global_step, t_env, h_val, temp, entropy_val):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, step_in_episode, global_step, t_env, h_val, temp, entropy_val])
