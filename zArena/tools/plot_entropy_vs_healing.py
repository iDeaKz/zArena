import pandas as pd
import matplotlib.pyplot as plt
import glob # To find the latest log file
import os

# Find the latest entropy log file
log_dir = "zArena_logs"
list_of_files = glob.glob(os.path.join(log_dir, "entropy_log_*.csv")) 
if not list_of_files:
    print(f"No entropy log files found in {log_dir} with prefix 'entropy_log_'. Exiting.")
    exit()

latest_file = max(list_of_files, key=os.path.getctime)
print(f"Plotting data from: {latest_file}")

df = pd.read_csv(latest_file)

if df.empty:
    print("Log file is empty. No data to plot.")
    exit()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(df["global_step"], df["H_t_value"], label="Ĥ(t) (for policy)", color="green", alpha=0.7)
plt.plot(df["global_step"], df["policy_entropy"], label="Policy Entropy", color="red", linestyle="--", alpha=0.7)
plt.xlabel("Global Step")
plt.ylabel("Value")
plt.title("Healing Signal (Policy) vs Policy Entropy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df["global_step"], df["temperature"], label="Softmax Temperature", color="blue", alpha=0.7)
plt.xlabel("Global Step")
plt.ylabel("Temperature")
plt.title("Policy Softmax Temperature over Time")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()