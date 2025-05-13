import csv
import os
from datetime import datetime

class EpisodeLogger:
    def __init__(self, log_dir="zArena_logs", file_prefix="symbolic_episode_log"):
        """
        Initializes the EpisodeLogger.

        Args:
            log_dir (str): Directory to save log files.
            file_prefix (str): Prefix for the log file name.
        """
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.log_dir, f"{file_prefix}_{timestamp}.csv")
        
        self.header = ["episode", "step", "time_env", "position_agent", "raw_reward", "shaped_reward", "h_value_from_model", "action_taken"] # Extended header
        
        with open(self.path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
        print(f"Logger initialized. Logging to: {self.path}")

    def log(self, episode, step, time_env, position_agent, raw_reward, shaped_reward, h_value_from_model, action_taken):
        """
        Logs a single step of an episode.

        Args:
            episode (int): Current episode number.
            step (int): Current step number within the episode.
            time_env (float): Current time `t` in the environment (for H_hat).
            position_agent (any): Agent's position or relevant state component.
            raw_reward (float): The base reward from the environment.
            shaped_reward (float): The reward after applying symbolic shaping.
            h_value_from_model (float): The Ĥ(t) value from the healing model.
            action_taken (any): The action taken by the agent.
        """
        with open(self.path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, step, time_env, position_agent, raw_reward, shaped_reward, h_value_from_model, action_taken])

    def log_episode_summary(self, episode, total_steps, total_raw_reward, total_shaped_reward, average_h_value):
        """
        Logs a summary for a completed episode.
        (This is an example, can be expanded or logged to a different file)
        """
        summary_path = os.path.join(self.log_dir, "episode_summaries.csv")
        summary_header = ["episode", "total_steps", "total_raw_reward", "total_shaped_reward", "average_h_value"]
        
        if not os.path.exists(summary_path):
            with open(summary_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(summary_header)

        with open(summary_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_steps, total_raw_reward, total_shaped_reward, average_h_value])