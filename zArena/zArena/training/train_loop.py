import numpy as np
from zArena.utils.logger import EpisodeLogger # General episode logger
from zArena.utils.entropy_logger import SymbolicEntropyLogger # Symbolic entropy logger
from zArena.utils.reward_shaping import shaped_reward, get_healing_model
from zArena.agents.symbolic_policy_agent import SymbolicPolicyAgent # Import the new agent

def train(env, episodes=100, reward_shaping_alpha=1.5, agent_lr=1e-3):
    """
    Training loop for zArena environments using SymbolicPolicyAgent.

    Args:
        env: An instance of a zEnv environment.
        episodes (int): Number of episodes to train for.
        reward_shaping_alpha (float): Alpha parameter for reward shaping.
        agent_lr (float): Learning rate for the agent.
    """
    # Assuming env.observation_space and env.action_space are available
    # For zHealingGrid, obs_space is Box(low=0, high=1, shape=(2,), dtype=np.float32)
    # action_space is Discrete(3)
    # We need to pass the shape for observation_space and dimension for action_space
    # If env doesn't have these properties directly, they need to be inferred or passed.
    # For zHealingGrid, let's assume:
    # obs_space_shape = (2,) # (position_norm, time)
    # action_space_dim = 3   # (left, stay, right)
    # A more general way would be:
    # obs_space_shape = env.observation_space().shape # If zEnv implements this
    # action_space_dim = env.action_space().n # If zEnv implements this for Discrete spaces

    # For zHealingGrid specifically:
    obs_shape = (2,) # Based on zHealingGrid's _get_obs()
    act_dim = 3      # Based on zHealingGrid's action space

    agent = SymbolicPolicyAgent(
        observation_space_shape=obs_shape, 
        action_space_dim=act_dim, 
        lr=agent_lr
    )
    
    general_logger = EpisodeLogger() # Your existing logger
    entropy_logger = SymbolicEntropyLogger() # New logger for entropy details

    healing_model = get_healing_model()
    global_step_counter = 0

    for ep in range(episodes):
        obs = env.reset() # obs from zHealingGrid is np.array([pos_norm, time_t])
        
        trajectory = []
        done = False
        total_steps_episode = 0
        total_raw_reward_episode = 0
        total_shaped_reward_episode = 0
        h_values_sum_episode = 0

        while not done:
            # Agent acts and returns action, h_val, temp, entropy
            action, h_val_from_act, temp_from_act, entropy_from_act = agent.act(obs)
            
            next_obs, raw_reward, done, info = env.step(action)
            
            # current_env_time is the `t` value from the *current* observation `obs`
            # which was used by agent.act() to calculate h_val_from_act
            current_env_time = obs[1] # time_t component of observation for zHealingGrid
            
            # Shaped reward is calculated based on the state *after* the action, using next_obs's time
            # For zHealingGrid, H(t) in reward uses env.t *after* step, which is next_obs[1]
            time_for_reward_shaping = next_obs[1]
            current_shaped_reward = shaped_reward(raw_reward, time_for_reward_shaping, alpha=reward_shaping_alpha, healing_model=healing_model)

            trajectory.append((obs, action, current_shaped_reward, next_obs, done))
            global_step_counter += 1
            
            # Log general step data
            general_logger.log(
                episode=ep + 1,
                step=total_steps_episode + 1,
                time_env=time_for_reward_shaping, # Log time component of next_obs
                position_agent=next_obs[0] * 4.0, # De-normalize position for zHealingGrid
                raw_reward=raw_reward,
                shaped_reward=current_shaped_reward,
                h_value_from_model=h_val_from_act, # H_val used for policy temperature
                action_taken=action
            )
            
            # Log symbolic entropy data
            entropy_logger.log(
                episode=ep + 1,
                step_in_episode=total_steps_episode + 1,
                global_step=global_step_counter,
                t_env=current_env_time, # Time `t` used by agent.act for h_val & temp
                h_val=h_val_from_act,
                temp=temp_from_act,
                entropy_val=entropy_from_act
            )
            
            obs = next_obs
            total_steps_episode += 1
            total_raw_reward_episode += raw_reward
            total_shaped_reward_episode += current_shaped_reward
            h_values_sum_episode += h_val_from_act # Sum H values used for policy temp
        
        agent.update(trajectory) # Agent learns from trajectory with shaped rewards

        avg_h_value_policy_episode = h_values_sum_episode / total_steps_episode if total_steps_episode > 0 else 0
        general_logger.log_episode_summary(
            ep + 1, 
            total_steps_episode, 
            total_raw_reward_episode, 
            total_shaped_reward_episode, 
            avg_h_value_policy_episode
        )

        print(f"Episode {ep + 1} finished after {total_steps_episode} steps. Total Shaped Reward: {total_shaped_reward_episode:.2f}. Avg H (policy): {avg_h_value_policy_episode:.2f}")

    print("Training complete.")
    print(f"General logs in: {general_logger.path}")
    print(f"Entropy logs in: {entropy_logger.log_path}")
