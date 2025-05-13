import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from zArena.agents.base_agent import zAgent
from zArena.utils.reward_shaping import get_healing_model # Ensure this path is correct

class SymbolicPolicyAgent(zAgent):
    def __init__(self, observation_space_shape, action_space_dim, lr=1e-3):
        super().__init__(observation_space_shape, action_space_dim)
        
        # Ensure observation_space_shape is a tuple, e.g., (2,) for zHealingGrid
        self.input_dim = np.prod(observation_space_shape) if isinstance(observation_space_shape, tuple) else observation_space_shape

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.healing_model = get_healing_model() # Uses the globally initialized healing_model

    def act(self, observation):
        # Ensure observation is a flat tensor for the linear layer
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.tensor(observation, dtype=torch.float32).flatten()
        else: # if it's already a tensor
            obs_tensor = observation.clone().detach().to(torch.float32).flatten()

        if obs_tensor.shape[0] != self.input_dim:
             raise ValueError(f"Observation shape mismatch. Expected {self.input_dim}, got {obs_tensor.shape[0]}")

        logits = self.model(obs_tensor)
        
        # Heal-aware entropy scaling
        # Assuming obs is [..., time] and time is the last element if obs_shape > 1
        # For zHealingGrid, obs is [position_norm, time], so observation[1] is time.
        # If observation_space_shape is just (N,), then observation[1] is the second element.
        current_time_for_healing_model = observation[-1] # Assuming time is the last feature in the observation
        
        h_val_array = self.healing_model.predict(np.array([current_time_for_healing_model]))
        
        # Ensure h_val is a scalar
        if isinstance(h_val_array, np.ndarray):
            h_val = h_val_array.item(0) if h_val_array.size == 1 else h_val_array[0]
        else: # if it's already a scalar like a float
            h_val = h_val_array

        temp = max(0.1, 1.0 - h_val) # Lower temp = more deterministic as Ĥ(t) rises
        
        probs = torch.softmax(logits / temp, dim=-1)
        
        # Entropy calculation: H(X) = - sum(p(x) * log(p(x)))
        # Add epsilon for numerical stability if probs can be 0
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item() 
        
        action = torch.multinomial(probs, num_samples=1).item()
        return action, h_val, temp, entropy # Return h_val, temp, and entropy for logging

    def update(self, trajectory, gamma=0.99):
        # trajectory is expected to be list of (obs, action, shaped_reward, next_obs, done)
        rewards = []
        log_action_probs = []
        
        for (obs, action, reward, _, _) in trajectory:
            rewards.append(reward)
            
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).flatten()
            else:
                obs_tensor = obs.clone().detach().to(torch.float32).flatten()

            logits = self.model(obs_tensor)
            log_probs_all_actions = torch.log_softmax(logits, dim=-1) # log_softmax for numerical stability
            log_action_probs.append(log_probs_all_actions[action])

        # Calculate discounted returns (Gt)
        discounted_returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_returns.insert(0, R)
        
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)
        
        # Normalize returns for stability (optional but often good practice)
        if len(discounted_returns) > 1:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9) # Epsilon for std dev
        elif len(discounted_returns) == 1:
            # Avoid division by zero if only one step; mean is the value itself, std is 0.
            # Not strictly necessary for REINFORCE with single step but good for general case.
             pass


        policy_loss = []
        for log_prob, G_t in zip(log_action_probs, discounted_returns):
            policy_loss.append(-log_prob * G_t) # REINFORCE loss: -log_prob(a|s) * Gt
        
        self.optimizer.zero_grad()
        # Summing the losses for all steps in the trajectory
        total_policy_loss = torch.stack(policy_loss).sum() 
        total_policy_loss.backward()
        self.optimizer.step()