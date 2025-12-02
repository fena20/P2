"""
PPO (Proximal Policy Optimization) with LSTM Policy for HVAC Control
Implements PPO algorithm with LSTM-based policy network for sequential decision making
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Tuple, List, Dict
import gymnasium as gym

class LSTMPolicyNetwork(nn.Module):
    """
    LSTM-based policy network for PPO.
    Outputs both action mean and standard deviation for continuous actions.
    """
    
    def __init__(self, state_dim, action_dim, hidden_size=128, num_layers=2):
        super(LSTMPolicyNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Policy head (mean)
        self.policy_mean = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Policy head (std)
        self.policy_std = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softplus()  # Ensure positive std
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state, hidden=None):
        """
        Forward pass.
        
        Args:
            state: (batch_size, seq_len, state_dim) or (batch_size, state_dim)
            hidden: LSTM hidden state
        
        Returns:
            action_mean: Mean of action distribution
            action_std: Standard deviation of action distribution
            value: State value estimate
            hidden: Updated hidden state
        """
        # Handle single timestep input
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward
        lstm_out, hidden = self.lstm(state, hidden)
        
        # Use last timestep
        last_hidden = lstm_out[:, -1, :]
        
        # Policy outputs
        action_mean = self.policy_mean(last_hidden)
        action_std = self.policy_std(last_hidden) + 1e-5  # Small epsilon for numerical stability
        
        # Value output
        value = self.value_head(last_hidden)
        
        return action_mean, action_std, value, hidden
    
    def get_action(self, state, hidden=None, deterministic=False):
        """Sample action from policy."""
        action_mean, action_std, value, hidden = self.forward(state, hidden)
        
        if deterministic:
            return action_mean, value, hidden
        
        # Sample from normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, action_log_prob, value, hidden

class PPOAgent:
    """
    PPO Agent with LSTM policy for HVAC control.
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 hidden_size=128, num_layers=2, device='cuda'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Policy network
        self.policy = LSTMPolicyNetwork(state_dim, action_dim, hidden_size, num_layers).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Action space bounds (for scaling)
        self.action_low = np.array([16.0, 0.0])
        self.action_high = np.array([26.0, 2.0])
        
    def scale_action(self, action):
        """Scale action from [-1, 1] to actual action space."""
        action = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        scaled = (action + 1) / 2 * (self.action_high - self.action_low) + self.action_low
        return scaled
    
    def update(self, states, actions, old_log_probs, rewards, values, dones, 
               next_values, num_epochs=10, batch_size=64):
        """
        Update policy using PPO algorithm.
        
        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, action_dim)
            old_log_probs: (batch_size, 1)
            rewards: (batch_size,)
            values: (batch_size, 1)
            dones: (batch_size,)
            next_values: (batch_size, 1)
            num_epochs: Number of update epochs
            batch_size: Batch size for updates
        """
        # Compute advantages using GAE (Generalized Advantage Estimation)
        advantages = self.compute_gae(rewards, values, next_values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Training loop
        total_loss = 0.0
        n_batches = len(states) // batch_size
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(states))
            
            for i in range(n_batches):
                batch_idx = indices[i*batch_size:(i+1)*batch_size]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Get current policy outputs
                action_mean, action_std, value_pred, _ = self.policy(batch_states)
                
                # Compute new log probabilities
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Policy loss (clipped)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(value_pred, batch_returns)
                
                # Entropy bonus
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / (num_epochs * n_batches)
    
    def compute_gae(self, rewards, values, next_values, dones, lambda_=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                last_gae = delta + self.gamma * lambda_ * last_gae
            
            advantages[t] = last_gae
        
        return advantages
    
    def save(self, path):
        """Save model."""
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        """Load model."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))

def train_ppo_agent(env, agent, num_episodes=1000, max_steps=1000, 
                   update_frequency=2048, save_path='models/reinforcement_learning/ppo_lstm.pth'):
    """
    Training loop for PPO agent.
    """
    episode_rewards = []
    episode_lengths = []
    
    # Storage for trajectories
    states_buffer = []
    actions_buffer = []
    rewards_buffer = []
    log_probs_buffer = []
    values_buffer = []
    dones_buffer = []
    
    hidden_state = None
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Convert state to sequence format for LSTM
        state_seq = deque([state], maxlen=10)  # Keep last 10 states
        
        for step in range(max_steps):
            # Prepare state sequence
            state_tensor = torch.FloatTensor(np.array(state_seq)).unsqueeze(0).to(agent.device)
            
            # Get action from policy
            action, log_prob, value, hidden_state = agent.policy.get_action(
                state_tensor, hidden_state
            )
            
            # Scale action
            scaled_action = agent.scale_action(action.squeeze().cpu())
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(scaled_action)
            done = terminated or truncated
            
            # Store transition
            states_buffer.append(np.array(state_seq))
            actions_buffer.append(action.squeeze().cpu().numpy())
            rewards_buffer.append(reward)
            log_probs_buffer.append(log_prob.item())
            values_buffer.append(value.item())
            dones_buffer.append(done)
            
            # Update state sequence
            state_seq.append(next_state)
            state = next_state
            
            episode_reward += reward
            episode_length += 1
            
            # Update if buffer is full
            if len(states_buffer) >= update_frequency:
                # Prepare data
                states = np.array(states_buffer)
                actions = np.array(actions_buffer)
                rewards = np.array(rewards_buffer)
                log_probs = np.array(log_probs_buffer)
                values = np.array(values_buffer)
                dones = np.array(dones_buffer)
                
                # Compute next values
                next_values = np.zeros_like(values)
                next_values[:-1] = values[1:]
                next_values[-1] = 0 if dones[-1] else values[-1]
                
                # Update agent
                loss = agent.update(states, actions, log_probs, rewards, values, 
                                  dones, next_values)
                
                # Clear buffers
                states_buffer = []
                actions_buffer = []
                rewards_buffer = []
                log_probs_buffer = []
                values_buffer = []
                dones_buffer = []
                
                print(f"Update at episode {episode}, step {step}, loss: {loss:.4f}")
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
            
            # Save model
            agent.save(save_path)
    
    return episode_rewards, episode_lengths
