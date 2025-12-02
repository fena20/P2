"""
Reinforcement Learning Agents for Building Energy Optimization
- Custom building environment
- PPO with LSTM policy for HVAC control
- Multi-agent RL for HVAC and lighting coordination
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib.pyplot as plt
import os


class BuildingEnergyEnv(gym.Env):
    """
    Custom Gymnasium environment for building energy management.
    
    State: temperature, humidity, time, occupancy, weather
    Actions: HVAC setpoint adjustments, lighting control
    Reward: -energy_cost - comfort_penalty
    """
    
    def __init__(self, data_processor=None, max_steps=1000):
        super(BuildingEnergyEnv, self).__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: continuous HVAC control + discrete lighting
        # HVAC: temperature setpoint change [-2, +2] degrees
        # Lighting: [0=off, 1=on]
        self.action_space = spaces.Box(
            low=np.array([-2.0, 0.0]),
            high=np.array([2.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space
        # [indoor_temp, indoor_humidity, outdoor_temp, outdoor_humidity,
        #  hour_sin, hour_cos, is_weekend, occupancy, comfort_index]
        self.observation_space = spaces.Box(
            low=np.array([10, 0, -20, 0, -1, -1, 0, 0, -3]),
            high=np.array([35, 100, 40, 100, 1, 1, 1, 1, 3]),
            dtype=np.float32
        )
        
        # Load historical data if provided
        self.data_processor = data_processor
        self.historical_data = None
        if data_processor is not None:
            rl_data = data_processor.prepare_rl_environment_data()
            self.historical_data = rl_data['data']
        
        # Environment state
        self.indoor_temp = 22.0
        self.indoor_humidity = 50.0
        self.outdoor_temp = 15.0
        self.outdoor_humidity = 60.0
        self.hour = 12
        self.is_weekend = 0
        self.occupancy = 1.0
        self.comfort_index = 0.0
        self.hvac_setpoint = 22.0
        self.lighting_state = 0
        
        # Cost parameters
        self.energy_cost_per_kwh = 0.12  # $/kWh
        self.comfort_penalty_weight = 0.5
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize from historical data if available
        if self.historical_data is not None and len(self.historical_data) > 0:
            idx = np.random.randint(0, len(self.historical_data) - self.max_steps)
            row = self.historical_data.iloc[idx]
            
            self.indoor_temp = row.get('T2', 22.0)
            self.indoor_humidity = row.get('RH2', 50.0)
            self.outdoor_temp = row.get('T_out', 15.0)
            self.outdoor_humidity = row.get('RH_out', 60.0)
            self.hour = row.get('hour', 12)
            self.is_weekend = row.get('is_weekend', 0)
        else:
            # Random initialization
            self.indoor_temp = np.random.uniform(20, 24)
            self.indoor_humidity = np.random.uniform(40, 60)
            self.outdoor_temp = np.random.uniform(5, 25)
            self.outdoor_humidity = np.random.uniform(30, 70)
            self.hour = np.random.randint(0, 24)
            self.is_weekend = np.random.randint(0, 2)
        
        self.hvac_setpoint = 22.0
        self.lighting_state = 0
        self.occupancy = self._get_occupancy()
        self.comfort_index = self._calculate_comfort()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return next state, reward, done, info."""
        # Parse action
        hvac_adjustment = np.clip(action[0], -2.0, 2.0)
        lighting_action = 1 if action[1] > 0.5 else 0
        
        # Update HVAC setpoint
        self.hvac_setpoint = np.clip(self.hvac_setpoint + hvac_adjustment, 18, 26)
        self.lighting_state = lighting_action
        
        # Simulate building dynamics
        self._update_building_state()
        
        # Calculate reward
        energy_consumption = self._calculate_energy_consumption()
        comfort_penalty = self._calculate_comfort_penalty()
        
        reward = -(energy_consumption * self.energy_cost_per_kwh + 
                   self.comfort_penalty_weight * comfort_penalty)
        
        # Update step counter
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Info dictionary
        info = {
            'energy': energy_consumption,
            'comfort_penalty': comfort_penalty,
            'indoor_temp': self.indoor_temp,
            'hvac_setpoint': self.hvac_setpoint,
            'comfort_index': self.comfort_index
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        """Construct observation vector."""
        hour_sin = np.sin(2 * np.pi * self.hour / 24)
        hour_cos = np.cos(2 * np.pi * self.hour / 24)
        
        obs = np.array([
            self.indoor_temp,
            self.indoor_humidity,
            self.outdoor_temp,
            self.outdoor_humidity,
            hour_sin,
            hour_cos,
            self.is_weekend,
            self.occupancy,
            self.comfort_index
        ], dtype=np.float32)
        
        return obs
    
    def _update_building_state(self):
        """Simulate building thermal dynamics."""
        # Simple thermal model: indoor temp moves toward setpoint and outdoor temp
        alpha = 0.3  # HVAC effectiveness
        beta = 0.1   # Heat transfer with outside
        
        # HVAC drives indoor temp toward setpoint
        self.indoor_temp += alpha * (self.hvac_setpoint - self.indoor_temp)
        
        # Heat transfer with outdoor environment
        self.indoor_temp += beta * (self.outdoor_temp - self.indoor_temp)
        
        # Add some noise
        self.indoor_temp += np.random.normal(0, 0.1)
        self.indoor_temp = np.clip(self.indoor_temp, 15, 30)
        
        # Humidity changes slightly
        self.indoor_humidity += np.random.normal(0, 1.0)
        self.indoor_humidity = np.clip(self.indoor_humidity, 20, 80)
        
        # Update time (advance by 10 minutes simulation)
        self.hour = (self.hour + 0.167) % 24  # 10 min = 0.167 hours
        
        # Update occupancy and comfort
        self.occupancy = self._get_occupancy()
        self.comfort_index = self._calculate_comfort()
    
    def _get_occupancy(self):
        """Estimate occupancy based on time of day."""
        hour_int = int(self.hour)
        
        # Weekday occupancy pattern
        if self.is_weekend == 0:
            if 0 <= hour_int < 7:
                return 0.8  # Sleeping
            elif 7 <= hour_int < 9:
                return 1.0  # Morning
            elif 9 <= hour_int < 17:
                return 0.2  # Away at work
            elif 17 <= hour_int < 23:
                return 1.0  # Evening
            else:
                return 0.8
        else:
            # Weekend: higher occupancy throughout day
            if 0 <= hour_int < 8:
                return 0.8
            else:
                return 1.0
    
    def _calculate_comfort(self):
        """Calculate thermal comfort index (simplified PMV)."""
        # Optimal: 20-24°C, 40-60% RH
        temp_deviation = (self.indoor_temp - 22) / 4
        humidity_penalty = np.abs(self.indoor_humidity - 50) / 50
        
        comfort = temp_deviation + 0.3 * humidity_penalty
        return np.clip(comfort, -3, 3)
    
    def _calculate_energy_consumption(self):
        """Calculate energy consumption in kWh."""
        # HVAC energy: proportional to temperature difference and setpoint
        temp_diff = abs(self.indoor_temp - self.hvac_setpoint)
        hvac_energy = 0.5 * temp_diff * 0.1  # kWh
        
        # Lighting energy
        lighting_energy = 0.05 * self.lighting_state  # kWh
        
        # Baseline appliances
        baseline_energy = 0.1  # kWh
        
        total_energy = hvac_energy + lighting_energy + baseline_energy
        return total_energy
    
    def _calculate_comfort_penalty(self):
        """Calculate comfort violation penalty."""
        # Penalty increases with occupancy
        comfort_violation = max(0, abs(self.comfort_index) - 1.0)
        penalty = comfort_violation * self.occupancy
        return penalty


class LSTMPolicyNetwork(nn.Module):
    """
    LSTM-based policy network for PPO.
    Handles sequential state information for better decision making.
    """
    
    def __init__(self, state_dim, action_dim, hidden_size=128, num_layers=1):
        super(LSTMPolicyNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for processing state sequences
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim * 2)  # mean and std for each action
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.hidden = None
    
    def forward(self, state, hidden=None):
        """
        Forward pass through network.
        
        Args:
            state: (batch, seq_len, state_dim) or (batch, state_dim)
            hidden: LSTM hidden state
        
        Returns:
            action_mean, action_std, value, hidden
        """
        # Add sequence dimension if needed
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
        
        # LSTM forward
        if hidden is None:
            lstm_out, hidden = self.lstm(state)
        else:
            lstm_out, hidden = self.lstm(state, hidden)
        
        # Use last timestep
        last_hidden = lstm_out[:, -1, :]
        
        # Policy (actor)
        policy_out = self.policy_head(last_hidden)
        action_dim = policy_out.shape[-1] // 2
        action_mean = policy_out[:, :action_dim]
        action_log_std = policy_out[:, action_dim:]
        action_std = torch.exp(torch.clamp(action_log_std, -20, 2))
        
        # Value (critic)
        value = self.value_head(last_hidden)
        
        return action_mean, action_std, value, hidden
    
    def reset_hidden(self, batch_size=1):
        """Reset LSTM hidden state."""
        self.hidden = None


class PPOAgent:
    """
    Proximal Policy Optimization agent with LSTM policy.
    """
    
    def __init__(self, state_dim, action_dim, hidden_size=128, 
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        
        # Policy network
        self.policy_net = LSTMPolicyNetwork(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Training buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Metrics
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
    
    def select_action(self, state, deterministic=False):
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, action_std, value, _ = self.policy_net(state_tensor)
        
        if deterministic:
            action = action_mean.squeeze().numpy()
        else:
            # Sample from Gaussian distribution
            dist = Normal(action_mean, action_std)
            action_tensor = dist.sample()
            action = action_tensor.squeeze().numpy()
            log_prob = dist.log_prob(action_tensor).sum(dim=-1)
            
            # Store for training
            self.states.append(state)
            self.actions.append(action)
            self.values.append(value.item())
            self.log_probs.append(log_prob.item())
        
        return action
    
    def store_reward(self, reward, done):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self, batch_size=64, n_epochs=10):
        """Update policy using PPO algorithm."""
        if len(self.states) < batch_size:
            return
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for epoch in range(n_epochs):
            # Get current policy predictions
            action_mean, action_std, values, _ = self.policy_net(states)
            
            # Compute log probabilities
            dist = Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            
            # Entropy bonus for exploration
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            
            self.policy_losses.append(policy_loss.item())
            self.value_losses.append(value_loss.item())
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def _compute_gae(self):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
            
            next_value = self.values[i]
        
        return returns, advantages
    
    def save(self, path):
        """Save agent."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Agent saved to {path}")
    
    def load(self, path):
        """Load agent."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Agent loaded from {path}")


class MultiAgentSystem:
    """
    Multi-agent system with separate agents for HVAC and lighting control.
    Implements cooperative multi-agent reinforcement learning.
    """
    
    def __init__(self, state_dim, hvac_action_dim=1, lighting_action_dim=1):
        # Separate agents for HVAC and lighting
        self.hvac_agent = PPOAgent(state_dim, hvac_action_dim, hidden_size=64)
        self.lighting_agent = PPOAgent(state_dim, lighting_action_dim, hidden_size=32)
        
        self.episode_rewards_hvac = []
        self.episode_rewards_lighting = []
    
    def select_actions(self, state, deterministic=False):
        """Select actions from both agents."""
        hvac_action = self.hvac_agent.select_action(state, deterministic)
        lighting_action = self.lighting_agent.select_action(state, deterministic)
        
        # Combine actions
        combined_action = np.concatenate([hvac_action, lighting_action])
        return combined_action
    
    def store_reward(self, reward, done):
        """Store rewards for both agents."""
        # In cooperative setting, both agents get same reward
        self.hvac_agent.store_reward(reward, done)
        self.lighting_agent.store_reward(reward, done)
    
    def update(self):
        """Update both agents."""
        self.hvac_agent.update()
        self.lighting_agent.update()
    
    def save(self, path_prefix):
        """Save both agents."""
        self.hvac_agent.save(f"{path_prefix}_hvac.pth")
        self.lighting_agent.save(f"{path_prefix}_lighting.pth")
    
    def load(self, path_prefix):
        """Load both agents."""
        self.hvac_agent.load(f"{path_prefix}_hvac.pth")
        self.lighting_agent.load(f"{path_prefix}_lighting.pth")


def train_ppo_agent(env, agent, n_episodes=500, max_steps=1000, 
                    update_frequency=2048, save_path='../models'):
    """
    Train PPO agent on building energy environment.
    
    Args:
        env: Building energy environment
        agent: PPO agent or MultiAgentSystem
        n_episodes: Number of training episodes
        max_steps: Max steps per episode
        update_frequency: Update policy every N steps
        save_path: Path to save models
    """
    os.makedirs(save_path, exist_ok=True)
    
    episode_rewards = []
    episode_energies = []
    episode_comforts = []
    total_steps = 0
    
    print("="*80)
    print("Training PPO Agent for Building Energy Control")
    print("="*80)
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_comfort = 0
        
        for step in range(max_steps):
            # Select action
            if isinstance(agent, MultiAgentSystem):
                action = agent.select_actions(state)
            else:
                action = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store reward
            agent.store_reward(reward, done)
            
            episode_reward += reward
            episode_energy += info['energy']
            episode_comfort += info['comfort_penalty']
            total_steps += 1
            
            state = next_state
            
            # Update policy
            if total_steps % update_frequency == 0:
                agent.update()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy / (step + 1))
        episode_comforts.append(episode_comfort / (step + 1))
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_energy = np.mean(episode_energies[-10:])
            avg_comfort = np.mean(episode_comforts[-10:])
            
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Energy: {avg_energy:.3f} kWh | "
                  f"Avg Comfort: {avg_comfort:.3f}")
        
        # Save best model
        if episode > 100 and episode_reward > max(episode_rewards[:-1]):
            if isinstance(agent, MultiAgentSystem):
                agent.save(os.path.join(save_path, 'best_multiagent'))
            else:
                agent.save(os.path.join(save_path, 'best_ppo_agent.pth'))
    
    print("="*80)
    print("Training Complete!")
    print("="*80)
    
    return episode_rewards, episode_energies, episode_comforts


if __name__ == "__main__":
    print("="*80)
    print("Testing RL Agents")
    print("="*80)
    
    # Test environment
    env = BuildingEnergyEnv()
    print(f"Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test PPO agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"\nTesting PPO Agent...")
    ppo_agent = PPOAgent(state_dim, action_dim)
    print(f"  Policy parameters: {sum(p.numel() for p in ppo_agent.policy_net.parameters()):,}")
    
    # Test multi-agent system
    print(f"\nTesting Multi-Agent System...")
    multi_agent = MultiAgentSystem(state_dim, hvac_action_dim=1, lighting_action_dim=1)
    print(f"  HVAC agent parameters: {sum(p.numel() for p in multi_agent.hvac_agent.policy_net.parameters()):,}")
    print(f"  Lighting agent parameters: {sum(p.numel() for p in multi_agent.lighting_agent.policy_net.parameters()):,}")
    
    print("\n" + "="*80)
    print("RL Agents Test Complete!")
    print("="*80)
