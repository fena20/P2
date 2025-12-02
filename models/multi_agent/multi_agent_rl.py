"""
Multi-Agent Reinforcement Learning System
Separate agents for HVAC and Lighting control with coordination mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import gymnasium as gym

class MultiAgentPolicyNetwork(nn.Module):
    """
    Policy network for multi-agent system.
    Can be shared or separate for each agent.
    """
    
    def __init__(self, state_dim, action_dim, agent_id='hvac', 
                 hidden_size=128, num_layers=2, use_lstm=True):
        super(MultiAgentPolicyNetwork, self).__init__()
        
        self.agent_id = agent_id
        self.use_lstm = use_lstm
        
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=state_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            feature_size = hidden_size
        else:
            self.fc1 = nn.Linear(state_dim, hidden_size)
            feature_size = hidden_size
        
        # Policy head
        self.policy_mean = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        self.policy_std = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softplus()
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, hidden=None):
        """Forward pass."""
        if self.use_lstm:
            if len(state.shape) == 2:
                state = state.unsqueeze(1)
            lstm_out, hidden = self.lstm(state, hidden)
            features = lstm_out[:, -1, :]
        else:
            features = F.relu(self.fc1(state))
            hidden = None
        
        action_mean = self.policy_mean(features)
        action_std = self.policy_std(features) + 1e-5
        value = self.value_head(features)
        
        return action_mean, action_std, value, hidden
    
    def get_action(self, state, hidden=None, deterministic=False):
        """Sample action."""
        action_mean, action_std, value, hidden = self.forward(state, hidden)
        
        if deterministic:
            return action_mean, value, hidden
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value, hidden

class HVACAgent:
    """HVAC Control Agent."""
    
    def __init__(self, state_dim, action_dim, device='cuda'):
        self.agent_id = 'hvac'
        self.policy = MultiAgentPolicyNetwork(
            state_dim, action_dim, agent_id='hvac'
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.device = device
        
        # Action bounds: [setpoint (16-26°C), mode (0-2)]
        self.action_low = np.array([16.0, 0.0])
        self.action_high = np.array([26.0, 2.0])
    
    def scale_action(self, action):
        """Scale action to HVAC action space."""
        action = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        scaled = (action + 1) / 2 * (self.action_high - self.action_low) + self.action_low
        return scaled

class LightingAgent:
    """Lighting Control Agent."""
    
    def __init__(self, state_dim, action_dim, device='cuda'):
        self.agent_id = 'lighting'
        self.policy = MultiAgentPolicyNetwork(
            state_dim, action_dim, agent_id='lighting'
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.device = device
        
        # Action bounds: [brightness (0-100%), dimming (0-1)]
        self.action_low = np.array([0.0, 0.0])
        self.action_high = np.array([100.0, 1.0])
    
    def scale_action(self, action):
        """Scale action to lighting action space."""
        action = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        scaled = (action + 1) / 2 * (self.action_high - self.action_low) + self.action_low
        return scaled

class MultiAgentBuildingEnv:
    """
    Multi-agent building environment with separate HVAC and Lighting control.
    """
    
    def __init__(self, data: np.ndarray, energy_data: np.ndarray,
                 lighting_data: np.ndarray = None, max_steps=1000):
        self.data = data
        self.energy_data = energy_data
        self.lighting_data = lighting_data if lighting_data is not None else np.zeros_like(energy_data)
        self.max_steps = max_steps
        self.current_step = 0
        
        # State space for each agent
        # Shared state: [indoor_temp, outdoor_temp, humidity, hour, day_of_week, current_energy]
        self.state_dim = 6
        
        # Action spaces
        self.hvac_action_dim = 2  # [setpoint, mode]
        self.lighting_action_dim = 2  # [brightness, dimming]
        
    def reset(self):
        """Reset environment."""
        self.current_step = 0
        state = self._get_state()
        return {'hvac': state, 'lighting': state}
    
    def _get_state(self):
        """Get current state."""
        if self.current_step >= len(self.data):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        obs = np.zeros(self.state_dim, dtype=np.float32)
        
        # Extract features (simplified)
        temp_cols = [i for i in range(min(9, self.data.shape[1]))]
        if temp_cols:
            obs[0] = np.mean([self.data[self.current_step, i] for i in temp_cols[:9]])
        
        if self.data.shape[1] > 9:
            obs[1] = self.data[self.current_step, 9]
        
        rh_start = 10
        if self.data.shape[1] > rh_start:
            rh_cols = [i for i in range(rh_start, min(rh_start+9, self.data.shape[1]))]
            if rh_cols:
                obs[2] = np.mean([self.data[self.current_step, i] for i in rh_cols])
        
        if self.data.shape[1] > 20:
            obs[3] = self.data[self.current_step, -2] if self.data.shape[1] > 20 else 12
            obs[4] = self.data[self.current_step, -1] if self.data.shape[1] > 21 else 0
        
        if self.current_step < len(self.energy_data):
            obs[5] = self.energy_data[self.current_step]
        
        return obs
    
    def step(self, hvac_action: np.ndarray, lighting_action: np.ndarray):
        """
        Execute one step with both agents.
        
        Returns:
            next_states: Dict with states for each agent
            rewards: Dict with rewards for each agent
            done: Whether episode is done
            info: Additional info
        """
        # Get current state
        state = self._get_state()
        
        # Parse HVAC action
        hvac_setpoint = np.clip(hvac_action[0], 16.0, 26.0)
        hvac_mode = int(np.clip(hvac_action[1], 0, 2))
        
        # Parse Lighting action
        brightness = np.clip(lighting_action[0], 0.0, 100.0)
        dimming = np.clip(lighting_action[1], 0.0, 1.0)
        
        # Calculate HVAC energy
        indoor_temp = state[0]
        base_energy = state[5] if state[5] > 0 else 50.0
        
        if hvac_mode == 0:
            hvac_energy = 0.0
        elif hvac_mode == 1:
            temp_diff = max(0, indoor_temp - hvac_setpoint)
            hvac_energy = temp_diff * 10.0
        else:
            temp_diff = max(0, hvac_setpoint - indoor_temp)
            hvac_energy = temp_diff * 12.0
        
        # Calculate lighting energy
        lighting_energy = brightness * 0.5 * (1 - dimming * 0.3)  # Dimming reduces energy
        
        # Total energy
        total_energy = base_energy + hvac_energy + lighting_energy
        
        # Comfort penalty for HVAC
        temp_diff_from_setpoint = abs(indoor_temp - hvac_setpoint)
        hvac_comfort_penalty = temp_diff_from_setpoint * 2.0
        
        # Visual comfort for lighting (prefer natural light during day)
        hour = int(state[3])
        if 6 <= hour <= 18:  # Daytime
            optimal_brightness = max(0, 100 - (hour - 12) ** 2 * 2)
        else:  # Nighttime
            optimal_brightness = 30
        
        lighting_comfort_penalty = abs(brightness - optimal_brightness) * 0.1
        
        # Rewards (negative costs)
        hvac_reward = -(hvac_energy * 0.01 + hvac_comfort_penalty)
        lighting_reward = -(lighting_energy * 0.005 + lighting_comfort_penalty)
        
        # Update step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Next states
        next_state = self._get_state()
        next_states = {'hvac': next_state, 'lighting': next_state}
        
        info = {
            'total_energy': total_energy,
            'hvac_energy': hvac_energy,
            'lighting_energy': lighting_energy,
            'hvac_setpoint': hvac_setpoint,
            'brightness': brightness
        }
        
        return next_states, {'hvac': hvac_reward, 'lighting': lighting_reward}, done, info

class MultiAgentPPO:
    """
    Multi-Agent PPO trainer with independent learning.
    """
    
    def __init__(self, hvac_agent: HVACAgent, lighting_agent: LightingAgent,
                 gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.hvac_agent = hvac_agent
        self.lighting_agent = lighting_agent
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def compute_gae(self, rewards, values, next_values, dones, lambda_=0.95):
        """Compute GAE."""
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
    
    def update_agent(self, agent, states, actions, old_log_probs, rewards, 
                    values, dones, next_values, num_epochs=10, batch_size=64):
        """Update a single agent."""
        advantages = self.compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        
        states = torch.FloatTensor(states).to(agent.device)
        actions = torch.FloatTensor(actions).to(agent.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(agent.device)
        advantages = torch.FloatTensor(advantages).to(agent.device)
        returns = torch.FloatTensor(returns).to(agent.device)
        
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
                
                action_mean, action_std, value_pred, _ = agent.policy(batch_states)
                
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(value_pred, batch_returns)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=0.5)
                agent.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / (num_epochs * n_batches)

def train_multi_agent_system(env, hvac_agent, lighting_agent, num_episodes=1000,
                            max_steps=1000, update_frequency=2048):
    """Training loop for multi-agent system."""
    trainer = MultiAgentPPO(hvac_agent, lighting_agent)
    
    # Buffers for each agent
    hvac_buffers = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 
                   'values': [], 'dones': []}
    lighting_buffers = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [],
                       'values': [], 'dones': []}
    
    episode_rewards = {'hvac': [], 'lighting': []}
    
    hvac_hidden = None
    lighting_hidden = None
    
    for episode in range(num_episodes):
        states = env.reset()
        episode_rewards_curr = {'hvac': 0, 'lighting': 0}
        
        hvac_state_seq = deque([states['hvac']], maxlen=10)
        lighting_state_seq = deque([states['lighting']], maxlen=10)
        
        for step in range(max_steps):
            # HVAC agent
            hvac_state_tensor = torch.FloatTensor(np.array(hvac_state_seq)).unsqueeze(0).to(hvac_agent.device)
            hvac_action, hvac_log_prob, hvac_value, hvac_hidden = hvac_agent.policy.get_action(
                hvac_state_tensor, hvac_hidden
            )
            hvac_scaled_action = hvac_agent.scale_action(hvac_action.squeeze().cpu())
            
            # Lighting agent
            lighting_state_tensor = torch.FloatTensor(np.array(lighting_state_seq)).unsqueeze(0).to(lighting_agent.device)
            lighting_action, lighting_log_prob, lighting_value, lighting_hidden = lighting_agent.policy.get_action(
                lighting_state_tensor, lighting_hidden
            )
            lighting_scaled_action = lighting_agent.scale_action(lighting_action.squeeze().cpu())
            
            # Step environment
            next_states, rewards, done, info = env.step(hvac_scaled_action, lighting_scaled_action)
            
            # Store transitions
            hvac_buffers['states'].append(np.array(hvac_state_seq))
            hvac_buffers['actions'].append(hvac_action.squeeze().cpu().numpy())
            hvac_buffers['rewards'].append(rewards['hvac'])
            hvac_buffers['log_probs'].append(hvac_log_prob.item())
            hvac_buffers['values'].append(hvac_value.item())
            hvac_buffers['dones'].append(done)
            
            lighting_buffers['states'].append(np.array(lighting_state_seq))
            lighting_buffers['actions'].append(lighting_action.squeeze().cpu().numpy())
            lighting_buffers['rewards'].append(rewards['lighting'])
            lighting_buffers['log_probs'].append(lighting_log_prob.item())
            lighting_buffers['values'].append(lighting_value.item())
            lighting_buffers['dones'].append(done)
            
            # Update state sequences
            hvac_state_seq.append(next_states['hvac'])
            lighting_state_seq.append(next_states['lighting'])
            
            episode_rewards_curr['hvac'] += rewards['hvac']
            episode_rewards_curr['lighting'] += rewards['lighting']
            
            # Update agents if buffers are full
            if len(hvac_buffers['states']) >= update_frequency:
                # Update HVAC agent
                hvac_next_values = np.zeros_like(hvac_buffers['values'])
                hvac_next_values[:-1] = hvac_buffers['values'][1:]
                hvac_next_values[-1] = 0 if hvac_buffers['dones'][-1] else hvac_buffers['values'][-1]
                
                hvac_loss = trainer.update_agent(
                    hvac_agent,
                    np.array(hvac_buffers['states']),
                    np.array(hvac_buffers['actions']),
                    np.array(hvac_buffers['log_probs']),
                    np.array(hvac_buffers['rewards']),
                    np.array(hvac_buffers['values']),
                    np.array(hvac_buffers['dones']),
                    hvac_next_values
                )
                
                # Update Lighting agent
                lighting_next_values = np.zeros_like(lighting_buffers['values'])
                lighting_next_values[:-1] = lighting_buffers['values'][1:]
                lighting_next_values[-1] = 0 if lighting_buffers['dones'][-1] else lighting_buffers['values'][-1]
                
                lighting_loss = trainer.update_agent(
                    lighting_agent,
                    np.array(lighting_buffers['states']),
                    np.array(lighting_buffers['actions']),
                    np.array(lighting_buffers['log_probs']),
                    np.array(lighting_buffers['rewards']),
                    np.array(lighting_buffers['values']),
                    np.array(lighting_buffers['dones']),
                    lighting_next_values
                )
                
                # Clear buffers
                for buffer in [hvac_buffers, lighting_buffers]:
                    for key in buffer:
                        buffer[key] = []
                
                print(f"Episode {episode}, Step {step}, "
                      f"HVAC Loss: {hvac_loss:.4f}, Lighting Loss: {lighting_loss:.4f}")
            
            if done:
                break
        
        episode_rewards['hvac'].append(episode_rewards_curr['hvac'])
        episode_rewards['lighting'].append(episode_rewards_curr['lighting'])
        
        if (episode + 1) % 100 == 0:
            avg_hvac_reward = np.mean(episode_rewards['hvac'][-100:])
            avg_lighting_reward = np.mean(episode_rewards['lighting'][-100:])
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"HVAC Avg Reward: {avg_hvac_reward:.2f}, "
                  f"Lighting Avg Reward: {avg_lighting_reward:.2f}")
    
    return episode_rewards
