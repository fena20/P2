# Physics-Informed Deep Reinforcement Learning (PI-DRL) Framework
## Residential Building Energy Management with AMPds2 Dataset

**Target Journal:** Applied Energy (Q1)  
**Research Area:** Cyber-Physical Energy Systems

---

## 📋 Overview

This repository implements a comprehensive **Physics-Informed Deep Reinforcement Learning (PI-DRL)** framework for residential building HVAC control. The key novelty is the integration of physical constraints (specifically, compressor cycling penalties) into the reward function to prevent hardware degradation while optimizing energy costs.

### Key Contributions

1. **Physics-Informed Reward Function**: Multi-objective reward with cycling penalty to prevent short-cycling
2. **1st-Order RC Thermal Model**: Building physics integrated into the RL environment
3. **Publication-Ready Visualizations**: Four key figures following Applied Energy standards
4. **Reproducible Results**: Three "Golden Tables" for complete experimental documentation

---

## 🏗️ Framework Architecture

```
PI-DRL Framework
├── Part 1: Physics-Informed Environment (SmartHomeEnv)
│   ├── AMPds2 Data Handler (1-minute resolution)
│   ├── State Space: Box(6,) [T_in, T_out, Solar, Price, Action, Time]
│   ├── Action Space: Discrete(2) [OFF, ON]
│   ├── RC Thermal Model: T_in^{t+1} = T_in^t + Δt × [...]
│   └── Multi-Objective Reward: R = -(w₁·Cost + w₂·Discomfort + w₃·Cycling)
│
├── Part 2: PPO Agent (stable-baselines3)
│   ├── Policy Network: MLP [128, 128]
│   ├── Callbacks: SaveBestModel, EvalCallback
│   └── Hyperparameters optimized for HVAC control
│
└── Part 3: ResultVisualizer (Publication Quality)
    ├── Figure 1: System Heartbeat (Micro-Dynamics)
    ├── Figure 2: Control Policy Heatmap (Explainability)
    ├── Figure 3: Multi-Objective Radar Chart
    ├── Figure 4: Energy Carpet Plot (Load Shifting)
    └── Tables 1-3: Hyperparameters, Performance, Ablation
```

---

## 🔬 The Physics-Informed Approach

### 1st-Order RC Thermal Model

The building thermal dynamics are modeled using a first-order RC (Resistance-Capacitance) circuit:

$$T_{in}^{t+1} = T_{in}^t + \Delta t \times \left[\frac{T_{out} - T_{in}}{R \cdot C} + \frac{Q_{HVAC} + Q_{Solar}}{C}\right]$$

Where:
- $R$ = Thermal resistance (°C/kW) - represents building insulation
- $C$ = Thermal capacitance (kWh/°C) - represents thermal mass
- $Q_{HVAC}$ = HVAC heating/cooling power (kW)
- $Q_{Solar}$ = Solar heat gain (kW)

### Multi-Objective Reward Function (THE NOVELTY)

$$R = -\left(w_1 \cdot \text{Cost} + w_2 \cdot \text{Discomfort} + w_3 \cdot \text{Cycling\_Penalty}\right)$$

**The Cycling Penalty** is the key physics-informed constraint:
- Penalizes switching more than once every 15 minutes
- Exponential penalty for rapid cycling
- Prevents compressor short-cycling that leads to hardware degradation

```python
if action != self.last_action:
    if self.time_since_switch < self.min_on_time:
        cycling_penalty = np.exp(
            (self.min_on_time - self.time_since_switch) / self.min_on_time
        ) - 1
```

---

## 📊 Output Figures (Journal Standard)

### Figure 1: System Heartbeat (Micro-Dynamics)
- **Purpose**: Demonstrate short-cycling prevention
- **Format**: Dual-axis plot (Compressor State vs Indoor Temperature)
- **Comparison**: Baseline Thermostat (frequent switching) vs PI-DRL Agent (stable runs)

### Figure 2: Control Policy Heatmap
- **Purpose**: Explainability and demand response visualization
- **X-axis**: Hour of Day (0-23)
- **Y-axis**: Outdoor Temperature (-5 to 35°C)
- **Color**: Probability of Action=ON
- **Insight**: Shows agent stays OFF during peak pricing hours (17:00-20:00)

### Figure 3: Multi-Objective Radar Chart
- **Metrics**: Energy Cost, Comfort Violation, Equipment Cycles, Peak Load, Carbon
- **Comparison**: Baseline (100%) vs PI-DRL (normalized)
- **Style**: Filled polygons with transparency

### Figure 4: Energy Carpet Plot (Load Shifting)
- **Purpose**: Visualize load shifting behavior
- **X-axis**: Day of Year
- **Y-axis**: Hour of Day
- **Color**: HVAC Power Consumption
- **Goal**: Show "red zones" shifting away from peak hours

---

## 📋 Golden Tables

### Table 1: Simulation & Hyperparameters
| Parameter | Value |
|-----------|-------|
| Thermal Resistance R | 2.0 °C/kW |
| Thermal Capacitance C | 10.0 kWh/°C |
| HVAC Power Q_hvac | 3.5 kW |
| COP η | 3.0 |
| Cost Weight w₁ | 1.0 |
| Discomfort Weight w₂ | 2.0 |
| Cycling Weight w₃ | 0.5 |
| Min ON/OFF Time | 15 min |
| Learning Rate α | 3e-4 |
| Discount Factor γ | 0.99 |

### Table 2: Quantitative Performance Comparison
| Method | Total Cost ($) | Discomfort (°C·h) | Switching Count |
|--------|---------------|-------------------|-----------------|
| Baseline Thermostat | X.XX | XX.XX | XX |
| PI-DRL (Proposed) | X.XX | XX.XX | XX |

### Table 3: Ablation Study
| Configuration | Cost | Discomfort | Switches | Hardware Risk |
|--------------|------|------------|----------|---------------|
| Full PI-DRL | ✓ | ✓ | ✓ | LOW |
| w/o Cycling (w₃=0) | ✓ | ✓ | ✗ HIGH | Destroys compressor |
| w/o Discomfort (w₂=0) | ✓ | ✗ HIGH | ✓ | Poor comfort |

---

## 🚀 Quick Start

### Installation

```bash
pip install stable-baselines3 gymnasium pandas numpy matplotlib seaborn shap tqdm rich
```

### Basic Usage

```python
from src.pi_drl_framework import (
    SmartHomeEnv, AMPds2DataLoader,
    create_ppo_agent, train_agent, evaluate_agent,
    ResultVisualizer, BaselineThermostat
)

# 1. Load data and create environment
data_loader = AMPds2DataLoader(n_days=365)
data = data_loader.load_data()
env = SmartHomeEnv(data=data)

# 2. Create and train PPO agent
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

vec_env = DummyVecEnv([lambda: Monitor(env)])
agent = create_ppo_agent(vec_env)
agent, metrics = train_agent(vec_env, agent, total_timesteps=100000)

# 3. Evaluate and visualize
eval_env = SmartHomeEnv(data=data)
results = evaluate_agent(eval_env, agent, n_episodes=10)

visualizer = ResultVisualizer('./figures')
visualizer.figure1_system_heartbeat(baseline_data, pidrl_data)
visualizer.figure3_radar_chart(baseline_metrics, pidrl_metrics)
```

### Run Complete Pipeline

```bash
python3 -c "from src.pi_drl_framework import main; main()"
```

### Run Tests

```bash
python3 test_pi_drl_framework.py
```

---

## 📁 File Structure

```
/workspace
├── src/
│   ├── pi_drl_framework.py      # Main PI-DRL framework
│   └── shap_explainability.py   # SHAP analysis module
├── figures/
│   ├── fig1_system_heartbeat.png
│   ├── fig2_control_policy_heatmap.png
│   ├── fig3_radar_chart.png
│   ├── fig4_energy_carpet.png
│   ├── table1_hyperparameters.csv
│   ├── table2_performance_comparison.csv
│   └── table3_ablation_study.csv
├── models/
│   ├── pi_drl_best.zip
│   └── pi_drl_best_final.zip
├── test_pi_drl_framework.py     # Test suite
├── requirements.txt
└── PI_DRL_README.md             # This file
```

---

## 🔧 Customization

### Environment Parameters

```python
env = SmartHomeEnv(
    data=data,
    R=2.0,              # Thermal resistance (°C/kW)
    C=10.0,             # Thermal capacitance (kWh/°C)
    Q_hvac=3.5,         # HVAC power (kW)
    eta_hvac=3.0,       # COP
    T_setpoint=21.0,    # Target temperature (°C)
    T_deadband=1.0,     # Comfort deadband (°C)
    w1_cost=1.0,        # Energy cost weight
    w2_discomfort=2.0,  # Comfort violation weight
    w3_cycling=0.5,     # Cycling penalty weight
    min_on_time=15,     # Min ON time (minutes)
    episode_length=1440 # 1 day = 1440 minutes
)
```

### PPO Hyperparameters

```python
agent = create_ppo_agent(
    env=vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)
```

---

## 📚 References

### Dataset
- **AMPds2**: Almanac of Minutely Power Dataset (Version 2)
- Source: https://github.com/Fateme9977/P3/tree/main/data
- Resolution: 1-minute intervals

### Key Papers
1. Schulman, J., et al. "Proximal Policy Optimization Algorithms" (2017)
2. Vazquez-Canteli, J.R., Nagy, Z. "Reinforcement learning for demand response" (2019)

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@article{pi_drl_hvac_2024,
  title={Physics-Informed Deep Reinforcement Learning for Residential HVAC Control with Cycling Constraints},
  author={Your Name},
  journal={Applied Energy},
  year={2024}
}
```

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- Stable-Baselines3 team for the excellent RL library
- AMPds2 dataset contributors
- OpenAI Gymnasium for the environment framework
