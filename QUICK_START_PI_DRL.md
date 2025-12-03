# Quick Start Guide - PI-DRL Framework

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Complete Pipeline

```bash
python3 src/pi_drl_main.py
```

This will:
- ✅ Create environment with synthetic AMPds2 data
- ✅ Train PPO agent (200K timesteps, ~2-4 hours on CPU)
- ✅ Evaluate PI-DRL agent
- ✅ Evaluate baseline controller
- ✅ Run ablation study
- ✅ Generate 4 publication figures
- ✅ Generate 3 publication tables

### Step 3: Check Results

```bash
ls results/pi_drl/
# models/    figures/    tables/
```

---

## 📊 Quick Customization

### Use Real AMPds2 Data

```bash
python3 src/pi_drl_main.py --data_path /path/to/ampds2_data.csv
```

### Skip Training (Use Existing Model)

```bash
python3 src/pi_drl_main.py --no_train
```

### Skip Ablation Study

```bash
python3 src/pi_drl_main.py --no_ablation
```

### Custom Training Duration

```bash
python3 src/pi_drl_main.py --timesteps 500000
```

---

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `src/pi_drl_main.py` | **Main entry point** - Run this! |
| `src/pi_drl_environment.py` | Physics-informed environment |
| `src/pi_drl_training.py` | PPO training script |
| `src/pi_drl_visualization.py` | Figure generation |
| `src/pi_drl_tables.py` | Table generation |

---

## 📈 Expected Outputs

### Figures (in `results/pi_drl/figures/`)
1. `figure1_system_heartbeat.png` - Micro-dynamics (2-hour zoom)
2. `figure2_control_policy_heatmap.png` - Policy explainability
3. `figure3_multi_objective_radar.png` - Multi-objective comparison
4. `figure4_energy_carpet_plot.png` - Load shifting visualization

### Tables (in `results/pi_drl/tables/`)
1. `table1_simulation_hyperparameters.csv` - Reproducibility parameters
2. `table2_performance_comparison.csv` - Quantitative results
3. `table3_ablation_study.csv` - Physics-informed validation

---

## 🔧 Python API Usage

### Basic Example

```python
from src.pi_drl_main import main

main(
    data_path=None,
    train_agent=True,
    run_ablation=True,
    total_timesteps=200000,
    n_eval_episodes=10,
    save_dir="./results/pi_drl"
)
```

### Custom Environment

```python
from src.pi_drl_environment import SmartHomeEnv

env = SmartHomeEnv(
    R=0.05,              # Thermal resistance
    C=0.5,               # Thermal capacitance
    min_cycle_time=15,   # 15-minute minimum cycle
    w3=5.0               # Cycling penalty weight
)
```

### Train Only

```python
from src.pi_drl_training import train_ppo_agent

model = train_ppo_agent(
    env=env,
    total_timesteps=200000,
    save_dir="./models"
)
```

---

## ⚡ Quick Test

Test all components:

```bash
python3 src/test_pi_drl.py
```

---

## 📚 Full Documentation

- **PI_DRL_README.md** - Comprehensive guide
- **IMPLEMENTATION_SUMMARY.md** - Technical details

---

## 🆘 Troubleshooting

### Issue: "Module not found"
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
- Reduce `batch_size` in training parameters
- Use CPU: Set `device='cpu'` in training

### Issue: "Training too slow"
- Reduce `total_timesteps` for testing
- Use GPU if available
- Reduce `n_steps` parameter

---

## ✅ Checklist

Before running:
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Python 3.8+ available
- [ ] ~5GB disk space for results
- [ ] 2-4 hours for full training (CPU)

After running:
- [ ] Check `results/pi_drl/figures/` for 4 figures
- [ ] Check `results/pi_drl/tables/` for 3 tables
- [ ] Review `results/pi_drl/models/` for saved models

---

**Ready to go!** 🎉

Run: `python3 src/pi_drl_main.py`
