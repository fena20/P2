# Edge AI with Hybrid RL and Deep Learning for Occupant-Centric Energy Optimization

## Research Project for Applied Energy Journal Submission

This repository contains the complete implementation of a novel Edge AI framework integrating hybrid reinforcement learning (RL) with deep learning for occupant-centric energy optimization in residential buildings.

## 📋 Project Overview

The project implements:
- **LSTM-based Deep Learning** for energy consumption prediction
- **PPO Reinforcement Learning** for adaptive HVAC and lighting control
- **Multi-Agent RL** for coordinated subsystem optimization
- **Federated Learning** for privacy-preserving model training
- **Edge AI Deployment** using TorchScript for embedded systems
- **PMV/PPD Thermal Comfort** modeling based on ISO 7730

## 📁 Repository Structure

```
├── main_energy_optimization.py     # Main implementation script
├── paper_draft.md                  # Complete research paper draft (~7000 words)
├── energydata_complete.csv         # Residential energy dataset
├── EDA_INSIGHTS_SUMMARY.md         # Exploratory data analysis summary
│
├── Tables/
│   ├── table1_summary_statistics.csv    # Dataset summary statistics
│   ├── table2_model_comparison.csv      # DL model performance comparison
│   ├── table3_federated_comparison.csv  # Federated vs centralized training
│   └── table4_scenario_comparison.csv   # Control scenario comparison
│
├── Figures/
│   ├── fig0_system_architecture.png     # System architecture diagram
│   ├── fig1_prediction_scatter.png      # Predicted vs actual energy
│   ├── fig2_savings_comparison.png      # Energy/cost/comfort bar charts
│   ├── fig3_time_series.png             # Time-series comparison
│   ├── fig4_sensitivity_heatmap.png     # Sensitivity analysis
│   └── fig5_pareto_front.png            # Pareto front visualization
│
├── edge_model.pt                   # TorchScript model for edge deployment
└── energy_optimization_project.zip # Complete project archive
```

## 🚀 Quick Start

### Requirements

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn gymnasium stable-baselines3 scipy
```

### Running the Project

```bash
python3 main_energy_optimization.py
```

This will:
1. Load and preprocess the energy consumption dataset
2. Train LSTM and Transformer models for energy prediction
3. Train PPO agents for single and multi-agent control
4. Run federated learning simulation across 3 clients
5. Export model to TorchScript for edge deployment
6. Compare baseline, MPC, and hybrid RL scenarios
7. Generate all publication-quality figures and tables

## 📊 Key Results

| Metric | Value |
|--------|-------|
| LSTM Prediction R² | 0.284 |
| Hybrid RL Energy Savings | 5.4% |
| Mean PPD (Thermal Comfort) | 12.2% |
| Edge Inference Latency | 0.569 ms |
| Federated Learning R² | 0.375 |

## 🔬 Methodology Highlights

### Deep Learning Architecture
- LSTM with 128 hidden units, 2 layers
- Sequence length: 24 time steps (4 hours)
- Early stopping with patience of 10 epochs

### Reinforcement Learning
- PPO algorithm with MLP policy
- State: [T_indoor, T_out, RH_indoor, energy_pred, hour_sin, hour_cos, PPD]
- Actions: HVAC setpoint adjustment {-2, -1, 0, +1, +2}°C
- Reward: -energy_cost - comfort_penalty

### Federated Learning
- FedAvg aggregation across 3 clients
- 5 training rounds
- Zero raw data transmission

### Thermal Comfort
- PMV/PPD model based on ISO 7730
- Metabolic rate: 1.2 met
- Clothing insulation: 0.7 clo

## 📄 Paper Draft

The complete research paper draft (`paper_draft.md`) includes:
- Abstract (200 words)
- Introduction with literature review
- Detailed methodology
- Comprehensive results analysis
- Discussion and limitations
- 25+ academic references

## 🎨 Generated Figures

All figures are publication-quality (300 DPI) with:
- Muted color palette suitable for academic journals
- Clear axis labels with units
- Legible fonts and tight layouts
- Professional legends and annotations

## 🔐 Privacy Features

The federated learning implementation ensures:
- No raw energy consumption data leaves local devices
- Only model parameters transmitted during training
- Privacy-preserving collaborative learning

## ⚡ Edge AI Deployment

The TorchScript model (`edge_model.pt`) enables:
- Sub-millisecond inference latency
- Cloud-independent operation
- Deployment on embedded systems (Raspberry Pi, PLCs)

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{edge_ai_building_energy_2024,
  title={Edge AI with Hybrid Reinforcement Learning and Deep Learning 
         for Occupant-Centric Optimization of Energy Consumption 
         in Residential Buildings},
  journal={Applied Energy},
  year={2024}
}
```

## 📝 License

This project is released for academic and research purposes.

## 🙏 Acknowledgments

- Dataset: Appliances Energy Prediction Dataset (Luis M. Candanedo)
- Libraries: PyTorch, Stable-Baselines3, Gymnasium, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
