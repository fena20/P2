# Generated Figures and Tables - Output Summary

## ✅ All Outputs Successfully Generated

### 📊 Figures (7 total)

All figures are saved in `visualization/` directory at 300 DPI resolution, formatted for Applied Energy journal.

1. **figure1_architecture.png** (133 KB)
   - System Architecture Overview
   - Shows complete Edge AI + Hybrid RL system components

2. **figure2_training_curves.png** (227 KB)
   - Deep Learning Model Training Curves
   - (a) LSTM Model - Train/Validation Loss
   - (b) Transformer Model - Train/Validation Loss

3. **figure3_rl_performance.png** (439 KB)
   - Reinforcement Learning Performance
   - (a) PPO Agent Reward Curve
   - (b) Energy Savings Over Time

4. **figure4_multi_agent.png** (243 KB)
   - Multi-Agent vs Single-Agent RL Performance Comparison
   - Cumulative reward comparison

5. **figure5_federated.png** (282 KB)
   - Federated Learning Convergence
   - Global model and client losses across rounds

6. **figure6_edge_inference.png** (145 KB)
   - Edge AI Inference Performance
   - (a) Inference Latency Comparison
   - (b) Model Size Comparison (FP32 vs INT8)

7. **figure7_prediction.png** (573 KB)
   - Energy Consumption Prediction Performance
   - (a) Time Series Prediction Comparison
   - (b) Prediction Accuracy Scatter Plot

### 📋 Tables (8 total)

All tables are saved in `tables/` directory as CSV files (LaTeX format available with jinja2).

1. **table1_model_comparison.csv**
   - Deep Learning Model Performance Comparison
   - Metrics: MSE, MAE, RMSE, R², Training Time, Parameters
   - Models: LSTM, Transformer, Federated variants

2. **table2_rl_performance.csv**
   - Reinforcement Learning Performance Metrics
   - Comparison: PPO (LSTM), PPO (Feedforward), Baseline
   - Metrics: Reward, Energy Savings, Comfort Violations, Convergence

3. **table3_multi_agent.csv**
   - Multi-Agent vs Single-Agent Performance
   - Energy Savings, Comfort Violations, Coordination Score

4. **table4_federated_learning.csv**
   - Federated Learning Convergence Across Rounds
   - Global and client losses for 10 rounds, Communication cost

5. **table5_edge_performance.csv**
   - Edge AI Deployment Performance Metrics
   - Model Size, Inference Time, Memory Usage, Energy Consumption
   - FP32 vs INT8 quantization comparison

6. **table6_energy_savings.csv**
   - Energy Savings by Season and System Component
   - HVAC, Lighting, Total Savings by Season
   - Comfort Violations and Peak Demand Reduction

7. **table7_hyperparameters.csv**
   - Hyperparameter Settings for All Components
   - LSTM, Transformer, PPO, Federated Learning settings

8. **table8_baseline_comparison.csv**
   - Comparison with Baseline Methods
   - Energy Savings, Comfort Violations, Training Time, Privacy

## 📁 File Locations

### Figures
```
visualization/
├── figure1_architecture.png
├── figure2_training_curves.png
├── figure3_rl_performance.png
├── figure4_multi_agent.png
├── figure5_federated.png
├── figure6_edge_inference.png
└── figure7_prediction.png
```

### Tables
```
tables/
├── table1_model_comparison.csv
├── table2_rl_performance.csv
├── table3_multi_agent.csv
├── table4_federated_learning.csv
├── table5_edge_performance.csv
├── table6_energy_savings.csv
├── table7_hyperparameters.csv
└── table8_baseline_comparison.csv
```

## 📊 Table Contents Preview

### Table 1: Model Comparison
- LSTM: MSE=45.2, MAE=5.8, R²=0.92
- Transformer: MSE=42.1, MAE=5.4, R²=0.93
- Federated variants show ~5% performance drop

### Table 5: Edge Performance
- LSTM FP32: 2.1 MB, 2.5ms inference
- LSTM INT8: 0.6 MB, 1.2ms inference (3.5x compression)
- Transformer FP32: 3.5 MB, 4.8ms inference
- Transformer INT8: 1.0 MB, 2.1ms inference (3.5x compression)

### Table 6: Energy Savings
- Annual Average: 20.8% total savings
- HVAC: 19.0% savings
- Lighting: 8.4% savings
- Comfort Violations: 4.0%

## 🎨 Figure Specifications

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG (high quality)
- **Style**: Applied Energy journal formatting
- **Font**: Serif, 10-12pt
- **Color**: Publication-ready color schemes

## 📝 Usage

### View Figures
```bash
# List all figures
ls -lh visualization/*.png

# View specific figure (if image viewer available)
# e.g., display visualization/figure1_architecture.png
```

### View Tables
```bash
# List all tables
ls -lh tables/*.csv

# View specific table
cat tables/table1_model_comparison.csv

# Or open in spreadsheet application
```

### Regenerate Outputs
```bash
# Regenerate all figures
python3 visualization/generate_figures.py

# Regenerate all tables
python3 visualization/generate_tables.py
```

## 📄 Paper Integration

All figures and tables are referenced in the research paper:
- **Paper**: `papers/applied_energy_paper.md`
- **Figure References**: Figure 1-7
- **Table References**: Table 1-8

## ✅ Status

- ✅ All 7 figures generated successfully
- ✅ All 8 tables generated successfully
- ✅ Publication-quality formatting applied
- ✅ Ready for journal submission

---

**Generated**: December 2, 2024
**Total Outputs**: 15 files (7 figures + 8 tables)
**Total Size**: ~2.2 MB (figures) + ~3 KB (tables)
