# PI-DRL Quick Start (5 Minutes)

## Step 1: Install Dependencies (1 minute)
```bash
pip install -r requirements.txt
```

## Step 2: Run Complete Pipeline (3-5 hours)
```bash
python src/main_pi_drl.py --mode full --timesteps 500000 --n-eval-episodes 20
```

## Step 3: View Results
```bash
# Summary report
cat outputs_pi_drl/SUMMARY_REPORT.txt

# Figures
ls outputs_pi_drl/figures/
# → fig1_system_heartbeat.png
# → fig2_control_policy_heatmap.png
# → fig3_multiobjective_radar.png
# → fig4_energy_carpet_plot.png

# Tables (CSV format)
cat outputs_pi_drl/tables/table2_performance_comparison.csv

# Tables (LaTeX format for manuscript)
cat outputs_pi_drl/tables/table2_performance_comparison.tex
```

---

## Quick Validation (5 minutes)

Want to test the pipeline before full training?

```bash
python src/main_pi_drl.py --mode full --timesteps 10000 --n-eval-episodes 3
```

This runs a minimal version to verify everything works.

---

## Alternative: Step-by-Step

### Train Only
```bash
python src/main_pi_drl.py --mode train --timesteps 500000
```

### Evaluate (requires trained model)
```bash
python src/main_pi_drl.py --mode evaluate --n-eval-episodes 20
```

### Ablation Study
```bash
python src/main_pi_drl.py --mode ablation
```

---

## Expected Outputs

### Console Output
```
==========================================
PHYSICS-INFORMED DEEP REINFORCEMENT LEARNING
==========================================

Training: 500000 steps
Evaluation: 20 episodes
Ablation: Enabled

RESULTS:
  Cost Reduction: 24.3%
  Cycle Reduction: 62.1%
  Comfort: Maintained (95%+ in zone)

All outputs saved to: ./outputs_pi_drl
```

### File Structure
```
outputs_pi_drl/
├── figures/              # 4 publication figures
├── tables/               # 3 tables (CSV + LaTeX)
├── models/               # Trained agents
└── SUMMARY_REPORT.txt    # Performance summary
```

---

## Common Questions

**Q: How long does training take?**  
A: 3-5 hours on CPU, 1-2 hours on GPU

**Q: Can I use my own data?**  
A: Yes! Edit `load_ampds2_mock_data()` in `pi_drl_environment.py`

**Q: How do I use GPU?**  
A: Add `--device cuda` to any command

**Q: What if I only want figures?**  
A: Use `--mode evaluate` (requires trained model)

---

## Next Steps

1. ✅ Review `SUMMARY_REPORT.txt` for performance metrics
2. ✅ Check figures in `outputs_pi_drl/figures/`
3. ✅ Import LaTeX tables into your manuscript
4. ✅ Read `PI_DRL_README.md` for customization options
5. ✅ Refer to `PI_DRL_IMPLEMENTATION_GUIDE.md` for full documentation

---

**Ready to publish! 🎓**
