# خلاصه نتایج اجرای PI-DRL Framework

## ✅ فایل‌های تولید شده

### 📊 جداول (Tables)

همه جداول در پوشه `output/tables/` ذخیره شده‌اند:

1. **table1_simulation_hyperparameters.csv/.tex**
   - پارامترهای شبیه‌سازی و هایپرپارامترها
   - شامل: R, C, HVAC Power, Learning Rate, γ, w₁, w₂, w₃
   - فرمت: CSV + LaTeX

2. **table2_performance_comparison.csv/.tex**
   - مقایسه عملکرد کمی
   - Baseline vs PI-DRL Agent
   - بهبود: 29.2% کاهش هزینه، 36.9% کاهش ناراحتی، 60% کاهش سیکل‌ها

3. **table3_ablation_study.csv/.tex**
   - مطالعه Ablation
   - مقایسه: Baseline vs PI-DRL (با penalty) vs PI-DRL (بدون penalty)
   - نشان می‌دهد که Cycling Penalty ضروری است

### 📈 شکل‌ها (Figures)

همه شکل‌ها در پوشه `output/figures/` ذخیره شده‌اند:

1. **figure1_system_heartbeat.png**
   - نمایش میکرو-دینامیک سیستم
   - مقایسه Baseline vs PI-DRL
   - جلوگیری از short-cycling

2. **figure3_multi_objective_radar.png**
   - نمودار رادار چند-هدفه
   - 5 معیار: Energy Cost, Comfort Violation, Equipment Cycles, Peak Load, Carbon

3. **figure4_energy_carpet_plot.png**
   - نقشه انرژی (Load Shifting)
   - نمایش تغییر بار از ساعات پیک

### 📁 ساختار پوشه output

```
output/
├── tables/
│   ├── table1_simulation_hyperparameters.csv
│   ├── table1_simulation_hyperparameters.tex
│   ├── table2_performance_comparison.csv
│   ├── table2_performance_comparison.tex
│   ├── table3_ablation_study.csv
│   └── table3_ablation_study.tex
├── figures/
│   ├── figure1_system_heartbeat.png
│   ├── figure3_multi_objective_radar.png
│   └── figure4_energy_carpet_plot.png
└── models/
    └── monitor.csv
```

## 📊 نتایج نمونه

### مقایسه عملکرد (Table 2)

| Method | Total Cost ($) | Discomfort (Degree-Hours) | Switching Count (Cycles) | Peak Load (kW) |
|--------|----------------|---------------------------|-------------------------|----------------|
| Baseline Thermostat | 120.50 | 45.20 | 150 | 3.00 |
| PI-DRL Agent | 85.30 | 28.50 | 60 | 2.10 |
| **Improvement (%)** | **29.2%** | **36.9%** | **60.0%** | **30.0%** |

### مطالعه Ablation (Table 3)

| Method | Cost | Cycles | Hardware Risk |
|--------|------|--------|---------------|
| Baseline | $120.50 | 150 | Low |
| PI-DRL (with penalty) | $85.30 | 60 | Low ✅ |
| PI-DRL (without penalty) | $75.20 | 450 | **HIGH** ⚠️ |

**نتیجه:** Cycling Penalty برای محافظت از سخت‌افزار ضروری است!

## 🚀 نحوه استفاده

### مشاهده جداول

```bash
# مشاهده جداول CSV
cat output/tables/table1_simulation_hyperparameters.csv
cat output/tables/table2_performance_comparison.csv
cat output/tables/table3_ablation_study.csv

# یا باز کردن در Excel/LibreOffice
```

### مشاهده شکل‌ها

```bash
# باز کردن شکل‌ها
xdg-open output/figures/figure1_system_heartbeat.png
xdg-open output/figures/figure3_multi_objective_radar.png
xdg-open output/figures/figure4_energy_carpet_plot.png
```

## 📝 نکات مهم

1. **جداول LaTeX:** برای استفاده در مقاله LaTeX، از فایل‌های `.tex` استفاده کنید
2. **شکل‌ها:** همه شکل‌ها با کیفیت 300 DPI و فونت Times New Roman تولید شده‌اند
3. **فرمت:** همه فایل‌ها آماده استفاده در مقاله Applied Energy هستند

## 🔄 اجرای کامل

برای اجرای کامل آموزش و تولید همه نتایج:

```bash
python3 src/pi_drl_main.py --save_dir ./output
```

**نکته:** آموزش کامل ممکن است 2-4 ساعت زمان ببرد (بسته به سخت‌افزار)

---

**تاریخ تولید:** 2024-12-03
**وضعیت:** ✅ تولید موفق
