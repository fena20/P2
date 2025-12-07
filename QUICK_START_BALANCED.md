# 🎯 Quick Start: Balanced PI-DRL HVAC Controller

## مشکل قبلی شما
- **نسخه اول:** Always OFF → Comfort فاجعه‌بار ❌
- **نسخه دوم:** Always ON → 0 cycles، هزینه بالا ❌

## ✅ راه‌حل: Balanced Reward Function

فایل جدید: **`src/pi_drl_hvac_controller_balanced.py`**

### تغییرات کلیدی

#### 1. Deadband-Aware Reward
```python
if T در deadband [19.5-22.5°C]:
    ✅ می‌تواند OFF باشد
    اما اگر T >= setpoint و ON است:
        penalty = 5.0  # غیرضروری!
```

#### 2. Reward Weights متعادل
```python
w_comfort_violation = 50.0   # شدید اما منطقی
w_unnecessary_on = 5.0       # 🔑 مانع Always ON
w_cost = 1.0                 # همیشه فعال
w_peak = 2.0                 # Peak shaving
```

## 🚀 نحوه استفاده

### گام 1: تنظیم data path
```python
# خط 58 فایل:
data_dir: str = r"C:\Users\FATEME\Downloads\dataverse_files"
```

### گام 2: اجرا
```bash
cd /workspace
python src/pi_drl_hvac_controller_balanced.py
```

### گام 3: بررسی نتایج
باید ببینید:
- ✅ Cycles > 0 (نه Always ON)
- ✅ Cost < Baseline
- ✅ Comfort < Baseline
- ✅ Avg peak power < 3.0 kW

## 📊 نتایج مورد انتظار

| Metric | Baseline | Target |
|--------|----------|--------|
| Cost | $1381 | $1240-1310 (-5 to -10%) |
| Comfort loss | 13076 | 8000-10000 (-25 to -40%) |
| Energy | 9770 kWh | 8800-9300 (-5 to -10%) |
| Cycles | ~100 | 80-150 |
| Peak power | ~2.5 kW | 1.8-2.2 (-20 to -30%) |

## 🔧 اگر هنوز مشکل داشت

### Always ON است؟
```python
w_unnecessary_on = 10.0  # افزایش از 5.0
```

### Always OFF است؟
```python
w_comfort_violation = 100.0  # افزایش از 50.0
```

### Cycling زیاد است؟
```python
w_switch = 1.0  # افزایش از 0.1
```

## 📚 مستندات کامل

1. `BALANCED_REWARD_STRATEGY.md` - توضیح کامل reward function
2. `COMPREHENSIVE_COMPARISON.md` - مقایسه سه نسخه
3. `COMFORT_FIRST_FIX_EXPLANATION.md` - چرا Always ON شد

## 🎯 فلسفه طراحی

> "ترموستات هوشمند = ترموستات معمولی + بهینه‌سازی هزینه"

Agent باید یاد بگیرد:
1. در deadband → مثل ترموستات (ON/OFF)
2. در deadband بالا → ترجیحاً OFF (صرفه‌جویی)
3. در peak hours + deadband → حتماً OFF (peak shaving)
4. خارج comfort → اورژانسی! (فوراً درست کن)

موفق باشید! 🚀
