# 📑 INDEX: راه‌حل مشکل Always-ON/Always-OFF در PI-DRL

## 🎯 شروع سریع

**فایل اصلی برای اجرا:**
```
src/pi_drl_hvac_controller_balanced.py
```

**مستندات:**
```
QUICK_START_BALANCED.md          ← شروع از اینجا! 
```

---

## 📂 ساختار فایل‌ها

### 1️⃣ کد Python (در پوشه `src/`)

| فایل | توضیح | وضعیت |
|------|-------|-------|
| `pi_drl_hvac_controller_balanced.py` | ✅ **راه‌حل نهایی** - Deadband-aware reward | استفاده کنید |
| `pi_drl_hvac_controller_fixed.py` | ❌ Always ON می‌شود | برای مقایسه |
| کد اصلی شما | ❌ Always OFF می‌شد | برای مقایسه |

### 2️⃣ مستندات (در روت)

#### مستندات کاربر (شروع از اینجا)
| فایل | محتوا | زمان مطالعه |
|------|-------|-------------|
| `QUICK_START_BALANCED.md` | راهنمای سریع استفاده | 5 دقیقه |
| `COMPREHENSIVE_COMPARISON.md` | مقایسه سه نسخه + troubleshooting | 15 دقیقه |
| `BALANCED_REWARD_STRATEGY.md` | توضیح کامل reward function | 20 دقیقه |

#### مستندات فنی (برای عمیق‌تر شدن)
| فایل | محتوا |
|------|-------|
| `COMFORT_FIRST_FIX_EXPLANATION.md` | چرا نسخه دوم Always ON شد |

---

## 🔑 مفاهیم کلیدی

### مشکل اصلی شما
```
نسخه 1 (Original):
  Cost weight بالا → Always OFF → Comfort فاجعه‌بار

نسخه 2 (Comfort-First fix):
  Comfort weight بالا → Always ON → 0 cycles، هزینه بالا
```

### راه‌حل (Balanced)
```python
# کلید موفقیت: Deadband-Aware Reward

if T در deadband [19.5-22.5]:
    if T >= setpoint و action=ON:
        penalty = w_unnecessary_on  # 🔑 مانع Always ON
    else:
        penalty = 0  # ✅ OK
        
elif T خارج comfort band:
    penalty = w_comfort_violation * (violation²)  # 🚨 اورژانسی
```

---

## 📊 نتایج مورد انتظار

| Metric | Baseline | Balanced Target | بهبود |
|--------|----------|-----------------|-------|
| Cost | $1381 | $1240-1310 | -5% to -10% ✅ |
| Comfort loss | 13076 | 8000-10000 | -25% to -40% ✅ |
| Energy | 9770 kWh | 8800-9300 | -5% to -10% ✅ |
| Cycles | ~100 | 80-150 | معقول ✅ |
| Peak power | ~2.5 kW | 1.8-2.2 kW | -20% to -30% ✅ |

---

## 🚀 دستور استفاده

### گام 1: تنظیم data path
```python
# ویرایش src/pi_drl_hvac_controller_balanced.py
# خط 58:
data_dir: str = r"C:\Users\FATEME\Downloads\dataverse_files"
```

### گام 2: اجرا
```bash
cd /workspace
python src/pi_drl_hvac_controller_balanced.py
```

### گام 3: بررسی خروجی
```
PHASE 3: Evaluation
  ↓
Baseline thermostat:
  Cost: $1381.72
  Comfort loss (band): 13076.29
  Cycles: ~100
  
PI-DRL agent:
  Cost: $XXXX  ← باید < 1381 باشد
  Comfort loss (band): XXXX  ← باید < 13076 باشد
  Cycles: XX  ← باید > 0 باشد!
```

---

## 🔧 Troubleshooting سریع

### مشکل: Agent هنوز Always ON است
```python
# در src/pi_drl_hvac_controller_balanced.py
# خط ~98:
w_unnecessary_on = 10.0  # افزایش از 5.0
```

### مشکل: Agent هنوز Always OFF است
```python
# خط ~96:
w_comfort_violation = 100.0  # افزایش از 50.0
```

### مشکل: Comfort loss هنوز بالا
```python
# خط ~88:
episode_length_days = 3  # افزایش از 2
total_timesteps = 300_000  # افزایش از 200_000
```

### مشکل: Peak shaving کار نمی‌کند
```python
# خط ~100:
w_peak = 5.0  # افزایش از 2.0
```

---

## 📖 ترتیب مطالعه پیشنهادی

1. **ابتدا:** `QUICK_START_BALANCED.md` (5 دقیقه)
2. **اجرا:** `src/pi_drl_hvac_controller_balanced.py`
3. **اگر مشکل داشت:** `COMPREHENSIVE_COMPARISON.md` → بخش Troubleshooting
4. **برای درک عمیق:** `BALANCED_REWARD_STRATEGY.md`

---

## 🎓 درس‌های کلیدی

### درس 1: Reward Shaping > Weight Tuning
```
❌ بد: فقط weights را تغییر دادن
✅ خوب: ساختار reward را تغییر دادن (deadband logic)
```

### درس 2: یادگیری از Baseline
```
Thermostat ساده:
  - در deadband → نگه می‌دارد (ON یا OFF)
  - خارج deadband → اصلاح می‌کند

RL باید این را یاد بگیرد و بهبود دهد!
```

### درس 3: "Comfort-first" ≠ "Always ON"
```
اشتباه: comfort penalty را خیلی بزرگ کردن
درست: unnecessary ON penalty اضافه کردن
```

### درس 4: Test Early
```
بعد از 10k steps → چک کنید
  - آیا cycling دارد؟
  - آیا فقط ON یا فقط OFF است؟
  
اگر بله → فوراً weights را تغییر دهید
```

---

## 💡 نکات طلایی

### ✅ کارهایی که باید انجام دهید
- Baseline را اول test کنید (برای مقایسه)
- Episode length >= 2 days
- Log files را حین training بررسی کنید
- فقط یک weight را در هر دفعه تغییر دهید

### ❌ کارهایی که نباید انجام دهید
- همه weights را باهم تغییر ندهید
- Training را خیلی زود stop نکنید (حداقل 100k steps)
- بدون test با baseline، نتیجه‌گیری نکنید
- Episode length < 1 day نگذارید

---

## 🆘 پشتیبانی

اگر بعد از اجرای کد، نتایج رضایت‌بخش نبود:

1. نتایج دقیق را یادداشت کنید:
   ```
   Cost: $XXXX (baseline: $1381)
   Comfort loss: XXXX (baseline: 13076)
   Cycles: XX (baseline: ~100)
   Avg peak power: X.XX kW
   ```

2. علائم را شناسایی کنید:
   - Cycles = 0 → Always ON
   - Avg power ≈ 0 → Always OFF
   - Cycles > 300 → Cycling زیاد

3. از جدول Troubleshooting استفاده کنید (در `COMPREHENSIVE_COMPARISON.md`)

4. اگر باز حل نشد، نتایج را به من بدهید تا fine-tune کنیم

---

## 📈 تکامل راه‌حل

```
مشکل اولیه (شما)
  ↓
Cost weight بالا
  ↓
Always OFF ❌
  ↓
Fix اول (من)
  ↓
Comfort weight بالا
  ↓
Always ON ❌
  ↓
Fix نهایی (من)
  ↓
Deadband-aware reward + Unnecessary ON penalty
  ↓
Balanced behavior ✅
```

---

## 🏁 خلاصه

**فایل اصلی:**
```
src/pi_drl_hvac_controller_balanced.py
```

**راهنمای سریع:**
```
QUICK_START_BALANCED.md
```

**Troubleshooting:**
```
COMPREHENSIVE_COMPARISON.md → بخش "Fine-Tuning"
```

**درک عمیق:**
```
BALANCED_REWARD_STRATEGY.md
```

موفق باشید! 🚀🎯
