# راهنمای کامل اجرا در Google Colab (فارسی)

## روش 1: استفاده از نوت‌بوک آماده (پیشنهادی)

### مرحله 1: آپلود نوت‌بوک
1. فایل `Colab_Quick_Start.ipynb` را در Google Colab باز کنید
2. یا فایل را در Google Drive آپلود کنید و از Colab باز کنید

### مرحله 2: آپلود فایل‌های کد
دو روش دارید:

#### روش A: آپلود مستقیم از کامپیوتر
1. در Colab، روی آیکون 📁 (Files) کلیک کنید
2. روی آیکون آپلود کلیک کنید
3. تمام فایل‌های پوشه `src/` را آپلود کنید:
   - `src/__init__.py`
   - `src/data_harmonization.py`
   - `src/feature_engineering.py`
   - `src/digital_twin.py`
   - `src/optimization.py`
   - `src/mcdm.py`
   - `src/utils.py`
4. فایل‌های `main.py` و `config.yaml` را هم آپلود کنید

#### روش B: استفاده از Google Drive
1. تمام فایل‌های پروژه را در Google Drive آپلود کنید
2. در Colab، کد زیر را اجرا کنید:
```python
from google.colab import drive
drive.mount('/content/drive')

# کپی فایل‌ها از Drive
!cp -r /content/drive/MyDrive/your_folder_name/* .
```

### مرحله 3: اجرا
سلول‌های نوت‌بوک را به ترتیب اجرا کنید (Shift + Enter)

---

## روش 2: استفاده از Git (اگر repository دارید)

### مرحله 1: کلون کردن repository
```python
!git clone https://github.com/yourusername/your-repo-name.git
%cd your-repo-name
```

### مرحله 2: نصب کتابخانه‌ها
```python
!pip install -r requirements.txt
```

### مرحله 3: اجرا
```python
!python main.py
```

---

## روش 3: ایجاد فایل‌ها مستقیماً در Colab

اگر نمی‌خواهید فایل‌ها را آپلود کنید، می‌توانید محتوای هر فایل را مستقیماً در Colab ایجاد کنید:

### مرحله 1: ایجاد ساختار
```python
import os
os.makedirs('src', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
```

### مرحله 2: ایجاد فایل‌ها
برای هر فایل Python، یک سلول جدید ایجاد کنید و محتوا را کپی کنید:

```python
# ایجاد src/data_harmonization.py
with open('src/data_harmonization.py', 'w', encoding='utf-8') as f:
    f.write('''# محتوای فایل data_harmonization.py را اینجا قرار دهید
    ''')
```

**نکته:** این روش برای فایل‌های بزرگ خیلی طولانی می‌شود.

---

## نکات مهم

### 1. استفاده از GPU (اختیاری)
برای سرعت بیشتر در آموزش مدل:
- Runtime → Change runtime type → GPU

### 2. محدودیت زمان اجرا
- Colab رایگان: 12 ساعت
- Colab Pro: 24 ساعت
- اگر اجرا طولانی شد، نتایج را ذخیره کنید

### 3. ذخیره نتایج
```python
# ذخیره در Google Drive
!cp -r results/ /content/drive/MyDrive/
!cp -r figures/ /content/drive/MyDrive/

# یا دانلود مستقیم
from google.colab import files
files.download('results/optimization_results.json')
```

### 4. مشکلات رایج

#### مشکل: ModuleNotFoundError
**راه حل:** مطمئن شوید فایل‌های `src/` را آپلود کرده‌اید

#### مشکل: FileNotFoundError برای config.yaml
**راه حل:** فایل `config.yaml` را در root directory آپلود کنید

#### مشکل: Out of Memory
**راه حل:** 
- اندازه dataset را کاهش دهید
- `population_size` و `n_generations` را در `config.yaml` کاهش دهید

---

## ساختار فایل‌های مورد نیاز

```
.
├── main.py                    ✅ ضروری
├── config.yaml                ✅ ضروری
├── src/
│   ├── __init__.py           ✅ ضروری
│   ├── data_harmonization.py ✅ ضروری
│   ├── feature_engineering.py ✅ ضروری
│   ├── digital_twin.py       ✅ ضروری
│   ├── optimization.py       ✅ ضروری
│   ├── mcdm.py              ✅ ضروری
│   └── utils.py             ✅ ضروری
└── Colab_Quick_Start.ipynb   ✅ پیشنهادی
```

---

## مثال اجرای سریع

```python
# 1. نصب کتابخانه‌ها
!pip install -q numpy pandas scikit-learn scipy matplotlib seaborn pymoo xgboost lightgbm catboost requests joblib tqdm pyyaml

# 2. ایجاد ساختار
import os
os.makedirs('src', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# 3. آپلود فایل‌ها (از طریق Files panel در Colab)

# 4. اجرا
import sys
sys.path.insert(0, '.')
from main import BuildingEnergyOptimizationPipeline

pipeline = BuildingEnergyOptimizationPipeline()
results = pipeline.run_complete_pipeline()
```

---

## پشتیبانی

اگر مشکلی پیش آمد:
1. مطمئن شوید تمام فایل‌ها را آپلود کرده‌اید
2. خطاها را بررسی کنید
3. ساختار پوشه‌ها را چک کنید

**موفق باشید! 🚀**
