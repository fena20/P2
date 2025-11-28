"""
اسکریپت سریع برای راه‌اندازی در Google Colab
این فایل را در Colab اجرا کنید تا همه چیز آماده شود
"""

print("🚀 شروع راه‌اندازی پروژه در Google Colab...")
print("=" * 60)

# 1. نصب کتابخانه‌ها
print("\n📦 مرحله 1: نصب کتابخانه‌ها...")
import subprocess
import sys

packages = [
    "numpy>=1.24.0", "pandas>=2.0.0", "scikit-learn>=1.3.0", 
    "scipy>=1.10.0", "matplotlib>=3.7.0", "seaborn>=0.12.0",
    "pymoo>=0.6.0", "xgboost>=2.0.0", "lightgbm>=4.0.0", 
    "catboost>=1.2.0", "requests>=2.31.0", "joblib>=1.3.0", 
    "tqdm>=4.65.0", "pyyaml>=6.0"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✅ کتابخانه‌ها نصب شدند!")

# 2. ایجاد ساختار پوشه‌ها
print("\n📁 مرحله 2: ایجاد ساختار پوشه‌ها...")
import os

directories = ['src', 'data', 'results', 'figures']
for dir_name in directories:
    os.makedirs(dir_name, exist_ok=True)
    print(f"  ✅ پوشه {dir_name}/ ایجاد شد")

# 3. بررسی فایل‌های مورد نیاز
print("\n🔍 مرحله 3: بررسی فایل‌های مورد نیاز...")
required_files = [
    'src/__init__.py',
    'src/data_harmonization.py',
    'src/feature_engineering.py',
    'src/digital_twin.py',
    'src/optimization.py',
    'src/mcdm.py',
    'src/utils.py',
    'main.py'
]

missing_files = []
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"  ✅ {file_path}")
    else:
        print(f"  ❌ {file_path} - یافت نشد!")
        missing_files.append(file_path)

if missing_files:
    print(f"\n⚠️  {len(missing_files)} فایل یافت نشد!")
    print("لطفا فایل‌های زیر را آپلود کنید:")
    for f in missing_files:
        print(f"  - {f}")
    print("\nبرای آپلود:")
    print("1. روی آیکون 📁 Files در سمت چپ کلیک کنید")
    print("2. روی آیکون ⬆️ Upload کلیک کنید")
    print("3. فایل‌ها را آپلود کنید")
else:
    print("\n✅ تمام فایل‌ها موجود هستند!")

# 4. ایجاد config.yaml اگر وجود ندارد
print("\n⚙️  مرحله 4: بررسی config.yaml...")
if not os.path.exists('config.yaml'):
    print("  ⚠️  config.yaml یافت نشد. در حال ایجاد...")
    config_content = """# Configuration file for Multi-Objective Building Energy Optimization Framework

data:
  source_url: "https://raw.githubusercontent.com/Fateme9977/P2/main/energydata_complete.csv"
  local_path: "data/energydata_complete.csv"
  target_column: "Appliances"
  train_datasets: [1, 2, 3]
  test_datasets: [4, 5, 6]

feature_engineering:
  enthalpy:
    cp_air: 1.006
    hfg: 2501.0
  ema_alpha: 0.3
  resample_freq: "1H"
  standardize: true

digital_twin:
  base_models:
    - name: "xgboost"
      params:
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.1
        random_state: 42
    - name: "lightgbm"
      params:
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.1
        random_state: 42
    - name: "catboost"
      params:
        iterations: 100
        depth: 6
        learning_rate: 0.1
        random_seed: 42
  meta_model:
    name: "ridge"
    params:
      alpha: 1.0
  cv_folds: 5
  random_state: 42

optimization:
  algorithm: "NSGA2"
  population_size: 50
  n_generations: 100
  decision_variables:
    T_set_heat:
      lower_bound: 18.0
      upper_bound: 24.0
    T_set_cool:
      lower_bound: 20.0
      upper_bound: 26.0
  deadband_min: 2.0
  discomfort:
    temp_weight: 1.0
    rh_weight: 0.5
    optimal_temp: 22.0
    optimal_rh: 50.0

mcdm:
  method: "TOPSIS"
  weights:
    energy: 0.5
    discomfort: 0.5

output:
  results_dir: "results"
  figures_dir: "figures"
  save_pareto_front: true
  save_optimization_history: true
"""
    with open('config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_content)
    print("  ✅ config.yaml ایجاد شد")
else:
    print("  ✅ config.yaml موجود است")

# 5. خلاصه
print("\n" + "=" * 60)
print("📋 خلاصه:")
print("=" * 60)

if not missing_files:
    print("\n✅ همه چیز آماده است!")
    print("\n🚀 برای اجرای pipeline:")
    print("""
import sys
sys.path.insert(0, '.')
from main import BuildingEnergyOptimizationPipeline

pipeline = BuildingEnergyOptimizationPipeline(config_path='config.yaml')
results = pipeline.run_complete_pipeline()
    """)
else:
    print("\n⚠️  لطفا ابتدا فایل‌های گمشده را آپلود کنید")
    print("سپس این اسکریپت را دوباره اجرا کنید")

print("\n" + "=" * 60)
print("✅ راه‌اندازی کامل شد!")
print("=" * 60)
