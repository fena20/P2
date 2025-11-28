"""
Setup script for Google Colab
این اسکریپت تمام فایل‌های لازم را در Colab ایجاد می‌کند
"""

import os
import urllib.request
import json

def create_directory_structure():
    """ایجاد ساختار پوشه‌ها"""
    directories = ['src', 'data', 'results', 'figures']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
    print("✅ ساختار پوشه‌ها ایجاد شد")

def download_file(url, save_path):
    """دانلود فایل از URL"""
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ فایل دانلود شد: {save_path}")
        return True
    except Exception as e:
        print(f"❌ خطا در دانلود {url}: {e}")
        return False

def create_config_file():
    """ایجاد فایل config.yaml"""
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
    print("✅ فایل config.yaml ایجاد شد")

if __name__ == "__main__":
    print("🚀 شروع راه‌اندازی پروژه در Google Colab...")
    create_directory_structure()
    create_config_file()
    print("\n✅ راه‌اندازی کامل شد!")
    print("\n📝 مراحل بعدی:")
    print("1. فایل‌های src/*.py را از repository دانلود کنید")
    print("2. یا از دستور git clone استفاده کنید")
    print("3. سپس main.py را اجرا کنید")
