#!/usr/bin/env python3
"""Step 4 – SHAP interpretation for the thermal intensity model."""

from __future__ import annotations

import argparse

import joblib

from recs2020_heatpump import default_pipeline_config
from recs2020_heatpump.io_utils import load_dataset
from recs2020_heatpump.shap_utils import (
    prepare_shap_inputs,
    save_shap_dependence_plots,
    save_shap_summary_plot,
    shap_importance_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis for XGBoost model.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_thermal_intensity.joblib",
        help="Serialized model filename.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of samples used for SHAP estimation.",
    )
    return parser.parse_args()


def select_feature_names(feature_names, base_names):
    selected = []
    for base in base_names:
        match = next((name for name in feature_names if base in name), None)
        if match:
            selected.append(match)
    return selected


def main() -> None:
    args = parse_args()
    config = default_pipeline_config()

    dataset_path = config.paths.data_cache / "recs2020_gas_heating.parquet"
    df = load_dataset(dataset_path)
    model_path = config.paths.models_dir / args.model_name
    pipeline = joblib.load(model_path)

    feature_cols = [
        col
        for col in config.model.numeric_features + config.model.categorical_features
        if col in df.columns
    ]
    X = df[feature_cols]

    (
        sample,
        X_processed,
        shap_values,
        base_values,
        feature_names,
    ) = prepare_shap_inputs(
        pipeline,
        X,
        sample_size=args.sample_size,
        random_state=config.model.random_state,
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    importance = shap_importance_table(shap_values, feature_names)
    table4_path = config.paths.tables_dir / "table4_shap_importance.csv"
    importance.to_csv(table4_path, index=False)

    figure6_path = config.paths.figures_dir / "figure6_shap_global.png"
    figure7_path = config.paths.figures_dir / "figure7_shap_dependence.png"

    save_shap_summary_plot(shap_values, X_processed, feature_names, figure6_path)
    selected_features = select_feature_names(
        feature_names,
        ["HDD65", "heated_floor_area_sqft", "envelope_class"],
    )
    if selected_features:
        save_shap_dependence_plots(
            shap_values,
            X_processed,
            feature_names,
            selected_features,
            figure7_path,
        )

    print("✅ SHAP analysis complete.")
    print(f"   Table 4 -> {table4_path}")
    print(f"   Figures -> {[figure6_path.name] + ([figure7_path.name] if selected_features else [])}")


if __name__ == "__main__":
    main()
