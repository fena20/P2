#!/usr/bin/env python3
"""Step 3 – Train XGBoost thermal intensity model."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from recs2020_heatpump import default_pipeline_config
from recs2020_heatpump.io_utils import load_dataset
from recs2020_heatpump.modeling import (
    error_breakdown,
    save_model,
    split_dataset,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost thermal intensity model.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_thermal_intensity.joblib",
        help="Filename for the serialized pipeline.",
    )
    return parser.parse_args()


def plot_prediction_scatter(y_true, y_pred, output_path):
    """Figure 5: predicted vs observed thermal intensity."""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolor="none")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Observed thermal intensity")
    ax.set_ylabel("Predicted thermal intensity")
    ax.set_title("Figure 5 – Predicted vs Observed (test set)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    config = default_pipeline_config()

    dataset_path = config.paths.data_cache / "recs2020_gas_heating.parquet"
    df = load_dataset(dataset_path)

    splits = split_dataset(df, config)
    pipeline, metrics = train_model(splits, config)

    model_path = config.paths.models_dir / args.model_name
    save_model(pipeline, model_path)

    table3_path = config.paths.tables_dir / "table3_xgboost_performance.csv"
    pd.DataFrame([metrics]).to_csv(table3_path, index=False)

    division_breakdown = error_breakdown(pipeline, splits, df, config, "DIVISION")
    env_breakdown = error_breakdown(pipeline, splits, df, config, "envelope_class")
    division_breakdown.to_csv(
        config.paths.tables_dir / "table3_division_breakdown.csv",
        index=False,
    )
    env_breakdown.to_csv(
        config.paths.tables_dir / "table3_envelope_breakdown.csv",
        index=False,
    )

    preds = pipeline.predict(splits.X_test)
    predictions_df = pd.DataFrame(
        {
            "DOEID": df.loc[splits.X_test.index, "DOEID"],
            "actual": splits.y_test,
            "predicted": preds,
        }
    )
    predictions_path = config.paths.data_cache / "xgb_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)

    figure5_path = config.paths.figures_dir / "figure5_model_performance.png"
    plot_prediction_scatter(splits.y_test, preds, figure5_path)

    print("✅ XGBoost model trained.")
    print(f"   Model     -> {model_path}")
    print(f"   Metrics   -> {table3_path}")
    print(f"   Figures   -> {figure5_path}")
    print(f"   Predictions cache -> {predictions_path}")


if __name__ == "__main__":
    main()
