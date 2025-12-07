from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def prepare_shap_inputs(
    pipeline: Pipeline,
    X: pd.DataFrame,
    sample_size: int = 5000,
    random_state: int = 42,
):
    """Return processed feature matrix, SHAP values, and feature names."""

    sample = X.sample(
        n=min(sample_size, len(X)),
        random_state=random_state,
    )
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_processed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    base_values = explainer.expected_value
    return sample, X_processed, shap_values, base_values, feature_names


def shap_importance_table(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Mean |SHAP| ranking."""

    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    df["normalized_importance_pct"] = 100 * df["mean_abs_shap"] / df[
        "mean_abs_shap"
    ].sum()
    return df


def save_shap_summary_plot(
    shap_values: np.ndarray,
    X_processed: np.ndarray,
    feature_names: List[str],
    output_path: Path,
) -> Path:
    """Figure 6: global SHAP importance."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def save_shap_dependence_plots(
    shap_values: np.ndarray,
    X_processed: np.ndarray,
    feature_names: List[str],
    selected_features: Iterable[str],
    output_path: Path,
) -> Path:
    """Figure 7: dependence plots for selected features."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(selected_features), figsize=(5 * len(selected_features), 4))

    if len(selected_features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, selected_features):
        shap.dependence_plot(
            feature,
            shap_values,
            X_processed,
            feature_names=feature_names,
            ax=ax,
            show=False,
        )
        ax.set_title(f"SHAP Dependence: {feature}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
