from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PipelineConfig


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    w_train: pd.Series
    w_val: pd.Series
    w_test: pd.Series


def _feature_lists(df: pd.DataFrame, config: PipelineConfig) -> Tuple[list[str], list[str]]:
    numeric = [f for f in config.model.numeric_features if f in df.columns]
    categorical = [f for f in config.model.categorical_features if f in df.columns]
    return numeric, categorical


def create_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ]
    )
    return preprocessor


def split_dataset(df: pd.DataFrame, config: PipelineConfig) -> DatasetSplits:
    """Create stratified train/val/test splits."""

    numeric, categorical = _feature_lists(df, config)
    features = numeric + categorical
    missing = [col for col in features + [config.model.target_col] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    strat_col = (
        df["DIVISION"].astype(str) + "_" + df["envelope_class"].astype(str)
    )

    X = df[features]
    y = df[config.model.target_col]
    weights = df[config.model.weight_col]

    X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
        X,
        y,
        weights,
        test_size=config.model.test_size,
        random_state=config.model.random_state,
        stratify=strat_col,
    )

    val_size_adjusted = config.model.validation_size / (1 - config.model.test_size)
    (
        X_train,
        X_val,
        y_train,
        y_val,
        w_train,
        w_val,
    ) = train_test_split(
        X_temp,
        y_temp,
        w_temp,
        test_size=val_size_adjusted,
        random_state=config.model.random_state,
        stratify=strat_col.loc[X_temp.index],
    )

    return DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        w_train=w_train,
        w_val=w_val,
        w_test=w_test,
    )


def train_model(
    splits: DatasetSplits,
    config: PipelineConfig,
) -> Tuple[Pipeline, Dict[str, float]]:
    """Train an XGBoost model with preprocessing pipeline."""

    numeric, categorical = _feature_lists(splits.X_train, config)
    preprocessor = create_preprocessor(numeric, categorical)

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        **config.model.xgb_params,
        random_state=config.model.random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_model),
        ]
    )

    pipeline.fit(
        splits.X_train,
        splits.y_train,
        model__sample_weight=splits.w_train,
    )

    metrics = evaluate_model(pipeline, splits, config)
    return pipeline, metrics


def evaluate_model(
    pipeline: Pipeline,
    splits: DatasetSplits,
    config: PipelineConfig,
) -> Dict[str, float]:
    """Compute RMSE/MAE/R2 on train/val/test sets."""

    metrics = {}
    for split_name, X, y, w in [
        ("train", splits.X_train, splits.y_train, splits.w_train),
        ("val", splits.X_val, splits.y_val, splits.w_val),
        ("test", splits.X_test, splits.y_test, splits.w_test),
    ]:
        preds = pipeline.predict(X)
        metrics[f"{split_name}_rmse"] = float(
            mean_squared_error(y, preds, sample_weight=w, squared=False)
        )
        metrics[f"{split_name}_mae"] = float(
            mean_absolute_error(y, preds, sample_weight=w)
        )
        metrics[f"{split_name}_r2"] = float(r2_score(y, preds, sample_weight=w))
    return metrics


def error_breakdown(
    pipeline: Pipeline,
    splits: DatasetSplits,
    df: pd.DataFrame,
    config: PipelineConfig,
    group_col: str,
) -> pd.DataFrame:
    """Return RMSE/MAE by group on the test split."""

    preds = pipeline.predict(splits.X_test)
    results = pd.DataFrame(
        {
            "actual": splits.y_test,
            "predicted": preds,
            "weight": splits.w_test,
            "group": df.loc[splits.X_test.index, group_col],
        }
    )

    rows = []
    for group, frame in results.groupby("group"):
        w = frame["weight"]
        rows.append(
            {
                group_col: group,
                "rmse": mean_squared_error(
                    frame["actual"], frame["predicted"], sample_weight=w, squared=False
                ),
                "mae": mean_absolute_error(
                    frame["actual"], frame["predicted"], sample_weight=w
                ),
                "r2": r2_score(frame["actual"], frame["predicted"], sample_weight=w),
            }
        )
    return pd.DataFrame(rows).sort_values(group_col)


def save_model(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
