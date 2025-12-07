from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .config import PipelineConfig


def discover_microdata_path(data_dir: Path, pattern: str = "recs2020_public") -> Path:
    """
    Return the first file that matches ``pattern`` within ``data_dir``.

    Parameters
    ----------
    data_dir
        Directory containing RECS 2020 microdata.
    pattern
        Substring to look for inside the file name.
    """

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory {data_dir} does not exist. "
            "Clone https://github.com/Fateme9977/DataR and set RECS2020_DATA_DIR."
        )

    candidates = sorted(data_dir.glob("*.csv"))
    for candidate in candidates:
        if pattern in candidate.name:
            return candidate

    available = ", ".join(c.name for c in candidates) or "no CSV files found"
    raise FileNotFoundError(
        f"Could not locate a RECS 2020 CSV matching '{pattern}' in {data_dir}. "
        f"Available files: {available}"
    )


def load_microdata(csv_path: Path, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load the RECS microdata CSV with optional column subset."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Microdata file not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=columns)
    return df


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Persist dataframe to Parquet for downstream steps."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a cached dataset (Parquet)."""

    if not path.exists():
        raise FileNotFoundError(
            f"Cached dataset not found at {path}. Run the previous stage first."
        )
    return pd.read_parquet(path)


def save_json(data: dict, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent))


def variable_table_from_meta(config: PipelineConfig) -> pd.DataFrame:
    """Create Table 1 (variable definitions) from metadata."""

    rows = [
        {
            "variable": meta.name,
            "description": meta.description,
            "unit": meta.unit,
            "source": meta.source,
            "role": meta.role,
            "recs_code": meta.code,
        }
        for meta in config.variable_definitions
    ]
    return pd.DataFrame(rows)


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate column present in the dataframe."""

    for col in candidates:
        if col in df.columns:
            return col
    return None
