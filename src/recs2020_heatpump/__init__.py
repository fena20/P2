"""
RECS 2020 Heat Pump Retrofit workflow utilities.

This package hosts shared configuration, data loading helpers, preprocessing
routines, modeling utilities, and optimization helpers that are reused across
the staged scripts defined in ``src/01_data_prep.py`` through
``src/07_tipping_point_maps.py``.
"""

from .config import (
    PipelineConfig,
    PipelinePaths,
    VariableMeta,
    default_pipeline_config,
)

__all__ = [
    "PipelineConfig",
    "PipelinePaths",
    "VariableMeta",
    "default_pipeline_config",
]
