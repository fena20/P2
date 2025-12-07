from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class VariableMeta:
    """Metadata definition used for Table 1."""

    name: str
    description: str
    unit: str
    source: str
    role: str
    code: str


@dataclass
class PipelinePaths:
    """Centralized path management for the workflow."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    data_dir: Path | None = None
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.data_dir is None:
            data_env = os.getenv("RECS2020_DATA_DIR")
            self.data_dir = Path(data_env).expanduser().resolve() if data_env else (
                self.project_root / "data"
            )
        if self.output_dir is None:
            self.output_dir = self.project_root / "output" / "recs2020"

    @property
    def data_cache(self) -> Path:
        return self.output_dir / "data"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    def ensure(self) -> None:
        for path in [
            self.output_dir,
            self.data_cache,
            self.tables_dir,
            self.figures_dir,
            self.models_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Hyper-parameters and feature definitions for the XGBoost model."""

    target_col: str = "thermal_intensity_btu_per_sqft_hdd"
    weight_col: str = "NWEIGHT"
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    numeric_features: List[str] = field(
        default_factory=lambda: [
            "HDD65",
            "heated_floor_area_sqft",
            "year_built",
            "occupants",
            "baseline_heating_energy_mmbtu",
            "household_income",
        ]
    )
    categorical_features: List[str] = field(
        default_factory=lambda: [
            "DIVISION",
            "REGIONC",
            "HOUSING_TYPE",
            "envelope_class",
            "window_quality",
            "draftiness",
            "EQUIPAGE",
            "EQUIPM",
        ]
    )
    xgb_params: Dict[str, int | float] = field(
        default_factory=lambda: {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "tree_method": "hist",
        }
    )


@dataclass
class EnvelopeConfig:
    """Rule-based scoring heuristics for envelope classification."""

    recent_year_threshold: int = 2000
    mid_year_threshold: int = 1980
    drafty_good: List[int] = field(default_factory=lambda: [1, 2])  # 1=never,2=seldom
    drafty_poor: List[int] = field(default_factory=lambda: [4, 5])  # 4=often,5=always
    window_good_codes: List[int] = field(default_factory=lambda: [1, 2])
    window_poor_codes: List[int] = field(default_factory=lambda: [4, 5])
    score_bins: Dict[str, range] = field(
        default_factory=lambda: {
            "poor": range(-10, 1),
            "medium": range(1, 3),
            "good": range(3, 10),
        }
    )


@dataclass
class ScenarioConfig:
    """Economic and emissions assumptions for retrofit analysis."""

    discount_rate: float = 0.05
    analysis_years: int = 15
    gas_price_per_mmbtu: float = 12.0
    electricity_price_per_kwh: float = 0.15
    gas_co2_kg_per_mmbtu: float = 53.06
    grid_co2_kg_per_kwh: float = 0.35
    price_adjustments_by_division: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {}
    )


@dataclass
class NSGAConfig:
    """Configuration for NSGA-II optimization."""

    population: int = 80
    generations: int = 150
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    seed: int = 42


@dataclass
class PipelineConfig:
    """Full configuration bundle for the RECS workflow."""

    paths: PipelinePaths = field(default_factory=PipelinePaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    nsga: NSGAConfig = field(default_factory=NSGAConfig)
    variable_definitions: List[VariableMeta] = field(default_factory=list)
    heat_energy_candidates: List[str] = field(
        default_factory=lambda: [
            "BTUNG",
            "BTUHEAT",
            "BTU_SPACE_HEAT",
            "BTUHS",
            "BTUSE",
        ]
    )
    floor_area_candidates: List[str] = field(
        default_factory=lambda: [
            "TOTSQFT_EN",
            "TOTSQFT",
            "HEATEDSQFT",
        ]
    )
    occupant_candidates: List[str] = field(default_factory=lambda: ["NUMHHS", "NHSLDMEM"])
    income_candidates: List[str] = field(default_factory=lambda: ["MONEYPY", "HINCP"])

    def __post_init__(self) -> None:
        self.paths.ensure()


VARIABLE_DEFINITIONS: List[VariableMeta] = [
    VariableMeta(
        name="DOEID",
        description="Unique household identifier in RECS 2020 microdata",
        unit="-",
        source="RECS 2020 microdata",
        role="identifier",
        code="DOEID",
    ),
    VariableMeta(
        name="NWEIGHT",
        description="Final person-level sample weight",
        unit="-",
        source="RECS 2020 microdata",
        role="weight",
        code="NWEIGHT",
    ),
    VariableMeta(
        name="FUELHEAT",
        description="Primary space-heating fuel",
        unit="categorical",
        source="RECS 2020 microdata",
        role="filter/feature",
        code="FUELHEAT",
    ),
    VariableMeta(
        name="HDD65",
        description="Annual heating degree days (base 65°F) for household climate",
        unit="degree-days",
        source="RECS 2020 microdata",
        role="feature/normalization",
        code="HDD65",
    ),
    VariableMeta(
        name="heated_floor_area_sqft",
        description="Conditioned floor area used for space heating",
        unit="square feet",
        source="Derived from TOTSQFT_EN",
        role="feature/normalization",
        code="TOTSQFT_EN",
    ),
    VariableMeta(
        name="baseline_heating_energy_mmbtu",
        description="Annual space-heating energy use",
        unit="MMBtu",
        source="Derived from BTUNG / BTUHEAT",
        role="target component",
        code="BTUNG",
    ),
    VariableMeta(
        name="thermal_intensity_btu_per_sqft_hdd",
        description="Heating energy normalized by area and climate",
        unit="BTU / (ft²·HDD)",
        source="Derived metric",
        role="target",
        code="I",
    ),
    VariableMeta(
        name="DIVISION",
        description="U.S. Census division",
        unit="categorical",
        source="RECS 2020 microdata",
        role="feature/stratification",
        code="DIVISION",
    ),
    VariableMeta(
        name="REGIONC",
        description="U.S. Census region",
        unit="categorical",
        source="RECS 2020 microdata",
        role="feature/stratification",
        code="REGIONC",
    ),
    VariableMeta(
        name="HOUSING_TYPE",
        description="Building type (single family, multi family, etc.)",
        unit="categorical",
        source="RECS 2020 microdata",
        role="feature",
        code="TYPEHUQ",
    ),
    VariableMeta(
        name="year_built",
        description="Construction year midpoint",
        unit="year",
        source="Derived from YEARMADE",
        role="feature",
        code="YEARMADE",
    ),
    VariableMeta(
        name="draftiness",
        description="Self-reported air leakage (draftiness)",
        unit="ordinal (1–5)",
        source="RECS 2020 microdata",
        role="feature",
        code="DRAFTY",
    ),
    VariableMeta(
        name="window_quality",
        description="Window insulation quality indicator",
        unit="categorical",
        source="RECS 2020 microdata",
        role="feature",
        code="WINDOWS",
    ),
    VariableMeta(
        name="EQUIPM",
        description="Main heating equipment type",
        unit="categorical",
        source="RECS 2020 microdata",
        role="feature",
        code="EQUIPM",
    ),
    VariableMeta(
        name="EQUIPAGE",
        description="Age of main heating equipment",
        unit="years",
        source="RECS 2020 microdata",
        role="feature",
        code="EQUIPAGE",
    ),
    VariableMeta(
        name="envelope_class",
        description="Categorical envelope class (poor/medium/good)",
        unit="categorical",
        source="Derived metric",
        role="feature/stratification",
        code="Envelope score",
    ),
    VariableMeta(
        name="household_income",
        description="Household income bracket midpoint",
        unit="USD",
        source="Derived from HINCP/MONEYPY",
        role="feature/context",
        code="HINCP",
    ),
]


def default_pipeline_config() -> PipelineConfig:
    """Return a configurable default configuration instance."""

    cfg = PipelineConfig(variable_definitions=VARIABLE_DEFINITIONS)
    return cfg
