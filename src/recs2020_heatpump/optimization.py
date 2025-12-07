from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .config import PipelineConfig
from .retrofit import (
    HeatPumpOption,
    RetrofitMeasure,
    build_archetypes,
    default_heat_pumps,
    default_retrofit_measures,
    evaluate_scenario,
)


class RetrofitProblem(ElementwiseProblem):
    """Element-wise NSGA-II problem for a single archetype."""

    def __init__(
        self,
        archetype: pd.Series,
        measures: Sequence[RetrofitMeasure],
        hp_options: Sequence[HeatPumpOption],
        config: PipelineConfig,
    ):
        self.archetype = archetype
        self.measures = list(measures)
        self.hp_options = [None] + list(hp_options)
        self.config = config

        xl = [0] * (len(self.measures) + 1)
        xu = [1] * len(self.measures) + [len(self.hp_options) - 1]

        super().__init__(
            n_var=len(xl),
            n_obj=2,
            n_constr=0,
            xl=np.array(xl),
            xu=np.array(xu),
            type_var=np.int64,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        measure_flags = x[:-1]
        hp_idx = int(x[-1])

        selected_measures = [
            measure for measure, flag in zip(self.measures, measure_flags) if flag == 1
        ]
        hp_option = self.hp_options[hp_idx]

        scenario = evaluate_scenario(
            archetype=self.archetype,
            measures=selected_measures,
            hp_option=hp_option,
            config=self.config,
        )
        out["F"] = np.array([scenario["annual_cost_usd"], scenario["annual_emissions_kg"]])


def run_nsga_for_archetype(
    archetype: pd.Series,
    config: PipelineConfig,
    measures: Sequence[RetrofitMeasure],
    hp_options: Sequence[HeatPumpOption],
) -> pd.DataFrame:
    """Execute NSGA-II for a single archetype and return Pareto solutions."""

    problem = RetrofitProblem(archetype, measures, hp_options, config)
    algorithm = NSGA2(pop_size=config.nsga.population)
    termination = get_termination("n_gen", config.nsga.generations)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=config.nsga.seed,
        verbose=False,
    )

    solutions = []
    for x, f in zip(result.X, result.F):
        measure_flags = x[:-1]
        hp_idx = int(x[-1])
        selected_measures = [
            measure for measure, flag in zip(measures, measure_flags) if flag == 1
        ]
        hp_option = [None] + list(hp_options)
        hp_choice = hp_option[hp_idx]

        scenario = evaluate_scenario(
            archetype=archetype,
            measures=selected_measures,
            hp_option=hp_choice,
            config=config,
        )
        scenario["objective_cost"] = f[0]
        scenario["objective_emissions"] = f[1]
        scenario["selected_binaries"] = x.tolist()
        solutions.append(scenario)

    return pd.DataFrame(solutions)


def run_nsga_pipeline(
    df: pd.DataFrame,
    config: PipelineConfig,
    measures: Sequence[RetrofitMeasure] | None = None,
    hp_options: Sequence[HeatPumpOption] | None = None,
) -> pd.DataFrame:
    """Run NSGA-II for each archetype and combine results."""

    measures = measures or default_retrofit_measures()
    hp_options = hp_options or default_heat_pumps()

    archetypes = build_archetypes(df, config)
    all_results = []
    for _, archetype in archetypes.iterrows():
        pareto_df = run_nsga_for_archetype(archetype, config, measures, hp_options)
        all_results.append(pareto_df)
    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)
