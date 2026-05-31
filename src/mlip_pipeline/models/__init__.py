"""mlip_pipeline.models — public re-export of all domain model packages.

Import from here for a stable public API::

    from mlip_pipeline.models import (
        # VASP
        VaspParallelConfig, ScalingPolicy,
        # Pipeline steps
        FitResult, ExploreResult, SelectionResult,
        LabelResult, ConvertResult, EvaluationResult,
        # Loop
        STEPS, STOP_REASONS, GenerationState, LoopResult,
        # Validate
        EosResult, ElasticResult, ValidationResult, ...
    )

Or import directly from the sub-module for clarity::

    from mlip_pipeline.models.validate import EosResult
    from mlip_pipeline.models.loop import GenerationState
"""
from mlip_pipeline.models.vasp import VaspParallelConfig, ScalingPolicy
from mlip_pipeline.models.steps import (
    PrepareTrainResult,
    FitResult,
    ExploreResult,
    SelectionResult,
    LabelResult,
    ConvertResult,
    EvaluationResult,
)
from mlip_pipeline.models.loop import STEPS, STOP_REASONS, GenerationState, LoopResult
from mlip_pipeline.models.validate import (
    EosResult,
    ElasticResult,
    MeltingResult,
    ThermalExpansionResult,
    VacancyResult,
    RdfResult,
    ValidationResult,
)

__all__ = [
    # vasp
    "VaspParallelConfig",
    "ScalingPolicy",
    # steps
    "PrepareTrainResult",
    "FitResult",
    "ExploreResult",
    "SelectionResult",
    "LabelResult",
    "ConvertResult",
    "EvaluationResult",
    # loop
    "STEPS",
    "STOP_REASONS",
    "GenerationState",
    "LoopResult",
    # validate
    "EosResult",
    "ElasticResult",
    "MeltingResult",
    "ThermalExpansionResult",
    "VacancyResult",
    "RdfResult",
    "ValidationResult",
]
