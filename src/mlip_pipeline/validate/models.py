"""Backward-compatibility shim.

All model classes now live in ``mlip_pipeline.models``.  This module
re-exports everything so that existing imports such as::

    from mlip_pipeline.validate.models import ValidationResult

continue to work without modification.
"""
from mlip_pipeline.models import (  # noqa: F401
    EosResult,
    ElasticResult,
    MeltingResult,
    ThermalExpansionResult,
    VacancyResult,
    RdfResult,
    ValidationResult,
)

__all__ = [
    "EosResult",
    "ElasticResult",
    "MeltingResult",
    "ThermalExpansionResult",
    "VacancyResult",
    "RdfResult",
    "ValidationResult",
]
