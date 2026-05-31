"""Backward-compatibility shim — all symbols now live in mlip_pipeline.models package.

Anything that previously imported from ``mlip_pipeline.models`` (the flat
module) will continue to work unchanged.  New code should import from the
package directly, e.g.::

    from mlip_pipeline.models import FitResult          # preferred
    from mlip_pipeline.models.loop import GenerationState  # most explicit
"""
from mlip_pipeline.models import *  # noqa: F401, F403
from mlip_pipeline.models import __all__  # noqa: F401
