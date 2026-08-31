"""The cell representation every other package is written against.

This exists because `Cell` used to live inside one dataset package while both
cohorts and `treemoco` depended on it, which made every candidate shared module
unmovable.
"""

from .cell_graph import (
    CellGraph,
    TYPE_APICAL,
    TYPE_AXON,
    TYPE_BASAL,
    TYPE_SOMA,
)
from .cohort import Cohort
from .jobs import N_WORKERS, run_jobs

__all__ = [
    "CellGraph",
    "Cohort",
    "N_WORKERS",
    "TYPE_APICAL",
    "TYPE_AXON",
    "TYPE_BASAL",
    "TYPE_SOMA",
    "run_jobs",
]
