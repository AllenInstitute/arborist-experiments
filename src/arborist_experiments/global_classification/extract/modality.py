"""Which SWC compartment types each modality keeps.

One table for both cohorts. It used to be two, and the Patch-seq copy lacked `basal`/`apical`,
which forced a context manager that temporarily injected those two keys into it so the
Patch-seq-authored `gaussian_baseline` could be called on a MICrONS run. The tables agreed on
every key they shared, so the union is a no-op for Patch-seq and the injection is gone.

Which modalities are *meaningful* on a given cohort is a separate question and not one
`filter_cell` can answer -- see `microns.modality.check_modality`.
"""

from typing import Optional

from arborist_experiments.utils.cells import (
    CellGraph, TYPE_APICAL, TYPE_AXON, TYPE_BASAL, TYPE_SOMA,
)


MODALITY_TYPES = {
    "full":     {TYPE_SOMA, TYPE_AXON, TYPE_BASAL, TYPE_APICAL},
    "axon":     {TYPE_SOMA, TYPE_AXON},
    "dendrite": {TYPE_SOMA, TYPE_BASAL, TYPE_APICAL},
    "basal":    {TYPE_SOMA, TYPE_BASAL},
    "apical":   {TYPE_SOMA, TYPE_APICAL},
}


def filter_cell(cell: CellGraph, modality: str) -> Optional[CellGraph]:
    if modality not in MODALITY_TYPES:
        raise ValueError(f"Unknown modality {modality!r}")
    return cell.filtered(MODALITY_TYPES[modality])
