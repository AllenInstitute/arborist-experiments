"""Compartment modalities available on the two MICrONS cohorts.

Excitatory (`EM_syn.hdf5`), measured on 800- and 3000-cell samples: 100% of cells have basal
dendrite, ~89% have apical, and only 0.43-0.5% have any axon node at all. `full` and `axon` are
therefore not meaningful there -- `full` would differ from `dendrite` for ~200 of 48042 cells
and produce a duplicate row at double the extraction cost, and `axon` would score on a 200-cell
cohort. `basal` and `apical` replace the axon-vs-dendrite axis with the split that data supports.

Inhibitory (`MICrONS_inh_swc`), measured over all 7893 files: types are {1, 2, 3} -- soma, axon,
basal -- and 5251 cells (66.5%) carry axon nodes. So `full`/`axon` are real there and
`basal`/`apical` are not: there is no type-4 node anywhere, which makes `basal` a duplicate of
`dendrite` and `apical` empty.

`filter_cell` stays a pure function of (cell, modality); the cohort-dependent gate is
`check_modality(source, modality)`, which the driver calls once before extraction so an
unavailable modality is a startup error rather than 48k identical failures inside a pool.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from arborist_experiments.global_classification.patchseq.data_loading import (
    Cell, TYPE_SOMA, TYPE_AXON, TYPE_BASAL, TYPE_APICAL,
)


MODALITY_TYPES = {
    "full":     {TYPE_SOMA, TYPE_AXON, TYPE_BASAL, TYPE_APICAL},
    "axon":     {TYPE_SOMA, TYPE_AXON},
    "dendrite": {TYPE_SOMA, TYPE_BASAL, TYPE_APICAL},
    "basal":    {TYPE_SOMA, TYPE_BASAL},
    "apical":   {TYPE_SOMA, TYPE_APICAL},
}

# Which modalities are meaningful depends on the cohort, not on `filter_cell`, so the gate is
# keyed by source and enforced by the driver before extraction starts. `full`/`axon` are real on
# the inhibitory cells (66.5% have axon) and meaningless on the excitatory ones (0.43%);
# `apical` is the reverse, since the inh SWCs have no type-4 node at all.
UNAVAILABLE = {
    "em_exc": {
        "full": "only 0.43% of excitatory cells have any axon node, so 'full' is 'dendrite' "
                "plus noise for ~200 of 48042 cells; use 'dendrite'",
        "axon": "only 0.43% of excitatory cells have any axon node; there is no axon cohort "
                "to score",
    },
    "microns_inh": {
        "apical": "the inhibitory SWCs contain no type-4 nodes; 'apical' would be empty",
        "basal": "the inhibitory SWCs have no apical, so 'basal' is identical to 'dendrite'",
    },
}


def check_modality(source: str, modality: str) -> None:
    """Raise if `modality` is not meaningful on `source`. A hard error beats a duplicate row."""
    if modality not in MODALITY_TYPES:
        raise ValueError(f"Unknown modality {modality!r}")
    reason = UNAVAILABLE.get(source, {}).get(modality)
    if reason:
        raise ValueError(f"modality {modality!r} is not available on cohort {source!r}: {reason}")


@contextmanager
def patchseq_modality_types():
    """Temporarily teach the Patch-seq `filter_cell` about `basal`/`apical`.

    `gaussian_baseline.run_gaussian_cv` is imported unchanged -- it is the baseline
    implementation and editing it would put its published numbers in doubt -- but it calls the
    Patch-seq `filter_cell`, which raises on modalities its own MODALITY_TYPES lacks. Patch-seq's
    `dendrite` set is {1,3,4}, identical to ours, so this only *adds* the two missing keys and is
    a no-op for the default modality.
    """
    import arborist_experiments.global_classification.patchseq.modality as ps

    assert ps.MODALITY_TYPES["dendrite"] == MODALITY_TYPES["dendrite"]
    added = {k: v for k, v in MODALITY_TYPES.items() if k not in ps.MODALITY_TYPES}
    ps.MODALITY_TYPES.update(added)
    try:
        yield
    finally:
        for k in added:
            del ps.MODALITY_TYPES[k]


def filter_cell(cell: Cell, modality: str) -> Optional[Cell]:
    if modality not in MODALITY_TYPES:
        raise ValueError(f"Unknown modality {modality!r}")
    return cell.filtered(MODALITY_TYPES[modality])
