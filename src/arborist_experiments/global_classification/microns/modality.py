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

`extract.modality.filter_cell` stays a pure function of (cell, modality); the cohort-dependent
gate is `check_modality(source, modality)`, which the driver calls once before extraction so an
unavailable modality is a startup error rather than 48k identical failures inside a pool. That
split is why only the gate lives here.
"""

from __future__ import annotations

from arborist_experiments.global_classification.extract import MODALITY_TYPES

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
