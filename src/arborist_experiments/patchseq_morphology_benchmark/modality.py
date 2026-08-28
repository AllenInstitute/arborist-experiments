from typing import Optional

import numpy as np

from .data_loading import Cell


MODALITY_TYPES = {
    "full":     {1, 2, 3, 4},
    "axon":     {1, 2},
    "dendrite": {1, 3, 4},
}


def filter_cell(cell: Cell, modality: str) -> Optional[Cell]:
    if modality not in MODALITY_TYPES:
        raise ValueError(f"Unknown modality {modality!r}")
    keep_types = MODALITY_TYPES[modality]

    types = cell.types
    parents = cell.parents
    n = len(types)

    keep_mask = np.array([t in keep_types for t in types], dtype=bool)
    if not keep_mask.any():
        return None
    if keep_mask.sum() < 3:
        return None

    non_soma_kept = np.array(
        [keep_mask[i] and int(types[i]) != 1 for i in range(n)], dtype=bool
    )
    if not non_soma_kept.any():
        return None

    old_to_new = -np.ones(n, dtype=np.int64)
    new_indices = np.flatnonzero(keep_mask)
    for new_i, old_i in enumerate(new_indices):
        old_to_new[old_i] = new_i

    new_parents = np.full(len(new_indices), -1, dtype=np.int64)
    new_synthetic = np.zeros(len(new_indices), dtype=bool)
    for new_i, old_i in enumerate(new_indices):
        p = int(parents[old_i])
        syn = bool(cell.synthetic_edge[old_i])
        while p != -1 and not keep_mask[p]:
            syn = syn or bool(cell.synthetic_edge[p])
            p = int(parents[p])
        new_parents[new_i] = -1 if p == -1 else int(old_to_new[p])
        new_synthetic[new_i] = syn

    return Cell(
        specimen_id=cell.specimen_id,
        types=types[keep_mask].copy(),
        parents=new_parents,
        xyz=cell.xyz[keep_mask].copy(),
        radius=cell.radius[keep_mask].copy(),
        synthetic_edge=new_synthetic,
    )
