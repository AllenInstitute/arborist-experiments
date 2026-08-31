from typing import Optional

from .data_loading import Cell


MODALITY_TYPES = {
    "full":     {1, 2, 3, 4},
    "axon":     {1, 2},
    "dendrite": {1, 3, 4},
}


def filter_cell(cell: Cell, modality: str) -> Optional[Cell]:
    if modality not in MODALITY_TYPES:
        raise ValueError(f"Unknown modality {modality!r}")
    return cell.filtered(MODALITY_TYPES[modality])
