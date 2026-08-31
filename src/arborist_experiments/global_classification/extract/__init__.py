"""The handcrafted feature extractors, written once and parameterised per cohort.

This package is the answer to a duplication that had grown to 937 lines across two cohort
packages, of which about 830 were identical: `features.py` and `radial_arbor_features.py` existed
twice, and the copies differed only in three density literals, two bin-edge arrays, two extra
entries in a lookup table, and prose. Everything cohort-specific is now a `BinningConfig` the
caller passes in, built in `patchseq.binning` and `microns.binning`.

Nothing here imports from `patchseq` or `microns`; the only dependency is `utils.cells`. That is
what makes a third cohort a config rather than a third copy.
"""

from .binning import BinningConfig
from .features import (
    FeatureResult,
    extract_density,
    extract_density_xy,
    extract_density_xz,
    extract_density_yz,
    extract_morphometrics,
    extract_persistence_image,
    extract_sholl,
)
from .modality import MODALITY_TYPES, filter_cell
from .radial_arbor_features import (
    MODALITY_GROUPS,
    STRUCTURE_GROUPINGS,
    extract_radial_arbors,
)
from .swc_adapter import cell_to_neurontree, cell_to_swc_df

__all__ = [
    "BinningConfig",
    "FeatureResult",
    "MODALITY_GROUPS",
    "MODALITY_TYPES",
    "STRUCTURE_GROUPINGS",
    "cell_to_neurontree",
    "cell_to_swc_df",
    "extract_density",
    "extract_density_xy",
    "extract_density_xz",
    "extract_density_yz",
    "extract_morphometrics",
    "extract_persistence_image",
    "extract_radial_arbors",
    "extract_sholl",
    "filter_cell",
]
