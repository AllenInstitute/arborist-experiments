"""Handcrafted morphology feature extractors, parameterised per dataset.

Every dataset-specific constant (density ranges, bin edges, Sholl radii, persistence-image grid) is an
explicit argument: each extractor declares the constants it reads, and a caller holding a
`BinningConfig` unpacks it at the call site. `BinningConfig` is exported because callers pass one
around, not because any extractor takes one. The only dependency inside this library is
`utils.cells`.

`extract_upright_morphometrics` is reachable from here but imported lazily, so it is the one export
that is not in the import block below -- see `__getattr__`.
"""

from importlib import import_module

from .binning import BinningConfig
from .features import (
    MORPHOMETRIC_FEATURES,
    density_feature_names,
    drop_zero_variance_and_impute,
    extract_density,
    extract_morphometrics,
    extract_persistence_image,
    extract_sholl,
    persistence_feature_names,
)
from .modality import MODALITY_TYPES, filter_cell
from .radial_arbor_features import (
    MODALITY_GROUPS,
    STRUCTURE_GROUPINGS,
    extract_radial_hist,
    extract_radial_sholl,
)
from .result import (
    FeatureResult,
    batch_names,
    concat_features,
    design_matrix,
    skips,
)
from .swc_adapter import cell_to_neurontree, cell_to_swc_df

# Resolved on first attribute access (PEP 562) rather than imported above, because
# `upright_morphometrics` pulls `neuron_morphology` and the `simple_tree` shim beneath it -- 37 modules,
# and a fork-only constraint, that a caller reaching for a density map should not inherit. Lazy rather
# than absent so that every extractor in this package is reachable by the same import.
_LAZY = {"extract_upright_morphometrics": "upright_morphometrics"}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)


__all__ = [
    "MODALITY_GROUPS",
    "MODALITY_TYPES",
    "MORPHOMETRIC_FEATURES",
    "STRUCTURE_GROUPINGS",
    "BinningConfig",
    "FeatureResult",
    "batch_names",
    "cell_to_neurontree",
    "cell_to_swc_df",
    "concat_features",
    "density_feature_names",
    "design_matrix",
    "drop_zero_variance_and_impute",
    "extract_density",
    "extract_morphometrics",
    "extract_persistence_image",
    "extract_radial_hist",
    "extract_radial_sholl",
    "extract_sholl",
    "extract_upright_morphometrics",
    "filter_cell",
    "persistence_feature_names",
    "skips",
]
