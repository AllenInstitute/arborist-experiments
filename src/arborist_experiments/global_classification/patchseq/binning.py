"""Patch-seq binning constants, derived from the cohort they bin."""

from __future__ import annotations

import numpy as np

from arborist_experiments.global_classification.extract import BinningConfig

# Ranges are the p0.5-p99.5 envelope of soma-centered swc_upright over all 1618 cells, rounded
# out: x +/-270, y -310..556, z -32..129. The earlier +/-500 / -300..1300 box was set for an
# absolute pia frame that this data is not in, so most bins were empty for every cell -- xz put
# 625 bins over a region where ~30 were ever occupied. bin_size 20 matches the paper.
DENSITY_CONFIG = {
    "density": True,
    "smooth": True,
    "sigma": 1,
    "bin_size": 20,
    "r_min_x": -300.0, "r_max_x": 300.0,
    "r_min_y": -350.0, "r_max_y": 600.0,
    "r_min_z":  -50.0, "r_max_z": 150.0,
}

PATCHSEQ = BinningConfig(
    name="patchseq",
    density_config=DENSITY_CONFIG,
    sholl_radii=list(np.arange(10.0, 610.0, 10.0)),
    persim_kwargs={
        "birth_range": (0.0, 500.0),
        "pers_range": (0.0, 500.0),
        "pixel_size": 25.0,
    },
    # p99.5 envelope of soma-centered swc_upright over all 1618 cells: in-plane radius 318 um,
    # y -310..556.
    r_edges=np.linspace(0.0, 320.0, 13),            # 12 in-plane radial bins (microns)
    z_edges=np.linspace(-350.0, 600.0, 13),         # 12 soma-relative depth bins (microns)
)
