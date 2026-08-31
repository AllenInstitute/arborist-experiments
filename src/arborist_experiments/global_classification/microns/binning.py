"""MICrONS binning constants, one config per cohort.

Both are derived by `derive_constants.py` on the cohort they bin, by the same rule as Patch-seq
(pooled-node p0.5-p99.5 per axis, rounded out to the next bin_size). `SHOLL_RADII` and
`PERSIM_KWARGS` are deliberately identical to Patch-seq's on both, so those rows stay directly
comparable to the published 0.380 / 0.830 baselines: per-cell max 3D radius and the birth/death
envelope both fall inside the Patch-seq ranges here.

Selecting a cohort is `CONFIGS[source]`. It used to be a call to `inh_constants
.apply_inh_constants()`, which rebound the excitatory constants in place before the first process
pool was forked, and whose whole risk surface -- a stray import-time call, a fork that happened
too early, a digest computed off stale copies -- was the price of not passing a value.
"""

from __future__ import annotations

import numpy as np

from arborist_experiments.global_classification.extract import BinningConfig

SHOLL_RADII = list(np.arange(10.0, 610.0, 10.0))

PERSIM_KWARGS = {
    "birth_range": (0.0, 500.0),
    "pers_range": (0.0, 500.0),
    "pixel_size": 25.0,
}

# Re-derived on all 49803 exc cells / 93.4M soma+dendrite nodes: x -150.3..148.6, y -179.9..454.7,
# z -137.0..132.4. The box is smaller than Patch-seq's in x and y and larger in z, so feature
# counts drop (xy 512 / xz 224 / yz 448 against 1440 / 750 / 1200): MICrONS EM arbors are
# laterally more compact, and z here is a real ~280 um extent rather than a ~200 um slice.
EM_EXC = BinningConfig(
    name="em_exc",
    density_config={
        "density": True,
        "smooth": True,
        "sigma": 1,
        "bin_size": 20,
        "r_min_x": -160.0, "r_max_x": 160.0,
        "r_min_y": -180.0, "r_max_y": 460.0,
        "r_min_z": -140.0, "r_max_z": 140.0,
    },
    sholl_radii=SHOLL_RADII,
    persim_kwargs=PERSIM_KWARGS,
    # in-plane radius p99.5 = 209.4 -> 220, and z_edges matches the density y range so both
    # representations bin the cortical axis over the same support.
    r_edges=np.linspace(0.0, 220.0, 13),            # 12 in-plane radial bins (microns)
    z_edges=np.linspace(-180.0, 460.0, 13),         # 12 soma-relative depth bins (microns)
)

# Derived over all 7824 inh cells / 36.2M nodes as the union of the dendrite/axon/full envelopes,
# so nothing is clipped for any of the three modalities and dendrite-vs-axon-vs-full stays an
# internal comparison at a fixed feature count. Inhibitory axons reach much further than any
# excitatory dendrite, which is the whole reason this is a second config.
MICRONS_INH = BinningConfig(
    name="microns_inh",
    density_config={
        "density": True,
        "smooth": True,
        "sigma": 1,
        "bin_size": 20,
        "r_min_x": -280.0, "r_max_x": 260.0,
        "r_min_y": -320.0, "r_max_y": 420.0,
        "r_min_z": -240.0, "r_max_z": 220.0,
    },
    sholl_radii=SHOLL_RADII,
    persim_kwargs=PERSIM_KWARGS,
    r_edges=np.linspace(0.0, 360.0, 13),            # axon in-plane p99.5 = 349.5 -> 360
    z_edges=np.linspace(-320.0, 420.0, 13),         # the density y range, as on the exc cohort
)

CONFIGS = {"em_exc": EM_EXC, "microns_inh": MICRONS_INH}
