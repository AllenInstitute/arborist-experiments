"""Binning constants for the inhibitory cohort, and the one call that activates them.

Why these differ from the excitatory ones. `derive_constants.derive` run on the inh cache
(7824 cells, 36.2M nodes) gives a much larger envelope, because inhibitory axons are
translaminar while the exc cohort is dendrite-only:

    struct set        x p0.5/p99.5      y p0.5/p99.5      z p0.5/p99.5    in-plane r p99.5
    dendrite (1,3)    -186.9 / 186.0    -226.2 / 383.0    -165.1 / 164.6      254.1
    axon     (1,2)    -277.4 / 249.5    -318.1 / 415.9    -224.2 / 211.8      349.5
    full     (1,2,3)  -257.2 / 234.7    -297.3 / 410.0    -212.8 / 201.2      330.7
    (exc, for reference: -150.3/148.6, -179.9/454.7, -137.0/132.4, 209.4)

The values below are the **union** over the three modalities, rounded outward to the next 20 um
by the same rule the exc and Patch-seq constants use. One constant set for all three inh
modalities is the point: dendrite-vs-axon-vs-full is an internal comparison, and it would be
confounded if each modality binned over its own support and so produced a different feature
count. Nothing is clipped for any modality as a result.

`SHOLL_*` and `PERSIM_KWARGS` are deliberately NOT changed -- per-cell max 3D radius is
p50/p90/p99 = 205/311/497 um, so the existing 600 um Sholl envelope covers 99% of cells, and
keeping them identical is what makes the inh Sholl and persistence rows comparable to the exc
and Patch-seq ones.

Activation is an explicit call, not an import side effect. It rebinds module-level names in
`features` and `radial_arbor_features` rather than parameterizing them, for the same reason the
port copied those modules in the first place: they are read *inside* ProcessPoolExecutor
workers. Rebinding in the parent before any pool is created is inherited by fork; threading a
config object through would not be. Doing it this way also means not one line of the excitatory
code path changes, so the already-published exc numbers cannot have moved --
`assert_exc_constants()` is the test of that.
"""

from __future__ import annotations

import numpy as np


# Union of the dendrite/axon/full envelopes above, rounded out to the next 20 um.
DENSITY_RANGES = {"x": (-280.0, 260.0), "y": (-320.0, 420.0), "z": (-240.0, 220.0)}
R_EDGES = np.linspace(0.0, 360.0, 13)      # axon in-plane p99.5 = 349.5 -> 360
Z_EDGES = np.linspace(-320.0, 420.0, 13)   # matches the density y range, as on the exc cohort

_EXC_R_MAX = 220.0
_EXC_Z = (-180.0, 460.0)

__all__ = ["DENSITY_RANGES", "R_EDGES", "Z_EDGES", "apply_inh_constants",
           "assert_exc_constants"]


def assert_exc_constants() -> None:
    """Fail if the excitatory constants are not the ones the exc runs were produced with."""
    from . import features, radial_arbor_features as raf

    assert raf.R_EDGES[-1] == _EXC_R_MAX, f"R_EDGES max is {raf.R_EDGES[-1]}, expected 220"
    assert (raf.Z_EDGES[0], raf.Z_EDGES[-1]) == _EXC_Z, f"Z_EDGES is {raf.Z_EDGES[[0, -1]]}"
    for key, want in (("r_min_x", -160.0), ("r_max_x", 160.0), ("r_min_y", -180.0),
                      ("r_max_y", 460.0), ("r_min_z", -140.0), ("r_max_z", 140.0)):
        got = features.DENSITY_CONFIG[key]
        assert got == want, f"DENSITY_CONFIG[{key!r}] is {got}, expected {want}"


def apply_inh_constants(log=print) -> None:
    """Rebind the binning constants in `features` and `radial_arbor_features` to the inh set.

    Must be called before any feature extraction, i.e. before the first process pool is forked.
    """
    from . import features, radial_arbor_features as raf

    assert_exc_constants()

    raf.R_EDGES = R_EDGES
    raf.Z_EDGES = Z_EDGES
    # _DISK_AREAS is derived from R_EDGES at import time, so rebinding R_EDGES alone would leave
    # the radial histogram normalized by the excitatory disc areas.
    areas = np.pi * (R_EDGES[1:] ** 2 - R_EDGES[:-1] ** 2)
    raf._DISK_AREAS = areas / areas.max()

    cfg = dict(features.DENSITY_CONFIG)
    for ax, (lo, hi) in DENSITY_RANGES.items():
        cfg[f"r_min_{ax}"], cfg[f"r_max_{ax}"] = lo, hi
    features.DENSITY_CONFIG = cfg

    log(f"inh constants active: R_EDGES 0..{R_EDGES[-1]:g}, "
        f"Z_EDGES {Z_EDGES[0]:g}..{Z_EDGES[-1]:g}, density x/y/z "
        f"{DENSITY_RANGES['x']}/{DENSITY_RANGES['y']}/{DENSITY_RANGES['z']}")
