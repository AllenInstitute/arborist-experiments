"""The excitatory binning constants must be exactly the ones the exc runs were produced with.

`apply_inh_constants` rebinds module globals rather than threading a config object, because
`features` and `radial_arbor_features` are read inside ProcessPoolExecutor workers and only
fork-inherited globals reach them. The cost of that choice is that a stray import-time call, or
an edited literal, would silently move the already-published excitatory numbers. This is the
test that would catch it -- so it must run in a process where nothing has activated the inh set.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from arborist_experiments.global_classification.microns import inh_constants


def test_exc_constants_are_intact_on_a_fresh_import():
    inh_constants.assert_exc_constants()


def test_activation_is_not_an_import_side_effect():
    """A fresh interpreter that imports everything must still pass the exc assertion."""
    code = (
        "from arborist_experiments.global_classification.microns import "
        "run_benchmark, features, radial_arbor_features, inh_constants, data_loading;"
        "inh_constants.assert_exc_constants();print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_apply_inh_constants_rebinds_everything_the_features_read():
    """Run in a subprocess: the rebind is global and would leak into the other tests."""
    code = (
        "import numpy as np;"
        "from arborist_experiments.global_classification.microns import "
        "features as F, radial_arbor_features as raf, inh_constants as IC;"
        "IC.apply_inh_constants(log=lambda m: None);"
        "assert raf.R_EDGES[-1] == IC.R_EDGES[-1];"
        "assert raf.Z_EDGES[0] == IC.Z_EDGES[0] and raf.Z_EDGES[-1] == IC.Z_EDGES[-1];"
        # _DISK_AREAS is derived from R_EDGES at import, so rebinding R_EDGES alone would leave
        # the radial histogram normalized by the excitatory annulus areas.
        "a = np.pi * (IC.R_EDGES[1:]**2 - IC.R_EDGES[:-1]**2);"
        "assert np.allclose(raf._DISK_AREAS, a / a.max());"
        "assert F.DENSITY_CONFIG['r_min_x'] == IC.DENSITY_RANGES['x'][0];"
        "assert F.DENSITY_CONFIG['r_max_y'] == IC.DENSITY_RANGES['y'][1];"
        # Sholl and persistence stay identical on purpose, so those rows remain comparable.
        "assert F.SHOLL_RADII[-1] == 600.0;"
        "assert F.PERSIM_KWARGS['birth_range'] == (0, 500);"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# The p0.5/p99.5 envelopes measured by derive_constants.py on the inh cache, per modality.
# The shipped constants are their union, so nothing is clipped for any of the three modalities
# and dendrite-vs-axon-vs-full stays an internal comparison at a fixed feature count.
INH_MEASURED = {
    "dendrite": {"x": (-186.9, 186.0), "y": (-226.2, 383.0), "z": (-165.1, 164.6), "r": 254.1},
    "axon":     {"x": (-277.4, 249.5), "y": (-318.1, 415.9), "z": (-224.2, 211.8), "r": 349.5},
    "full":     {"x": (-257.2, 234.7), "y": (-297.3, 410.0), "z": (-212.8, 201.2), "r": 330.7},
}


@pytest.mark.parametrize("modality", sorted(INH_MEASURED))
def test_inh_ranges_contain_every_modality_envelope(modality):
    env = INH_MEASURED[modality]
    for ax in "xyz":
        lo, hi = inh_constants.DENSITY_RANGES[ax]
        assert lo <= env[ax][0] and hi >= env[ax][1], f"{modality} {ax} clipped"
    assert inh_constants.R_EDGES[-1] >= env["r"]
    # Z_EDGES bins the cortical axis over the same support as the density y range, as on exc.
    assert (inh_constants.Z_EDGES[0], inh_constants.Z_EDGES[-1]) == inh_constants.DENSITY_RANGES["y"]


def test_bin_edges_are_multiples_of_the_bin_size():
    from arborist_experiments.global_classification.microns.derive_constants import BIN_SIZE

    for lo, hi in inh_constants.DENSITY_RANGES.values():
        assert lo % BIN_SIZE == 0 and hi % BIN_SIZE == 0
    # 12 bins each, matching the exc and Patch-seq layouts so feature counts stay comparable.
    assert len(inh_constants.R_EDGES) == 13 and len(inh_constants.Z_EDGES) == 13
