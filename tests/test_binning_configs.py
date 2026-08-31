"""The published binning constants must be exactly the ones the published runs were produced with.

Each cohort's constants are a `BinningConfig` passed to the extractors, replacing the module-global
rebinding that `apply_inh_constants` used to do. That rebinding's stated reason -- that the
extractors run inside ProcessPoolExecutor workers and only fork-inherited globals reach them --
was wrong: `utils.cells.run_jobs` passes its `*job_args` through `ex.map`, which pickles them, so
a config travels exactly as `modality` always did. `test_a_config_survives_the_worker_boundary`
is the check that keeps that true.

What the old design risked was silently moving already-published numbers -- a stray import-time
activation, a fork that happened too early, a digest computed off stale copies. A value cannot be
moved by any of those now, so the remaining exposure is an edited literal. `constants_digest` is
the exact test of that: the published excitatory feature cache directory is named after it, so
pinning `em_exc` to 2a91b6b7 pins every constant a feature value depends on at once.
"""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from arborist_experiments.global_classification.microns import feature_cache
from arborist_experiments.global_classification.microns.binning import (
    EM_EXC,
    MICRONS_INH,
)
from arborist_experiments.global_classification.patchseq.binning import PATCHSEQ

# The one published feature cache directory is
# `em_exc_n160_21e1abc7f7ce_r0_c2a91b6b7`. If this digest moves, every matrix under
# scratch/features is orphaned and the exc rows are no longer the exc rows.
PUBLISHED_DIGESTS = {"em_exc": "2a91b6b7", "microns_inh": "a761cf82", "patchseq": "55d9e4ce"}


def _ranges(cfg) -> dict[str, tuple[float, float]]:
    return {ax: (cfg.density_config[f"r_min_{ax}"], cfg.density_config[f"r_max_{ax}"])
            for ax in "xyz"}


@pytest.mark.parametrize("cfg", [EM_EXC, MICRONS_INH, PATCHSEQ], ids=lambda c: c.name)
def test_digest_is_the_published_one(cfg):
    assert feature_cache.constants_digest(cfg) == PUBLISHED_DIGESTS[cfg.name]


def test_exc_constants_are_the_ones_the_exc_runs_used():
    """The digest above would catch any of these, but not say which one moved."""
    assert EM_EXC.r_edges[-1] == 220.0
    assert (EM_EXC.z_edges[0], EM_EXC.z_edges[-1]) == (-180.0, 460.0)
    assert _ranges(EM_EXC) == {"x": (-160.0, 160.0), "y": (-180.0, 460.0), "z": (-140.0, 140.0)}


def test_the_three_configs_are_distinguishable():
    """Same-digest configs would collide in the feature cache and cross-contaminate two cohorts."""
    digests = {feature_cache.constants_digest(c) for c in (EM_EXC, MICRONS_INH, PATCHSEQ)}
    assert len(digests) == 3


@pytest.mark.parametrize("cfg", [EM_EXC, MICRONS_INH], ids=lambda c: c.name)
def test_sholl_and_persistence_stay_identical_to_patchseq(cfg):
    """Deliberate: it is what keeps those MICrONS rows comparable to the Patch-seq baselines."""
    assert cfg.sholl_radii == PATCHSEQ.sholl_radii
    assert cfg.persim_kwargs == PATCHSEQ.persim_kwargs


def test_disk_areas_follow_r_edges():
    """`disk_areas` is derived, so a config cannot carry the wrong annulus normalization.

    Rebinding `R_EDGES` without recomputing `_DISK_AREAS` was the specific bug the old
    module-global scheme could produce, and `apply_inh_constants` had to remember not to.
    """
    for cfg in (EM_EXC, MICRONS_INH, PATCHSEQ):
        a = np.pi * (cfg.r_edges[1:] ** 2 - cfg.r_edges[:-1] ** 2)
        assert np.array_equal(cfg.disk_areas, a / a.max())


def _worker_view(cfg) -> tuple[str, float, float, float]:
    return (cfg.name, float(cfg.r_edges[-1]), float(cfg.disk_areas[0]),
            float(cfg.density_config["r_min_x"]))


def test_a_config_survives_the_worker_boundary():
    """The claim that replaced the fork argument: a config reaches a pool worker intact."""
    want = _worker_view(MICRONS_INH)
    round_tripped = pickle.loads(pickle.dumps(MICRONS_INH))
    assert _worker_view(round_tripped) == want
    assert (feature_cache.constants_digest(round_tripped)
            == feature_cache.constants_digest(MICRONS_INH))
    with ProcessPoolExecutor(max_workers=1) as ex:
        assert list(ex.map(_worker_view, [MICRONS_INH])) == [want]


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
    ranges = _ranges(MICRONS_INH)
    for ax in "xyz":
        lo, hi = ranges[ax]
        assert lo <= env[ax][0] and hi >= env[ax][1], f"{modality} {ax} clipped"
    assert MICRONS_INH.r_edges[-1] >= env["r"]
    # z_edges bins the cortical axis over the same support as the density y range, as on exc.
    assert (MICRONS_INH.z_edges[0], MICRONS_INH.z_edges[-1]) == ranges["y"]


def test_bin_edges_are_multiples_of_the_bin_size():
    from arborist_experiments.global_classification.microns.derive_constants import (
        BIN_SIZE,
    )

    for lo, hi in _ranges(MICRONS_INH).values():
        assert lo % BIN_SIZE == 0 and hi % BIN_SIZE == 0
    # 12 bins each, matching the exc and Patch-seq layouts so feature counts stay comparable.
    assert len(MICRONS_INH.r_edges) == 13 and len(MICRONS_INH.z_edges) == 13
