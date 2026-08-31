"""Verification suite for the MICrONS cache and cohort (plan section 10, items 1-4).

Every source read here is ascending and contiguous (rows 0..N). Random access into
EM_syn.hdf5 costs 37-400 ms/cell against 2.4 ms/cell sequential, so a test that sampled
random rows would take longer than the whole extraction.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from arborist_experiments.global_classification.microns.cache_extract import SOURCE_PATH
from arborist_experiments.global_classification.microns.data_loading import (
    CACHE_PATH,
    EXC_COARSE_CLASSES,
    LABELS_PATH,
    TYPE_APICAL,
    TYPE_BASAL,
    TYPE_SOMA,
    build_cohort,
    cell_from_cache,
    class_counts,
    load_cache,
    load_labels,
)

N_FIDELITY = 20        # source rows re-read for the byte-level fidelity check
N_INVARIANT = 2000     # cells checked for the Cell contract

# Measured on the completed cache. These are asserted exactly: any drift means the join,
# the label file, or the extraction changed, and every number downstream moves with it.
EXPECTED_N_CACHE = 49803
EXPECTED_N_META = 49795
EXPECTED_N_JOINED = 48789
EXPECTED_N_COHORT = 48042
EXPECTED_COUNTS = {
    "L4IT": 12213, "L6CT": 10149, "L2IT": 9237, "L3IT": 6149,
    "L6IT": 4960, "L5IT": 3836, "L5ET": 1036, "L5NP": 462,
}

pytestmark = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH), reason=f"cache not built at {CACHE_PATH}"
)


@pytest.fixture(scope="module")
def cache():
    return load_cache(CACHE_PATH)


@pytest.fixture(scope="module")
def cohort():
    return build_cohort(cache_path=CACHE_PATH, labels_path=LABELS_PATH,
                        label_column="coarse_label", log=lambda m: None)


def test_cache_is_complete_and_self_consistent(cache):
    assert len(cache.specimens) == EXPECTED_N_CACHE
    assert len(cache.offsets) == EXPECTED_N_CACHE + 1
    assert cache.offsets[0] == 0
    assert int(cache.offsets[-1]) == len(cache.types) == len(cache.parents) == len(cache.xyz)
    assert np.all(np.diff(cache.offsets) > 0)
    with h5py.File(CACHE_PATH, "r") as f:
        assert bool(f.attrs["complete"])
        assert int(f.attrs["n_cells_done"]) == EXPECTED_N_CACHE
        assert int(np.sum(f["n_nodes"][:])) == int(cache.offsets[-1])
        assert len(f["meta_specimens"]) == EXPECTED_N_META


@pytest.mark.skipif(not os.path.exists(SOURCE_PATH), reason="source hdf5 not mounted")
def test_cache_matches_source_on_ascending_rows(cache):
    """Re-read the source, re-center, and demand equality with the cache."""
    with h5py.File(SOURCE_PATH, "r") as src:
        shape = src["swc_upright/shape"][:N_FIDELITY]
        spec = src["swc_upright/specimens"]
        checked = 0
        for i in range(N_FIDELITY):
            sid = spec[i]
            sid = sid.decode() if isinstance(sid, bytes) else str(sid)
            row = cache.row_by_id.get(sid)
            if row is None:
                continue                                  # skipped by an invariant
            n = int(shape[i, 0])
            xyz = np.asarray(src["swc_upright/data"][i], dtype=np.float64).reshape(n, 3)
            types = np.asarray(src["swc_types/data"][i]).astype(np.int32).ravel()
            parents = np.asarray(src["swc_parents/data"][i]).astype(np.int64).ravel()
            xyz = xyz - xyz[types == TYPE_SOMA].mean(axis=0)

            cell = cell_from_cache(cache, row)
            assert cell.specimen_id == sid
            assert len(cell.types) == n
            np.testing.assert_array_equal(cell.types, types)
            np.testing.assert_array_equal(cell.parents, parents)
            # float32 storage of coordinates up to ~1150 um before centering.
            np.testing.assert_allclose(cell.xyz, xyz, atol=1e-3)
            checked += 1
    assert checked >= N_FIDELITY - 2


def test_cell_invariants(cache):
    for row in range(N_INVARIANT):
        c = cell_from_cache(cache, row)
        n = len(c.types)
        assert n >= 3
        assert set(np.unique(c.types).tolist()) <= {1, 2, 3, 4}
        soma = np.flatnonzero(c.types == TYPE_SOMA)
        roots = np.flatnonzero(c.parents == -1)
        assert len(soma) == 1 and len(roots) == 1 and roots[0] == soma[0]
        assert c.parents.min() >= -1 and c.parents.max() < n
        assert not (c.parents == np.arange(n)).any()
        assert np.isfinite(c.xyz).all()
        # Soma at the origin on all three axes: x,y came that way, z was centered here.
        assert np.abs(c.xyz[soma[0]]).max() < 1e-3
        assert np.all(c.radius == 1.0)
        assert not c.synthetic_edge.any()
        # uint8, not int32: types is now the SkeletonGraph node-attr registry's dtype.
        assert c.types.dtype == np.uint8 and c.parents.dtype == np.int64
        assert c.xyz.dtype == np.float64


def test_cortical_axis_is_upright_y_with_pia_positive(cache):
    """Two independent proofs that CORTICAL_AXIS=1 with pia at +y."""
    finite = np.isfinite(cache.pia_y) & np.isfinite(cache.wm_y)
    frac = float(np.mean((cache.pia_y[finite] > 0) & (cache.wm_y[finite] < 0)))
    assert frac >= 0.99, f"pia_y > 0 > wm_y for only {frac:.3f} of cells"

    above = 0
    total = 0
    for row in range(N_INVARIANT):
        c = cell_from_cache(cache, row)
        ap, ba = c.types == TYPE_APICAL, c.types == TYPE_BASAL
        if not (ap.any() and ba.any()):
            continue
        total += 1
        above += c.xyz[ap, 1].mean() > c.xyz[ba, 1].mean()
    assert total > 0.8 * N_INVARIANT
    assert above / total >= 0.95, f"apical above basal for only {above / total:.3f}"


def test_column_two_centering_was_needed_and_applied():
    """soma_origin proves the fix: x,y arrived at 0, column 2 was absolute and got subtracted."""
    with h5py.File(CACHE_PATH, "r") as f:
        origin = f["soma_origin"][:]
    assert np.abs(origin[:, :2]).max() < 1e-3
    assert origin[:, 2].min() > 500.0 and origin[:, 2].max() < 1500.0


def test_labels_and_cohort_counts(cohort):
    labels = load_labels(LABELS_PATH, "coarse_label")
    assert len(labels) > 60000
    assert all(isinstance(k, str) for k in list(labels)[:100])

    assert len(cohort) == EXPECTED_N_COHORT
    assert cohort.label_names == sorted(EXC_COARSE_CLASSES)
    assert class_counts(cohort) == EXPECTED_COUNTS
    assert sum(EXPECTED_COUNTS.values()) == EXPECTED_N_COHORT

    ids = [c.specimen_id for c in cohort.cells]
    assert len(set(ids)) == len(ids), "duplicate cells in cohort"
    for c, k in zip(cohort.cells[:500], cohort.labels[:500]):
        assert cohort.subclass_by_id[c.specimen_id] == cohort.label_names[int(k)]


def test_join_and_exc_only_drop_counts():
    joined = build_cohort(cache_path=CACHE_PATH, labels_path=LABELS_PATH,
                          exc_only=False, min_class_size=1, log=lambda m: None)
    assert len(joined) == EXPECTED_N_JOINED
    counts = class_counts(joined)
    non_exc = sum(v for k, v in counts.items() if k not in EXC_COARSE_CLASSES)
    assert non_exc == EXPECTED_N_JOINED - EXPECTED_N_COHORT


def test_soma_depth_orders_the_laminar_labels(cohort):
    """The single strongest evidence that the join, the frame and the axis are all right.

    If the label join were broken or the sign flipped, this ordering would not appear.
    """
    order = ["L2IT", "L3IT", "L4IT", "L5IT", "L5ET", "L5NP", "L6IT", "L6CT"]
    depth = np.asarray(cohort.pia_y, dtype=float)      # soma-to-pia along +y
    means = {}
    for name in order:
        k = cohort.label_names.index(name)
        means[name] = float(np.nanmean(depth[cohort.labels == k]))
    seq = [means[n] for n in order]
    assert seq == sorted(seq), f"non-monotonic soma depth: {means}"
    assert 100 < seq[0] < 200 and 550 < seq[-1] < 700
