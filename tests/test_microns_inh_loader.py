"""Verification suite for the MICrONS inhibitory cache, cohort and derived upright frame.

Mirrors test_microns_loader.py, plus the two things that are specific to this cohort and that
every inhibitory number depends on: the geometric root-id -> nucleus-id join, and the rotation
into the upright frame. Both are validated against quantities that had no part in producing
them -- the join by the gap to the runner-up nucleus, the rotation by laminar label ordering on
inhibitory cells (the fit used excitatory ones only).

Source SWCs are small individual CSVs, so unlike the excitatory suite there is no sequential
access constraint and the fidelity check samples spread-out rows.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from arborist_experiments.microns_morphology_benchmark.data_loading import (
    INH_CACHE_PATH, INH_LABELS_PATH, INH_COARSE_CLASSES, TYPE_APICAL, TYPE_AXON, TYPE_BASAL,
    TYPE_SOMA, build_inh_cohort, cell_from_cache, class_counts, load_cache,
)
from arborist_experiments.microns_morphology_benchmark.inh_id_map import (
    ACCEPT_UM, MAP_PATH, SWC_DIR, UPRIGHT_PATH, load_map, load_upright, upright_matrix,
)


N_FIDELITY = 25        # source files re-read for the byte-level fidelity check
N_INVARIANT = 2000     # cells checked for the Cell contract

# Measured on the completed cache. Asserted exactly: any drift means the join, the label file
# or the extraction changed, and every number downstream moves with it.
EXPECTED_N_SWC = 7893
EXPECTED_N_RESOLVED = 7824
EXPECTED_N_LABELLED = 7550
EXPECTED_N_COHORT = 7094
EXPECTED_COUNTS = {
    "PV": 2967, "ITC": 1261, "MC": 1246, "DTC": 339, "NMC": 283, "NGC": 249,
    "L1": 234, "ITCperi": 168, "AltBasket": 166, "AltDTC": 152, "ChC": 29,
}
# Root-ID (filename) matching would resolve only this many, which is why the join is geometric.
EXPECTED_N_ROOT_AGREES = 5796
EXPECTED_THETA_DEG = 4.5758

pytestmark = pytest.mark.skipif(
    not os.path.exists(INH_CACHE_PATH), reason=f"inh cache not built at {INH_CACHE_PATH}"
)


@pytest.fixture(scope="module")
def cache():
    return load_cache(INH_CACHE_PATH)


@pytest.fixture(scope="module")
def cohort():
    return build_inh_cohort(cache_path=INH_CACHE_PATH, labels_path=INH_LABELS_PATH,
                            log=lambda m: None)


@pytest.fixture(scope="module")
def id_map():
    return load_map(MAP_PATH)


# --- the join ------------------------------------------------------------------------------

def test_join_is_unambiguous_and_beats_root_id_matching(id_map):
    assert len(id_map) == EXPECTED_N_SWC
    resolved = id_map[id_map["nucleus_id"] >= 0]
    assert len(resolved) == EXPECTED_N_RESOLVED
    assert resolved["nucleus_id"].is_unique, "two SWCs joined to the same nucleus"

    # The soma node *is* the nucleus centroid, so a real match sits at distance 0.
    assert resolved["soma_dist_um"].max() <= ACCEPT_UM
    assert np.percentile(resolved["soma_dist_um"], 99) < 1e-6
    # And the runner-up is far away, so the 2 um threshold is nowhere near a decision boundary.
    assert resolved["second_dist_um"].min() > 3.0

    assert int(id_map["root_agrees"].sum()) == EXPECTED_N_ROOT_AGREES
    assert EXPECTED_N_ROOT_AGREES < EXPECTED_N_RESOLVED, "geometric join must recover cells"

    unresolved = id_map[id_map["nucleus_id"] < 0]
    assert set(unresolved["match_method"]) <= {"no_soma_node", "no_nucleus_within_radius"}


# --- the frame -----------------------------------------------------------------------------

def test_upright_is_a_pure_rotation_about_global_z():
    up = load_upright(UPRIGHT_PATH)
    z = up["z_is_global_z"]
    assert abs(z["coef"][2] - 1.0) < 0.01 and abs(z["coef"][0]) < 0.01 and abs(z["coef"][1]) < 0.01
    assert z["rms_resid_um"] < 2.0
    # Two independent cortical boundaries must agree on the one free angle.
    assert abs(up["pia_fit"]["theta_deg"] - up["wm_fit"]["theta_deg"]) < 0.05
    assert up["theta_deg"] == pytest.approx(EXPECTED_THETA_DEG, abs=1e-3)
    # A rotation has |(a, b)| == 1; it is not imposed by the fit, so it is a real check.
    for key in ("pia_fit", "wm_fit"):
        assert abs(up[key]["norm_ab"] - 1.0) < 0.01

    R = upright_matrix(up["theta_deg"])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_cache_frame_matches_the_recorded_theta(cache):
    import h5py
    import pandas as pd

    up = load_upright(UPRIGHT_PATH)
    with h5py.File(INH_CACHE_PATH, "r") as f:
        assert f.attrs["upright_theta_deg"] == pytest.approx(up["theta_deg"], abs=1e-9)
        soma_origin = f["soma_origin"][:]
        root_id = f["root_id"][:]

    # soma_origin is the *global* soma, so re-deriving the rotation from it and the source file
    # is an end-to-end check that the stored coordinates really are xyz_up - origin.
    R = upright_matrix(up["theta_deg"])
    rows = np.linspace(0, len(root_id) - 1, N_FIDELITY).astype(int)
    for row in rows:
        df = pd.read_csv(os.path.join(SWC_DIR, f"{int(root_id[row])}.csv"),
                         usecols=["type", "x", "y", "z"])
        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        soma = xyz[df["type"].to_numpy() == TYPE_SOMA].mean(axis=0)
        assert np.allclose(soma, soma_origin[row], atol=1e-2)

        lo, hi = int(cache.offsets[row]), int(cache.offsets[row + 1])
        assert hi - lo == len(xyz)
        expected = (xyz @ R.T) - (soma @ R.T)
        assert np.allclose(cache.xyz[lo:hi], expected, atol=1e-2)


def test_pia_is_positive_and_wm_negative(cohort):
    pia = np.asarray(cohort.pia_y, dtype=float)
    wm = np.asarray(cohort.wm_y, dtype=float)
    assert (pia > 0).mean() == 1.0
    assert (wm < 0).mean() > 0.99
    assert (pia - wm).mean() == pytest.approx(695.0, abs=5.0)   # cortical thickness


def test_soma_depth_orders_the_inhibitory_labels(cohort):
    """The rotation was fitted on excitatory cells only, so this is out-of-sample validation.

    Only the ends are asserted, not a full ordering: most inhibitory classes are translaminar
    and have no expected depth rank. L1 and ChC are superficial by definition; the descending
    (DTC) types are deep.
    """
    depth = np.asarray(cohort.pia_y, dtype=float)
    lab = np.array([cohort.label_names[int(k)] for k in cohort.labels])
    mean = {n: depth[lab == n].mean() for n in cohort.label_names}

    by_depth = sorted(mean, key=mean.get)
    assert mean["L1"] < 150.0 and mean["ChC"] < 150.0
    assert set(by_depth[:2]) == {"L1", "ChC"}
    assert min(mean["DTC"], mean["AltDTC"]) > 450.0
    assert set(by_depth[-2:]) == {"DTC", "AltDTC"}


# --- cache and cells ------------------------------------------------------------------------

def test_cache_is_complete_and_self_consistent(cache):
    assert len(cache.specimens) == EXPECTED_N_RESOLVED
    assert len(cache.offsets) == EXPECTED_N_RESOLVED + 1
    assert int(cache.offsets[-1]) == len(cache.types) == len(cache.parents) == len(cache.xyz)
    assert cache.radius is not None and len(cache.radius) == len(cache.types)
    assert len(cache.row_by_id) == len(cache.specimens)
    assert set(np.unique(cache.types).tolist()) == {TYPE_SOMA, TYPE_AXON, TYPE_BASAL}


def test_cell_invariants(cache):
    n = min(N_INVARIANT, len(cache.specimens))
    for row in range(n):
        cell = cell_from_cache(cache, row)
        m = len(cell.types)
        assert m >= 3
        assert set(np.unique(cell.types).tolist()) <= {TYPE_SOMA, TYPE_AXON, TYPE_BASAL}
        soma = np.flatnonzero(cell.types == TYPE_SOMA)
        roots = np.flatnonzero(cell.parents == -1)
        assert len(soma) == 1 and len(roots) == 1 and roots[0] == soma[0]
        assert cell.parents.min() >= -1 and cell.parents.max() < m
        assert not (cell.parents == np.arange(m)).any()
        assert np.isfinite(cell.xyz).all()
        assert np.abs(cell.xyz[soma[0]]).max() < 1e-4
        # Radius is discarded by default so the two cohorts' morphometrics mean the same thing.
        assert (cell.radius == 1.0).all()
        assert not cell.synthetic_edge.any()


def test_real_radius_is_available_but_off_by_default(cache):
    plain = cell_from_cache(cache, 0)
    real = cell_from_cache(cache, 0, real_radius=True)
    assert (plain.radius == 1.0).all()
    assert real.radius.min() > 0 and real.radius.max() > 1.0


def test_no_apical_anywhere(cache):
    assert not (cache.types == TYPE_APICAL).any()


def test_axon_prevalence_makes_the_axon_modality_scoreable(cache):
    with_axon = 0
    for row in range(len(cache.specimens)):
        lo, hi = int(cache.offsets[row]), int(cache.offsets[row + 1])
        with_axon += bool((cache.types[lo:hi] == TYPE_AXON).any())
    frac = with_axon / len(cache.specimens)
    # 66.5% measured. The excitatory cohort's 0.43% is why full/axon are blocked there.
    assert 0.6 < frac < 0.7


# --- cohort ---------------------------------------------------------------------------------

def test_labels_and_cohort_counts(cohort):
    assert len(cohort) == EXPECTED_N_COHORT
    assert cohort.label_names == sorted(INH_COARSE_CLASSES)
    assert class_counts(cohort) == EXPECTED_COUNTS
    assert sum(EXPECTED_COUNTS.values()) == EXPECTED_N_COHORT
    ids = [c.specimen_id for c in cohort.cells]
    assert len(set(ids)) == len(ids)
    for c, k in zip(cohort.cells, cohort.labels):
        assert cohort.subclass_by_id[c.specimen_id] == cohort.label_names[int(k)]


def test_inh_only_drops_the_excitatory_labelled_contaminants():
    """The mirror image of R-7: 456 exc-labelled cells sit in the inhibitory file."""
    loose = build_inh_cohort(cache_path=INH_CACHE_PATH, labels_path=INH_LABELS_PATH,
                             inh_only=False, log=lambda m: None)
    strict = build_inh_cohort(cache_path=INH_CACHE_PATH, labels_path=INH_LABELS_PATH,
                              inh_only=True, log=lambda m: None)
    assert len(loose) > len(strict)
    extra = set(loose.label_names) - set(strict.label_names)
    assert extra and extra <= {"L2IT", "L3IT", "L4IT", "L5ET", "L5IT", "L5NP", "L6CT", "L6IT"}
