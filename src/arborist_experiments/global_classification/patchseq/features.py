from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np
import pandas as pd

import networkx as nx
from morphopy.neurontree.NeuronTree import NeuronTree
from morphopy.computation import feature_presentation
from morphopy.computation.feature_presentation import (
    compute_density_maps,
    compute_morphometric_statistics,
    get_persistence,
)
from morphopy.neurontree.utils import angle_between
from persim import PersistenceImager


def _get_path_angles_safe(self):
    root = self.get_root()
    branchpoints = list(self.get_branchpoints())
    if root in branchpoints:
        branchpoints.remove(root)
    tips = set(self.get_tips().tolist())
    G = self.get_graph()
    successors = nx.dfs_successors(G, root)
    nodes_data = G.nodes(data=True)
    path_angle: dict = {}
    for u, v in self.edges():
        if v in branchpoints or v in tips:
            continue
        if v not in successors:
            continue
        e1 = nodes_data[v]["pos"] - nodes_data[u]["pos"]
        for w in successors[v]:
            e2 = nodes_data[w]["pos"] - nodes_data[v]["pos"]
            path_angle[v] = angle_between(e1, e2) * 180.0 / np.pi
    return path_angle


NeuronTree.get_path_angles = _get_path_angles_safe


_EMPTY_SAFE_REDUCTIONS = ("max", "min", "amax", "amin", "mean", "median", "percentile")


class _EmptySafeNumpy:
    """numpy proxy whose reductions return NaN on empty input instead of raising.

    morphopy computes e.g. max_branch_angle as np.max(branch_angles), which raises
    ValueError for an unbranched arbor. That is common in the axon modality (a stub
    axon with no branch points), and dropping those cells silently shrinks the cohort
    in a class-correlated way. The statistic is genuinely undefined there, so NaN --
    which the caller then column-imputes -- keeps the cohort fixed across modalities.
    """

    def __getattr__(self, name):
        attr = getattr(np, name)
        if name not in _EMPTY_SAFE_REDUCTIONS:
            return attr

        def safe(a, *args, **kwargs):
            if np.size(a) == 0:
                return np.nan
            return attr(a, *args, **kwargs)

        return safe


@contextmanager
def _empty_safe_reductions():
    original = feature_presentation.np
    feature_presentation.np = _EmptySafeNumpy()
    try:
        yield
    finally:
        feature_presentation.np = original

# N_WORKERS is re-exported, not used here: the drivers log `features.N_WORKERS`.
from arborist_experiments.utils.cells import N_WORKERS, run_jobs
from .data_loading import Cell, Cohort, load_precomputed_morphometrics
from .modality import filter_cell
from .swc_adapter import cell_to_neurontree


log = logging.getLogger(__name__)


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

SHOLL_RADII = list(np.arange(10.0, 610.0, 10.0))

PERSIM_KWARGS = dict(
    birth_range=(0.0, 500.0),
    pers_range=(0.0, 500.0),
    pixel_size=25.0,
)


@dataclass
class FeatureResult:
    specimen_ids: list[str]
    X: np.ndarray                       # (n_cells, n_features)
    feature_names: list[str]
    skipped: list[tuple[str, str]]      # (specimen_id, reason)


def _build_nt(cell: Cell, modality: str):
    sub = filter_cell(cell, modality)
    if sub is None:
        return None, "no nodes for modality"
    try:
        return cell_to_neurontree(sub), None
    except Exception as e:
        return None, f"neurontree build: {type(e).__name__}: {e}"


# Re-exported under the private name the sibling modules and drivers already import.
_run_jobs = run_jobs


def _job_density(cell: Cell, modality: str, axis: str):
    nt, err = _build_nt(cell, modality)
    if err is not None:
        return cell.specimen_id, None, None, err
    try:
        return cell.specimen_id, _density_projection(nt, axis), None, None
    except Exception as e:
        return cell.specimen_id, None, None, f"density_{axis}: {type(e).__name__}: {e}"


def _job_persistence(cell: Cell, modality: str):
    nt, err = _build_nt(cell, modality)
    if err is not None:
        return cell.specimen_id, None, None, err
    try:
        p: pd.DataFrame = get_persistence(nt)
        b = p["birth"].to_numpy(dtype=float)
        d = p["death"].to_numpy(dtype=float)
        births = np.minimum(b, d)
        deaths = np.maximum(b, d)
        mask = deaths > births
        diagram = np.stack([births[mask], deaths[mask]], axis=1)
        if len(diagram) == 0:
            return cell.specimen_id, None, None, "persistence: empty diagram"
        img = PersistenceImager(**PERSIM_KWARGS).transform(diagram, skew=False)
        return cell.specimen_id, np.asarray(img).ravel(), None, None
    except Exception as e:
        return cell.specimen_id, None, None, f"persistence: {type(e).__name__}: {e}"


def _job_sholl(cell: Cell, modality: str):
    nt, err = _build_nt(cell, modality)
    if err is not None:
        return cell.specimen_id, None, None, err
    try:
        intersections, _intervals = nt.get_sholl_intersection_profile(
            proj="xy", centroid="soma", radii=SHOLL_RADII
        )
        v = np.asarray(intersections, dtype=float)
        if len(v) != len(SHOLL_RADII):
            v = np.pad(v, (0, len(SHOLL_RADII) - len(v)), constant_values=0.0)[:len(SHOLL_RADII)]
        return cell.specimen_id, v, None, None
    except Exception as e:
        return cell.specimen_id, None, None, f"sholl: {type(e).__name__}: {e}"


def _job_morphometrics(cell: Cell, modality: str):
    nt, err = _build_nt(cell, modality)
    if err is not None:
        return cell.specimen_id, None, None, err
    try:
        with _empty_safe_reductions():
            df = compute_morphometric_statistics(nt)
        row = df.iloc[0]
        numeric = row.apply(lambda x: float(x) if isinstance(x, (int, float, np.number)) else np.nan)
        return cell.specimen_id, numeric.to_numpy(dtype=float), list(numeric.index), None
    except Exception as e:
        return cell.specimen_id, None, None, f"morphometrics: {type(e).__name__}: {e}"


def _stack(rows: list[tuple[str, np.ndarray]], names: list[str], skipped) -> FeatureResult:
    ids = [r[0] for r in rows]
    X = np.vstack([r[1] for r in rows]) if rows else np.zeros((0, len(names)))
    return FeatureResult(specimen_ids=ids, X=X, feature_names=names, skipped=skipped)


def _density_projection(nt, axis: str) -> np.ndarray:
    d = compute_density_maps(nt, config_params=DENSITY_CONFIG)
    return np.asarray(d[f"{axis}_proj"]["data"]).ravel()


def _collect_fixed_width(results, label: str, name_fn) -> FeatureResult:
    rows: list[tuple[str, np.ndarray]] = []
    skipped: list[tuple[str, str]] = []
    names: Optional[list[str]] = None
    for sid, v, _local_names, err in results:
        if err is not None:
            skipped.append((sid, err)); continue
        if names is None:
            names = name_fn(len(v))
        elif len(v) != len(names):
            skipped.append((sid, f"{label}: shape mismatch {len(v)} vs {len(names)}"))
            continue
        rows.append((sid, v))
    return _stack(rows, names or [], skipped)


def extract_density(cohort: Cohort, modality: str, axis: str) -> FeatureResult:
    results = _run_jobs(cohort, _job_density, modality, axis)
    return _collect_fixed_width(
        results, f"density_{axis}", lambda n: [f"density_{axis}_{i}" for i in range(n)]
    )


def extract_density_xy(cohort, modality):
    return extract_density(cohort, modality, "xy")


def extract_density_xz(cohort, modality):
    return extract_density(cohort, modality, "xz")


def extract_density_yz(cohort, modality):
    return extract_density(cohort, modality, "yz")


def extract_persistence_image(cohort: Cohort, modality: str) -> FeatureResult:
    results = _run_jobs(cohort, _job_persistence, modality)
    return _collect_fixed_width(
        results, "persistence", lambda n: [f"persim_{i}" for i in range(n)]
    )


def extract_sholl(cohort: Cohort, modality: str) -> FeatureResult:
    results = _run_jobs(cohort, _job_sholl, modality)
    return _collect_fixed_width(
        results, "sholl", lambda n: [f"sholl_r{int(r)}" for r in SHOLL_RADII]
    )


def extract_morphometrics(cohort: Cohort, modality: str) -> FeatureResult:
    rows: list[tuple[str, np.ndarray]] = []
    skipped: list[tuple[str, str]] = []
    names: Optional[list[str]] = None
    for sid, v, local_names, err in _run_jobs(cohort, _job_morphometrics, modality):
        if err is not None:
            skipped.append((sid, err)); continue
        if names is None:
            names = local_names
        elif local_names != names:
            skipped.append((sid, "morphometrics: column mismatch"))
            continue
        rows.append((sid, v))
    if rows:
        X = np.vstack([r[1] for r in rows])
        # Radius is constant (the file has none), so the thickness statistics are constant too.
        col_ok = ~np.all(np.isnan(X), axis=0)
        col_ok &= np.nanstd(np.where(np.isnan(X), np.nan, X), axis=0) > 0
        X = X[:, col_ok]
        names = [n for n, ok in zip(names, col_ok) if ok]
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        return FeatureResult(
            specimen_ids=[r[0] for r in rows], X=X, feature_names=names, skipped=skipped,
        )
    return _stack(rows, names or [], skipped)


def extract_precomputed_morphometrics(cohort: Cohort, met_path: str) -> FeatureResult:
    ids, X_all, feats = load_precomputed_morphometrics(met_path)
    id_to_row = {i: k for k, i in enumerate(ids)}
    cohort_ids = [c.specimen_id for c in cohort.cells]

    rows: list[tuple[str, np.ndarray]] = []
    skipped: list[tuple[str, str]] = []
    for sid in cohort_ids:
        k = id_to_row.get(sid)
        if k is None:
            skipped.append((sid, "precomputed_morphometrics: not in MET table"))
            continue
        rows.append((sid, X_all[k]))
    if not rows:
        return _stack(rows, feats, skipped)
    X = np.vstack([r[1] for r in rows])
    col_ok = ~np.all(np.isnan(X), axis=0)
    X = X[:, col_ok]
    kept_feats = [n for n, ok in zip(feats, col_ok) if ok]
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return FeatureResult(
        specimen_ids=[r[0] for r in rows], X=X, feature_names=kept_feats, skipped=skipped,
    )
