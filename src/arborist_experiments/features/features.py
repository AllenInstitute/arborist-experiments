"""The MorphoPy-derived representations.

Density projections, persistence images, Sholl profiles and morphometric statistics. Everything
that varies by cohort arrives as an argument -- the bin edges, the imager kwargs, the Sholl radii --
so nothing here knows which dataset it is running on.

Each extractor names the constants it reads and no others, which is why none of them takes a
`BinningConfig`: a driver holding one unpacks it at the call site. `extract_morphometrics` takes no
constant at all, because morphopy's morphometric statistics are lengths and angles rather than
histograms and no bin edge reaches them.

`_build_nt` is looked up as a module global on purpose: a caller can rebind it to swap in a
different graph builder for every extractor at once, and fork carries that into the workers.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

import networkx as nx
import numpy as np
import pandas as pd
from morphopy.computation import feature_presentation
from morphopy.computation.feature_presentation import (
    compute_density_maps,
    compute_morphometric_statistics,
    get_persistence,
)
from morphopy.neurontree.NeuronTree import NeuronTree
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

from arborist.skeleton_graph import SkeletonGraph

from .mapping import map_graphs
from .modality import filter_cell
from .result import FeatureResult, batch_names, usable
from .swc_adapter import cell_to_neurontree

log = logging.getLogger(__name__)

# What these four ask of a graph, checked in the parent before it is pickled. Every one of them goes
# through `filter_cell` -> `filtered` -> `cell_to_neurontree` -> `cell_to_swc_df`, which reads `type`
# (nothing else can drive a modality filter) and `radius` (it reaches morphopy's thickness, surface
# and volume statistics). A dataset without measured radii must set one explicitly in its loader;
# an extractor does not invent it.
REQ = {"keys": ("type", "radius")}

# Sholl and morphometrics additionally measure from the soma outwards --
# `get_sholl_intersection_profile(centroid="soma")` -- so a graph without one cannot be used. Density
# and persistence need neither a soma node nor a particular frame.
REQ_SOMA = {"keys": ("type", "radius"), "needs_soma": True}

# morphopy's own column order, and there is nothing to derive it from but this list. It is fixed in
# code rather than by data: `compute_morphometric_statistics` fills one dict in exactly this order and
# returns `pd.DataFrame.from_dict(z, orient="index").T`, and the one conditional it contains -- the
# `path_angles` branch -- writes the same three keys either way, so no cell can permute the index or
# shorten it. `_job_morphometrics` raises rather than adapting if a morphopy version ever disagrees,
# which is the point of declaring it: read off whichever cell ran first, a changed order would
# silently relabel every column.
MORPHOMETRIC_FEATURES = [
    "branch_points", "width", "depth", "height", "tips", "stems", "total_length",
    "avg_thickness", "max_thickness", "total_surface", "total_volume",
    "max_path_dist_to_soma", "max_branch_order",
    "max_path_angle", "min_path_angle", "median_path_angle", "mean_soma_exit_angle",
    "max_segment_path_length", "median_intermediate_segment_pl", "median_terminal_segment_pl",
    "log_max_tortuosity", "log_min_tortuosity", "log_median_tortuosity",
    "max_branch_angle", "min_branch_angle", "mean_branch_angle",
    "max_degree", "tree_asymmetry",
]

# Which pair of axes each density projection bins, in morphopy's order -- `compute_density_maps`
# builds `bins["xy"] = (n_bins_x, n_bins_y)` and `np.histogramdd` returns that shape, so the raveled
# vector is row-major over the first axis of the pair.
DENSITY_AXES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}


def _axis_ranges(axis: str, x_range, y_range, z_range) -> tuple[tuple[float, float], ...]:
    """The two ranges the projection reads, raising by name for either one that is missing.

    A missing range is not a defaultable argument: morphopy falls back to *that cell's* own extent, so
    both the width and the values become properties of the cell and no two cells are comparable. The
    third range is not required, because a projection does not read it -- passing one is not an error,
    it just does nothing, which is why it is not rejected either.
    """
    if axis not in DENSITY_AXES:
        raise ValueError(f"unknown projection {axis!r}, expected one of {sorted(DENSITY_AXES)}")
    given = {"x": x_range, "y": y_range, "z": z_range}
    out = []
    for ax in DENSITY_AXES[axis]:
        if given[ax] is None:
            raise ValueError(
                f"the {axis} projection bins over {ax}, so it needs {ax}_range -- without it morphopy "
                f"would bin this axis over each cell's own extent and no two cells would be comparable")
        lo, hi = given[ax]
        out.append((float(lo), float(hi)))
    return tuple(out)


def _n_bins(axis_ranges, bin_size: float, n_bins) -> tuple[int, int]:
    """morphopy's bin counts for the projection's two axes, known without a morphology.

    `compute_density_maps` computes `np.ceil((max - min) / bin_size)` per axis and lets an explicit
    `n_bins_<ax>` override it -- that is the whole rule, and it is reproduced rather than called
    because the function it lives in wants a morphology before it will tell anyone the answer.

    `n_bins` overrides the count only. It does not remove the need for a range: morphopy still takes
    the histogram's `range=` from the config or the cell, so a count without a range would fix the
    width while leaving the values per-cell. `_axis_ranges` requires them either way.
    """
    if n_bins is not None:
        a, b = n_bins
        return int(a), int(b)
    return tuple(int(np.ceil((hi - lo) / bin_size)) for lo, hi in axis_ranges)


def density_feature_names(
    axis: str, *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    bin_size: float = 20.0,
    n_bins: tuple[int, int] | None = None,
) -> list[str]:
    """The column names of a density projection, known before any graph is read.

    The width is the product of the two axes' bin counts, which depends on the ranges, so there is no
    fixed constant to check it against; `_job_density` checks every row it computes instead.

    Takes `extract_density`'s geometry parameters and not its value-only ones (`density`, `smooth`,
    `sigma`, `resample_distance`): none of those move a column count, so accepting them would only
    mislead. A caller holding a `BinningConfig` names the geometry rather than splatting the bundle
    in.
    """
    a, b = _n_bins(_axis_ranges(axis, x_range, y_range, z_range), bin_size, n_bins)
    return [f"density_{axis}_{i}" for i in range(a * b)]


def _morphopy_config(axis, x_range, y_range, z_range, bin_size, n_bins, density, smooth, sigma,
                     resample_distance) -> dict:
    """The explicit parameters reassembled as the `config_params` dict morphopy reads.

    Rebuilt here rather than taken from a caller so that the extractor's signature is the list of
    constants it reads. Every key morphopy looks up is written, with no reliance on its own defaults:
    it reaches all of them through `config_params.get(name, default)`, so a key this function forgot
    would be silently defaulted rather than raise.

    The range for the axis this projection does *not* read is passed through when the caller gave one.
    `compute_density_maps` computes all six projections on every call and this one keeps just the
    requested one, so the third range cannot reach the returned values -- but it does set the bin count
    of the projections that get discarded, and a cell that is flat along an unranged axis would bin it
    into `ceil(0 / bin_size) = 0` bins. Dropping the third range would turn that into a raise inside
    morphopy for a cell that otherwise works.
    """
    cfg = {
        "density": density,
        "smooth": smooth,
        "sigma": sigma,
        "bin_size": bin_size,
        "distance": resample_distance,
    }
    for ax, given in (("x", x_range), ("y", y_range), ("z", z_range)):
        if given is not None:
            cfg[f"r_min_{ax}"], cfg[f"r_max_{ax}"] = float(given[0]), float(given[1])
    if n_bins is not None:
        for ax, n in zip(DENSITY_AXES[axis], n_bins):
            cfg[f"n_bins_{ax}"] = int(n)
    return cfg


def persistence_feature_names(
    *,
    birth_range: tuple[float, float],
    pers_range: tuple[float, float],
    pixel_size: float,
) -> list[str]:
    """The column names of a persistence image, known before any graph is read.

    persim is asked rather than imitated: `PersistenceImager(...).resolution` is the pixel grid it
    will produce, derived from these three parameters by the same code that then fills it. Note that
    persim truncates rather than rounds --
    `int(width / pixel_size)` -- so a range that is not a whole number of pixels loses the remainder
    off the top edge, which is a good enough reason not to reimplement the line.

    These are `extract_persistence_image`'s pixel-grid parameters and not its four value-only ones
    (`weight`, `weight_params`, `kernel`, `kernel_params`): none of those move a column count, and the
    same split holds for `density_feature_names`.
    """
    nx, ny = PersistenceImager(
        birth_range=birth_range, pers_range=pers_range, pixel_size=pixel_size).resolution
    return [f"persim_{i}" for i in range(nx * ny)]


def _check_width(v: np.ndarray, n_expected: int, what: str) -> None:
    """Raise if morphopy's actual width disagrees with the one declared from the config.

    Outside every `except` in this module on purpose. A declared width that is wrong is a bug in the
    declaration, and the two ways of absorbing it are both silent: as a per-graph reason it empties
    the cohort one cell at a time, and as a `design_matrix` width disagreement it drops every row.
    Cheap enough to run per cell, which is what makes it fail on the first worker to finish rather
    than after a whole extraction.
    """
    if len(v) != n_expected:
        raise ValueError(
            f"{what}: got {len(v)} values, the config declares {n_expected} -- the width rule in "
            f"this module and the one in the library have diverged")


def _build_nt(cell: SkeletonGraph, modality: str):
    sub = filter_cell(cell, modality)
    if sub is None:
        return None, "no nodes for modality"
    try:
        return cell_to_neurontree(sub), None
    except Exception as e:
        return None, f"neurontree build: {type(e).__name__}: {e}"


def _job_density(cell: SkeletonGraph, modality: str, config: dict, axis: str,
                 n_expected: int):
    nt, err = _build_nt(cell, modality)
    if err is not None:
        return None, None, err
    try:
        v = _density_projection(nt, config, axis)
    except Exception as e:
        return None, None, f"density_{axis}: {type(e).__name__}: {e}"
    _check_width(v, n_expected, f"density_{axis}")
    return v, None, None


def _job_persistence(cell: SkeletonGraph, modality: str, persim_kwargs: dict, n_expected: int):
    nt, err = _build_nt(cell, modality)
    if err is not None:
        return None, None, err
    try:
        p: pd.DataFrame = get_persistence(nt)
        b = p["birth"].to_numpy(dtype=float)
        d = p["death"].to_numpy(dtype=float)
        births = np.minimum(b, d)
        deaths = np.maximum(b, d)
        mask = deaths > births
        diagram = np.stack([births[mask], deaths[mask]], axis=1)
        if len(diagram) == 0:
            return None, None, "persistence: empty diagram"
        img = PersistenceImager(**persim_kwargs).transform(diagram, skew=False)
        v = np.asarray(img).ravel()
    except Exception as e:
        return None, None, f"persistence: {type(e).__name__}: {e}"
    _check_width(v, n_expected, "persistence")
    return v, None, None


def _job_sholl(cell: SkeletonGraph, modality: str, radii: list[float]):
    nt, err = _build_nt(cell, modality)
    if err is not None:
        return None, None, err
    try:
        intersections, _intervals = nt.get_sholl_intersection_profile(
            proj="xy", centroid="soma", radii=radii
        )
        v = np.asarray(intersections, dtype=float)
        if len(v) != len(radii):
            v = np.pad(v, (0, len(radii) - len(v)), constant_values=0.0)[:len(radii)]
        return v, None, None
    except Exception as e:
        return None, None, f"sholl: {type(e).__name__}: {e}"


def _job_morphometrics(cell: SkeletonGraph, modality: str):
    nt, err = _build_nt(cell, modality)
    if err is not None:
        return None, None, err
    try:
        with _empty_safe_reductions():
            df = compute_morphometric_statistics(nt)
        row = df.iloc[0]
        numeric = row.apply(lambda x: float(x) if isinstance(x, (int, float, np.number)) else np.nan)
    except Exception as e:
        return None, None, f"morphometrics: {type(e).__name__}: {e}"
    if list(numeric.index) != MORPHOMETRIC_FEATURES:
        # Outside the `except` for the same reason `_check_width` is: absorbing this as a per-graph
        # reason would empty the cohort silently, and absorbing it as a per-row column set would let
        # a renamed or reordered statistic through under the old name.
        raise ValueError(
            f"morphometrics: morphopy returned {list(numeric.index)}, this module declares "
            f"{MORPHOMETRIC_FEATURES}")
    return numeric.to_numpy(dtype=float), None, None


def _density_projection(nt, density_config: dict, axis: str) -> np.ndarray:
    d = compute_density_maps(nt, config_params=density_config)
    return np.asarray(d[f"{axis}_proj"]["data"]).ravel()


def _aligned(results, names: list[str]) -> list[FeatureResult]:
    """One `FeatureResult` per graph, every element carrying the declared column names.

    Skipped elements included, which is what lets an all-skipped batch still report its width. One
    list object is shared by all of them, so a 48k-cell cohort holds one copy of the names.
    """
    return [
        FeatureResult(X=v, feature_names=names, skipped=err)
        for v, _local_names, err in results
    ]


def extract_density(
    graphs: list[SkeletonGraph], modality: str, axis: str, *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    bin_size: float = 20.0,
    n_bins: tuple[int, int] | None = None,
    density: bool = True,
    smooth: bool = True,
    sigma: float = 1.0,
    resample_distance: float = 1.0,
) -> list[FeatureResult]:
    """A 2-D density projection of the arbor, one `FeatureResult` per input graph.

    `axis` is `"xy"`, `"xz"` or `"yz"`, and the two ranges it names are required -- see `_axis_ranges`
    for why a range has no usable default. Every other parameter is morphopy's own default:

      * `bin_size`      bin width in microns along both axes; the count is `ceil(range / bin_size)`.
      * `n_bins`        `(n_a, n_b)` overriding that count for the projection's two axes.
      * `density`       normalise to a probability density (`np.histogramdd(density=...)`) rather
                        than return raw counts, so values scale with bin volume.
      * `smooth`,`sigma`  Gaussian-filter the histogram afterwards; `sigma` is in BINS, not microns.
      * `resample_distance`  the step, in microns, at which the arbor is resampled into the point
                        cloud that gets binned. Affects every value and nothing about the width.

    A `BinningConfig` stores these as one `density_config` dict, morphopy's `config_params`;
    `BinningConfig.density_kwargs()` unpacks it into these parameters.

    The width comes from these parameters, not from the first cell that succeeded, so a batch in
    which every graph was skipped still reports one. `density_feature_names` reproduces morphopy's bin
    rule rather than calling it, so every row that is computed is checked against the declared width.
    """
    names = density_feature_names(axis, x_range=x_range, y_range=y_range, z_range=z_range,
                                  bin_size=bin_size, n_bins=n_bins)
    config = _morphopy_config(axis, x_range, y_range, z_range, bin_size, n_bins,
                              density, smooth, sigma, resample_distance)
    return _aligned(
        map_graphs(graphs, REQ, _job_density, modality, config, axis, len(names)),
        names,
    )


def extract_persistence_image(
    graphs: list[SkeletonGraph], modality: str, *,
    birth_range: tuple[float, float],
    pers_range: tuple[float, float],
    pixel_size: float,
    weight=None,
    weight_params: dict | None = None,
    kernel=None,
    kernel_params: dict | None = None,
) -> list[FeatureResult]:
    """A persistence image of the arbor's barcode, one `FeatureResult` per input graph.

    These are `PersistenceImager`'s own parameters, forwarded to it. The three that fix the pixel grid
    have no default because persim's own -- `(0.0, 1.0)` for both ranges, `0.2` for `pixel_size` -- are
    for unit-scale diagrams: a neuron's births and deaths are path distances in microns, so accepting
    that default would put every cell's whole diagram outside a 1 um image and hand back 25 zeros.
    Being told beats being guessed at here, which is also why the density projections' ranges are
    required.

      * `birth_range`, `pers_range`  the image's box, in the same microns the barcode is in. persim
                        skews death to persistence itself when asked; this module passes `skew=False`
                        to `transform` and hands it (birth, death) pairs, so `pers_range` is read
                        against DEATH.
      * `pixel_size`    the side of one square pixel, in microns. The grid is
                        `int(width / pixel_size)` by `int(height / pixel_size)` -- truncated.
      * `weight`, `weight_params`  how much each point of the diagram contributes; persim's default is
                        `pers ** n` with `{"n": 1.0}`, and its `pers` is the second coordinate -- so
                        under `skew=False` a point is weighted by its death, not by its persistence.
      * `kernel`, `kernel_params`  the density each point is smeared with; persim's default is a
                        Gaussian with `sigma = [[1, 0], [0, 1]]` -- in microns SQUARED, being a
                        covariance rather than a width.

    All four value-only parameters are forwarded as-is, `None` included, because persim reads `None`
    as "use my default" -- so passing none of them gives persim's defaults. Two constraints
    on the callables that are not persim's: they are pickled to the extraction workers, so a lambda
    will not survive the trip, and `feature_cache.constants_digest` json-dumps `cfg.persim_kwargs`, so
    a callable stored in a cohort config would break cache identity rather than change it.

    A `BinningConfig` stores these as one `persim_kwargs` dict; `BinningConfig.persistence_kwargs()`
    unpacks it into these parameters.

    The width comes from those three parameters: the same numbers asked of
    `PersistenceImager.resolution` give it before any graph is read, so a skipped element carries the
    names too. An empty diagram is a per-graph reason, not a width question.
    """
    names = persistence_feature_names(
        birth_range=birth_range, pers_range=pers_range, pixel_size=pixel_size)
    persim_kwargs = {
        "birth_range": birth_range, "pers_range": pers_range, "pixel_size": pixel_size,
        "weight": weight, "weight_params": weight_params,
        "kernel": kernel, "kernel_params": kernel_params,
    }
    return _aligned(
        map_graphs(graphs, REQ, _job_persistence, modality, persim_kwargs, len(names)),
        names,
    )


def extract_sholl(
    graphs: list[SkeletonGraph], modality: str, *, sholl_radii: list[float]
) -> list[FeatureResult]:
    """Sholl intersections, one `FeatureResult` per graph.

    Unlike density and persistence this knows its width upfront -- it is `sholl_radii`, and
    `_job_sholl` pads or truncates to it -- so a skipped element carries the names too and an
    all-skipped batch still reports a width. That is also the whole of this feature's dependence on
    its cohort: one list of radii in microns.
    """
    names = [f"sholl_r{int(r)}" for r in sholl_radii]
    return _aligned(
        map_graphs(graphs, REQ_SOMA, _job_sholl, modality, sholl_radii), names)


def extract_morphometrics(
    graphs: list[SkeletonGraph], modality: str
) -> list[FeatureResult]:
    """One `FeatureResult` per input graph, `MORPHOMETRIC_FEATURES` wide, with its NaNs intact.

    The columns are declared rather than taken from whichever cell ran first, so a skipped
    element carries them too and an all-skipped batch still has a width. A cell whose statistics come
    back under different names is a raise, not a row -- see `MORPHOMETRIC_FEATURES`.

    Takes no cohort constant, which is a fact about the feature rather than an omission: morphopy's
    morphometric statistics are lengths, counts and angles, so no bin edge reaches them and this row
    is comparable across datasets without any of them agreeing on an envelope.

    Dropping zero-variance columns and imputing NaNs depend on the whole cohort, so they are not done
    here: call `drop_zero_variance_and_impute` once over the cohort.
    """
    return _aligned(
        map_graphs(graphs, REQ_SOMA, _job_morphometrics, modality),
        list(MORPHOMETRIC_FEATURES))


def drop_zero_variance_and_impute(results: list[FeatureResult]) -> list[FeatureResult]:
    """Drop columns that are constant across the cohort, then fill NaNs with column means.

    Cohort-level by nature, which is why it is not inside `extract_morphometrics`: both the surviving
    column set and the imputed values depend on which graphs were passed, so a caller that extracts
    in chunks must run this once over the whole cohort or get a different feature space per chunk.

    Only morphometrics wants it. For density and radial_hist an all-zero block is a true zero
    density rather than a missing measurement, and dropping those columns would make the width
    depend on the cohort.
    """
    names = batch_names(results)
    live = [r for r in results if usable(r, names)]
    if not live:
        return list(results)

    X = np.vstack([r.X for r in live])
    # A dataset with constant radius makes the thickness statistics constant, for example.
    col_ok = ~np.all(np.isnan(X), axis=0)
    col_ok &= np.nanstd(np.where(np.isnan(X), np.nan, X), axis=0) > 0
    X = X[:, col_ok]
    kept = [n for n, ok in zip(names, col_ok) if ok]
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    out, k = [], 0
    for r in results:
        if usable(r, names):
            out.append(FeatureResult(X=X[k], feature_names=kept, skipped=None))
            k += 1
        elif r.X is None:
            # A skipped element carries the surviving column set too, so `batch_names` reads the
            # same width whichever element it reaches first.
            out.append(FeatureResult(X=None, feature_names=kept, skipped=r.skipped))
        else:
            out.append(r)
    return out
