"""Tier A/B/B' morphometrics: the 51 columns that need nothing but an uprighted reconstruction.

Layer-aligned morphometrics need the pia surface and layer drawings; these 51 need only an
uprighted SWC, so they can be computed for any reconstruction in an upright frame. Definitions come
from `neuron_morphology` itself, called the way `skeleton_keys.cmds.process_morphology_features`
calls it, so the columns carry the same names and semantics as skeleton_keys' output:

  * `dimension` at COORD_TYPE.NODE only; bias_x = abs(bias_xyz[0]), bias_y = bias_xyz[1] signed,
    extent_x = width, extent_y = height.
  * `soma_percentile` -> (x, y); `moments_along_max_distance_projection` -> mean and std, apical
    only, measured at COORD_TYPE.BIFURCATION (the upstream default).
  * `total_surface_area`, `mean_diameter`, `calculate_number_of_stems` on dendrites only.
  * basal stem_exit binned down/side/up at theta <= 1/3, <= 2/3, else; axon exit_theta and
    exit_distance from the stem closest to the soma.

Two public functions, because there are two honest ways in. `upright_morphometrics` takes the five
arrays `neuron_morphology` actually reads -- it wants an SWC node table, not a graph -- and is what
a caller holding raw SWC columns should use. `extract_upright_morphometrics` takes graphs and
returns the aligned `list[FeatureResult]` every other extractor here returns.

`extract_upright_morphometrics` is re-exported from `features/__init__.py` *lazily*, through that
module's `__getattr__`. An eager re-export is what is ruled out: importing this module imports
`neuron_morphology`, and nothing else in the package should have to pay for that (nor depend on the
vendored `SimpleTree` shim) to reach a density map. Either import path works; both land here, and the
first attribute access is what pulls the dependency.

Two structural facts a caller inherits:

  * `neuron_morphology.morphology` opens with `from allensdk.core.simple_tree import SimpleTree` and
    `Morphology` subclasses it, so that one import sits underneath every column below. Rather than
    install allensdk for it, `simple_tree.py` beside this file carries the class and the block after
    `# isort: split` registers it in `sys.modules` under the dotted name upstream asks for -- so
    upstream's import statement resolves with no allensdk on the path at all. The two stub parent
    packages exist for the `import allensdk.core.simple_tree` spelling; the `from ... import` spelling
    upstream uses needs only the leaf. This module is the only one anywhere that imports
    `neuron_morphology`, which is why the registration lives here rather than in a caller's discipline.
  * A process pool must be forked, never spawned. That registration mutates the parent's
    `sys.modules`; a forked child inherits it, a spawned one re-imports `neuron_morphology` from
    scratch and dies on an allensdk that is not installed. `map_graphs` is safe (`run_jobs` uses
    `ProcessPoolExecutor` on the default Linux fork context); joblib callers must pass
    `backend="multiprocessing"`.

No modality parameter. Every column names its own compartment -- `axon_extent_y`,
`basal_dendrite_num_branches` -- so a modality filter would either duplicate what the names already
say or silently blank columns that are the point of the row.
"""
import numpy as np
from arborist.skeleton_graph import SkeletonGraph

from arborist_experiments.features.mapping import map_graphs
from arborist_experiments.features.result import FeatureResult
from arborist_experiments.utils.cells import as_cell

# isort: split
# Every import below depends on this block, which is why `# isort: split` is here -- it stops the
# sorter hoisting them above the registration. Idempotent, and a no-op on a second import.
import sys
import types

from arborist_experiments.features import simple_tree

if "allensdk.core.simple_tree" not in sys.modules:
    for _name in ("allensdk", "allensdk.core"):
        if _name not in sys.modules:
            _stub = types.ModuleType(_name)
            _stub.__path__ = []
            sys.modules[_name] = _stub
    sys.modules["allensdk.core.simple_tree"] = simple_tree
    sys.modules["allensdk.core"].simple_tree = simple_tree

from neuron_morphology.constants import APICAL_DENDRITE, AXON, BASAL_DENDRITE
from neuron_morphology.feature_extractor.data import Data
from neuron_morphology.features.branching.bifurcations import num_outer_bifurcations
from neuron_morphology.features.dimension import dimension
from neuron_morphology.features.intrinsic import max_branch_order, num_branches
from neuron_morphology.features.path import (
    early_branch_path,
    max_path_distance,
    mean_contraction,
)
from neuron_morphology.features.size import (
    max_euclidean_distance,
    mean_diameter,
    total_length,
    total_surface_area,
)
from neuron_morphology.features.soma import (
    calculate_number_of_stems,
    calculate_soma_surface,
    calculate_stem_exit_and_distance,
    soma_percentile,
)
from neuron_morphology.features.statistics.moments_along_max_distance_projection import (
    moments_along_max_distance_projection,
)
from neuron_morphology.morphology import Morphology

COMPARTMENTS = (("apical_dendrite", APICAL_DENDRITE), ("basal_dendrite", BASAL_DENDRITE),
                ("axon", AXON))
PER_COMPARTMENT = ("bias_x", "bias_y", "extent_x", "extent_y", "soma_percentile_x",
                   "soma_percentile_y", "num_branches", "max_branch_order", "total_length",
                   "max_euclidean_distance", "max_path_distance", "mean_contraction")
DENDRITE_ONLY = ("total_surface_area", "mean_diameter")
RADIUS_DERIVED = ("apical_dendrite_total_surface_area", "apical_dendrite_mean_diameter",
                  "basal_dendrite_total_surface_area", "basal_dendrite_mean_diameter",
                  "soma_surface_area")


def feature_names():
    names = []
    for comp, _ in COMPARTMENTS:
        names += [f"{comp}_{f}" for f in PER_COMPARTMENT]
        if comp != "axon":
            names += [f"{comp}_{f}" for f in DENDRITE_ONLY]
    names += [
        "basal_dendrite_calculate_number_of_stems",
        "soma_surface_area",
        "apical_dendrite_early_branch_path",
        "apical_dendrite_num_outer_bifurcations",
        "apical_dendrite_mean_moments_along_max_distance_projection",
        "apical_dendrite_std_moments_along_max_distance_projection",
        "basal_dendrite_stem_exit_down",
        "basal_dendrite_stem_exit_side",
        "basal_dendrite_stem_exit_up",
        "axon_exit_theta",
        "axon_exit_distance",
    ]
    return names


FEATURES = feature_names()
INDEX = {n: i for i, n in enumerate(FEATURES)}

_B_SUFFIXES = ("_bias_y", "_extent_y", "_soma_percentile_y", "_stem_exit_down", "_stem_exit_side",
               "_stem_exit_up")

TIER_OF = {}
for n in FEATURES:
    if n.endswith(("_bias_x", "_extent_x", "_soma_percentile_x")):
        TIER_OF[n] = "B'"
    elif n.endswith(_B_SUFFIXES) or n == "axon_exit_theta":
        TIER_OF[n] = "B"
    else:
        TIER_OF[n] = "A"


def build_morphology(types, parents, xyz, radius):
    nodes = [
        {"id": int(i), "type": int(t), "x": float(x), "y": float(y), "z": float(z),
         "radius": float(r), "parent": int(p)}
        for i, (t, p, (x, y, z), r) in enumerate(zip(types, parents, xyz, radius))
    ]
    return Morphology(nodes, node_id_cb=lambda n: n["id"],
                      parent_id_cb=lambda n: n["parent"])


def upright_morphometrics(types, parents, xyz, radius, synthetic=None) -> np.ndarray:
    """The 51 tier A/B/B' values for one cell, NaN where there is nothing to measure.

    NaN is a compartment the cell does not have, or a statistic with no support on the cable it does
    have -- `mean_contraction` on an unbranched arbor, for instance. It never means a measurement
    that failed, so a consumer must not impute it without deciding which of the two it is imputing.

    `synthetic` marks nodes whose parent link was invented by `reparent_to_soma`. The three
    soma-attachment features are measured on the un-reparented graph, because a fabricated link
    does not just add cable there -- it makes the fragment root look like a genuine stem.
    `calculate_number_of_stems` counts it (inflating the basal count ~5x on fragmented cells) and
    `calculate_stem_exit_and_distance` reports distance 0 for it, which then wins the axon
    "closest stem" contest. Every other feature filters on node_types and so never sees a
    synthetic link, whose parent is always the soma.
    """
    row = np.full(len(FEATURES), np.nan)
    data = Data(build_morphology(types, parents, xyz, radius))
    if synthetic is not None and synthetic.any():
        attach_parents = parents.copy()
        attach_parents[synthetic] = -1
        attach = Data(build_morphology(types, attach_parents, xyz, radius))
    else:
        attach = data

    for comp, nt in COMPARTMENTS:
        if not np.any(types == nt):
            continue
        ntl = [nt]
        dim = dimension(data, node_types=ntl)
        row[INDEX[f"{comp}_bias_x"]] = abs(dim["bias_xyz"][0])
        row[INDEX[f"{comp}_bias_y"]] = dim["bias_xyz"][1]
        row[INDEX[f"{comp}_extent_x"]] = dim["width"]
        row[INDEX[f"{comp}_extent_y"]] = dim["height"]
        pct = soma_percentile(data, node_types=ntl)
        row[INDEX[f"{comp}_soma_percentile_x"]] = pct[0]
        row[INDEX[f"{comp}_soma_percentile_y"]] = pct[1]
        row[INDEX[f"{comp}_num_branches"]] = num_branches(data, node_types=ntl)
        row[INDEX[f"{comp}_max_branch_order"]] = max_branch_order(data, node_types=ntl)
        row[INDEX[f"{comp}_total_length"]] = total_length(data, node_types=ntl)
        row[INDEX[f"{comp}_max_euclidean_distance"]] = max_euclidean_distance(
            data, node_types=ntl)
        row[INDEX[f"{comp}_max_path_distance"]] = max_path_distance(data, node_types=ntl)
        row[INDEX[f"{comp}_mean_contraction"]] = mean_contraction(data, node_types=ntl)
        if comp != "axon":
            row[INDEX[f"{comp}_total_surface_area"]] = total_surface_area(data, node_types=ntl)
            row[INDEX[f"{comp}_mean_diameter"]] = mean_diameter(data, node_types=ntl)

    row[INDEX["soma_surface_area"]] = calculate_soma_surface(data)

    if np.any(types == BASAL_DENDRITE):
        row[INDEX["basal_dendrite_calculate_number_of_stems"]] = calculate_number_of_stems(
            attach, node_types=[BASAL_DENDRITE])
        stem = calculate_stem_exit_and_distance(attach, node_types=[BASAL_DENDRITE])
        if stem:
            # if/elif/else, not three independent sums: neuron_morphology returns NaN theta
            # when a stem root sits exactly on the soma centroid, and upstream's chain sends
            # NaN to `up` (every comparison is False). Independent sums would drop it and the
            # three fractions would not sum to 1.
            down = side = up = 0
            for theta, _ in stem:
                if theta <= 1 / 3:
                    down += 1
                elif theta <= 2 / 3:
                    side += 1
                else:
                    up += 1
            n = len(stem)
            row[INDEX["basal_dendrite_stem_exit_down"]] = down / n
            row[INDEX["basal_dendrite_stem_exit_side"]] = side / n
            row[INDEX["basal_dendrite_stem_exit_up"]] = up / n

    if np.any(types == APICAL_DENDRITE):
        row[INDEX["apical_dendrite_early_branch_path"]] = early_branch_path(
            data, node_types=[APICAL_DENDRITE])
        row[INDEX["apical_dendrite_num_outer_bifurcations"]] = num_outer_bifurcations(
            data, node_types=[APICAL_DENDRITE])
        mom = moments_along_max_distance_projection(data, node_types=[APICAL_DENDRITE])
        row[INDEX["apical_dendrite_mean_moments_along_max_distance_projection"]] = mom["mean"]
        row[INDEX["apical_dendrite_std_moments_along_max_distance_projection"]] = mom["std"]

    if np.any(types == AXON):
        ax = calculate_stem_exit_and_distance(attach, node_types=[AXON])
        if ax:
            # Upstream's explicit scan from inf rather than min(): same result on ties, but it
            # also matches upstream if a distance is ever NaN.
            closest_distance, closest_theta = np.inf, 0.0
            for theta, dist in ax:
                if dist < closest_distance:
                    closest_distance, closest_theta = dist, theta
            row[INDEX["axon_exit_theta"]] = closest_theta
            row[INDEX["axon_exit_distance"]] = closest_distance
    return row


# `radius` because five columns are diameters and surface areas, and a soma because every one of the
# 51 is measured from it: `dimension` centres on `get_root()`, `soma_percentile` compares against it,
# and `calculate_stem_exit_and_distance` reads `get_soma()`. That root is also what the reparenting
# below establishes -- upstream needs one root, and a fragmented reconstruction has many.
#
# `soma_at_origin` is deliberately *not* required, unlike the radial features': every column is
# measured relative to the root node rather than to the origin, so all 51 are translation-invariant
# and a cell in a foreign frame would still be measured correctly. What tiers B and B' additionally
# assume -- that +y points toward pia -- cannot be checked from one graph, so it is stated here
# rather than pretended to be a check.
REQ = {"keys": ("type", "radius"), "needs_soma": True}


def _job_upright_morphometrics(graph):
    # Reparenting here rather than in `map_graphs`' shared `as_cell` call because it is this
    # feature's decision: upstream needs a single root, and the invented links are exactly what the
    # `synthetic` mask then excludes from the soma-attachment columns. `as_cell` ORs the mask, so a
    # cell whose loader already reparented keeps its own.
    cell = as_cell(graph, reparent_orphans=True)
    feats = cell.node_feats
    row = upright_morphometrics(feats["type"], feats["parent"], cell.node_xyz, feats["radius"],
                                feats["synthetic_edge"])
    return row, FEATURES, None


def extract_upright_morphometrics(graphs: list[SkeletonGraph]) -> list[FeatureResult]:
    """One `FeatureResult` per input graph, in order, 51 columns wide.

    The width is known before any graph is read, so a skipped element carries the column names too
    and an all-skipped batch still reports a width.

    There is no per-graph runtime reason here: a cell that satisfies `REQ` always yields a row, NaN
    where it gives upstream nothing to measure -- which is a measurement, not a skip. Nothing is
    caught, so a `neuron_morphology` failure propagates instead of becoming a silently missing row.
    """
    return [
        FeatureResult(X=v, feature_names=FEATURES, skipped=err)
        for v, _local_names, err in map_graphs(graphs, REQ, _job_upright_morphometrics)
    ]
