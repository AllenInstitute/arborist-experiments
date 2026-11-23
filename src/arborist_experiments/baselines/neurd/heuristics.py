"""
Created on Mon November 24 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Geometric rules from NEURD's automated proofreading (Celii et al., Nature
2025) evaluated at the root of a rooted subgraph. NEURD flags merge errors
on a soma-rooted skeleton with heuristics on node degree, angles between
adjacent branches and width jumps. Fragments have no soma and TreeSamples
carry no radii, so this module keeps the orientation-free rules:

- high degree: four or more branches meet at the node (NEURD high-degree
  branching / crossovers);
- pass-through: two branches leave the node nearly collinear, i.e. a fiber
  runs straight through it (NEURD cross-roads / crossover match);
- fork: two branches leave the node nearly parallel (NEURD double-back and
  fork-divergence rules, which are angle thresholds relative to the soma);
- bend: a degree-2 node turns sharply (NEURD internal bend).

Each rule is a threshold on one statistic; ``rule_margins`` returns how far
each statistic is past its threshold so sites can be ranked, not just
flagged. NEURD's angle thresholds are relative to the soma direction and do
not transfer directly, so the defaults here are set so that an ordinary
bifurcation (branches 40-90 degrees apart, continuing within 15-45 degrees
of the parent) does not fire; they should be tuned on validation sites.
"""

import numpy as np


def root_directions(sample, comparison_distance=20.0, offset=0.0):
    """
    Unit direction of each branch leaving the root.

    Positions along each root curve are recovered by cumulative sum of its
    finite differences (the root is the origin). The direction runs from the
    point at arc length ``offset`` to the point at arc length
    ``offset + comparison_distance``, or to the curve's end if shorter.

    Parameters
    ----------
    sample : TreeSample
        Rooted subgraph from ``arborist.data.datasets``, built by
        ``build_tree_sample``.
    comparison_distance : float, optional
        Arc length (microns) over which each direction is measured.
        Default is 20.
    offset : float, optional
        Arc length (microns) skipped before measuring, to step past local
        wobble at the node. Default is 0.

    Returns
    -------
    numpy.ndarray
        Shape (n_root_curves, 3) unit vectors; zero rows for curves with no
        extent.
    """
    dirs = []
    for k in sample.root_curve_indices:
        xyz = np.cumsum(np.asarray(sample.curves[k], dtype=np.float64), axis=0)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
        start = xyz[min(np.searchsorted(arc, offset), len(xyz) - 1)]
        end = xyz[min(np.searchsorted(arc, offset + comparison_distance), len(xyz) - 1)]
        v = end - start
        norm = np.linalg.norm(v)
        dirs.append(v / norm if norm > 0 else v)
    return np.asarray(dirs, dtype=np.float64).reshape(-1, 3)


def pairwise_angles(directions):
    """
    Angles in degrees between every pair of direction vectors.

    Parameters
    ----------
    directions : numpy.ndarray
        Shape (n, 3) unit vectors.

    Returns
    -------
    numpy.ndarray
        Shape (n * (n - 1) / 2,) angles in [0, 180]; empty for n < 2.
    """
    n = len(directions)
    if n < 2:
        return np.zeros(0)
    i, j = np.triu_indices(n, k=1)
    cos = np.clip((directions[i] * directions[j]).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def rule_margins(
    sample,
    high_degree_min=4,
    pass_through_angle=165.0,
    fork_angle=30.0,
    bend_angle=80.0,
    comparison_distance=20.0,
    offset=0.0,
):
    """
    Signed margin of each NEURD rule at the root; positive means the rule
    fires.

    Parameters
    ----------
    sample : TreeSample
        Rooted subgraph from ``arborist.data.datasets``, built by
        ``build_tree_sample``.
    high_degree_min : int, optional
        Root degree at and above which the node is flagged. Default is 4.
    pass_through_angle : float, optional
        Minimum angle (degrees) between two branches for them to count as
        one fiber passing straight through the node. Default is 165.
    fork_angle : float, optional
        Maximum angle (degrees) between two branches for them to count as a
        near-parallel fork. Default is 30.
    bend_angle : float, optional
        Minimum turn (degrees) at a degree-2 node to flag it; the two
        branches then meet at an angle below 180 - bend_angle. Default is 80.
    comparison_distance, offset : float, optional
        See ``root_directions``.

    Returns
    -------
    dict
        Keys "high_degree" (branches past the threshold, offset by half a
        branch so the threshold itself fires), "pass_through", "fork",
        "bend" (degrees past the threshold). Rules that do not apply to the
        node's degree are -inf.
    """
    directions = root_directions(sample, comparison_distance, offset)
    degree = len(directions)
    angles = pairwise_angles(directions)
    margins = {
        "high_degree": float(degree - high_degree_min + 0.5),
        "pass_through": -np.inf,
        "fork": -np.inf,
        "bend": -np.inf,
    }
    if degree >= 3:
        margins["pass_through"] = float(angles.max() - pass_through_angle)
        margins["fork"] = float(fork_angle - angles.min())
    elif degree == 2:
        margins["bend"] = float((180.0 - bend_angle) - angles[0])
    return margins
