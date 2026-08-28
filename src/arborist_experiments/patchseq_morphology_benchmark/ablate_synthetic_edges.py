"""Rerun the MorphoPy-based representations with the fabricated soma links removed.

`_reparent_to_soma` invents a parent link from the soma to every disconnected fragment so
morphopy sees one connected tree. `cell_to_swc_df` does not carry the `synthetic_edge` flag,
so those links reach morphopy as traced cable. Measured on the dendrite modality: they are
0.4% of edges but 29.4% of cable length, 31.9% of the point cloud `compute_density_maps` bins
(they are the only edges long enough for `resample_nodes(d=1)` to interpolate), and they
inflate the Sholl profile 1.52x, mostly in the outer half where the fabricated 96 um spokes
cross every circle.

Two arms, because morphopy's features split on whether they need a connected tree:

  prune    -- drop the synthetic edges from the graph and leave the forest. `compute_density_maps`
              and `get_sholl_intersection_profile` only iterate edges, so this is exact: no real
              cable is lost and nothing else changes. Edges are removed after NeuronTree
              construction so `t[pid == -1] = 1` has already run and no fragment root gets
              relabelled as soma.
  somacomp -- drop them, then keep only the soma's connected component. Required by
              `get_persistence` and `compute_morphometric_statistics`, which walk from the root
              and raise IndexError on a forest. This is NOT a clean ablation: it also discards
              real traced cable, 20% cohort-wide and strongly class-correlated (L5 PT retains
              56%, L2/3 IT 99%), and drops 21 cells whose soma component has under 3 nodes.
              density_xy and sholl are run in this arm too, as controls: comparing them against
              the prune arm prices the fragment loss for representations where both are possible.

Baseline numbers are not recomputed -- scratch/benchmarks/v2 used the same cohort, seed and CV
settings, and the radial ablation reproduced its radial_arbors row to 4 decimals.

Dendrite modality only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter

import networkx as nx
import numpy as np

from arborist_experiments.morphology_cv import classifier
from arborist_experiments.patchseq_morphology_benchmark import features as F
from arborist_experiments.morphology_cv.classifier import run_cv
from arborist_experiments.patchseq_morphology_benchmark.data_loading import Cell, build_cohort
from arborist_experiments.patchseq_morphology_benchmark.modality import filter_cell
from arborist_experiments.cells.plots import plot_confusion, plot_embedding
from arborist_experiments.patchseq_morphology_benchmark.swc_adapter import cell_to_neurontree
from arborist_experiments.patchseq_morphology_benchmark.run_benchmark import (
    _log, _next_version_dir, _write_rows,
)


EXTRACTORS = {
    "density_xy":              F.extract_density_xy,
    "density_xz":              F.extract_density_xz,
    "density_yz":              F.extract_density_yz,
    "sholl":                   F.extract_sholl,
    "persistence_image":       F.extract_persistence_image,
    "morphometric_statistics": F.extract_morphometrics,
}

ARMS = {
    "prune":    ["density_xy", "density_xz", "density_yz", "sholl"],
    "somacomp": ["persistence_image", "morphometric_statistics", "density_xy", "sholl"],
}

SMALL_COMPONENT = "soma component < 3 nodes"


def _synthetic_edges(sub: Cell) -> list[tuple[int, int]]:
    """Synthetic edges as NeuronTree node-id pairs. cell_to_swc_df numbers nodes 1..N in order."""
    child = np.flatnonzero((sub.parents != -1) & sub.synthetic_edge)
    return [(int(sub.parents[i]) + 1, int(i) + 1) for i in child]


def _build_nt_prune(cell: Cell, modality: str):
    sub = filter_cell(cell, modality)
    if sub is None:
        return None, "no nodes for modality"
    try:
        nt = cell_to_neurontree(sub)
        nt.get_graph().remove_edges_from(_synthetic_edges(sub))
        return nt, None
    except Exception as e:
        return None, f"neurontree build: {type(e).__name__}: {e}"


def _build_nt_somacomp(cell: Cell, modality: str):
    sub = filter_cell(cell, modality)
    if sub is None:
        return None, "no nodes for modality"
    try:
        child = np.flatnonzero(sub.parents != -1)
        real = child[~sub.synthetic_edge[child]]
        g = nx.Graph()
        g.add_nodes_from(range(len(sub.types)))
        g.add_edges_from([(int(sub.parents[i]), int(i)) for i in real])
        soma = np.flatnonzero(sub.types == 1)
        keep = np.zeros(len(sub.types), dtype=bool)
        keep[sorted(nx.node_connected_component(g, int(soma[0]) if len(soma) else 0))] = True
        if keep.sum() < 3:
            return None, SMALL_COMPONENT

        old_to_new = -np.ones(len(sub.types), dtype=np.int64)
        old_to_new[keep] = np.arange(int(keep.sum()))
        parents = np.array(
            [-1 if (sub.parents[i] == -1 or not keep[sub.parents[i]])
             else old_to_new[sub.parents[i]] for i in np.flatnonzero(keep)],
            dtype=np.int64,
        )
        pruned = Cell.from_arrays(
            specimen_id=sub.specimen_id,
            types=sub.types[keep].copy(),
            parents=parents,
            xyz=sub.xyz[keep].copy(),
            radius=sub.radius[keep].copy(),
            synthetic_edge=np.zeros(int(keep.sum()), dtype=bool),
        )
        if int((parents == -1).sum()) != 1:
            return None, f"soma component has {int((parents == -1).sum())} roots"
        return cell_to_neurontree(pruned), None
    except Exception as e:
        return None, f"neurontree build: {type(e).__name__}: {e}"


BUILDERS = {"prune": _build_nt_prune, "somacomp": _build_nt_somacomp}


def _mirror_ablation(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            dst = f"{PREFIX}_{name}" if name in GENERIC_NAMES else name
            shutil.copy2(src, os.path.join(dst_dir, dst))


def _synthetic_inventory(cohort, modality: str) -> dict:
    n_edges = n_syn = n_cells_frag = 0
    for cell in cohort.cells:
        sub = filter_cell(cell, modality)
        if sub is None:
            continue
        child = np.flatnonzero(sub.parents != -1)
        syn = int(sub.synthetic_edge[child].sum())
        n_edges += len(child)
        n_syn += syn
        n_cells_frag += int(syn > 0)
    return {"n_edges": n_edges, "n_synthetic": n_syn, "n_cells_fragmented": n_cells_frag}


def _write_desc(out_dir: str, cohort, inv: dict, summary: dict) -> None:
    lines = ["Rerun of the morphopy-based representations with synthetic (fabricated) "
             "soma-to-fragment edges removed.", ""]
    lines += ["* Parameters",
              f"    modality = {MODALITY}",
              f"    arms = {ARMS!r}",
              f"    outer_k = {OUTER_K}", f"    inner_k = {INNER_K}",
              f"    seed = {SEED}", f"    min_class_size = {MIN_CLASS_SIZE}",
              f"    n_classes = {len(cohort.label_names)}",
              f"    label = MET.meta/subclass", f"    coordinates = swc_upright",
              f"    met_path = {MET_PATH}", f"    swc_path = {SWC_PATH}", ""]
    lines += ["* Arms",
              "    prune    -- synthetic edges removed from the graph, forest kept. Exact for",
              "                compute_density_maps and get_sholl_intersection_profile (edge",
              "                iteration only); no real cable lost.",
              "    somacomp -- synthetic edges removed, then only the soma's connected component",
              "                kept. Required by get_persistence and",
              "                compute_morphometric_statistics, which need a connected tree.",
              "                CONFOUNDED: also discards real cable (20% cohort-wide, and",
              "                class-correlated: L5 PT retains 56%, L2/3 IT 99%) and drops cells",
              "                whose soma component has under 3 nodes. density_xy and sholl run",
              "                here as controls against the prune arm.", ""]
    lines += [f"* {MODALITY} edge inventory",
              f"    edges = {inv['n_edges']:,}",
              f"    synthetic = {inv['n_synthetic']:,} "
              f"({100 * inv['n_synthetic'] / max(1, inv['n_edges']):.2f}%)",
              f"    cells with >=1 synthetic edge = {inv['n_cells_fragmented']}", ""]
    lines += ["* Classifier (identical to run_benchmark.py)",
              "    StandardScaler -> PCA(min(n/2, 200, n_features)) -> elastic-net LogisticRegression",
              "    grid: C in [0.1, 1.0, 10.0], l1_ratio in [0.0, 0.5, 1.0]; scoring neg_log_loss",
              "    metrics: log_loss, accuracy, balanced_accuracy, f1_macro, mcc", ""]
    lines += ["* Baseline for comparison",
              "    /root/capsule/scratch/benchmarks/v2/benchmark_metrics.csv (same cohort, seed,",
              "    CV settings; not recomputed here)", ""]
    lines += ["* Results (balanced_accuracy, mean +/- std over outer folds)"]
    for (arm, rep), (m, s, nf, nc) in summary.items():
        lines.append(f"    {arm:9s} {rep:24s} {nf:5d} feat  n={nc:5d}   {m:.4f} +/- {s:.4f}")
    lines += ["", "* Full metrics", "    see synthetic_ablation_metrics.csv"]
    with open(os.path.join(out_dir, "desc.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(args):
    out_dir = _next_version_dir(ARCHIVE_DIR)
    fh = open(os.path.join(out_dir, "run_log.txt"), "w")
    _log(fh, f"Durable output dir: {out_dir} (mirrored to {RESULTS_DIR})")
    _log(fh, f"Cores: {os.cpu_count()}  cv jobs: {classifier.N_JOBS}  "
             f"feature workers: {F.N_WORKERS}")

    cohort = build_cohort(met_path=MET_PATH, swc_path=SWC_PATH, min_class_size=MIN_CLASS_SIZE)
    id_to_label = {c.specimen_id: int(l) for c, l in zip(cohort.cells, cohort.labels)}
    _log(fh, f"Cohort: n_cells={len(cohort)} n_classes={len(cohort.label_names)}")

    inv = _synthetic_inventory(cohort, MODALITY)
    _log(fh, f"{MODALITY} edges={inv['n_edges']:,} synthetic={inv['n_synthetic']:,} "
             f"({100 * inv['n_synthetic'] / inv['n_edges']:.2f}%) "
             f"fragmented cells={inv['n_cells_fragmented']}")
    if inv["n_synthetic"] == 0:
        raise SystemExit("no synthetic edges in this modality -- nothing to ablate")

    csv_path = os.path.join(out_dir, "synthetic_ablation_metrics.csv")
    summary = {}
    t_start = time.time()
    with open(csv_path, "w", newline="") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["feature_rep", "modality", "metric", "mean", "std",
                         "n_cells", "n_classes", "per_fold"])
        for arm, reps in ARMS.items():
            # The extractors read _build_nt as a module global and fan out with fork, so
            # rebinding it here reaches the workers without touching benchmark code.
            F._build_nt = BUILDERS[arm]
            _log(fh, f"########## arm {arm}: _build_nt -> {F._build_nt.__name__} ##########")
            for rep in reps:
                t0 = time.time()
                fr = EXTRACTORS[rep](cohort, MODALITY)
                reasons = Counter(r for _sid, r in fr.skipped)
                _log(fh, f"--- {arm} x {rep} --- {len(fr.specimen_ids)} cells, "
                         f"{fr.X.shape[1]} features, {len(fr.skipped)} skipped, "
                         f"extract {time.time() - t0:.1f}s")
                for r, n in reasons.most_common():
                    _log(fh, f"    skip[{n}]: {r}")
                if arm == "somacomp" and reasons.get(SMALL_COMPONENT, 0) == 0:
                    raise SystemExit(f"{SMALL_COMPONENT!r} never fired -- the _build_nt "
                                     f"rebinding did not reach the workers")

                y = np.array([id_to_label[s] for s in fr.specimen_ids], dtype=np.int64)
                present = np.unique(y)
                class_names = [cohort.label_names[i] for i in present]
                y = np.array([np.where(present == v)[0][0] for v in y], dtype=np.int64)

                t0 = time.time()
                cv = run_cv(fr.X, y, class_names=class_names,
                            outer_k=OUTER_K, inner_k=INNER_K, seed=SEED)
                _log(fh, f"  cv in {time.time() - t0:.1f}s")
                for metric, vals in cv.metrics.items():
                    _log(fh, f"    {metric:18s} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
                summary[(arm, rep)] = (float(np.mean(cv.metrics["balanced_accuracy"])),
                                       float(np.std(cv.metrics["balanced_accuracy"])),
                                       int(fr.X.shape[1]), int(cv.n_samples))

                tag = f"{rep}_{arm}"
                _write_rows(writer, csv_fh, tag, MODALITY, cv)
                plot_confusion(cv.confusion, class_names,
                               os.path.join(out_dir, f"confusion_{tag}.png"),
                               title=f"{rep} / {MODALITY} / {arm} (n={cv.n_samples})")
                try:
                    plot_embedding(fr.X, y, class_names,
                                   os.path.join(out_dir, f"embedding_{tag}.png"),
                                   method="umap", title=f"{rep} / {MODALITY} / {arm}")
                except Exception as e:
                    _log(fh, f"  embedding failed: {type(e).__name__}: {e}")
                _write_desc(out_dir, cohort, inv, summary)
                _mirror_ablation(out_dir, RESULTS_DIR)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({
            "modality": MODALITY, "arms": ARMS, "outer_k": OUTER_K, "inner_k": INNER_K,
            "seed": SEED, "min_class_size": MIN_CLASS_SIZE,
            "met_path": MET_PATH, "swc_path": SWC_PATH,
            "dendrite_edge_inventory": inv,
            "classes": cohort.label_names, "label": "MET.meta/subclass",
            "coordinates": "swc_upright",
            "compare_to": "/root/capsule/scratch/benchmarks/v2/benchmark_metrics.csv",
        }, f, indent=2, sort_keys=True)
    _log(fh, f"Done in {time.time() - t_start:.1f}s. Durable copy: {out_dir}")
    fh.close()
    _mirror_ablation(out_dir, RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rerun morphopy representations without fabricated soma-to-fragment edges.")
    args = parser.parse_args()

    MET_PATH = "/root/capsule/data/patchseq_MET/patchseq_MET.hdf5"
    SWC_PATH = "/root/capsule/data/patchseq_SWC/patchseq_SWC.hdf5"
    RESULTS_DIR = "/root/capsule/results"
    ARCHIVE_DIR = "/root/capsule/scratch/synthetic_edge_ablation"

    MIN_CLASS_SIZE = 25
    SEED = 0
    OUTER_K = 5
    INNER_K = 3
    MODALITY = "dendrite"

    PREFIX = "synthetic_ablation"
    GENERIC_NAMES = {"desc.txt", "run_log.txt", "run_config.json"}

    main(args)
