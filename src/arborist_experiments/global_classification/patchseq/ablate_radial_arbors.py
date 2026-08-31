"""Block ablation of the radial_arbors representation.

`radial_arbors` concatenates two feature families per arbor group -- a cylindrical
(depth, in-plane radius) cable histogram and a 3D Sholl profile -- so its benchmark row does
not say which family carries the signal, and its margin over the standalone `sholl` row is
confounded by containing a Sholl block itself.

This splits the vector by column prefix and runs each block through the benchmark's own
classifier at identical settings (5x3 nested CV, seed 0), so the rows are directly comparable
to benchmark_metrics.csv. The combined block is included as well: it reproduces the
`radial_arbors` row, which makes this artifact self-contained and checks that the v2 numbers
replicate.

Features are extracted once per modality and sliced, so all three blocks see exactly the same
cells and the same numbers -- no re-extraction differences between them.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from arborist_experiments.global_classification.cv import classifier
from arborist_experiments.global_classification.cv.classifier import run_cv
from arborist_experiments.global_classification.patchseq.data_loading import build_cohort
from arborist_experiments.utils.cells.plots import plot_confusion, plot_embedding
from arborist_experiments.global_classification.extract import extract_radial_arbors
from arborist_experiments.global_classification.patchseq.binning import PATCHSEQ
from arborist_experiments.global_classification.rows import write_rows
from arborist_experiments.utils.runs import (
    log as _log,
    mirror_dir,
    next_version_dir as _next_version_dir,
)


BLOCKS = {
    "radial_hist":  ("hist_",),
    "radial_sholl": ("sholl_",),
    "radial_both":  ("hist_", "sholl_"),
}


def _mirror_ablation(src_dir: str, dst_dir: str) -> None:
    mirror_dir(src_dir, dst_dir, prefix=f"{PREFIX}_", generic_names=GENERIC_NAMES)


def _write_desc(out_dir: str, cohort, n_cells_by_mod: dict, summary: dict) -> None:
    lines = ["Block ablation of radial_arbors: cable histogram vs 3D Sholl profile.", ""]
    lines += ["* Parameters",
              f"    blocks = {sorted(BLOCKS)!r}",
              f"    modalities = {MODALITIES!r}",
              f"    outer_k = {OUTER_K}", f"    inner_k = {INNER_K}",
              f"    seed = {SEED}", f"    min_class_size = {MIN_CLASS_SIZE}",
              f"    n_classes = {len(cohort.label_names)}",
              f"    label = MET.meta/subclass", f"    coordinates = swc_upright",
              f"    met_path = {MET_PATH}", f"    swc_path = {SWC_PATH}", ""]
    lines += ["* Blocks (columns of the radial_arbors vector, sliced by name prefix)",
              "    radial_hist  -- 12 depth x 12 in-plane-radius cable histogram per group",
              "    radial_sholl -- 20 soma-centered 3D Sholl radii per group",
              "    radial_both  -- both; reproduces the `radial_arbors` benchmark row", ""]
    lines += ["* Classifier (identical to run_benchmark.py)",
              "    StandardScaler -> PCA(min(n/2, 200, n_features)) -> elastic-net LogisticRegression",
              "    grid: C in [0.1, 1.0, 10.0], l1_ratio in [0.0, 0.5, 1.0]; scoring neg_log_loss",
              "    metrics: log_loss, accuracy, balanced_accuracy, f1_macro, mcc",
              "    no class_weight -- accuracy is raw; balanced_accuracy is the class-equal read", ""]
    lines += ["* Cells per modality"]
    lines += [f"    {m} = {n}" for m, n in n_cells_by_mod.items()]
    lines += ["", "* Results (balanced_accuracy, mean +/- std over outer folds)"]
    for (block, mod), (m, s, nf) in summary.items():
        lines.append(f"    {block:13s} {mod:9s} {nf:4d} feat   {m:.4f} +/- {s:.4f}")
    lines += ["", "* Full metrics", "    see radial_ablation_metrics.csv"]
    with open(os.path.join(out_dir, "desc.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(args):
    out_dir = _next_version_dir(ARCHIVE_DIR)
    fh = open(os.path.join(out_dir, "run_log.txt"), "w")
    _log(fh, f"Durable output dir: {out_dir} (mirrored to {RESULTS_DIR})")
    _log(fh, f"Cores: {os.cpu_count()}  cv jobs: {classifier.N_JOBS}")

    cohort = build_cohort(met_path=MET_PATH, swc_path=SWC_PATH, min_class_size=MIN_CLASS_SIZE)
    id_to_label = {c.specimen_id: int(l) for c, l in zip(cohort.cells, cohort.labels)}
    _log(fh, f"Cohort: n_cells={len(cohort)} n_classes={len(cohort.label_names)}")

    csv_path = os.path.join(out_dir, "radial_ablation_metrics.csv")
    summary, n_cells_by_mod = {}, {}
    t_start = time.time()
    with open(csv_path, "w", newline="") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["feature_rep", "modality", "metric", "mean", "std",
                         "n_cells", "n_classes", "per_fold"])
        for mod in MODALITIES:
            fr = extract_radial_arbors(cohort, mod, PATCHSEQ)
            n_cells_by_mod[mod] = len(fr.specimen_ids)
            _log(fh, f"=== {mod}: {len(fr.specimen_ids)} cells, {fr.X.shape[1]} features, "
                     f"{len(fr.skipped)} skipped ===")
            for sid, reason in fr.skipped:
                _log(fh, f"    skip {sid}: {reason}")

            y = np.array([id_to_label[s] for s in fr.specimen_ids], dtype=np.int64)
            classes_in_y = np.unique(y)
            class_names = [cohort.label_names[i] for i in classes_in_y]
            y = np.array([np.where(classes_in_y == v)[0][0] for v in y], dtype=np.int64)

            for block, prefixes in sorted(BLOCKS.items()):
                cols = np.flatnonzero([n.startswith(prefixes) for n in fr.feature_names])
                X = fr.X[:, cols]
                _log(fh, f"--- {block} x {mod} --- {X.shape[1]} features")
                t0 = time.time()
                cv = run_cv(X, y, class_names=class_names,
                            outer_k=OUTER_K, inner_k=INNER_K, seed=SEED)
                _log(fh, f"  cv: outer={OUTER_K} inner={INNER_K} in {time.time() - t0:.1f}s")
                for metric, vals in cv.metrics.items():
                    _log(fh, f"    {metric:18s} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
                summary[(block, mod)] = (float(np.mean(cv.metrics["balanced_accuracy"])),
                                         float(np.std(cv.metrics["balanced_accuracy"])),
                                         int(X.shape[1]))

                write_rows(writer, block, mod, cv, csv_fh=csv_fh)
                stem = f"{block}_{mod}"
                plot_confusion(cv.confusion, class_names,
                               os.path.join(out_dir, f"confusion_{stem}.png"),
                               title=f"{block} / {mod} (n={cv.n_samples})")
                try:
                    plot_embedding(X, y, class_names,
                                   os.path.join(out_dir, f"embedding_{stem}.png"),
                                   method="umap", title=f"{block} / {mod}")
                except Exception as e:
                    _log(fh, f"  embedding failed: {type(e).__name__}: {e}")
                _write_desc(out_dir, cohort, n_cells_by_mod, summary)
                _mirror_ablation(out_dir, RESULTS_DIR)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({
            "blocks": {k: list(v) for k, v in BLOCKS.items()},
            "modalities": MODALITIES, "outer_k": OUTER_K, "inner_k": INNER_K,
            "seed": SEED, "min_class_size": MIN_CLASS_SIZE,
            "met_path": MET_PATH, "swc_path": SWC_PATH,
            "n_cells_by_modality": n_cells_by_mod,
            "classes": cohort.label_names, "label": "MET.meta/subclass",
            "coordinates": "swc_upright",
            "compare_to": "/root/capsule/scratch/benchmarks/v2/benchmark_metrics.csv",
        }, f, indent=2, sort_keys=True)
    _log(fh, f"Done in {time.time() - t_start:.1f}s. Durable copy: {out_dir}")
    fh.close()
    _mirror_ablation(out_dir, RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablate the radial_arbors feature blocks.")
    args = parser.parse_args()

    MET_PATH = "/root/capsule/data/patchseq_MET/patchseq_MET.hdf5"
    SWC_PATH = "/root/capsule/data/patchseq_SWC/patchseq_SWC.hdf5"
    RESULTS_DIR = "/root/capsule/results"
    ARCHIVE_DIR = "/root/capsule/scratch/radial_ablation"

    MIN_CLASS_SIZE = 25
    SEED = 0
    OUTER_K = 5
    INNER_K = 3

    PREFIX = "radial_ablation"
    GENERIC_NAMES = {"desc.txt", "run_log.txt", "run_config.json"}
    MODALITIES = ["full", "axon", "dendrite"]

    main(args)
