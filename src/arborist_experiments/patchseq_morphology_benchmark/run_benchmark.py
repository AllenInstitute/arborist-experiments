"""Patch-seq morphology benchmark driver.

Reproduces the Laturnus, Kobak & Berens 2020 feature-representation benchmarks on
the Allen Patch-seq cohort. Predicts `MET.meta/subclass` (14-way, floor 25) from
morphology features derived from `swc_upright` (pia-normalized) reconstructions,
using multinomial elastic-net logistic regression with nested stratified CV.

Outputs are written to a fresh version directory under /root/capsule/scratch/ and
mirrored into /root/capsule/results/ after every rep x modality. /root/capsule/results
is wiped between sessions, so it can never be the only copy -- a previous 28-hour run
was lost that way.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np

from arborist_experiments.morphology_cv import classifier, persistence
from arborist_experiments.patchseq_morphology_benchmark import features
from arborist_experiments.patchseq_morphology_benchmark.data_loading import (
    Cohort, build_cohort, load_precomputed_morphometrics,
)
from arborist_experiments.patchseq_morphology_benchmark.features import (
    extract_density_xy, extract_density_xz, extract_density_yz,
    extract_persistence_image, extract_sholl, extract_morphometrics,
    extract_precomputed_morphometrics,
)
from arborist_experiments.morphology_cv.classifier import run_cv
from arborist_experiments.patchseq_morphology_benchmark.gaussian_baseline import run_gaussian_cv
from arborist_experiments.patchseq_morphology_benchmark.radial_arbor_features import extract_radial_arbors
from arborist_experiments.cells.plots import plot_confusion, plot_embedding
from arborist_experiments.experiment_harness import (
    log as _log,
    mirror_dir,
    next_version_dir as _next_version_dir,
    write_rows,
)


EXTRACTORS = {
    "density_xy":               ("morph", extract_density_xy),
    "density_xz":               ("morph", extract_density_xz),
    "density_yz":               ("morph", extract_density_yz),
    "persistence_image":        ("morph", extract_persistence_image),
    "sholl":                    ("morph", extract_sholl),
    "morphometric_statistics":  ("morph", extract_morphometrics),
    "precomputed_morphometrics":("baseline", extract_precomputed_morphometrics),
    "gaussian_arbor":           ("generative", run_gaussian_cv),
    "radial_arbors":            ("morph", extract_radial_arbors),
}


def _sanity_prints(cohort, fh) -> None:
    _log(fh, f"Cohort: n_cells={len(cohort)} n_classes={len(cohort.label_names)}")
    counts = defaultdict(int)
    for lbl in cohort.labels:
        counts[cohort.label_names[lbl]] += 1
    for name in cohort.label_names:
        _log(fh, f"  class {name}: {counts[name]}")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(cohort), size=min(5, len(cohort)), replace=False)
    for i in idx:
        y_range = float(np.ptp(cohort.cells[i].xyz[:, 1]))
        _log(fh, f"  cell {cohort.cells[i].specimen_id}: swc_upright y-ptp = {y_range:.1f} μm")


def _subsample_stratified(cohort, per_class: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    keep = []
    for cls in range(len(cohort.label_names)):
        idx = np.where(cohort.labels == cls)[0]
        rng.shuffle(idx)
        keep.extend(idx[:per_class].tolist())
    keep = sorted(keep)
    from arborist_experiments.patchseq_morphology_benchmark.data_loading import Cohort
    return Cohort(
        cells=[cohort.cells[i] for i in keep],
        labels=cohort.labels[keep],
        label_names=cohort.label_names,
        subclass_by_id=cohort.subclass_by_id,
    )


def _align_features_and_labels(fr, cohort):
    id_to_label = {c.specimen_id: int(l) for c, l in zip(cohort.cells, cohort.labels)}
    y = np.array([id_to_label[sid] for sid in fr.specimen_ids], dtype=np.int64)
    return fr.X, y


def _mirror(src_dir: str, dst_dir: str) -> None:
    mirror_dir(src_dir, dst_dir, prefix="benchmark_", generic_names=GENERIC_NAMES)


def _write_run_config(out_dir: str, extra: dict) -> None:
    cfg = {
        "met_path": MET_PATH, "swc_path": SWC_PATH,
        "min_class_size": MIN_CLASS_SIZE, "seed": SEED,
        "coordinates": "swc_upright",
        "label": "MET.meta/subclass",
    }
    cfg.update(extra)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def _write_desc(out_dir: str, cohort, rep_list, mod_list, outer_k, inner_k, smoke,
                solver: str, dtype: str) -> None:
    lines = ["Patch-seq morphology feature-representation benchmark.", ""]
    lines += ["* Run parameters",
              f"    smoke = {bool(smoke)}",
              f"    representations = {rep_list!r}",
              f"    modalities = {mod_list!r}",
              f"    outer_k = {outer_k}", f"    inner_k = {inner_k}",
              f"    seed = {SEED}", f"    min_class_size = {MIN_CLASS_SIZE}",
              f"    solver = {solver}   dtype = {dtype}",
              f"    n_cells = {len(cohort)}", f"    n_classes = {len(cohort.label_names)}",
              f"    label = MET.meta/subclass", f"    coordinates = swc_upright",
              f"    met_path = {MET_PATH}", f"    swc_path = {SWC_PATH}", ""]
    counts = defaultdict(int)
    for lbl in cohort.labels:
        counts[cohort.label_names[lbl]] += 1
    solver_line = ("    multinomial elastic-net logistic regression (saga, max_iter=5000)"
                   if solver == "saga" else
                   f"    multinomial elastic-net logistic regression (GPU FISTA, {dtype});"
                   " same objective as saga, see morphology_cv.gpu_logreg")
    lines += ["* Classifier",
              solver_line,
              "    StandardScaler -> PCA(min(n/2, 200, n_features)) -> LogisticRegression",
              "    grid: C in [0.1, 1.0, 10.0], l1_ratio in [0.0, 0.5, 1.0]",
              "    inner-loop scoring: neg_log_loss",
              "    metrics: log_loss, accuracy, balanced_accuracy, f1_macro, mcc",
              "    no class_weight -- accuracy is raw; balanced_accuracy and f1_macro are the",
              "    class-sensitive reads on a 12.5x imbalanced cohort", ""]
    lines += ["* Class counts"]
    lines += [f"    {name} = {counts[name]}" for name in cohort.label_names]
    lines += ["", "* Results",
              "    benchmark_metrics.csv    one row per rep x modality x metric; best_params and",
              "                             grid_boundary flag a hyperparameter on a grid edge",
              "    oof_<rep>_<mod>.npz      out-of-fold probabilities, one row per cell, with",
              "                             (y, fold, specimen_ids, class_names). Every number in",
              "                             the metrics CSV is recomputable from this, so a",
              "                             re-slice or a new metric costs no refit.",
              "    confusion_<rep>_<mod>.csv  fold=all is the aggregate the PNG shows; 0..k-1 are",
              "                             the individual outer folds"]
    with open(os.path.join(out_dir, "desc.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_manifest(cohort, out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["specimen_id", "subclass"])
        for c, lbl in zip(cohort.cells, cohort.labels):
            w.writerow([c.specimen_id, cohort.label_names[int(lbl)]])


def main(args):
    if args.smoke:
        out_dir = os.path.join(ARCHIVE_DIR, "smoke")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = _next_version_dir(ARCHIVE_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fh = open(os.path.join(out_dir, "run_log.txt"), "w")
    _log(fh, f"Durable output dir: {out_dir} (mirrored to {RESULTS_DIR})")
    _log(fh, f"Cores: {os.cpu_count()}  extract workers: {features.N_WORKERS}  cv jobs: {classifier.N_JOBS}")

    _log(fh, "Building cohort...")
    cohort = build_cohort(
        met_path=MET_PATH,
        swc_path=SWC_PATH,
        min_class_size=MIN_CLASS_SIZE,
    )
    _sanity_prints(cohort, fh)

    subset_tag = ""
    if args.subset == "precomputed":
        # Restrict to the cells precomputed_morphometrics is scored on, so other reps can be
        # compared against it without the cohort itself being a confound. The class floor stays
        # applied to the whole cohort first, so classes below 25 can survive in the subset.
        have = set(load_precomputed_morphometrics(MET_PATH)[0])
        keep = [i for i, c in enumerate(cohort.cells) if c.specimen_id in have]
        cohort = Cohort(cells=[cohort.cells[i] for i in keep], labels=cohort.labels[keep],
                        label_names=cohort.label_names, subclass_by_id=cohort.subclass_by_id)
        subset_tag = f"_sub{len(keep)}"
        _log(fh, f"Restricted to precomputed_morphometrics cells: n_cells={len(keep)} "
                 f"n_classes={len(np.unique(cohort.labels))}")

    _write_manifest(cohort, os.path.join(out_dir, "cohort_manifest.csv"))

    if args.smoke:
        cohort = _subsample_stratified(cohort, per_class=SMOKE_PER_CLASS, seed=SEED)
        _log(fh, f"Smoke subsampled cohort: n_cells={len(cohort)}")
        rep_list = SMOKE_REPS
        mod_list = SMOKE_MODALITIES
        outer_k, inner_k = SMOKE_OUTER_K, SMOKE_INNER_K
    else:
        rep_list = args.representations or REPRESENTATIONS
        mod_list = args.modalities or MODALITIES
        outer_k, inner_k = OUTER_K, INNER_K

    _write_run_config(out_dir, {
        "smoke": bool(args.smoke), "outer_k": outer_k, "inner_k": inner_k,
        "representations": rep_list, "modalities": mod_list,
        "n_cells": len(cohort), "classes": cohort.label_names,
        "subset": args.subset,
        "solver": args.solver, "dtype": args.dtype,
    })
    _write_desc(out_dir, cohort, rep_list, mod_list, outer_k, inner_k, args.smoke,
                args.solver, args.dtype)
    _mirror(out_dir, RESULTS_DIR)

    csv_path = os.path.join(out_dir, "benchmark_metrics.csv")
    with open(csv_path, "w", newline="") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow([
            "feature_rep", "modality", "metric", "mean", "std",
            "n_cells", "n_classes", "per_fold", "best_params", "grid_boundary",
        ])

        for rep in rep_list:
            kind, extractor = EXTRACTORS[rep]
            rep_label = f"{rep}{subset_tag}"
            for mod in mod_list:
                if kind == "baseline" and mod != "full":
                    continue
                _log(fh, f"--- {rep_label} × {mod} ---")
                t0 = time.time()

                if kind == "generative":
                    cv, skipped = extractor(cohort, mod, outer_k=outer_k,
                                            inner_k=inner_k, seed=SEED)
                    _log(fh, f"  gaussian cv: {len(skipped)} skipped, "
                             f"outer={outer_k} inner={inner_k} in {time.time() - t0:.1f}s")
                    for sid, reason in skipped[:10]:
                        _log(fh, f"    skip {sid}: {reason}")
                    if len(skipped) > 10:
                        _log(fh, f"    ...and {len(skipped) - 10} more.")
                    if cv is None:
                        _log(fh, "  too few cells or degenerate labels, skipping.")
                        continue
                    _log(fh, f"  temperatures: {[p['temperature'] for p in cv.best_params]}")
                    write_rows(writer, rep_label, mod, cv, csv_fh=csv_fh)
                    # No per-fold matrix and no OOF here: run_gaussian_cv builds its CVResult by
                    # hand and only has the aggregate.
                    persistence.write_confusion(
                        os.path.join(out_dir, f"confusion_{rep_label}_{mod}.csv"),
                        cv.confusion, cv.class_names)
                    plot_confusion(
                        cv.confusion, cv.class_names,
                        os.path.join(out_dir, f"confusion_{rep_label}_{mod}.png"),
                        title=f"{rep_label} / {mod} (n={cv.n_samples})",
                    )
                    _mirror(out_dir, RESULTS_DIR)
                    continue

                if kind == "baseline":
                    fr = extractor(cohort, MET_PATH)
                else:
                    fr = extractor(cohort, mod)
                _log(fh, f"  extract: {len(fr.specimen_ids)} cells, {fr.X.shape[1]} features, "
                         f"{len(fr.skipped)} skipped in {time.time() - t0:.1f}s")
                if fr.skipped:
                    for sid, reason in fr.skipped[:10]:
                        _log(fh, f"    skip {sid}: {reason}")
                    if len(fr.skipped) > 10:
                        _log(fh, f"    ...and {len(fr.skipped) - 10} more.")

                if len(fr.specimen_ids) < 30:
                    _log(fh, "  too few cells, skipping classifier.")
                    continue

                X, y = _align_features_and_labels(fr, cohort)
                classes_in_y = np.unique(y)
                class_names = [cohort.label_names[i] for i in classes_in_y]
                y_remap = np.array([np.where(classes_in_y == v)[0][0] for v in y], dtype=np.int64)
                if len(classes_in_y) < 2:
                    _log(fh, "  degenerate label set, skipping.")
                    continue

                t0 = time.time()
                cv = run_cv(X, y_remap, class_names=class_names,
                            outer_k=outer_k, inner_k=inner_k, seed=SEED,
                            solver=args.solver, dtype=args.dtype,
                            log=lambda m: _log(fh, m))
                _log(fh, f"  cv: outer={outer_k} inner={inner_k} in {time.time() - t0:.1f}s")
                _log(fh, f"  balanced_accuracy = "
                         f"{np.mean(cv.metrics['balanced_accuracy']):.4f}")
                _log(fh, f"  best_params = {cv.best_params}")

                write_rows(writer, rep_label, mod, cv, csv_fh=csv_fh)

                stem = f"{rep_label}_{mod}"
                persistence.write_oof(os.path.join(out_dir, f"oof_{stem}.npz"),
                                      cv.oof_proba, y_remap, cv.oof_fold,
                                      fr.specimen_ids, class_names)
                persistence.write_confusion(os.path.join(out_dir, f"confusion_{stem}.csv"),
                                            cv.confusion, class_names,
                                            per_fold=cv.confusion_per_fold)
                plot_confusion(
                    cv.confusion, class_names,
                    os.path.join(out_dir, f"confusion_{stem}.png"),
                    title=f"{rep_label} / {mod} (n={cv.n_samples})",
                )
                if not args.smoke:
                    try:
                        plot_embedding(
                            X, y_remap, class_names,
                            os.path.join(out_dir, f"embedding_{stem}.png"),
                            method="umap",
                            title=f"{rep_label} / {mod}",
                        )
                    except Exception as e:
                        _log(fh, f"  embedding failed: {type(e).__name__}: {e}")
                _mirror(out_dir, RESULTS_DIR)

    _log(fh, f"Done. Durable copy: {out_dir}")
    fh.close()
    _mirror(out_dir, RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke test.")
    parser.add_argument("--representations", nargs="+", default=None,
                        help="Subset of the representations to score. Defaults to all.")
    parser.add_argument("--modalities", nargs="+", default=None,
                        help="Subset of the modalities to score. Defaults to all three.")
    parser.add_argument("--subset", default="none", choices=["none", "precomputed"],
                        help="'precomputed' restricts the cohort to the cells "
                             "precomputed_morphometrics is scored on, for a matched comparison.")
    parser.add_argument("--solver", default="gpu", choices=["gpu", "saga"],
                        help="Elastic-net optimizer. Same objective either way; gpu (FISTA) is "
                             "~4x faster and reaches a lower objective at the C=10 corner. saga "
                             "is kept for the equivalence tests.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                        help="Iterate precision for the gpu solver; reductions are always float64.")
    args = parser.parse_args()

    MET_PATH = "/root/capsule/data/patchseq_MET/patchseq_MET.hdf5"
    SWC_PATH = "/root/capsule/data/patchseq_SWC/patchseq_SWC.hdf5"
    RESULTS_DIR = "/root/capsule/results"
    ARCHIVE_DIR = "/root/capsule/scratch/benchmarks"

    MIN_CLASS_SIZE = 25
    SEED = 0

    GENERIC_NAMES = {"desc.txt", "run_log.txt", "run_config.json"}

    OUTER_K = 5
    INNER_K = 3

    REPRESENTATIONS = [
        "density_xy", "density_xz", "density_yz",
        "persistence_image", "sholl", "morphometric_statistics",
        "precomputed_morphometrics", "gaussian_arbor", "radial_arbors",
    ]
    MODALITIES = ["full", "axon", "dendrite"]

    SMOKE_PER_CLASS = 20
    SMOKE_REPS = ["morphometric_statistics"]
    SMOKE_MODALITIES = ["full"]
    SMOKE_OUTER_K = 3
    SMOKE_INNER_K = 2

    main(args)
