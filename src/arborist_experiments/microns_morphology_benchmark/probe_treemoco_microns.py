"""Score a joint-corpus TreeMoCo artifact against one MICrONS cohort.

One run per (source, mode), per the platform convention -- never a loop over cohorts. The encoder was
pretrained on both cohorts at once (label-free, so nothing leaks), but every scored number here lives
inside a single cohort's label space: 8 excitatory classes or 11 inhibitory ones, never a joint
classifier, which would be reading has-apical vs has-axon rather than morphology.

  frozen    The root embedding z through the benchmark's own nested-CV elastic net
            (`classifier.run_cv`, GPU FISTA solver). This is the row that lines up with
            radial_arbors, morphometric_statistics and the rest in benchmark_metrics.csv.
  finetune  Per outer fold: reload the pretrained encoder, hold out 15% of the fold's training cells
            for early stopping on balanced accuracy, fine-tune encoder + linear head, predict the
            fold's test cells. No class_weight, matching the benchmark's deliberate choice.

`finetune` re-collapses the cohort's cells rather than reusing the pretraining corpus, because it
needs trees indexed by cohort row. Collapse is ~1 min across 16 cores, so caching it would buy
nothing and add a staleness failure mode.

The outer split replicates `classifier.run_cv`'s StratifiedKFold(shuffle=True, random_state=SEED),
so frozen and finetune score identical partitions of identical cells and their difference is
attributable to the head alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from arborist_experiments.morphology_cv import persistence
from arborist_experiments.morphology_cv.classifier import CVResult, run_cv
from arborist_experiments.cells.plots import plot_confusion
from arborist_experiments.treemoco.augment import (
    PRESETS, AugmentConfig, augment, identity_view,
)
from arborist_experiments.treemoco.batching import collate
from arborist_experiments.treemoco.model import (
    TreeClassifier, TreeEncoder,
)
from arborist_experiments.treemoco.tree_features import (
    CollapsedTree, collapse,
)

from .data_loading import build_cohort_from_source, class_counts, subsample_per_class
from .features import _run_jobs
from .inh_constants import apply_inh_constants
from arborist_experiments.experiment_harness import (
    log as _log,
    mirror_dir,
    next_version_dir as _next_version_dir,
    write_rows,
)


REP_NAME = "treemoco_joint"

# The corpus keys are source-prefixed because 416 nucleus ids appear in both caches; see
# pretrain_treemoco_microns. This maps a benchmark source name onto its prefix.
KEY_PREFIX = {"em_exc": "exc", "microns_inh": "inh"}


def _mirror(src_dir: str, dst_dir: str) -> None:
    mirror_dir(src_dir, dst_dir, prefix="treemoco_probe_", generic_names=GENERIC_NAMES)


def latest_artifact(base: str) -> str:
    vs = sorted((int(d[1:]) for d in os.listdir(base)
                 if d.startswith("v") and d[1:].isdigit()), reverse=True)
    for v in vs:
        path = os.path.join(base, f"v{v}", "embeddings.h5")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no embeddings.h5 under any version dir in {base}")


def load_artifact(path: str) -> tuple[dict[str, int], np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        keys = [s.decode() if isinstance(s, bytes) else str(s) for s in f["pooled/specimen_ids"][:]]
        X = f["pooled/X"][:]
        attrs = {k: v for k, v in f.attrs.items()}
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path} has duplicate keys; the source prefix is missing or broken")
    return {k: i for i, k in enumerate(keys)}, X, attrs


def align_to_cohort(cohort, row_by_key: dict[str, int], X: np.ndarray,
                    source: str) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings and labels in cohort row order. Raises rather than silently dropping cells."""
    prefix = KEY_PREFIX[source]
    missing = [c.specimen_id for c in cohort.cells if f"{prefix}:{c.specimen_id}" not in row_by_key]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(cohort.cells)} cohort cells have no embedding "
            f"(e.g. {missing[:5]}); the artifact was built from a different cache"
        )
    rows = [row_by_key[f"{prefix}:{c.specimen_id}"] for c in cohort.cells]
    return X[rows].astype(np.float64), cohort.labels.astype(np.int64)


def _job_collapse(cell):
    try:
        return cell.specimen_id, collapse(cell), None
    except Exception as e:
        return cell.specimen_id, None, f"collapse: {type(e).__name__}: {e}"


def collapse_cohort(cohort, log=print) -> tuple[list, np.ndarray]:
    """Collapsed whole-cell trees and the cohort row index of each. No modality filter."""
    t0 = time.time()
    results = _run_jobs(SimpleNamespace(cells=cohort.cells), _job_collapse)
    trees, rows, skipped = [], [], []
    for i, (sid, tree, err) in enumerate(results):
        if err is not None:
            skipped.append(f"{sid}: {err}")
            continue
        trees.append(tree)
        rows.append(i)
    sizes = np.array([len(t.parents) for t in trees])
    log(f"collapsed {len(trees)} trees ({len(skipped)} skipped), nodes median "
        f"{int(np.median(sizes))} p90 {int(np.percentile(sizes, 90))} max {int(sizes.max())} "
        f"({time.time() - t0:.1f}s)")
    return trees, np.array(rows, dtype=np.int64)


def outer_folds(y: np.ndarray, outer_k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros((len(y), 1)), y))


def fold_metrics(y_true, y_pred, y_proba, classes) -> dict:
    return {
        "log_loss": float(log_loss(y_true, y_proba, labels=classes)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=classes,
                                   zero_division=0.0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


class _LabelledDataset(Dataset):
    def __init__(self, trees, labels, aug: AugmentConfig, train: bool):
        self.trees, self.labels, self.aug, self.train = trees, labels, aug, train
        self._rng: np.random.Generator | None = None

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, i):
        if self.train:
            if self._rng is None:
                self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
            view = augment(self.trees[i], self.aug, self._rng)
        else:
            view = identity_view(self.trees[i], self.aug)
        return view, int(self.labels[i])


def _collate_labelled(items):
    return collate([v for v, _ in items]), torch.tensor([y for _, y in items], dtype=torch.long)


@torch.no_grad()
def _predict(net: nn.Module, trees, labels, aug, device, batch_size) -> np.ndarray:
    net.eval()
    loader = DataLoader(_LabelledDataset(trees, labels, aug, train=False),
                        batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=_collate_labelled)
    return np.concatenate([torch.softmax(net(b.to(device)), dim=1).cpu().numpy()
                           for b, _ in loader], axis=0)


def finetune_fold(state: dict, trees: list[CollapsedTree], y: np.ndarray, tr: np.ndarray,
                  te: np.ndarray, aug: AugmentConfig, d_in: int, n_classes: int, device,
                  log) -> tuple[np.ndarray, dict]:
    encoder = TreeEncoder(d_in=d_in, d_hidden=D_HIDDEN, d_phi=D_PHI)
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                             if k.startswith("encoder.")})
    net = TreeClassifier(encoder, n_classes).to(device)

    sub_tr, sub_va = train_test_split(tr, test_size=FT_VAL_FRAC, random_state=SEED, stratify=y[tr])
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    loader = DataLoader(
        _LabelledDataset([trees[i] for i in sub_tr], y[sub_tr], aug, train=True),
        batch_size=FT_BATCH_SIZE, shuffle=True, drop_last=len(sub_tr) > FT_BATCH_SIZE,
        num_workers=NUM_WORKERS, collate_fn=_collate_labelled,
        persistent_workers=NUM_WORKERS > 0,
    )

    best_score, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(FT_EPOCHS):
        net.train()
        tot = 0.0
        for batch, yb in loader:
            loss = nn.functional.cross_entropy(net(batch.to(device)), yb.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            tot += loss.item()

        proba = _predict(net, [trees[i] for i in sub_va], y[sub_va], aug, device, FT_BATCH_SIZE)
        score = balanced_accuracy_score(y[sub_va], proba.argmax(axis=1))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if epoch % 5 == 0 or stale >= FT_PATIENCE:
            log(f"    ft epoch {epoch + 1}/{FT_EPOCHS} loss {tot / max(len(loader), 1):.4f} "
                f"val_bal_acc {score:.4f} (best {best_score:.4f} @ {best_epoch + 1})")
        if stale >= FT_PATIENCE:
            break

    net.load_state_dict(best_state)
    proba = _predict(net, [trees[i] for i in te], y[te], aug, device, FT_BATCH_SIZE)
    return proba, {k: v.cpu() for k, v in best_state.items()}


def run_finetune(trees, y, class_names, state, aug, d_in, ckpt_prefix,
                 log) -> tuple[CVResult, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = np.arange(len(class_names))
    metrics = {k: [] for k in ("log_loss", "accuracy", "balanced_accuracy", "f1_macro", "mcc")}
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    oof_proba = np.full((len(y), len(classes)), np.nan)
    oof_fold = np.full(len(y), -1, dtype=np.int64)
    best_params = []

    for k, (tr, te) in enumerate(outer_folds(y, OUTER_K, SEED)):
        t0 = time.time()
        log(f"  fold {k + 1}/{OUTER_K}: {len(tr)} train / {len(te)} test")
        proba, weights = finetune_fold(state, trees, y, tr, te, aug, d_in, len(class_names),
                                       device, log)
        y_pred = proba.argmax(axis=1)
        oof_proba[te] = proba
        oof_fold[te] = k
        best_params.append({"mode": "finetune", "epochs": FT_EPOCHS, "lr": FT_LR})
        torch.save(weights, f"{ckpt_prefix}_fold{k + 1}.pth")
        for name, value in fold_metrics(y[te], y_pred, proba, classes).items():
            metrics[name].append(value)
        conf += confusion_matrix(y[te], y_pred, labels=classes)
        log(f"  fold {k + 1}: acc {metrics['accuracy'][-1]:.4f} "
            f"bal_acc {metrics['balanced_accuracy'][-1]:.4f} ({time.time() - t0:.1f}s)")

    cv = CVResult(metrics=metrics, confusion=conf, best_params=best_params, n_samples=len(y),
                  n_features=D_HIDDEN, n_classes=len(class_names), class_names=class_names)
    return cv, {"proba": oof_proba, "y": y, "fold": oof_fold}


def _write_desc(out_dir: str, cfg: dict, stats: dict) -> None:
    lines = [f"TreeMoCo probe: {cfg['mode']} mode on the {cfg['source']} cohort, from a joint "
             "excitatory+inhibitory pretraining artifact.", ""]
    lines += ["* Parameters"]
    for k, v in cfg.items():
        lines.append(f"    {k} = {v!r}")
    lines += ["", "* Run statistics"]
    lines += [f"    {k} = {v}" for k, v in stats.items()]
    lines += ["", "* Reading the results",
              "    - The encoder saw both cohorts during pretraining, but pretraining is",
              "      label-free and this probe is scored inside one cohort's label space, so the",
              "      has-apical / has-axon cohort marker has nothing to leak into.",
              "    - Whole-cell input, no modality filter: excitatory cells contribute",
              "      soma+basal+apical, inhibitory ones soma+basal+axon. The hand-designed rows in",
              "      benchmark_metrics.csv are per-modality, so compare this against the",
              "      excitatory 'dendrite' rows and the inhibitory 'full' rows.",
              "    - Chance balanced accuracy is 1/n_classes; the chance_corrected column is",
              "      (BA - chance)/(1 - chance), the only form comparable across cohorts.",
              "    - `frozen` shares the classifier, grid, folds and seed with run_benchmark, so",
              "      its row is directly comparable to the hand-designed representations.",
              "      `finetune` replaces that head, so it is comparable to `frozen` and to the",
              "      Patch-seq finetune row, not to the elastic-net rows."]
    with open(os.path.join(out_dir, "desc.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(args):
    out_dir = (os.path.join(ARCHIVE_DIR, "smoke") if args.smoke
               else _next_version_dir(ARCHIVE_DIR))
    os.makedirs(out_dir, exist_ok=True)
    fh = open(os.path.join(out_dir, "run_log.txt"), "w")
    _log(fh, f"Durable output dir: {out_dir} (mirrored to {RESULTS_DIR})")

    if args.source == "microns_inh":
        apply_inh_constants(log=lambda m: _log(fh, m))

    artifact = args.artifact or latest_artifact(PRETRAIN_DIR)
    row_by_key, X_all, attrs = load_artifact(artifact)
    model_config = json.loads(attrs["model_config"])
    aug = PRESETS[attrs["augment"]](include_type=model_config["d_in"] != 30)
    _log(fh, f"Artifact: {artifact}  ({len(row_by_key)} trees, d_in={model_config['d_in']}, "
             f"augment={attrs['augment']}, epochs={attrs['epochs']})")

    # Passed explicitly rather than inherited: coarse_label is what every hand-designed row in
    # benchmark_metrics.csv was scored on, so it is the thing that makes this row comparable, and
    # coarse/fine are separate annotation rounds rather than nested (R-6) -- a fine_label run would
    # be a different task, not a refinement.
    build_kwargs = {"label_column": LABEL_COLUMN}
    build_kwargs |= ({"exc_only": True} if args.source == "em_exc" else {"inh_only": True})
    cohort = build_cohort_from_source(args.source, log=lambda m: _log(fh, m), **build_kwargs)
    if args.limit:
        # Per-class, not the first N rows: a head slice can leave a class with fewer members than
        # OUTER_K and StratifiedKFold then raises, which turns a smoke run into a spurious failure.
        cohort = subsample_per_class(cohort, max(1, args.limit // len(cohort.label_names)),
                                     seed=SEED)
        _log(fh, f"--limit {args.limit}: subsampled per class to {len(cohort)} cells")
    y = cohort.labels.astype(np.int64)
    class_names = cohort.label_names
    chance = 1.0 / len(class_names)
    counts = class_counts(cohort)
    _log(fh, f"Cohort: {len(y)} cells, {len(class_names)} classes, chance {chance:.4f} {counts}")

    run_cfg = {"source": args.source, "mode": args.mode, "artifact": artifact,
               "label_column": LABEL_COLUMN,
               "state": args.state, "solver": args.solver, "limit": args.limit,
               "seed": SEED, "outer_k": OUTER_K, "inner_k": INNER_K,
               "d_hidden": D_HIDDEN, "d_phi": D_PHI, "model_config": model_config,
               "augment": str(attrs["augment"]), "pretrain_epochs": int(attrs["epochs"]),
               "pretrain_config": json.loads(attrs["pretrain_config"]),
               "modality": "whole cell (no filter)", "n_cells": len(y),
               "class_counts": counts, "chance": chance,
               "finetune": {"epochs": FT_EPOCHS, "lr": FT_LR, "batch_size": FT_BATCH_SIZE,
                            "weight_decay": FT_WEIGHT_DECAY, "val_frac": FT_VAL_FRAC,
                            "patience": FT_PATIENCE, "grad_clip": GRAD_CLIP,
                            "num_workers": NUM_WORKERS}}
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2, sort_keys=True, default=str)

    csv_path = os.path.join(out_dir, "probe_metrics.csv")
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(["feature_rep", "mode", "metric", "mean", "std", "n_cells", "n_features",
                     "n_classes", "per_fold", "chance_corrected", "best_params",
                     "grid_boundary"])

    stats: dict = {}
    t0 = time.time()
    if args.mode == "frozen":
        X, y = align_to_cohort(cohort, row_by_key, X_all, args.source)
        _log(fh, f"frozen: X {X.shape}, running nested CV (solver={args.solver})")
        cv = run_cv(X, y, class_names, outer_k=OUTER_K, inner_k=INNER_K, seed=SEED,
                    solver=args.solver, log=lambda m: _log(fh, f"  {m}"))
        # align_to_cohort returns cohort row order, so the ids line up with the oof rows.
        oof = {"proba": cv.oof_proba, "y": y, "fold": cv.oof_fold,
               "specimen_ids": [c.specimen_id for c in cohort.cells]}
    else:
        trees, rows = collapse_cohort(cohort, log=lambda m: _log(fh, m))
        if len(rows) != len(y):
            _log(fh, f"finetune: {len(y) - len(rows)} cells dropped by collapse")
            y = y[rows]
        state = torch.load(args.state or os.path.join(os.path.dirname(artifact),
                                                      "treemoco_all.pth"),
                           map_location="cpu", weights_only=True)
        _log(fh, f"finetune: {len(trees)} trees, {FT_EPOCHS} max epochs, "
                 f"patience {FT_PATIENCE}, lr {FT_LR}")
        cv, oof = run_finetune(trees, y, class_names, state, aug, model_config["d_in"],
                               os.path.join(out_dir, f"finetuned_{args.source}"),
                               log=lambda m: _log(fh, m))
        oof["specimen_ids"] = [cohort.cells[i].specimen_id for i in rows]

    write_rows(writer, REP_NAME, args.mode, cv, chance)
    csv_fh.flush()
    ba = float(np.mean(cv.metrics["balanced_accuracy"]))
    _log(fh, f"{args.mode}: balanced_accuracy = {ba:.4f} (chance {chance:.4f}, "
             f"chance-corrected {(ba - chance) / (1 - chance):.4f})")
    _log(fh, f"  best_params = {cv.best_params}")

    persistence.write_confusion(
        os.path.join(out_dir, f"confusion_{args.source}_{args.mode}.csv"),
        cv.confusion, class_names, per_fold=getattr(cv, "confusion_per_fold", None))
    plot_confusion(cv.confusion, class_names,
                   os.path.join(out_dir, f"confusion_{args.source}_{args.mode}.png"),
                   title=f"{REP_NAME} {args.mode} / {args.source}")
    persistence.write_oof(os.path.join(out_dir, f"oof_{args.source}_{args.mode}.npz"),
                          oof["proba"], oof["y"], oof["fold"], oof["specimen_ids"], class_names)

    stats["n_cells"] = len(y)
    stats["n_classes"] = len(class_names)
    stats["balanced_accuracy"] = round(ba, 4)
    stats["chance_corrected"] = round((ba - chance) / (1 - chance), 4)
    stats["total_seconds"] = round(time.time() - t0, 1)
    csv_fh.close()

    _write_desc(out_dir, {"source": args.source, "mode": args.mode, "artifact": artifact,
                          "solver": args.solver, "outer_k": OUTER_K, "inner_k": INNER_K,
                          "seed": SEED, "d_hidden": D_HIDDEN,
                          "d_in": model_config["d_in"], "augment": attrs["augment"],
                          "pretrain_epochs": int(attrs["epochs"]),
                          "modality": "whole cell (no filter)",
                          "ft_epochs": FT_EPOCHS, "ft_lr": FT_LR,
                          "ft_batch_size": FT_BATCH_SIZE}, stats)
    _mirror(out_dir, RESULTS_DIR)
    _log(fh, f"Done in {stats['total_seconds']}s. Durable copy: {out_dir}")
    fh.close()
    _mirror(out_dir, RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score a joint TreeMoCo artifact against one MICrONS cohort.")
    parser.add_argument("--source", default="em_exc", choices=sorted(KEY_PREFIX))
    parser.add_argument("--mode", default="frozen", choices=["frozen", "finetune"])
    parser.add_argument("--artifact", default=None,
                        help="embeddings.h5; defaults to the latest pretraining version dir.")
    parser.add_argument("--state", default=None,
                        help="treemoco_all.pth; defaults to the artifact's sibling.")
    parser.add_argument("--solver", default="gpu", choices=["gpu", "saga"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR = "/root/capsule/results"
    ARCHIVE_DIR = "/root/capsule/scratch/microns_treemoco_probes"
    PRETRAIN_DIR = "/root/capsule/scratch/microns_treemoco_pretrain"
    GENERIC_NAMES = {"desc.txt", "run_log.txt", "run_config.json", "probe_metrics.csv"}

    SEED = 0
    LABEL_COLUMN = "coarse_label"
    OUTER_K = 5
    INNER_K = 3
    D_HIDDEN = 128
    D_PHI = 128
    GRAD_CLIP = 5.0
    NUM_WORKERS = 10

    FT_EPOCHS = 100
    FT_LR = 1e-4
    FT_BATCH_SIZE = 64
    FT_WEIGHT_DECAY = 1e-4
    FT_VAL_FRAC = 0.15
    FT_PATIENCE = 15

    if args.smoke:
        args.limit = args.limit or 400
        FT_EPOCHS = 3
        FT_PATIENCE = 2

    main(args)
