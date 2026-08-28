"""Score a TreeMoCo pretraining artifact against the Patch-seq subclass benchmark.

Three modes, all reporting the same five metrics over the same outer folds so their rows are directly
comparable to each other and to benchmark_metrics.csv:

  frozen    The root embedding z through `classifier.run_cv` -- the benchmark's own nested-CV
            elastic net. This is the row that lines up with radial_arbors, morphometric_statistics
            and the rest.
  finetune  Per outer fold: reload the pretrained encoder, hold out 15% of the fold's training cells
            for early stopping on balanced accuracy, fine-tune encoder + linear head, predict the
            fold's test cells. No class_weight, matching the benchmark's deliberate choice on a
            12.5x-imbalanced cohort.
  per_fold  Per outer fold: re-run pretraining on that fold's training cells only, then probe. The
            leakage-free number; the gap against `frozen` is what the transductive headline costs.
            Defaults to the `full` modality alone because it repeats pretraining 5 times.

Both learned modes reuse `classifier._make_pipeline` and the grid so the classifier is identical to
run_cv's, and replicate its outer StratifiedKFold(shuffle=True, random_state=seed) split, so every
mode scores the same five partitions of the same cells.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from arborist_experiments.morphology_cv import classifier
from arborist_experiments.morphology_cv.classifier import CVResult, run_cv
from ..data_loading import Cohort, build_cohort, load_precomputed_morphometrics
from ..plots import plot_confusion, plot_embedding
from ..probe_path_encodings import load_pooled
from ..run_benchmark import _log, _next_version_dir, _write_rows
from .augment import PRESETS, AugmentConfig, augment, identity_view
from .batching import collate
from .model import TreeClassifier, TreeEncoder
from .pretrain import PretrainConfig, embed, pretrain
from .pretrain_treemoco import collapse_cohort
from .tree_features import N_FEATURES, N_FEATURES_TYPED, CollapsedTree


REP_NAME = "treemoco"


def modality_columns(names: list[str], modality: str) -> np.ndarray:
    """Columns of one modality's z block.

    Not `probe_path_encodings.modality_columns`: that treats "full" as every column because a path
    encoding only has axon and dendrite groups, whereas here "full" is its own encoder trained on the
    whole tree and must not be the union of all three.
    """
    return np.flatnonzero(np.array([n.startswith(f"z_{modality}_") for n in names]))


def _mirror_probe(src_dir: str, dst_dir: str) -> None:
    """Mirror into the flat results dir, prefixing the generic filenames so nothing is clobbered."""
    os.makedirs(dst_dir, exist_ok=True)
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        dst_name = f"{REP_NAME}_probe_{name}" if name in GENERIC_NAMES else name
        shutil.copy2(src, os.path.join(dst_dir, dst_name))


def outer_folds(y: np.ndarray, outer_k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The same partitions `classifier.run_cv` uses, so every mode scores identical folds."""
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


def _empty_metrics() -> dict:
    return {"log_loss": [], "accuracy": [], "balanced_accuracy": [], "f1_macro": [], "mcc": []}


# --- mode: finetune ---
class _LabelledDataset(Dataset):
    def __init__(self, trees: list[CollapsedTree], labels: np.ndarray, aug: AugmentConfig,
                 train: bool):
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
                  te: np.ndarray, aug: AugmentConfig, cfg: PretrainConfig, n_classes: int,
                  log) -> tuple[np.ndarray, dict]:
    """Fine-tune the pretrained encoder on fold `tr`.

    Returns the test-set class probabilities and the selected weights. The weights are returned
    rather than dropped so a fold's model can be re-applied to any cell later -- re-scoring on a
    cohort subset otherwise costs a full retrain.
    """
    device = torch.device(cfg.device)
    d_in = N_FEATURES_TYPED if aug.include_type else N_FEATURES
    encoder = TreeEncoder(d_in=d_in, d_hidden=cfg.d_hidden, d_phi=cfg.d_phi)
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                             if k.startswith("encoder.")})
    net = TreeClassifier(encoder, n_classes).to(device)

    sub_tr, sub_va = train_test_split(tr, test_size=FT_VAL_FRAC, random_state=SEED,
                                      stratify=y[tr])
    train_trees = [trees[i] for i in sub_tr]
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    loader = DataLoader(_LabelledDataset(train_trees, y[sub_tr], aug, train=True),
                        batch_size=FT_BATCH_SIZE, shuffle=True, drop_last=len(sub_tr) > FT_BATCH_SIZE,
                        num_workers=cfg.num_workers, collate_fn=_collate_labelled,
                        persistent_workers=cfg.num_workers > 0)

    best_score, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(FT_EPOCHS):
        net.train()
        tot = 0.0
        for batch, yb in loader:
            loss = nn.functional.cross_entropy(net(batch.to(device)), yb.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item()

        proba = _predict(net, [trees[i] for i in sub_va], y[sub_va], aug, device, FT_BATCH_SIZE)
        score = balanced_accuracy_score(y[sub_va], proba.argmax(axis=1))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if epoch % 10 == 0 or stale >= FT_PATIENCE:
            log(f"    ft epoch {epoch + 1}/{FT_EPOCHS} loss {tot / max(len(loader), 1):.4f} "
                f"val_bal_acc {score:.4f} (best {best_score:.4f} @ {best_epoch + 1})")
        if stale >= FT_PATIENCE:
            break

    net.load_state_dict(best_state)
    proba = _predict(net, [trees[i] for i in te], y[te], aug, device, FT_BATCH_SIZE)
    return proba, {k: v.cpu() for k, v in best_state.items()}


# --- mode: per_fold ---
def elastic_net_fold(X: np.ndarray, y: np.ndarray, tr: np.ndarray,
                     te: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """One outer fold of run_cv's classifier: inner grid search on `tr`, predict `te`."""
    pipe = classifier._make_pipeline(len(tr), X.shape[1], SEED)
    inner = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, GRID, cv=inner, scoring="neg_log_loss",
                      n_jobs=classifier.N_JOBS, refit=True)
    gs.fit(X[tr], y[tr])
    return gs.predict(X[te]), gs.predict_proba(X[te]), gs.best_params_


def run_learned_mode(mode: str, trees: list[CollapsedTree], y: np.ndarray, class_names: list[str],
                     aug: AugmentConfig, cfg: PretrainConfig, state: dict | None, log,
                     ckpt_prefix: str | None = None) -> tuple[CVResult, dict]:
    """Outer-fold loop shared by `finetune` and `per_fold`.

    Also returns the out-of-fold predictions, so the run can be re-scored on any cohort subset
    without retraining. `ckpt_prefix`, if given, is the path stem the per-fold weights are saved to.
    """
    classes = np.arange(len(class_names))
    metrics, best_params = _empty_metrics(), []
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    n_features = cfg.d_hidden
    oof_proba = np.full((len(y), len(classes)), np.nan, dtype=np.float64)
    oof_fold = np.full(len(y), -1, dtype=np.int64)

    for k, (tr, te) in enumerate(outer_folds(y, OUTER_K, SEED)):
        t0 = time.time()
        log(f"  fold {k + 1}/{OUTER_K}: {len(tr)} train / {len(te)} test")
        if mode == "finetune":
            proba, fold_weights = finetune_fold(state, trees, y, tr, te, aug, cfg,
                                                len(class_names), log)
            y_pred = proba.argmax(axis=1)
            best_params.append({"mode": "finetune", "epochs": FT_EPOCHS, "lr": FT_LR})
            if ckpt_prefix:
                torch.save(fold_weights, f"{ckpt_prefix}_fold{k + 1}.pth")
        else:
            fold_state, history = pretrain([trees[i] for i in tr], aug, cfg,
                                           log=lambda m: log(f"    fold {k + 1} {m}"))
            encoder = TreeEncoder(d_in=N_FEATURES_TYPED if aug.include_type else N_FEATURES,
                                  d_hidden=cfg.d_hidden, d_phi=cfg.d_phi)
            encoder.load_state_dict({kk[len("encoder."):]: v for kk, v in fold_state.items()
                                     if kk.startswith("encoder.")})
            X = embed(encoder, trees, aug, cfg)
            n_features = X.shape[1]
            y_pred, proba, bp = elastic_net_fold(X, y, tr, te)
            if history:
                bp["final_pretrain_loss"] = round(history[-1].loss, 4)
            best_params.append(bp)

        oof_proba[te] = proba
        oof_fold[te] = k
        for name, value in fold_metrics(y[te], y_pred, proba, classes).items():
            metrics[name].append(value)
        conf += confusion_matrix(y[te], y_pred, labels=classes)
        log(f"  fold {k + 1}: acc {metrics['accuracy'][-1]:.4f} "
            f"bal_acc {metrics['balanced_accuracy'][-1]:.4f} ({time.time() - t0:.1f}s)")

    cv = CVResult(metrics=metrics, confusion=conf, best_params=best_params,
                  n_samples=len(y), n_features=n_features, n_classes=len(class_names),
                  class_names=class_names)
    return cv, {"proba": oof_proba, "y": y, "fold": oof_fold}


def _relabel(y: np.ndarray, label_names: list[str]) -> tuple[np.ndarray, list[str]]:
    present = np.unique(y)
    remap = {v: i for i, v in enumerate(present)}
    return (np.array([remap[v] for v in y], dtype=np.int64),
            [label_names[i] for i in present])


def main(args):
    if args.smoke:
        out_dir = os.path.join(ARCHIVE_DIR, "smoke")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = _next_version_dir(ARCHIVE_DIR)
    fh = open(os.path.join(out_dir, "run_log.txt"), "w")
    _log(fh, f"Durable output dir: {out_dir} (mirrored to {RESULTS_DIR})")
    _log(fh, f"Mode: {args.mode}  artifact: {args.artifact}")

    run_dir = os.path.dirname(os.path.abspath(args.artifact))
    with open(os.path.join(run_dir, "run_config.json")) as f:
        pretrain_cfg = json.load(f)
    with open(os.path.join(run_dir, "model_config.json")) as f:
        model_config = json.load(f)
    # include_type is not part of the preset, so it has to be restored from the pretraining run or
    # the encoder would be rebuilt at the wrong input width. Artifacts predating the flag lack the key.
    aug = PRESETS[pretrain_cfg["augment"]](
        include_type=bool(pretrain_cfg["augment_config"].get("include_type", False)))
    cfg = PretrainConfig(d_hidden=model_config["d_hidden"], d_phi=model_config["d_phi"],
                         d_proj=model_config["d_proj"],
                         projector_bn=model_config["projector_bn"],
                         epochs=args.epochs or pretrain_cfg["pretrain_config"]["epochs"],
                         batch_size=pretrain_cfg["pretrain_config"]["batch_size"],
                         lr=pretrain_cfg["pretrain_config"]["lr"],
                         weight_decay=pretrain_cfg["pretrain_config"]["weight_decay"],
                         temperature=pretrain_cfg["pretrain_config"]["temperature"],
                         queue_size=pretrain_cfg["pretrain_config"]["queue_size"],
                         momentum=pretrain_cfg["pretrain_config"]["momentum"],
                         grad_clip=pretrain_cfg["pretrain_config"]["grad_clip"],
                         num_workers=NUM_WORKERS, seed=SEED)
    _log(fh, f"Pretraining preset: {aug.name}  encoder d_hidden={cfg.d_hidden}  device={cfg.device}")

    cohort = build_cohort(met_path=MET_PATH, swc_path=SWC_PATH, min_class_size=MIN_CLASS_SIZE)
    _log(fh, f"Cohort: n_cells={len(cohort.cells)} n_classes={len(cohort.label_names)}")

    subset_tag = ""
    if args.subset == "precomputed":
        # The precomputed_morphometrics benchmark row is scored on only the cells present in the MET
        # morphometric table, so comparing against it needs the same cells. The class floor stays
        # applied to the whole cohort first, as run_benchmark does, which is why classes below 25
        # survive here.
        have = set(load_precomputed_morphometrics(MET_PATH)[0])
        keep = [i for i, c in enumerate(cohort.cells) if c.specimen_id in have]
        cohort = Cohort(cells=[cohort.cells[i] for i in keep], labels=cohort.labels[keep],
                        label_names=cohort.label_names, subclass_by_id=cohort.subclass_by_id)
        subset_tag = f"_sub{len(keep)}"
        _log(fh, f"Restricted to the precomputed_morphometrics cells: n_cells={len(keep)} "
                 f"n_classes={len(np.unique(cohort.labels))}")

    modalities = args.modalities or (["full"] if args.mode == "per_fold"
                                     else ["full", "axon", "dendrite"])
    _log(fh, f"Modalities: {modalities}")

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({"mode": args.mode, "artifact": os.path.abspath(args.artifact),
                   "pretrain_run_config": pretrain_cfg, "model_config": model_config,
                   "modalities": modalities, "outer_k": OUTER_K, "inner_k": INNER_K,
                   "seed": SEED, "min_class_size": MIN_CLASS_SIZE,
                   "subset": args.subset, "n_cells_scored": len(cohort.cells),
                   "label": "MET.meta/subclass", "coordinates": "swc_upright",
                   "finetune": {"epochs": FT_EPOCHS, "lr": FT_LR, "batch_size": FT_BATCH_SIZE,
                                "weight_decay": FT_WEIGHT_DECAY, "val_frac": FT_VAL_FRAC,
                                "patience": FT_PATIENCE, "class_weight": None,
                                "train_views": "augmented", "eval_views": "identity"},
                   "per_fold_epochs": cfg.epochs}, f, indent=2, sort_keys=True)

    if args.mode == "frozen":
        X_all, ids, names, attrs = load_pooled(args.artifact)
        _log(fh, f"  pooled X {X_all.shape}, {len(ids)} cells, {len(names)} features")
        row_of = {sid: i for i, sid in enumerate(ids)}
        keep = [i for i, c in enumerate(cohort.cells) if c.specimen_id in row_of]
        missing = len(cohort.cells) - len(keep)
        X_all = X_all[[row_of[cohort.cells[i].specimen_id] for i in keep]]
        y_all, class_names = _relabel(cohort.labels[keep], cohort.label_names)
        _log(fh, f"  matched {len(keep)} cells, {missing} missing from artifact")
    else:
        y_all, class_names = None, None

    csv_path = os.path.join(out_dir, "probe_metrics.csv")
    summary: dict = {}
    with open(csv_path, "w", newline="") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["feature_rep", "modality", "metric", "mean", "std",
                         "n_cells", "n_classes", "per_fold"])
        # A 0-epoch artifact is the random-encoder control; it has to be labelled apart from a
        # pretrained one or its rows are indistinguishable in a merged metrics table.
        tag = "_rand" if pretrain_cfg["pretrain_config"]["epochs"] == 0 else ""
        tag += "_type" if aug.include_type else ""
        rep = f"{REP_NAME}_{aug.name}{tag}_{args.mode}{subset_tag}"
        for mod in modalities:
            _log(fh, f"--- {rep} x {mod} ---")
            t0 = time.time()
            if args.mode == "frozen":
                cols = modality_columns(names, mod)
                X = X_all[:, cols]
                n_zero = int((~X.any(axis=1)).sum())
                _log(fh, f"  {X.shape[1]} features, {n_zero} cells with an all-zero block")
                cv = run_cv(X, y_all, class_names=class_names, outer_k=OUTER_K,
                            inner_k=INNER_K, seed=SEED)
            else:
                trees, rows, skipped = collapse_cohort(cohort, mod)
                y, class_names = _relabel(cohort.labels[rows], cohort.label_names)
                _log(fh, f"  {len(trees)} trees ({len(skipped)} cells lack this modality), "
                         f"{len(class_names)} classes")
                state = None
                if args.mode == "finetune":
                    ckpt = os.path.join(run_dir, f"treemoco_{mod}.pth")
                    state = torch.load(ckpt, map_location="cpu", weights_only=True)
                    _log(fh, f"  checkpoint {os.path.basename(ckpt)}")
                stem_ck = f"{rep}_{mod}"
                cv, oof = run_learned_mode(
                    args.mode, trees, y, class_names, aug, cfg, state, lambda m: _log(fh, m),
                    ckpt_prefix=os.path.join(out_dir, f"finetuned_{stem_ck}")
                    if args.mode == "finetune" else None)
                np.savez_compressed(
                    os.path.join(out_dir, f"oof_{stem_ck}.npz"),
                    specimen_ids=np.array([cohort.cells[i].specimen_id for i in rows], dtype=object),
                    class_names=np.array(class_names, dtype=object), **oof)
                X = None
            _log(fh, f"  finished in {time.time() - t0:.1f}s")
            for metric, vals in cv.metrics.items():
                _log(fh, f"    {metric:9s} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
            summary[mod] = {k: (float(np.mean(v)), float(np.std(v)))
                            for k, v in cv.metrics.items()}

            _write_rows(writer, csv_fh, rep, mod, cv)
            stem = f"{rep}_{mod}"
            plot_confusion(cv.confusion, cv.class_names,
                           os.path.join(out_dir, f"confusion_{stem}.png"),
                           title=f"{rep} / {mod} (n={cv.n_samples})")
            if X is not None:
                try:
                    plot_embedding(X, y_all, cv.class_names,
                                   os.path.join(out_dir, f"embedding_{stem}.png"),
                                   method="umap", title=f"{rep} / {mod}")
                except Exception as e:
                    _log(fh, f"  embedding failed: {type(e).__name__}: {e}")
            _mirror_probe(out_dir, RESULTS_DIR)

    lines = [f"TreeMoCo probe ({args.mode}) of {os.path.abspath(args.artifact)}", ""]
    lines += ["* Probe parameters",
              f"    representation = {rep!r}",
              f"    mode = {args.mode!r}",
              f"    augment_preset = {aug.name!r}",
              f"    modalities = {modalities!r}",
              f"    outer_k = {OUTER_K}", f"    inner_k = {INNER_K}",
              f"    seed = {SEED}", f"    min_class_size = {MIN_CLASS_SIZE}",
              f"    subset = {args.subset!r}",
              f"    n_cells_scored = {len(cohort.cells)}",
              "    label = 'MET.meta/subclass'"]
    if args.mode == "finetune":
        lines += [f"    ft_epochs = {FT_EPOCHS}", f"    ft_lr = {FT_LR}",
                  f"    ft_batch_size = {FT_BATCH_SIZE}",
                  f"    ft_weight_decay = {FT_WEIGHT_DECAY}",
                  f"    ft_val_frac = {FT_VAL_FRAC}", f"    ft_patience = {FT_PATIENCE}",
                  "    class_weight = None"]
    if args.mode == "per_fold":
        lines += [f"    per_fold_pretrain_epochs = {cfg.epochs}"]
    lines += ["", "* Pretraining hyperparameters (from the artifact)"]
    lines += [f"    {k} = {v!r}" for k, v in sorted(pretrain_cfg["pretrain_config"].items())]
    lines += ["", "* Results (mean +/- std over outer folds)"]
    for mod, mets in summary.items():
        lines.append(f"    {mod}:")
        for metric, (m, s) in mets.items():
            lines.append(f"        {metric:9s} {m:.4f} +/- {s:.4f}")
    with open(os.path.join(out_dir, "desc.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    _log(fh, f"Done. Durable copy: {out_dir}")
    fh.close()
    _mirror_probe(out_dir, RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a TreeMoCo artifact with the benchmark "
                                                 "classifier.")
    parser.add_argument("--artifact",
                        default="/root/capsule/scratch/treemoco_pretrain/v1/embeddings.h5",
                        help="embeddings.h5 written by pretrain_treemoco.py.")
    parser.add_argument("--mode", default="frozen", choices=["frozen", "finetune", "per_fold"])
    parser.add_argument("--modalities", nargs="+", default=None,
                        help="Defaults to all three, or just 'full' for per_fold, which "
                             "repeats pretraining once per outer fold.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the per-fold pretraining epochs (per_fold mode).")
    parser.add_argument("--subset", default="none", choices=["none", "precomputed"],
                        help="'precomputed' restricts the cohort to the 921 cells the "
                             "precomputed_morphometrics benchmark row is scored on.")
    parser.add_argument("--smoke", action="store_true", help="Short run into a smoke/ dir.")
    args = parser.parse_args()

    MET_PATH = "/root/capsule/data/patchseq_MET/patchseq_MET.hdf5"
    SWC_PATH = "/root/capsule/data/patchseq_SWC/patchseq_SWC.hdf5"
    RESULTS_DIR = "/root/capsule/results"
    ARCHIVE_DIR = "/root/capsule/scratch/treemoco_probes"

    MIN_CLASS_SIZE = 25
    SEED = 0
    OUTER_K = 5
    INNER_K = 3
    NUM_WORKERS = 4
    GRID = {"clf__C": [0.1, 1.0, 10.0], "clf__l1_ratio": [0.0, 0.5, 1.0]}
    GENERIC_NAMES = {"desc.txt", "run_log.txt", "run_config.json", "probe_metrics.csv"}

    FT_EPOCHS = 100
    FT_LR = 1e-4
    FT_BATCH_SIZE = 64
    FT_WEIGHT_DECAY = 1e-4
    FT_VAL_FRAC = 0.15
    FT_PATIENCE = 15

    if args.smoke:
        FT_EPOCHS = 2
        FT_PATIENCE = 2
        if args.epochs is None:
            args.epochs = 2

    main(args)
