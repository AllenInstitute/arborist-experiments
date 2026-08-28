"""Driver: contrastively pretrain TreeMoCo on the Patch-seq cohort and dump root embeddings.

One Code Ocean run per augmentation preset (`--augment upright` or `--augment paper`); the three
modalities are looped inside the run, as run_benchmark.py does. Writes to
scratch/treemoco_pretrain/vX/ and mirrors into results/ after every modality, since results/ is wiped
between sessions.

The embeddings artifact deliberately uses the schema probe_path_encodings.load_pooled already reads:
pooled/X with one row per cohort cell and feature names of the form z_<modality>_<dim>, so
`modality_columns` slices out a modality by name. A cell that has no nodes of a modality gets a zero
block, matching how path_encodings pools an absent group.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

from ..data_loading import Cohort, build_cohort
from ..features import _run_jobs
from ..modality import filter_cell
from .augment import PRESETS
from .model import TreeEncoder
from .pretrain import PretrainConfig, embed, pretrain
from .tree_features import collapse, n_features


MODALITIES = ("full", "axon", "dendrite")
_STR = h5py.string_dtype()


def _log(fh, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def _next_version_dir(base: str) -> str:
    os.makedirs(base, exist_ok=True)
    used = {int(d[1:]) for d in os.listdir(base) if d.startswith("v") and d[1:].isdigit()}
    out = os.path.join(base, f"v{max(used, default=0) + 1}")
    os.makedirs(out)
    return out


def _mirror_file(src: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    name = os.path.basename(src)
    if name in GENERIC_NAMES:
        name = f"treemoco_pretrain_{name}"
    tmp = os.path.join(dst_dir, name + ".partial")
    with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
        while chunk := fsrc.read(8 << 20):
            fdst.write(chunk)
    os.replace(tmp, os.path.join(dst_dir, name))


def _job_collapse(cell, modality: str):
    sub = filter_cell(cell, modality)
    if sub is None:
        return cell.specimen_id, None, "no nodes for modality"
    try:
        return cell.specimen_id, collapse(sub), None
    except Exception as e:
        return cell.specimen_id, None, f"collapse: {type(e).__name__}: {e}"


def collapse_cohort(cohort: Cohort, modality: str) -> tuple[list, np.ndarray, list[str]]:
    """Collapsed trees for one modality, the cohort row index of each, and the skip reasons."""
    results = _run_jobs(cohort, _job_collapse, modality)
    trees, rows, skipped = [], [], []
    for i, (sid, tree, err) in enumerate(results):
        if err is not None:
            skipped.append(f"{sid}: {err}")
            continue
        trees.append(tree)
        rows.append(i)
    return trees, np.array(rows, dtype=np.int64), skipped


def feature_names(d_hidden: int) -> list[str]:
    return [f"z_{m}_{d:03d}" for m in MODALITIES for d in range(d_hidden)]


def _init_artifact(path: str, specimen_ids: list[str], d_hidden: int, attrs: dict) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("pooled/X", shape=(len(specimen_ids), len(MODALITIES) * d_hidden),
                         dtype="f4", fillvalue=0.0)
        f.create_dataset("pooled/specimen_ids", data=np.array(specimen_ids, dtype=object),
                         dtype=_STR)
        f.create_dataset("pooled/feature_names",
                         data=np.array(feature_names(d_hidden), dtype=object), dtype=_STR)
        for k, v in attrs.items():
            f.attrs[k] = v


def _write_history(path: str, history) -> None:
    with open(path, "w") as f:
        f.write("epoch,loss,top1,embed_std,lr\n")
        for s in history:
            f.write(f"{s.epoch},{s.loss:.6f},{s.top1:.6f},{s.embed_std:.6f},{s.lr:.6e}\n")


def _write_desc(out_dir: str, cfg: dict, stats: dict) -> None:
    lines = ["TreeMoCo contrastive pretraining on the Patch-seq cohort "
             "(Chen et al. 2022, NeurIPS).", ""]
    lines += ["* Parameters"]
    for k, v in cfg.items():
        if isinstance(v, dict):
            lines += [f"    {k}.{ik} = {iv!r}" for ik, iv in v.items()]
        else:
            lines.append(f"    {k} = {v!r}")
    lines += ["", "* Run statistics"]
    lines += [f"    {k} = {v}" for k, v in stats.items()]
    lines += ["", "* Departures from the paper",
              "    - 30th node feature is_synthetic flags parent links invented by",
              "      _reparent_to_soma (42% of cells are fragmented); those links keep their",
              "      topology but their branch geometry is zeroed.",
              "    - F_E is built from 9 resampled points (8 edge vectors -> R^24); the paper",
              "      says 8 vertices, which cannot give 24 dims. Branches are stored at 33",
              "      points and L/C measured there, because a 9-point resample loses a median",
              "      2.2% of arc length and >5% on 21% of branches.",
              "    - Both Tree-LSTM cells use one forget gate over the summed child state, as",
              "      the paper's Eqns 1-4 are written. Tai et al.'s per-child forget gate is",
              "      not expressible for LSTM2, whose input c_{k,1} postdates its children.",
              "    - No shuffling-BN (single GPU), so query and key batches share BN statistics.",
              "    - queue_size and momentum are scaled down from MoCo's ImageNet defaults: the",
              "      queue cannot exceed the cohort, and 0.999 barely moves the key encoder in",
              "      200 epochs of ~25 steps."]
    if cfg.get("augment_config", {}).get("include_type"):
        lines += ["    - Node features carry a soma/axon/dendrite one-hot, so d_in is 33 not 30.",
                  "      The paper's nodes have no identity; without it the encoder cannot tell an",
                  "      axonal branch from a dendritic one, a distinction every other",
                  "      representation in the benchmark gets for free from being extracted per",
                  "      modality and concatenated. It is exempt from attribute masking.",
                  "      Basal (3) and apical (4) dendrite are merged on purpose: the apical call",
                  "      follows from having judged the cell pyramidal, so keeping it separate",
                  "      would feed the subclass label back in as a node feature."]
    with open(os.path.join(out_dir, "desc.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(args):
    if args.smoke:
        out_dir = os.path.join(ARCHIVE_DIR, "smoke")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = _next_version_dir(ARCHIVE_DIR)

    fh = open(os.path.join(out_dir, "run_log.txt"), "w")
    _log(fh, f"Durable output dir: {out_dir} (mirrored to {RESULTS_DIR})")

    aug = PRESETS[args.augment](include_type=args.node_type)
    cfg = PretrainConfig(
        d_hidden=D_HIDDEN, d_phi=D_PHI, d_proj=D_PROJ, projector_bn=not args.no_projector_bn,
        epochs=args.epochs, batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        temperature=TEMPERATURE, queue_size=QUEUE_SIZE, momentum=MOMENTUM,
        grad_clip=GRAD_CLIP, num_workers=NUM_WORKERS, seed=SEED,
    )
    _log(fh, f"Augmentation preset: {aug.name}  device: {cfg.device}")

    cohort = build_cohort(met_path=MET_PATH, swc_path=SWC_PATH, min_class_size=MIN_CLASS_SIZE)
    if args.limit:
        cohort = Cohort(cells=cohort.cells[:args.limit], labels=cohort.labels[:args.limit],
                        label_names=cohort.label_names, subclass_by_id=cohort.subclass_by_id)
    specimen_ids = [c.specimen_id for c in cohort.cells]
    _log(fh, f"Cohort: {len(specimen_ids)} cells, {len(cohort.label_names)} classes")

    model_config = {"d_in": n_features(aug.include_type),
                    "d_hidden": cfg.d_hidden, "d_phi": cfg.d_phi,
                    "d_proj": cfg.d_proj, "projector_bn": cfg.projector_bn}
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2, sort_keys=True)

    run_cfg = {"augment": aug.name, "augment_config": {k: (list(v) if isinstance(v, tuple) else v)
                                                       for k, v in vars(aug).items()},
               "pretrain_config": cfg.as_dict(), "model_config": model_config,
               "modalities": list(MODALITIES), "n_cells": len(specimen_ids),
               "min_class_size": MIN_CLASS_SIZE, "coordinates": "swc_upright",
               "met_path": MET_PATH, "swc_path": SWC_PATH, "seed": SEED}
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2, sort_keys=True)

    h5_path = os.path.join(out_dir, "embeddings.h5")
    _init_artifact(h5_path, specimen_ids, cfg.d_hidden, {
        "augment": aug.name, "n_cells": len(specimen_ids), "epochs": cfg.epochs,
        "coordinates": "swc_upright", "model_config": json.dumps(model_config),
        "pretrain_config": json.dumps(cfg.as_dict()), "seed": SEED,
    })

    stats: dict = {}
    t_start = time.time()
    for mi, modality in enumerate(MODALITIES):
        t0 = time.time()
        trees, rows, skipped = collapse_cohort(cohort, modality)
        sizes = np.array([len(t) for t in trees])
        _log(fh, f"{modality}: {len(trees)} trees ({len(skipped)} skipped), collapsed nodes "
                 f"median {int(np.median(sizes))} p90 {int(np.percentile(sizes, 90))} "
                 f"max {sizes.max()}  ({time.time() - t0:.1f}s)")

        state, history = pretrain(trees, aug, cfg, log=lambda m, _m=modality: _log(fh, f"  {_m} {m}"))
        ckpt = os.path.join(out_dir, f"treemoco_{modality}.pth")
        torch.save(state, ckpt)
        hist_path = os.path.join(out_dir, f"pretrain_history_{modality}.csv")
        _write_history(hist_path, history)

        encoder = TreeEncoder(d_in=model_config["d_in"], d_hidden=cfg.d_hidden, d_phi=cfg.d_phi)
        encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                                 if k.startswith("encoder.")})
        Z = embed(encoder, trees, aug, cfg)
        lo = mi * cfg.d_hidden
        with h5py.File(h5_path, "a") as f:
            f["pooled/X"][rows, lo:lo + cfg.d_hidden] = Z.astype(np.float32)

        stats[f"n_trees_{modality}"] = len(trees)
        stats[f"n_skipped_{modality}"] = len(skipped)
        stats[f"embed_z_std_{modality}"] = round(float(Z.std(axis=0).mean()), 5)
        if history:
            stats[f"final_loss_{modality}"] = round(history[-1].loss, 4)
            stats[f"final_top1_{modality}"] = round(history[-1].top1, 4)
            stats[f"final_embed_std_{modality}"] = round(history[-1].embed_std, 5)
        _log(fh, f"{modality}: done in {time.time() - t0:.1f}s"
                 + (f"  final loss {history[-1].loss:.4f} top1 {history[-1].top1:.3f} "
                    f"embed_std {history[-1].embed_std:.4f}" if history else "  (control run)"))

        for path in (ckpt, hist_path, h5_path, os.path.join(out_dir, "run_log.txt")):
            _mirror_file(path, RESULTS_DIR)

    stats["total_seconds"] = round(time.time() - t_start, 1)
    _write_desc(out_dir, run_cfg, stats)
    for name in ("desc.txt", "run_config.json", "model_config.json", "run_log.txt"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            _mirror_file(path, RESULTS_DIR)
    _log(fh, f"Done in {stats['total_seconds']}s. Durable copy: {out_dir}")
    fh.close()
    _mirror_file(os.path.join(out_dir, "run_log.txt"), RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain TreeMoCo on the Patch-seq cohort.")
    parser.add_argument("--augment", default="upright", choices=sorted(PRESETS),
                        help="Augmentation preset: upright keeps the pia frame, paper does not.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--no_projector_bn", action="store_true",
                        help="Reproduce the paper's Table 5 ablation.")
    parser.add_argument("--node_type", action="store_true",
                        help="Append a soma/axon/dendrite one-hot to the node features "
                             "(d_in 30 -> 33). Not in the paper.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N cells.")
    parser.add_argument("--smoke", action="store_true", help="Small run into a smoke/ dir.")
    args = parser.parse_args()
    if args.smoke:
        if args.limit is None:
            args.limit = 100
        args.epochs = min(args.epochs, 5)

    MET_PATH = "/root/capsule/data/patchseq_MET/patchseq_MET.hdf5"
    SWC_PATH = "/root/capsule/data/patchseq_SWC/patchseq_SWC.hdf5"
    RESULTS_DIR = "/root/capsule/results"
    ARCHIVE_DIR = "/root/capsule/scratch/treemoco_pretrain"

    MIN_CLASS_SIZE = 25
    SEED = 0
    GENERIC_NAMES = {"desc.txt", "run_log.txt", "run_config.json", "model_config.json"}

    D_HIDDEN = 128
    D_PHI = 128
    D_PROJ = 128
    BATCH_SIZE = 64
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.2
    QUEUE_SIZE = 1024
    MOMENTUM = 0.99
    GRAD_CLIP = 5.0
    NUM_WORKERS = 4

    main(args)
