"""Driver: contrastively pretrain TreeMoCo on both MICrONS cohorts jointly, dump root embeddings.

The library modules under arborist_experiments.treemoco (tree_features, augment, batching,
model, pretrain) are imported unchanged -- they take a `Cell` and know nothing about which cohort it
came from. Only the orchestration is rewritten, because three things differ here:

  - **No modality axis.** The Patch-seq driver loops full/axon/dendrite and runs `filter_cell` per
    modality. That exists because the hand-designed representations emit fixed-length per-compartment
    blocks and `full` vs `dendrite` changes the feature layout. TreeMoCo consumes the tree itself, so
    there is nothing to keep matched: every node the cell has is passed in. Excitatory cells
    contribute soma+basal+apical, inhibitory ones soma+basal+axon, and the type one-hot is what lets
    the encoder tell those apart (TYPE_GROUPS merges basal and apical, so the apical call -- which
    presupposes having judged the cell pyramidal -- is not fed back in as a node feature).
  - **One encoder over both cohorts.** Pretraining is label-free, so the "has apical / has axon"
    separation that makes a *joint classifier* read the acquisition pipeline cannot leak here: there
    is no cohort label to predict. Probing and fine-tuning happen per cohort, inside each one's own
    label space, so the scored comparisons stay within-cohort.
  - **All cached cells, not just the labelled ones.** 57627 trees against the 55136 that survive
    labelling. Self-supervised training does not need the labels, so the unlabelled remainder is free
    data.

416 nucleus ids appear in *both* caches -- the same neuron reconstructed and annotated by two
pipelines, and genuinely different results (median soma-relative y extent 321 um in the excitatory
file vs 474 um in the inhibitory one; the excitatory copy carries apical where the inhibitory copy
carries axon). Both copies are kept: they are two legitimate annotations, and at 0.72% of the corpus
the false-negative cost of a duplicate landing in the MoCo queue is far below the noise floor. The
consequence that does matter is that specimen ids are NOT unique across the corpus, so every key
here is source-prefixed (`exc:195614`, `inh:195614`). Keying by bare id would silently merge them
and hand a probe the wrong cohort's embedding.

Two hyperparameters are re-derived rather than inherited, by the same rule their Patch-seq values
were chosen under (pretrain.py's docstring states the rule: the queue cannot exceed the dataset, and
the momentum must actually move the key encoder within the run's step count):

    queue_size  1024 -> 16384   1024 was capped by the 1618-tree Patch-seq cohort. 57627 trees at
                                batch 64 is ~900 steps/epoch, so 16384 refills every ~18 steps --
                                more negatives and fresher ones.
    momentum    0.99 -> 0.999   0.99 was chosen because 0.999 "barely moves the key encoder in 200
                                epochs of ~25 steps" (5k steps). Here an epoch is ~900 steps, so
                                0.999 is back in its intended regime and is MoCo's own default.

Both are CLI flags, so the Patch-seq values remain one argument away.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from arborist_experiments.treemoco.augment import PRESETS
from arborist_experiments.treemoco.model import TreeEncoder
from arborist_experiments.treemoco.pretrain import (
    PretrainConfig, embed, pretrain,
)
from arborist_experiments.treemoco.tree_features import (
    collapse, n_features,
)

from .data_loading import CACHE_PATH, INH_CACHE_PATH, cell_from_cache, load_cache
from .features import _run_jobs
from arborist_experiments.experiment_harness import (
    log as _log,
    mirror_file,
    next_version_dir as _next_version_dir,
)


SOURCE_CACHES = {"exc": CACHE_PATH, "inh": INH_CACHE_PATH}
_STR = h5py.string_dtype()


def _mirror_file(src: str, dst_dir: str) -> None:
    mirror_file(src, dst_dir, prefix="microns_treemoco_pretrain_", generic_names=GENERIC_NAMES)


def _job_collapse(cell):
    """Collapse the whole cell -- no modality filter. See the module docstring."""
    try:
        return cell.specimen_id, collapse(cell), None
    except Exception as e:
        return cell.specimen_id, None, f"collapse: {type(e).__name__}: {e}"


def build_corpus(limit: int | None = None, log=print) -> tuple[list, list[str], list[str]]:
    """Collapsed trees for every cached cell in both cohorts.

    Returns the trees, their source-prefixed keys, and the skip reasons. Order is exc then inh, and
    within a source it is cache row order, so a rerun reproduces the corpus exactly.
    """
    trees: list = []
    keys: list[str] = []
    skipped: list[str] = []
    for source, path in SOURCE_CACHES.items():
        cache = load_cache(path)
        n = len(cache.specimens) if limit is None else min(limit, len(cache.specimens))
        cells = [cell_from_cache(cache, row) for row in range(n)]
        t0 = time.time()
        results = _run_jobs(SimpleNamespace(cells=cells), _job_collapse)
        n_ok = 0
        for sid, tree, err in results:
            if err is not None:
                skipped.append(f"{source}:{sid}: {err}")
                continue
            trees.append(tree)
            keys.append(f"{source}:{sid}")
            n_ok += 1
        sizes = np.array([len(t.parents) for t in trees[-n_ok:]]) if n_ok else np.zeros(1)
        log(f"{source}: {n_ok}/{n} trees collapsed ({len(results) - n_ok} skipped), "
            f"collapsed nodes median {int(np.median(sizes))} "
            f"p90 {int(np.percentile(sizes, 90))} max {int(sizes.max())} "
            f"({time.time() - t0:.1f}s)")
    return trees, keys, skipped


def feature_names(d_hidden: int) -> list[str]:
    return [f"z_all_{d:03d}" for d in range(d_hidden)]


def _write_artifact(path: str, keys: list[str], Z: np.ndarray, attrs: dict) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("pooled/X", data=Z.astype(np.float32))
        f.create_dataset("pooled/specimen_ids", data=np.array(keys, dtype=object), dtype=_STR)
        f.create_dataset("pooled/feature_names",
                         data=np.array(feature_names(Z.shape[1]), dtype=object), dtype=_STR)
        for k, v in attrs.items():
            f.attrs[k] = v


def _write_history(path: str, history) -> None:
    with open(path, "w") as f:
        f.write("epoch,loss,top1,embed_std,lr\n")
        for s in history:
            f.write(f"{s.epoch},{s.loss:.6f},{s.top1:.6f},{s.embed_std:.6f},{s.lr:.6e}\n")


def _write_desc(out_dir: str, cfg: dict, stats: dict) -> None:
    lines = ["TreeMoCo contrastive pretraining on both MICrONS cohorts jointly "
             "(Chen et al. 2022, NeurIPS).", ""]
    lines += ["* Parameters"]
    for k, v in cfg.items():
        if isinstance(v, dict):
            lines += [f"    {k}.{ik} = {iv!r}" for ik, iv in v.items()]
        else:
            lines.append(f"    {k} = {v!r}")
    lines += ["", "* Run statistics"]
    lines += [f"    {k} = {v}" for k, v in stats.items()]
    lines += ["", "* Differences from the Patch-seq pretraining run",
              "    - No modality axis: the whole cell is collapsed, so excitatory cells contribute",
              "      soma+basal+apical and inhibitory ones soma+basal+axon. One encoder, one z",
              "      block, feature names z_all_<ddd>. Nothing is filtered by filter_cell.",
              "    - Both cohorts in one corpus. Pretraining is label-free, so the has-apical /",
              "      has-axon split that would let a joint *classifier* read the acquisition",
              "      pipeline cannot leak: there is no cohort label to predict. Probes are",
              "      per-cohort and stay inside each one's label space.",
              "    - All cached cells, labelled or not (57627 vs the 55136 that survive labelling).",
              "    - Keys are source-prefixed. 416 nucleus ids are in both caches -- the same",
              "      neuron annotated twice, with genuinely different results -- so a bare id is",
              "      not a unique key. Both copies are kept in the corpus.",
              "    - is_synthetic (node feature 30) is identically zero: MICrONS trees arrive",
              "      connected with one root that is the soma, so no parent link was invented.",
              "      On Patch-seq it flagged 42% of cells. That column is dead weight here.",
              "    - queue_size and momentum re-derived for the corpus size; see the module",
              "      docstring for the rule and the Patch-seq values."]
    if cfg.get("augment_config", {}).get("include_type"):
        lines += ["    - Node features carry a soma/axon/dendrite one-hot (d_in 33, not 30). This",
                  "      matters more here than on Patch-seq: the corpus mixes cells whose arbors",
                  "      are apical-labelled with cells whose arbors are axon-labelled, and without",
                  "      the one-hot the encoder cannot tell those apart. TYPE_GROUPS merges basal",
                  "      and apical on purpose -- the apical call presupposes having judged the cell",
                  "      pyramidal, so keeping it separate would feed the label back in."]
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

    aug = PRESETS[args.augment](include_type=not args.no_node_type)
    cfg = PretrainConfig(
        d_hidden=D_HIDDEN, d_phi=D_PHI, d_proj=D_PROJ, projector_bn=not args.no_projector_bn,
        epochs=args.epochs, batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        temperature=TEMPERATURE, queue_size=args.queue_size, momentum=args.momentum,
        grad_clip=GRAD_CLIP, num_workers=NUM_WORKERS, seed=SEED,
    )
    _log(fh, f"Augmentation preset: {aug.name} (include_type={aug.include_type})  "
             f"device: {cfg.device}  d_in: {n_features(aug.include_type)}")

    t0 = time.time()
    trees, keys, skipped = build_corpus(limit=args.limit, log=lambda m: _log(fh, m))
    sizes = np.array([len(t.parents) for t in trees])
    n_exc = sum(k.startswith("exc:") for k in keys)
    _log(fh, f"Corpus: {len(trees)} trees ({n_exc} exc + {len(trees) - n_exc} inh), "
             f"{len(skipped)} skipped, {int(sizes.sum())} collapsed nodes total "
             f"({time.time() - t0:.1f}s)")
    if skipped:
        with open(os.path.join(out_dir, "skipped.txt"), "w") as f:
            f.write("\n".join(skipped) + "\n")

    model_config = {"d_in": n_features(aug.include_type), "d_hidden": cfg.d_hidden,
                    "d_phi": cfg.d_phi, "d_proj": cfg.d_proj, "projector_bn": cfg.projector_bn}
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2, sort_keys=True)

    run_cfg = {"augment": aug.name,
               "augment_config": {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in vars(aug).items()},
               "pretrain_config": cfg.as_dict(), "model_config": model_config,
               "n_trees": len(trees), "n_exc": n_exc, "n_inh": len(trees) - n_exc,
               "steps_per_epoch": len(trees) // cfg.batch_size,
               "caches": SOURCE_CACHES, "coordinates": "upright, soma-centered",
               "modality": "whole cell (no filter)", "seed": SEED}
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2, sort_keys=True)

    _log(fh, f"Pretraining: {cfg.epochs} epochs x {len(trees) // cfg.batch_size} steps "
             f"= {cfg.epochs * (len(trees) // cfg.batch_size)} steps, "
             f"queue {cfg.queue_size}, momentum {cfg.momentum}")
    t_train = time.time()
    state, history = pretrain(trees, aug, cfg, log=lambda m: _log(fh, f"  {m}"))
    train_s = time.time() - t_train
    ckpt = os.path.join(out_dir, "treemoco_all.pth")
    torch.save(state, ckpt)
    hist_path = os.path.join(out_dir, "pretrain_history.csv")
    _write_history(hist_path, history)
    for path in (ckpt, hist_path, os.path.join(out_dir, "run_log.txt")):
        _mirror_file(path, RESULTS_DIR)

    encoder = TreeEncoder(d_in=model_config["d_in"], d_hidden=cfg.d_hidden, d_phi=cfg.d_phi)
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                             if k.startswith("encoder.")})
    t_embed = time.time()
    Z = embed(encoder, trees, aug, cfg)
    _log(fh, f"Embedded {Z.shape[0]} trees -> {Z.shape[1]} dims ({time.time() - t_embed:.1f}s)")

    h5_path = os.path.join(out_dir, "embeddings.h5")
    _write_artifact(h5_path, keys, Z, {
        "augment": aug.name, "n_trees": len(trees), "epochs": cfg.epochs,
        "coordinates": "upright, soma-centered", "modality": "whole cell (no filter)",
        "model_config": json.dumps(model_config), "pretrain_config": json.dumps(cfg.as_dict()),
        "key_format": "<source>:<specimen_id>", "seed": SEED,
    })

    stats = {"n_trees": len(trees), "n_exc": n_exc, "n_inh": len(trees) - n_exc,
             "n_skipped": len(skipped),
             "collapsed_nodes_median": int(np.median(sizes)),
             "collapsed_nodes_p90": int(np.percentile(sizes, 90)),
             "collapsed_nodes_max": int(sizes.max()),
             "embed_z_std": round(float(Z.std(axis=0).mean()), 5),
             "pretrain_seconds": round(train_s, 1)}
    if history:
        stats["final_loss"] = round(history[-1].loss, 4)
        stats["final_top1"] = round(history[-1].top1, 4)
        stats["final_embed_std"] = round(history[-1].embed_std, 5)
    stats["total_seconds"] = round(time.time() - t0, 1)
    _write_desc(out_dir, run_cfg, stats)

    for name in ("embeddings.h5", "desc.txt", "run_config.json", "model_config.json",
                 "run_log.txt", "skipped.txt"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            _mirror_file(path, RESULTS_DIR)
    _log(fh, f"Done in {stats['total_seconds']}s. Durable copy: {out_dir}")
    fh.close()
    _mirror_file(os.path.join(out_dir, "run_log.txt"), RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain TreeMoCo jointly on both MICrONS cohorts.")
    parser.add_argument("--augment", default="upright", choices=sorted(PRESETS),
                        help="upright keeps the pia frame (CORTICAL_AXIS=1 matches both caches); "
                             "paper PCA-normalizes and throws the laminar frame away.")
    parser.add_argument("--epochs", type=int, default=60,
                        help="~900 steps/epoch on the full corpus, so 60 epochs is ~54k steps "
                             "against the Patch-seq run's ~5k.")
    parser.add_argument("--queue_size", type=int, default=16384,
                        help="Patch-seq used 1024, capped by its 1618-tree cohort.")
    parser.add_argument("--momentum", type=float, default=0.999,
                        help="Patch-seq used 0.99 because its epochs were only ~25 steps.")
    parser.add_argument("--no_node_type", action="store_true",
                        help="Drop the soma/axon/dendrite one-hot (d_in 33 -> 30). The one-hot is "
                             "ON by default here: the corpus mixes apical-labelled and "
                             "axon-labelled arbors and the encoder needs to tell them apart.")
    parser.add_argument("--no_projector_bn", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N cells of EACH cache.")
    parser.add_argument("--smoke", action="store_true", help="Small run into a smoke/ dir.")
    args = parser.parse_args()
    if args.smoke:
        if args.limit is None:
            args.limit = 150
        args.epochs = min(args.epochs, 3)
        args.queue_size = min(args.queue_size, 256)

    RESULTS_DIR = "/root/capsule/results"
    ARCHIVE_DIR = "/root/capsule/scratch/microns_treemoco_pretrain"
    GENERIC_NAMES = {"desc.txt", "run_log.txt", "run_config.json", "model_config.json",
                     "skipped.txt", "pretrain_history.csv", "embeddings.h5"}

    SEED = 0
    D_HIDDEN = 128
    D_PHI = 128
    D_PROJ = 128
    BATCH_SIZE = 64
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.2
    GRAD_CLIP = 5.0
    # Measured, not assumed: raising this from 4 (the Patch-seq value) to 10 changed nothing --
    # 98 s/epoch either way, T4 at ~38%. The bottleneck is the level-synchronous Tree-LSTM walk in
    # batching.py, which launches one kernel per height band and so is latency- not throughput-bound;
    # more loader workers cannot help. Left at 10 because it costs nothing, but do not read it as a
    # tuned value. Note that changing it reshuffles each worker's augmentation RNG stream, so it does
    # change the run -- just not the objective or the data distribution.
    NUM_WORKERS = 10

    main(args)
