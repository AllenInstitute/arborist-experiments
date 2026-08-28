"""Path encodings for the Patch-seq cohort.

Runs the pretrained arborist CurveEncoder over every unbranched path in each Patch-seq
reconstruction and writes the per-path latents to a versioned HDF5 artifact, alongside a
pooled per-cell feature matrix.

The encoder was trained on paths from SkeletonGraph(node_spacing=5), and CurveDataset applies
no scale normalization -- it only first-differences -- so the latents are only meaningful if
the input diffs carry the same absolute magnitude the encoder saw in training. Patch-seq nodes
sit at a median 1.15 um spacing against that 5 um target, so every path is arc-length resampled
to 5 um before differencing. See resample_path and prepare_curve for the two places this
matters.

Outputs go to a fresh version directory under /root/capsule/scratch/path_encodings/ and are
mirrored into /root/capsule/results/ after every chunk. /root/capsule/results is wiped between
sessions, so it can never be the only copy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import h5py
import networkx as nx
import numpy as np
import torch

from arborist.data.datasets import collate_curves
from arborist.models.curve_transformer import CurveAutoencoder

from arborist_experiments.patchseq_morphology_benchmark.data_loading import (
    Cohort, TYPE_SOMA, _iter_swc, build_cohort,
)
from arborist_experiments.patchseq_morphology_benchmark.features import _run_jobs
from arborist_experiments.patchseq_morphology_benchmark.radial_arbor_features import (
    STRUCTURE_GROUPINGS, _real_edges,
)


GROUP_NAMES = ["axon", "dendrite"]


# --- Path extraction ---
def extract_paths(cell, struct_ids: list[int]) -> list[np.ndarray]:
    """Node-index paths between irreducible (degree != 2) nodes, over traced edges only.

    Mirrors SkeletonGraph.irreducible_paths, which defines the unit the encoder was trained
    on. The soma joins the group so that arbor-to-soma edges connect and the soma acts as a
    hub terminating each stem; _real_edges drops the links _reparent_to_soma invented, which
    are 32% of edge length and were never traced.
    """
    child = _real_edges(cell, struct_ids, include_soma=True)
    if len(child) == 0:
        return []

    graph = nx.Graph()
    graph.add_edges_from(zip(cell.parents[child].tolist(), child.tolist()))
    irreducible = {n for n in graph.nodes if graph.degree(n) != 2}

    paths = []
    visited_edges = set()
    for source in irreducible:
        for nb in graph.neighbors(source):
            edge = frozenset((source, nb))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            path = [source, nb]
            prev, curr = source, nb
            while curr not in irreducible:
                nxt = next((n for n in graph.neighbors(curr) if n != prev), None)
                if nxt is None:
                    break
                visited_edges.add(frozenset((curr, nxt)))
                path.append(nxt)
                prev, curr = curr, nxt
            paths.append(np.asarray(path, dtype=np.int64))
    return paths


def resample_path(xyz: np.ndarray, step: float) -> np.ndarray | None:
    """Points spaced exactly `step` um along the path's arc length.

    A trailing partial step is dropped rather than appended: the whole point of resampling is
    that every diff has the magnitude the encoder trained on, and a short final diff would
    break that. Returns None for a path shorter than one step.
    """
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 0])          # np.interp needs increasing arc length
    xyz = xyz[keep]
    if len(xyz) < 2:
        return None
    arc = np.concatenate([[0.0], np.cumsum(seg[seg > 0])])
    if arc[-1] < step:
        return None
    t = np.arange(0.0, arc[-1], step)
    return np.stack([np.interp(t, arc, xyz[:, k]) for k in range(3)], axis=1)


def prepare_curve(xyz: np.ndarray, segment_len: int) -> tuple[np.ndarray, int, int]:
    """First differences, zero-padded up to a whole number of segment tokens.

    CurveEncoder slices to n_segments * segment_len points, so any tail beyond a multiple of
    segment_len is dropped from the geometry while still perturbing the arc-length positional
    encoding. Padding up to the next multiple keeps every real point inside a segment. Zero
    diffs contribute no arc length, so arc_pos stays correct, and padding short curves up to
    one full token follows MergeGraphDataset's min_curve_len.
    """
    n_real = len(xyz)
    n_target = int(np.ceil(n_real / segment_len) * segment_len)
    diffs = np.zeros((n_target, 3), dtype=np.float32)
    diffs[1:n_real] = np.diff(xyz, axis=0)
    return diffs, n_real, n_target - n_real


def _job_paths(cell, node_spacing: float, segment_len: int, min_points: int):
    """Resampled, differenced curves for both arbor groups. Runs in a worker process."""
    groups: dict[str, dict] = {}
    for group in GROUP_NAMES:
        curves, lengths, n_points, n_padded = [], [], [], []
        n_dropped, cable_dropped = 0, 0.0
        for path in extract_paths(cell, STRUCTURE_GROUPINGS[group]):
            if np.all(cell.types[path] == TYPE_SOMA):
                continue                              # soma-internal, belongs to no arbor
            xyz = cell.xyz[path]
            raw_length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
            resampled = resample_path(xyz, node_spacing)
            if resampled is None or len(resampled) < min_points:
                n_dropped += 1
                cable_dropped += raw_length
                continue
            diffs, n_real, n_pad = prepare_curve(resampled, segment_len)
            curves.append(diffs)
            lengths.append(float((n_real - 1) * node_spacing))
            n_points.append(n_real)
            n_padded.append(n_pad)
        groups[group] = {
            "curves": curves,
            "cable_length": np.asarray(lengths, dtype=np.float32),
            "n_points": np.asarray(n_points, dtype=np.int32),
            "n_padded": np.asarray(n_padded, dtype=np.int32),
            "n_dropped": n_dropped,
            "cable_dropped": cable_dropped,
            "cable_total": float(cable_dropped + sum(lengths)),
        }
    return cell.specimen_id, groups


# --- Encoding ---
def encode_curves(model, curves: list[np.ndarray], device: str, batch_size: int) -> np.ndarray:
    """Per-curve CurveEncoder latents.

    Curves are sorted by length so each batch pads little. Every curve is already a multiple
    of segment_len, so the batch maximum is too and the encoder's per-segment mask
    (mask[:, ::segment_len]) lands on segment boundaries.
    """
    latent_dim = model.config["latent_dim"]
    if not curves:
        return np.zeros((0, latent_dim), dtype=np.float32)

    order = np.argsort([len(c) for c in curves], kind="stable")
    out = np.empty((len(curves), latent_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            padded, mask = collate_curves([curves[i] for i in idx])
            z, _ = model.encoder(padded.to(device), mask.to(device))
            out[idx] = z.float().cpu().numpy()
    return out


def pool_latents(z: np.ndarray, cable_length: np.ndarray, group: np.ndarray,
                 latent_dim: int) -> np.ndarray:
    """Cable-length-weighted mean and std of per-path latents, per arbor group.

    Length weighting is what demotes the zero-padded stubs. A group with no cable yields an
    all-zero block -- a true zero that needs no imputation, matching radial_arbor_features.
    """
    blocks = []
    for name in GROUP_NAMES:
        m = group == name
        w = cable_length[m].astype(np.float64)
        if not m.any() or w.sum() <= 0:
            blocks.append(np.zeros(2 * latent_dim, dtype=np.float32))
            continue
        w = w / w.sum()
        zz = z[m].astype(np.float64)
        mu = (w[:, None] * zz).sum(axis=0)
        var = (w[:, None] * (zz - mu) ** 2).sum(axis=0)
        blocks.append(np.concatenate([mu, np.sqrt(np.maximum(var, 0.0))]).astype(np.float32))
    return np.concatenate(blocks)


def pooled_feature_names(latent_dim: int) -> list[str]:
    return [
        f"{stat}_{g}_z{d:03d}"
        for g in GROUP_NAMES
        for stat in ("mean", "std")
        for d in range(latent_dim)
    ]


# --- Checkpoint loading ---
def _val_loss_from_name(name: str) -> float:
    match = re.search(r"-(\d+\.\d+)\.pth$", name)
    return float(match.group(1)) if match else float("inf")


def resolve_checkpoint(train_root: str, latent_dim: int) -> tuple[str, str]:
    """Best-val-loss v6 checkpoint for `latent_dim`, with its sidecar model_config.json."""
    subdir = os.path.join(train_root, f"curve_autoencoder_v6_latent{latent_dim}")
    ckpts = [f for f in os.listdir(subdir) if f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints in {subdir}")
    best = min(ckpts, key=_val_loss_from_name)
    return os.path.join(subdir, best), os.path.join(subdir, "model_config.json")


def load_encoder(ckpt_path: str, config_path: str, device: str):
    """CurveAutoencoder from a bare state_dict plus its sidecar config.

    CurveAutoencoder.load() expects checkpoint["config"] and ["model_state_dict"], but
    train.py saves model.state_dict() directly, so it raises on every checkpoint under data/.
    This follows evaluate.py's loading path instead.
    """
    with open(config_path) as f:
        config = json.load(f)
    model = CurveAutoencoder(**config)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


# --- Artifact ---
_STR = h5py.string_dtype()


def _init_artifact(path: str, latent_dim: int, attrs: dict) -> None:
    with h5py.File(path, "w") as f:
        for name, shape, dtype in [
            ("path_latents/z", (0, latent_dim), "f4"),
            ("path_latents/specimen_id", (0,), _STR),
            ("path_latents/group", (0,), _STR),
            ("path_latents/cable_length", (0,), "f4"),
            ("path_latents/n_points", (0,), "i4"),
            ("path_latents/n_padded", (0,), "i4"),
            ("pooled/X", (0, 4 * latent_dim), "f4"),
            ("pooled/specimen_ids", (0,), _STR),
        ]:
            maxshape = (None,) + shape[1:]
            f.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=True)
        f.create_dataset(
            "pooled/feature_names",
            data=np.array(pooled_feature_names(latent_dim), dtype=object), dtype=_STR,
        )
        for k, v in attrs.items():
            f.attrs[k] = v


def _append(f: h5py.File, name: str, values) -> None:
    dset = f[name]
    values = np.asarray(values) if not isinstance(values, np.ndarray) else values
    n = len(values)
    if n == 0:
        return
    dset.resize(dset.shape[0] + n, axis=0)
    dset[-n:] = values


def _mirror_file(src: str, dst_dir: str) -> None:
    """Copy one file into the flat results dir, prefixing the generic names.

    results/ is shared with the benchmark and probe runs, which write their own desc.txt /
    run_log.txt / run_config.json; an unprefixed copy silently overwrites theirs. Written via
    a .partial rename so a mirror interrupted mid-copy never leaves a truncated file behind.
    """
    os.makedirs(dst_dir, exist_ok=True)
    name = os.path.basename(src)
    if name in GENERIC_NAMES:
        name = f"path_encodings_{name}"
    tmp = os.path.join(dst_dir, name + ".partial")
    with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
        while chunk := fsrc.read(8 << 20):
            fdst.write(chunk)
    os.replace(tmp, os.path.join(dst_dir, name))


def _write_desc(out_dir: str, cfg: dict, stats: dict) -> None:
    lines = ["Path encodings for the Patch-seq cohort.", ""]
    lines += ["* Parameters"]
    lines += [f"    {k} = {v!r}" for k, v in cfg.items()]
    lines += ["", "* Retention"]
    lines += [f"    {k} = {v}" for k, v in stats.items()]
    with open(os.path.join(out_dir, "desc.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# --- Driver ---
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


def build_input_cohort(met_path: str, swc_path: str, min_class_size: int,
                       all_cells: bool) -> Cohort:
    if not all_cells:
        return build_cohort(met_path=met_path, swc_path=swc_path,
                            min_class_size=min_class_size)
    cells = list(_iter_swc(swc_path))
    return Cohort(cells=cells, labels=np.zeros(len(cells), dtype=np.int64),
                  label_names=["unlabeled"])


def main(args):
    if args.smoke:
        out_dir = os.path.join(ARCHIVE_DIR, "smoke")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = _next_version_dir(ARCHIVE_DIR)

    fh = open(os.path.join(out_dir, "run_log.txt"), "w")
    _log(fh, f"Durable output dir: {out_dir} (mirrored to {RESULTS_DIR})")

    ckpt_path, config_path = resolve_checkpoint(TRAIN_ROOT, args.latent_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_encoder(ckpt_path, config_path, device)
    latent_dim = model.config["latent_dim"]
    segment_len = model.config["segment_len"]
    _log(fh, f"Encoder: {os.path.basename(ckpt_path)} on {device}")
    _log(fh, f"  latent_dim={latent_dim} segment_len={segment_len} "
             f"node_spacing={NODE_SPACING} min_points={args.min_points}")

    cohort = build_input_cohort(MET_PATH, SWC_PATH, MIN_CLASS_SIZE, args.all_cells)
    cells = cohort.cells[:args.limit] if args.limit else cohort.cells
    _log(fh, f"Cells to encode: {len(cells)}"
             f"{' (limited)' if args.limit else ''}")

    cfg = {
        "checkpoint": ckpt_path, "model_config": model.config,
        "node_spacing": NODE_SPACING, "segment_len": segment_len,
        "min_points": args.min_points, "n_cells": len(cells),
        "all_cells": bool(args.all_cells), "min_class_size": MIN_CLASS_SIZE,
        "coordinates": "swc_upright", "swc_path": SWC_PATH, "met_path": MET_PATH,
        "seed": SEED,
    }
    h5_path = os.path.join(out_dir, "latents.h5")
    _init_artifact(h5_path, latent_dim, {
        "checkpoint": ckpt_path, "node_spacing": NODE_SPACING,
        "segment_len": segment_len, "min_points": args.min_points,
        "n_cells_requested": len(cells), "coordinates": "swc_upright",
        "model_config": json.dumps(model.config), "seed": SEED,
    })
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    totals = {
        "n_paths": 0, "n_dropped": 0, "diff_norm_sum": 0.0, "diff_norm_count": 0,
        **{f"cable_total_{g}": 0.0 for g in GROUP_NAMES},
        **{f"cable_dropped_{g}": 0.0 for g in GROUP_NAMES},
        **{f"n_paths_{g}": 0 for g in GROUP_NAMES},
        **{f"cells_without_{g}": 0 for g in GROUP_NAMES},
    }

    t_start = time.time()
    for start in range(0, len(cells), CHUNK_SIZE):
        chunk = cells[start:start + CHUNK_SIZE]
        sub = Cohort(cells=chunk, labels=np.zeros(len(chunk), dtype=np.int64),
                     label_names=cohort.label_names)
        results = _run_jobs(sub, _job_paths, NODE_SPACING, segment_len, args.min_points)

        curves, sids, groups, lengths, n_points, n_padded = [], [], [], [], [], []
        spans = []
        for sid, per_group in results:
            span_start = len(curves)
            for g in GROUP_NAMES:
                d = per_group[g]
                curves.extend(d["curves"])
                sids.extend([sid] * len(d["curves"]))
                groups.extend([g] * len(d["curves"]))
                lengths.append(d["cable_length"])
                n_points.append(d["n_points"])
                n_padded.append(d["n_padded"])
                totals["n_dropped"] += d["n_dropped"]
                totals[f"n_paths_{g}"] += len(d["curves"])
                totals[f"cable_total_{g}"] += d["cable_total"]
                totals[f"cable_dropped_{g}"] += d["cable_dropped"]
                if len(d["curves"]) == 0:
                    totals[f"cells_without_{g}"] += 1
            spans.append((sid, span_start, len(curves)))

        z = encode_curves(model, curves, device, BATCH_SIZE)
        group_arr = np.array(groups, dtype=object)
        length_arr = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.float32)

        pooled_rows, pooled_ids = [], []
        for sid, lo, hi in spans:
            pooled_rows.append(
                pool_latents(z[lo:hi], length_arr[lo:hi], group_arr[lo:hi], latent_dim))
            pooled_ids.append(sid)

        with h5py.File(h5_path, "a") as f:
            _append(f, "path_latents/z", z)
            _append(f, "path_latents/specimen_id", np.array(sids, dtype=object))
            _append(f, "path_latents/group", group_arr)
            _append(f, "path_latents/cable_length", length_arr)
            _append(f, "path_latents/n_points",
                    np.concatenate(n_points) if n_points else np.zeros(0, dtype=np.int32))
            _append(f, "path_latents/n_padded",
                    np.concatenate(n_padded) if n_padded else np.zeros(0, dtype=np.int32))
            _append(f, "pooled/X",
                    np.vstack(pooled_rows) if pooled_rows
                    else np.zeros((0, 4 * latent_dim), dtype=np.float32))
            _append(f, "pooled/specimen_ids", np.array(pooled_ids, dtype=object))

        totals["n_paths"] += len(curves)
        for c, npt in zip(curves, np.concatenate(n_points) if n_points else []):
            norms = np.linalg.norm(c[1:int(npt)], axis=1)
            totals["diff_norm_sum"] += float(norms.sum())
            totals["diff_norm_count"] += len(norms)

        done = min(start + CHUNK_SIZE, len(cells))
        _log(fh, f"  cells {done}/{len(cells)}  paths so far {totals['n_paths']}  "
                 f"({time.time() - t_start:.1f}s)")
        _mirror_file(h5_path, RESULTS_DIR)

    mean_norm = totals["diff_norm_sum"] / max(totals["diff_norm_count"], 1)
    _log(fh, "")
    _log(fh, f"Encoded {totals['n_paths']} paths from {len(cells)} cells "
             f"in {time.time() - t_start:.1f}s")
    _log(fh, f"Mean real diff norm: {mean_norm:.3f} um (target {NODE_SPACING})")
    _log(fh, f"Paths dropped below min_points={args.min_points}: {totals['n_dropped']}")
    for g in GROUP_NAMES:
        tot = totals[f"cable_total_{g}"]
        kept = 100.0 * (1.0 - totals[f"cable_dropped_{g}"] / tot) if tot > 0 else 0.0
        _log(fh, f"  {g}: {totals[f'n_paths_{g}']} paths, {kept:.1f}% of traced cable "
                 f"retained, {totals[f'cells_without_{g}']} cells with none")

    stats = {
        "n_paths": totals["n_paths"], "n_dropped": totals["n_dropped"],
        "mean_diff_norm_um": round(mean_norm, 4),
        **{f"n_paths_{g}": totals[f"n_paths_{g}"] for g in GROUP_NAMES},
        **{f"cells_without_{g}": totals[f"cells_without_{g}"] for g in GROUP_NAMES},
    }
    _write_desc(out_dir, cfg, stats)
    for name in ("run_log.txt", "run_config.json", "desc.txt"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            _mirror_file(path, RESULTS_DIR)
    _mirror_file(h5_path, RESULTS_DIR)
    _log(fh, f"Done. Durable copy: {out_dir}")
    fh.close()
    _mirror_file(os.path.join(out_dir, "run_log.txt"), RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate path encodings for Patch-seq cells.")
    parser.add_argument("--latent_dim", type=int, default=64,
                        choices=[8, 16, 32, 64, 128],
                        help="Selects the v6 checkpoint subdirectory.")
    parser.add_argument("--min_points", type=int, default=3,
                        help="Drop paths yielding fewer resampled points than this.")
    parser.add_argument("--all_cells", action="store_true",
                        help="Encode every cell in the SWC file, not just the labelled cohort.")
    parser.add_argument("--limit", type=int, default=None, help="Encode only the first N cells.")
    parser.add_argument("--smoke", action="store_true", help="Small run into a smoke/ dir.")
    args = parser.parse_args()
    if args.smoke and args.limit is None:
        args.limit = 20

    MET_PATH = "/root/capsule/data/patchseq_MET/patchseq_MET.hdf5"
    SWC_PATH = "/root/capsule/data/patchseq_SWC/patchseq_SWC.hdf5"
    TRAIN_ROOT = "/root/capsule/data/curve_autoencoder_train_v6"
    RESULTS_DIR = "/root/capsule/results"
    ARCHIVE_DIR = "/root/capsule/scratch/path_encodings"

    MIN_CLASS_SIZE = 25
    SEED = 0

    GENERIC_NAMES = {"desc.txt", "run_log.txt", "run_config.json"}

    NODE_SPACING = 5.0
    BATCH_SIZE = 256
    CHUNK_SIZE = 200

    main(args)
