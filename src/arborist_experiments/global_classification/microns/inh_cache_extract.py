"""One-time extraction of `data/MICrONS_inh_swc` into the same cache layout as the exc cache.

Deliberately writes the identical dataset layout, so `data_loading.load_cache` and
`cell_from_cache` are reused verbatim rather than forked. Two datasets are added that the exc
cache has no analogue for: `root_id` (the source filename, kept for provenance since it is
*not* the join key) and `radius` (the inh SWCs carry real radii; the exc file has none).

Differences from `cache_extract.py`, all measured over all 7893 files:

  - **The frame is global, not upright.** Every cell is rotated into the upright frame with the
    matrix derived in `inh_id_map.derive_upright` before being soma-centered. Nothing downstream
    can tell the difference, which is the point -- `CORTICAL_AXIS = 1`, `R_EDGES` and `Z_EDGES`
    then mean the same thing they mean for the exc cohort.
  - **Types are {1, 2, 3}: soma, axon, basal. No apical.** 5251 of 7893 cells (66.5%) have axon
    nodes, so this is the cohort where the axon-vs-dendrite question the exc data structurally
    could not ask is actually testable.
  - `specimens` holds the **nucleus id**, resolved by `inh_id_map`, so the label join is the same
    string-keyed join the exc path uses. 69 of 7893 files are unresolved (5 have no soma node,
    64 have no nucleus within 2 um) and are skipped.
  - Trees are clean: exactly one root, and the root is the soma, in all 7888 files that have a
    soma. So there is no `_reparent_to_soma` here either and `synthetic_edge` stays all-False.
  - 36.6M nodes total, median 2647 per cell -- larger than the exc median of 1713 because of the
    axons.
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from .cache_extract import _extend
from .inh_id_map import (
    CACHE_DIR,
    MAP_PATH,
    SWC_DIR,
    UPRIGHT_PATH,
    load_map,
    load_upright,
    upright_matrix,
)

CACHE_VERSION = 1
OUT_PATH = os.path.join(CACHE_DIR, "microns_inh_swc.h5")
LABELS_PATH = os.path.join(CACHE_DIR, "inh_labels.csv")

TYPE_SOMA = 1
VALID_TYPES = (1, 2, 3)


def _check_invariants(types, parents, xyz) -> str | None:
    n = len(types)
    if n < 3:
        return f"only {n} nodes"
    if not np.isin(types, VALID_TYPES).all():
        return f"unexpected types {sorted(set(types.tolist()) - set(VALID_TYPES))}"
    soma = np.flatnonzero(types == TYPE_SOMA)
    if len(soma) != 1:
        return f"{len(soma)} soma nodes"
    roots = np.flatnonzero(parents == -1)
    if len(roots) != 1:
        return f"{len(roots)} roots"
    if roots[0] != soma[0]:
        return f"root {roots[0]} is not the soma {soma[0]}"
    if parents.min() < -1 or parents.max() >= n:
        return f"parents out of range [{parents.min()}, {parents.max()}]"
    if (parents == np.arange(n)).any():
        return "self-parent"
    if not np.isfinite(xyz).all():
        return "non-finite coordinates"
    return None


def _init_cache(out: h5py.File, theta_deg: float) -> None:
    out.attrs["cache_version"] = CACHE_VERSION
    out.attrs["source_dir"] = SWC_DIR
    out.attrs["upright_theta_deg"] = theta_deg
    out.attrs["n_cells_done"] = 0
    out.attrs["complete"] = False
    kw = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    out.create_dataset("specimens", shape=(0,), maxshape=(None,), dtype="S8", **kw)
    out.create_dataset("root_id", shape=(0,), maxshape=(None,), dtype=np.int64, **kw)
    out.create_dataset("source_row", shape=(0,), maxshape=(None,), dtype=np.int64, **kw)
    out.create_dataset("n_nodes", shape=(0,), maxshape=(None,), dtype=np.int32, **kw)
    out.create_dataset("offsets", data=np.zeros(1, dtype=np.int64), maxshape=(None,), **kw)
    out.create_dataset("soma_origin", shape=(0, 3), maxshape=(None, 3), dtype=np.float32, **kw)
    out.create_dataset("pia_y", shape=(0,), maxshape=(None,), dtype=np.float32, **kw)
    out.create_dataset("wm_y", shape=(0,), maxshape=(None,), dtype=np.float32, **kw)
    out.create_dataset("xyz", shape=(0, 3), maxshape=(None, 3), dtype=np.float32,
                       chunks=(65536, 3), **kw)
    out.create_dataset("radius", shape=(0,), maxshape=(None,), dtype=np.float32,
                       chunks=(262144,), **kw)
    out.create_dataset("types", shape=(0,), maxshape=(None,), dtype=np.uint8,
                       chunks=(262144,), **kw)
    out.create_dataset("parents", shape=(0,), maxshape=(None,), dtype=np.int32,
                       chunks=(262144,), **kw)


def _append(out: h5py.File, recs: list[dict]) -> None:
    if not recs:
        return
    _extend(out["xyz"], np.concatenate([r["xyz"] for r in recs]))
    _extend(out["radius"], np.concatenate([r["radius"] for r in recs]))
    _extend(out["types"], np.concatenate([r["types"] for r in recs]))
    _extend(out["parents"], np.concatenate([r["parents"] for r in recs]))

    n_nodes = np.array([r["n_nodes"] for r in recs], dtype=np.int64)
    off = out["offsets"]
    _extend(off, int(off[-1]) + np.cumsum(n_nodes))

    _extend(out["specimens"], np.array([r["specimen_id"] for r in recs], dtype="S8"))
    _extend(out["root_id"], np.array([r["root_id"] for r in recs], dtype=np.int64))
    _extend(out["source_row"], np.array([r["source_row"] for r in recs], dtype=np.int64))
    _extend(out["n_nodes"], n_nodes.astype(np.int32))
    _extend(out["soma_origin"], np.stack([r["soma_origin"] for r in recs]))
    _extend(out["pia_y"], np.array([r["pia_y"] for r in recs], dtype=np.float32))
    _extend(out["wm_y"], np.array([r["wm_y"] for r in recs], dtype=np.float32))


def extract(swc_dir: str = SWC_DIR, out_path: str = OUT_PATH, map_path: str = MAP_PATH,
            upright_path: str = UPRIGHT_PATH, block: int = 500, limit: int | None = None) -> str:
    import pandas as pd

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = os.path.join(os.path.dirname(out_path), "inh_extract_log.txt")
    skipped_path = os.path.join(os.path.dirname(out_path), "inh_skipped.csv")

    up = load_upright(upright_path)
    R = upright_matrix(up["theta_deg"])
    imap = load_map(map_path).set_index("root_id")

    files = sorted(f for f in os.listdir(swc_dir) if f.endswith(".csv"))
    if limit is not None:
        files = files[:limit]

    with h5py.File(out_path, "w") as out, open(log_path, "w") as logf, \
            open(skipped_path, "w") as skipf:
        _init_cache(out, up["theta_deg"])
        skipf.write("source_row,root_id,reason\n")

        recs, skips = [], []
        t0 = time.time()
        for row, name in enumerate(files):
            root = int(os.path.splitext(name)[0])
            m = imap.loc[root]
            if int(m["nucleus_id"]) < 0:
                skips.append((row, root, str(m["match_method"])))
                continue
            df = pd.read_csv(os.path.join(swc_dir, name),
                             usecols=["id", "type", "parent", "x", "y", "z", "radius"])
            ids = df["id"].to_numpy()
            types = df["type"].to_numpy().astype(np.int32)
            parents = df["parent"].to_numpy().astype(np.int64)
            xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)

            # Parents reference `id`, which is 0..n-1 in every file; assert rather than assume,
            # because a non-contiguous id column would silently corrupt the tree.
            if not (ids == np.arange(len(ids))).all():
                skips.append((row, root, "id column is not 0..n-1"))
                continue
            reason = _check_invariants(types, parents, xyz)
            if reason is not None:
                skips.append((row, root, reason))
                continue

            xyz_up = xyz @ R.T
            origin = xyz_up[types == TYPE_SOMA].mean(axis=0)
            recs.append({
                "specimen_id": str(int(m["nucleus_id"])), "root_id": root, "source_row": row,
                "n_nodes": len(ids), "xyz": (xyz_up - origin).astype(np.float32),
                "radius": df["radius"].to_numpy(dtype=np.float32),
                "types": types.astype(np.uint8), "parents": parents.astype(np.int32),
                "soma_origin": xyz[types == TYPE_SOMA].mean(axis=0).astype(np.float32),
                "pia_y": float(m["pia_y"]), "wm_y": float(m["wm_y"]),
            })

            if len(recs) >= block or row == len(files) - 1:
                _append(out, recs)
                out.attrs["n_cells_done"] = int(out["specimens"].shape[0])
                out.flush()
                skipf.writelines(f"{r_},{rid},{why}\n" for r_, rid, why in skips)
                skipf.flush()
                dt = time.time() - t0
                msg = (f"rows ..{row + 1}: {out['specimens'].shape[0]} kept, "
                       f"{len(skips)} skipped this block, {dt:.1f}s")
                print(msg, flush=True)
                logf.write(msg + "\n")
                logf.flush()
                recs, skips = [], []
                t0 = time.time()

        if recs or skips:
            _append(out, recs)
            skipf.writelines(f"{r_},{rid},{why}\n" for r_, rid, why in skips)
            out.attrs["n_cells_done"] = int(out["specimens"].shape[0])
        out.attrs["complete"] = True
        n = int(out["specimens"].shape[0])
    print(f"wrote {out_path}: {n} cells", flush=True)
    return out_path


def convert_labels(map_path: str = MAP_PATH, out_csv: str = LABELS_PATH) -> str:
    """Nucleus-keyed labels for the resolved inh cells, in the same schema as coarse_labels.csv."""
    import pandas as pd

    from .cache_extract import LABELS_PARQUET

    lab = pd.read_parquet(LABELS_PARQUET)
    lab["cell_id"] = lab["cell_id"].astype(str)
    resolved = load_map(map_path)
    resolved = resolved[resolved["nucleus_id"] >= 0]
    keep = set(resolved["nucleus_id"].astype(int).astype(str))
    out = lab[lab["cell_id"].isin(keep)].copy()
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(out)} of {len(resolved)} resolved cells carry a label")
    print(out["coarse_label"].value_counts().to_string())
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--swc_dir", default=SWC_DIR)
    p.add_argument("--out", default=OUT_PATH)
    p.add_argument("--map", default=MAP_PATH)
    p.add_argument("--upright", default=UPRIGHT_PATH)
    p.add_argument("--block", type=int, default=500)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--labels", action="store_true")
    args = p.parse_args()

    if args.labels:
        convert_labels(args.map)
    t0 = time.time()
    extract(args.swc_dir, args.out, args.map, args.upright, args.block, args.limit)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
