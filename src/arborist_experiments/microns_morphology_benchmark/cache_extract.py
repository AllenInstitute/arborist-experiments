"""One-time extraction of EM_syn.hdf5 into a flat, fast-to-read cache.

Why a cache at all: the source stores each cell as one element of a vlen object dataset with
chunks=(1,). Reads are 15-160x slower out of ascending order because the vlen global heap lays
adjacent cells out adjacently, so only a sequential scan faults in useful blocks:

    contiguous ascending   2.4 ms/cell (3 arrays)  ->  ~2 min for all 49803
    sparse-random          37-400 ms/cell          ->  0.5-5.4 h
    contiguous slice read  pathological (a 500-element slice did not finish in 600 s)

Every read here is therefore a single ascending element-wise pass, and nothing downstream
touches the source file again.
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np


CACHE_VERSION = 1

SOURCE_PATH = "/root/capsule/data/EM_exc_conf/EM_syn.hdf5"
LABELS_PARQUET = "/root/capsule/data/microns_fine_labels/microns_fine_labels.parquet"
CACHE_DIR = "/root/capsule/scratch/microns_cache/v1"

TYPE_SOMA = 1
VALID_TYPES = (1, 2, 3, 4)


def _decode(x) -> str:
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def _boundary_y(poly: np.ndarray) -> float:
    """y of the pia/wm trace at the point closest to the soma in x.

    The traces are in the same soma-centered upright frame as swc_upright (soma at x=0),
    so this is a signed soma-to-boundary distance along the cortical axis.
    """
    if poly is None or poly.ndim != 2 or len(poly) == 0:
        return np.nan
    return float(poly[np.argmin(np.abs(poly[:, 0])), 1])


def _check_invariants(types: np.ndarray, parents: np.ndarray, xyz: np.ndarray) -> str | None:
    n = len(types)
    if n < 3:
        return f"only {n} nodes"
    if len(parents) != n or len(xyz) != n:
        return f"length mismatch types={n} parents={len(parents)} xyz={len(xyz)}"
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
    # The frame convention the whole port depends on: soma at exactly (0, 0) in x and y,
    # column 2 absolute. If a future file drops this, fail loudly rather than mis-center.
    if abs(xyz[soma[0], 0]) > 1e-3 or abs(xyz[soma[0], 1]) > 1e-3:
        return f"soma not at x=y=0: {xyz[soma[0]]}"
    return None


def extract(
    source_path: str = SOURCE_PATH,
    out_path: str | None = None,
    block: int = 2000,
    limit: int | None = None,
) -> str:
    out_path = out_path or os.path.join(CACHE_DIR, "em_exc_swc.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = os.path.join(os.path.dirname(out_path), "extract_log.txt")
    skipped_path = os.path.join(os.path.dirname(out_path), "skipped.csv")

    with h5py.File(source_path, "r") as src:
        n_src = src["swc_upright/data"].shape[0]
        n_total = n_src if limit is None else min(limit, n_src)

        t0 = time.time()
        shapes = {k: src[f"{k}/shape"][:] for k in ("swc_upright", "pia_upright", "wm_upright")}
        meta_specimens = [_decode(s) for s in src["meta/specimens"][:]]
        print(f"read shape tables in {time.time() - t0:.1f}s", flush=True)
        # swc_upright/specimens is vlen and costs ~4 ms/element, so read only the rows this
        # invocation needs. It is NOT in meta/specimens order -- the 8 conformal-stage failures
        # are interleaved, not appended -- so it cannot be substituted by the cheap meta read.
        t0 = time.time()
        spec_ds = src["swc_upright/specimens"]
        specimens = [_decode(spec_ds[i]) for i in range(n_total)]
        spec_read_s = time.time() - t0
        print(f"read {n_total} specimen ids in {spec_read_s:.1f}s", flush=True)

        done = _resume_point(out_path)
        if done >= n_total:
            print(f"cache already complete at {done} cells")
            return out_path

        mode = "r+" if done > 0 else "w"
        with h5py.File(out_path, mode) as out, \
                open(log_path, "a") as logf, open(skipped_path, "a" if done else "w") as skipf:
            if mode == "w":
                _init_cache(out, meta_specimens, spec_read_s)
                skipf.write("source_row,specimen_id,reason\n")

            for lo in range(done, n_total, block):
                hi = min(lo + block, n_total)
                t0 = time.time()
                recs, skips = _read_block(src, shapes, specimens, lo, hi)
                _append(out, recs)
                out.attrs["n_cells_done"] = hi
                out.flush()
                for row, sid, reason in skips:
                    skipf.write(f"{row},{sid},{reason}\n")
                skipf.flush()
                dt = time.time() - t0
                msg = (f"rows {lo}-{hi}: {len(recs)} kept, {len(skips)} skipped, "
                       f"{dt:.1f}s ({dt / max(1, hi - lo) * 1000:.2f} ms/cell)")
                print(msg, flush=True)
                logf.write(msg + "\n")
                logf.flush()

            out.attrs["complete"] = True
    return out_path


def _init_cache(out: h5py.File, meta_specimens: list[str], spec_read_s: float) -> None:
    out.attrs["cache_version"] = CACHE_VERSION
    out.attrs["source_path"] = SOURCE_PATH
    out.attrs["n_cells_done"] = 0
    out.attrs["complete"] = False
    out.attrs["specimens_read_seconds"] = spec_read_s
    out.create_dataset("meta_specimens", data=np.array(meta_specimens, dtype="S8"))
    kw = dict(compression="gzip", compression_opts=4, shuffle=True)
    out.create_dataset("specimens", shape=(0,), maxshape=(None,), dtype="S8", **kw)
    out.create_dataset("source_row", shape=(0,), maxshape=(None,), dtype=np.int64, **kw)
    out.create_dataset("n_nodes", shape=(0,), maxshape=(None,), dtype=np.int32, **kw)
    out.create_dataset("offsets", data=np.zeros(1, dtype=np.int64), maxshape=(None,), **kw)
    out.create_dataset("soma_origin", shape=(0, 3), maxshape=(None, 3), dtype=np.float32, **kw)
    out.create_dataset("pia_y", shape=(0,), maxshape=(None,), dtype=np.float32, **kw)
    out.create_dataset("wm_y", shape=(0,), maxshape=(None,), dtype=np.float32, **kw)
    out.create_dataset("xyz", shape=(0, 3), maxshape=(None, 3), dtype=np.float32,
                       chunks=(65536, 3), **kw)
    out.create_dataset("types", shape=(0,), maxshape=(None,), dtype=np.uint8,
                       chunks=(262144,), **kw)
    out.create_dataset("parents", shape=(0,), maxshape=(None,), dtype=np.int32,
                       chunks=(262144,), **kw)


def _resume_point(out_path: str) -> int:
    """Rows already committed, after truncating any partial tail from a killed run."""
    if not os.path.exists(out_path):
        return 0
    with h5py.File(out_path, "r+") as out:
        done = int(out.attrs.get("n_cells_done", 0))
        if done == 0:
            return 0
        # Per-cell datasets are extended after the node datasets, so trusting n_cells_done
        # and truncating everything back to it removes any half-written block.
        n_nodes = out["n_nodes"][:done]
        total = int(out["offsets"][done])
        assert int(n_nodes.sum()) == total, "cache offsets inconsistent; delete and re-extract"
        for name in ("specimens", "source_row", "n_nodes", "pia_y", "wm_y"):
            out[name].resize((done,))
        out["soma_origin"].resize((done, 3))
        out["offsets"].resize((done + 1,))
        out["xyz"].resize((total, 3))
        for name in ("types", "parents"):
            out[name].resize((total,))
    print(f"resuming from row {done}")
    return done


def _read_block(src, shapes, specimens, lo: int, hi: int):
    """Read rows [lo, hi) in ascending order. Ascending order is the whole point -- see module docstring."""
    up_d, ty_d, pa_d = src["swc_upright/data"], src["swc_types/data"], src["swc_parents/data"]
    pia_d, wm_d = src["pia_upright/data"], src["wm_upright/data"]
    swc_s, pia_s, wm_s = shapes["swc_upright"], shapes["pia_upright"], shapes["wm_upright"]

    recs, skips = [], []
    for i in range(lo, hi):
        n = int(swc_s[i, 0])
        xyz = np.asarray(up_d[i], dtype=np.float64).reshape(n, 3)
        types = np.asarray(ty_d[i]).astype(np.int32).ravel()
        parents = np.asarray(pa_d[i]).astype(np.int64).ravel()

        reason = _check_invariants(types, parents, xyz)
        if reason is not None:
            skips.append((i, specimens[i], reason))
            continue

        origin = xyz[types == TYPE_SOMA].mean(axis=0)
        pia = np.asarray(pia_d[i], dtype=np.float64).reshape(tuple(int(s) for s in pia_s[i]))
        wm = np.asarray(wm_d[i], dtype=np.float64).reshape(tuple(int(s) for s in wm_s[i]))
        recs.append(dict(
            specimen_id=specimens[i], source_row=i, n_nodes=n,
            xyz=(xyz - origin).astype(np.float32), types=types.astype(np.uint8),
            parents=parents.astype(np.int32), soma_origin=origin.astype(np.float32),
            pia_y=_boundary_y(pia), wm_y=_boundary_y(wm),
        ))
    return recs, skips


def _extend(ds, data) -> None:
    k = len(data)
    if k == 0:
        return
    n = ds.shape[0]
    ds.resize((n + k,) + ds.shape[1:])
    ds[n:] = data


def _append(out: h5py.File, recs: list[dict]) -> None:
    if not recs:
        return
    _extend(out["xyz"], np.concatenate([r["xyz"] for r in recs]))
    _extend(out["types"], np.concatenate([r["types"] for r in recs]))
    _extend(out["parents"], np.concatenate([r["parents"] for r in recs]))

    n_nodes = np.array([r["n_nodes"] for r in recs], dtype=np.int64)
    off = out["offsets"]
    base = int(off[-1])
    _extend(off, base + np.cumsum(n_nodes))

    _extend(out["specimens"], np.array([r["specimen_id"] for r in recs], dtype="S8"))
    _extend(out["source_row"], np.array([r["source_row"] for r in recs], dtype=np.int64))
    _extend(out["n_nodes"], n_nodes.astype(np.int32))
    _extend(out["soma_origin"], np.stack([r["soma_origin"] for r in recs]))
    _extend(out["pia_y"], np.array([r["pia_y"] for r in recs], dtype=np.float32))
    _extend(out["wm_y"], np.array([r["wm_y"] for r in recs], dtype=np.float32))


def convert_labels(parquet_path: str = LABELS_PARQUET, out_csv: str | None = None) -> str:
    """Parquet -> CSV once, so no benchmark run needs pyarrow on an un-rebuilt image."""
    import pandas as pd

    out_csv = out_csv or os.path.join(CACHE_DIR, "coarse_labels.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.read_parquet(parquet_path)
    df["cell_id"] = df["cell_id"].astype(str)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(df)} rows, columns {list(df.columns)}")
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=SOURCE_PATH)
    p.add_argument("--out", default=os.path.join(CACHE_DIR, "em_exc_swc.h5"))
    p.add_argument("--block", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--labels", action="store_true", help="also convert the label parquet to CSV")
    args = p.parse_args()

    if args.labels:
        convert_labels()
    t0 = time.time()
    out = extract(args.source, args.out, block=args.block, limit=args.limit)
    print(f"cache at {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
