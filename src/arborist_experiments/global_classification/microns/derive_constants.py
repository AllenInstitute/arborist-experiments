"""Derive the binning constants for this cohort from the cache, and report them.

This script never edits code. It prints the percentile table; the numbers are then copied into
features.py / radial_arbor_features.py by hand with a comment naming this derivation, which is
how the Patch-seq constants are documented and what makes them auditable.

Rule, identical to the Patch-seq one: pooled-node p0.5-p99.5 per axis, rounded outward to the
next bin_size multiple. Reads the cache, never the source file.
"""

from __future__ import annotations

import argparse

import numpy as np

from .data_loading import CACHE_PATH, load_cache


BIN_SIZE = 20.0


def _round_out(lo: float, hi: float, bin_size: float = BIN_SIZE) -> tuple[float, float]:
    return (float(np.floor(lo / bin_size) * bin_size), float(np.ceil(hi / bin_size) * bin_size))


def derive(cache_path: str = CACHE_PATH, max_cells: int | None = None,
           struct_types: tuple[int, ...] = (1, 3, 4)) -> dict:
    cache = load_cache(cache_path)
    n_cells = len(cache.specimens) if max_cells is None else min(max_cells, len(cache.specimens))
    end = int(cache.offsets[n_cells])
    xyz = cache.xyz[:end].astype(np.float64)
    types = cache.types[:end]

    # Dendrite is the only modality scored on the excitatory cohort, so the envelope is derived
    # on soma+basal+apical by default. The inhibitory cohort also scores axon and full, whose
    # envelopes are much larger, so struct_types is selectable there.
    keep = np.isin(types, struct_types)
    xyz = xyz[keep]

    out: dict = {"n_cells": n_cells, "n_nodes": int(len(xyz)),
                 "struct_types": list(struct_types)}
    for ax, name in enumerate("xyz"):
        lo, hi = np.percentile(xyz[:, ax], [0.5, 99.5])
        r_lo, r_hi = _round_out(lo, hi)
        out[name] = dict(p0_5=float(lo), p99_5=float(hi), r_min=r_lo, r_max=r_hi,
                         n_bins=int(round((r_hi - r_lo) / BIN_SIZE)))

    inplane = np.linalg.norm(xyz[:, [0, 2]], axis=1)
    out["inplane_r_p99_5"] = float(np.percentile(inplane, 99.5))

    r3 = np.linalg.norm(xyz, axis=1)
    out["node_r3_p99_5"] = float(np.percentile(r3, 99.5))

    # Per-cell max radius: how much of a fixed-length Sholl profile is identically zero.
    per_cell_max = []
    for i in range(n_cells):
        lo_i, hi_i = int(cache.offsets[i]), int(cache.offsets[i + 1])
        t = cache.types[lo_i:hi_i]
        m = np.isin(t, (1, 3, 4))
        if m.any():
            per_cell_max.append(float(np.linalg.norm(cache.xyz[lo_i:hi_i][m], axis=1).max()))
    pc = np.array(per_cell_max)
    out["per_cell_max_r_pct"] = {p: float(np.percentile(pc, p)) for p in (50, 90, 99, 100)}
    return out


def report(d: dict) -> str:
    L = [f"Derived on {d['n_cells']} cells / {d['n_nodes']} soma+dendrite nodes.",
         f"Rule: pooled-node p0.5-p99.5 per axis, rounded out to the next {BIN_SIZE:g} um.", ""]
    for name in "xyz":
        a = d[name]
        L.append(f"  {name}: p0.5={a['p0_5']:8.1f}  p99.5={a['p99_5']:8.1f}"
                 f"  ->  r_min_{name}={a['r_min']:7.1f} r_max_{name}={a['r_max']:7.1f}"
                 f"  ({a['n_bins']} bins)")
    nx, ny, nz = (d[k]["n_bins"] for k in "xyz")
    L += ["",
          f"  density feature counts: xy={nx * ny}  xz={nx * nz}  yz={ny * nz}",
          "",
          f"  in-plane radius p99.5 = {d['inplane_r_p99_5']:.1f}  -> R_EDGES max",
          f"  3D node radius p99.5   = {d['node_r3_p99_5']:.1f}",
          "  per-cell max radius p50/p90/p99/max = "
          + " / ".join(f"{d['per_cell_max_r_pct'][p]:.0f}" for p in (50, 90, 99, 100)),
          ]
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default=CACHE_PATH)
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    text = report(derive(args.cache, args.max_cells))
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
