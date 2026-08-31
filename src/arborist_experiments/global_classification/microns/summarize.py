"""Assemble the cross-run comparison: does the Patch-seq ranking survive 31x more cells?

Reads `benchmark_metrics.csv` from several version directories and emits one table of
chance-corrected balanced accuracy per representation per run, plus the rank correlation
against the Patch-seq baseline.

Balanced accuracy is not comparable across cohorts with different class counts (13 Patch-seq,
8 MICrONS, 7 at n=1500 where L5NP falls below the floor), so the comparison is done on
`(BA - 1/K) / (1 - 1/K)` and the raw numbers are printed alongside it, never instead of it.
"""

from __future__ import annotations

import argparse
import csv
import os

# scratch/benchmarks/v2, the Patch-seq run this port is compared against (n=1556, 13 classes).
# precomputed_morphometrics (0.830) is listed for completeness but has no MICrONS counterpart.
PATCHSEQ_BA = {
    "precomputed_morphometrics": 0.830,
    "radial_arbors": 0.686,
    "morphometric_statistics": 0.560,
    "density_xy": 0.465,
    "density_yz": 0.430,
    "persistence_image": 0.414,
    "sholl": 0.380,
    "gaussian_arbor": 0.262,
    "density_xz": 0.165,
}
PATCHSEQ_BA_DENDRITE = {
    "radial_arbors": 0.622,
    "morphometric_statistics": 0.610,
    "density_xy": 0.540,
    "sholl": 0.490,
    "density_yz": 0.489,
    "persistence_image": 0.442,
    "gaussian_arbor": 0.307,
    "density_xz": 0.183,
}
PATCHSEQ_N_CLASSES = 13

ARCHIVE_DIR = "/root/capsule/scratch/microns_benchmarks"


def _corrected(ba: float, k: int) -> float:
    chance = 1.0 / k
    return (ba - chance) / (1.0 - chance)


def read_run(version_dir: str) -> dict:
    path = os.path.join(version_dir, "benchmark_metrics.csv")
    out: dict = {"dir": version_dir, "ba": {}, "n_cells": None, "n_classes": None}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["metric"] != "balanced_accuracy":
                continue
            out["ba"][r["feature_rep"]] = float(r["mean"])
            out["n_cells"] = int(r["n_cells"])
            out["n_classes"] = int(r["n_classes"])
    return out


def _spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def report(runs: list[dict], reference: str = "dendrite") -> str:
    ref_ba = PATCHSEQ_BA_DENDRITE if reference == "dendrite" else PATCHSEQ_BA
    reps = [r for r in sorted(ref_ba, key=lambda k: -ref_ba[k])]

    head = f"{'representation':26s} {'PS':>6s} {'PS_cc':>6s}"
    for run in runs:
        head += f" | {'n=' + str(run['n_cells']):>15s}"
    lines = [(f"Patch-seq reference column: {reference} modality, n=1556, "
             f"{PATCHSEQ_N_CLASSES} classes (chance {1 / PATCHSEQ_N_CLASSES:.3f})"),
             "PS_cc and the run columns are chance-corrected (BA - 1/K)/(1 - 1/K); raw BA in ().",
             "", head, "-" * len(head)]

    for rep in reps:
        row = f"{rep:26s} {ref_ba[rep]:6.3f} {_corrected(ref_ba[rep], PATCHSEQ_N_CLASSES):6.3f}"
        for run in runs:
            ba = run["ba"].get(rep)
            if ba is None:
                row += f" | {'-':>15s}"
            else:
                row += f" | {_corrected(ba, run['n_classes']):6.3f} ({ba:5.3f})"
        lines.append(row)

    lines.append("")
    for run in runs:
        shared = [r for r in reps if r in run["ba"]]
        rho = _spearman([ref_ba[r] for r in shared],
                        [run["ba"][r] for r in shared])
        lines.append(f"  n={run['n_cells']:<6d} K={run['n_classes']}  "
                     f"Spearman rho vs Patch-seq over {len(shared)} shared reps = {rho:+.3f}")

    depth = [(run["n_cells"], run["ba"].get("soma_depth")) for run in runs]
    if any(d[1] is not None for d in depth):
        lines += ["", "  soma_depth control (the number every row must beat, not chance):"]
        for n, ba in depth:
            if ba is not None:
                lines.append(f"    n={n:<6d} BA={ba:.4f}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("versions", nargs="+", help="version dirs, or bare names like v3")
    p.add_argument("--archive", default=ARCHIVE_DIR)
    p.add_argument("--reference", default="dendrite", choices=["dendrite", "full"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = []
    for v in args.versions:
        d = v if os.path.isdir(v) else os.path.join(args.archive, v)
        runs.append(read_run(d))
    runs.sort(key=lambda r: r["n_cells"])

    text = report(runs, args.reference)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
