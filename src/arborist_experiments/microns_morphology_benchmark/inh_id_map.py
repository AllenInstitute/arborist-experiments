"""Resolve `data/MICrONS_inh_swc` root IDs to label nucleus IDs, and derive the upright frame.

Two problems have to be solved before the inhibitory cells can be benchmarked, and both are
solved here so the network and the excitatory cache are each touched exactly once.

**1. The ID gap.** SWC filenames are 18-digit segmentation root IDs; `microns_fine_labels`
`cell_id` values are 4-6 digit nucleus IDs. Direct overlap is zero.

CAVE cannot bridge them: every endpoint (`global.daf-apis.com/info/...`, materialization v3,
`minnie.microns-daf.com`) returns 302 to Google OAuth, anonymous `minnie65_public` access is
unavailable, and `caveclient` is not installed. A CAVE route would need a user-supplied token.

The route that works needs no new credential. `gs://allen-minnie-phase3/public_data/
cell_types_microns_1078_auto.feather` is a single 7.5 MB object (read with the `GCS_TOKEN`
service account, read-only) carrying `target_id` (nucleus), `pt_root_id` and `pt_position`.
Root ID is *not* a stable key across segmentation versions -- only 5797 of our 7893 filenames
match a `pt_root_id` -- but nucleus position is. The first node (`type == 1`) of each local SWC
*is* the nucleus centroid: measured over all 7893 cells, 7825 sit at distance **0.000 um** from
a manifest `pt_position` (p99 = 0.0), while the nearest *other* nucleus is at least 3.82 um
away. So the geometric join is both unambiguous and 35% higher-yield than filename matching,
and the 2028 cells whose root ID disagrees are recovered rather than silently mis-joined.

**2. The frame.** The inhibitory SWCs are in the global volume frame, not upright, so
`CORTICAL_AXIS = 1`, `R_EDGES` and `Z_EDGES` are meaningless on them until a rotation is known.

It is derived from the excitatory cache rather than assumed. `soma_origin[:,2]` regressed on
global soma position gives coefficients (8e-6, 2e-5, 0.999984) with 0.71 um rms residual --
i.e. **upright z is global z, unrotated**, so the transform is a pure rotation about z and there
is only one angle to find. Regressing the cached `pia_y` and `wm_y` on global (x, y) with 3-MAD
trimming gives theta = +4.5809 and +4.5758 degrees from two independent cortical boundaries,
agreeing to 0.005 degrees, with the |(a,b)| = 1 unit-norm constraint satisfied to 0.4% and 0.03%
without being imposed. Residual rms 17.1 / 12.4 um is pia curvature, not axis error.

The same two fits double as affine maps from global soma position to `pia_y` / `wm_y`, which is
what the `soma_depth` control needs and what validates the whole chain: applied to the
inhibitory somas, depth puts L1 interneurons shallowest (87.8 um) and orders the excitatory
contaminants monotonically L2IT 146 < L3IT 289 < L4IT 331 < L5IT 403 < L5ET 461 < L6CT 482 <
L6IT 484 -- the plan's acceptance test, on cells that had no part in the fit. Derived cortical
thickness is 695.1 +- 0.6 um.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


CACHE_DIR = "/root/capsule/scratch/microns_cache/v1"
SWC_DIR = "/root/capsule/data/MICrONS_inh_swc"
EXC_CACHE = os.path.join(CACHE_DIR, "em_exc_swc.h5")
MANIFEST_LOCAL = os.path.join(CACHE_DIR, "cell_types_microns_1078_auto.feather")
MAP_PATH = os.path.join(CACHE_DIR, "inh_id_map.csv")
UPRIGHT_PATH = os.path.join(CACHE_DIR, "inh_upright.json")

MANIFEST_BUCKET = "allen-minnie-phase3"
MANIFEST_BLOB = "public_data/cell_types_microns_1078_auto.feather"

# `pt_position` is nanometres; the SWCs are microns.
NM_PER_UM = 1000.0
# The nearest *other* nucleus to any matched soma is 3.82 um away, so anything in [0, 2] is
# unambiguous and the threshold is nowhere near a decision boundary.
ACCEPT_UM = 2.0
TRIM_MAD = 3.0
TRIM_ITERS = 12

__all__ = [
    "fetch_manifest", "read_soma_nodes", "geometric_join", "derive_upright",
    "build_map", "load_map", "load_upright", "upright_matrix",
]


def fetch_manifest(local_path: str = MANIFEST_LOCAL, log=print) -> str:
    """Download the nucleus manifest once, read-only, using the GCS_TOKEN service account."""
    if os.path.exists(local_path):
        log(f"manifest already cached: {local_path}")
        return local_path
    from google.cloud import storage
    from google.oauth2 import service_account

    raw = os.environ.get("GCS_TOKEN")
    if not raw:
        raise RuntimeError(
            "GCS_TOKEN is unset. data/gcs-token.json is revoked (invalid JWT signature) and "
            "must not be used; the manifest cannot be fetched without a working credential."
        )
    info = json.loads(raw)
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/devstorage.read_only"]
    )
    client = storage.Client(project=info["project_id"], credentials=creds)
    blob = client.bucket(MANIFEST_BUCKET).blob(MANIFEST_BLOB)
    data = blob.download_as_bytes()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
    log(f"fetched gs://{MANIFEST_BUCKET}/{MANIFEST_BLOB} -> {local_path} ({len(data)} bytes)")
    return local_path


def read_soma_nodes(swc_dir: str = SWC_DIR, log=print) -> tuple[np.ndarray, np.ndarray]:
    """Root IDs (from filenames) and their soma-node xyz in microns, filename-sorted."""
    import pandas as pd

    files = sorted(f for f in os.listdir(swc_dir) if f.endswith(".csv"))
    roots = np.array([int(os.path.splitext(f)[0]) for f in files], dtype=np.int64)
    soma = np.empty((len(files), 3), dtype=np.float64)
    for k, name in enumerate(files):
        # The soma is NOT always row 0 -- 30 of 7893 files start on a dendrite node, so the
        # whole type column has to be scanned rather than the first row sampled.
        df = pd.read_csv(os.path.join(swc_dir, name), usecols=["type", "x", "y", "z"])
        hit = np.flatnonzero(df["type"].to_numpy() == 1)
        if len(hit) > 1:
            raise ValueError(f"{name}: {len(hit)} soma nodes, expected at most 1")
        # 5 of 7893 files carry no soma node at all. The join keys on the soma being the nucleus
        # centroid, so those cannot be resolved and are left as NaN -> unresolved.
        soma[k] = df[["x", "y", "z"]].to_numpy()[hit[0]] if len(hit) else np.nan
    n_missing = int(np.isnan(soma[:, 0]).sum())
    log(f"read {len(files)} files from {swc_dir}; {n_missing} have no soma node")
    return roots, soma


def geometric_join(roots, soma, manifest_path: str = MANIFEST_LOCAL, log=print):
    """Nearest-nucleus join with an acceptance radius and a uniqueness check.

    Returns a DataFrame with one row per SWC: nucleus_id (-1 if unresolved), soma_dist_um,
    second_dist_um, root_agrees, match_method.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    mf = pd.read_feather(manifest_path)
    nuc = mf["target_id"].to_numpy()
    mroot = mf["pt_root_id"].to_numpy()
    pos = mf[["pt_position_x", "pt_position_y", "pt_position_z"]].to_numpy() / NM_PER_UM
    if len(np.unique(nuc)) != len(nuc):
        raise ValueError("manifest target_id is not unique; the join would be ambiguous")

    finite = np.isfinite(soma).all(axis=1)
    probe = np.where(finite[:, None], soma, 0.0)
    dist, idx = cKDTree(pos).query(probe, k=2)
    accept = finite & (dist[:, 0] <= ACCEPT_UM)
    # A hit is only trustworthy if the runner-up is clearly further away.
    ambiguous = accept & (dist[:, 1] <= ACCEPT_UM)
    if ambiguous.any():
        raise ValueError(f"{int(ambiguous.sum())} somas have two nuclei within {ACCEPT_UM} um")

    nucleus_id = np.where(accept, nuc[idx[:, 0]], -1)
    root_agrees = accept & (mroot[idx[:, 0]] == roots)
    out = pd.DataFrame({
        "root_id": roots,
        "nucleus_id": nucleus_id,
        "soma_dist_um": np.where(accept, dist[:, 0], np.nan),
        "second_dist_um": dist[:, 1],
        "root_agrees": root_agrees,
        "match_method": np.where(accept, "soma_centroid",
                                 np.where(finite, "no_nucleus_within_radius", "no_soma_node")),
        "soma_x": soma[:, 0], "soma_y": soma[:, 1], "soma_z": soma[:, 2],
    })
    log(f"geometric join: {int(accept.sum())}/{len(roots)} resolved within {ACCEPT_UM} um "
        f"(median {np.nanmedian(out['soma_dist_um']):.4f}, "
        f"p99 {np.nanpercentile(out['soma_dist_um'], 99):.4f}); "
        f"nearest rival >= {dist[accept, 1].min():.2f} um")
    log(f"  root_id would have matched {int(root_agrees.sum())} of them -- "
        f"{int(accept.sum() - root_agrees.sum())} cells recovered that filename matching loses")
    return out


def _robust_plane(xy: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Least squares `y ~ a*x0 + b*x1 + c` with iterative 3-MAD trimming.

    Trimming matters: `pia_y`/`wm_y` come from the nearest vertex of a per-cell boundary
    polyline, and ~1-2% of cells get a vertex from the wrong part of a curved boundary (68 have
    a negative pia_y). Those outliers pull |(a,b)| away from 1, which is the quantity used to
    confirm the model is a rigid rotation.
    """
    A = np.c_[xy, np.ones(len(xy))]
    keep = np.isfinite(y) & np.isfinite(A).all(axis=1)
    coef = np.zeros(3)
    for _ in range(TRIM_ITERS):
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        r = y - A @ coef
        med = np.median(r[keep])
        mad = np.median(np.abs(r[keep] - med)) * 1.4826
        keep = np.isfinite(r) & (np.abs(r - med) <= TRIM_MAD * max(mad, 1e-9))
    r = y - A @ coef
    return coef, {
        "n_kept": int(keep.sum()), "n_total": int(len(y)),
        "norm_ab": float(np.hypot(coef[0], coef[1])),
        "theta_deg": float(np.degrees(np.arctan2(coef[0], coef[1]))),
        "rms_resid_um": float(np.sqrt((r[keep] ** 2).mean())),
    }


def derive_upright(exc_cache: str = EXC_CACHE, manifest_path: str = MANIFEST_LOCAL, log=print):
    """Derive the global->upright rotation and the pia/wm depth affines from the exc cache."""
    import h5py
    import pandas as pd

    mf = pd.read_feather(manifest_path)
    gpos = {int(t): (x, y, z) for t, x, y, z in zip(
        mf["target_id"], mf["pt_position_x"], mf["pt_position_y"], mf["pt_position_z"])}

    with h5py.File(exc_cache, "r") as f:
        spec = [s.decode() for s in f["specimens"][:]]
        soma_origin = f["soma_origin"][:]
        pia = f["pia_y"][:].astype(np.float64)
        wm = f["wm_y"][:].astype(np.float64)

    rows = [k for k, s in enumerate(spec) if s.strip().isdigit() and int(s) in gpos]
    g = np.array([gpos[int(spec[k])] for k in rows], dtype=np.float64) / NM_PER_UM
    log(f"upright derivation uses {len(rows)} of {len(spec)} excitatory cells")

    # First: is the transform a pure rotation about z? If upright z is global z then the third
    # coefficient is 1 and the other two are 0, and only one angle remains to be found.
    A3 = np.c_[g, np.ones(len(g))]
    cz, *_ = np.linalg.lstsq(A3, soma_origin[rows, 2].astype(np.float64), rcond=None)
    z_resid = float(np.sqrt(((soma_origin[rows, 2] - A3 @ cz) ** 2).mean()))
    log(f"  soma_origin[:,2] ~ global: coef={np.round(cz[:3], 6).tolist()} "
        f"|coef|={np.linalg.norm(cz[:3]):.6f} rms={z_resid:.3f} um  -> upright z IS global z")
    if abs(np.linalg.norm(cz[:3]) - 1.0) > 0.01 or abs(cz[2] - 1.0) > 0.01:
        raise ValueError("upright z is not global z; the pure-z-rotation assumption fails")

    cp, sp = _robust_plane(g[:, :2], pia[rows])
    cw, sw = _robust_plane(g[:, :2], wm[rows])
    for name, s in (("pia_y", sp), ("wm_y ", sw)):
        log(f"  {name}: kept {s['n_kept']}/{s['n_total']}  |a,b|={s['norm_ab']:.5f}  "
            f"theta={s['theta_deg']:+.4f} deg  rms={s['rms_resid_um']:.2f} um")
    dtheta = abs(sp["theta_deg"] - sw["theta_deg"])
    log(f"  two independent boundaries agree on theta to {dtheta:.4f} deg")
    if dtheta > 0.5:
        raise ValueError(f"pia and wm disagree on the cortical axis by {dtheta:.3f} deg")

    # wm_y is the better-conditioned fit (|a,b| = 1.00027 vs 0.99646), so it sets theta; the
    # difference is 0.005 deg either way.
    theta = float(np.radians(sw["theta_deg"]))
    return {
        "theta_deg": float(np.degrees(theta)),
        "theta_source": "wm_y",
        "pia_affine": cp.tolist(),
        "wm_affine": cw.tolist(),
        "pia_fit": sp, "wm_fit": sw,
        "z_is_global_z": {"coef": cz[:3].tolist(), "rms_resid_um": z_resid},
        "n_exc_used": len(rows),
    }


def upright_matrix(theta_deg: float) -> np.ndarray:
    """Rows are the upright basis vectors in global coordinates: `xyz_up = R @ xyz_global`.

    The downward cortical direction is d = (sin t, cos t, 0), so upright +y (toward pia, matching
    `swc_upright` where pia_upright y > 0) is -d, upright z is global z, and upright x closes a
    right-handed frame as y_up x z_up.
    """
    t = np.radians(theta_deg)
    s, c = np.sin(t), np.cos(t)
    y_up = np.array([-s, -c, 0.0])
    z_up = np.array([0.0, 0.0, 1.0])
    x_up = np.cross(y_up, z_up)
    R = np.stack([x_up, y_up, z_up])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
    return R


def build_map(swc_dir: str = SWC_DIR, exc_cache: str = EXC_CACHE,
              manifest: str = MANIFEST_LOCAL, map_path: str = MAP_PATH,
              upright_path: str = UPRIGHT_PATH, log=print) -> None:
    import pandas as pd

    fetch_manifest(manifest, log=log)
    roots, soma = read_soma_nodes(swc_dir, log=log)
    df = geometric_join(roots, soma, manifest, log=log)

    up = derive_upright(exc_cache, manifest, log=log)
    A = np.c_[soma[:, :2], np.ones(len(soma))]
    df["pia_y"] = A @ np.array(up["pia_affine"])
    df["wm_y"] = A @ np.array(up["wm_affine"])
    thick = df["pia_y"] - df["wm_y"]
    log(f"  derived cortical thickness at inh somas: {thick.mean():.1f} +- {thick.std():.1f} um")

    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    df.to_csv(map_path, index=False)
    with open(upright_path, "w") as f:
        json.dump(up, f, indent=2)
    unresolved = df[df["nucleus_id"] < 0]
    unresolved.to_csv(os.path.join(os.path.dirname(map_path), "inh_unresolved.csv"), index=False)
    log(f"wrote {map_path} ({len(df)} rows, {len(unresolved)} unresolved) and {upright_path}")


def load_map(map_path: str = MAP_PATH):
    import pandas as pd

    if not os.path.exists(map_path):
        raise RuntimeError(f"{map_path} missing; run `python -m ...inh_id_map` first")
    return pd.read_csv(map_path)


def load_upright(upright_path: str = UPRIGHT_PATH) -> dict:
    if not os.path.exists(upright_path):
        raise RuntimeError(f"{upright_path} missing; run `python -m ...inh_id_map` first")
    with open(upright_path) as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--swc_dir", default=SWC_DIR)
    p.add_argument("--exc_cache", default=EXC_CACHE)
    p.add_argument("--manifest", default=MANIFEST_LOCAL)
    p.add_argument("--out", default=MAP_PATH)
    p.add_argument("--upright_out", default=UPRIGHT_PATH)
    args = p.parse_args()
    build_map(args.swc_dir, args.exc_cache, args.manifest, args.out, args.upright_out)


if __name__ == "__main__":
    main()
