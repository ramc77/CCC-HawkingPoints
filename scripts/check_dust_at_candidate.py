#!/usr/bin/env python
"""
scripts/check_dust_at_candidate.py

Compare Planck 353 GHz thermal-dust brightness in a candidate disc against
several control discs at similar galactic latitude. If the candidate's mean
brightness is markedly elevated over controls, the candidate direction is
dust-contaminated and any CMB-map anomaly there is a foreground residual.

Usage:
    python -m scripts.check_dust_at_candidate \\
        --map353 data/raw/HFI_SkyMap_353_2048_R3.01_full.fits \\
        --gal-l 133.2 --gal-b -24.0 \\
        --radius-deg 5.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import healpy as hp


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--map353", type=Path, required=True,
                    help="Planck 353 GHz single-frequency FITS map")
    p.add_argument("--gal-l", type=float, required=True)
    p.add_argument("--gal-b", type=float, required=True)
    p.add_argument("--radius-deg", type=float, default=5.0)
    p.add_argument("--n-controls", type=int, default=8,
                    help="Number of control discs at similar |b|")
    p.add_argument("--seed", type=int, default=20260707)
    return p.parse_args(argv)


def _disc_stats(tmap, nside, vec, radius_deg):
    disc = hp.query_disc(nside, vec, np.radians(radius_deg))
    vals = tmap[disc]
    finite = vals[np.isfinite(vals)]
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "abs_mean": float(np.mean(np.abs(finite))),
    }


def main(argv=None):
    args = parse_args(argv)
    print(f"[Dust] Loading {args.map353} ...")
    tmap = hp.read_map(args.map353)
    nside = hp.get_nside(tmap)
    print(f"[Dust] nside = {nside}")

    lon, lat = np.array([args.gal_l]), np.array([args.gal_b])
    cand_vec = hp.ang2vec(lon, lat, lonlat=True)[0]
    cand = _disc_stats(tmap, nside, cand_vec, args.radius_deg)
    print(f"[Dust] Candidate (l={args.gal_l:.2f}, b={args.gal_b:.2f}): "
          f"mean={cand['mean']:.3g}, std={cand['std']:.3g}, "
          f"|mean|={cand['abs_mean']:.3g}, n={cand['n']}")

    # Draw n_controls random positions at similar |b| but different l
    rng = np.random.default_rng(args.seed)
    target_b = args.gal_b
    controls = []
    tries = 0
    while len(controls) < args.n_controls and tries < 200:
        tries += 1
        cand_l = float(rng.uniform(0, 360))
        # Avoid near-candidate longitudes
        if abs(cand_l - args.gal_l) < 15 or abs(cand_l - args.gal_l - 360) < 15:
            continue
        vec = hp.ang2vec(cand_l, target_b, lonlat=True)
        stats = _disc_stats(tmap, nside, vec, args.radius_deg)
        if stats["n"] >= 100 and np.isfinite(stats["mean"]):
            controls.append({"l": cand_l, "b": target_b, **stats})

    control_means = np.array([c["abs_mean"] for c in controls])
    control_stds = np.array([c["std"] for c in controls])
    print(f"[Dust] {len(controls)} control discs at b = {target_b:.1f}°:")
    for c in controls:
        print(f"[Dust]   l={c['l']:7.2f}: mean={c['mean']:+.3g}, "
              f"|mean|={c['abs_mean']:.3g}, std={c['std']:.3g}")

    ctrl_absmean_med = float(np.median(control_means))
    ctrl_std_med = float(np.median(control_stds))
    ratio_absmean = cand["abs_mean"] / max(ctrl_absmean_med, 1e-30)
    ratio_std = cand["std"] / max(ctrl_std_med, 1e-30)
    print()
    print("[Dust] === VERDICT ===")
    print(f"[Dust]   candidate |mean| / median control |mean| = {ratio_absmean:.2f}")
    print(f"[Dust]   candidate std   / median control std     = {ratio_std:.2f}")
    print()
    if ratio_absmean > 2.0 or ratio_std > 2.0:
        print("[Dust] Candidate direction shows > 2x brighter dust than same-|b| "
              "controls. Consistent with a dust-foreground residual.")
    elif ratio_absmean > 1.3 or ratio_std > 1.3:
        print("[Dust] Candidate direction shows mildly elevated dust (~30-100%). "
              "Compatible with a small foreground contribution, but not diagnostic.")
    else:
        print("[Dust] Candidate direction shows dust brightness comparable to "
              "same-|b| controls. Foreground contamination is not obvious; the "
              "CMB-map anomaly cannot be attributed to a dust residual from "
              "this cross-check alone.")


if __name__ == "__main__":
    sys.exit(main())
