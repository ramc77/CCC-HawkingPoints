#!/usr/bin/env python
"""
scripts/inspect_candidate.py

Plot a gnomonic projection of a real CMB map around a candidate position.
Use this to eyeball the actual patch the pipeline scored — much more direct
than external tools, since we already have the FITS on disk.

Usage:
    python -m scripts.inspect_candidate \\
        --map data/raw/COM_CMB_IQU-smica_2048_R3.00_full.fits \\
        --gal-l 289.16 --gal-b 66.44 \\
        --patch-size-deg 10 \\
        --out results/figures/inspect_l289_b66.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--map", type=Path, required=True)
    p.add_argument("--gal-l", type=float, required=True,
                    help="Galactic longitude (deg) of the candidate center")
    p.add_argument("--gal-b", type=float, required=True,
                    help="Galactic latitude (deg) of the candidate center")
    p.add_argument("--patch-size-deg", type=float, default=10.0,
                    help="Full width of the plotted patch")
    p.add_argument("--target-nside", type=int, default=512,
                    help="Downgrade to this nside for plotting (matches pipeline)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"Loading {args.map} ...")
    tmap = hp.read_map(args.map)
    nside = hp.get_nside(tmap)
    if nside != args.target_nside:
        print(f"Degrading nside {nside} → {args.target_nside}")
        tmap = hp.ud_grade(tmap, args.target_nside)

    # Auto-detect K vs µK using the same heuristic as run_real_data.py
    med_abs = float(np.median(np.abs(tmap[np.isfinite(tmap)])))
    if med_abs < 1e-3:
        tmap = tmap * 1e6
        unit = "µK"
    else:
        unit = "µK"  # assume already
    print(f"Plotting patch at (l, b) = ({args.gal_l:.2f}°, {args.gal_b:.2f}°), "
          f"width {args.patch_size_deg}°, unit {unit}")

    # Gnomonic projection: healpy's gnomview centers on (l, b) via rot=(l, b, 0)
    # in galactic frame. reso is arcmin per pixel.
    n_pix = 512
    reso_arcmin = args.patch_size_deg * 60.0 / n_pix
    fig = plt.figure(figsize=(7, 7))
    hp.gnomview(
        tmap, coord="G", rot=(args.gal_l, args.gal_b, 0.0),
        xsize=n_pix, reso=reso_arcmin,
        title=f"SMICA around (l={args.gal_l:.2f}°, b={args.gal_b:.2f}°)",
        unit=unit, cmap="RdBu_r",
        min=-500, max=500, fig=fig.number,
    )
    hp.graticule(dpar=1, dmer=1, coord="G", verbose=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {args.out}")

    # Also report a few basic stats on the patch pixels themselves
    lon, lat = np.array([args.gal_l]), np.array([args.gal_b])
    center_vec = hp.ang2vec(lon, lat, lonlat=True)[0]
    disc = hp.query_disc(args.target_nside, center_vec,
                          np.radians(args.patch_size_deg / 2))
    vals = tmap[disc]
    finite = vals[np.isfinite(vals)]
    if finite.size:
        print(f"Patch pixel stats (n={finite.size}): "
              f"min={finite.min():.2f} {unit}, "
              f"max={finite.max():.2f} {unit}, "
              f"std={finite.std():.2f} {unit}")


if __name__ == "__main__":
    sys.exit(main())
