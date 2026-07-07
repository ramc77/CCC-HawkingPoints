#!/usr/bin/env python
"""
scripts/build_fallback_mask.py

Build a rough galactic + high-variance mask while waiting for the official
Planck common confidence mask to be reachable. Not a substitute for the
official mask in the paper — use this only for the exploratory second pass
so we can distinguish "galactic foreground residual" from "actual candidate."

Two exclusion criteria, both physically motivated:

  1. Galactic latitude cut: |b| < GAL_CUT_DEG (default 20°) removes the
     regions where Planck's component separation is known to leave the
     largest residuals.

  2. High-variance patches: compute the temperature variance in a moving
     disc, mask any pixel whose local variance exceeds a percentile
     (default 95th) of the whole-sky distribution. Removes localized bright
     residuals (point-source contamination, dust knots).

Both criteria are approximate. Reproduce your final results with the
official Planck common mask before writing anything up.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import healpy as hp

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--map", type=Path, required=True,
                    help="A Planck component-separated CMB map (used only to size the mask)")
    p.add_argument("--out", type=Path, required=True,
                    help="Output FITS mask path (0/1)")
    p.add_argument("--gal-cut-deg", type=float, default=20.0,
                    help="Excise |b| < this (degrees)")
    p.add_argument("--variance-percentile", type=float, default=95.0,
                    help="Excise pixels above this percentile of local variance")
    p.add_argument("--variance-disc-deg", type=float, default=1.0,
                    help="Radius of disc used for local-variance smoothing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"[Mask] Loading {args.map} ...")
    tmap = hp.read_map(args.map)
    nside = hp.get_nside(tmap)
    npix = tmap.size
    print(f"[Mask] nside={nside}, npix={npix}")

    # 1. Galactic-latitude cut. hp.pix2ang with lonlat=True returns (lon, lat)
    # in DEGREES — using that form directly so there's no colatitude/latitude
    # confusion. lat is exactly galactic b since this file is in galactic frame.
    _lon_deg, b_deg = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    gal_mask = np.abs(b_deg) > args.gal_cut_deg
    print(f"[Mask] Galactic |b|>{args.gal_cut_deg}° keeps {gal_mask.mean():.2%}")

    # 2. Local-variance cut. Use a rough proxy: smooth |T|² over the disc scale
    # and take the moving average. We downgrade to a lower nside for the
    # variance calc to keep it cheap.
    calc_nside = min(nside, 128)
    print(f"[Mask] Estimating local variance at nside={calc_nside} ...")
    t_lo = hp.ud_grade(tmap, calc_nside)
    var_lo = hp.smoothing(t_lo ** 2, fwhm=np.radians(args.variance_disc_deg),
                           verbose=False) - hp.smoothing(t_lo, fwhm=np.radians(args.variance_disc_deg),
                                                           verbose=False) ** 2
    var_hi = hp.ud_grade(var_lo, nside)
    thresh = float(np.percentile(var_hi[np.isfinite(var_hi)],
                                    args.variance_percentile))
    var_mask = var_hi < thresh
    print(f"[Mask] Local-var below p{args.variance_percentile:.0f} keeps "
          f"{var_mask.mean():.2%}")

    mask = (gal_mask & var_mask).astype(np.float64)
    kept = float(mask.sum() / mask.size)
    print(f"[Mask] Combined mask keeps {kept:.2%} of sky "
          f"(good for {int(kept * npix / 1e6)} M pixels)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(args.out, mask, overwrite=True)
    print(f"[Mask] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
