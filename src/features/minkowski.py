"""
Phase 2: Minkowski functionals of local excursion sets.

Given a gnomonic patch around a candidate center, at each threshold nu we look
at the excursion set {T > nu * sigma} and compute three morphological summaries:

  V0 = area fraction
  V1 = perimeter length (per unit area)
  V2 = Euler characteristic (genus) — connected components minus holes

For a purely Gaussian isotropic field these have known analytic forms in nu.
A localized Hawking-point bump distorts them in a specific way (extra positive
excursion, isolated component) — so tracking V0/V1/V2 at a few threshold levels
gives the density model a shape-sensitive but template-free channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy import ndimage

from src.features.needlets import gnomonic_patch


DEFAULT_THRESHOLDS_SIGMA: tuple[float, ...] = (-1.0, 0.0, 1.0)


@dataclass
class MinkowskiFunctionals:
    thresholds: tuple[float, ...]
    v0_area: np.ndarray       # area fraction of excursion set
    v1_perimeter: np.ndarray  # perimeter (in pixel units)
    v2_genus: np.ndarray      # Euler characteristic


def _perimeter_pixels(mask: np.ndarray) -> int:
    """
    4-connected boundary length: count pixel edges where the excursion mask
    flips value. Equivalent to counting pixels in mask XOR shift-by-one along
    each axis. Boundary of the patch itself is not counted (we do not pad).
    """
    if mask.size == 0:
        return 0
    horiz = int(np.sum(mask[:, :-1] != mask[:, 1:]))
    vert = int(np.sum(mask[:-1, :] != mask[1:, :]))
    return horiz + vert


def _euler_characteristic(mask: np.ndarray) -> int:
    """
    2-D Euler characteristic chi = C - H, where C is the number of connected
    components of the excursion set (foreground) and H is the number of holes
    (connected components of the background that are enclosed by foreground).

    ndimage.label uses 4-connectivity by default; we pair foreground 4-conn with
    background 8-conn to satisfy the standard adjacency convention (Kong-Rosenfeld).
    """
    if mask.size == 0:
        return 0
    _, n_components = ndimage.label(mask)
    bg = ~mask
    # Trim background components that touch the frame — those are "outside",
    # not holes. Only interior background components count as holes.
    bg_labeled, _ = ndimage.label(bg, structure=np.ones((3, 3), dtype=int))
    touches_frame = set()
    for edge in (bg_labeled[0, :], bg_labeled[-1, :],
                  bg_labeled[:, 0], bg_labeled[:, -1]):
        touches_frame.update(int(v) for v in edge if v != 0)
    all_bg = {int(v) for v in np.unique(bg_labeled) if v != 0}
    n_holes = len(all_bg - touches_frame)
    return int(n_components - n_holes)


def minkowski_functionals(tmap: np.ndarray, nside: int, center_vec: np.ndarray,
                          thresholds_sigma: Sequence[float] = DEFAULT_THRESHOLDS_SIGMA,
                          patch_size_deg: float = 5.0,
                          n_pix: int = 128) -> MinkowskiFunctionals:
    """
    Compute V0/V1/V2 for excursion sets of the patch at each nu * sigma,
    where sigma is the patch's own standard deviation. Using the patch's own
    sigma (rather than a global CMB sigma) makes the functionals dimensionless
    and comparable across patches at different galactic latitudes.
    """
    patch = gnomonic_patch(tmap, nside, center_vec,
                            patch_size_deg=patch_size_deg, n_pix=n_pix)
    good = np.isfinite(patch)
    if not good.any():
        n = len(thresholds_sigma)
        return MinkowskiFunctionals(tuple(thresholds_sigma),
                                     np.full(n, np.nan),
                                     np.full(n, np.nan),
                                     np.full(n, np.nan))

    # Robust sigma via MAD (1.4826 × median absolute deviation from the median).
    # Using np.std here would inflate sigma when the patch contains a bright
    # point source, biasing every downstream Minkowski excursion. On pure
    # Gaussian nulls MAD ≈ std, so calibration barely shifts; on real data with
    # unresolved sources it is dramatically more robust.
    good_vals = patch[good]
    center_val = float(np.median(good_vals))
    mad = float(np.median(np.abs(good_vals - center_val)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(good_vals)) or 1.0
    mean = center_val

    v0 = np.zeros(len(thresholds_sigma))
    v1 = np.zeros(len(thresholds_sigma))
    v2 = np.zeros(len(thresholds_sigma))
    area = float(good.sum())
    for i, nu in enumerate(thresholds_sigma):
        excursion = (patch - mean) > (nu * sigma)
        excursion &= good
        v0[i] = float(excursion.sum()) / area
        v1[i] = _perimeter_pixels(excursion) / area
        v2[i] = _euler_characteristic(excursion)

    return MinkowskiFunctionals(tuple(thresholds_sigma), v0, v1, v2)


def minkowski_feature_dict(mf: MinkowskiFunctionals) -> dict:
    feats: dict[str, float] = {}
    for i, nu in enumerate(mf.thresholds):
        tag = f"nu{nu:+.1f}sigma".replace(".", "p").replace("+", "p").replace("-", "m")
        feats[f"mink_v0_{tag}"] = float(mf.v0_area[i])
        feats[f"mink_v1_{tag}"] = float(mf.v1_perimeter[i])
        feats[f"mink_v2_{tag}"] = float(mf.v2_genus[i])
    return feats
