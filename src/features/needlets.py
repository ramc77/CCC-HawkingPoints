"""
Phase 2: needlet / spherical-wavelet coefficients on flat sky patches.

We use a flat-sky Mexican-hat-wavelet approximation across ~5 dyadic scales.
This is the MVP-appropriate simplification of a full spherical needlet transform
(pys2let / s2fft is the natural upgrade path — flagged here and in CLAUDE.md).

Each patch is a gnomonic (tangent-plane) projection of a disc around a chosen
sky position. That lets us reuse cheap 2-D scale-space machinery from scipy
without leaving the physically-motivated feature framing PySR later regresses on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import healpy as hp
from scipy.ndimage import gaussian_filter


# Default scales must all fit inside the default patch. See the saturation
# check in `needlet_coeffs` below: sigma_pix > n_pix/3 nulls the scale. With
# the default 5° × 128 patch that caps usable scales at ~1.6°. Widen the patch
# (PatchFeatureConfig.patch_size_deg / patch_n_pix) if you want larger scales.
DEFAULT_SCALES_DEG: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)


@dataclass
class NeedletCoeffs:
    center_vec: np.ndarray
    scales_deg: tuple[float, ...]
    coeff_mean: np.ndarray   # per-scale mean of the wavelet response over the patch
    coeff_var: np.ndarray    # per-scale variance
    coeff_absmax: np.ndarray # per-scale peak |response| — sensitive to localized bumps


def gnomonic_patch(tmap: np.ndarray, nside: int, center_vec: np.ndarray,
                    patch_size_deg: float = 5.0,
                    n_pix: int = 128) -> np.ndarray:
    """
    Gnomonic (tangent-plane) projection of tmap onto an (n_pix, n_pix) grid
    centered on center_vec. patch_size_deg is the full width, so pixel scale is
    patch_size_deg / n_pix.

    We use healpy's projector at fixed rotation angles that make the tangent
    plane axis-aligned to the local (theta, phi) basis. That is good enough for
    scale-space features — we do not need exact Cartesian coordinates, only a
    smoothly-sampled small-angle image.
    """
    lon, lat = hp.vec2ang(np.asarray(center_vec), lonlat=True)  # deg
    lon = float(np.atleast_1d(lon)[0])
    lat = float(np.atleast_1d(lat)[0])
    xsize = int(n_pix)
    reso_arcmin = (patch_size_deg / xsize) * 60.0
    projector = hp.projector.GnomonicProj(rot=(lon, lat, 0.0),
                                          xsize=xsize,
                                          reso=reso_arcmin)
    vec2pix = lambda x, y, z: hp.vec2pix(nside, x, y, z)  # noqa: E731
    return projector.projmap(tmap, vec2pix)


def _mexican_hat(patch: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Difference-of-Gaussians approximation of the Mexican-hat wavelet."""
    a = gaussian_filter(patch, sigma_pix)
    b = gaussian_filter(patch, sigma_pix * 1.6)  # standard DoG ratio (Marr-Hildreth)
    return a - b


def needlet_coeffs(tmap: np.ndarray, nside: int, center_vec: np.ndarray,
                    scales_deg: Sequence[float] = DEFAULT_SCALES_DEG,
                    patch_size_deg: float = 5.0,
                    n_pix: int = 128) -> NeedletCoeffs:
    """
    Compute per-scale wavelet-response statistics (mean, var, |peak|) on a
    gnomonic patch around center_vec.

    scales_deg are the wavelet FWHM-ish scales; we convert to pixel-space sigma
    using the same reso as the projection. That keeps the wavelet scale tied to
    an angular scale, not to grid resolution — which is what matters when we
    compare features extracted at different nside.
    """
    patch = gnomonic_patch(tmap, nside, center_vec,
                            patch_size_deg=patch_size_deg, n_pix=n_pix)
    pix_deg = patch_size_deg / n_pix

    means, vars_, absmax = [], [], []
    for s_deg in scales_deg:
        sigma_pix = float(s_deg) / pix_deg
        # Skip cleanly with NaN — downstream code drops NaN rows before training
        # the flow, so the schema stays stable but bad scales don't poison it.
        # Lower bound: scale finer than the projection resolution.
        # Upper bound: sigma_pix >= n_pix / 4 means the Gaussian is wider than
        # ~half the patch — DoG saturates and the reported variance is dominated
        # by boundary averaging rather than field structure.
        if sigma_pix < 0.5 or sigma_pix > n_pix / 3:
            means.append(np.nan)
            vars_.append(np.nan)
            absmax.append(np.nan)
            continue
        w = _mexican_hat(patch, sigma_pix)
        means.append(float(np.nanmean(w)))
        vars_.append(float(np.nanvar(w)))
        absmax.append(float(np.nanmax(np.abs(w))))

    return NeedletCoeffs(
        center_vec=center_vec,
        scales_deg=tuple(scales_deg),
        coeff_mean=np.array(means),
        coeff_var=np.array(vars_),
        coeff_absmax=np.array(absmax),
    )


def needlet_feature_dict(coeffs: NeedletCoeffs) -> dict:
    """Named needlet features for one patch (see Phase 2 / PySR requirement)."""
    feats: dict[str, float] = {}
    for i, s in enumerate(coeffs.scales_deg):
        tag = f"needlet_s{i}_{s:.2f}deg".replace(".", "p")
        feats[f"{tag}_mean"] = float(coeffs.coeff_mean[i])
        feats[f"{tag}_var"] = float(coeffs.coeff_var[i])
        feats[f"{tag}_absmax"] = float(coeffs.coeff_absmax[i])
    return feats
