"""
Phase 2: ring-variance profile features.

Generalizes the classical Gurzadyan-Penrose ring-variance statistic (a single
inner-ring / outer-ring comparison) into a full multi-annulus feature vector, which
the flow density model (Phase 4) and PySR distillation (Phase 6) operate on.

Features are returned as a named dict, deliberately not a bare array — PySR must
regress onto physically interpretable quantities (see CLAUDE.md).
"""
from dataclasses import dataclass
import numpy as np
import healpy as hp


@dataclass
class RingProfile:
    center_vec: np.ndarray
    radii_deg: np.ndarray   # bin edges in degrees, increasing, e.g. linspace(0, 2.0, 9)
    ring_mean: np.ndarray
    ring_var: np.ndarray
    n_pix: np.ndarray


def ring_profile(tmap: np.ndarray, nside: int, center_vec: np.ndarray,
                  radii_deg: np.ndarray) -> RingProfile:
    """
    Mean and variance of temperature in concentric annuli around center_vec.

    center_vec: unit 3-vector (e.g. from hp.ang2vec or hp.pix2vec).
    radii_deg: increasing bin edges in degrees, e.g. np.linspace(0, 2.0, 9) for 8
        annuli out to 2 degrees opening angle (matches the AMNP-2018 Hawking point scale).
    """
    radii_rad = np.radians(np.asarray(radii_deg, dtype=float))
    max_radius = radii_rad[-1]

    disc_pix = hp.query_disc(nside, center_vec, max_radius)
    if disc_pix.size == 0:
        raise ValueError("Empty disc — check nside / center_vec / radius units")

    pix_vecs = np.array(hp.pix2vec(nside, disc_pix)).T
    cos_ang = np.clip(pix_vecs @ center_vec, -1.0, 1.0)
    ang = np.arccos(cos_ang)

    n_bins = len(radii_rad) - 1
    ring_mean = np.full(n_bins, np.nan)
    ring_var = np.full(n_bins, np.nan)
    n_pix = np.zeros(n_bins, dtype=int)

    vals = tmap[disc_pix]
    for i in range(n_bins):
        sel = (ang >= radii_rad[i]) & (ang < radii_rad[i + 1])
        n_pix[i] = int(sel.sum())
        if n_pix[i] > 0:
            ring_mean[i] = vals[sel].mean()
            ring_var[i] = vals[sel].var()

    return RingProfile(center_vec=center_vec, radii_deg=np.asarray(radii_deg),
                        ring_mean=ring_mean, ring_var=ring_var, n_pix=n_pix)


def ring_variance_statistic(profile: RingProfile) -> float:
    """
    Classical Gurzadyan-Penrose-style statistic: variance in the innermost ring
    relative to the mean variance of all outer annuli. Low values flag anomalously
    quiet circles — this is the quantity the PySR distillation (Phase 6) should be
    checked against as an internal validation baseline.
    """
    valid = ~np.isnan(profile.ring_var)
    if valid.sum() < 2:
        return np.nan
    var_valid = profile.ring_var[valid]
    inner = var_valid[0]
    outer_mean = np.nanmean(var_valid[1:])
    if outer_mean == 0:
        return np.nan
    return inner / outer_mean


def feature_vector(profile: RingProfile) -> dict:
    """Named feature dict for one patch — ring means/variances plus the summary stat."""
    feats = {}
    for i, (m, v) in enumerate(zip(profile.ring_mean, profile.ring_var)):
        feats[f"ring{i}_mean"] = float(m)
        feats[f"ring{i}_var"] = float(v)
    feats["ring_variance_stat"] = float(ring_variance_statistic(profile))
    return feats
