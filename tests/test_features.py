"""
Unit tests for src/features/ needlets, minkowski, and the combined feature vector.

All tests use small synthetic HEALPix maps drawn from a toy power spectrum via
healpy.synfast, so we exercise the same code path production uses without pulling
in CAMB. nside is small (32-64) to keep the suite fast.
"""
from __future__ import annotations

import numpy as np
import healpy as hp
import pytest

from src.features.needlets import needlet_coeffs, needlet_feature_dict, DEFAULT_SCALES_DEG
from src.features.minkowski import minkowski_functionals, minkowski_feature_dict
from src.features.feature_vector import (
    PatchFeatureConfig,
    extract_patch_features,
    feature_order,
)


def _gaussian_map(nside: int, seed: int = 0, white: bool = True) -> np.ndarray:
    """Toy CMB-like map. white=True gives pure white noise; else 1/l^2 red spectrum."""
    lmax = 3 * nside - 1
    rng = np.random.default_rng(seed)
    if white:
        return rng.normal(0.0, 1.0, size=hp.nside2npix(nside))
    ell = np.arange(lmax + 1)
    cl = np.zeros_like(ell, dtype=float)
    cl[2:] = 1.0 / (ell[2:] ** 2)
    return hp.synfast(cl, nside=nside, lmax=lmax, new=True)


def test_needlet_coeff_variance_grows_with_scale_on_red_field():
    """
    On a 1/l^2 red-tilted field, larger wavelet scales sit at lower ell where the
    input spectrum has more power — so the DoG bandpass carries larger variance
    at larger scales, at least while the wavelet still fits comfortably inside
    the patch. We stay well below the patch size (0.5° and 1.0° in a 5° patch,
    so sigma_pix maxes at ~26 in 128 pixels) and average over independent
    realizations so a single-draw fluctuation cannot flake the assertion.
    """
    nside = 64
    small_var, big_var = [], []
    for seed in range(6):
        tmap = _gaussian_map(nside, seed=seed, white=False)
        center = np.array(hp.pix2vec(nside, 12345 + seed * 111))
        coeffs = needlet_coeffs(tmap, nside, center, scales_deg=(0.5, 1.0))
        small_var.append(coeffs.coeff_var[0])
        big_var.append(coeffs.coeff_var[1])
    small_mean = float(np.nanmean(small_var))
    big_mean = float(np.nanmean(big_var))
    assert big_mean > small_mean, (
        f"red-field DoG variance not monotonic in scale: "
        f"0.5°={small_mean:.4g}, 1.0°={big_mean:.4g}"
    )


def test_needlet_feature_dict_schema_stable():
    nside = 32
    tmap = _gaussian_map(nside, seed=2)
    center = np.array(hp.pix2vec(nside, 0))
    c = needlet_coeffs(tmap, nside, center, scales_deg=DEFAULT_SCALES_DEG)
    feats = needlet_feature_dict(c)
    assert len(feats) == 3 * len(DEFAULT_SCALES_DEG)
    for k in feats:
        assert k.startswith("needlet_")


def test_minkowski_v0_matches_gaussian_expectation_at_zero_threshold():
    """
    For a mean-subtracted Gaussian field, the excursion set {T > 0} covers ~half
    the pixels — so V0 at threshold 0 should be near 0.5 within a few percent.
    """
    nside = 64
    tmap = _gaussian_map(nside, seed=3)
    center = np.array(hp.pix2vec(nside, 42))
    mf = minkowski_functionals(tmap, nside, center,
                                thresholds_sigma=(-1.0, 0.0, 1.0))
    zero_v0 = mf.v0_area[1]
    assert 0.4 < zero_v0 < 0.6, f"expected V0(0) near 0.5, got {zero_v0}"


def test_minkowski_v2_positive_for_bright_bump():
    """
    Injecting a bright localized bump on top of noise should give a positive
    Euler characteristic at the high threshold (an isolated bright component).
    """
    nside = 64
    tmap = _gaussian_map(nside, seed=4)
    center = np.array(hp.pix2vec(nside, 100))
    disc = hp.query_disc(nside, center, np.radians(0.3))
    tmap[disc] += 30.0

    mf = minkowski_functionals(tmap, nside, center,
                                thresholds_sigma=(0.0, 2.0),
                                patch_size_deg=4.0, n_pix=64)
    # At +2 sigma the bump should stand out as at least one isolated component.
    assert mf.v2_genus[1] >= 1
    assert mf.v0_area[1] > 0


def test_feature_vector_schema_is_deterministic():
    """
    feature_order returns the finite-only subset of extract_patch_features'
    keys in the same relative order (see feature_order's docstring — it filters
    out columns that are structurally NaN for this cfg).
    """
    cfg = PatchFeatureConfig(nside=32,
                              radii_deg=np.linspace(0, 2.0, 5),
                              patch_size_deg=4.0, patch_n_pix=64)
    tmap = _gaussian_map(cfg.nside, seed=5)
    center = np.array(hp.pix2vec(cfg.nside, 0))
    all_keys = list(extract_patch_features(tmap, center, cfg).keys())
    order = feature_order(cfg)
    assert set(order).issubset(set(all_keys))
    positions = [all_keys.index(n) for n in order]
    assert positions == sorted(positions), "feature_order did not preserve schema ordering"


def test_ring_variance_of_white_noise_is_scale_invariant():
    """
    Original Gurzadyan-Penrose-style sanity check: on pure white noise every
    ring's variance should be statistically equal, so the ring-variance stat
    (inner / mean(outer)) should sit near 1 within Poisson noise — but only
    once every ring is populated enough for its variance to be meaningful.

    At nside=128, HEALPix resolution is ~0.46°, so a ring narrower than that
    is a single pixel with zero variance and the ratio degenerates to 0. We
    use 4 rings x 0.5° here (~4 pixels per innermost ring) so the test
    exercises what it claims to test. Production configs run at nside>=512
    where 8 rings x 0.25° are already well-populated.
    """
    nside = 128
    rng = np.random.default_rng(7)
    tmap = rng.normal(0.0, 1.0, size=hp.nside2npix(nside))
    cfg = PatchFeatureConfig(nside=nside, radii_deg=np.linspace(0, 2.0, 5))
    stats = []
    for pix in range(0, 50):  # aggregate across patches — one draw is too noisy alone
        center = np.array(hp.pix2vec(nside, pix * 1000))
        feats = extract_patch_features(tmap, center, cfg)
        stats.append(feats["ring_variance_stat"])
    mean_stat = float(np.nanmean(stats))
    assert 0.3 < mean_stat < 3.0, f"ring-variance stat far from 1: {mean_stat}"
