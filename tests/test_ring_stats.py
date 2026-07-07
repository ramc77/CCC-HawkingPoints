"""
Unit tests for src/features/ring_stats.py. Uses synthetic maps built directly (no CAMB
dependency) so these run fast and without the heavy simulation stack.
"""
import numpy as np
import healpy as hp
import pytest

from src.features.ring_stats import ring_profile, ring_variance_statistic, feature_vector


def test_ring_profile_uniform_map_has_zero_variance():
    """A perfectly flat map should give exactly zero variance in every ring."""
    nside = 64
    tmap = np.full(hp.nside2npix(nside), 5.0)
    center_vec = np.array(hp.pix2vec(nside, 0))
    radii_deg = np.linspace(0, 2.0, 5)

    profile = ring_profile(tmap, nside, center_vec, radii_deg)

    assert np.allclose(profile.ring_mean[~np.isnan(profile.ring_mean)], 5.0)
    assert np.allclose(profile.ring_var[~np.isnan(profile.ring_var)], 0.0, atol=1e-10)


def test_ring_variance_statistic_flags_quiet_inner_ring():
    """
    A map with an artificially quiet (low-variance) inner ring embedded in noisy
    outer rings should give a ring_variance_stat well below 1.
    """
    nside = 128
    rng = np.random.default_rng(0)
    tmap = rng.normal(0.0, 10.0, size=hp.nside2npix(nside))

    radii_deg = np.linspace(0, 2.0, 5)
    center_vec = np.array(hp.pix2vec(nside, 0))
    inner_pix = hp.query_disc(nside, center_vec, np.radians(radii_deg[1]))
    tmap[inner_pix] = 0.0  # flatten exactly the innermost ring to near-zero variance

    profile = ring_profile(tmap, nside, center_vec, radii_deg)
    stat = ring_variance_statistic(profile)

    assert stat < 0.1  # inner ring variance should be tiny relative to noisy outer rings


def test_feature_vector_keys_are_named_and_stable():
    """PySR distillation needs named features, not a bare array — check schema shape."""
    nside = 64
    tmap = np.zeros(hp.nside2npix(nside))
    center_vec = np.array(hp.pix2vec(nside, 0))
    radii_deg = np.linspace(0, 2.0, 5)

    profile = ring_profile(tmap, nside, center_vec, radii_deg)
    feats = feature_vector(profile)

    n_rings = len(radii_deg) - 1
    expected_keys = {f"ring{i}_mean" for i in range(n_rings)} | \
                    {f"ring{i}_var" for i in range(n_rings)} | \
                    {"ring_variance_stat"}
    assert set(feats.keys()) == expected_keys


def test_ring_profile_raises_on_empty_disc():
    nside = 64
    tmap = np.zeros(hp.nside2npix(nside))
    center_vec = np.array(hp.pix2vec(nside, 0))
    with pytest.raises(ValueError):
        ring_profile(tmap, nside, center_vec, radii_deg=np.array([0.0]))
