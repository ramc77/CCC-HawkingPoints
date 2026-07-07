"""
Tests for src/simulations/gaussian_null.py.

CAMB is a heavy compiled dependency — we skip the real theory-spectrum call and
inject a synthetic 1/l^2 red-tilted C_ell so the tests run fast in any env with
healpy installed. The point of the recovery test is to confirm synfast + our
config wiring reproduce whatever theory C_ell you feed in, not to certify CAMB.
"""
from __future__ import annotations

import numpy as np
import healpy as hp
import pytest

from src.simulations.gaussian_null import (
    NullSimConfig,
    generate_null_map,
    build_null_feature_table,
    _sample_centers,
)
from src.features.feature_vector import PatchFeatureConfig


def _toy_cl(lmax: int) -> np.ndarray:
    ell = np.arange(lmax + 1)
    cl = np.zeros_like(ell, dtype=float)
    cl[2:] = 1000.0 / (ell[2:] ** 2 + 1.0)  # arbitrary uK^2 scale
    return cl


def test_generate_null_map_recovers_input_cl():
    """
    Average anafast-recovered C_ell over many null realizations should match the
    input theory C_ell within a few percent, across a broad ell band well below
    the Nyquist scale. This is the standard end-to-end check on Phase 3.
    """
    nside = 64
    lmax = 2 * nside
    cl_in = _toy_cl(lmax)

    n_sims = 40
    recovered = np.zeros_like(cl_in)
    for i in range(n_sims):
        cfg = NullSimConfig(nside=nside, lmax=lmax,
                             beam_fwhm_arcmin=0.0, seed=42 + i)
        tmap = generate_null_map(cfg, cl_tt=cl_in)
        cl_out = hp.anafast(tmap, lmax=lmax)
        recovered += cl_out
    recovered /= n_sims

    ell_band = slice(10, lmax - 10)
    ratio = recovered[ell_band] / cl_in[ell_band]
    med = float(np.median(ratio))
    assert 0.85 < med < 1.15, f"median recovered/input C_ell ratio off: {med}"


def test_sample_centers_respects_mask():
    nside = 32
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix)
    good = np.arange(0, npix, 4)   # keep every 4th pixel
    mask[good] = 1

    rng = np.random.default_rng(0)
    vecs = _sample_centers(nside, 50, mask, rng)
    pix = hp.vec2pix(nside, vecs[:, 0], vecs[:, 1], vecs[:, 2])
    assert set(pix.tolist()).issubset(set(good.tolist()))


def test_build_null_feature_table_shape_and_sim_ids():
    """
    Small end-to-end: 3 sims x 4 patches gives 12 rows, each carrying a stable
    named feature schema, sim_id, and center coordinates.
    """
    nside = 32
    lmax = 2 * nside
    cl_in = _toy_cl(lmax)

    base = NullSimConfig(nside=nside, lmax=lmax, beam_fwhm_arcmin=0.0, seed=1)
    fcfg = PatchFeatureConfig(nside=nside,
                                radii_deg=np.linspace(0, 2.0, 5),
                                patch_size_deg=4.0, patch_n_pix=48)

    rows = build_null_feature_table(n_sims=3, n_patches_per_sim=4,
                                     base_cfg=base, feat_cfg=fcfg,
                                     cl_tt=cl_in, seed0=123)
    assert len(rows) == 12
    sim_ids = sorted({r["sim_id"] for r in rows})
    assert sim_ids == [0, 1, 2]
    for r in rows:
        assert "ring_variance_stat" in r
        assert -1.0 <= r["center_x"] <= 1.0
