"""
Tests for src/models/flow_density.py.

We test the Gaussian baseline directly (no torch dependency) — this is enough
to verify the split_by_sim guard, the feature-matrix builder, and the required
sensitivity behaviour: injected Gaussian bumps on top of null patches must
receive systematically higher anomaly scores than untouched null patches.

The MAFDensity flow is exercised only if torch + normflows are importable, so
CI without the heavy stack still passes.
"""
from __future__ import annotations

import numpy as np
import healpy as hp
import pytest

from src.simulations.gaussian_null import NullSimConfig, build_null_feature_table
from src.features.feature_vector import PatchFeatureConfig, feature_order
from src.models.flow_density import (
    GaussianDensity,
    split_by_sim,
    rows_to_matrix,
)
from src.validation.injection_test import hawking_point_template


def _toy_cl(lmax: int) -> np.ndarray:
    ell = np.arange(lmax + 1)
    cl = np.zeros_like(ell, dtype=float)
    cl[2:] = 1000.0 / (ell[2:] ** 2 + 1.0)
    return cl


def test_split_by_sim_keeps_sims_whole():
    rows = [{"sim_id": i // 3, "patch_id": i, "x": float(i)} for i in range(30)]
    train, val = split_by_sim(rows, val_frac=0.3, seed=0)
    tr_sims = {r["sim_id"] for r in train}
    va_sims = {r["sim_id"] for r in val}
    assert tr_sims.isdisjoint(va_sims)
    assert tr_sims | va_sims == {r["sim_id"] for r in rows}


def _build_null_and_bumped(n_sims: int = 6, n_patches: int = 6, nside: int = 32,
                             amplitude: float = 60.0, sigma_deg: float = 1.0):
    lmax = 2 * nside
    cl_in = _toy_cl(lmax)
    base = NullSimConfig(nside=nside, lmax=lmax, beam_fwhm_arcmin=0.0, seed=7)
    fcfg = PatchFeatureConfig(nside=nside,
                                radii_deg=np.linspace(0, 2.0, 5),
                                patch_size_deg=4.0, patch_n_pix=48)

    null_rows = build_null_feature_table(
        n_sims=n_sims, n_patches_per_sim=n_patches,
        base_cfg=base, feat_cfg=fcfg, cl_tt=cl_in, seed0=101,
    )

    # Build a second sim ensemble, but with a bright Gaussian bump injected at
    # every patch center — this is the sensitivity probe from CLAUDE.md.
    from src.simulations.gaussian_null import generate_null_map
    from src.features.feature_vector import extract_patch_features

    rng = np.random.default_rng(999)
    bumped_rows: list[dict] = []
    for sim_id in range(n_sims):
        cfg = NullSimConfig(nside=nside, lmax=lmax,
                             beam_fwhm_arcmin=0.0, seed=500 + sim_id)
        tmap = generate_null_map(cfg, cl_tt=cl_in)
        for j in range(n_patches):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            tmap_bumped = tmap + hawking_point_template(nside, v, amplitude, sigma_deg)
            feats = extract_patch_features(tmap_bumped, v, fcfg)
            feats["sim_id"] = sim_id
            feats["patch_id"] = j
            bumped_rows.append(feats)

    return null_rows, bumped_rows, fcfg


def test_gaussian_baseline_scores_injections_higher_than_null():
    """
    Sanity sensitivity: patches with a strong injected Hawking-point-like
    Gaussian bump should score higher (more anomalous) than clean null patches,
    even under a whitened multivariate Gaussian baseline. If this fails, the
    feature vector isn't picking up the injected signal at all.
    """
    null_rows, bumped_rows, fcfg = _build_null_and_bumped()
    train_rows, _val_rows = split_by_sim(null_rows, val_frac=0.25, seed=0)

    fnames = feature_order(fcfg)
    Xtr = rows_to_matrix(train_rows, fnames)
    Xnull = rows_to_matrix(null_rows, fnames)
    Xinj = rows_to_matrix(bumped_rows, fnames)

    model = GaussianDensity().fit(Xtr)
    s_null = model.anomaly_score(Xnull)
    s_inj = model.anomaly_score(Xinj)

    assert np.median(s_inj) > np.median(s_null), (
        f"injections not scored higher than null: "
        f"median inj={np.median(s_inj)}, median null={np.median(s_null)}"
    )


def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


@pytest.mark.skipif(not (_has("torch") and _has("normflows")),
                     reason="MAFDensity requires torch + normflows")
def test_maf_density_runs_end_to_end_and_beats_null_on_injections():
    from src.models.flow_density import MAFDensity
    null_rows, bumped_rows, fcfg = _build_null_and_bumped(n_sims=8, n_patches=8)
    train_rows, val_rows = split_by_sim(null_rows, val_frac=0.25, seed=0)
    fnames = feature_order(fcfg)
    Xtr = rows_to_matrix(train_rows, fnames)
    Xval = rows_to_matrix(val_rows, fnames)
    Xinj = rows_to_matrix(bumped_rows, fnames)

    model = MAFDensity(n_layers=3, hidden=32, n_epochs=30, batch=32)
    model.fit(Xtr, Xval)
    s_val = model.anomaly_score(Xval)
    s_inj = model.anomaly_score(Xinj)
    assert np.median(s_inj) > np.median(s_val)
