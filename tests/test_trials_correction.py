"""
Calibration test: on a held-out ensemble of pure-Gaussian control simulations
(none of which were used to build the null CDF), the reported map-wide-max
p-values must be approximately uniform on (0, 1].

Uniformity is the standard calibration check: if the reported p-values aren't
uniform under the null, every downstream detection is either over- or under-
confident, and any Hawking-point claim built on it would be indefensible.

We avoid CAMB and torch here — the score function is just numpy on the feature
vector, and the null simulations use a synthetic 1/l^2 spectrum.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from src.simulations.gaussian_null import NullSimConfig, build_null_feature_table
from src.features.feature_vector import PatchFeatureConfig
from src.calibration.trials_correction import (
    build_null_distribution,
    calibrate_real_map,
)


def _toy_cl(lmax: int) -> np.ndarray:
    ell = np.arange(lmax + 1)
    cl = np.zeros_like(ell, dtype=float)
    cl[2:] = 1000.0 / (ell[2:] ** 2 + 1.0)
    return cl


def _row_scorer(row: dict) -> float:
    """
    A stand-in for the Phase-6 distilled statistic. We use log(1 + |kurtosis|)
    + a needlet-variance weight — arbitrary but stable, so the calibration test
    is really testing the null-CDF math, not the choice of statistic.
    """
    k = row.get("local_kurtosis", 0.0)
    v = row.get("needlet_s2_1p00deg_var", 0.0)
    if not np.isfinite(k) or not np.isfinite(v):
        return float("-inf")
    return float(np.log1p(abs(k)) + v)


def _make_rows(n_sims: int, n_patches: int, seed0: int, nside: int = 32):
    lmax = 2 * nside
    cl_in = _toy_cl(lmax)
    base = NullSimConfig(nside=nside, lmax=lmax, beam_fwhm_arcmin=0.0)
    fcfg = PatchFeatureConfig(nside=nside,
                                radii_deg=np.linspace(0, 2.0, 5),
                                patch_size_deg=4.0, patch_n_pix=48)
    return build_null_feature_table(n_sims=n_sims, n_patches_per_sim=n_patches,
                                     base_cfg=base, feat_cfg=fcfg,
                                     cl_tt=cl_in, seed0=seed0)


def test_null_pvalues_are_uniform_on_control_ensemble():
    """
    Two ensembles, disjoint seeds:
      1. calibration ensemble  → build the null CDF of map-wide max.
      2. control ensemble       → treat each sim as if it were the "real" map,
                                    ask for its p-value against the calibration
                                    CDF, and check the p-values are uniform.

    We use Kolmogorov-Smirnov against U(0,1). This is the load-bearing check on
    the whole calibration stage — a genuinely mis-calibrated pipeline will give
    ks_stat well above 0.3 even at n=80 sims. We use n=80 (not n=30) to make
    ks_stat under a truly-uniform null concentrate below ~0.15, and threshold
    at p > 0.001 to absorb residual small-n noise while still catching real
    breakage. Production calibration (Phase 7) uses N_sims>=1000 — this test
    is a fast dev-time smoke check that the machinery works, not a full
    calibration certification.
    """
    calib_rows = _make_rows(n_sims=80, n_patches=15, seed0=111)
    control_rows = _make_rows(n_sims=80, n_patches=15, seed0=999_111)

    null_calib = build_null_distribution(calib_rows, scorer=_row_scorer, top_k=3)

    from collections import defaultdict
    per_sim: "defaultdict[int, list[dict]]" = defaultdict(list)
    for r in control_rows:
        per_sim[r["sim_id"]].append(r)

    p_values = []
    for sim_id, rows in per_sim.items():
        report = calibrate_real_map(rows, null_calib, scorer=_row_scorer)
        p_values.append(report.p_value_max)
    p_values = np.array(p_values)

    ks_stat, ks_p = scipy_stats.kstest(p_values, "uniform")
    assert ks_p > 0.001, (
        f"p-values reject U(0,1) at KS p={ks_p:.4f}: min={p_values.min():.3f}, "
        f"max={p_values.max():.3f}, ks_stat={ks_stat:.3f}. This is the calibration "
        f"check — investigate before proceeding to real data."
    )


def test_pvalue_of_max_never_zero_with_shift():
    """The +1/(n+1) shift guarantees no reported p-value is exactly 0."""
    from src.calibration.trials_correction import NullMaxCalibration
    calib = NullMaxCalibration(
        max_scores=np.array([1.0, 2.0, 3.0, 4.0]),
        top_k_scores=np.zeros((4, 1)),
        top_k_reported=1,
        n_sims=4,
    )
    assert calib.pvalue_of_max(1000.0) > 0.0
    assert calib.pvalue_of_max(1000.0) == pytest.approx(1 / 5)


def test_calibrate_real_map_returns_topk_pvalues_in_bounds():
    calib_rows = _make_rows(n_sims=10, n_patches=8, seed0=222)
    fake_real_rows = _make_rows(n_sims=1, n_patches=20, seed0=333)
    # strip sim_id — real data doesn't have one
    for r in fake_real_rows:
        r.pop("sim_id", None)

    null_calib = build_null_distribution(calib_rows, scorer=_row_scorer, top_k=5)
    report = calibrate_real_map(fake_real_rows, null_calib, scorer=_row_scorer)
    assert report.top_k_pvalues.shape == (5,)
    assert np.all((report.top_k_pvalues > 0) & (report.top_k_pvalues <= 1))
