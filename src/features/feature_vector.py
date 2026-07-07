"""
Phase 2: combined per-patch feature vector.

Assembles a single stable, named schema from three physically-motivated feature
families:

  ring stats     : Gurzadyan-Penrose-generalized radial mean/variance profile
  needlets       : per-scale wavelet-response mean/var/|peak|
  minkowski      : V0/V1/V2 of local excursion sets at a few threshold levels
  local moments  : local kurtosis + skewness of the patch (a cheap outlier probe
                   that turns out to be exactly what catches the Bodnia-2024
                   single-pixel failure mode)

The schema is deliberately named — PySR (Phase 6) regresses onto named quantities,
never onto raw pixels. `FEATURE_ORDER` fixes column order so the parquet caches
from Phase 3 are safe to join across runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
from scipy import stats

from src.features.ring_stats import ring_profile, feature_vector as ring_feature_dict
from src.features.needlets import (
    DEFAULT_SCALES_DEG,
    needlet_coeffs,
    needlet_feature_dict,
    gnomonic_patch,
)
from src.features.minkowski import (
    DEFAULT_THRESHOLDS_SIGMA,
    minkowski_functionals,
    minkowski_feature_dict,
)


@dataclass
class PatchFeatureConfig:
    nside: int = 512
    radii_deg: np.ndarray = field(default_factory=lambda: np.linspace(0, 2.0, 9))
    needlet_scales_deg: tuple[float, ...] = DEFAULT_SCALES_DEG
    minkowski_thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS_SIGMA
    patch_size_deg: float = 5.0
    patch_n_pix: int = 128


def _local_moments(tmap: np.ndarray, cfg: PatchFeatureConfig,
                    center_vec: np.ndarray) -> dict:
    patch = gnomonic_patch(tmap, cfg.nside, center_vec,
                            patch_size_deg=cfg.patch_size_deg,
                            n_pix=cfg.patch_n_pix)
    vals = patch[np.isfinite(patch)]
    if vals.size < 2:
        return {"local_kurtosis": float("nan"), "local_skewness": float("nan")}
    return {
        "local_kurtosis": float(stats.kurtosis(vals, fisher=True, bias=False)),
        "local_skewness": float(stats.skew(vals, bias=False)),
    }


def extract_patch_features(tmap: np.ndarray, center_vec: np.ndarray,
                            cfg: Optional[PatchFeatureConfig] = None) -> dict:
    """
    Compute the full named feature dict for one (map, center) pair. Callers
    supply center_vec as a unit 3-vector (typically hp.pix2vec for a chosen
    HEALPix pixel).
    """
    cfg = cfg or PatchFeatureConfig()

    profile = ring_profile(tmap, cfg.nside, center_vec, cfg.radii_deg)
    ring_feats = ring_feature_dict(profile)

    needlets = needlet_coeffs(tmap, cfg.nside, center_vec,
                               scales_deg=cfg.needlet_scales_deg,
                               patch_size_deg=cfg.patch_size_deg,
                               n_pix=cfg.patch_n_pix)
    needlet_feats = needlet_feature_dict(needlets)

    mf = minkowski_functionals(tmap, cfg.nside, center_vec,
                                thresholds_sigma=cfg.minkowski_thresholds,
                                patch_size_deg=cfg.patch_size_deg,
                                n_pix=cfg.patch_n_pix)
    mf_feats = minkowski_feature_dict(mf)

    moments = _local_moments(tmap, cfg, center_vec)

    out: dict[str, float] = {}
    out.update(ring_feats)
    out.update(needlet_feats)
    out.update(mf_feats)
    out.update(moments)
    return out


def feature_order(cfg: Optional[PatchFeatureConfig] = None) -> list[str]:
    """
    Deterministic column order for the feature vector, so parquet caches remain
    self-consistent across sessions. Callers should always project their dicts
    through this order before stacking to arrays.

    We probe with a fixed random draw rather than a zero map so ring_variance_stat,
    kurtosis, and skewness (all NaN on constant input) come back finite. Features
    that are STRUCTURALLY NaN for this cfg (e.g. needlet scales that saturate on
    the patch — see the sigma_pix > n_pix/3 guard in needlets.py) are dropped
    here so `rows_to_matrix` downstream doesn't lose every row to a permanently-
    NaN column.
    """
    cfg = cfg or PatchFeatureConfig()
    import healpy as hp
    rng = np.random.default_rng(0)
    dummy = rng.normal(0.0, 1.0, size=hp.nside2npix(cfg.nside))
    center = np.array(hp.pix2vec(cfg.nside, 0))
    feats = extract_patch_features(dummy, center, cfg)
    return [k for k, v in feats.items() if np.isfinite(v)]
