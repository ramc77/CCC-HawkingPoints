"""
Phase 3: Gaussian LambdaCDM null simulation generator.

These realizations define "normal" CMB for the flow density model (Phase 4) and the
calibration null ensemble (Phase 7). No Hawking-point template is ever injected here —
see injection_test.py (Phase 5) for that, which is validation-only and must never feed
back into training or calibration (CLAUDE.md, Risks section).

`build_null_feature_table` walks N_sims Gaussian realizations, samples patch centers
from a mask-aware pixel pool, extracts the Phase-2 named feature vector at each patch,
and writes the whole table to a parquet cache — the flow (Phase 4) and calibration
(Phase 7) both consume this cache, so this step must not be re-run casually.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import healpy as hp

from src.features.feature_vector import PatchFeatureConfig, extract_patch_features


@dataclass
class NullSimConfig:
    nside: int = 512
    lmax: int = 1500
    beam_fwhm_arcmin: float = 5.0        # placeholder — read the real value from the
                                          # Planck/WMAP manifest (Phase 1) before production runs
    noise_uk_arcmin: float = 0.0          # set > 0 to add white instrument noise
    seed: Optional[int] = None
    cosmology: dict = field(default_factory=dict)  # overrides for the CAMB params below


def get_cl_theory(cfg: NullSimConfig) -> np.ndarray:
    """
    Fiducial Planck-2018-like LambdaCDM TT power spectrum, in uK^2, for ell = 0..lmax.
    Requires the `camb` package (heavy compiled dependency — install in the hawkingml
    conda env, not expected to run in a lightweight sandbox).
    """
    import camb

    pars = camb.CAMBparams()
    cosmo = dict(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544)
    cosmo.update(cfg.cosmology)
    pars.set_cosmology(**cosmo)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
    pars.set_for_lmax(cfg.lmax, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
    cl_tt = powers["total"][:, 0]
    return cl_tt[: cfg.lmax + 1]


def generate_null_map(cfg: NullSimConfig, mask: Optional[np.ndarray] = None,
                       cl_tt: Optional[np.ndarray] = None) -> np.ndarray:
    """
    One Gaussian LambdaCDM realization: synfast from a theory C_ell, beam-convolved,
    optionally noised and masked.

    Pass cl_tt explicitly to skip the CAMB call (e.g. reuse one theory spectrum across
    many realizations — CAMB itself is deterministic, only synfast's random draw varies).
    """
    if cl_tt is None:
        cl_tt = get_cl_theory(cfg)

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    fwhm_rad = np.radians(cfg.beam_fwhm_arcmin / 60.0)
    tmap = hp.synfast(cl_tt, nside=cfg.nside, lmax=cfg.lmax, fwhm=fwhm_rad, new=True)

    if cfg.noise_uk_arcmin > 0:
        pix_area_arcmin2 = hp.nside2pixarea(cfg.nside, degrees=True) * 3600.0
        sigma_pix = cfg.noise_uk_arcmin / np.sqrt(pix_area_arcmin2)
        tmap = tmap + np.random.normal(0.0, sigma_pix, size=tmap.shape)

    if mask is not None:
        tmap = tmap * mask

    return tmap


# ---------------------------------------------------------------------------
# Ensemble driver — build the Phase-3 feature cache for the flow (Phase 4)
# and the trials-factor calibration (Phase 7).
# ---------------------------------------------------------------------------


def _sample_centers(nside: int, n_centers: int,
                     mask: Optional[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    Draw n_centers HEALPix pixel indices uniformly from the unmasked sky.
    Returns an (n_centers, 3) array of unit vectors — one row per patch center.
    """
    npix = hp.nside2npix(nside)
    if mask is not None:
        good = np.where(mask > 0)[0]
        if good.size == 0:
            raise ValueError("mask has zero unmasked pixels")
        pix = rng.choice(good, size=n_centers, replace=False if n_centers <= good.size else True)
    else:
        pix = rng.integers(0, npix, size=n_centers)
    vecs = np.array(hp.pix2vec(nside, pix)).T
    return vecs


def build_null_feature_table(n_sims: int,
                              n_patches_per_sim: int,
                              base_cfg: NullSimConfig,
                              feat_cfg: Optional[PatchFeatureConfig] = None,
                              mask: Optional[np.ndarray] = None,
                              cl_tt: Optional[np.ndarray] = None,
                              seed0: int = 20260705) -> "list[dict]":
    """
    Generate `n_sims` independent Gaussian LambdaCDM realizations, sample
    n_patches_per_sim mask-aware patches from each, and extract the Phase-2
    feature vector at every patch.

    Returns a list of row-dicts (one per patch). Every row carries the sim_id
    so downstream splits (train/val/test in Phase 4, held-out control in
    Phase 7) can partition BY SIMULATION rather than by patch — see the docstring
    warning about leaked correlated structure inside a single sim.
    """
    feat_cfg = feat_cfg or PatchFeatureConfig(nside=base_cfg.nside)
    if cl_tt is None:
        cl_tt = get_cl_theory(base_cfg)

    rng = np.random.default_rng(seed0)
    rows: list[dict] = []
    for sim_id in range(n_sims):
        cfg = NullSimConfig(nside=base_cfg.nside,
                             lmax=base_cfg.lmax,
                             beam_fwhm_arcmin=base_cfg.beam_fwhm_arcmin,
                             noise_uk_arcmin=base_cfg.noise_uk_arcmin,
                             seed=seed0 + sim_id + 1,
                             cosmology=base_cfg.cosmology)
        tmap = generate_null_map(cfg, mask=mask, cl_tt=cl_tt)
        centers = _sample_centers(base_cfg.nside, n_patches_per_sim, mask, rng)
        for j, center in enumerate(centers):
            feats = extract_patch_features(tmap, center, feat_cfg)
            feats["sim_id"] = int(sim_id)
            feats["patch_id"] = int(j)
            feats["center_x"] = float(center[0])
            feats["center_y"] = float(center[1])
            feats["center_z"] = float(center[2])
            rows.append(feats)
    return rows


def write_feature_parquet(rows: Iterable[dict], out_path: Path | str) -> Path:
    """
    Write a rows-of-dicts feature table to parquet. We defer the import so the
    core simulation code works in environments without pyarrow — parquet is
    only mandatory for the Phase 4 training input.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        df = pd.DataFrame(list(rows))
        df.to_parquet(out_path, index=False)
    except ImportError as e:  # pragma: no cover — CI installs pandas/pyarrow
        raise RuntimeError(
            "pandas + pyarrow (or fastparquet) required to write feature parquet; "
            "install the hawkingml env from environment.yml"
        ) from e
    return out_path
