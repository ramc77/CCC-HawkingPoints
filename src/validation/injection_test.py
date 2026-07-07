"""
Phase 5: injection-based sensitivity check.

VALIDATION ONLY. This tells us what signal amplitude this pipeline can plausibly
detect, using the Hawking-point Gaussian radial template from An, Meissner, Nurowski
& Penrose (2018). It must never be used to train the flow density model (Phase 4) or
to build the calibration null ensemble (Phase 7) — see CLAUDE.md, Risks section. Doing
so would reintroduce exactly the template-dependence this project is designed to avoid.
"""
from typing import Sequence, Optional
import numpy as np
import healpy as hp

from src.simulations.gaussian_null import NullSimConfig, generate_null_map
from src.features.ring_stats import ring_profile, feature_vector


def hawking_point_template(nside: int, center_vec: np.ndarray,
                            amplitude_uk: float, sigma_deg: float,
                            radius_deg: float = 3.0) -> np.ndarray:
    """
    Gaussian radial temperature bump centered at center_vec:
        T(theta) = amplitude_uk * exp(-theta^2 / (2 * sigma_deg^2))
    following the AMNP-2018 Hawking point template. Returns a full-sky map with the
    bump added at pixels within radius_deg and zero elsewhere — add this to a null map.
    """
    template = np.zeros(hp.nside2npix(nside))
    disc_pix = hp.query_disc(nside, center_vec, np.radians(radius_deg))
    pix_vecs = np.array(hp.pix2vec(nside, disc_pix)).T
    cos_ang = np.clip(pix_vecs @ center_vec, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_ang))
    template[disc_pix] = amplitude_uk * np.exp(-theta_deg**2 / (2.0 * sigma_deg**2))
    return template


def sensitivity_sweep(amplitudes_uk: Sequence[float], sigma_deg: float = 1.0,
                       nside: int = 512, n_realizations: int = 20,
                       radii_deg: Optional[np.ndarray] = None,
                       seed0: int = 0) -> dict:
    """
    For each injection amplitude, inject a Hawking-point template at a random sky
    location into n_realizations independent null simulations, compute the
    ring-variance statistic, and return the distribution of scores per amplitude.

    A 0.0 uK entry in amplitudes_uk gives the no-injection control distribution —
    always include it so the sweep is read relative to the null, not in isolation.
    """
    if radii_deg is None:
        radii_deg = np.linspace(0, 2.0, 9)

    # theory C_ell is deterministic in CAMB — compute once and reuse across realizations
    cfg0 = NullSimConfig(nside=nside, seed=seed0)
    cl_tt = None  # computed lazily inside generate_null_map on first call if left None

    results = {}
    for amp in amplitudes_uk:
        scores = []
        for i in range(n_realizations):
            cfg = NullSimConfig(nside=nside, seed=seed0 + i)
            tmap = generate_null_map(cfg, cl_tt=cl_tt)

            rng = np.random.default_rng(seed0 + i)
            center_vec = rng.normal(size=3)
            center_vec /= np.linalg.norm(center_vec)

            if amp > 0:
                tmap = tmap + hawking_point_template(nside, center_vec, amp, sigma_deg)

            profile = ring_profile(tmap, nside, center_vec, radii_deg)
            feats = feature_vector(profile)
            scores.append(feats["ring_variance_stat"])
        results[amp] = np.array(scores)
    return results


def summarize_sweep(results: dict) -> None:
    """Print mean/std of the ring-variance statistic at each injected amplitude."""
    for amp, scores in results.items():
        valid = scores[~np.isnan(scores)]
        if valid.size == 0:
            print(f"amplitude={amp:6.1f} uK   no valid patches")
            continue
        print(f"amplitude={amp:6.1f} uK   mean_stat={valid.mean():.4f}   "
              f"std={valid.std():.4f}   n={valid.size}")


if __name__ == "__main__":
    # AMNP-2018 quoted amplitudes are roughly in the tens-of-uK range at ~1 deg scale —
    # this grid brackets that. Use small nside/n_realizations here for a quick local
    # check; production sensitivity curves (Phase 5 proper) should use nside matching
    # the real data (e.g. 2048) and n_realizations >= 100.
    amplitudes = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    out = sensitivity_sweep(amplitudes, n_realizations=20, nside=512)
    summarize_sweep(out)
