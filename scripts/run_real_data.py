#!/usr/bin/env python
"""
scripts/run_real_data.py

Score a real Planck or WMAP component-separated CMB map against the calibrated
null CDF built during the control experiment (Phases 3-7).

Design:
  We DO NOT recompute the null ensemble here — that already lives in the parquet
  cache written by scripts/run_control_experiment.py. This script:

    1. Loads results/tables/null_features.parquet (all rows tagged train/calib/control).
    2. Re-fits the Gaussian density on the train rows (seconds).
    3. Re-runs the ridge/PySR distillation on the flow scores (seconds).
    4. Re-builds the trials-corrected null CDF of the distilled statistic from
       the calib rows (seconds).
    5. Loads a real Planck/WMAP FITS map + mask.
    6. Extracts patches from the unmasked sky, scores each with the distilled
       statistic, reports top-K vs. the null CDF.
    7. Applies the Bodnia-2024 single-pixel-mask check to any candidate below
       the trials-corrected significance threshold.
    8. Writes results/tables/real_candidates.csv and results/REAL_RESULTS.md.

Usage:
  # Score Planck SMICA (assuming fetch_data.py has run):
  python -m scripts.run_real_data \\
      --map data/raw/COM_CMB_IQU-smica_2048_R3.00_full.fits \\
      --mask data/raw/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits \\
      --map-label planck_smica

  # Point to any local FITS file (works even if fetch_data URLs are wrong):
  python -m scripts.run_real_data --map ~/Downloads/my_map.fits --map-label smica_local

Cross-pipeline validation (Phase 8): re-run this script per component-separation
map and cross-match the candidate lists — a real Hawking-point-like signal must
appear consistently in every map at (roughly) the same sky location.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import healpy as hp

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.feature_vector import (  # noqa: E402
    PatchFeatureConfig,
    extract_patch_features,
    feature_order,
)
from src.models.flow_density import (  # noqa: E402
    GaussianDensity,
    rows_to_matrix,
    split_by_sim,
)
from src.distillation.pysr_distill import (  # noqa: E402
    build_evaluator,
    distill,
)
from src.calibration.trials_correction import (  # noqa: E402
    build_null_distribution,
    calibrate_real_map,
)
from src.validation.cross_pipeline import rescore_masking_extreme_pixel  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--parquet", type=Path,
                    default=Path("results/tables/null_features.parquet"),
                    help="Feature parquet from the control experiment")
    p.add_argument("--map", type=Path, required=True,
                    help="Real CMB temperature map (FITS)")
    p.add_argument("--mask", type=Path, default=None,
                    help="Optional confidence mask (FITS). Pixels with value==0 are ignored")
    p.add_argument("--map-label", type=str, required=True,
                    help="Short name for this map (e.g. planck_smica), used in outputs")
    p.add_argument("--nside", type=int, default=512,
                    help="Target nside — real map is degraded to this if higher")
    p.add_argument("--n-patches", type=int, default=3000,
                    help="Patches to sample from the unmasked sky")
    p.add_argument("--top-k", type=int, default=10,
                    help="Number of highest-scoring candidates to report")
    p.add_argument("--alpha", type=float, default=0.01,
                    help="Trials-corrected significance threshold for the single-pixel check")
    p.add_argument("--single-pixel-radius-deg", type=float, default=2.0,
                    help="Radius of disc used in the Bodnia-2024 single-pixel scrub. "
                         "2° covers compact point sources up to ~1° wide after ud_grade "
                         "and any beam smearing; drop to ~0.5° if too aggressive.")
    p.add_argument("--use-pysr", action="store_true",
                    help="Use PySR for distillation (else ridge fallback)")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Rebuild the calibrated pipeline from the parquet cache
# ---------------------------------------------------------------------------


def load_ensembles(parquet_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    if "ensemble" not in df.columns:
        raise RuntimeError(
            f"{parquet_path} has no 'ensemble' column — re-run the control "
            "experiment with the current run_control_experiment.py."
        )
    train = df[df["ensemble"] == "train"].to_dict("records")
    calib = df[df["ensemble"] == "calib"].to_dict("records")
    control = df[df["ensemble"] == "control"].to_dict("records")
    return train, calib, control


def rebuild_pipeline(train_rows: list[dict], calib_rows: list[dict],
                       fnames: list[str], use_pysr: bool = False):
    print(f"[Rebuild] Fitting GaussianDensity on {len(train_rows)} train rows ...")
    train_actual, _val = split_by_sim(train_rows, val_frac=0.2, seed=42)
    Xtr = rows_to_matrix(train_actual, fnames)
    model = GaussianDensity().fit(Xtr)

    print(f"[Rebuild] Distilling flow anomaly score onto {len(fnames)} features ...")
    X_dist = rows_to_matrix(train_actual, fnames)
    y_dist = model.anomaly_score(X_dist)
    distillation = distill(X_dist, y_dist, fnames,
                            use_pysr=use_pysr, niterations=100, complexity=20)
    scorer = build_evaluator(distillation)
    print(f"[Rebuild] Distilled expression R²={distillation.best_r2:.4f}, "
          f"resembles GP={distillation.resembles_classical}")

    print(f"[Rebuild] Building trials-corrected null CDF from "
          f"{len({r['sim_id'] for r in calib_rows})} calibration sims ...")
    null_calib = build_null_distribution(calib_rows, scorer=scorer, top_k=10)
    threshold = float(np.quantile(null_calib.max_scores, 1.0 - 0.01))
    print(f"[Rebuild] Null CDF built (n={null_calib.n_sims}). "
          f"α=0.01 threshold on distilled max: {threshold:.3f}")
    return model, distillation, scorer, null_calib


# ---------------------------------------------------------------------------
# Real map loading + patch sampling
# ---------------------------------------------------------------------------


def load_real_map(map_path: Path, mask_path: Path | None,
                    target_nside: int) -> tuple[np.ndarray, np.ndarray | None]:
    print(f"[RealMap] Loading {map_path} ...")
    # Planck component-separation maps: TQU triple. We take I (intensity, index 0).
    # WMAP ILC: single-column T map. healpy.read_map returns column 0 by default.
    tmap = hp.read_map(map_path, verbose=False)
    orig_nside = hp.get_nside(tmap)
    print(f"[RealMap] Original nside={orig_nside}, npix={tmap.size}")
    if orig_nside != target_nside:
        print(f"[RealMap] Degrading nside {orig_nside} → {target_nside}")
        tmap = hp.ud_grade(tmap, target_nside)

    # Planck component-separated maps are in K_CMB; our simulated null was in µK.
    # If the median absolute value looks like Kelvin (< 1e-3), assume it's K and
    # convert. Otherwise assume µK. This is a heuristic — verify against the
    # source's own header for a paper-grade run.
    med_abs = float(np.median(np.abs(tmap[np.isfinite(tmap)])))
    if med_abs < 1e-3:
        print(f"[RealMap] median|T| = {med_abs:.3e} → treating as K_CMB, converting to µK")
        tmap = tmap * 1e6
    else:
        print(f"[RealMap] median|T| = {med_abs:.3e} → treating as µK")

    mask = None
    if mask_path is not None:
        print(f"[RealMap] Loading mask {mask_path} ...")
        mask = hp.read_map(mask_path, verbose=False)
        if hp.get_nside(mask) != target_nside:
            mask = hp.ud_grade(mask, target_nside)
        # Planck common mask is 0/1; degrade may leave fractional values.
        mask = (mask > 0.5).astype(np.float64)
        f_kept = float(mask.sum() / mask.size)
        print(f"[RealMap] Mask: {f_kept:.2%} of sky kept")

    return tmap, mask


def sample_real_patch_centers(nside: int, mask: np.ndarray | None,
                                n_patches: int, seed: int = 20260706) -> np.ndarray:
    """Draw n_patches HEALPix pixel centers from the unmasked sky."""
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    if mask is not None:
        good = np.where(mask > 0)[0]
        if good.size < n_patches:
            print(f"[RealMap] Only {good.size} unmasked pixels; sampling with replacement")
            pix = rng.choice(good, size=n_patches, replace=True)
        else:
            pix = rng.choice(good, size=n_patches, replace=False)
    else:
        pix = rng.integers(0, npix, size=n_patches)
    vecs = np.array(hp.pix2vec(nside, pix)).T
    return vecs


# ---------------------------------------------------------------------------
# Score real map + report
# ---------------------------------------------------------------------------


def score_real_map_patches(tmap: np.ndarray, centers: np.ndarray,
                             feat_cfg: PatchFeatureConfig,
                             scorer, fnames: list[str]) -> list[dict]:
    rows: list[dict] = []
    n = centers.shape[0]
    for i, c in enumerate(centers):
        feats = extract_patch_features(tmap, c, feat_cfg)
        if not all(np.isfinite(feats.get(k, np.nan)) for k in fnames):
            continue
        row = dict(feats)
        row["center_x"] = float(c[0])
        row["center_y"] = float(c[1])
        row["center_z"] = float(c[2])
        row["score"] = float(scorer(row))
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"[Score] {i + 1}/{n} patches scored")
    return rows


def report(rows: list[dict], null_calib, args, tmap: np.ndarray,
             feat_cfg: PatchFeatureConfig, scorer) -> Path:
    from src.validation.cross_pipeline import rescore_masking_extreme_pixel

    scored = [r for r in rows if np.isfinite(r["score"])]
    scored.sort(key=lambda r: r["score"], reverse=True)
    top = scored[: args.top_k]

    threshold_alpha = float(np.quantile(null_calib.max_scores, 1.0 - args.alpha))
    n_null = null_calib.n_sims

    tables = args.output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    csv_path = tables / f"real_candidates_{args.map_label}.csv"

    with csv_path.open("w") as f:
        # Coordinates: (l, b) in galactic frame (Planck component-sep maps).
        f.write("rank,map_label,center_x,center_y,center_z,gal_l_deg,gal_b_deg,"
                "score,score_after_scrub,p_value,passes_single_pixel_check\n")
        candidate_records = []
        for k, cand in enumerate(top):
            c = np.array([cand["center_x"], cand["center_y"], cand["center_z"]])
            pval = float((np.sum(null_calib.max_scores >= cand["score"]) + 1)
                         / (n_null + 1))
            passes_pixel = True
            scrubbed_score = float("nan")
            if pval < args.alpha:
                scrubbed_score, _ = rescore_masking_extreme_pixel(
                    tmap, feat_cfg.nside, c,
                    args.single_pixel_radius_deg,
                    scorer,
                    lambda t, cc: extract_patch_features(t, cc, feat_cfg),
                )
                passes_pixel = bool(scrubbed_score > threshold_alpha)
                verdict = ("PASS (still significant after scrub)"
                            if passes_pixel else "FAIL (drops below threshold)")
                print(f"[Check] rank {k+1}: original={cand['score']:.2f}  "
                      f"scrubbed={scrubbed_score:.2f}  "
                      f"threshold={threshold_alpha:.2f}  → {verdict}",
                      flush=True)
            # Planck component-separated maps live in GALACTIC coordinates.
            # We report (l, b) directly — do not mislabel these as RA/Dec.
            lon, lat = hp.vec2ang(c, lonlat=True)
            gal_l = float(np.atleast_1d(lon)[0])
            gal_b = float(np.atleast_1d(lat)[0])
            f.write(f"{k+1},{args.map_label},{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},"
                    f"{gal_l:.4f},{gal_b:.4f},"
                    f"{cand['score']:.6g},{scrubbed_score:.6g},"
                    f"{pval:.6g},{int(passes_pixel)}\n")
            candidate_records.append({
                "rank": k + 1, "score": cand["score"], "p_value": pval,
                "scrubbed_score": scrubbed_score,
                "gal_l": gal_l, "gal_b": gal_b,
                "passes_pixel_check": passes_pixel,
            })
    print(f"[Report] Candidates CSV: {csv_path}")

    md_path = args.output_dir / f"REAL_RESULTS_{args.map_label}.md"
    with md_path.open("w") as f:
        f.write(f"# Real-data results — {args.map_label}\n\n")
        f.write("Applied the calibrated CCC-HawkingPoints pipeline (Phases 3-7,\n")
        f.write("certified via `scripts/run_control_experiment.py`) to the real map.\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Map: `{args.map}`\n")
        f.write(f"- Mask: `{args.mask}`\n")
        f.write(f"- nside: {args.nside}\n")
        f.write(f"- Patches sampled: {len(scored)} / {args.n_patches} requested\n")
        f.write(f"- Null CDF: {n_null} calibration sims (from parquet)\n")
        f.write(f"- Trials-corrected α = {args.alpha}  →  score threshold "
                f"{threshold_alpha:.3f}\n\n")
        f.write("## Top candidates\n\n")
        f.write("| Rank | Gal l (°) | Gal b (°) | Score | p-value | "
                "Single-pixel check |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in candidate_records:
            pix_check = ("PASS" if r["passes_pixel_check"]
                          else "**FAIL**") if r["p_value"] < args.alpha else "n/a"
            f.write(f"| {r['rank']} | {r['gal_l']:.2f} | {r['gal_b']:.2f} | "
                    f"{r['score']:.3g} | {r['p_value']:.3g} | {pix_check} |\n")

        f.write("\n## Interpretation\n\n")
        significant = [r for r in candidate_records if r["p_value"] < args.alpha]
        pixel_survivors = [r for r in significant if r["passes_pixel_check"]]
        if not significant:
            f.write("**Calibrated non-detection.** No patch exceeds the trials-\n")
            f.write("corrected significance threshold at α = "
                    f"{args.alpha}. This is a complete, publishable result — see\n")
            f.write("CLAUDE.md Sec. 7. The pipeline's sensitivity floor (from the\n")
            f.write("control experiment's Phase 5 curve) sets the amplitude at\n")
            f.write("which a real signal would have been detectable.\n\n")
            f.write("Cross-pipeline validation (Phase 8): re-run this script on the\n")
            f.write("other Planck component-separation maps (NILC, SEVEM, Commander)\n")
            f.write("and on WMAP. A calibrated non-detection replicated across all\n")
            f.write("five maps is a strong constraint on Penrose CCC.\n")
        elif not pixel_survivors:
            f.write(f"**{len(significant)} candidate(s) cleared the trials-corrected\n")
            f.write("significance threshold but ALL failed the Bodnia-2024 single-pixel\n")
            f.write("check** — the signal disappears when the brightest/darkest pixel\n")
            f.write("in the candidate disc is masked. This is the Bodnia et al. 2024\n")
            f.write("failure mode: single outlier pixels faking Hawking-point-like\n")
            f.write("structure. Not a cosmological detection.\n")
        else:
            f.write(f"**{len(pixel_survivors)} candidate(s) cleared BOTH the trials-\n")
            f.write("corrected significance threshold AND the Bodnia-2024 single-pixel\n")
            f.write("check.** Next: run this same script on the other Planck component-\n")
            f.write("separation maps and WMAP. A real Hawking-point-like signal must\n")
            f.write("appear at consistent sky coordinates across all five maps —\n")
            f.write("candidates that don't replicate are foreground residuals, not\n")
            f.write("cosmological.\n")
    print(f"[Report] Markdown summary: {md_path}")
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.parquet.exists():
        print(f"ERROR: {args.parquet} not found. Run the control experiment first:")
        print("  python -m scripts.run_control_experiment "
              "--nside 512 --n-sims 200 --n-patches 30 --skip-flow")
        return 2

    if not args.map.exists():
        print(f"ERROR: real map not found at {args.map}")
        return 2

    train_rows, calib_rows, _control_rows = load_ensembles(args.parquet)
    print(f"[Rebuild] Loaded parquet: train={len(train_rows)}, "
          f"calib={len(calib_rows)} rows")

    # Match the feature config to the control run: same ring binning, same
    # patch size — else the real feature vector wouldn't be in the same space
    # as the trained distilled statistic.
    n_rings = 5 if args.nside < 256 else 9
    feat_cfg = PatchFeatureConfig(
        nside=args.nside,
        radii_deg=np.linspace(0, 2.0, n_rings),
        patch_size_deg=5.0,
        patch_n_pix=128,
    )
    fnames = feature_order(feat_cfg)
    print(f"[Rebuild] Active feature columns: {len(fnames)}")

    _model, _distillation, scorer, null_calib = rebuild_pipeline(
        train_rows, calib_rows, fnames, use_pysr=args.use_pysr
    )

    tmap, mask = load_real_map(args.map, args.mask, args.nside)
    centers = sample_real_patch_centers(args.nside, mask, args.n_patches)
    print(f"[Score] Extracting features + scoring {centers.shape[0]} patches ...")
    rows = score_real_map_patches(tmap, centers, feat_cfg, scorer, fnames)
    print(f"[Score] Scored {len(rows)} patches (dropped "
          f"{centers.shape[0] - len(rows)} with NaN features)")

    report(rows, null_calib, args, tmap, feat_cfg, scorer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
