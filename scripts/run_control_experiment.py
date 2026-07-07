#!/usr/bin/env python
"""
scripts/run_control_experiment.py

Reviewer-expectation control test for the CCC-HawkingPoints pipeline.

Runs Phases 3-7 end-to-end on Gaussian-only ΛCDM simulations and reports whether
the pipeline correctly returns a calibrated non-detection on a null sky. CLAUDE.md
Sec. 7 calls this out as a mandatory reviewer-expectation step: given the field's
history (Gurzadyan-Penrose 2010/2011 → Jow-Scott 2020 correction), no Hawking-point
detection on real data can be taken seriously unless the same pipeline first
demonstrably returns no false positives on Gaussian control data.

What this script does, top to bottom:

  Phase 3  Build three independent null-sim ensembles: train (flow), calib
            (null CDF), control (KS uniformity check). Cache to parquet.
  Phase 4  Train a MAF flow on train (with GaussianDensity fallback if torch
            errors), report train/val NLL.
  Phase 5  Inject Hawking-point-like Gaussian bumps at a grid of amplitudes onto
            fresh null realizations; score with the trained flow; produce a
            sensitivity curve saying "this method could detect X µK at scale Y°."
  Phase 6  Fit PySR (or ridge fallback) to distill the flow's anomaly score into
            a symbolic expression; log the full Pareto front; check whether the
            best simple expression resembles the classical Gurzadyan-Penrose
            ring-variance statistic (a strong internal validation).
  Phase 7  Build the empirical CDF of the distilled statistic's MAP-WIDE MAX on
            the calibration set. Score each control sim as if it were "real data,"
            get its trials-corrected p-value, and Kolmogorov-Smirnov it against
            U(0,1). This is THE calibration certification the pipeline needs to
            clear before real data.

Outputs:
  results/tables/null_features.parquet
  results/tables/control_sensitivity_curve.csv
  results/tables/pysr_pareto.csv
  results/tables/pysr_best_expression.txt
  results/figures/control_sensitivity_curve.png
  results/figures/control_pvalue_uniformity.png
  results/CONTROL_RESULTS.md

Usage:
  # Quick sanity check (few minutes at nside=128)
  python -m scripts.run_control_experiment --nside 128 --n-sims 200

  # Real dev pass (couple of hours at nside=512)
  python -m scripts.run_control_experiment --nside 512 --n-sims 200

  # Production certification (many hours at nside=512)
  python -m scripts.run_control_experiment --nside 512 --n-sims 1000 --use-pysr
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats as scipy_stats

# Repo root on sys.path so `from src.xxx` works even when the script is invoked
# directly (not just via `python -m scripts.run_control_experiment`).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.simulations.gaussian_null import (  # noqa: E402
    NullSimConfig,
    build_null_feature_table,
    generate_null_map,
    get_cl_theory,
    write_feature_parquet,
)
from src.features.feature_vector import (  # noqa: E402
    PatchFeatureConfig,
    extract_patch_features,
    feature_order,
)
from src.models.flow_density import (  # noqa: E402
    GaussianDensity,
    MAFDensity,
    rows_to_matrix,
    split_by_sim,
)
from src.distillation.pysr_distill import (  # noqa: E402
    build_evaluator,
    distill,
    save_pareto_table,
)
from src.calibration.trials_correction import (  # noqa: E402
    build_null_distribution,
    calibrate_real_map,
)
from src.validation.injection_test import hawking_point_template  # noqa: E402


# ---------------------------------------------------------------------------
# CLI + config
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--nside", type=int, default=128,
                    help="HEALPix resolution (default 128 for quick dev; 512 for real)")
    p.add_argument("--n-sims", type=int, default=200,
                    help="Total null simulations (default 200 dev; 1000+ for production)")
    p.add_argument("--n-patches", type=int, default=20,
                    help="Patch centers sampled per simulation")
    p.add_argument("--output-dir", type=Path, default=Path("results"),
                    help="Where to write tables/figures/CONTROL_RESULTS.md")
    p.add_argument("--use-pysr", action="store_true",
                    help="Use PySR for Phase 6 (requires Julia). Default: ridge fallback")
    p.add_argument("--skip-flow", action="store_true",
                    help="Skip MAF, use GaussianDensity baseline (seconds, not minutes)")
    p.add_argument("--flow-layers", type=int, default=3,
                    help="MAF depth (default 3 for dev; try 6-8 for production)")
    p.add_argument("--flow-hidden", type=int, default=32,
                    help="MAF hidden width (default 32 dev; 64-128 for production)")
    p.add_argument("--flow-epochs", type=int, default=50,
                    help="MAF training epochs (default 50 dev; 200+ for production)")
    return p.parse_args(argv)


def build_configs(args) -> tuple[NullSimConfig, PatchFeatureConfig]:
    """
    Choose sim + feature configs sized appropriately for the chosen nside.

    At low nside the coarse pixel resolution can't populate narrow rings, so we
    use fewer, wider rings; at higher nside we go back to the CLAUDE.md defaults.
    """
    sim_cfg = NullSimConfig(
        nside=args.nside,
        lmax=min(3 * args.nside - 1, 2000),
        beam_fwhm_arcmin=5.0,   # Planck-ish, not real — replace with the manifest value
        noise_uk_arcmin=0.0,     # clean control run: no instrument noise
        seed=None,
        cosmology={},
    )
    # Ring binning tied to nside: at nside=128 (~0.46° per pixel) an 0.25° ring
    # is a single pixel, so we cap at 5 rings x 0.5°. At nside>=256 we can
    # afford the finer 8 rings x 0.25° that CLAUDE.md's methods list uses.
    n_rings = 5 if args.nside < 256 else 9
    feat_cfg = PatchFeatureConfig(
        nside=args.nside,
        radii_deg=np.linspace(0, 2.0, n_rings),
        patch_size_deg=5.0,
        patch_n_pix=128,
    )
    return sim_cfg, feat_cfg


# ---------------------------------------------------------------------------
# Phase 3: null ensemble builders
# ---------------------------------------------------------------------------


def build_ensemble(n_sims: int, n_patches: int,
                    sim_cfg: NullSimConfig, feat_cfg: PatchFeatureConfig,
                    cl_tt: np.ndarray, seed0: int, label: str) -> list[dict]:
    print(f"[Phase 3] Building {label} ensemble: "
          f"{n_sims} sims × {n_patches} patches ...")
    rows = build_null_feature_table(
        n_sims=n_sims,
        n_patches_per_sim=n_patches,
        base_cfg=sim_cfg,
        feat_cfg=feat_cfg,
        cl_tt=cl_tt,
        seed0=seed0,
    )
    for r in rows:
        r["ensemble"] = label
    print(f"[Phase 3] {label}: {len(rows)} patches extracted")
    return rows


# ---------------------------------------------------------------------------
# Phase 4: flow density
# ---------------------------------------------------------------------------


def phase4_train_flow(train_rows: list[dict], val_rows: list[dict],
                       fnames: list[str], use_maf: bool = True,
                       n_layers: int = 3, hidden: int = 32,
                       n_epochs: int = 50):
    Xtr = rows_to_matrix(train_rows, fnames)
    Xval = rows_to_matrix(val_rows, fnames)
    print(f"[Phase 4] X_train: {Xtr.shape}, X_val: {Xval.shape}")

    if use_maf:
        try:
            model = MAFDensity(n_layers=n_layers, hidden=hidden, lr=1e-3,
                                n_epochs=n_epochs, batch=256, seed=0)
            model.fit(Xtr, Xval)
            train_nll = float(np.mean(model.anomaly_score(Xtr)))
            val_nll = float(np.mean(model.anomaly_score(Xval)))
            print(f"[Phase 4] MAF trained. train NLL={train_nll:.4f}, "
                  f"val NLL={val_nll:.4f}")
            return model, {"backend": "MAFDensity",
                           "train_nll": train_nll, "val_nll": val_nll}
        except Exception as e:  # torch missing, CUDA issue, whatever
            print(f"[Phase 4] MAF failed ({type(e).__name__}: {e}); "
                  "falling back to GaussianDensity")

    model = GaussianDensity().fit(Xtr)
    train_nll = float(np.mean(model.anomaly_score(Xtr)))
    val_nll = float(np.mean(model.anomaly_score(Xval)))
    print(f"[Phase 4] Gaussian baseline. train NLL={train_nll:.4f}, "
          f"val NLL={val_nll:.4f}")
    return model, {"backend": "GaussianDensity",
                   "train_nll": train_nll, "val_nll": val_nll}


# ---------------------------------------------------------------------------
# Phase 5: sensitivity sweep
# ---------------------------------------------------------------------------


def phase5_sensitivity_sweep(model, sim_cfg: NullSimConfig,
                              feat_cfg: PatchFeatureConfig,
                              cl_tt: np.ndarray, fnames: list[str],
                              amplitudes_uk: list[float],
                              n_realizations: int = 30,
                              sigma_deg: float = 1.0,
                              seed0: int = 7_777) -> dict:
    """
    Paired-injection sweep: for each of n_realizations independent null draws
    we score EVERY amplitude at the SAME sky location. Paired sampling makes
    the amplitude ladder much less noisy at fixed n_realizations.
    """
    print(f"[Phase 5] Sensitivity sweep: {len(amplitudes_uk)} amplitudes "
          f"× {n_realizations} realizations (paired)")
    results: dict[float, list[float]] = {float(a): [] for a in amplitudes_uk}
    for i in range(n_realizations):
        cfg = NullSimConfig(
            nside=sim_cfg.nside, lmax=sim_cfg.lmax,
            beam_fwhm_arcmin=sim_cfg.beam_fwhm_arcmin,
            noise_uk_arcmin=sim_cfg.noise_uk_arcmin,
            seed=seed0 + i,
        )
        base_tmap = generate_null_map(cfg, cl_tt=cl_tt)
        rng = np.random.default_rng(seed0 * 7 + i * 13)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        for amp in amplitudes_uk:
            if amp > 0:
                tmap = base_tmap + hawking_point_template(
                    sim_cfg.nside, v, amp, sigma_deg
                )
            else:
                tmap = base_tmap
            feats = extract_patch_features(tmap, v, feat_cfg)
            X_one = np.array([[feats[k] for k in fnames]])
            if np.isfinite(X_one).all():
                results[float(amp)].append(float(model.anomaly_score(X_one)[0]))
    for amp in amplitudes_uk:
        arr = np.array(results[float(amp)])
        results[float(amp)] = arr
        m = float(np.nanmean(arr)) if arr.size else float("nan")
        print(f"[Phase 5]   amp={amp:6.1f} µK: n={arr.size}, mean_score={m:.3f}")
    return results


# ---------------------------------------------------------------------------
# Phase 6: PySR distillation
# ---------------------------------------------------------------------------


def phase6_distill(model, train_rows: list[dict], fnames: list[str],
                    use_pysr: bool = False):
    print("[Phase 6] Symbolic distillation of the flow's anomaly score ...")
    X = rows_to_matrix(train_rows, fnames)
    y = model.anomaly_score(X)
    result = distill(X, y, fnames, use_pysr=use_pysr,
                      niterations=100, complexity=20)
    print(f"[Phase 6] Best expression (R²={result.best_r2:.4f}):")
    print(f"          {result.best_expression}")
    print(f"[Phase 6] Resembles classical ring-variance stat: "
          f"{result.resembles_classical}")
    print(f"          {result.resembles_classical_note}")
    return result


# ---------------------------------------------------------------------------
# Phase 7: trials-corrected calibration + KS uniformity certification
# ---------------------------------------------------------------------------


def phase7_calibrate(calib_rows: list[dict], control_rows: list[dict],
                      scorer: Callable[[dict], float]):
    print("[Phase 7] Building trials-corrected null CDF of distilled statistic ...")
    null_calib = build_null_distribution(calib_rows, scorer=scorer, top_k=5)
    p05 = float(np.quantile(null_calib.max_scores, 0.05))
    p50 = float(np.quantile(null_calib.max_scores, 0.5))
    p95 = float(np.quantile(null_calib.max_scores, 0.95))
    print(f"[Phase 7] Null CDF (n={null_calib.n_sims}): "
          f"map-wide-max p05={p05:.3f}, p50={p50:.3f}, p95={p95:.3f}")

    per_sim: dict[int, list[dict]] = defaultdict(list)
    for r in control_rows:
        per_sim[r["sim_id"]].append(r)

    p_values: list[float] = []
    for _, rows in per_sim.items():
        report = calibrate_real_map(rows, null_calib, scorer=scorer)
        p_values.append(report.p_value_max)
    p_arr = np.array(p_values)
    ks_stat, ks_p = scipy_stats.kstest(p_arr, "uniform")
    verdict = "PASS" if ks_p > 0.001 else "FAIL"
    print(f"[Phase 7] KS uniformity on {len(p_arr)} held-out sims: "
          f"p_KS={ks_p:.4f}, ks_stat={ks_stat:.4f}  →  {verdict}")
    return null_calib, p_arr, float(ks_stat), float(ks_p), verdict


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_sensitivity_csv(sensitivity: dict, out_path: Path) -> None:
    with out_path.open("w") as f:
        f.write("amplitude_uK,n,mean_score,std_score,max_score\n")
        for amp, arr in sorted(sensitivity.items()):
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                f.write(f"{amp},0,nan,nan,nan\n")
            else:
                f.write(f"{amp},{arr.size},"
                        f"{arr.mean():.6g},{arr.std():.6g},{arr.max():.6g}\n")


def _write_figures(sensitivity: dict, p_values: np.ndarray,
                    ks_p: float, verdict: str, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[Report] Matplotlib unavailable ({e}); skipping figures")
        return

    amps = sorted(sensitivity.keys())
    means = [float(np.nanmean(sensitivity[a])) for a in amps]
    stds = [float(np.nanstd(sensitivity[a])) for a in amps]
    plt.figure(figsize=(6, 4))
    plt.errorbar(amps, means, yerr=stds, marker="o", capsize=3)
    plt.xlabel("Injected Hawking-point amplitude (µK)")
    plt.ylabel("Flow anomaly score (−log p)")
    plt.title("Control-run sensitivity curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "control_sensitivity_curve.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(p_values, bins=15, density=True, edgecolor="k", alpha=0.7)
    plt.axhline(1.0, color="red", linestyle="--", label="Uniform expectation")
    plt.xlabel("p_value_of_max on held-out control sims")
    plt.ylabel("Density")
    plt.title(f"Calibration KS p={ks_p:.4f}  ({verdict})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "control_pvalue_uniformity.png", dpi=120)
    plt.close()
    print(f"[Report] Figures written to {out_dir}")


def write_report(output_dir: Path, args, flow_info: dict, sensitivity: dict,
                  distillation, p_values: np.ndarray, ks_stat: float, ks_p: float,
                  verdict: str, n_train: int, n_calib: int, n_control: int) -> Path:
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    _write_sensitivity_csv(sensitivity, tables / "control_sensitivity_curve.csv")
    save_pareto_table(distillation, out_dir=tables)
    _write_figures(sensitivity, p_values, ks_p, verdict, figures)

    md_path = output_dir / "CONTROL_RESULTS.md"
    with md_path.open("w") as f:
        f.write("# Control experiment — CCC-HawkingPoints pipeline\n\n")
        f.write("Reviewer-expectation control test on Gaussian-only ΛCDM simulations.\n")
        f.write("The pipeline must return a calibrated non-detection here BEFORE it\n")
        f.write("is trusted on real Planck/WMAP maps (CLAUDE.md Sec. 7).\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- nside: {args.nside}\n")
        f.write(f"- N_sims: {args.n_sims} "
                f"(train={n_train}, calib={n_calib}, control={n_control})\n")
        f.write(f"- Patches per sim: {args.n_patches}\n")
        f.write(f"- Flow backend: {flow_info['backend']}\n")
        f.write(f"- PySR enabled: {args.use_pysr}\n\n")

        f.write("## Phase 4 — Flow density\n\n")
        f.write(f"- Train NLL: {flow_info['train_nll']:.4f}\n")
        f.write(f"- Val NLL: {flow_info['val_nll']:.4f}\n\n")
        f.write("A large train/val NLL gap suggests overfitting — shrink the flow\n")
        f.write("(fewer layers / smaller hidden) or grow N_sims.\n\n")

        f.write("## Phase 5 — Sensitivity curve\n\n")
        f.write("| Injected amp (µK) | n | mean flow score | std |\n")
        f.write("|---|---|---|---|\n")
        for amp, arr in sorted(sensitivity.items()):
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                f.write(f"| {amp:.1f} | 0 | n/a | n/a |\n")
            else:
                f.write(f"| {amp:.1f} | {arr.size} | "
                        f"{arr.mean():.4g} | {arr.std():.4g} |\n")
        f.write("\nSee `results/figures/control_sensitivity_curve.png`.\n\n")

        f.write("## Phase 6 — Symbolic distillation\n\n")
        f.write(f"- Best expression: `{distillation.best_expression}`\n")
        f.write(f"- R² vs flow score: {distillation.best_r2:.4f}\n")
        f.write(f"- Resembles classical ring-variance stat: "
                f"**{distillation.resembles_classical}**\n")
        f.write(f"  ({distillation.resembles_classical_note})\n")
        f.write("- Pareto front: `results/tables/pysr_pareto.csv`\n\n")

        f.write("## Phase 7 — Trials-corrected calibration\n\n")
        f.write(f"- Null CDF built from {n_calib} calibration sims\n")
        f.write(f"- Held-out control ensemble: {n_control} sims\n")
        f.write(f"- KS uniformity: p_KS = **{ks_p:.4f}**, ks_stat = {ks_stat:.4f}\n")
        f.write(f"- Verdict: **{verdict}**\n\n")
        f.write("See `results/figures/control_pvalue_uniformity.png`.\n\n")

        f.write("## Bottom line\n\n")
        if verdict == "PASS":
            f.write("**GO** — control run confirms calibrated non-detection.\n")
            f.write("The pipeline is ready to be applied to real Planck/WMAP maps.\n\n")
            f.write("Next: `python -m src.simulations.fetch_data` to download the\n")
            f.write("maps, then run the equivalent driver on real data and cross-\n")
            f.write("pipeline validate per Phase 8.\n")
        else:
            f.write("**NO-GO** — control run failed KS uniformity.\n")
            f.write("Do NOT run on real data yet. Likely causes to investigate:\n\n")
            f.write("- Feature vector contains a statistic that isn't null-invariant.\n")
            f.write("- Train/val split leaked correlated patches (check split_by_sim).\n")
            f.write("- N_sims too small for a stable tail — rerun at 500-1000.\n")
            f.write("- Distilled statistic overfits training draws — check the\n")
            f.write("  Pareto front, try higher complexity budget or fewer features.\n")
    print(f"[Report] {md_path}")
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg, feat_cfg = build_configs(args)
    print(f"[Init] nside={args.nside}, lmax={sim_cfg.lmax}, "
          f"n_rings={len(feat_cfg.radii_deg) - 1}, "
          f"radii_deg={list(feat_cfg.radii_deg)}")

    print("[Init] Computing fiducial ΛCDM C_ell via CAMB "
          "(one-time; reused across all sims) ...")
    cl_tt = get_cl_theory(sim_cfg)

    # Split budget: 50% flow train, 30% null-CDF calib, 20% held-out KS control.
    # Rationale: flow needs the most, CDF needs enough to resolve its tail, and
    # KS uniformity gets whatever's left (must be >= ~40 to give the KS test any
    # power — see test_null_pvalues_are_uniform_on_control_ensemble).
    n_train = int(args.n_sims * 0.5)
    n_calib = int(args.n_sims * 0.3)
    n_control = args.n_sims - n_train - n_calib
    print(f"[Init] Sim split: train={n_train}, calib={n_calib}, control={n_control}")

    train_rows = build_ensemble(n_train, args.n_patches, sim_cfg, feat_cfg,
                                  cl_tt, seed0=100_000, label="train")
    calib_rows = build_ensemble(n_calib, args.n_patches, sim_cfg, feat_cfg,
                                  cl_tt, seed0=200_000, label="calib")
    control_rows = build_ensemble(n_control, args.n_patches, sim_cfg, feat_cfg,
                                    cl_tt, seed0=300_000, label="control")

    parquet_path = args.output_dir / "tables" / "null_features.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_feature_parquet(train_rows + calib_rows + control_rows, parquet_path)
        print(f"[Phase 3] Feature parquet: {parquet_path}")
    except RuntimeError as e:
        print(f"[Phase 3] Parquet write skipped ({e}); continuing in-memory")

    train_actual, val_rows = split_by_sim(train_rows, val_frac=0.2, seed=42)
    print(f"[Init] Train sims after val split: "
          f"train={len({r['sim_id'] for r in train_actual})}, "
          f"val={len({r['sim_id'] for r in val_rows})}")

    fnames = feature_order(feat_cfg)
    print(f"[Init] Active feature columns: {len(fnames)}")

    model, flow_info = phase4_train_flow(train_actual, val_rows, fnames,
                                            use_maf=not args.skip_flow,
                                            n_layers=args.flow_layers,
                                            hidden=args.flow_hidden,
                                            n_epochs=args.flow_epochs)

    amplitudes_uk = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
    sensitivity = phase5_sensitivity_sweep(model, sim_cfg, feat_cfg, cl_tt,
                                              fnames, amplitudes_uk,
                                              n_realizations=30)

    distillation = phase6_distill(model, train_actual, fnames,
                                     use_pysr=args.use_pysr)
    scorer_fn = build_evaluator(distillation)

    null_calib, p_values, ks_stat, ks_p, verdict = phase7_calibrate(
        calib_rows, control_rows, scorer_fn
    )

    write_report(args.output_dir, args, flow_info, sensitivity, distillation,
                  p_values, ks_stat, ks_p, verdict,
                  n_train, n_calib, n_control)

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
