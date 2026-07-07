"""
Phase 8: cross-pipeline validation and final candidate report.

A candidate patch is only reported as a genuine possible Hawking-point signal
if it clears three independent filters (per CLAUDE.md, Sec. 2 stage 6):

  (a) Trials-corrected significance: its Phase-6 distilled statistic exceeds
      the Phase-7 null distribution's threshold at the requested alpha level.

  (b) Cross-pipeline consistency: the same sky location + angular scale is
      flagged in every Planck component-separation map (SMICA, NILC, SEVEM,
      Commander) and in WMAP. A candidate appearing in only one map is very
      likely a foreground / component-separation residual, not a cosmological
      signal.

  (c) Not-a-single-pixel: after masking the brightest / darkest single pixel
      inside the candidate disc and re-scoring, the statistic must still clear
      the threshold. This is the explicit check for the Bodnia et al. 2024
      failure mode — a single hot/cold pixel that fakes a low-variance circle.

The module exports one high-level entry point, `run_cross_pipeline`, plus
`write_final_report` which produces:

  results/tables/candidates.csv
  results/RESULTS.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence
import numpy as np
import healpy as hp

from src.calibration.trials_correction import (
    NullMaxCalibration,
    calibrate_real_map,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MapCandidate:
    map_name: str          # e.g. "planck_smica"
    center_vec: tuple[float, float, float]
    ring_scale_deg: float  # angular scale from the ring-stats config
    score: float
    p_value: float
    passes_single_pixel_check: bool


@dataclass
class CrossPipelineResult:
    per_map: dict[str, list[MapCandidate]]
    matched_candidates: list[dict] = field(default_factory=list)
    sensitivity_curve: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cross-map matching
# ---------------------------------------------------------------------------


def _angular_separation(a: tuple[float, float, float],
                          b: tuple[float, float, float]) -> float:
    v1 = np.asarray(a); v2 = np.asarray(b)
    dot = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def match_candidates_across_maps(per_map: dict[str, list[MapCandidate]],
                                    match_radius_deg: float = 0.5,
                                    required_map_count: int | None = None
                                    ) -> list[dict]:
    """
    A candidate is "matched" if a patch within match_radius_deg of it appears in
    every required map. required_map_count defaults to len(per_map) (must appear
    in every map); pass a smaller value to test cross-map consistency under
    relaxed requirements (useful when WMAP resolution can't resolve the finer
    Planck-scale candidates).
    """
    map_names = list(per_map.keys())
    required = required_map_count if required_map_count is not None else len(map_names)
    anchor_map = map_names[0]
    matched = []
    for anchor in per_map[anchor_map]:
        cluster = {anchor_map: anchor}
        for other in map_names[1:]:
            hit = next(
                (c for c in per_map[other]
                 if _angular_separation(anchor.center_vec, c.center_vec)
                 <= match_radius_deg),
                None,
            )
            if hit is not None:
                cluster[other] = hit
        if len(cluster) >= required:
            matched.append({
                "center_vec": anchor.center_vec,
                "n_maps": len(cluster),
                "worst_p_value": max(c.p_value for c in cluster.values()),
                "all_pass_single_pixel_check":
                    all(c.passes_single_pixel_check for c in cluster.values()),
                "per_map": {name: {
                    "score": c.score,
                    "p_value": c.p_value,
                    "passes_single_pixel_check": c.passes_single_pixel_check,
                } for name, c in cluster.items()},
            })
    return matched


# ---------------------------------------------------------------------------
# Bodnia-style single-pixel check
# ---------------------------------------------------------------------------


def rescore_masking_extreme_pixel(tmap: np.ndarray, nside: int,
                                    center_vec: np.ndarray,
                                    check_radius_deg: float,
                                    row_scorer: Callable[[dict], float],
                                    feat_extractor: Callable[[np.ndarray, np.ndarray], dict],
                                    n_sigma: float = 3.0,
                                    ) -> tuple[float, dict]:
    """
    Mask every pixel inside the check disc whose |T - median| exceeds
    n_sigma × local MAD-derived std, re-extract features, re-score.

    Previously we masked only the single brightest pixel (following the letter
    of Bodnia et al. 2024). That misses **compact-cluster** point sources — a
    handful of contiguous hot pixels from an unresolved radio galaxy — where
    the pipeline flags the source but no single pixel dominates. Masking all
    n_sigma-excess pixels in the disc catches both.

    Median absolute deviation (MAD) rather than raw std gives a robust local
    scale that isn't itself blown up by the outliers we're trying to remove.
    """
    disc = hp.query_disc(nside, center_vec, np.radians(check_radius_deg))
    vals = tmap[disc]
    finite = np.isfinite(vals)
    if not finite.any():
        return float("-inf"), {}

    v = vals[finite]
    local_median = float(np.median(v))
    mad = float(np.median(np.abs(v - local_median)))
    sigma_est = 1.4826 * mad          # MAD → gaussian-consistent σ
    if sigma_est == 0.0:
        sigma_est = float(np.std(v))  # degenerate all-equal disc

    excess = np.abs(v - local_median) > n_sigma * sigma_est
    victim_pix = disc[finite][excess]
    if victim_pix.size == 0:
        # No excess pixel — score would be identical, no need to re-extract.
        return float(row_scorer(feat_extractor(tmap, center_vec))), {}

    tmap_scrubbed = tmap.copy()
    tmap_scrubbed[victim_pix] = local_median
    feats = feat_extractor(tmap_scrubbed, center_vec)
    return float(row_scorer(feats)), feats


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------


def run_cross_pipeline(
    real_map_rows: dict[str, list[dict]],
    tmap_by_map: dict[str, np.ndarray],
    nside_by_map: dict[str, int],
    null_calibration: NullMaxCalibration,
    row_scorer: Callable[[dict], float],
    feat_extractor: Callable[[np.ndarray, np.ndarray], dict],
    top_k: int = 5,
    alpha: float = 0.01,
    single_pixel_check_radius_deg: float = 1.0,
) -> CrossPipelineResult:
    """
    Score every provided map, retain candidates above the alpha threshold from
    the calibration null, apply the single-pixel check, and then match across
    maps.
    """
    per_map: dict[str, list[MapCandidate]] = {}
    for map_name, rows in real_map_rows.items():
        report = calibrate_real_map(rows, null_calibration, scorer=row_scorer)
        keep_ks = [k for k in range(top_k) if report.top_k_pvalues[k] < alpha]
        cands: list[MapCandidate] = []
        for k in keep_ks:
            # Find the row whose scorer output = top_k_scores[k]. We recompute
            # scorer(row) rather than storing indices upstream — costs nothing
            # given top_k is small.
            target = report.top_k_scores[k]
            row = next(r for r in rows
                       if np.isclose(row_scorer(r), target))
            center = np.array([row["center_x"], row["center_y"], row["center_z"]])
            new_score, _ = rescore_masking_extreme_pixel(
                tmap_by_map[map_name], nside_by_map[map_name], center,
                single_pixel_check_radius_deg,
                row_scorer, feat_extractor,
            )
            # Pass the check if the score after scrubbing still lies above the
            # null distribution's alpha threshold.
            n_null = null_calibration.n_sims
            threshold = float(np.quantile(null_calibration.max_scores,
                                            1.0 - alpha))
            passes = bool(new_score > threshold)
            cands.append(MapCandidate(
                map_name=map_name,
                center_vec=tuple(float(x) for x in center),
                ring_scale_deg=float(row.get("ring_scale_deg", 2.0)),
                score=float(target),
                p_value=float(report.top_k_pvalues[k]),
                passes_single_pixel_check=passes,
            ))
        per_map[map_name] = cands

    matched = match_candidates_across_maps(per_map)
    return CrossPipelineResult(per_map=per_map, matched_candidates=matched)


# ---------------------------------------------------------------------------
# Final report artifacts
# ---------------------------------------------------------------------------


def write_final_report(result: CrossPipelineResult,
                        sensitivity_curve: dict | None,
                        out_dir_tables: Path | str = "results/tables",
                        out_dir_root: Path | str = "results") -> tuple[Path, Path]:
    """
    Writes results/tables/candidates.csv and results/RESULTS.md.

    The results markdown is deliberately blunt about the outcome. A calibrated
    non-detection with a stated sensitivity limit is a complete, publishable
    result for this pipeline (CLAUDE.md, "Reviewer expectation"). It is not a
    failure and must not be written up as one.
    """
    tables = Path(out_dir_tables)
    tables.mkdir(parents=True, exist_ok=True)
    root = Path(out_dir_root)
    root.mkdir(parents=True, exist_ok=True)

    csv_path = tables / "candidates.csv"
    with csv_path.open("w") as f:
        f.write("map_name,center_x,center_y,center_z,score,p_value,"
                "passes_single_pixel_check\n")
        for map_name, cands in result.per_map.items():
            for c in cands:
                cx, cy, cz = c.center_vec
                f.write(f"{map_name},{cx:.6f},{cy:.6f},{cz:.6f},"
                        f"{c.score:.6g},{c.p_value:.6g},"
                        f"{int(c.passes_single_pixel_check)}\n")

    md_path = root / "RESULTS.md"
    all_pass = [
        m for m in result.matched_candidates
        if m["all_pass_single_pixel_check"]
        and m["worst_p_value"] < 0.01
    ]
    with md_path.open("w") as f:
        f.write("# CCC-HawkingPoints — results\n\n")
        f.write("## Outcome\n\n")
        if all_pass:
            f.write(f"{len(all_pass)} candidate(s) survive all three filters: "
                    "trials-corrected significance, cross-pipeline consistency "
                    "across every input map, and the Bodnia-2024 single-pixel "
                    "check. Detail below.\n\n")
        else:
            f.write("Calibrated non-detection.\n\n"
                    "No patch survives all three filters simultaneously: "
                    "trials-corrected significance, cross-pipeline consistency, "
                    "and the Bodnia-2024 single-pixel check. Per CLAUDE.md this "
                    "is a complete, publishable result — see the sensitivity "
                    "curve below for the amplitude limits this constraint applies to.\n\n")
        f.write("## Sensitivity curve\n\n")
        if sensitivity_curve:
            f.write("| injected amplitude (uK) | mean score | std score | n |\n")
            f.write("|---|---|---|---|\n")
            for amp, scores in sorted(sensitivity_curve.items()):
                arr = np.asarray(scores)
                valid = arr[np.isfinite(arr)]
                if valid.size == 0:
                    f.write(f"| {amp:.1f} | n/a | n/a | 0 |\n")
                else:
                    f.write(f"| {amp:.1f} | {valid.mean():.4g} | {valid.std():.4g} | "
                            f"{valid.size} |\n")
        else:
            f.write("Run Phase 5 (`src/validation/injection_test.py`) and pass the "
                    "output dict into `write_final_report` to populate this table.\n")
        f.write("\n## Per-map candidate counts\n\n")
        for map_name, cands in result.per_map.items():
            f.write(f"- **{map_name}**: {len(cands)} candidates above alpha threshold "
                    f"({sum(c.passes_single_pixel_check for c in cands)} pass single-pixel check)\n")
        f.write("\n## Matched candidates (cross-pipeline)\n\n")
        if not result.matched_candidates:
            f.write("None.\n")
        for m in result.matched_candidates:
            f.write(f"- center={m['center_vec']}, n_maps={m['n_maps']}, "
                    f"worst_p={m['worst_p_value']:.3g}, "
                    f"single_pixel_check_all_pass={m['all_pass_single_pixel_check']}\n")
    return csv_path, md_path
