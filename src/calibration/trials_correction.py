"""
Phase 7: trials-factor / look-elsewhere-corrected calibration.

The central lesson of Jow & Scott (2020) is that per-patch p-values are the
wrong reference distribution — the field of view contains millions of patches,
so the correct calibration is against the distribution of the MAP-WIDE MAXIMUM
of the detection statistic under the null hypothesis.

Concretely, for each null simulation:
    1. Score every patch through the distilled statistic (Phase 6).
    2. Record only the maximum score across the whole map.
The empirical CDF of these maxima IS the trials-corrected null. A real map's
maximum score is significant only relative to this distribution — never relative
to a single-patch null.

Top-N candidate patches on the real map get p-values in the same reference
frame: the fraction of null sims whose k-th largest score exceeds the candidate's
k-th largest score, for each k. This keeps the top-N calibration honest (a
top-3 candidate has to beat top-3s of the null sims, not top-1s).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np


@dataclass
class NullMaxCalibration:
    """
    Frozen empirical null distribution of the map-wide max of the statistic.
    """
    max_scores: np.ndarray            # (n_sims,) — one map-wide max per null sim
    top_k_scores: np.ndarray          # (n_sims, K) — top-K per sim, K = top_k_reported
    top_k_reported: int
    n_sims: int

    def pvalue_of_max(self, observed_max: float) -> float:
        """
        Two-sided-safe upper-tail p-value: fraction of null sims whose map-wide
        max exceeds the observed maximum. Uses the standard +1/(n+1) shift so a
        finite null ensemble cannot report exactly p=0 (which would be an
        overconfident lie about the tail).
        """
        n_exceed = int(np.sum(self.max_scores >= observed_max))
        return (n_exceed + 1) / (self.n_sims + 1)

    def pvalue_of_topk(self, observed_topk: np.ndarray) -> np.ndarray:
        """
        Per-k trials-corrected p-value: fraction of null sims whose k-th largest
        score >= the candidate's k-th largest, for k = 1..K.
        """
        observed_topk = np.asarray(observed_topk)
        K = observed_topk.size
        pvals = np.zeros(K)
        for k in range(K):
            n_exceed = int(np.sum(self.top_k_scores[:, k] >= observed_topk[k]))
            pvals[k] = (n_exceed + 1) / (self.n_sims + 1)
        return pvals


def _map_scores_by_sim(rows: Sequence[dict],
                        scorer: Callable[[dict], float]) -> "dict[int, np.ndarray]":
    """
    Group per-patch scores by sim_id and return a dict {sim_id: array of scores}.
    Rows without a sim_id are ignored (belongs to real data, not the null).
    """
    grouped: dict[int, list[float]] = {}
    for r in rows:
        sim_id = r.get("sim_id")
        if sim_id is None:
            continue
        val = scorer(r)
        if np.isfinite(val):
            grouped.setdefault(int(sim_id), []).append(float(val))
    return {sid: np.array(v) for sid, v in grouped.items()}


def build_null_distribution(rows: Sequence[dict],
                              scorer: Callable[[dict], float],
                              top_k: int = 5) -> NullMaxCalibration:
    """
    Walk the Phase-3 null feature table, score every patch through `scorer` (the
    distilled statistic from Phase 6, or the flow's own anomaly score for a
    diagnostic run), and reduce each simulation to its map-wide max + top-K
    highest scores.
    """
    per_sim = _map_scores_by_sim(rows, scorer)
    sim_ids = sorted(per_sim.keys())
    if not sim_ids:
        raise ValueError("no null-sim rows found — expected rows carrying sim_id")

    maxes = np.array([per_sim[s].max() for s in sim_ids])
    topk_list = []
    for s in sim_ids:
        scores = np.sort(per_sim[s])[::-1]
        if scores.size >= top_k:
            topk_list.append(scores[:top_k])
        else:  # tiny sim in a test — pad with -inf so it counts as no candidate
            padded = np.full(top_k, -np.inf)
            padded[: scores.size] = scores
            topk_list.append(padded)
    topk = np.vstack(topk_list)
    return NullMaxCalibration(
        max_scores=maxes,
        top_k_scores=topk,
        top_k_reported=top_k,
        n_sims=len(sim_ids),
    )


def score_real_map(rows: Sequence[dict],
                     scorer: Callable[[dict], float]) -> np.ndarray:
    """Vector of per-patch scores for a real-data map's rows."""
    vals = [scorer(r) for r in rows]
    return np.array([v for v in vals if np.isfinite(v)])


@dataclass
class CalibrationReport:
    observed_max: float
    p_value_max: float
    top_k_scores: np.ndarray
    top_k_pvalues: np.ndarray
    n_null_sims: int


def calibrate_real_map(real_rows: Sequence[dict],
                        null_calibration: NullMaxCalibration,
                        scorer: Callable[[dict], float]) -> CalibrationReport:
    """
    Score a real map through `scorer`, compare its map-wide max and top-K
    candidates against the pre-built null distribution, and return a report.
    """
    real_scores = np.sort(score_real_map(real_rows, scorer))[::-1]
    K = null_calibration.top_k_reported
    if real_scores.size < K:
        raise ValueError(f"real map has {real_scores.size} scorable patches, need >= {K}")
    top_k = real_scores[:K]
    obs_max = float(top_k[0])
    p_max = null_calibration.pvalue_of_max(obs_max)
    p_topk = null_calibration.pvalue_of_topk(top_k)
    return CalibrationReport(observed_max=obs_max,
                              p_value_max=p_max,
                              top_k_scores=top_k,
                              top_k_pvalues=p_topk,
                              n_null_sims=null_calibration.n_sims)
