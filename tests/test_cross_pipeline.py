"""
Tests for src/validation/cross_pipeline.py.

We only exercise the deterministic pieces here — angular matching and the
report-writing skeleton. The full pipeline driver (`run_cross_pipeline`) is
integration-heavy; its dependencies (a trained flow, a distilled statistic, a
real map) are tested in their own suites.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from src.validation.cross_pipeline import (
    CrossPipelineResult,
    MapCandidate,
    match_candidates_across_maps,
    write_final_report,
)


def _cand(map_name: str, vec: tuple[float, float, float], p: float = 0.001,
            pass_pixel: bool = True) -> MapCandidate:
    return MapCandidate(
        map_name=map_name, center_vec=vec, ring_scale_deg=1.0,
        score=10.0, p_value=p, passes_single_pixel_check=pass_pixel,
    )


def test_match_candidates_all_maps_agree_within_radius():
    v_common = (1.0, 0.0, 0.0)
    v_close = (0.9999, 0.01, 0.0); v_close = tuple(np.array(v_close) / np.linalg.norm(v_close))
    v_far = (0.0, 1.0, 0.0)

    per_map = {
        "planck_smica": [_cand("planck_smica", v_common), _cand("planck_smica", v_far)],
        "planck_nilc":  [_cand("planck_nilc",  v_close)],
        "planck_sevem": [_cand("planck_sevem", v_common)],
    }
    matched = match_candidates_across_maps(per_map, match_radius_deg=1.0)
    assert len(matched) == 1
    assert matched[0]["n_maps"] == 3


def test_write_final_report_marks_non_detection_when_no_matches(tmp_path):
    result = CrossPipelineResult(per_map={"planck_smica": [], "wmap9_ilc": []})
    csv, md = write_final_report(result, sensitivity_curve=None,
                                    out_dir_tables=tmp_path / "tables",
                                    out_dir_root=tmp_path)
    assert csv.exists() and md.exists()
    md_text = md.read_text()
    assert "Calibrated non-detection" in md_text
    assert "sensitivity curve" in md_text.lower()


def test_write_final_report_includes_sensitivity_table(tmp_path):
    result = CrossPipelineResult(per_map={"planck_smica": []})
    sensitivity = {0.0: np.array([0.9, 1.0, 1.1]),
                    20.0: np.array([1.5, 1.6, 1.4])}
    _, md = write_final_report(result, sensitivity_curve=sensitivity,
                                 out_dir_tables=tmp_path / "tables",
                                 out_dir_root=tmp_path)
    text = md.read_text()
    assert "0.0 |" in text and "20.0 |" in text


def test_write_final_report_flags_detection_when_matches_pass(tmp_path):
    matched = [{
        "center_vec": (1.0, 0.0, 0.0),
        "n_maps": 5,
        "worst_p_value": 0.001,
        "all_pass_single_pixel_check": True,
        "per_map": {},
    }]
    result = CrossPipelineResult(per_map={"planck_smica": []},
                                    matched_candidates=matched)
    _, md = write_final_report(result, sensitivity_curve=None,
                                 out_dir_tables=tmp_path / "tables",
                                 out_dir_root=tmp_path)
    text = md.read_text()
    assert "survive all three filters" in text
