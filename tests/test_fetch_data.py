"""
Tests for src/simulations/fetch_data.py.

CI must not hit the network. We drive fetch_all with a fake downloader that
writes a tiny FITS-like blob to disk, then verify_manifest re-reads the manifest
and re-hashes the files. Corruption and truncation are then simulated locally
to confirm verify_manifest actually catches them.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import yaml

from src.simulations.fetch_data import (
    MapSpec,
    fetch_all,
    verify_manifest,
)


FITS_HEADER = b"SIMPLE  =                    T"  # begins with the 9-byte magic


def _fake_specs() -> tuple[MapSpec, ...]:
    return (
        MapSpec(key="planck_smica",
                url="http://example.invalid/COM_CMB_IQU-smica_2048_R3.00_full.fits",
                release="Planck PR3 (2018)", kind="map"),
        MapSpec(key="planck_common_mask",
                url="http://example.invalid/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
                release="Planck PR3 (2018)", kind="mask"),
    )


def _make_fake_downloader(payloads: dict[str, bytes]):
    def dl(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payloads[url])
    return dl


def test_fetch_all_writes_manifest_and_files(tmp_path):
    specs = _fake_specs()
    payloads = {
        specs[0].url: FITS_HEADER + b"\x00" * 128,
        specs[1].url: FITS_HEADER + b"\xff" * 64,
    }
    manifest_path = fetch_all(data_raw=tmp_path,
                              specs=specs,
                              downloader=_make_fake_downloader(payloads),
                              today=date(2026, 7, 5))
    assert manifest_path.exists()

    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)
    keys = {e["key"] for e in manifest["entries"]}
    assert keys == {"planck_smica", "planck_common_mask"}
    for e in manifest["entries"]:
        assert e["downloaded_on"] == "2026-07-05"
        assert e["bytes"] > 0
        assert len(e["sha256"]) == 64


def test_verify_manifest_clean(tmp_path):
    specs = _fake_specs()
    payloads = {s.url: FITS_HEADER + b"\x00" * 32 for s in specs}
    fetch_all(data_raw=tmp_path, specs=specs,
              downloader=_make_fake_downloader(payloads))
    assert verify_manifest(tmp_path) == []


def test_verify_manifest_detects_corruption(tmp_path):
    specs = _fake_specs()
    payloads = {s.url: FITS_HEADER + b"\x00" * 32 for s in specs}
    fetch_all(data_raw=tmp_path, specs=specs,
              downloader=_make_fake_downloader(payloads))

    victim = tmp_path / "COM_CMB_IQU-smica_2048_R3.00_full.fits"
    contents = victim.read_bytes()
    victim.write_bytes(contents[:-1] + b"\x01")  # flip the last byte

    problems = verify_manifest(tmp_path)
    assert any("checksum mismatch" in p for p in problems), problems


def test_verify_manifest_detects_non_fits(tmp_path):
    """
    Simulates the case where the archive returns an HTML error page instead of
    the FITS file — the checksum will be consistent with the HTML page, but the
    FITS magic-byte check should still flag it.
    """
    specs = _fake_specs()[:1]
    html = b"<!DOCTYPE html><html><body>404 Not Found</body></html>"
    fetch_all(data_raw=tmp_path, specs=specs,
              downloader=_make_fake_downloader({specs[0].url: html}))
    problems = verify_manifest(tmp_path)
    assert any("not a FITS file" in p for p in problems), problems


def test_verify_manifest_detects_missing_file(tmp_path):
    specs = _fake_specs()
    payloads = {s.url: FITS_HEADER + b"\x00" * 16 for s in specs}
    fetch_all(data_raw=tmp_path, specs=specs,
              downloader=_make_fake_downloader(payloads))
    (tmp_path / "COM_CMB_IQU-smica_2048_R3.00_full.fits").unlink()
    problems = verify_manifest(tmp_path)
    assert any("missing file" in p for p in problems), problems
