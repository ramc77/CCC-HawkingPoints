"""
Phase 1: fetch Planck + WMAP CMB maps and masks into data/raw/.

Sources
-------
- Planck PR3 component-separated maps (SMICA, NILC, SEVEM, Commander) and the
  common confidence mask, from the Planck Legacy Archive (PLA).
- WMAP 9-year foreground-reduced ILC temperature map and temperature analysis
  mask, from LAMBDA (NASA GSFC).

Every download writes an entry into data/raw/MANIFEST.yaml recording the exact
URL, release, filename, SHA-256, byte size, and download date, so a re-run is
reproducible. `verify_manifest` re-checks checksums without touching the network
— that is what the unit tests call.

Exact URLs are release-dependent. The catalogue below encodes the Planck PR3
2018 / WMAP 9-year release URLs. If a URL 404s, update the entry and re-run —
do not silently substitute a different file.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional
import hashlib
import urllib.request
import yaml


# Primary source: IRSA (NASA/IPAC) mirror of Planck PR3, direct HTTP paths.
# The Planck Legacy Archive AIO endpoint at pla.esac.esa.int has been unreliable
# (500 errors from LDAP-backend misconfig, 2026-07). If IRSA also goes down,
# fetch maps manually from the PLA web UI at https://pla.esac.esa.int and drop
# the FITS files directly into data/raw/ — run_real_data.py takes any local FITS.
IRSA_PR3_CMB_BASE = ("https://irsa.ipac.caltech.edu/data/Planck/release_3/"
                     "all-sky-maps/maps/component-maps/cmb/")
IRSA_PR3_MASKS_BASE = ("https://irsa.ipac.caltech.edu/data/Planck/release_3/"
                        "ancillary-data/masks/")
LAMBDA_BASE = "https://lambda.gsfc.nasa.gov/data/map/dr5/dfp/ilc/"


@dataclass(frozen=True)
class MapSpec:
    key: str                # short id used as manifest key + filename stem
    url: str
    release: str            # e.g. "Planck PR3 (2018)" — human-readable, goes into manifest
    kind: str               # "map" | "mask"


PLANCK_PR3_MAPS: tuple[MapSpec, ...] = (
    MapSpec(
        key="planck_smica",
        url=IRSA_PR3_CMB_BASE + "COM_CMB_IQU-smica_2048_R3.00_full.fits",
        release="Planck PR3 (2018)",
        kind="map",
    ),
    MapSpec(
        key="planck_nilc",
        url=IRSA_PR3_CMB_BASE + "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
        release="Planck PR3 (2018)",
        kind="map",
    ),
    MapSpec(
        key="planck_sevem",
        url=IRSA_PR3_CMB_BASE + "COM_CMB_IQU-sevem_2048_R3.00_full.fits",
        release="Planck PR3 (2018)",
        kind="map",
    ),
    MapSpec(
        key="planck_commander",
        url=IRSA_PR3_CMB_BASE + "COM_CMB_IQU-commander_2048_R3.00_full.fits",
        release="Planck PR3 (2018)",
        kind="map",
    ),
    MapSpec(
        key="planck_common_mask",
        url=IRSA_PR3_MASKS_BASE + "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
        release="Planck PR3 (2018)",
        kind="mask",
    ),
)

WMAP9_MAPS: tuple[MapSpec, ...] = (
    MapSpec(
        key="wmap9_ilc",
        url=LAMBDA_BASE + "wmap_ilc_9yr_v5.fits",
        release="WMAP 9-year (2013)",
        kind="map",
    ),
    MapSpec(
        key="wmap9_temperature_mask",
        url=LAMBDA_BASE + "wmap_temperature_kq85_analysis_mask_r9_9yr_v5.fits",
        release="WMAP 9-year (2013)",
        kind="mask",
    ),
)

ALL_MAPS: tuple[MapSpec, ...] = PLANCK_PR3_MAPS + WMAP9_MAPS


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # urllib is fine — these are static ~100 MB files, no auth. For very large
    # downloads a resumable client would be nicer; not worth the extra dep here.
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)


def _default_filename(spec: MapSpec) -> str:
    # Preserve the archive's filename when possible — it encodes release, nside,
    # component method. Fall back to the manifest key if the URL has no basename.
    tail = spec.url.rsplit("/", 1)[-1].split("=", 1)[-1]
    return tail or f"{spec.key}.fits"


def fetch_all(data_raw: Path | str = "data/raw",
              specs: Iterable[MapSpec] = ALL_MAPS,
              downloader=_download,
              today: Optional[date] = None) -> Path:
    """
    Download every MapSpec into `data_raw/`, then write MANIFEST.yaml.

    `downloader` and `today` are injectable so the unit test can drive the whole
    flow with local fixtures — see tests/test_fetch_data.py. In normal use both
    fall through to the real network / real clock.
    """
    data_raw = Path(data_raw)
    data_raw.mkdir(parents=True, exist_ok=True)

    entries = []
    for spec in specs:
        filename = _default_filename(spec)
        dest = data_raw / filename
        if not dest.exists():
            downloader(spec.url, dest)
        entries.append({
            "key": spec.key,
            "filename": filename,
            "url": spec.url,
            "release": spec.release,
            "kind": spec.kind,
            "sha256": _sha256(dest),
            "bytes": dest.stat().st_size,
            "downloaded_on": (today or date.today()).isoformat(),
        })

    manifest_path = data_raw / "MANIFEST.yaml"
    with manifest_path.open("w") as f:
        yaml.safe_dump({"entries": entries}, f, sort_keys=False)
    return manifest_path


def verify_manifest(data_raw: Path | str = "data/raw") -> list[str]:
    """
    Re-hash every file listed in data_raw/MANIFEST.yaml and compare against the
    recorded sha256. Also checks the FITS magic bytes (`SIMPLE  =` at offset 0)
    so a truncated download is caught even if a stale manifest happens to match.

    Returns a list of human-readable problem strings — empty list == all good.
    Never touches the network. Safe to run in CI.
    """
    data_raw = Path(data_raw)
    manifest = data_raw / "MANIFEST.yaml"
    if not manifest.exists():
        return [f"missing manifest: {manifest}"]

    with manifest.open() as f:
        entries = yaml.safe_load(f).get("entries", [])

    problems: list[str] = []
    for e in entries:
        path = data_raw / e["filename"]
        if not path.exists():
            problems.append(f"missing file: {path}")
            continue
        actual_hash = _sha256(path)
        if actual_hash != e["sha256"]:
            problems.append(f"checksum mismatch on {path.name}: "
                             f"expected {e['sha256']}, got {actual_hash}")
        with path.open("rb") as f:
            head = f.read(9)
        if head != b"SIMPLE  =":
            problems.append(f"not a FITS file: {path.name} (bad magic {head!r})")
    return problems


if __name__ == "__main__":
    manifest_path = fetch_all()
    print(f"Wrote {manifest_path}")
    for p in verify_manifest():
        print("PROBLEM:", p)
