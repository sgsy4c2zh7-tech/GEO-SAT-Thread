#!/usr/bin/env python3
"""Fetch official NOAA/NCEP WSA-ENLIL data for SWIFT.

Products used
-------------
1) NOAA SWPC Earth/L1 WSA-ENLIL time series:
   https://services.swpc.noaa.gov/json/enlil_time_series.json
   Fields include time_tag, v_r, earth_particles_per_cm3 and cloud.

2) NCEP NOMADS WSA inner-boundary velocity FITS at 21.5 R_sun:
   https://nomads.ncep.noaa.gov/pub/data/nccf/com/wsa_enlil/prod/
   The latest wsa_vel_21.5rs_*_gong.fits file is reduced to a compact
   ecliptic/longitude speed boundary for browser visualization.

The public SWPC JSON is a 1-D L1 time series, not the full 3-D ENLIL grid.
The FITS boundary is real WSA model input at 21.5 R_sun.  SWIFT uses these
real operational model products without storing the ~131 MB suball.nc file.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "enlil"
OUT.mkdir(parents=True, exist_ok=True)
LATEST = OUT / "latest.json"
ARCHIVE = OUT / "forecast-archive.json"
BOUNDARY = OUT / "wsa_boundary.json"
INDEX = OUT / "index.json"

SWPC_ENLIL = "https://services.swpc.noaa.gov/json/enlil_time_series.json"
NOMADS_ROOT = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/wsa_enlil/prod"
UA = "SWIFT-Space-Weather-Research/1.3 (+GitHub Actions)"
TIMEOUT = 75
ARCHIVE_KEEP_DAYS = 120


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(v: Any) -> datetime | None:
    if not v:
        return None
    s = str(v).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if len(s) == 19 and s[10] == " ":
        s = s.replace(" ", "T") + "+00:00"
    try:
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        return None


def fnum(v: Any, default: float | None = None) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": UA, "Accept": "*/*"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_enlil_timeseries(now: datetime) -> dict[str, Any]:
    raw = get(SWPC_ENLIL).json()
    rows = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict):
            continue
        t = parse_time(r.get("time_tag") or r.get("time"))
        v = fnum(r.get("v_r") or r.get("speed"))
        n = fnum(r.get("earth_particles_per_cm3") or r.get("density"))
        if not t or v is None:
            continue
        rows.append({
            "time": iso_z(t),
            "v_r": round(v, 3),
            "earth_particles_per_cm3": round(n, 4) if n is not None else None,
            "cloud": r.get("cloud"),
            "source": "NOAA_SWPC_WSA_ENLIL_L1",
        })
    rows.sort(key=lambda x: x["time"])
    future = [r for r in rows if parse_time(r["time"]) and parse_time(r["time"]) >= now - timedelta(hours=1)]
    return {
        "source_url": SWPC_ENLIL,
        "records": rows,
        "forecast": future,
        "count": len(rows),
        "forecast_count": len(future),
    }


def discover_latest_wsa_fits(now: datetime) -> tuple[str, str]:
    # Search today and yesterday to survive publication delays around 00 UTC.
    candidates: list[tuple[datetime, str, str]] = []
    for d in [now.date(), (now - timedelta(days=1)).date()]:
        day = d.strftime("%Y%m%d")
        url = f"{NOMADS_ROOT}/wsa_enlil.{day}/"
        try:
            text = get(url).text
        except Exception:
            continue
        for name in re.findall(r'href=["\']([^"\']+\.fits)["\']', text, flags=re.I):
            # Ambient bi-hourly boundary files plus possible mrid-based CME runs.
            if "wsa_vel_21.5rs" not in name.lower():
                continue
            m = re.search(r"_(\d{2})_gong(?:_z)?\.fits$", name, flags=re.I)
            hour = int(m.group(1)) if m else 0
            ts = datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc)
            candidates.append((ts, url + name, name))
    if not candidates:
        raise RuntimeError("No WSA 21.5 R_sun FITS found on NOMADS")
    _, url, name = max(candidates, key=lambda x: x[0])
    return url, name


def reduce_fits_to_boundary(blob: bytes, source_url: str, source_name: str, now: datetime) -> dict[str, Any]:
    with fits.open(BytesIO(blob), memmap=False) as hdul:
        arrays = []
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if isinstance(data, np.ndarray) and data.ndim >= 2 and np.issubdtype(data.dtype, np.number):
                arr = np.asarray(data, dtype=float)
                while arr.ndim > 2:
                    arr = arr[0]
                arrays.append((arr.size, arr, dict(hdu.header)))
        if not arrays:
            raise RuntimeError("No numeric 2-D image found in WSA FITS")
        _, arr, header = max(arrays, key=lambda x: x[0])

    arr = np.where(np.isfinite(arr), arr, np.nan)
    # Astropy applies BSCALE/BZERO when reading.  Select likely physical speed image.
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise RuntimeError("WSA FITS contains no finite values")

    # Orient longitude along the longest axis in the typical WSA lat x lon grid.
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    nlat, nlon = arr.shape
    lat = np.linspace(-90.0, 90.0, nlat)
    eq = np.abs(lat) <= 12.0
    if eq.sum() < 1:
        eq = np.ones(nlat, dtype=bool)
    profile = np.nanmedian(arr[eq, :], axis=0)

    # If the selected image is not physically plausible, find a simple linear scale
    # only when metadata suggests such a scale.  Normally astropy already applied it.
    med = float(np.nanmedian(profile))
    if not (150 <= med <= 1800):
        # A second heuristic for files containing m/s rather than km/s.
        if 150000 <= med <= 1800000:
            profile = profile / 1000.0
        else:
            raise RuntimeError(f"Selected WSA FITS image does not look like km/s (median={med:.3g})")

    # Downsample to 180 longitude sectors by circular interpolation.
    src_x = np.linspace(0.0, 360.0, nlon, endpoint=False)
    dst_x = np.linspace(0.0, 360.0, 180, endpoint=False)
    ext_x = np.r_[src_x, 360.0]
    ext_y = np.r_[profile, profile[0]]
    out = np.interp(dst_x, ext_x, ext_y)
    out = np.clip(out, 150, 1800)

    return {
        "updated_at": iso_z(now),
        "source": "NCEP_WSA_21.5Rs_GONG",
        "source_url": source_url,
        "source_file": source_name,
        "inner_boundary_rsun": 21.5,
        "latitude_band_deg": [-12, 12],
        "method": "median ecliptic latitude band from operational WSA FITS; circularly resampled to 180 longitudes",
        "grid_shape_original": [int(nlat), int(nlon)],
        "header_excerpt": {k: header.get(k) for k in ("DATE", "DATE-OBS", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "BUNIT") if k in header},
        "speed_by_longitude": [
            {"longitude_deg": round(float(lon), 3), "speed_km_s": round(float(v), 2)}
            for lon, v in zip(dst_x, out)
        ],
        "stats": {
            "min_km_s": round(float(np.nanmin(out)), 2),
            "median_km_s": round(float(np.nanmedian(out)), 2),
            "max_km_s": round(float(np.nanmax(out)), 2),
        },
    }


def main() -> None:
    now = utcnow()
    errors = []
    ts = {"records": [], "forecast": [], "count": 0, "forecast_count": 0, "source_url": SWPC_ENLIL}
    try:
        ts = fetch_enlil_timeseries(now)
    except Exception as e:
        errors.append({"product": "enlil_time_series", "error": str(e)[:300]})

    boundary = None
    try:
        url, name = discover_latest_wsa_fits(now)
        boundary = reduce_fits_to_boundary(get(url).content, url, name, now)
        save_json(BOUNDARY, boundary)
    except Exception as e:
        errors.append({"product": "wsa_21.5rs_fits", "error": str(e)[:300]})
        boundary = load_json(BOUNDARY, None)

    latest = {
        "updated_at": iso_z(now),
        "model": "NOAA/NCEP WSA-ENLIL operational ingest",
        "time_series_source": ts.get("source_url"),
        "wsa_boundary_file": "wsa_boundary.json" if boundary else None,
        "records": ts.get("records", []),
        "forecast": ts.get("forecast", []),
        "errors": errors,
        "status": "ok" if ts.get("records") else "warning",
    }
    save_json(LATEST, latest)

    # Compact run archive.  Keep only the L1/Earth-line rows, not binary model files.
    old = load_json(ARCHIVE, {"items": []}) or {"items": []}
    items = [x for x in old.get("items", []) if isinstance(x, dict)]
    issued = iso_z(now)
    for r in ts.get("forecast", []):
        t = parse_time(r.get("time"))
        if not t:
            continue
        items.append({
            "issued_at": issued,
            "target_time": r["time"],
            "lead_hours": round((t - now).total_seconds() / 3600, 2),
            "v_r": r.get("v_r"),
            "earth_particles_per_cm3": r.get("earth_particles_per_cm3"),
            "cloud": r.get("cloud"),
        })
    cutoff = now - timedelta(days=ARCHIVE_KEEP_DAYS)
    dedup = {}
    for x in items:
        it = parse_time(x.get("issued_at"))
        if not it or it < cutoff:
            continue
        key = (x.get("issued_at"), x.get("target_time"))
        dedup[key] = x
    items = sorted(dedup.values(), key=lambda x: (x.get("issued_at", ""), x.get("target_time", "")))
    save_json(ARCHIVE, {"updated_at": iso_z(now), "retention_days": ARCHIVE_KEEP_DAYS, "items": items})

    save_json(INDEX, {
        "updated_at": iso_z(now),
        "latest": "latest.json",
        "forecast_archive": "forecast-archive.json",
        "wsa_boundary": "wsa_boundary.json" if boundary else None,
        "sources": [SWPC_ENLIL, NOMADS_ROOT],
        "errors": errors,
    })
    print(json.dumps({"enlil_rows": len(ts.get("records", [])), "forecast_rows": len(ts.get("forecast", [])), "wsa_boundary": bool(boundary), "errors": errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
