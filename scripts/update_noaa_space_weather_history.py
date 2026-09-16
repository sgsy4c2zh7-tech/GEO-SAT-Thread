#!/usr/bin/env python3
"""Update NOAA solar wind / IMF / Kp history for SWIFT reports.

Outputs:
- docs/data/noaa/wind_history.json
- docs/data/noaa/mag_history.json
- docs/data/noaa/kp_history.json
- docs/data/noaa/status.json

Compatibility copies:
- docs/data/noaa_wind_history.json
- docs/data/noaa_imf_history.json
- docs/data/noaa_kp_history.json

Why this script exists:
SWPC JSON product formats occasionally change and real-time streams can be interrupted.
This script keeps the old local history, appends any newly available records, and never
wipes the archive just because a current download fails.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "noaa"
OUT.mkdir(parents=True, exist_ok=True)
LEGACY = ROOT / "docs" / "data"
LEGACY.mkdir(parents=True, exist_ok=True)
KEEP_DAYS = int(os.environ.get("NOAA_HISTORY_KEEP_DAYS", "31"))
TIMEOUT = int(os.environ.get("NOAA_FETCH_TIMEOUT", "60"))

URLS = {
    "wind": [
        "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
        "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json",
        "https://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json",
    ],
    "mag": [
        "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
        "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json",
        "https://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json",
    ],
    "kp": [
        "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
        "https://services.swpc.noaa.gov/products/noaa-estimated-planetary-k-index-1-minute.json",
    ],
}

OUT_FILES = {
    "wind": OUT / "wind_history.json",
    "mag": OUT / "mag_history.json",
    "kp": OUT / "kp_history.json",
}
LEGACY_FILES = {
    "wind": LEGACY / "noaa_wind_history.json",
    "mag": LEGACY / "noaa_imf_history.json",
    "kp": LEGACY / "noaa_kp_history.json",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Some feeds use unix seconds, others milliseconds.
        v = float(value)
        if v > 10_000_000_000:
            v /= 1000.0
        try:
            return datetime.fromtimestamp(v, timezone.utc)
        except Exception:
            return None
    s = str(value).strip()
    if not s or s.lower() in {"time_tag", "time", "datetime", "date"}:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if len(s) == 16 and s[10] == "T":
        s += ":00+00:00"
    if len(s) == 16 and s[10] == " ":
        s = s.replace(" ", "T") + ":00+00:00"
    if len(s) == 19 and s[10] == " ":
        s = s.replace(" ", "T") + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    # Common SWPC table format: 2026-09-16 09:00:00.000
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(str(value), fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def num(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def get_first(obj: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in obj and obj[name] not in (None, ""):
            return obj[name]
    return None


def fetch_json(url: str) -> Any:
    headers = {
        "User-Agent": "SWIFT-Space-Weather-Report/1.0 (+GitHub Actions)",
        "Accept": "application/json,text/plain,*/*",
    }
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def normalize_table_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "history", "forecast"):
            if isinstance(payload.get(key), list):
                return normalize_table_rows(payload[key])
        return [payload]
    if not isinstance(payload, list):
        return []
    if not payload:
        return []
    if isinstance(payload[0], dict):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload[0], list):
        header = [str(x).strip() for x in payload[0]]
        rows = []
        for raw in payload[1:]:
            if not isinstance(raw, list):
                continue
            rows.append({header[i] if i < len(header) else f"col{i}": raw[i] for i in range(len(raw))})
        return rows
    return []


def normalize_wind(payload: Any, source_url: str) -> list[dict[str, Any]]:
    out = []
    for r in normalize_table_rows(payload):
        t = parse_time(get_first(r, ["time_tag", "time", "timestamp", "datetime", "date", "created_at", "0"]))
        speed = num(get_first(r, ["speed", "proton_speed", "bulk_speed", "solar_wind_speed", "V", "v", "2"]))
        density = num(get_first(r, ["density", "proton_density", "n", "1"]))
        temperature = num(get_first(r, ["temperature", "temp", "T", "3"]))
        if not t or speed is None:
            continue
        out.append({
            "time": iso_z(t),
            "speed": round(speed, 3),
            "density": round(density, 3) if density is not None else None,
            "temperature": round(temperature, 3) if temperature is not None else None,
            "source": "NOAA_SWPC",
            "source_url": source_url,
        })
    return out


def normalize_mag(payload: Any, source_url: str) -> list[dict[str, Any]]:
    out = []
    for r in normalize_table_rows(payload):
        t = parse_time(get_first(r, ["time_tag", "time", "timestamp", "datetime", "date", "created_at", "0"]))
        bt = num(get_first(r, ["bt", "total_field", "Btotal", "6", "1"]))
        bx = num(get_first(r, ["bx_gsm", "bx", "2"]))
        by = num(get_first(r, ["by_gsm", "by", "3"]))
        bz = num(get_first(r, ["bz_gsm", "bz", "4"]))
        phi = num(get_first(r, ["phi_gsm", "phi", "5"]))
        theta = num(get_first(r, ["theta_gsm", "theta", "6"]))
        if not t or bz is None:
            continue
        out.append({
            "time": iso_z(t),
            "bt": round(bt, 3) if bt is not None else None,
            "bx": round(bx, 3) if bx is not None else None,
            "by": round(by, 3) if by is not None else None,
            "bz": round(bz, 3),
            "phi": round(phi, 3) if phi is not None else None,
            "theta": round(theta, 3) if theta is not None else None,
            "source": "NOAA_SWPC",
            "source_url": source_url,
        })
    return out


def normalize_kp(payload: Any, source_url: str) -> list[dict[str, Any]]:
    out = []
    for r in normalize_table_rows(payload):
        t = parse_time(get_first(r, ["time_tag", "time", "timestamp", "datetime", "date", "created_at", "0"]))
        kp = num(get_first(r, ["kp_index", "kp", "estimated_kp", "Kp", "1"]))
        if not t or kp is None:
            continue
        out.append({
            "time": iso_z(t),
            "kp": round(kp, 3),
            "source": "NOAA_SWPC",
            "source_url": source_url,
        })
    return out


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    for key in ("records", "data", "items", "history"):
        if isinstance(obj.get(key), list):
            return [r for r in obj[key] if isinstance(r, dict)]
    return []


def merge_records(existing: list[dict[str, Any]], new: list[dict[str, Any]], keep_days: int) -> list[dict[str, Any]]:
    cutoff = utcnow() - timedelta(days=keep_days)
    by_time: dict[str, dict[str, Any]] = {}
    for r in existing + new:
        t = parse_time(r.get("time") or r.get("time_tag") or r.get("timestamp"))
        if not t or t < cutoff:
            continue
        key = iso_z(t)
        item = dict(r)
        item["time"] = key
        by_time[key] = item
    return [by_time[k] for k in sorted(by_time.keys())]


def coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    times = [parse_time(r.get("time")) for r in records]
    times = [t for t in times if t]
    if not times:
        return {"count": 0, "days": 0, "start": None, "end": None}
    start, end = min(times), max(times)
    return {
        "count": len(times),
        "days": round(max(0.0, (end - start).total_seconds() / 86400.0), 3),
        "start": iso_z(start),
        "end": iso_z(end),
    }


def write_history(kind: str, records: list[dict[str, Any]], status: dict[str, Any]) -> None:
    payload = {
        "updated_at": iso_z(utcnow()),
        "kind": kind,
        "retention_days": KEEP_DAYS,
        "status": status,
        "coverage": coverage(records),
        "records": records,
    }
    OUT_FILES[kind].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LEGACY_FILES[kind].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_kind(kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalizer = {"wind": normalize_wind, "mag": normalize_mag, "kp": normalize_kp}[kind]
    new_records: list[dict[str, Any]] = []
    errors = []
    used_urls = []
    for url in URLS[kind]:
        try:
            payload = fetch_json(url)
            rows = normalizer(payload, url)
            if rows:
                new_records.extend(rows)
                used_urls.append({"url": url, "records": len(rows)})
        except Exception as e:
            errors.append({"url": url, "error": str(e)[:240]})
    existing = load_existing(OUT_FILES[kind]) or load_existing(LEGACY_FILES[kind])
    merged = merge_records(existing, new_records, KEEP_DAYS)
    status = {
        "ok": bool(new_records) or bool(existing),
        "new_records": len(new_records),
        "existing_records": len(existing),
        "merged_records": len(merged),
        "used_urls": used_urls,
        "errors": errors,
    }
    return merged, status


def main() -> None:
    all_status = {"updated_at": iso_z(utcnow()), "keep_days": KEEP_DAYS, "products": {}}
    for kind in ("wind", "mag", "kp"):
        records, status = collect_kind(kind)
        write_history(kind, records, status)
        all_status["products"][kind] = {**status, "coverage": coverage(records)}
        print(f"{kind}: new={status['new_records']} merged={len(records)} errors={len(status['errors'])}")
    (OUT / "status.json").write_text(json.dumps(all_status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
