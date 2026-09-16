#!/usr/bin/env python3
"""Accumulate NOAA SWPC wind / IMF / Kp history without dead fallback URLs.

Key changes:
- Uses the current NOAA RTSW wind/mag feeds and the standard planetary Kp product.
- Merges BOTH the newer docs/data/noaa/*.json files and the older
  docs/data/noaa-wind|noaa-imf|noaa-kp/history.json archives.
- Keeps wind/mag for 45 days and Kp for 180 days.
- Never wipes accumulated history because one fetch fails.

Outputs:
- docs/data/noaa/wind_history.json
- docs/data/noaa/mag_history.json
- docs/data/noaa/kp_history.json
- docs/data/noaa/status.json

Compatibility copies:
- docs/data/noaa_wind_history.json
- docs/data/noaa_imf_history.json
- docs/data/noaa_kp_history.json
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
OUT = DATA / "noaa"
OUT.mkdir(parents=True, exist_ok=True)

TIMEOUT = int(os.environ.get("NOAA_FETCH_TIMEOUT", "60"))
KEEP_DAYS = {"wind": 45, "mag": 45, "kp": 180}

URLS = {
    "wind": ["https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"],
    "mag": ["https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"],
    "kp": ["https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"],
}

OUT_FILES = {
    "wind": OUT / "wind_history.json",
    "mag": OUT / "mag_history.json",
    "kp": OUT / "kp_history.json",
}
COMPAT_FILES = {
    "wind": DATA / "noaa_wind_history.json",
    "mag": DATA / "noaa_imf_history.json",
    "kp": DATA / "noaa_kp_history.json",
}
LEGACY_INPUTS = {
    "wind": [DATA / "noaa-wind" / "history.json", DATA / "noaa_wind_history.json"],
    "mag": [DATA / "noaa-imf" / "history.json", DATA / "noaa_imf_history.json"],
    "kp": [DATA / "noaa-kp" / "history.json", DATA / "noaa_kp_history.json"],
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
        v = float(value)
        if v > 10_000_000_000:
            v /= 1000.0
        try:
            return datetime.fromtimestamp(v, timezone.utc)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if len(s) == 16 and s[10] in ("T", " "):
        s = s.replace(" ", "T") + ":00+00:00"
    elif len(s) == 19 and s[10] == " ":
        s = s.replace(" ", "T") + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
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
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def get_first(obj: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in obj and obj[name] not in (None, ""):
            return obj[name]
    return None


def fetch_json(url: str) -> Any:
    headers = {
        "User-Agent": "SWIFT-Space-Weather/2.0 (+GitHub Actions)",
        "Accept": "application/json,text/plain,*/*",
    }
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def table_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "history", "forecast"):
            if isinstance(payload.get(key), list):
                return table_rows(payload[key])
        return [payload]
    if not isinstance(payload, list) or not payload:
        return []
    if isinstance(payload[0], dict):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload[0], list):
        header = [str(x).strip() for x in payload[0]]
        return [
            {header[i] if i < len(header) else f"col{i}": row[i] for i in range(len(row))}
            for row in payload[1:]
            if isinstance(row, list)
        ]
    return []


def normalize(kind: str, payload: Any, source_url: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in table_rows(payload):
        t = parse_time(get_first(r, ["time_tag", "time", "timestamp", "datetime", "date", "created_at", "0"]))
        if not t:
            continue
        if kind == "wind":
            speed = num(get_first(r, ["speed", "proton_speed", "bulk_speed", "solar_wind_speed", "V", "v", "2"]))
            if speed is None:
                continue
            density = num(get_first(r, ["density", "proton_density", "n", "1"]))
            temp = num(get_first(r, ["temperature", "temp", "T", "3"]))
            out.append({
                "time": iso_z(t), "speed": round(speed, 3),
                "density": round(density, 3) if density is not None else None,
                "temperature": round(temp, 3) if temp is not None else None,
                "source": "NOAA_SWPC", "source_url": source_url,
            })
        elif kind == "mag":
            bz = num(get_first(r, ["bz_gsm", "bz", "4"]))
            if bz is None:
                continue
            out.append({
                "time": iso_z(t),
                "bt": num(get_first(r, ["bt", "total_field", "Btotal", "1"])),
                "bx": num(get_first(r, ["bx_gsm", "bx", "2"])),
                "by": num(get_first(r, ["by_gsm", "by", "3"])),
                "bz": round(bz, 3),
                "source": "NOAA_SWPC", "source_url": source_url,
            })
        else:
            kp = num(get_first(r, ["kp_index", "kp", "Kp", "1"]))
            if kp is None or not (0 <= kp <= 9):
                continue
            out.append({
                "time": iso_z(t), "kp": round(kp, 3),
                "source": "NOAA_PLANETARY_K", "source_url": source_url,
            })
    return out


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for key in ("records", "items", "data", "history"):
            if isinstance(obj.get(key), list):
                return [r for r in obj[key] if isinstance(r, dict)]
    return []


def canonicalize(kind: str, r: dict[str, Any]) -> dict[str, Any] | None:
    t = parse_time(r.get("time") or r.get("time_tag") or r.get("timestamp"))
    if not t:
        return None
    if kind == "wind":
        speed = num(r.get("speed") or r.get("proton_speed") or r.get("v"))
        if speed is None:
            return None
        return {
            "time": iso_z(t), "speed": round(speed, 3),
            "density": num(r.get("density") or r.get("proton_density")),
            "temperature": num(r.get("temperature") or r.get("temp")),
            "source": r.get("source") or "NOAA_SWPC",
            "source_url": r.get("source_url"),
        }
    if kind == "mag":
        bz = num(r.get("bz") if r.get("bz") is not None else r.get("bz_gsm"))
        if bz is None:
            return None
        return {
            "time": iso_z(t),
            "bt": num(r.get("bt")),
            "bx": num(r.get("bx") if r.get("bx") is not None else r.get("bx_gsm")),
            "by": num(r.get("by") if r.get("by") is not None else r.get("by_gsm")),
            "bz": round(bz, 3),
            "source": r.get("source") or "NOAA_SWPC",
            "source_url": r.get("source_url"),
        }
    kp = num(r.get("kp") if r.get("kp") is not None else r.get("kp_raw"))
    if kp is None or not (0 <= kp <= 9):
        return None
    return {
        "time": iso_z(t), "kp": round(kp, 3),
        "source": r.get("source") or "NOAA_PLANETARY_K",
        "source_url": r.get("source_url"),
    }


def merge(kind: str, chunks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cutoff = utcnow() - timedelta(days=KEEP_DAYS[kind])
    by_time: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        for raw in chunk:
            r = canonicalize(kind, raw)
            if not r:
                continue
            t = parse_time(r["time"])
            if not t or t < cutoff:
                continue
            by_time[r["time"]] = r
    return [by_time[k] for k in sorted(by_time)]


def coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    ts = [parse_time(r.get("time")) for r in records]
    ts = [x for x in ts if x]
    if not ts:
        return {"count": 0, "days": 0, "start": None, "end": None}
    start, end = min(ts), max(ts)
    return {
        "count": len(records),
        "days": round((end - start).total_seconds() / 86400, 3),
        "start": iso_z(start),
        "end": iso_z(end),
    }


def main() -> None:
    all_status = {"updated_at": iso_z(utcnow()), "products": {}}
    for kind in ("wind", "mag", "kp"):
        new_records: list[dict[str, Any]] = []
        used, errors = [], []
        for url in URLS[kind]:
            try:
                rows = normalize(kind, fetch_json(url), url)
                new_records.extend(rows)
                used.append({"url": url, "records": len(rows)})
            except Exception as e:
                errors.append({"url": url, "error": str(e)[:240]})

        sources = [load_records(OUT_FILES[kind])]
        sources.extend(load_records(p) for p in LEGACY_INPUTS[kind])
        sources.append(new_records)
        merged = merge(kind, sources)
        status = {
            "ok": bool(merged),
            "new_records": len(new_records),
            "merged_records": len(merged),
            "used_urls": used,
            "errors": errors,
            "legacy_sources_merged": [str(p.relative_to(ROOT)) for p in LEGACY_INPUTS[kind] if p.exists()],
        }
        payload = {
            "updated_at": iso_z(utcnow()),
            "kind": kind,
            "retention_days": KEEP_DAYS[kind],
            "status": status,
            "coverage": coverage(merged),
            "records": merged,
        }
        OUT_FILES[kind].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        COMPAT_FILES[kind].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        all_status["products"][kind] = {**status, "coverage": payload["coverage"]}
        print(kind, payload["coverage"])

    (OUT / "status.json").write_text(json.dumps(all_status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
