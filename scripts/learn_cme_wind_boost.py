#!/usr/bin/env python3
"""Learn how much solar wind speed increases after predicted CME arrivals.

Outputs:
- docs/data/swift-wind/cme-boost-model.json
- docs/data/cme-arrivals/latest.json

Model idea:
For each Earth-directed CME arrival candidate, compare the pre-arrival solar wind
baseline with the post-arrival peak speed.
  baseline = median Vsw in [arrival-12h, arrival-3h]
  peak     = max Vsw in [arrival, arrival+18h]
  delta_v  = max(0, peak - baseline)
The resulting delta_v samples are grouped by impact class and speed bin. This is
kept separate from the main solar wind predictor, so CME speed increase can be
learned independently.
"""
from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA = ROOT / "docs" / "data"
OUT_MODEL = DOCS_DATA / "swift-wind" / "cme-boost-model.json"
OUT_ARR = DOCS_DATA / "cme-arrivals" / "latest.json"
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
OUT_ARR.parent.mkdir(parents=True, exist_ok=True)

AU_KM = 149_597_870.0
R_SUN_KM = 695_700.0
R0_KM = 21.5 * R_SUN_KM


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)
    s = str(v).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if len(s) == 16 and s[10] == "T":
        s += ":00+00:00"
    if len(s) == 19 and s[10] == " ":
        s = s.replace(" ", "T") + "+00:00"
    try:
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(str(v), fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def num(v: Any, default: float | None = None) -> float | None:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_records(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for key in ("records", "data", "items", "history", "forecast"):
            if isinstance(obj.get(key), list):
                return [r for r in obj[key] if isinstance(r, dict)]
    return []


def load_wind_history() -> list[dict[str, Any]]:
    paths = [
        DOCS_DATA / "noaa" / "wind_history.json",
        DOCS_DATA / "noaa_wind_history.json",
        DOCS_DATA / "swift-wind" / "history.json",
    ]
    out = []
    for path in paths:
        for r in extract_records(load_json(path)):
            t = parse_time(r.get("time") or r.get("time_tag") or r.get("timestamp"))
            v = num(r.get("speed") or r.get("observed_speed") or r.get("solar_wind_speed") or r.get("v"))
            if t and v is not None:
                out.append({"time": iso_z(t), "_t": t, "speed": v})
    dedup = {}
    for r in out:
        dedup[r["time"]] = r
    return sorted(dedup.values(), key=lambda x: x["_t"])


def activity_time_from_id(s: str) -> datetime | None:
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}(?::\d{2})?)", str(s or ""))
    if not m:
        return None
    ss = f"{m.group(1)}T{m.group(2)}Z"
    return parse_time(ss)


def angular_sep(lon_deg: float, lat_deg: float) -> float:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_d = math.cos(lat) * math.cos(lon)
    cos_d = max(-1.0, min(1.0, cos_d))
    return math.degrees(math.acos(cos_d))


def classify_hit(sep: float, half: float) -> tuple[str, float, float]:
    if half <= 0:
        return "MISS", 0.0, 0.0
    ratio = sep / half
    if ratio <= 0.50:
        return "CORE-HIT", 1.0, 0.95
    if ratio <= 0.85:
        x = (ratio - 0.50) / 0.35
        return "BODY-HIT", 0.85 - 0.25 * x, 0.85 - 0.15 * x
    if ratio <= 1.00:
        x = (ratio - 0.85) / 0.15
        return "FLANK-HIT", 0.55 - 0.25 * x, 0.55 - 0.20 * x
    return "MISS", 0.0, 0.02


def ballistic_arrival(t0: datetime, speed: float) -> tuple[datetime, float]:
    speed = max(250.0, min(3000.0, speed or 500.0))
    hours = (AU_KM - R0_KM) / speed / 3600.0
    hours = max(10.0, min(168.0, hours))
    return t0 + timedelta(hours=hours), hours


def normalize_donki_analysis(c: dict[str, Any]) -> dict[str, Any] | None:
    raw_lon = num(c.get("longitude"), 0.0) or 0.0
    lon = raw_lon  # DONKI positive west, same convention as index.html
    lat = num(c.get("latitude"), 0.0) or 0.0
    speed = num(c.get("speed"), 500.0) or 500.0
    half = num(c.get("halfAngle"), 30.0) or 30.0
    t0 = parse_time(c.get("activityStartTime") or c.get("startTime")) or activity_time_from_id(c.get("activityID") or c.get("associatedCMEID") or c.get("id")) or parse_time(c.get("time21_5"))
    if not t0:
        return None

    enlil_arrival = None
    for e in c.get("enlilList") or []:
        if isinstance(e, dict) and e.get("estimatedShockArrivalTime"):
            enlil_arrival = parse_time(e.get("estimatedShockArrivalTime"))
            if enlil_arrival:
                break

    sep = angular_sep(lon, lat)
    impact, geom, conf = classify_hit(sep, half)
    if enlil_arrival:
        impact = "ENLIL-HIT"
        geom = max(geom, 1.0)
        conf = max(conf, 1.0)
        arrival = enlil_arrival
        transit_hours = (arrival - t0).total_seconds() / 3600.0
    else:
        if impact == "MISS":
            arrival, transit_hours = ballistic_arrival(t0, speed)
        else:
            speed_factor = 1.0 if impact == "CORE-HIT" else 0.9 if impact == "BODY-HIT" else 0.65
            arrival, transit_hours = ballistic_arrival(t0, speed * speed_factor)

    return {
        "id": c.get("activityID") or c.get("associatedCMEID") or c.get("id") or f"DONKI-{iso_z(t0)}",
        "source": "DONKI",
        "impact_class": impact,
        "earth_candidate": impact != "MISS",
        "confidence": round(conf, 3),
        "geometry_multiplier": round(geom, 3),
        "t0": iso_z(t0),
        "arrival_time": iso_z(arrival),
        "transit_hours": round(transit_hours, 2),
        "speed": round(speed, 1),
        "effective_speed": round(speed * max(geom, 0.25), 1),
        "half_angle": round(half, 1),
        "longitude": round(lon, 2),
        "latitude": round(lat, 2),
        "angular_separation": round(sep, 2),
    }


def load_donki_cmes() -> list[dict[str, Any]]:
    paths: list[Path] = []
    index_paths = [DOCS_DATA / "donki" / "index.json", DOCS_DATA / "donki_index.json"]
    for idx_path in index_paths:
        idx = load_json(idx_path)
        if not isinstance(idx, dict):
            continue
        for f in idx.get("files") or []:
            if not isinstance(f, dict):
                continue
            p = f.get("path")
            if not p:
                continue
            pth = (ROOT / p).resolve() if str(p).startswith("./") else (ROOT / str(p)).resolve()
            if pth.exists():
                paths.append(pth)
        if idx.get("latest"):
            paths.append(DOCS_DATA / "donki" / str(idx["latest"]))
    paths.extend((DOCS_DATA / "donki").glob("*.json"))

    out: list[dict[str, Any]] = []
    seen_paths = set()
    for path in paths:
        if path in seen_paths or not path.exists() or path.name == "index.json":
            continue
        seen_paths.add(path)
        obj = load_json(path)
        if not isinstance(obj, dict):
            continue
        analyses = obj.get("sources", {}).get("cmeAnalysis", {}).get("data", [])
        cmes = obj.get("sources", {}).get("cme", {}).get("data", [])
        for a in analyses if isinstance(analyses, list) else []:
            if isinstance(a, dict):
                item = normalize_donki_analysis(a)
                if item:
                    out.append(item)
        for cme in cmes if isinstance(cmes, list) else []:
            if not isinstance(cme, dict):
                continue
            analyses2 = cme.get("cmeAnalyses") or []
            if analyses2:
                for a in analyses2:
                    if isinstance(a, dict):
                        aa = dict(a)
                        aa.setdefault("activityID", cme.get("activityID"))
                        aa.setdefault("activityStartTime", cme.get("startTime"))
                        item = normalize_donki_analysis(aa)
                        if item:
                            out.append(item)
    # Deduplicate by ID + arrival hour.
    dedup = {}
    for c in out:
        at = parse_time(c.get("arrival_time"))
        if not at:
            continue
        key = f"{c.get('id')}|{at.strftime('%Y%m%d%H')}"
        dedup[key] = c
    return sorted(dedup.values(), key=lambda x: x["arrival_time"])


def window_values(wind: list[dict[str, Any]], start: datetime, end: datetime) -> list[float]:
    return [float(r["speed"]) for r in wind if start <= r["_t"] <= end and num(r.get("speed")) is not None]


def learn_samples(cmes: list[dict[str, Any]], wind: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    now = utcnow()
    for c in cmes:
        arrival = parse_time(c.get("arrival_time"))
        if not arrival or arrival > now - timedelta(hours=2):
            continue
        pre = window_values(wind, arrival - timedelta(hours=12), arrival - timedelta(hours=3))
        post = window_values(wind, arrival, arrival + timedelta(hours=18))
        if len(pre) < 3 or len(post) < 3:
            continue
        baseline = statistics.median(pre)
        peak = max(post)
        delta = max(0.0, peak - baseline)
        sample = dict(c)
        sample.update({
            "baseline_speed": round(baseline, 2),
            "post_peak_speed": round(peak, 2),
            "observed_delta_v": round(delta, 2),
            "pre_count": len(pre),
            "post_count": len(post),
        })
        samples.append(sample)
    return samples


def median_safe(vals: list[float], default: float) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.median(vals) if vals else default


def build_model(samples: list[dict[str, Any]]) -> dict[str, Any]:
    classes = ["ENLIL-HIT", "CORE-HIT", "BODY-HIT", "FLANK-HIT", "CACTUS-WIDE-HALO", "CACTUS-BROAD"]
    default_by_class = {
        "ENLIL-HIT": 160.0,
        "CORE-HIT": 150.0,
        "BODY-HIT": 90.0,
        "FLANK-HIT": 45.0,
        "CACTUS-WIDE-HALO": 60.0,
        "CACTUS-BROAD": 35.0,
    }
    by_class = {}
    for cls in classes:
        vals = [num(s.get("observed_delta_v"), 0.0) or 0.0 for s in samples if s.get("impact_class") == cls]
        by_class[cls] = {
            "count": len(vals),
            "median_delta_v": round(median_safe(vals, default_by_class[cls]), 2),
            "mean_delta_v": round(sum(vals) / len(vals), 2) if vals else default_by_class[cls],
            "default_used": not bool(vals),
        }

    # Simple global slope delta_v ~ intercept + slope*(speed-500) + geometry_factor.
    usable = [(num(s.get("speed")), num(s.get("geometry_multiplier")), num(s.get("observed_delta_v"))) for s in samples]
    usable = [(v, g, d) for v, g, d in usable if v is not None and g is not None and d is not None]
    slope = 0.10
    intercept = 40.0
    if len(usable) >= 3:
        xs = [(v - 500.0) * max(g, 0.1) for v, g, _ in usable]
        ys = [d for _, _, d in usable]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            intercept = my - slope * mx
            slope = max(-0.1, min(0.5, slope))
            intercept = max(0.0, min(220.0, intercept))

    return {
        "updated_at": iso_z(utcnow()),
        "model": "SWIFT CME Wind Boost Learner",
        "definition": "observed_delta_v = max(0, post_arrival_peak_speed - pre_arrival_median_speed)",
        "windows": {
            "pre_baseline": "arrival-12h to arrival-3h median",
            "post_peak": "arrival to arrival+18h max",
        },
        "sample_count": len(samples),
        "global": {
            "intercept": round(intercept, 4),
            "speed_geometry_slope": round(slope, 6),
            "formula": "delta_v = class_median*0.55 + max(0, intercept + slope*(speed-500)*geometry)*0.45",
        },
        "by_impact_class": by_class,
        "samples": samples[-200:],
    }


def expected_boost(c: dict[str, Any], model: dict[str, Any]) -> float:
    cls = str(c.get("impact_class") or "BODY-HIT")
    klass = model.get("by_impact_class", {}).get(cls, {})
    class_med = num(klass.get("median_delta_v"), 70.0) or 70.0
    glob = model.get("global", {})
    intercept = num(glob.get("intercept"), 40.0) or 40.0
    slope = num(glob.get("speed_geometry_slope"), 0.10) or 0.10
    speed = num(c.get("speed"), 500.0) or 500.0
    geom = max(0.1, num(c.get("geometry_multiplier"), 0.5) or 0.5)
    pred = 0.55 * class_med + 0.45 * max(0.0, intercept + slope * (speed - 500.0) * geom)
    return max(0.0, min(450.0, pred))


def main() -> None:
    wind = load_wind_history()
    cmes = load_donki_cmes()
    samples = learn_samples(cmes, wind)
    model = build_model(samples)
    for c in cmes:
        c["expected_wind_speed_increase"] = round(expected_boost(c, model), 2)
    arr_payload = {
        "updated_at": iso_z(utcnow()),
        "source": "DONKI cache + learned CME wind boost",
        "count": len(cmes),
        "arrivals": cmes[-300:],
    }
    OUT_MODEL.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_ARR.write_text(json.dumps(arr_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CME arrivals={len(cmes)} samples={len(samples)} wind_history={len(wind)}")


if __name__ == "__main__":
    main()
