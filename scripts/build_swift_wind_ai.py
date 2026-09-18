#!/usr/bin/env python3
"""Build the SWIFT Solar Wind AI forecast used by the dashboard and Excel report.

This script intentionally writes BOTH `records` and `forecast` into
`docs/data/swift-wind/latest.json`.

Why:
- The dashboard loader expects `latest.json.records`.
- Older/nonlinear SWIFT Wind workflows wrote only `latest.json.forecast`.
- That schema mismatch forced the dashboard into its browser nonlinear fallback,
  even when the Python forecast itself had been generated correctly.

Inputs (first available / merged):
- docs/data/noaa/wind_history.json
- docs/data/noaa_wind_history.json
- docs/data/noaa-wind/history.json
- live NOAA SWPC RTSW / plasma products (best effort)
- docs/data/wsa/index.json + latest WSA file (optional)
- docs/data/cme-arrivals/latest.json (preferred CME arrival list)
- docs/data/swift-wind/cme-boost-model.json (independent learned CME delta-V)

Outputs:
- docs/data/swift-wind/latest.json
- docs/data/swift-wind/index.json
- docs/data/swift-wind/coefficients.json
- docs/data/swift-wind/verification.json
- docs/data/swift-wind/forecast-archive.json
- docs/data/swift-wind/accuracy.json
- docs/data/swift-wind/accuracy-history.json
- docs/data/swift-wind/history.json   (accuracy compatibility alias)

Main accuracy definition shown as "Wind AI accuracy":
- hit rate within ±50 km/s over the evaluation window.
"""
from __future__ import annotations

import json
import math
import statistics
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA = ROOT / "docs" / "data"
OUT = DOCS_DATA / "swift-wind"
OUT.mkdir(parents=True, exist_ok=True)

LATEST = OUT / "latest.json"
INDEX = OUT / "index.json"
COEFFICIENTS = OUT / "coefficients.json"
VERIFICATION = OUT / "verification.json"
ARCHIVE = OUT / "forecast-archive.json"
ACCURACY = OUT / "accuracy.json"
ACCURACY_HISTORY = OUT / "accuracy-history.json"
ACCURACY_HISTORY_ALIAS = OUT / "history.json"
CME_BOOST_MODEL = OUT / "cme-boost-model.json"
CME_ARRIVALS = DOCS_DATA / "cme-arrivals" / "latest.json"

NOAA_HISTORY_PATHS = [
    DOCS_DATA / "noaa" / "wind_history.json",
    DOCS_DATA / "noaa_wind_history.json",
    DOCS_DATA / "noaa-wind" / "history.json",
]

NOAA_LIVE_URLS = [
    "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
]

NOW = datetime.now(timezone.utc)
FORECAST_HOURS = 120
PAST_RECORD_HOURS = 72
STEP_HOURS = 1
KEEP_ARCHIVE = 12000


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
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
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(str(value), fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def num(v: Any, default: float | None = None) -> float | None:
    if v is None or v == "":
        return default
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def median(values: Iterable[float], default: float | None = None) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(vals) if vals else default


def mean(values: Iterable[float], default: float | None = None) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return (sum(vals) / len(vals)) if vals else default


def sigmoid(x: float) -> float:
    if x >= 60:
        return 1.0
    if x <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"load failed: {path}: {exc}")
        return default


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", path)


def normalize_table_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "history", "forecast"):
            if isinstance(payload.get(key), list):
                return normalize_table_rows(payload[key])
        return [payload]
    if not isinstance(payload, list) or not payload:
        return []
    if isinstance(payload[0], dict):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload[0], list):
        header = [str(x).strip() for x in payload[0]]
        out = []
        for row in payload[1:]:
            if not isinstance(row, list):
                continue
            out.append({header[i] if i < len(header) else f"col{i}": row[i] for i in range(len(row))})
        return out
    return []


def first_value(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def normalize_wind(payload: Any, source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in normalize_table_rows(payload):
        t = parse_time(first_value(row, ("time_tag", "time", "timestamp", "datetime", "date", "0")))
        speed = num(first_value(row, ("speed", "proton_speed", "bulk_speed", "solar_wind_speed", "v", "V", "2")))
        density = num(first_value(row, ("density", "proton_density", "n", "1")))
        if not t or speed is None or not (200 <= speed <= 1200):
            continue
        out.append({
            "time": iso_z(t),
            "_t": t,
            "speed": float(speed),
            "density": float(density) if density is not None else None,
            "source": source,
        })
    return out


def fetch_json(url: str, timeout: int = 45) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SWIFT-Wind-AI/0.6 (+GitHub Actions)",
            "Accept": "application/json,text/plain,*/*",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def load_wind_observations() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []

    for path in NOAA_HISTORY_PATHS:
        obj = load_json(path)
        if obj is None:
            continue
        nrows = normalize_wind(obj, f"local:{path.relative_to(ROOT)}")
        if nrows:
            rows.extend(nrows)
            sources.append(str(path.relative_to(ROOT)))

    live_errors = []
    for url in NOAA_LIVE_URLS:
        try:
            payload = fetch_json(url)
            nrows = normalize_wind(payload, url)
            if nrows:
                rows.extend(nrows)
                sources.append(url)
                # One healthy real-time source is enough; keep going only if tiny.
                if len(nrows) > 200:
                    break
        except Exception as exc:
            live_errors.append(f"{url}: {exc}")

    dedup: dict[str, dict[str, Any]] = {}
    cutoff = NOW - timedelta(days=40)
    for row in rows:
        t = row["_t"]
        if t < cutoff or t > NOW + timedelta(hours=1):
            continue
        key = iso_z(t)
        # Later sources overwrite earlier ones; live data therefore wins.
        item = dict(row)
        item["time"] = key
        dedup[key] = item

    obs = sorted(dedup.values(), key=lambda x: x["_t"])
    latest = obs[-1]["_t"] if obs else None
    age_min = (NOW - latest).total_seconds() / 60.0 if latest else None
    meta = {
        "count": len(obs),
        "sources": sources,
        "latest_time": iso_z(latest) if latest else None,
        "age_minutes": round(age_min, 1) if age_min is not None else None,
        "fresh": bool(age_min is not None and age_min <= 120),
        "live_errors": live_errors,
    }
    return obs, meta


def load_wsa_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wsa_dir = DOCS_DATA / "wsa"
    idx = load_json(wsa_dir / "index.json", {}) or {}
    candidates: list[Path] = []
    if idx.get("latest"):
        candidates.append(wsa_dir / str(idx["latest"]))
    candidates.extend([wsa_dir / "latest.json"])
    for path in candidates:
        obj = load_json(path)
        if not isinstance(obj, dict):
            continue
        out = []
        for r in obj.get("records", []) if isinstance(obj.get("records"), list) else []:
            t = parse_time(r.get("time") or r.get("timestamp"))
            v = num(r.get("speed"))
            if t and v is not None:
                out.append({"time": t, "speed": float(v)})
        if out:
            out.sort(key=lambda x: x["time"])
            return out, {"file": str(path.relative_to(ROOT)), "count": len(out)}
    return [], {"file": None, "count": 0}


def interpolate_wsa(wsa: list[dict[str, Any]], t: datetime) -> float | None:
    if not wsa:
        return None
    if t <= wsa[0]["time"]:
        return wsa[0]["speed"]
    if t >= wsa[-1]["time"]:
        return wsa[-1]["speed"]
    prev = wsa[0]
    for nxt in wsa[1:]:
        if prev["time"] <= t <= nxt["time"]:
            den = (nxt["time"] - prev["time"]).total_seconds()
            if den <= 0:
                return prev["speed"]
            f = (t - prev["time"]).total_seconds() / den
            return prev["speed"] + (nxt["speed"] - prev["speed"]) * f
        prev = nxt
    return None


def load_cme_arrivals() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obj = load_json(CME_ARRIVALS, {}) or {}
    raw = obj.get("arrivals", []) if isinstance(obj, dict) else []
    out = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict):
            continue
        at = parse_time(r.get("arrival_time") or r.get("predicted_arrival") or r.get("estimated_arrival"))
        if not at:
            continue
        out.append({
            **r,
            "_arrival": at,
            "expected_wind_speed_increase": num(r.get("expected_wind_speed_increase")),
        })
    out.sort(key=lambda x: x["_arrival"])
    return out, {"file": str(CME_ARRIVALS.relative_to(ROOT)), "count": len(out)}


def load_boost_model() -> dict[str, Any]:
    obj = load_json(CME_BOOST_MODEL, {})
    return obj if isinstance(obj, dict) else {}


def expected_boost_from_model(cme: dict[str, Any], model: dict[str, Any]) -> float:
    explicit = num(cme.get("expected_wind_speed_increase"))
    if explicit is not None:
        return clamp(explicit, 0, 450)
    cls = str(cme.get("impact_class") or "BODY-HIT")
    class_row = model.get("by_impact_class", {}).get(cls, {}) if isinstance(model.get("by_impact_class"), dict) else {}
    class_med = num(class_row.get("median_delta_v"), 70.0) or 70.0
    glob = model.get("global", {}) if isinstance(model.get("global"), dict) else {}
    intercept = num(glob.get("intercept"), 40.0) or 40.0
    slope = num(glob.get("speed_geometry_slope"), 0.10) or 0.10
    speed = num(cme.get("speed"), 500.0) or 500.0
    geom = max(0.1, num(cme.get("geometry_multiplier"), 0.5) or 0.5)
    pred = 0.55 * class_med + 0.45 * max(0.0, intercept + slope * (speed - 500.0) * geom)
    return clamp(pred, 0, 450)


def main() -> None:
    obs, noaa_meta = load_wind_observations()
    if len(obs) < 24:
        raise SystemExit(f"Not enough NOAA wind observations: {len(obs)}")

    wsa, wsa_meta = load_wsa_records()
    cmes, cme_meta = load_cme_arrivals()
    boost_model = load_boost_model()

    # Cache simple arrays for repeated windows.
    def values_between(start: datetime, end: datetime) -> list[float]:
        return [r["speed"] for r in obs if start <= r["_t"] <= end]

    def speed_near(t: datetime, hours: float = 0.75) -> float | None:
        vals = values_between(t - timedelta(hours=hours), t + timedelta(hours=hours))
        return median(vals)

    def context_speed(t: datetime, hours: float = 6.0) -> float:
        # IMPORTANT: for hindcast verification use only observations at/before target time.
        vals = values_between(t - timedelta(hours=hours), t - timedelta(minutes=5))
        if vals:
            return float(median(vals, 400.0))
        # Future target: use the most recent observed context at NOW.
        vals = values_between(NOW - timedelta(hours=hours), NOW)
        return float(median(vals, 400.0))

    def context_trend(t: datetime, hours: float = 12.0) -> float:
        end = min(t - timedelta(minutes=5), NOW)
        rows = [r for r in obs if end - timedelta(hours=hours) <= r["_t"] <= end]
        if len(rows) < 6:
            return 0.0
        q = max(2, len(rows) // 4)
        first = float(median([x["speed"] for x in rows[:q]], rows[0]["speed"]))
        last = float(median([x["speed"] for x in rows[-q:]], rows[-1]["speed"]))
        dh = max(1.0, (rows[-1]["_t"] - rows[0]["_t"]).total_seconds() / 3600.0)
        return clamp((last - first) / dh, -18.0, 18.0)

    def fit_rotation_shift() -> int:
        recent = [r for r in obs if NOW - timedelta(hours=24) <= r["_t"] <= NOW]
        if len(recent) < 12:
            return 0
        sample = recent[::max(1, len(recent) // 18)]
        best = (1e18, 0)
        for shift in range(-36, 37, 3):
            errs = []
            for r in sample:
                past = r["_t"] - timedelta(days=27.27) + timedelta(hours=shift)
                p = speed_near(past, 2.5)
                if p is not None:
                    errs.append(abs(p - r["speed"]))
            if len(errs) >= 5:
                score = float(median(errs, 999.0)) + abs(shift) * 0.55
                if score < best[0]:
                    best = (score, shift)
        return int(best[1])

    rotation_shift_h = fit_rotation_shift()

    def rotation27(t: datetime) -> tuple[float | None, int]:
        center = t - timedelta(days=27.27) + timedelta(hours=rotation_shift_h)
        vals = values_between(center - timedelta(hours=4), center + timedelta(hours=4))
        if vals:
            return float(median(vals)), len(vals)
        vals = values_between(center - timedelta(hours=12), center + timedelta(hours=12))
        if vals:
            return float(median(vals)), len(vals)
        return None, 0

    def cme_boost_at(t: datetime) -> tuple[float, dict[str, Any] | None]:
        total = 0.0
        best = None
        for c in cmes:
            if c.get("earth_candidate") is False:
                continue
            arrival = c["_arrival"]
            dh = (t - arrival).total_seconds() / 3600.0
            if dh < -18 or dh > 36:
                continue
            amp = expected_boost_from_model(c, boost_model)
            # Fast rise near arrival, slower decay afterwards.
            if dh < 0:
                shape = math.exp(-((dh / 6.0) ** 2))
            else:
                shape = math.exp(-dh / 15.0)
            add = amp * shape
            total += add
            if best is None or add > best["boost"]:
                best = {
                    "id": c.get("id"),
                    "arrival_time": iso_z(arrival),
                    "impact_class": c.get("impact_class"),
                    "cme_speed": c.get("speed"),
                    "expected_delta_v": round(amp, 1),
                    "boost": round(add, 1),
                }
        return clamp(total, 0, 450), best

    default_coeff = {
        "version": "SWIFT-Wind-AI-v0.6-python-records",
        "updated_at": iso_z(NOW),
        "weights": {
            "recent": 0.34,
            "rotation27": 0.39,
            "hss_nonlinear": 0.17,
            "wsa": 0.10,
            "cme": 1.00,
        },
        "calibration": {"gain": 1.0, "offset": 0.0},
        "limits": {
            "min_speed": 260.0,
            "max_speed": 950.0,
            "max_delta_per_hour": 32.0,
            "ema_alpha": 0.48,
        },
        "notes": [
            "records and forecast are both emitted for dashboard compatibility.",
            "Historical fit uses context available before each target time (walk-forward), not current-time leakage.",
            "CME speed increase is learned separately in cme-boost-model.json and added as an independent pulse.",
            "Primary wind accuracy is the hit rate within ±50 km/s.",
        ],
    }

    loaded = load_json(COEFFICIENTS, {}) or {}
    coeff = dict(default_coeff)
    if isinstance(loaded, dict):
        coeff.update({k: v for k, v in loaded.items() if k not in ("weights", "calibration", "limits")})
        ww = dict(default_coeff["weights"])
        if isinstance(loaded.get("weights"), dict):
            for k in ww:
                if num(loaded["weights"].get(k)) is not None:
                    ww[k] = float(num(loaded["weights"].get(k)))
        # Force CME weight to a sane range because old v0.4 files used 0.05.
        ww["cme"] = clamp(ww.get("cme", 1.0), 0.65, 1.35)
        coeff["weights"] = ww
        cc = dict(default_coeff["calibration"])
        if isinstance(loaded.get("calibration"), dict):
            cc["gain"] = float(num(loaded["calibration"].get("gain"), 1.0))
            cc["offset"] = float(num(loaded["calibration"].get("offset"), 0.0))
        coeff["calibration"] = cc
        ll = dict(default_coeff["limits"])
        if isinstance(loaded.get("limits"), dict):
            for k in ll:
                if num(loaded["limits"].get(k)) is not None:
                    ll[k] = float(num(loaded["limits"].get(k)))
        coeff["limits"] = ll

    def raw_components(t: datetime, calibration: bool = True) -> tuple[float, float, dict[str, Any]]:
        lead_h = max(0.0, (t - NOW).total_seconds() / 3600.0)
        rec = context_speed(t)
        trend = context_trend(t)
        rot, rot_n = rotation27(t)
        if rot is None:
            rot = rec
        wsa_speed = interpolate_wsa(wsa, t)

        # Current-state term decays with forecast lead time; historical walk-forward gets full local context.
        if t <= NOW:
            rec_term = rec + clamp(trend * 3.0, -45.0, 45.0)
        else:
            decay = math.exp(-lead_h / 24.0)
            rec_term = rec + decay * clamp(trend * min(lead_h, 12.0), -50.0, 50.0)

        hss_gate = sigmoid((rot - 455.0) / 45.0)
        hss_term = 375.0 + 250.0 * hss_gate + 30.0 * math.tanh((rot - 500.0) / 120.0)

        bg_sources = {
            "recent": (coeff["weights"]["recent"], rec_term),
            "rotation27": (coeff["weights"]["rotation27"], rot),
            "hss_nonlinear": (coeff["weights"]["hss_nonlinear"], hss_term),
        }
        if wsa_speed is not None:
            bg_sources["wsa"] = (coeff["weights"]["wsa"], wsa_speed)

        den = sum(max(0.0, wgt) for wgt, _ in bg_sources.values()) or 1.0
        background = sum(max(0.0, wgt) * val for wgt, val in bg_sources.values()) / den

        cme_boost, cme_meta = cme_boost_at(t)
        enhanced = background + coeff["weights"]["cme"] * cme_boost

        background = clamp(background, coeff["limits"]["min_speed"], coeff["limits"]["max_speed"])
        enhanced = clamp(enhanced, coeff["limits"]["min_speed"], coeff["limits"]["max_speed"])

        if calibration:
            gain = float(num(coeff["calibration"].get("gain"), 1.0))
            offset = float(num(coeff["calibration"].get("offset"), 0.0))
            background = clamp(gain * background + offset, coeff["limits"]["min_speed"], coeff["limits"]["max_speed"])
            enhanced = clamp(gain * enhanced + offset, coeff["limits"]["min_speed"], coeff["limits"]["max_speed"])

        meta = {
            "persistence_speed": round(rec, 2),
            "recent_trend_kms_per_h": round(trend, 3),
            "rotation27_speed": round(rot, 2),
            "rotation27_shift_hours": rotation_shift_h,
            "rotation27_samples": rot_n,
            "hss_gate": round(hss_gate, 4),
            "hss_speed": round(hss_term, 2),
            "wsa_speed": round(wsa_speed, 2) if wsa_speed is not None else None,
            "cme_effect_weighted": round(cme_boost, 2),
            "nearest_cme": cme_meta,
        }
        return background, enhanced, meta

    # Fit gain/offset on a WALK-FORWARD hindcast. This fixes the old leakage bug where
    # historical verification accidentally reused the current 6h solar wind context.
    fit_pairs: list[tuple[float, float]] = []
    eval_rows = [r for r in obs if NOW - timedelta(days=7) <= r["_t"] <= NOW - timedelta(hours=1)]
    eval_rows = eval_rows[::max(1, len(eval_rows) // 220)]
    for r in eval_rows:
        _, pred, _ = raw_components(r["_t"], calibration=False)
        fit_pairs.append((pred, r["speed"]))

    if len(fit_pairs) >= 16:
        xs = [x for x, _ in fit_pairs]
        ys = [y for _, y in fit_pairs]
        mx = float(mean(xs, 0.0))
        my = float(mean(ys, 0.0))
        var = sum((x - mx) ** 2 for x in xs)
        cov = sum((x - mx) * (y - my) for x, y in fit_pairs)
        gain_new = clamp(cov / var if var > 1e-9 else 1.0, 0.78, 1.18)
        offset_new = clamp(my - gain_new * mx, -100.0, 100.0)
        old_gain = float(num(coeff["calibration"].get("gain"), 1.0))
        old_offset = float(num(coeff["calibration"].get("offset"), 0.0))
        # Moderate update, avoiding a single noisy day over-correcting the model.
        gain = 0.65 * old_gain + 0.35 * gain_new
        offset = 0.65 * old_offset + 0.35 * offset_new
        coeff["calibration"] = {"gain": round(gain, 5), "offset": round(offset, 3)}
        fit_errs = [(gain * x + offset) - y for x, y in fit_pairs]
        coeff["last_fit"] = {
            "time": iso_z(NOW),
            "fit_count": len(fit_pairs),
            "gain_candidate": round(gain_new, 5),
            "offset_candidate": round(offset_new, 3),
            "gain_applied": round(gain, 5),
            "offset_applied": round(offset, 3),
            "mae": round(float(mean([abs(e) for e in fit_errs], 0.0)), 2),
            "bias": round(float(mean(fit_errs, 0.0)), 2),
        }

    coeff["updated_at"] = iso_z(NOW)
    coeff["rotation27_shift_hours"] = rotation_shift_h
    coeff["cme_boost_sample_count"] = int(boost_model.get("sample_count", 0) or 0)

    def hourly_floor(dt: datetime) -> datetime:
        return dt.replace(minute=0, second=0, microsecond=0)

    now_hour = hourly_floor(NOW)
    records: list[dict[str, Any]] = []

    # Past 72 h, including the current hour -> 73 rows.
    past_times = [now_hour - timedelta(hours=h) for h in range(PAST_RECORD_HOURS, -1, -1)]
    for t in past_times:
        observed = speed_near(t, 0.75)
        bg, enhanced, meta = raw_components(t, calibration=True)
        records.append({
            "time": iso_z(t),
            "observed_speed": round(observed, 2) if observed is not None else None,
            "wsa_speed": meta["wsa_speed"],
            "persistence_speed": meta["persistence_speed"],
            "rotation27_speed": meta["rotation27_speed"],
            "swift_background_speed": round(bg, 2),
            "swift_cme_enhanced_speed": round(enhanced, 2),
            "cme_effect_weighted": meta["cme_effect_weighted"],
            "features": meta,
            "kind": "hindcast",
        })

    # Future 120 h -> 120 rows. Apply smoothing only to FUTURE output, not hindcast metrics.
    raw_future = []
    for h in range(1, FORECAST_HOURS + 1):
        t = now_hour + timedelta(hours=h)
        bg, enhanced, meta = raw_components(t, calibration=True)
        raw_future.append({"time": t, "background": bg, "enhanced": enhanced, "meta": meta})

    alpha = clamp(float(num(coeff["limits"].get("ema_alpha"), 0.48)), 0.05, 0.95)
    max_delta = clamp(float(num(coeff["limits"].get("max_delta_per_hour"), 32.0)), 5.0, 80.0)
    prev_enh = records[-1]["swift_cme_enhanced_speed"] if records else context_speed(NOW)
    prev_bg = records[-1]["swift_background_speed"] if records else context_speed(NOW)
    forecast: list[dict[str, Any]] = []

    for rr in raw_future:
        bg_ema = alpha * rr["background"] + (1.0 - alpha) * prev_bg
        enh_ema = alpha * rr["enhanced"] + (1.0 - alpha) * prev_enh
        bg = prev_bg + clamp(bg_ema - prev_bg, -max_delta, max_delta)
        enh = prev_enh + clamp(enh_ema - prev_enh, -max_delta, max_delta)
        bg = clamp(bg, coeff["limits"]["min_speed"], coeff["limits"]["max_speed"])
        enh = clamp(enh, coeff["limits"]["min_speed"], coeff["limits"]["max_speed"])
        prev_bg, prev_enh = bg, enh
        meta = rr["meta"]
        rec = {
            "time": iso_z(rr["time"]),
            "lead_hours": round((rr["time"] - NOW).total_seconds() / 3600.0, 2),
            "observed_speed": None,
            "wsa_speed": meta["wsa_speed"],
            "persistence_speed": meta["persistence_speed"],
            "rotation27_speed": meta["rotation27_speed"],
            "swift_background_speed": round(bg, 2),
            "swift_cme_enhanced_speed": round(enh, 2),
            "cme_effect_weighted": meta["cme_effect_weighted"],
            "speed": round(enh, 2),
            "speed_raw_nonlinear": round(rr["enhanced"], 2),
            "features": meta,
            "kind": "forecast",
        }
        forecast.append(rec)
        records.append(rec)

    # records count = 73 past + 120 future = 193, matching the dashboard's preferred range.
    def verify_hours(hours: int) -> dict[str, Any]:
        start = NOW - timedelta(hours=hours)
        items = []
        for row in records:
            if row.get("kind") != "hindcast":
                continue
            t = parse_time(row.get("time"))
            obs_v = num(row.get("observed_speed"))
            pred_v = num(row.get("swift_cme_enhanced_speed"))
            bg_v = num(row.get("swift_background_speed"))
            if not t or t < start or obs_v is None or pred_v is None or bg_v is None:
                continue
            items.append({
                "time": row["time"],
                "observed_speed": round(obs_v, 2),
                "predicted_speed": round(pred_v, 2),
                "background_speed": round(bg_v, 2),
                "error": round(pred_v - obs_v, 2),
                "background_error": round(bg_v - obs_v, 2),
            })

        def metric(key: str) -> dict[str, Any]:
            errs = [x[key] for x in items]
            if not errs:
                return {"count": 0, "status": "learning"}
            abs_errs = [abs(x) for x in errs]
            return {
                "count": len(errs),
                "mae": round(float(mean(abs_errs, 0.0)), 2),
                "bias": round(float(mean(errs, 0.0)), 2),
                "rmse": round(math.sqrt(float(mean([e * e for e in errs], 0.0))), 2),
                "hit_rate_50kms": round(sum(1 for e in abs_errs if e <= 50.0) / len(errs) * 100.0, 1),
                "hit_rate_75kms": round(sum(1 for e in abs_errs if e <= 75.0) / len(errs) * 100.0, 1),
                "hit_rate_100kms": round(sum(1 for e in abs_errs if e <= 100.0) / len(errs) * 100.0, 1),
            }

        return {
            "enhanced": metric("error"),
            "background": metric("background_error"),
            "items": items[-300:],
        }

    # 7d/30d metrics are evaluated separately from the 193 display records.
    def verify_days(days: int) -> dict[str, Any]:
        candidates = [r for r in obs if NOW - timedelta(days=days) <= r["_t"] <= NOW - timedelta(hours=1)]
        # Roughly hourly samples to control runtime while preserving the full window.
        sampled = candidates[::max(1, len(candidates) // max(24 * days, 1))]
        items = []
        for r in sampled:
            bg, pred, _ = raw_components(r["_t"], calibration=True)
            err = pred - r["speed"]
            items.append({
                "time": iso_z(r["_t"]),
                "observed_speed": round(r["speed"], 2),
                "predicted_speed": round(pred, 2),
                "error": round(err, 2),
                "abs_error": round(abs(err), 2),
            })
        if not items:
            return {"count": 0, "status": "learning"}
        errs = [x["error"] for x in items]
        abs_errs = [abs(x) for x in errs]
        return {
            "count": len(items),
            "mae": round(float(mean(abs_errs, 0.0)), 2),
            "bias": round(float(mean(errs, 0.0)), 2),
            "rmse": round(math.sqrt(float(mean([e * e for e in errs], 0.0))), 2),
            "hit_rate_50kms": round(sum(1 for x in abs_errs if x <= 50) / len(items) * 100.0, 1),
            "hit_rate_75kms": round(sum(1 for x in abs_errs if x <= 75) / len(items) * 100.0, 1),
            "hit_rate_100kms": round(sum(1 for x in abs_errs if x <= 100) / len(items) * 100.0, 1),
            "items": items[-500:],
        }

    verification = {
        "updated_at": iso_z(NOW),
        "definition": {
            "primary_accuracy": "hit_rate_50kms",
            "primary_accuracy_text": "|predicted - observed| <= 50 km/s",
        },
        "last_24h": verify_hours(24),
        "last_72h": verify_hours(72),
        "last_7d": verify_days(7),
        "last_30d": verify_days(30),
    }

    v24 = verification["last_24h"]["enhanced"]
    b24 = verification["last_24h"]["background"]
    latest_payload = {
        "updated_at": iso_z(NOW),
        "model": "SWIFT-Wind-AI-v0.6-python-records",
        "source": "NOAA history + live NOAA RTSW + 27.27-day recurrence + optional WSA + learned CME boost",
        "forecast_days": FORECAST_HOURS / 24,
        "history_hours": PAST_RECORD_HOURS,
        "step_hours": STEP_HOURS,
        "target": "±50 km/s",
        "formula": "Vsw = calibrated(background_recent_rotation27_HSS_WSA) + learned_CME_deltaV; future output is EMA/slew limited",
        "current": forecast[0] if forecast else None,
        "records": records,
        "forecast": forecast,
        "verification": verification,
        "coefficients": coeff,
        "inputs": {
            "noaa_wind": noaa_meta,
            "wsa": wsa_meta,
            "cme_arrivals": cme_meta,
            "cme_boost_model": {
                "file": str(CME_BOOST_MODEL.relative_to(ROOT)),
                "sample_count": int(boost_model.get("sample_count", 0) or 0),
                "updated_at": boost_model.get("updated_at"),
            },
        },
        "health": {
            "status": "ok" if noaa_meta.get("fresh") else "warning",
            "noaa_fresh": noaa_meta.get("fresh"),
            "noaa_age_minutes": noaa_meta.get("age_minutes"),
            "records_count": len(records),
            "forecast_count": len(forecast),
        },
    }

    # Forecast archive for true forecast-vs-observation verification later.
    archive_obj = load_json(ARCHIVE, {"items": []}) or {"items": []}
    archive_items = archive_obj.get("items", []) if isinstance(archive_obj, dict) else []
    archive_items = [x for x in archive_items if isinstance(x, dict)][-KEEP_ARCHIVE:]
    issued_at = iso_z(NOW)
    for r in forecast:
        archive_items.append({
            "issued_at": issued_at,
            "target_time": r["time"],
            "predicted_speed": r["swift_cme_enhanced_speed"],
            "background_speed": r["swift_background_speed"],
            "cme_effect_weighted": r["cme_effect_weighted"],
            "lead_hours": r["lead_hours"],
        })
    archive_items = archive_items[-KEEP_ARCHIVE:]

    # Accuracy history with the exact schema already consumed by index.html.
    hist_obj = load_json(ACCURACY_HISTORY, {"history": []}) or {"history": []}
    hist_rows = hist_obj.get("history", []) if isinstance(hist_obj, dict) else []
    hist_rows = [x for x in hist_rows if isinstance(x, dict)]
    accuracy_row = {
        "time": iso_z(NOW),
        "updated_at": iso_z(NOW),
        # Main percentage = real ±50 km/s hit rate, not the old synthetic MAE score.
        "enhanced_accuracy_score": v24.get("hit_rate_50kms"),
        "enhanced_hit_rate_50": v24.get("hit_rate_50kms"),
        "enhanced_hit_rate_100": v24.get("hit_rate_100kms"),
        "enhanced_mae": v24.get("mae"),
        "enhanced_bias": v24.get("bias"),
        "enhanced_rmse": v24.get("rmse"),
        "background_accuracy_score": b24.get("hit_rate_50kms"),
        "background_hit_rate_50": b24.get("hit_rate_50kms"),
        "background_hit_rate_100": b24.get("hit_rate_100kms"),
        "background_mae": b24.get("mae"),
        "background_bias": b24.get("bias"),
        "background_rmse": b24.get("rmse"),
        "definition": "accuracy_score = ±50 km/s hit rate over last 24h",
        "count": v24.get("count", 0),
    }
    # Avoid duplicate rows within 20 minutes if manually re-run repeatedly.
    hist_rows = [r for r in hist_rows if parse_time(r.get("time") or r.get("updated_at")) and parse_time(r.get("time") or r.get("updated_at")) < NOW - timedelta(minutes=20)]
    hist_rows.append(accuracy_row)
    hist_rows = hist_rows[-1000:]
    accuracy_payload = {
        "updated_at": iso_z(NOW),
        "definition": "Wind AI main accuracy = hit rate within ±50 km/s over last 24h",
        "latest": accuracy_row,
        "history": hist_rows,
        "last_24h": verification["last_24h"],
        "last_72h": verification["last_72h"],
        "last_7d": verification["last_7d"],
        "last_30d": verification["last_30d"],
    }

    save_json(LATEST, latest_payload)
    save_json(COEFFICIENTS, coeff)
    save_json(VERIFICATION, verification)
    save_json(ARCHIVE, {"updated_at": iso_z(NOW), "items": archive_items})
    save_json(ACCURACY, accuracy_payload)
    save_json(ACCURACY_HISTORY, accuracy_payload)
    save_json(ACCURACY_HISTORY_ALIAS, accuracy_payload)

    files = []
    for f in sorted(OUT.glob("*.json")):
        if f.name == "index.json":
            continue
        files.append({
            "file": f.name,
            "path": f"./data/swift-wind/{f.name}",
            "size_bytes": f.stat().st_size,
        })
    save_json(INDEX, {
        "updated_at": iso_z(NOW),
        "latest": "latest.json",
        "coefficients": "coefficients.json",
        "verification": "verification.json",
        "archive": "forecast-archive.json",
        "accuracy": "accuracy.json",
        "accuracy_history": "accuracy-history.json",
        "records_count": len(records),
        "forecast_count": len(forecast),
        "files": files,
    })

    print(json.dumps({
        "model": latest_payload["model"],
        "records": len(records),
        "forecast": len(forecast),
        "noaa": noaa_meta,
        "rotation27_shift_hours": rotation_shift_h,
        "cme_arrivals": len(cmes),
        "cme_boost_samples": boost_model.get("sample_count", 0),
        "wind_accuracy_24h_within_50": v24.get("hit_rate_50kms"),
        "wind_mae_24h": v24.get("mae"),
        "calibration": coeff.get("calibration"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
