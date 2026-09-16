#!/usr/bin/env python3
"""Build a lightweight SWIFT Bz forecast AI product.

Output:
- docs/data/swift-bz/latest.json

Forecast design:
- Uses recent NOAA IMF Bz persistence.
- Uses 27.27-day recurrence when archived IMF exists.
- Adds CME-arrival risk windows from docs/data/cme-arrivals/latest.json.
- Produces 3-hour bins for five days. The Excel report charts the first three days.

This is intentionally simple and transparent. It is not meant to declare the CME
internal field direction; CME contribution is represented as increased probability
of southward Bz and a wider negative-tail Bz-min forecast.
"""
from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
OUT = DATA / "swift-bz" / "latest.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

BIN_HOURS = 3
FORECAST_DAYS = 5
RECURRENCE_DAYS = 27.27


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


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


def records(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("records", "items", "history", "data", "forecast", "arrivals"):
            if isinstance(obj.get(key), list):
                return [x for x in obj[key] if isinstance(x, dict)]
    return []


def load_mag_history() -> list[dict[str, Any]]:
    paths = [
        DATA / "noaa" / "mag_history.json",
        DATA / "noaa_imf_history.json",
    ]
    out = []
    for p in paths:
        for r in records(load_json(p)):
            t = parse_time(r.get("time") or r.get("time_tag") or r.get("timestamp"))
            bz = num(r.get("bz") or r.get("bz_gsm") or r.get("imf_bz"))
            bt = num(r.get("bt") or r.get("total_field"))
            by = num(r.get("by") or r.get("by_gsm"))
            if t and bz is not None:
                out.append({"time": iso_z(t), "_t": t, "bz": bz, "bt": bt, "by": by})
    dedup = {}
    for r in out:
        dedup[r["time"]] = r
    return sorted(dedup.values(), key=lambda x: x["_t"])


def load_cme_arrivals() -> list[dict[str, Any]]:
    out = []
    for r in records(load_json(DATA / "cme-arrivals" / "latest.json")):
        at = parse_time(r.get("arrival_time") or r.get("arrival") or r.get("estimatedShockArrivalTime"))
        if at:
            x = dict(r)
            x["_arrival"] = at
            out.append(x)
    return sorted(out, key=lambda x: x["_arrival"])


def median(vals: list[float], default: float = 0.0) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.median(vals) if vals else default


def quantile(vals: list[float], q: float, default: float = 0.0) -> float:
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals:
        return default
    idx = max(0, min(len(vals) - 1, int(round((len(vals) - 1) * q))))
    return vals[idx]


def near_records(hist: list[dict[str, Any]], center: datetime, hours: float) -> list[dict[str, Any]]:
    start = center - timedelta(hours=hours)
    end = center + timedelta(hours=hours)
    return [r for r in hist if start <= r["_t"] <= end]


def cme_risk_for_time(cmes: list[dict[str, Any]], t: datetime) -> tuple[float, dict[str, Any] | None]:
    best_score = 0.0
    best = None
    for c in cmes:
        arrival = c["_arrival"]
        dt_h = abs((t - arrival).total_seconds() / 3600.0)
        if dt_h > 24:
            continue
        impact = str(c.get("impact_class") or "").upper()
        if "ENLIL" in impact or "CORE" in impact:
            base = 1.0
        elif "BODY" in impact:
            base = 0.75
        elif "FLANK" in impact:
            base = 0.45
        else:
            base = 0.25
        conf = num(c.get("confidence"), 0.5) or 0.5
        geom = num(c.get("geometry_multiplier"), 0.5) or 0.5
        # Bell-shaped influence around arrival.
        shape = math.exp(-(dt_h / 10.0) ** 2)
        score = base * conf * max(0.1, geom) * shape
        if score > best_score:
            best_score = score
            best = c
    return max(0.0, min(1.0, best_score)), best


def g_scale_from_kp(kp: float | None) -> str:
    if kp is None:
        return "G?"
    if kp >= 9:
        return "G5"
    if kp >= 8:
        return "G4"
    if kp >= 7:
        return "G3"
    if kp >= 6:
        return "G2"
    if kp >= 5:
        return "G1"
    return "G0"


def build_forecast(hist: list[dict[str, Any]], cmes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utcnow()
    recent = [r for r in hist if r["_t"] >= now - timedelta(hours=6)]
    recent_bz = median([r["bz"] for r in recent], 0.0)
    recent_bt = median([r["bt"] for r in recent if r.get("bt") is not None], 6.0)
    recent_sigma = max(1.5, statistics.pstdev([r["bz"] for r in recent]) if len(recent) >= 3 else 2.5)
    climatology_neg_q = quantile([r["bz"] for r in hist[-4000:]], 0.10, -5.0)
    forecast = []
    steps = int(FORECAST_DAYS * 24 / BIN_HOURS)
    for i in range(steps):
        start = now + timedelta(hours=i * BIN_HOURS)
        end = start + timedelta(hours=BIN_HOURS)
        lead_h = i * BIN_HOURS
        rec_center = start - timedelta(days=RECURRENCE_DAYS)
        rec_rows = near_records(hist, rec_center, 2.0)
        rec_bz = median([r["bz"] for r in rec_rows], None) if rec_rows else None
        rec_bt = median([r["bt"] for r in rec_rows if r.get("bt") is not None], None) if rec_rows else None
        w_recent = max(0.15, 0.75 * math.exp(-lead_h / 36.0))
        w_rec = 0.35 if rec_bz is not None else 0.0
        w_clim = max(0.0, 1.0 - w_recent - w_rec)
        base_bz = w_recent * recent_bz + w_clim * 0.0 + (w_rec * rec_bz if rec_bz is not None else 0.0)
        base_bt = max(2.0, w_recent * recent_bt + w_clim * 5.0 + (w_rec * (rec_bt or 5.0) if rec_bt is not None else 0.0))
        cme_score, cme = cme_risk_for_time(cmes, start)
        # CME score increases Bt and negative-tail risk, not a deterministic negative Bz assertion.
        bt_forecast = base_bt + 10.0 * cme_score
        bz_min = min(base_bz - 1.65 * recent_sigma - 10.0 * cme_score, climatology_neg_q - 5.0 * cme_score)
        south_prob = 100.0 / (1.0 + math.exp((base_bz + 1.0) / 2.4))
        south_prob += 38.0 * cme_score
        south_prob = max(2.0, min(98.0, south_prob))
        if south_prob >= 75:
            risk = "HIGH"
        elif south_prob >= 55:
            risk = "MODERATE"
        else:
            risk = "LOW"
        forecast.append({
            "time": iso_z(start),
            "end_time": iso_z(end),
            "lead_hours": lead_h,
            "bz_forecast": round(base_bz, 2),
            "bz_min_forecast": round(bz_min, 2),
            "bt_forecast": round(bt_forecast, 2),
            "southward_bz_probability": round(south_prob, 1),
            "bz_risk": risk,
            "confidence": round(max(0.2, min(0.9, 0.75 - lead_h / 240.0 + (0.1 if rec_bz is not None else 0.0))), 2),
            "cme_risk_score": round(cme_score, 3),
            "cme_id": cme.get("id") if cme else None,
            "model_components": {
                "recent_bz": round(recent_bz, 2),
                "rotation27_bz": round(rec_bz, 2) if rec_bz is not None else None,
                "recent_weight": round(w_recent, 3),
                "rotation27_weight": round(w_rec, 3),
            },
        })
    return forecast


def verification(hist: list[dict[str, Any]]) -> dict[str, Any]:
    # A transparent persistence baseline verification for current model readiness.
    now = utcnow()
    rows = [r for r in hist if now - timedelta(days=7) <= r["_t"] <= now]
    pairs = []
    by_time = {r["_t"].replace(minute=0, second=0, microsecond=0): r for r in hist}
    for r in rows:
        prev_t = r["_t"].replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
        prev = by_time.get(prev_t)
        if prev:
            pairs.append((prev["bz"], r["bz"]))
    if not pairs:
        return {"count": 0, "direction_hit_rate_threshold60": None, "bz_min_mae": None, "note": "Not enough history yet"}
    direction_hit = 0
    abs_err = []
    for pred, obs in pairs:
        prob = 100.0 / (1.0 + math.exp((pred + 1.0) / 2.4))
        pred_south = prob >= 60
        obs_south = obs < 0
        direction_hit += int(pred_south == obs_south)
        abs_err.append(abs(pred - obs))
    return {
        "count": len(pairs),
        "direction_hit_rate_threshold60": round(100.0 * direction_hit / len(pairs), 1),
        "bz_min_mae": round(sum(abs_err) / len(abs_err), 2),
        "method": "3h persistence baseline until enough trained hindcasts are available",
    }


def main() -> None:
    hist = load_mag_history()
    cmes = load_cme_arrivals()
    forecast = build_forecast(hist, cmes) if hist else []
    max_risk = max(forecast, key=lambda r: r.get("southward_bz_probability", 0), default=None)
    payload = {
        "updated_at": iso_z(utcnow()),
        "model": "SWIFT Bz AI v1 browser/report compatible",
        "forecast_days": FORECAST_DAYS,
        "cadence_hours": BIN_HOURS,
        "inputs": {
            "imf_history_records": len(hist),
            "cme_arrivals": len(cmes),
            "uses_rotation27_days": RECURRENCE_DAYS,
        },
        "current": forecast[0] if forecast else None,
        "max_risk": max_risk,
        "verification": {"last_7d": verification(hist)},
        "forecast": forecast,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Bz forecast rows={len(forecast)} mag_history={len(hist)} cmes={len(cmes)}")


if __name__ == "__main__":
    main()
