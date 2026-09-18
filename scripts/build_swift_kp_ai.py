#!/usr/bin/env python3
"""SWIFT Kp AI v0.8 - adaptive residual / self-calibrating 72 h forecast.

Design goals
------------
1. Forecast only the next 72 hours (3-hour cadence).
2. Published Kp floor is 0.33 (internal residual may be lower).
3. Predict *change from a recent observed-Kp anchor* instead of absolute Kp.
4. Gate uncertain SWIFT Bz and SWIFT Wind inputs by their verified skill/readiness.
5. Learn lead-day bias from real archived forecasts after observations arrive.
6. Learn bias ONLY from this model version; old-model bias is never transferred.
7. Add a bounded recent-Kp trend term so a rising/declining observed Kp is not over-damped.

Outputs
-------
docs/data/swift-kp/latest.json
docs/data/swift-kp/forecast.json
docs/data/swift-kp/verification.json
docs/data/swift-kp/leadtime-skill.json
docs/data/swift-kp/forecast-archive.json
docs/data/swift-kp/coefficients.json
docs/data/swift-kp/forecast.txt

Notes
-----
- Kp=0 is physically possible. The 0.33 floor here is an operational/UI model
  choice requested for SWIFT, not a statement that Kp=0 cannot occur.
- CME internal Bz is not deterministic; its effect enters only through the Bz
  probability/readiness product and the solar-wind forecast.
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
OUT = DATA / "swift-kp"
OUT.mkdir(parents=True, exist_ok=True)

LATEST = OUT / "latest.json"
FORECAST_JSON = OUT / "forecast.json"
VERIFICATION = OUT / "verification.json"
LEAD_SKILL = OUT / "leadtime-skill.json"
ARCHIVE = OUT / "forecast-archive.json"
COEF = OUT / "coefficients.json"
TXT = OUT / "forecast.txt"
INDEX = OUT / "index.json"

MODEL = "SWIFT-Kp-AI-v0.8-adaptive-residual"
FORECAST_HOURS = 72
STEP_HOURS = 3
Kp_FLOOR = 0.33


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
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
    if len(s) == 16 and s[10] in ("T", " "):
        s = s.replace(" ", "T") + ":00+00:00"
    elif len(s) == 19 and s[10] == " ":
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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def mean(vals: list[float], default: float = 0.0) -> float:
    return sum(vals) / len(vals) if vals else default


def median(vals: list[float], default: float = 0.0) -> float:
    vals = [x for x in vals if x is not None and math.isfinite(x)]
    return statistics.median(vals) if vals else default


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def records(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for key in ("records", "items", "data", "history", "forecast"):
            if isinstance(obj.get(key), list):
                return [r for r in obj[key] if isinstance(r, dict)]
    return []


def round_kp_third(x: float) -> float:
    vals = [0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67,
            3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0, 5.33, 5.67,
            6.0, 6.33, 6.67, 7.0, 7.33, 7.67, 8.0, 8.33, 8.67, 9.0]
    x = clamp(x, Kp_FLOOR, 9.0)
    return min(vals, key=lambda v: abs(v - x))


def g_scale(kp: float) -> str:
    if kp >= 9: return "G5"
    if kp >= 8: return "G4"
    if kp >= 7: return "G3"
    if kp >= 6: return "G2"
    if kp >= 5: return "G1"
    return "G0"


def merge_scalar_series(paths: list[Path], value_fn, valid_fn) -> list[dict[str, Any]]:
    by_time: dict[datetime, dict[str, Any]] = {}
    for p in paths:
        for r in records(load_json(p, {})):
            t = parse_time(r.get("time") or r.get("time_tag") or r.get("timestamp"))
            v = value_fn(r)
            if t and v is not None and valid_fn(v):
                key = t.replace(second=0, microsecond=0)
                by_time[key] = {"time": t, "value": float(v)}
    return [by_time[k] for k in sorted(by_time)]


KP_OBS = merge_scalar_series(
    [DATA / "noaa" / "kp_history.json", DATA / "noaa-kp" / "history.json", DATA / "noaa_kp_history.json"],
    lambda r: fnum(r.get("kp") if r.get("kp") is not None else r.get("kp_raw")),
    lambda x: 0 <= x <= 9,
)

WIND_OBS = merge_scalar_series(
    [DATA / "noaa" / "wind_history.json", DATA / "noaa-wind" / "history.json", DATA / "noaa_wind_history.json"],
    lambda r: fnum(r.get("speed") or r.get("proton_speed") or r.get("v")),
    lambda x: 200 <= x <= 1200,
)

IMF_OBS: list[dict[str, Any]] = []
_imf_by: dict[datetime, dict[str, Any]] = {}
for p in [DATA / "noaa" / "mag_history.json", DATA / "noaa-imf" / "history.json", DATA / "noaa_imf_history.json"]:
    for r in records(load_json(p, {})):
        t = parse_time(r.get("time") or r.get("time_tag"))
        bz = fnum(r.get("bz") if r.get("bz") is not None else r.get("bz_gsm"))
        bt = fnum(r.get("bt"))
        if t and bz is not None:
            _imf_by[t.replace(second=0, microsecond=0)] = {"time": t, "bz": bz, "bt": bt}
IMF_OBS = [_imf_by[k] for k in sorted(_imf_by)]


def nearest(rows: list[dict[str, Any]], t: datetime, max_hours: float) -> dict[str, Any] | None:
    if not rows:
        return None
    best = min(rows, key=lambda r: abs((r["time"] - t).total_seconds()))
    return best if abs((best["time"] - t).total_seconds()) <= max_hours * 3600 else None


def observed_kp(t: datetime) -> float | None:
    r = nearest(KP_OBS, t, 1.7)
    return r["value"] if r else None


def observed_wind(t: datetime) -> float | None:
    r = nearest(WIND_OBS, t, 2.0)
    return r["value"] if r else None


def observed_imf(t: datetime) -> dict[str, float]:
    rr = [r for r in IMF_OBS if abs((r["time"] - t).total_seconds()) <= 1.5 * 3600]
    if not rr:
        return {"south_prob": 0.35, "south_bz": 0.0, "bt": 5.0}
    bz = [r["bz"] for r in rr]
    bt = [r["bt"] for r in rr if r.get("bt") is not None]
    return {
        "south_prob": sum(1 for x in bz if x < 0) / len(bz),
        "south_bz": max(0.0, -min(bz)),
        "bt": median(bt, 5.0),
    }


def recent_kp_anchor(ref: datetime, hours: int = 12) -> float:
    vals: list[tuple[float, float]] = []
    for r in KP_OBS:
        age_h = (ref - r["time"]).total_seconds() / 3600
        if 0 <= age_h <= hours:
            # Exponential weight, newest observations matter most.
            w = math.exp(-age_h / 5.0)
            vals.append((r["value"], w))
    if not vals:
        return 1.5
    return sum(v * w for v, w in vals) / sum(w for _, w in vals)


def rolling_kp_median(ref: datetime, days: int = 7) -> float:
    vals = [r["value"] for r in KP_OBS if ref - timedelta(days=days) <= r["time"] <= ref]
    return median(vals, 1.5)


NOW = now_utc()
CURRENT_ANCHOR = recent_kp_anchor(NOW)
QUIET_BASELINE = rolling_kp_median(NOW, 7)

def recent_kp_trend(ref: datetime, hours: int = 12) -> float:
    """Robust Kp change per 3 hours using medians of recent and earlier halves.

    Positive means Kp has been rising. It is intentionally bounded because Kp is
    discrete/noisy and the trend should not dominate Wind/Bz drivers.
    """
    recent = [r["value"] for r in KP_OBS if ref - timedelta(hours=6) <= r["time"] <= ref]
    prior = [r["value"] for r in KP_OBS if ref - timedelta(hours=hours) <= r["time"] < ref - timedelta(hours=6)]
    if not recent or not prior:
        return 0.0
    # Difference across roughly 6 h -> convert to a 3 h tendency.
    return clamp((median(recent, CURRENT_ANCHOR) - median(prior, CURRENT_ANCHOR)) * 0.5, -0.67, 0.67)

RECENT_TREND_3H = recent_kp_trend(NOW)

# ---------- Forecast inputs ----------
wind_model = load_json(DATA / "swift-wind" / "latest.json", {}) or {}
wind_fc: list[dict[str, Any]] = []
for r in wind_model.get("forecast") or wind_model.get("records") or []:
    t = parse_time(r.get("time") or r.get("target_time") or r.get("start_time"))
    v = fnum(r.get("speed") or r.get("swift_cme_enhanced_speed") or r.get("predicted_speed"))
    if t and v is not None:
        wind_fc.append({"time": t, "speed": v})
wind_fc.sort(key=lambda r: r["time"])

wind_acc = load_json(DATA / "swift-wind" / "accuracy.json", {}) or {}
wind_latest_acc = wind_acc.get("latest") or (wind_acc.get("summary") or {}).get("last_24h") or {}
wind_n = int(wind_latest_acc.get("count", 0) or 0)
wind_hit50 = fnum(wind_latest_acc.get("hit_rate_50kms") or wind_latest_acc.get("enhanced_hit_rate_50"), None)
if wind_n < 12 or wind_hit50 is None:
    WIND_GATE = 0.45
else:
    WIND_GATE = clamp(0.35 + 0.55 * (wind_hit50 / 100.0), 0.40, 0.90)

bz_model = load_json(DATA / "swift-bz" / "latest.json", {}) or {}
bz_fc: list[dict[str, Any]] = []
for r in bz_model.get("forecast") or []:
    t = parse_time(r.get("time"))
    if t:
        bz_fc.append({
            "time": t,
            "south_prob": fnum(r.get("southward_bz_probability"), 35.0),
            "bz_min": fnum(r.get("bz_min_forecast"), -2.0),
            "bt": fnum(r.get("bt_forecast"), 5.0),
            "row_conf": fnum(r.get("confidence"), 0.30),
            "driver": str(r.get("driver") or ""),
        })
bz_fc.sort(key=lambda r: r["time"])

bz_ready = bz_model.get("readiness") or (bz_model.get("verification") or {}).get("readiness") or {}
bz_model_gate = fnum(bz_ready.get("kp_input_confidence"), None)
if bz_model_gate is None:
    # Legacy/fallback Bz products must be treated conservatively.
    v7 = (bz_model.get("verification") or {}).get("last_7d") or {}
    n = int(v7.get("count", 0) or 0)
    current_driver = str((bz_model.get("current") or {}).get("driver") or "").lower()
    if "fallback" in current_driver or n < 32:
        bz_model_gate = 0.22
    elif n < 100:
        bz_model_gate = 0.40
    else:
        bz_model_gate = 0.60


def forecast_wind(t: datetime) -> float:
    r = nearest(wind_fc, t, 2.2)
    if r:
        raw = r["speed"]
    else:
        recent = [x["value"] for x in WIND_OBS if NOW - timedelta(hours=6) <= x["time"] <= NOW]
        raw = median(recent, 400.0)
    # Pull low-confidence wind forecasts toward a neutral 425 km/s.
    return 425.0 + WIND_GATE * (raw - 425.0)


def forecast_bz_features(t: datetime) -> dict[str, float]:
    r = nearest(bz_fc, t, 2.2)
    if not r:
        return {"south_prob": 0.35, "south_bz": 0.0, "bt": 5.0, "gate": 0.10}
    gate = clamp(bz_model_gate * r["row_conf"], 0.08, 0.80)
    if "fallback" in r["driver"].lower():
        gate = min(gate, 0.16)
    p_raw = clamp(r["south_prob"] / 100.0, 0.0, 1.0)
    south_bz_raw = max(0.0, -r["bz_min"])
    bt_raw = max(2.0, r["bt"])
    return {
        "south_prob": 0.35 + gate * (p_raw - 0.35),
        "south_bz": gate * south_bz_raw,
        "bt": 5.0 + gate * (bt_raw - 5.0),
        "gate": gate,
    }


def dvdt(t: datetime, speed_fn) -> float:
    a = speed_fn(t - timedelta(hours=3))
    b = speed_fn(t)
    if a is None or b is None:
        return 0.0
    return (b - a) / 3.0

# ---------- Residual model ----------
# Features are deliberately modest. The model predicts Delta-Kp relative to a
# persistence/climatology anchor, not absolute Kp.
def feature_vector(v: float, bz: dict[str, float], dv_kmh: float) -> list[float]:
    v_excess = max(0.0, (v - 450.0) / 150.0)
    v_high = 1.0 / (1.0 + math.exp(-(v - 550.0) / 70.0))
    dv_pos = max(0.0, dv_kmh / 40.0)
    south = bz["south_bz"] / 8.0
    coupling = (v / 450.0) * bz["south_prob"] * (bz["south_bz"] / 6.0)
    bt_excess = max(0.0, (bz["bt"] - 5.0) / 8.0)
    return [1.0, v_excess, v_high, dv_pos, bz["south_prob"], south, coupling, bt_excess]

FEATURE_NAMES = ["bias", "v_excess", "v_high", "dv_pos", "south_prob", "south_bz", "coupling", "bt_excess"]
DEFAULT_BETA = [-0.15, 0.42, 0.18, 0.20, 0.20, 0.55, 0.28, 0.12]


def solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[piv][i]) < 1e-9:
            continue
        M[i], M[piv] = M[piv], M[i]
        d = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= d
        for r in range(n):
            if r == i:
                continue
            f = M[r][i]
            for j in range(i, n + 1):
                M[r][j] -= f * M[i][j]
    return [M[i][n] for i in range(n)]


def fit_residual_model() -> tuple[list[float], dict[str, Any]]:
    samples: list[tuple[list[float], float]] = []
    cutoff = NOW - timedelta(days=45)
    for target in KP_OBS:
        t = target["time"]
        if t < cutoff or t > NOW - timedelta(hours=1):
            continue
        v = observed_wind(t)
        if v is None:
            continue
        # Baseline may use only Kp observations strictly before target to avoid leakage.
        prior_vals = []
        for r in KP_OBS:
            age_h = (t - r["time"]).total_seconds() / 3600
            if 0 < age_h <= 12:
                w = math.exp(-age_h / 5.0)
                prior_vals.append((r["value"], w))
        if not prior_vals:
            continue
        base = sum(x*w for x,w in prior_vals) / sum(w for _,w in prior_vals)
        bz = observed_imf(t)
        prev_v = observed_wind(t - timedelta(hours=3))
        dv = 0.0 if prev_v is None else (v - prev_v) / 3.0
        x = feature_vector(v, bz, dv)
        y = target["value"] - base
        samples.append((x, y))

    if len(samples) < 48:
        return DEFAULT_BETA[:], {"status": "fallback", "fit_count": len(samples)}

    p = len(FEATURE_NAMES)
    lam = 1.8  # stronger ridge than old absolute-Kp model
    ATA = [[0.0] * p for _ in range(p)]
    ATy = [0.0] * p
    for x, y in samples:
        for i in range(p):
            ATy[i] += x[i] * y
            for j in range(p):
                ATA[i][j] += x[i] * x[j]
    for i in range(p):
        ATA[i][i] += lam
    beta = solve_linear(ATA, ATy)

    # Keep the operational model conservative and physically sensible.
    limits = [(-1.0, 0.8), (0.0, 1.2), (-0.2, 0.8), (0.0, 0.8),
              (-0.3, 1.0), (0.0, 1.6), (0.0, 1.2), (-0.2, 0.8)]
    beta = [clamp(beta[i], *limits[i]) for i in range(p)]
    errs = [sum(beta[i]*x[i] for i in range(p)) - y for x, y in samples]
    return beta, {
        "status": "fitted",
        "fit_count": len(samples),
        "mae_residual": round(mean([abs(e) for e in errs]), 3),
        "bias_residual": round(mean(errs), 3),
        "sample_start": iso(cutoff),
        "sample_end": iso(NOW),
    }


BETA, FIT_INFO = fit_residual_model()

# ---------- Forecast archive / bias correction ----------
old_archive = (load_json(ARCHIVE, {}) or {}).get("items", [])[-16000:]
scored: list[dict[str, Any]] = []
for a in old_archive:
    if a.get("scored"):
        scored.append(a)
        continue
    tt = parse_time(a.get("target_time"))
    if not tt or tt > NOW - timedelta(hours=1):
        scored.append(a)
        continue
    obs = observed_kp(tt)
    if obs is None:
        scored.append(a)
        continue
    pred = fnum(a.get("predicted_kp"), 0.0)
    err = pred - obs
    q = dict(a)
    q.update({
        "scored": True,
        "observed_kp": round(obs, 2),
        "error": round(err, 3),
        "abs_error": round(abs(err), 3),
        "hit_within_1kp": abs(err) <= 1.0,
        "hit_within_067kp": abs(err) <= 0.67,
        "g_scale_hit": g_scale(pred) == g_scale(obs),
    })
    scored.append(q)

def model_scored_for_day(day: int) -> list[dict[str, Any]]:
    cutoff = NOW - timedelta(days=30)
    return [a for a in scored if a.get("scored") and a.get("model") == MODEL and
            int(a.get("lead_day", 0)) == day and parse_time(a.get("target_time")) and
            parse_time(a["target_time"]) >= cutoff]


def robust_bias(errors: list[float]) -> float:
    """Median-centered Huber-style bias estimate, resistant to isolated storm misses."""
    if not errors:
        return 0.0
    med = statistics.median(errors)
    clipped = [clamp(e, med - 1.5, med + 1.5) for e in errors]
    return 0.65 * med + 0.35 * mean(clipped)


def lead_bias_correction(lead_h: float) -> dict[str, float]:
    day = max(1, min(3, int(max(0.0, lead_h) // 24) + 1))
    own = model_scored_for_day(day)
    errors = [fnum(x.get("error"), 0.0) for x in own]
    n = len(errors)
    bias = robust_bias(errors)
    # Gentle adaptation: avoid swinging from "too high" to "too low".
    if n < 16:
        alpha = 0.0
    elif n < 50:
        alpha = 0.15
    elif n < 150:
        alpha = 0.25
    else:
        alpha = 0.35
    correction = clamp(alpha * bias, -0.75, 0.75)
    return {"day": day, "bias": bias, "alpha": alpha, "correction": correction, "n": n, "source": "v0.8-self-only"}


def anchor_for_lead(lead_h: float) -> float:
    # Near term follows the latest observed Kp; by 72 h it relaxes toward the
    # recent 7-day median rather than toward zero.
    w = math.exp(-max(0.0, lead_h) / 30.0)
    return w * CURRENT_ANCHOR + (1.0 - w) * QUIET_BASELINE


def predict_raw(t: datetime) -> tuple[float, dict[str, Any]]:
    lead_h = max(0.0, (t - NOW).total_seconds() / 3600.0)
    anchor = anchor_for_lead(lead_h)
    v = forecast_wind(t)
    bz = forecast_bz_features(t)
    dv = dvdt(t, forecast_wind)
    x = feature_vector(v, bz, dv)
    driver = sum(BETA[i] * x[i] for i in range(len(BETA)))
    bc = lead_bias_correction(lead_h)

    # Observed-Kp trend matters most in the first 18 h, then rapidly decays.
    trend_steps = min(4.0, lead_h / 3.0)
    trend_delta = clamp(RECENT_TREND_3H * trend_steps * math.exp(-lead_h / 18.0), -0.85, 0.85)
    y = anchor + driver + trend_delta - bc["correction"]

    # Storm gate: without high-confidence southward IMF, high wind alone should not
    # jump Kp several units above the current observed state.
    storm_score = (
        0.35 * clamp((v - 500.0) / 250.0, 0.0, 1.0) +
        0.35 * clamp(bz["south_bz"] / 7.0, 0.0, 1.0) +
        0.20 * clamp((bz["south_prob"] - 0.45) / 0.35, 0.0, 1.0) +
        0.10 * clamp((bz["bt"] - 7.0) / 8.0, 0.0, 1.0)
    )
    if storm_score < 0.30:
        cap = max(anchor + 1.15, 3.0)
    elif storm_score < 0.50:
        cap = max(anchor + 1.8, 4.0)
    elif storm_score < 0.70:
        cap = max(anchor + 2.7, 5.3)
    else:
        cap = 9.0
    y = clamp(y, Kp_FLOOR, cap)

    return y, {
        "anchor_kp": round(anchor, 3),
        "current_anchor_kp": round(CURRENT_ANCHOR, 3),
        "quiet_7d_median_kp": round(QUIET_BASELINE, 3),
    "recent_trend_3h": round(RECENT_TREND_3H, 3),
        "driver_delta_kp": round(driver, 3),
        "recent_trend_3h": round(RECENT_TREND_3H, 3),
        "trend_delta_kp": round(trend_delta, 3),
        "wind_speed_effective": round(v, 1),
        "wind_gate": round(WIND_GATE, 3),
        "bz_gate": round(bz["gate"], 3),
        "southward_probability_effective": round(bz["south_prob"]*100, 1),
        "south_bz_effective": round(bz["south_bz"], 2),
        "storm_score": round(storm_score, 3),
        "storm_cap": round(cap, 2),
        "lead_bias": {k: round(v, 3) if isinstance(v, float) else v for k, v in bc.items()},
    }

# ---------- Build 72 h forecast ----------
start = NOW.replace(minute=0, second=0, microsecond=0)
start = start.replace(hour=(start.hour // 3) * 3)
if start < NOW:
    start += timedelta(hours=3)

forecast: list[dict[str, Any]] = []
for i in range(FORECAST_HOURS // STEP_HOURS):
    t = start + timedelta(hours=i * STEP_HOURS)
    raw, diag = predict_raw(t)
    kp = round_kp_third(raw)
    forecast.append({
        "time": iso(t),
        "start_time": iso(t),
        "end_time": iso(t + timedelta(hours=STEP_HOURS)),
        "lead_hours": round((t - NOW).total_seconds()/3600.0, 1),
        "kp": kp,
        "kp_raw": round(raw, 3),
        "g_scale": g_scale(kp),
        "diagnostics": diag,
    })

issued = iso(NOW)
for r in forecast:
    scored.append({
        "issued_at": issued,
        "target_time": r["time"],
        "lead_hours": r["lead_hours"],
        "lead_day": max(1, min(3, int(max(0, r["lead_hours"]) // 24) + 1)),
        "predicted_kp": r["kp"],
        "model": MODEL,
        "scored": False,
    })
scored = scored[-16000:]


def skill(rows_in: list[dict[str, Any]]) -> dict[str, Any]:
    rr = [x for x in rows_in if x.get("scored")]
    if not rr:
        return {"count": 0, "status": "learning"}
    errs = [fnum(x.get("error"), 0.0) for x in rr]
    return {
        "count": len(rr),
        "mae": round(mean([abs(e) for e in errs]), 3),
        "bias": round(mean(errs), 3),
        "rmse": round(math.sqrt(mean([e*e for e in errs])), 3),
        "hit_rate_within_1kp": round(100*sum(bool(x.get("hit_within_1kp")) for x in rr)/len(rr), 1),
        "hit_rate_within_067kp": round(100*sum(bool(x.get("hit_within_067kp")) for x in rr)/len(rr), 1),
        "g_scale_hit_rate": round(100*sum(bool(x.get("g_scale_hit")) for x in rr)/len(rr), 1),
    }

cut30 = NOW - timedelta(days=30)
new_scored = [x for x in scored if x.get("scored") and x.get("model") == MODEL and
              parse_time(x.get("target_time")) and parse_time(x["target_time"]) >= cut30]
leadtime = {
    "updated_at": iso(NOW),
    "model": MODEL,
    "overall": skill(new_scored),
    "by_lead_day": [
        {"lead_day": d, **skill([x for x in new_scored if int(x.get("lead_day", 0)) == d])}
        for d in range(1, 4)
    ],
    "adaptive_bias": {
        str(d): lead_bias_correction((d-1)*24 + 12) for d in range(1, 4)
    },
}

# Simple recent observed-vs-anchor diagnostic (not future forecast skill).
diag_rows = []
for r in KP_OBS:
    if NOW - timedelta(days=7) <= r["time"] <= NOW - timedelta(hours=1):
        diag_rows.append(r)
verification = {
    "updated_at": iso(NOW),
    "model": MODEL,
    "note": "Primary model quality metric is forecast-archive leadtime_skill. Residual-fit diagnostics are secondary.",
    "observed_kp_records_total": len(KP_OBS),
    "observed_wind_records_total": len(WIND_OBS),
    "observed_imf_records_total": len(IMF_OBS),
    "current_anchor_kp": round(CURRENT_ANCHOR, 3),
    "quiet_7d_median_kp": round(QUIET_BASELINE, 3),
    "fit": FIT_INFO,
    "new_model_scored_30d": len(new_scored),
}

coef_payload = {
    "version": MODEL,
    "updated_at": iso(NOW),
    "feature_names": FEATURE_NAMES,
    "beta": [round(x, 6) for x in BETA],
    "fit": FIT_INFO,
    "kp_floor": Kp_FLOOR,
    "forecast_hours": FORECAST_HOURS,
    "wind_gate": round(WIND_GATE, 3),
    "bz_model_gate": round(float(bz_model_gate), 3),
}

latest = {
    "updated_at": iso(NOW),
    "model": MODEL,
    "forecast_hours": FORECAST_HOURS,
    "forecast_days": 3,
    "step_hours": STEP_HOURS,
    "kp_floor": Kp_FLOOR,
    "current_observed_anchor": round(CURRENT_ANCHOR, 3),
    "quiet_7d_median": round(QUIET_BASELINE, 3),
    "recent_trend_3h": round(RECENT_TREND_3H, 3),
    "input_confidence": {
        "wind_gate": round(WIND_GATE, 3),
        "wind_accuracy_count": wind_n,
        "wind_hit_rate_50": wind_hit50,
        "bz_model": bz_model.get("model"),
        "bz_model_gate": round(float(bz_model_gate), 3),
    },
    "forecast": forecast,
    "current": forecast[0] if forecast else None,
    "max_kp": max(forecast, key=lambda x: x["kp"]) if forecast else None,
    "leadtime_skill": leadtime,
    "verification": verification,
    "coefficients": coef_payload,
    "formula": "Kp = observed-Kp anchor + decaying observed trend + fitted Delta-Kp Wind/Bz drivers - v0.8 self-only robust bias correction",
}

save_json(LATEST, latest)
save_json(FORECAST_JSON, {"updated_at": iso(NOW), "model": MODEL, "forecast": forecast})
save_json(VERIFICATION, verification)
save_json(LEAD_SKILL, leadtime)
save_json(ARCHIVE, {"updated_at": iso(NOW), "items": scored})
save_json(COEF, coef_payload)
save_json(INDEX, {
    "updated_at": iso(NOW),
    "latest": "latest.json",
    "forecast": "forecast.json",
    "verification": "verification.json",
    "leadtime_skill": "leadtime-skill.json",
    "archive": "forecast-archive.json",
    "coefficients": "coefficients.json",
})
TXT.write_text("\n".join([
    ":Product: SWIFT 72-hour Kp Forecast v0.8",
    f":Issued: {NOW.strftime('%Y %b %d %H%M UTC')}",
    f"# Kp floor={Kp_FLOOR}; adaptive residual model; self-only robust bias; confidence-gated Wind/Bz",
] + [f"{r['time']}  Kp={r['kp']:.2f}  {r['g_scale']}" for r in forecast]) + "\n", encoding="utf-8")

print(json.dumps({
    "model": MODEL,
    "current_observed_anchor": round(CURRENT_ANCHOR, 3),
    "quiet_7d_median": round(QUIET_BASELINE, 3),
    "fit": FIT_INFO,
    "wind_gate": round(WIND_GATE, 3),
    "bz_model_gate": round(float(bz_model_gate), 3),
    "first_forecast": forecast[0] if forecast else None,
    "max_kp": latest["max_kp"],
}, ensure_ascii=False, indent=2))
