#!/usr/bin/env python3
"""SWIFT-RB-CHARGE v1.6.1: GEO >2 MeV electron forecast + charging scenarios + verification.

Inputs (all optional/tolerant):
  docs/data/swift-kp/latest.json
  docs/data/swift-bz/latest.json
  docs/data/swift-wind/latest.json
  docs/data/noaa/{kp,mag,wind}_history.json
  docs/data/satellite-risk/latest.json
  docs/data/swift-charging/observed.json

Outputs:
  docs/data/swift-charging/latest.json
  docs/data/swift-charging/estimated-history.json (observation-driven estimates; never truth)
  docs/data/swift-charging/electron-history.json (30-day UI electron observations)
  docs/data/swift-charging/electron-research-history.json (longer training archive)
  docs/data/swift-charging/electron-model.json (lagged empirical + AI residual model)
  docs/data/swift-charging/model.json
  docs/data/swift-charging/forecast-archive.json
  docs/data/swift-charging/verification.json
  docs/data/swift-charging/observed.json  (pruned to latest 30 days)

Important scientific boundary:
- The forecast values are reference-spacecraft estimates, not direct GOES bus-potential telemetry.
- Surface charging uses a reference current-balance model driven by GOES low-energy differential particle flux when available; it remains a scenario estimate until validated spacecraft-potential observations are supplied.
- Internal charging is an equivalent reference-dielectric field, not a measured internal field in GOES hardware.
- Verification is counted only against rows in observed.json. Model-generated values never score themselves.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from urllib.request import Request, urlopen
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
    from sklearn.linear_model import Ridge, HuberRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:
    np = None
    Ridge = None
    HuberRegressor = None
    HistGradientBoostingRegressor = None

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
DATA = ROOT / "docs" / "data"
OUT = DATA / "swift-charging"
OUT.mkdir(parents=True, exist_ok=True)

FORECAST_HOURS = 72
STEP_HOURS = 3
ARCHIVE_DAYS = 45
OBS_KEEP_DAYS = 30
TRAIN_DAYS = 180
VERIFY_DAYS = 30
NOMINAL_LEADS = (24, 48, 72)
LEAD_TOL_H = 4.5
OBS_MATCH_H = 1.6

# Reference model coefficients. The fitted model keeps this same feature definition.
DEFAULT_SURFACE = {
    "intercept": 0.15,
    "kp": 0.38,
    "south_bz_nt": 0.10,
    "wind_excess_100kms": 0.22,
}
DEFAULT_INTERNAL = {
    "intercept": 0.03,
    "log10_e2mev_above_100": 0.075,
    "kp_excess_3": 0.028,
    "wind_excess_100kms": 0.018,
    "fluence_24h_1e8": 0.020,
}
DEFAULT_DIFFERENTIAL_RATIO = 0.38
RIDGE_LAMBDA = 1.2
MIN_TRAIN = 12
ELECTRON_UI_KEEP_DAYS = 30
ELECTRON_RESEARCH_KEEP_DAYS = 730
ELECTRON_MIN_TRAIN = 96
ELECTRON_LAGS_H = (0, 3, 6, 12, 24, 48, 72)
ELECTRON_MEMORY_H = (6, 24, 72)
ELECTRON_FLOOR = 1.0
MATERIAL_SCENARIO = {
    "name": "reference-GEO-dielectric-v1",
    "epsilon_r": 3.2,
    "conductivity_s_per_m": 1.0e-16,
    "shield_transport_response": 0.20,
    "j_ref_a_per_m2_at_1e4": 5.0e-10,
    "flux_ref_cm2_s_sr": 1.0e4,
    "flux_power": 0.85,
    "max_field_mvm": 5.0,
}


SURFACE_SCENARIO = {
    "name": "reference-GEO-surface-current-balance-v1",
    # Differential flux products are treated as omnidirectional particle flux.
    # The collection factor converts that flux to an effective incident surface flux.
    "collection_factor": 0.25,
    "photoelectron_current_a_m2": 4.0e-5,
    "photoelectron_characteristic_v": 2.0,
    "secondary_yield_max": 1.2,
    "secondary_yield_peak_keV": 0.35,
    "backscatter_fraction": 0.12,
    "areal_leakage_resistance_ohm_m2": 2.0e10,
    "min_potential_kv": -30.0,
    "max_potential_kv": 5.0,
}
DIFFERENTIAL_SURFACE_SCENARIO = {
    "name": "shadowed-low-emission-surface-v1",
    "collection_factor": 0.25,
    "photoelectron_current_a_m2": 2.0e-7,
    "photoelectron_characteristic_v": 2.0,
    "secondary_yield_max": 0.75,
    "secondary_yield_peak_keV": 0.45,
    "backscatter_fraction": 0.08,
    "areal_leakage_resistance_ohm_m2": 8.0e10,
    "min_potential_kv": -30.0,
    "max_potential_kv": 5.0,
}
SURFACE_PLASMA_KEEP_DAYS = 30
SURFACE_PLASMA_STALE_HOURS = 12.0
ELEMENTARY_CHARGE_C = 1.602176634e-19



def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso_z(d: datetime) -> str:
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def load_json(path: Path, default: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if path.parent == OUT:
            raise RuntimeError(f"Refusing to overwrite unreadable charging archive: {path}") from exc
    return default


def rows(obj: Any, keys: Iterable[str] = ("forecast", "records", "items", "history", "data", "kp_history", "bz_history", "wind_history")) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in keys:
            if isinstance(obj.get(k), list):
                return [x for x in obj[k] if isinstance(x, dict)]
    return []


def first_json(paths: list[Path]) -> Any:
    for p in paths:
        x = load_json(p)
        if x is not None:
            return x
    return None


def series_from(obj: Any, value_names: list[str]) -> list[tuple[datetime, float]]:
    out = []
    for r in rows(obj):
        t = parse_time(r.get("time") or r.get("target_time") or r.get("time_tag") or r.get("timestamp") or r.get("datetime"))
        if not t:
            continue
        v = None
        for k in value_names:
            v = num(r.get(k))
            if v is not None:
                break
        if v is not None:
            out.append((t, v))
    out.sort(key=lambda x: x[0])
    return out


def nearest(series: list[tuple[datetime, float]], t: datetime, max_hours: float = 5.0) -> float | None:
    if not series:
        return None
    best = min(series, key=lambda x: abs((x[0] - t).total_seconds()))
    if abs((best[0] - t).total_seconds()) > max_hours * 3600:
        return None
    return best[1]


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Small dense Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [list(map(float, a[i])) + [float(b[i])] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            raise ValueError("singular")
        m[c], m[piv] = m[piv], m[c]
        div = m[c][c]
        m[c] = [x / div for x in m[c]]
        for r in range(n):
            if r == c:
                continue
            f = m[r][c]
            if f == 0:
                continue
            m[r] = [m[r][j] - f * m[c][j] for j in range(n + 1)]
    return [m[i][-1] for i in range(n)]


def ridge_fit(xrows: list[list[float]], y: list[float], lam: float = RIDGE_LAMBDA) -> list[float] | None:
    if not xrows or len(xrows) != len(y):
        return None
    p = len(xrows[0])
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for x, yy in zip(xrows, y):
        for i in range(p):
            xty[i] += x[i] * yy
            for j in range(p):
                xtx[i][j] += x[i] * x[j]
    for i in range(1, p):  # do not regularize intercept
        xtx[i][i] += lam
    try:
        return solve_linear(xtx, xty)
    except Exception:
        return None


def load_drivers() -> tuple[list[tuple[datetime, float]], list[tuple[datetime, float]], list[tuple[datetime, float]], dict[str, Any]]:
    kp_obj = load_json(DATA / "swift-kp" / "latest.json", {}) or {}
    bz_obj = load_json(DATA / "swift-bz" / "latest.json", {}) or {}
    wind_obj = load_json(DATA / "swift-wind" / "latest.json", {}) or {}
    kp = series_from(kp_obj, ["kp", "kp_forecast", "estimated_kp"])
    bz = series_from(bz_obj, ["bz_min_forecast", "bz_forecast", "bz_min", "bz"])
    wind = series_from(wind_obj, ["predicted_speed", "swift_cme_enhanced_speed", "forecast_speed", "speed"])
    return kp, bz, wind, {"kp": kp_obj, "bz": bz_obj, "wind": wind_obj}


def load_history_drivers() -> tuple[list[tuple[datetime, float]], list[tuple[datetime, float]], list[tuple[datetime, float]]]:
    kp = series_from(first_json([DATA / "noaa" / "kp_history.json", DATA / "noaa_kp_history.json"]) or {}, ["kp", "kp_index", "estimated_kp"])
    bz = series_from(first_json([DATA / "noaa" / "mag_history.json", DATA / "noaa_imf_history.json"]) or {}, ["bz", "bz_gsm", "bz_min"])
    wind = series_from(first_json([DATA / "noaa" / "wind_history.json", DATA / "noaa_wind_history.json"]) or {}, ["speed", "observed_speed", "solar_wind_speed"])
    return kp, bz, wind



def _fetch_goes_electron_7d() -> tuple[list[tuple[datetime, float]], str]:
    if os.environ.get("SWIFT_CHARGE_OFFLINE") == "1":
        return [], "offline"
    try:
        req = Request(
            "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json",
            headers={"User-Agent": "SWIFT-RB-CHARGE/1.6"},
        )
        with urlopen(req, timeout=25) as response:
            payload = json.load(response)
        out = []
        for row in rows(payload):
            if str(row.get("energy", "")).replace(" ", "").lower() != ">=2mev":
                continue
            if row.get("quality_flag") not in (None, 0, "0"):
                continue
            t = parse_time(row.get("time_tag"))
            v = num(row.get("flux"))
            if t and v is not None and v >= 0:
                out.append((t, v))
        return out, f"NOAA fetched {len(out)} valid points"
    except Exception as exc:
        return [], f"NOAA fetch failed: {type(exc).__name__}: {exc}"


def _dedupe_series(series: list[tuple[datetime, float]], cutoff: datetime, now: datetime) -> list[tuple[datetime, float]]:
    d: dict[str, tuple[datetime, float]] = {}
    for t, v in series:
        if cutoff <= t <= now + timedelta(hours=1) and v is not None and math.isfinite(v) and v >= 0:
            d[iso_z(t)] = (t, float(v))
    return sorted(d.values(), key=lambda x: x[0])


def refresh_electron_histories(now: datetime) -> dict[str, Any]:
    """Maintain a 30-day UI archive and a separate long training archive.

    The training archive is intentionally not displayed as charging truth. It stores
    the measured >2 MeV electron environment used to train/verify the electron model.
    """
    ui_path = OUT / "electron-history.json"
    research_path = OUT / "electron-research-history.json"
    ui_old = series_from(load_json(ui_path, {}) or {}, ["electron_flux_gt2mev"])
    research_old = series_from(load_json(research_path, {}) or {}, ["electron_flux_gt2mev"])
    fetched, status = _fetch_goes_electron_7d()

    extra = []
    for path in (DATA / "noaa" / "electron_history.json", DATA / "noaa" / "electron_gt2mev_history.json"):
        extra += series_from(load_json(path, {}) or {}, ["electron_flux_gt2mev", "electron_gt2mev", "flux"])

    sat = load_json(DATA / "satellite-risk" / "latest.json", {}) or {}
    item = ((sat.get("goes_space_environment_now") or {}).get("electron_gt2mev") or {})
    if isinstance(item, dict):
        t = parse_time(item.get("time") or item.get("time_tag") or item.get("timestamp"))
        v = num(item.get("flux"), num(item.get("value")))
        if t and v is not None and v >= 0:
            extra.append((t, v))

    all_series = research_old + ui_old + fetched + extra
    research = _dedupe_series(all_series, now - timedelta(days=ELECTRON_RESEARCH_KEEP_DAYS), now)
    ui = _dedupe_series(research, now - timedelta(days=ELECTRON_UI_KEEP_DAYS), now)

    atomic_json(research_path, {
        "updated_at": iso_z(now),
        "retention_days": ELECTRON_RESEARCH_KEEP_DAYS,
        "purpose": "electron forecast training/verification; not spacecraft charging truth",
        "records": [{"time": iso_z(t), "electron_flux_gt2mev": v} for t, v in research],
    })
    atomic_json(ui_path, {
        "updated_at": iso_z(now),
        "retention_days": ELECTRON_UI_KEEP_DAYS,
        "purpose": "UI display of measured GOES >2 MeV electron environment",
        "fetch_status": status,
        "records": [{"time": iso_z(t), "electron_flux_gt2mev": v} for t, v in ui],
    })
    return {"ui": ui, "research": research, "fetch_status": status}


def current_e2_flux(electron_series: list[tuple[datetime, float]] | None = None) -> float | None:
    electron_series = electron_series or []
    now = utcnow()
    recent = [(t, v) for t, v in electron_series if t <= now and now - t <= timedelta(hours=3)]
    if recent:
        return max(recent, key=lambda x: x[0])[1]
    sat = load_json(DATA / "satellite-risk" / "latest.json", {}) or {}
    candidates = [
        (((sat.get("goes_space_environment_now") or {}).get("electron_gt2mev") or {}).get("flux")),
        (((sat.get("goes_space_environment_now") or {}).get("electron_gt2mev") or {}).get("value")),
        sat.get("electron_gt2mev_flux"),
    ]
    for v in candidates:
        x = num(v)
        if x is not None and x > 0:
            return x
    return None


def _hourly_mean(series: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    bins: dict[datetime, list[float]] = {}
    for t, v in series:
        h = t.replace(minute=0, second=0, microsecond=0)
        bins.setdefault(h, []).append(v)
    out = []
    for t in sorted(bins):
        vals = [x for x in bins[t] if math.isfinite(x) and x >= 0]
        if vals:
            out.append((t, sum(vals) / len(vals)))
    return out


def _past_value(series: list[tuple[datetime, float]], t: datetime, max_age_h: float) -> float | None:
    pts = [(tt, v) for tt, v in series if tt <= t and (t - tt).total_seconds() <= max_age_h * 3600]
    return max(pts, key=lambda x: x[0])[1] if pts else None


def _ewma_past(series: list[tuple[datetime, float]], t: datetime, tau_h: float, lookback_h: float | None = None) -> tuple[float | None, float]:
    lookback_h = lookback_h or max(24.0, 4.0 * tau_h)
    pts = [(tt, v) for tt, v in series if tt <= t and (t - tt).total_seconds() <= lookback_h * 3600]
    if not pts:
        return None, 0.0
    weights, vals = [], []
    for tt, v in pts:
        age = (t - tt).total_seconds() / 3600.0
        weights.append(math.exp(-age / max(1e-6, tau_h)))
        vals.append(v)
    den = sum(weights)
    if den <= 0:
        return None, 0.0
    # rough hourly coverage relative to requested lookback
    coverage = min(1.0, len({tt.replace(minute=0, second=0, microsecond=0) for tt, _ in pts}) / max(1.0, lookback_h))
    return sum(w * v for w, v in zip(weights, vals)) / den, coverage


def _history_driver_series() -> dict[str, list[tuple[datetime, float]]]:
    kp_obj = first_json([DATA / "noaa" / "kp_history.json", DATA / "noaa_kp_history.json"]) or {}
    mag_obj = first_json([DATA / "noaa" / "mag_history.json", DATA / "noaa_imf_history.json"]) or {}
    wind_obj = first_json([DATA / "noaa" / "wind_history.json", DATA / "noaa_wind_history.json"]) or {}
    kp = series_from(kp_obj, ["kp", "kp_index", "estimated_kp"])
    bz = series_from(mag_obj, ["bz", "bz_gsm", "bz_min"])
    wind = series_from(wind_obj, ["speed", "observed_speed", "solar_wind_speed"])
    density = series_from(wind_obj, ["density", "proton_density", "density_cm3"])
    return {"kp": kp, "bz": bz, "wind": wind, "density": density}


def _feature_vector(issue_t: datetime, e_hour: list[tuple[datetime, float]], drivers: dict[str, list[tuple[datetime, float]]]) -> tuple[list[float] | None, dict[str, Any]]:
    features: list[float] = []
    meta: dict[str, Any] = {"issue_time": iso_z(issue_t), "coverage": {}}

    # Autoregressive electron memory.
    for lag in ELECTRON_LAGS_H:
        v = _past_value(e_hour, issue_t - timedelta(hours=lag), 1.6)
        if v is None or v < 0:
            return None, {"reason": f"missing electron lag {lag}h"}
        features.append(math.log10(max(ELECTRON_FLOOR, v)))

    # Driver memories. No future driver observation is used.
    for tau in ELECTRON_MEMORY_H:
        vv, cv = _ewma_past(drivers["wind"], issue_t, tau)
        nn, cn = _ewma_past(drivers["density"], issue_t, tau)
        bb, cb = _ewma_past([(t, max(0.0, -v)) for t, v in drivers["bz"]], issue_t, tau)
        kk, ck = _ewma_past(drivers["kp"], issue_t, tau)
        if vv is None or bb is None or kk is None:
            return None, {"reason": f"missing driver memory tau={tau}h"}
        if nn is None:
            nn = 5.0
            cn = 0.0
        pd = 1.6726e-6 * max(0.01, nn) * vv * vv
        features.extend([
            vv / 500.0,
            math.log(max(0.01, nn)),
            math.log(max(1e-4, pd)),
            bb / 10.0,
            kk / 5.0,
        ])
        meta["coverage"][str(tau)] = {"wind": cv, "density": cn, "south_bz": cb, "kp": ck}

    # Short-term pressure change / interactions.
    v0 = _past_value(drivers["wind"], issue_t, 2.0)
    n0 = _past_value(drivers["density"], issue_t, 2.0)
    v3 = _past_value(drivers["wind"], issue_t - timedelta(hours=3), 2.0)
    n3 = _past_value(drivers["density"], issue_t - timedelta(hours=3), 2.0)
    bz24, _ = _ewma_past([(t, max(0.0, -v)) for t, v in drivers["bz"]], issue_t, 24.0)
    wind24, _ = _ewma_past(drivers["wind"], issue_t, 24.0)
    dens24, _ = _ewma_past(drivers["density"], issue_t, 24.0)
    if all(x is not None for x in (v0, n0, v3, n3)):
        p0 = 1.6726e-6 * max(0.01, n0) * v0 * v0
        p3 = 1.6726e-6 * max(0.01, n3) * v3 * v3
        dlogp = max(0.0, math.log(max(1e-6, p0)) - math.log(max(1e-6, p3)))
    else:
        dlogp = 0.0
    features.extend([
        dlogp,
        (max(0.01, dens24 or 5.0) / 5.0) * ((wind24 or 425.0) / 500.0),
        ((bz24 or 0.0) / 10.0) * dlogp,
    ])

    # Low-DOF UT/season terms. For a fixed GEO longitude these are a repeatable MLT proxy.
    ut = issue_t.hour + issue_t.minute / 60.0
    doy = issue_t.timetuple().tm_yday
    features.extend([
        math.sin(2 * math.pi * ut / 24.0), math.cos(2 * math.pi * ut / 24.0),
        math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25),
    ])
    meta["feature_count"] = len(features)
    return features, meta


def _fit_ridge_standardized(X, y, alpha: float = 2.0):
    """Standardized robust empirical fit; Huber first, ridge fallback."""
    if np is None or Ridge is None:
        return None
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    model = None
    if HuberRegressor is not None:
        try:
            model = HuberRegressor(alpha=0.01, epsilon=1.35, max_iter=500)
            model.fit(Xs, y)
        except Exception:
            model = None
    if model is None:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Xs, y)
    return model, mu, sd

def _predict_ridge(pack, X):
    model, mu, sd = pack
    arr = np.asarray(X, dtype=float)
    return model.predict((arr - mu) / sd)


def _electron_training_samples(e_series: list[tuple[datetime, float]], now: datetime) -> tuple[list[datetime], list[list[float]], dict[int, list[float]], list[dict[str, Any]]]:
    e_series = [(t,v) for t,v in e_series if t >= now-timedelta(days=TRAIN_DAYS+5) and t <= now]
    e_hour = _hourly_mean(e_series)
    drivers = _history_driver_series()
    drivers = {k:[(t,v) for t,v in ser if t >= now-timedelta(days=TRAIN_DAYS+5) and t <= now] for k,ser in drivers.items()}
    issue_times, X, y_by_lead, metas = [], [], {h: [] for h in range(3, FORECAST_HOURS + 1, STEP_HOURS)}, []
    # Use only times old enough that all requested targets are already observed.
    candidates = [t for t, _ in e_hour if t <= now - timedelta(hours=FORECAST_HOURS) and t.hour % STEP_HOURS == 0]
    for t in candidates:
        fv, meta = _feature_vector(t, e_hour, drivers)
        if fv is None:
            continue
        targets = []
        ok = True
        for h in y_by_lead:
            v = _past_value(e_hour, t + timedelta(hours=h), 1.6)
            if v is None:
                ok = False
                break
            targets.append(math.log10(max(ELECTRON_FLOOR, v)))
        if not ok:
            continue
        issue_times.append(t); X.append(fv); metas.append(meta)
        for h, yy in zip(y_by_lead, targets):
            y_by_lead[h].append(yy)
    return issue_times, X, y_by_lead, metas


def _expanding_oof(X, y) -> tuple[list[int], list[float]]:
    n = len(y)
    if np is None or n < 60:
        return [], []
    preds, idxs = [], []
    # Expanding windows: 50->65, 65->80, 80->100 percent.
    cuts = [(0.50, 0.65), (0.65, 0.80), (0.80, 1.00)]
    for a, b in cuts:
        tr_end = max(30, int(n * a))
        va_end = max(tr_end + 1, int(n * b))
        if tr_end >= n or va_end <= tr_end:
            continue
        pack = _fit_ridge_standardized(X[:tr_end], y[:tr_end])
        if not pack:
            continue
        pp = _predict_ridge(pack, X[tr_end:va_end])
        idxs.extend(range(tr_end, va_end))
        preds.extend([float(v) for v in pp])
    return idxs, preds


def _fit_residual_ai(Xr, yr):
    """Chronological residual AI: early OOF residuals train AI; later OOF residuals choose shrinkage."""
    if np is None or HistGradientBoostingRegressor is None or len(yr) < 60:
        return None, 0.0, {"status":"insufficient_oof"}
    Xr = np.asarray(Xr, dtype=float)
    yr = np.asarray(yr, dtype=float)
    cut = max(36, int(len(yr) * 0.70))
    if cut >= len(yr) - 12:
        return None, 0.0, {"status":"insufficient_calibration_tail"}
    ai_cal = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=100, max_depth=3,
        min_samples_leaf=max(12, cut//20), l2_regularization=1.0, random_state=42,
    )
    ai_cal.fit(Xr[:cut], yr[:cut])
    g = ai_cal.predict(Xr[cut:])
    yv = yr[cut:]
    den = float(np.dot(g, g))
    w = float(np.clip(np.dot(yv, g) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
    base_mae = float(np.mean(np.abs(yv)))
    corrected = yv - w * g
    corrected_mae = float(np.mean(np.abs(corrected)))
    if not math.isfinite(corrected_mae) or corrected_mae >= base_mae:
        w = 0.0
        corrected = yv
        corrected_mae = base_mae
    ai_final = None
    if w > 0.05:
        ai_final = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=100, max_depth=3,
            min_samples_leaf=max(12, len(yr)//20), l2_regularization=1.0, random_state=42,
        )
        ai_final.fit(Xr, yr)
    meta = {
        "status":"active" if ai_final is not None else "rejected_no_skill_gain",
        "calibration_count": int(len(yv)),
        "base_mae_dex": round(base_mae,5),
        "corrected_mae_dex": round(corrected_mae,5),
        "weight": round(w,4),
        "corrected_residuals": [float(v) for v in corrected],
    }
    return ai_final, w, meta



def build_electron_model(e_series: list[tuple[datetime, float]], now: datetime) -> dict[str, Any]:
    issue_times, X, y_by_lead, _ = _electron_training_samples(e_series, now)
    lead_models: dict[str, Any] = {}
    for h in range(3, FORECAST_HOURS + 1, STEP_HOURS):
        y = y_by_lead[h]
        entry: dict[str, Any] = {
            "lead_hours": h, "sample_count": len(y), "status": "bootstrap_reference",
            "ai_active": False, "ai_weight": 0.0,
        }
        if len(y) >= ELECTRON_MIN_TRAIN and np is not None and Ridge is not None:
            pack = _fit_ridge_standardized(X, y)
            if pack:
                ridge, mu, sd = pack
                entry["status"] = "ridge_empirical"
                entry["ridge_intercept"] = float(ridge.intercept_)
                entry["ridge_coef"] = [float(v) for v in ridge.coef_]
                entry["feature_mean"] = [float(v) for v in mu]
                entry["feature_scale"] = [float(v) for v in sd]

                idxs, emp_oof = _expanding_oof(X, y)
                if idxs:
                    residuals = [float(y[i] - p) for i, p in zip(idxs, emp_oof)]
                    entry["oof_count"] = len(residuals)
                    # AI learns only OOF residuals. It is optional and shrunk by an OOF-derived weight.
                    Xr = np.asarray([X[i] for i in idxs], dtype=float)
                    yr = np.asarray(residuals, dtype=float)
                    ai, w, aimeta = _fit_residual_ai(Xr, yr)
                    entry["ai_active"] = bool(ai is not None and w > 0.05)
                    entry["ai_weight"] = round(w, 4)
                    entry["ai_training_count"] = len(yr)
                    entry["ai_kind"] = "HistGradientBoostingRegressor-time-ordered-residual"
                    entry["ai_validation"] = {k:v for k,v in aimeta.items() if k != "corrected_residuals"}
                    rr = np.asarray(aimeta.get("corrected_residuals") or residuals, dtype=float)
                    entry["residual_q05_dex"] = float(np.quantile(rr, 0.05))
                    entry["residual_q50_dex"] = float(np.quantile(rr, 0.50))
                    entry["residual_q95_dex"] = float(np.quantile(rr, 0.95))
        lead_models[str(h)] = entry

    payload = {
        "model_family": "lagged empirical electron forecast + optional OOF AI residual",
        "version": "SWIFT-RB-ELECTRON-v0.2",
        "updated_at": iso_z(now),
        "training_samples_common": len(X),
        "feature_definition": {
            "electron_lags_hours": list(ELECTRON_LAGS_H),
            "driver_memory_hours": list(ELECTRON_MEMORY_H),
            "drivers": ["Vsw", "log_density", "log_proton_dynamic_pressure", "southward_Bz", "Kp"],
            "interactions": ["positive_delta_log_Pd_3h", "density24*wind24", "southBz24*delta_log_Pd"],
            "seasonal": ["sin/cos UT", "sin/cos DOY"],
            "no_future_observation_rule": True,
        },
        "leads": lead_models,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    payload["model_id"] = f"SWIFT-RB-ELECTRON-v0.2-{digest}"
    atomic_json(OUT / "electron-model.json", payload)
    return payload


def _runtime_models_for_leads(e_series: list[tuple[datetime, float]], now: datetime):
    """Refit deterministic sklearn objects for this run; JSON stores audit coefficients/metadata."""
    issue_times, X, y_by_lead, _ = _electron_training_samples(e_series, now)
    runtime = {}
    if np is None or Ridge is None:
        return runtime, X, y_by_lead
    for h, y in y_by_lead.items():
        if len(y) < ELECTRON_MIN_TRAIN:
            continue
        pack = _fit_ridge_standardized(X, y)
        if not pack:
            continue
        idxs, emp_oof = _expanding_oof(X, y)
        ai = None; w = 0.0
        if idxs:
            residuals = np.asarray([y[i] - p for i, p in zip(idxs, emp_oof)], dtype=float)
            Xr = np.asarray([X[i] for i in idxs], dtype=float)
            ai, w, _ = _fit_residual_ai(Xr, residuals)
        runtime[h] = {"ridge": pack, "ai": ai, "weight": w}
    return runtime, X, y_by_lead


def _bootstrap_electron_log(issue_t: datetime, lead_h: int, e_hour, drivers) -> float:
    f0 = _past_value(e_hour, issue_t, 2.0)
    y0 = math.log10(max(ELECTRON_FLOOR, f0 if f0 is not None else 100.0))
    kp24, _ = _ewma_past(drivers["kp"], issue_t, 24.0)
    bz24, _ = _ewma_past([(t, max(0.0, -v)) for t, v in drivers["bz"]], issue_t, 24.0)
    wind24, _ = _ewma_past(drivers["wind"], issue_t, 24.0)
    g = 0.22 * max(0.0, (kp24 or 2.0) - 3.0) + 0.025 * max(0.0, (bz24 or 0.0) - 2.0) + 0.16 * max(0.0, ((wind24 or 425.0) - 450.0) / 150.0)
    decay = math.exp(-lead_h / 48.0)
    target = clamp(2.0 + g, 1.5, 5.8)
    return decay * y0 + (1.0 - decay) * target


def electron_forecast_rows(e_series: list[tuple[datetime, float]], electron_model: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    e_hour = _hourly_mean(e_series)
    drivers = _history_driver_series()
    fv, fmeta = _feature_vector(now.replace(minute=0, second=0, microsecond=0), e_hour, drivers)
    runtime, _, _ = _runtime_models_for_leads(e_series, now)
    out = []
    current = current_e2_flux(e_series)
    for h in range(0, FORECAST_HOURS + 1, STEP_HOURS):
        t = now + timedelta(hours=h)
        if h == 0:
            flux = current if current is not None else 100.0
            y_emp = math.log10(max(ELECTRON_FLOOR, flux))
            delta = 0.0; w = 0.0; mode = "observed_initial" if current is not None else "neutral_initial"
            q05 = q95 = y_emp
        else:
            mode = "bootstrap_reference"
            y_emp = _bootstrap_electron_log(now, h, e_hour, drivers)
            delta = 0.0; w = 0.0
            entry = (electron_model.get("leads") or {}).get(str(h), {})
            rt = runtime.get(h)
            if fv is not None and rt:
                y_emp = float(_predict_ridge(rt["ridge"], [fv])[0])
                mode = "ridge_empirical"
                if rt["ai"] is not None and rt["weight"] > 0.05:
                    delta = float(rt["ai"].predict(np.asarray([fv], dtype=float))[0])
                    w = float(rt["weight"])
                    mode = "empirical_plus_ai_residual"
            y_hybrid = y_emp + w * delta
            q05 = y_hybrid + float(entry.get("residual_q05_dex", -0.45))
            q95 = y_hybrid + float(entry.get("residual_q95_dex", 0.45))
            flux = 10 ** clamp(y_hybrid, 0.0, 7.0)
        if h == 0:
            y_hybrid = y_emp
        out.append({
            "time": iso_z(t), "lead_hours": h,
            "electron_flux_gt2mev": round(float(flux), 4),
            "electron_log10_empirical": round(float(y_emp), 5),
            "electron_ai_delta_dex": round(float(delta), 5),
            "electron_ai_weight": round(float(w), 4),
            "electron_log10_hybrid": round(float(y_hybrid), 5),
            "electron_q05": round(10 ** clamp(float(q05), 0.0, 7.0), 4),
            "electron_q95": round(10 ** clamp(float(q95), 0.0, 7.0), 4),
            "electron_model_mode": mode,
            "feature_quality": fmeta,
        })
    return out


def _interp_flux(points: list[tuple[datetime, float]], t: datetime) -> float | None:
    pts = sorted(points, key=lambda x: x[0])
    if not pts:
        return None
    before = [p for p in pts if p[0] <= t]
    after = [p for p in pts if p[0] >= t]
    if not before or not after:
        p = before[-1] if before else after[0]
        if abs((p[0]-t).total_seconds()) <= 3.5*3600:
            return p[1]
        return None
    a, b = before[-1], after[0]
    if a[0] == b[0]:
        return a[1]
    if (t-a[0]).total_seconds() > 4*3600 or (b[0]-t).total_seconds() > 4*3600:
        return None
    f = (t-a[0]).total_seconds() / (b[0]-a[0]).total_seconds()
    return a[1] + f*(b[1]-a[1])


def exact_fluence(points: list[tuple[datetime, float]], start: datetime, end: datetime) -> tuple[float | None, float]:
    """Trapezoidal fluence over exact interval. Returns (fluence, coverage)."""
    if end <= start:
        return 0.0, 1.0
    knots = [start] + [t for t, _ in points if start < t < end] + [end]
    knots = sorted(set(knots))
    total = 0.0; covered = 0.0
    for a, b in zip(knots[:-1], knots[1:]):
        fa, fb = _interp_flux(points, a), _interp_flux(points, b)
        dt = (b-a).total_seconds()
        if fa is None or fb is None:
            continue
        total += 0.5*(fa+fb)*dt
        covered += dt
    duration = (end-start).total_seconds()
    coverage = covered/duration if duration > 0 else 1.0
    return (total if coverage >= 0.80 else None), coverage


def material_internal_field_forecast(e_observed: list[tuple[datetime, float]], electron_fc: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Exact first-order dielectric state update for a declared reference scenario."""
    eps0 = 8.8541878128e-12
    eps = eps0 * MATERIAL_SCENARIO["epsilon_r"]
    sigma = MATERIAL_SCENARIO["conductivity_s_per_m"]
    tau_s = eps / sigma if sigma > 0 else math.inf
    response = MATERIAL_SCENARIO["shield_transport_response"]
    jref = MATERIAL_SCENARIO["j_ref_a_per_m2_at_1e4"]
    fref = MATERIAL_SCENARIO["flux_ref_cm2_s_sr"]
    power = MATERIAL_SCENARIO["flux_power"]

    def j_eff(flux):
        return max(0.0, jref * response * (max(0.0, flux)/fref) ** power)

    # Initialize E by spinning up on up to 7 days of observed electron history.
    E = 0.0
    hist = [(t, v) for t, v in e_observed if now - timedelta(days=7) <= t <= now]
    hist_hour = _hourly_mean(hist)
    if len(hist_hour) >= 2:
        for (ta, fa), (tb, fb) in zip(hist_hour[:-1], hist_hour[1:]):
            dt = min((tb-ta).total_seconds(), 3*3600)
            j = j_eff(0.5*(fa+fb))
            if sigma > 0:
                decay = math.exp(-dt/tau_s)
                E = E*decay + (j/sigma)*(1-decay)
            else:
                E += j*dt/eps

    # Forecast at 3 h steps.
    out = []
    fc_points = [(parse_time(r["time"]), float(r["electron_flux_gt2mev"])) for r in electron_fc if parse_time(r.get("time"))]
    combined = [(t, v) for t, v in e_observed if now-timedelta(hours=48) <= t <= now] + fc_points
    for idx, r in enumerate(electron_fc):
        t = parse_time(r["time"])
        flux = float(r["electron_flux_gt2mev"])
        if idx > 0:
            prev_t = parse_time(electron_fc[idx-1]["time"])
            dt = (t-prev_t).total_seconds()
            j = j_eff(0.5*(float(electron_fc[idx-1]["electron_flux_gt2mev"]) + flux))
            if sigma > 0:
                decay = math.exp(-dt/tau_s)
                E = E*decay + (j/sigma)*(1-decay)
            else:
                E += j*dt/eps
        fluence, coverage = exact_fluence(combined, t-timedelta(hours=24), t)
        out.append({
            "time": r["time"], "lead_hours": r["lead_hours"],
            "internal_field_mvm": round(clamp(E/1e6, 0.0, MATERIAL_SCENARIO["max_field_mvm"]), 5),
            "electron_fluence_24h": None if fluence is None else round(fluence, 3),
            "fluence_coverage": round(coverage, 3),
            "j_eff_a_per_m2": round(j_eff(flux), 14),
            "material_scenario": MATERIAL_SCENARIO["name"],
        })
    return out

def _energy_kev(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    txt = str(v).strip().lower().replace(" ", "")
    # SWPC realtime differential products normally expose energy in keV, but
    # tolerate strings/ranges and explicit units.
    nums = []
    cur = ""
    for ch in txt:
        if ch.isdigit() or ch in ".eE+-":
            cur += ch
        else:
            if cur:
                try: nums.append(float(cur))
                except Exception: pass
                cur = ""
    if cur:
        try: nums.append(float(cur))
        except Exception: pass
    if not nums:
        return None
    x = sum(nums[:2]) / min(2, len(nums)) if len(nums) >= 2 and "-" in txt else nums[0]
    if "mev" in txt:
        x *= 1000.0
    elif "ev" in txt and "kev" not in txt:
        x /= 1000.0
    return x if math.isfinite(x) else None


def _flux_per_kev(row: dict[str, Any]) -> float | None:
    f = num(row.get("flux"))
    if f is None or f < 0:
        return None
    units = str(row.get("units") or row.get("unit") or "").lower()
    # SWPC GOES real-time differential particle JSON is typically
    # cm^-2 s^-1 keV^-1. Keep conversions explicit if metadata says otherwise.
    if "mev" in units and ("-1" in units or "/" in units):
        f /= 1000.0
    elif "ev" in units and "kev" not in units and ("-1" in units or "/" in units):
        f *= 1000.0
    return f


def _fetch_goes_low_energy(kind: str) -> tuple[list[dict[str, Any]], str]:
    if os.environ.get("SWIFT_CHARGE_OFFLINE") == "1":
        return [], "offline"
    fname = "differential-electrons-7-day.json" if kind == "electron" else "differential-protons-7-day.json"
    url = f"https://services.swpc.noaa.gov/json/goes/primary/{fname}"
    try:
        req = Request(url, headers={"User-Agent":"SWIFT-RB-CHARGE/1.6.1"})
        with urlopen(req, timeout=30) as response:
            payload = json.load(response)
        out = []
        for row in rows(payload):
            t = parse_time(row.get("time_tag") or row.get("time"))
            e = _energy_kev(row.get("energy"))
            f = _flux_per_kev(row)
            q = row.get("quality_flag")
            if not t or e is None or f is None:
                continue
            if q not in (None, 0, "0"):
                continue
            # MPS-LO range: 30 eV to 30 keV. Exclude the high-energy channels.
            if 0.03 <= e <= 30.0:
                out.append({"time":iso_z(t), "_t":t, "energy_kev":e, "flux_per_kev":f,
                            "satellite":row.get("satellite"), "source":fname})
        return out, f"{fname}: {len(out)} valid low-energy rows"
    except Exception as exc:
        return [], f"{fname} fetch failed: {type(exc).__name__}: {exc}"


def _channel_widths_kev(energies: list[float]) -> dict[float, float]:
    es = sorted(set(e for e in energies if e > 0))
    if not es:
        return {}
    if len(es) == 1:
        return {es[0]: max(0.01, es[0] * 0.12)}
    edges = []
    for a, b in zip(es[:-1], es[1:]):
        edges.append(math.sqrt(a*b))
    lo = es[0] * es[0] / edges[0]
    hi = es[-1] * es[-1] / edges[-1]
    full = [lo] + edges + [hi]
    return {e:max(1e-6, full[i+1]-full[i]) for i,e in enumerate(es)}


def _surface_spectrum_moment(rows_in: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows_in:
        return None
    # Multiple angular records may share a channel/time. Median them first.
    by_e: dict[float, list[float]] = {}
    for r in rows_in:
        e = num(r.get("energy_kev")); f = num(r.get("flux_per_kev"))
        if e is None or f is None or e <= 0 or f < 0:
            continue
        by_e.setdefault(round(e,6), []).append(f)
    if not by_e:
        return None
    widths = _channel_widths_kev(list(by_e.keys()))
    integral = 0.0
    weighted_e = 0.0
    for e, vals in by_e.items():
        vals = sorted(vals)
        n = len(vals)
        med = vals[n//2] if n % 2 else 0.5*(vals[n//2-1]+vals[n//2])
        dE = widths.get(e, max(0.01,e*0.12))
        part = max(0.0, med) * dE
        integral += part
        weighted_e += e * part
    char_e = weighted_e / integral if integral > 0 else 1.0
    return {"integral_flux_cm2_s": integral, "characteristic_energy_kev": char_e}


def _plasma_moments_from_rows(e_rows: list[dict[str, Any]], i_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 5-minute bins keep electron and ion observations aligned without inventing interpolation.
    def key5(t: datetime):
        minute = (t.minute // 5) * 5
        return t.replace(minute=minute, second=0, microsecond=0)
    eb: dict[datetime, list[dict[str,Any]]] = {}
    ib: dict[datetime, list[dict[str,Any]]] = {}
    for r in e_rows: eb.setdefault(key5(r["_t"]), []).append(r)
    for r in i_rows: ib.setdefault(key5(r["_t"]), []).append(r)
    out = []
    all_t = sorted(set(eb) | set(ib))
    for t in all_t:
        em = _surface_spectrum_moment(eb.get(t,[]))
        im = _surface_spectrum_moment(ib.get(t,[]))
        if not em and not im:
            continue
        out.append({
            "time":iso_z(t), "_t":t,
            "electron_integral_flux_cm2_s": em["integral_flux_cm2_s"] if em else None,
            "electron_characteristic_keV": em["characteristic_energy_kev"] if em else None,
            "ion_integral_flux_cm2_s": im["integral_flux_cm2_s"] if im else None,
            "ion_characteristic_keV": im["characteristic_energy_kev"] if im else None,
        })
    return out


def refresh_surface_plasma_history(now: datetime) -> dict[str, Any]:
    path = OUT / "surface-plasma-history.json"
    old = load_json(path,{}) or {}
    saved = {}
    for r in rows(old, ("records","items","data")):
        t = parse_time(r.get("time"))
        if t and t >= now-timedelta(days=SURFACE_PLASMA_KEEP_DAYS):
            saved[iso_z(t)] = {**r, "_t":t}
    er, es = _fetch_goes_low_energy("electron")
    ir, is_ = _fetch_goes_low_energy("ion")
    for r in _plasma_moments_from_rows(er,ir):
        if r["_t"] >= now-timedelta(days=SURFACE_PLASMA_KEEP_DAYS):
            saved[r["time"]] = r
    recs = [saved[k] for k in sorted(saved)]
    atomic_json(path,{
        "updated_at":iso_z(now),"retention_days":SURFACE_PLASMA_KEEP_DAYS,
        "source":"NOAA/SWPC GOES primary differential electrons/protons; low-energy channels <=30 keV",
        "electron_fetch_status":es,"ion_fetch_status":is_,
        "records":[{k:v for k,v in r.items() if k!="_t"} for r in recs],
    })
    return {"records":recs,"electron_fetch_status":es,"ion_fetch_status":is_}


def _latest_plasma(plasma_rows: list[dict[str,Any]], t: datetime, max_age_h: float = SURFACE_PLASMA_STALE_HOURS) -> dict[str,Any] | None:
    pts=[r for r in plasma_rows if isinstance(r.get("_t"),datetime) and r["_t"]<=t and (t-r["_t"]).total_seconds()<=max_age_h*3600]
    return max(pts,key=lambda r:r["_t"]) if pts else None


def _secondary_yield(energy_kev: float, scenario: dict[str,Any]) -> float:
    e=max(1e-5,energy_kev)
    emax=max(1e-5,float(scenario["secondary_yield_peak_keV"]))
    dmax=max(0.0,float(scenario["secondary_yield_max"]))
    x=e/emax
    return max(0.0, min(5.0, dmax*x*math.exp(1.0-x)))


def _surface_current_components(v_kv: float, plasma: dict[str,float], scenario: dict[str,Any]) -> dict[str,float]:
    # Effective incident particle current density from omnidirectional differential flux.
    cf=float(scenario["collection_factor"])
    fe=max(0.0,float(plasma.get("electron_integral_flux_cm2_s") or 0.0))
    fi=max(0.0,float(plasma.get("ion_integral_flux_cm2_s") or 0.0))
    ee=max(0.03,float(plasma.get("electron_characteristic_keV") or 1.0))
    ei=max(0.03,float(plasma.get("ion_characteristic_keV") or 1.0))
    je0=ELEMENTARY_CHARGE_C*cf*fe*1.0e4
    ji0=ELEMENTARY_CHARGE_C*cf*fi*1.0e4

    # Maxwellian-like collection response using characteristic energies as scale parameters.
    if v_kv < 0:
        je=je0*math.exp(max(-50.0,v_kv/ee))
        ji=ji0*(1.0+min(100.0,abs(v_kv)/ei))
    else:
        je=je0*(1.0+min(100.0,v_kv/ee))
        ji=ji0*math.exp(max(-50.0,-v_kv/ei))

    delta=_secondary_yield(ee,scenario)
    jse=delta*je
    jbs=float(scenario["backscatter_fraction"])*je

    # Photoelectron emission is suppressed as the surface goes positive.
    jph0=float(scenario["photoelectron_current_a_m2"])*float(plasma.get("sunlit_fraction",1.0))
    vph=max(1e-6,float(scenario["photoelectron_characteristic_v"]))/1000.0 # kV
    jph=jph0 if v_kv<=0 else jph0*math.exp(-v_kv/vph)

    rarea=max(1.0,float(scenario["areal_leakage_resistance_ohm_m2"]))
    jleak=(v_kv*1000.0)/rarea

    net=ji+jph+jse+jbs-je-jleak
    return {"net_a_m2":net,"electron_a_m2":je,"ion_a_m2":ji,"photo_a_m2":jph,
            "secondary_a_m2":jse,"backscatter_a_m2":jbs,"leak_a_m2":jleak,
            "secondary_yield":delta}


def solve_surface_potential_kv(plasma: dict[str,float], scenario: dict[str,Any]) -> tuple[float,dict[str,float]]:
    lo=float(scenario["min_potential_kv"]); hi=float(scenario["max_potential_kv"])
    flo=_surface_current_components(lo,plasma,scenario)["net_a_m2"]
    fhi=_surface_current_components(hi,plasma,scenario)["net_a_m2"]
    if flo==0: return lo,_surface_current_components(lo,plasma,scenario)
    if fhi==0: return hi,_surface_current_components(hi,plasma,scenario)
    if flo*fhi < 0:
        a,b,fa=lo,hi,flo
        for _ in range(80):
            m=0.5*(a+b); fm=_surface_current_components(m,plasma,scenario)["net_a_m2"]
            if abs(fm)<1e-12 or (b-a)<1e-5:
                return m,_surface_current_components(m,plasma,scenario)
            if fa*fm<=0: b=m
            else: a=m; fa=fm
        m=0.5*(a+b); return m,_surface_current_components(m,plasma,scenario)

    # No bracketed root: choose minimum absolute current imbalance on a dense grid.
    best=None
    for j in range(351):
        v=lo+(hi-lo)*j/350.0
        comp=_surface_current_components(v,plasma,scenario)
        score=abs(comp["net_a_m2"])
        if best is None or score<best[0]: best=(score,v,comp)
    return best[1],best[2]


def surface_plasma_for_forecast(t: datetime, now: datetime, plasma_rows: list[dict[str,Any]], kp: float, bz: float, wind: float) -> tuple[dict[str,float],str]:
    base=_latest_plasma(plasma_rows,now)
    if base:
        # Future low-energy plasma is not observed. Persist the measured spectral moments
        # and apply bounded storm-response scaling to particle flux amplitudes only.
        kp0=2.0; south=max(0.0,-bz)
        e_scale=math.exp(clamp(0.10*(kp-kp0)+0.018*south+0.055*((wind-425.0)/100.0),-2.0,2.2))
        i_scale=math.exp(clamp(0.06*(kp-kp0)+0.010*south+0.035*((wind-425.0)/100.0),-1.5,1.6))
        return {
            "electron_integral_flux_cm2_s": max(0.0,float(base.get("electron_integral_flux_cm2_s") or 0.0))*e_scale,
            "electron_characteristic_keV": max(0.03,float(base.get("electron_characteristic_keV") or 1.0)),
            "ion_integral_flux_cm2_s": max(0.0,float(base.get("ion_integral_flux_cm2_s") or 0.0))*i_scale,
            "ion_characteristic_keV": max(0.03,float(base.get("ion_characteristic_keV") or 1.0)),
            "sunlit_fraction": 1.0,
        },"mps-lo-persistence+storm-scaling"
    # Explicit fallback: a scenario plasma, not a measurement.
    eflux=1.5e7*math.exp(clamp(0.12*(kp-2.0)+0.02*max(0.0,-bz)+0.05*((wind-425)/100),-2,2))
    iflux=8.0e6*math.exp(clamp(0.07*(kp-2.0)+0.01*max(0.0,-bz)+0.03*((wind-425)/100),-2,2))
    return {
        "electron_integral_flux_cm2_s":eflux,"electron_characteristic_keV":1.5+0.25*max(0,kp-2),
        "ion_integral_flux_cm2_s":iflux,"ion_characteristic_keV":2.0+0.2*max(0,kp-2),
        "sunlit_fraction":1.0,
    },"reference-plasma-proxy"



def observed_rows(now: datetime | None = None) -> list[dict[str, Any]]:
    """Load independent charging observations and enforce 30-day retention.

    These rows are the only verification truth. Forecast/model outputs are never
    copied into this file as observations.
    """
    now = now or utcnow()
    cutoff = now - timedelta(days=OBS_KEEP_DAYS)
    obj = load_json(OUT / "observed.json", {}) or {}
    out = []
    for r in rows(obj, ("observed", "records", "items", "data")):
        t = parse_time(r.get("time"))
        if not t or t < cutoff or t > now + timedelta(hours=2):
            continue
        if r.get("data_kind") in ("estimate", "reconstruction", "forecast"):
            continue
        out.append({
            "time": iso_z(t), "_t": t,
            "surface_kv": num(r.get("surface_kv")),
            "differential_kv": num(r.get("differential_kv")),
            "internal_field_mvm": num(r.get("internal_field_mvm"), num(r.get("internal_mvm"))),
            "electron_flux_gt2mev": num(r.get("electron_flux_gt2mev")),
            "source": r.get("source") or "charging observation/reference target",
        })
    out.sort(key=lambda r: r["_t"])
    return out


def persist_observed_rows(obs: list[dict[str, Any]], now: datetime) -> None:
    payload = {
        "updated_at": iso_z(now),
        "retention_days": OBS_KEEP_DAYS,
        "observation_rule": "Independent/validated charging observations only; model outputs are never written here as truth.",
        "observed": [{k: v for k, v in r.items() if k != "_t"} for r in obs],
    }
    (OUT / "observed.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fit_coefficients(obs: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    hk, hb, hw = load_history_drivers()
    train = [r for r in obs if r["_t"] >= now - timedelta(days=TRAIN_DAYS)]

    sx, sy = [], []
    ix, iy = [], []
    for r in train:
        kp = nearest(hk, r["_t"], 4.5)
        bz = nearest(hb, r["_t"], 4.5)
        wind = nearest(hw, r["_t"], 4.5)
        if kp is None or bz is None or wind is None:
            continue
        if r.get("surface_kv") is not None:
            sx.append([1.0, kp, max(0.0, -bz), max(0.0, (wind - 400.0) / 100.0)])
            sy.append(abs(float(r["surface_kv"])))
        if r.get("internal_field_mvm") is not None and r.get("electron_flux_gt2mev") not in (None, 0):
            f = max(1.0, float(r["electron_flux_gt2mev"]))
            ix.append([1.0, max(0.0, math.log10(f) - 2.0), max(0.0, kp - 3.0), max(0.0, (wind - 400.0) / 100.0)])
            iy.append(max(0.0, float(r["internal_field_mvm"])))

    surface = dict(DEFAULT_SURFACE)
    internal = dict(DEFAULT_INTERNAL)
    surface_status = "default_reference"
    internal_status = "default_reference"

    surface = {"model":"current_balance","scenario":SURFACE_SCENARIO["name"],"differential_scenario":DIFFERENTIAL_SURFACE_SCENARIO["name"]}
    surface_status = f"current_balance_scenario; validated_surface_targets_n={len(sx)}"

    if len(ix) >= MIN_TRAIN:
        beta = ridge_fit(ix, iy)
        if beta:
            internal.update({
                "intercept": clamp(beta[0], 0.0, 1.5),
                "log10_e2mev_above_100": clamp(beta[1], 0.0, 1.0),
                "kp_excess_3": clamp(beta[2], 0.0, 0.5),
                "wind_excess_100kms": clamp(beta[3], 0.0, 0.5),
            })
            internal_status = f"ridge_learned_n={len(ix)}"

    payload = {
        "surface": surface,
        "internal": internal,
        "differential_ratio": None,
        "training": {
            "window_days": TRAIN_DAYS,
            "surface_samples": len(sx),
            "internal_samples": len(ix),
            "surface_status": surface_status,
            "internal_status": internal_status,
        },
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    payload["model_id"] = f"SWIFT-CHARGE-v0.2-{digest}"
    payload["updated_at"] = iso_z(now)
    return payload


def driver_at(series: list[tuple[datetime, float]], t: datetime, default: float) -> float:
    v = nearest(series, t, 7.0)
    if v is not None:
        return v
    if series:
        # use nearest edge for sparse forecast files
        return min(series, key=lambda x: abs((x[0] - t).total_seconds()))[1]
    return default



def forecast_rows(model: dict[str, Any], now: datetime, electron_history: list[tuple[datetime, float]], electron_model: dict[str, Any], surface_plasma_rows: list[dict[str,Any]]) -> list[dict[str, Any]]:
    kp_s, bz_s, wind_s, raw = load_drivers()
    electron_fc = electron_forecast_rows(electron_history, electron_model, now)
    internal_fc = material_internal_field_forecast(electron_history, electron_fc, now)
    internal_by_t = {r["time"]: r for r in internal_fc}
    out = []
    for er in electron_fc:
        lead = int(er["lead_hours"])
        t = parse_time(er["time"])
        kp = clamp(driver_at(kp_s, t, 2.0), 0.0, 9.0)
        bz = clamp(driver_at(bz_s, t, -1.0), -80.0, 80.0)
        wind = clamp(driver_at(wind_s, t, 425.0), 200.0, 1800.0)

        plasma, surface_mode = surface_plasma_for_forecast(t, now, surface_plasma_rows, kp, bz, wind)
        surface_kv, surface_components = solve_surface_potential_kv(plasma, SURFACE_SCENARIO)
        differential_surface_kv, differential_components = solve_surface_potential_kv(plasma, DIFFERENTIAL_SURFACE_SCENARIO)
        differential_kv = clamp(abs(differential_surface_kv - surface_kv), 0.0, 35.0)
        ir = internal_by_t.get(er["time"], {})

        out.append({
            "time": er["time"], "lead_hours": lead,
            "surface_kv": round(surface_kv, 3),
            "differential_kv": round(differential_kv, 3),
            "surface_secondary_kv": round(differential_surface_kv, 3),
            "surface_model_mode": surface_mode,
            "surface_plasma_e_flux_cm2_s": round(float(plasma.get("electron_integral_flux_cm2_s") or 0.0),3),
            "surface_plasma_i_flux_cm2_s": round(float(plasma.get("ion_integral_flux_cm2_s") or 0.0),3),
            "surface_e_char_kev": round(float(plasma.get("electron_characteristic_keV") or 0.0),4),
            "surface_i_char_kev": round(float(plasma.get("ion_characteristic_keV") or 0.0),4),
            "surface_e_current_a_m2": round(float(surface_components.get("electron_a_m2") or 0.0),12),
            "surface_i_current_a_m2": round(float(surface_components.get("ion_a_m2") or 0.0),12),
            "surface_photo_current_a_m2": round(float(surface_components.get("photo_a_m2") or 0.0),12),
            "surface_secondary_yield": round(float(surface_components.get("secondary_yield") or 0.0),5),
            "internal_field_mvm": ir.get("internal_field_mvm"),
            "electron_flux_gt2mev": er["electron_flux_gt2mev"],
            "electron_flux_empirical": round(10 ** clamp(er["electron_log10_empirical"], 0.0, 7.0), 4),
            "electron_ai_delta_dex": er["electron_ai_delta_dex"],
            "electron_ai_weight": er["electron_ai_weight"],
            "electron_q05": er["electron_q05"],
            "electron_q95": er["electron_q95"],
            "electron_model_mode": er["electron_model_mode"],
            "electron_fluence_24h": ir.get("electron_fluence_24h"),
            "fluence_coverage": ir.get("fluence_coverage"),
            "j_eff_a_per_m2": ir.get("j_eff_a_per_m2"),
            "material_scenario": ir.get("material_scenario"),
            "kp": round(kp, 3),
            "bz_min_nt": round(bz, 3),
            "wind_kms": round(wind, 2),
            "source": "SWIFT-RB electron hybrid + reference charging scenarios",
        })
    return out

def load_archive() -> list[dict[str, Any]]:
    obj = load_json(OUT / "forecast-archive.json", {}) or {}
    return [r for r in rows(obj, ("items", "records", "data")) if isinstance(r, dict)]


def update_archive(archive: list[dict[str, Any]], fc: list[dict[str, Any]], model_id: str, now: datetime) -> list[dict[str, Any]]:
    new = []
    for r in archive:
        it = parse_time(r.get("issued_at"))
        tt = parse_time(r.get("target_time") or r.get("time"))
        if it and tt and it >= now - timedelta(days=ARCHIVE_DAYS):
            new.append(r)
    for r in fc:
        new.append({
            "issued_at": iso_z(now),
            "target_time": r["time"],
            "lead_hours": r["lead_hours"],
            "model": model_id,
            "surface_kv": r["surface_kv"],
            "differential_kv": r["differential_kv"],
            "surface_secondary_kv": r.get("surface_secondary_kv"),
            "surface_model_mode": r.get("surface_model_mode"),
            "internal_field_mvm": r["internal_field_mvm"],
            "electron_flux_gt2mev": r["electron_flux_gt2mev"],
            "electron_flux_empirical": r.get("electron_flux_empirical"),
            "electron_ai_delta_dex": r.get("electron_ai_delta_dex"),
            "electron_ai_weight": r.get("electron_ai_weight"),
            "electron_q05": r.get("electron_q05"),
            "electron_q95": r.get("electron_q95"),
        })
    # dedupe exact issue-target-model triple
    d = {}
    for r in new:
        key = (r.get("issued_at"), r.get("target_time"), r.get("model"))
        d[key] = r
    out = list(d.values())
    out.sort(key=lambda r: (str(r.get("issued_at")), str(r.get("target_time"))))
    return out


def nearest_observed(obs: list[dict[str, Any]], t: datetime) -> dict[str, Any] | None:
    if not obs:
        return None
    best = min(obs, key=lambda r: abs((r["_t"] - t).total_seconds()))
    if abs((best["_t"] - t).total_seconds()) > OBS_MATCH_H * 3600:
        return None
    return best


def metrics(pairs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    good = [p for p in pairs if num(p.get("error")) is not None]
    if not good:
        return {"count": 0, "status": "learning"}
    errs = [float(p["error"]) for p in good]
    ae = [abs(x) for x in errs]
    out = {
        "count": len(good),
        "status": "verified",
        "mae": round(sum(ae) / len(ae), 4),
        "bias": round(sum(errs) / len(errs), 4),
        "rmse": round(math.sqrt(sum(x * x for x in errs) / len(errs)), 4),
    }
    if key == "surface_kv":
        out["hit_rate_1kv"] = round(100 * sum(x <= 1.0 for x in ae) / len(ae), 1)
        out["hit_rate_2kv"] = round(100 * sum(x <= 2.0 for x in ae) / len(ae), 1)
    else:
        out["hit_rate_0_1mvm"] = round(100 * sum(x <= 0.1 for x in ae) / len(ae), 1)
        out["hit_rate_0_2mvm"] = round(100 * sum(x <= 0.2 for x in ae) / len(ae), 1)
    return out


def build_verification(archive: list[dict[str, Any]], obs: list[dict[str, Any]], model_id: str, now: datetime) -> dict[str, Any]:
    pairs_by_kind: dict[str, dict[str, list[dict[str, Any]]]] = {"surface": {}, "internal": {}}
    skill_surface, skill_internal = [], []
    for lead in NOMINAL_LEADS:
        candidates = []
        for r in archive:
            if r.get("model") != model_id:
                continue
            issued = parse_time(r.get("issued_at")); target = parse_time(r.get("target_time"))
            lh = num(r.get("lead_hours"))
            if not issued or not target or lh is None:
                continue
            if target < now - timedelta(days=VERIFY_DAYS) or target > now + timedelta(hours=1):
                continue
            if abs(lh - lead) > LEAD_TOL_H:
                continue
            o = nearest_observed(obs, target)
            if not o:
                continue
            candidates.append((target, abs(lh - lead), r, o))
        # one closest-to-nominal forecast per target
        chosen: dict[str, tuple] = {}
        for item in candidates:
            key = iso_z(item[0])
            if key not in chosen or item[1] < chosen[key][1]:
                chosen[key] = item
        sp, ip = [], []
        for target, _, r, o in chosen.values():
            pv = num(r.get("surface_kv")); ov = num(o.get("surface_kv"))
            if pv is not None and ov is not None:
                sp.append({
                    "issued_at": r.get("issued_at"), "target_time": iso_z(target), "lead_hours": r.get("lead_hours"),
                    "predicted": pv, "observed": ov, "error": round(pv - ov, 4), "model": model_id,
                })
            pv = num(r.get("internal_field_mvm")); ov = num(o.get("internal_field_mvm"))
            if pv is not None and ov is not None:
                ip.append({
                    "issued_at": r.get("issued_at"), "target_time": iso_z(target), "lead_hours": r.get("lead_hours"),
                    "predicted": pv, "observed": ov, "error": round(pv - ov, 5), "model": model_id,
                })
        sm = metrics(sp, "surface_kv"); sm["nominal_lead_hours"] = lead
        im = metrics(ip, "internal_field_mvm"); im["nominal_lead_hours"] = lead
        skill_surface.append(sm); skill_internal.append(im)
        pairs_by_kind["surface"][f"{lead}h"] = sp
        pairs_by_kind["internal"][f"{lead}h"] = ip

    return {
        "updated_at": iso_z(now),
        "model": model_id,
        "observation_rule": "Only docs/data/swift-charging/observed.json rows are verification truth; model outputs are never self-scored.",
        "nominal_lead_skill": {"surface": skill_surface, "internal": skill_internal},
        "lead_pairs": pairs_by_kind,
    }


def _electron_obs_near(e_series: list[tuple[datetime, float]], t: datetime, max_h: float = 1.6) -> float | None:
    if not e_series:
        return None
    best = min(e_series, key=lambda x: abs((x[0]-t).total_seconds()))
    if abs((best[0]-t).total_seconds()) > max_h*3600:
        return None
    return best[1]


def build_electron_verification(archive: list[dict[str, Any]], e_series: list[tuple[datetime, float]], now: datetime) -> dict[str, Any]:
    leads = []
    pairs = {}
    for lead in NOMINAL_LEADS:
        cand = []
        for r in archive:
            target = parse_time(r.get("target_time"))
            lh = num(r.get("lead_hours"))
            if not target or lh is None or abs(lh-lead) > LEAD_TOL_H:
                continue
            if target < now-timedelta(days=VERIFY_DAYS) or target > now+timedelta(hours=1):
                continue
            pred = num(r.get("electron_flux_gt2mev"))
            obs = _electron_obs_near(e_series, target)
            if pred is None or pred <= 0 or obs is None or obs <= 0:
                continue
            cand.append((target, abs(lh-lead), r, obs))
        chosen = {}
        for item in cand:
            key = iso_z(item[0])
            if key not in chosen or item[1] < chosen[key][1]:
                chosen[key] = item
        pp = []
        for target, _, r, obs in chosen.values():
            pred = float(r["electron_flux_gt2mev"])
            err = math.log10(max(ELECTRON_FLOOR,pred)) - math.log10(max(ELECTRON_FLOOR,obs))
            pp.append({
                "issued_at": r.get("issued_at"), "target_time": iso_z(target),
                "lead_hours": r.get("lead_hours"), "predicted": pred, "observed": obs,
                "error_dex": round(err, 5), "ratio": round(pred/obs, 5),
            })
        errs = [p["error_dex"] for p in pp]
        if errs:
            ae = [abs(e) for e in errs]
            m = {
                "nominal_lead_hours": lead, "count": len(errs), "status": "verified",
                "mae_dex": round(sum(ae)/len(ae),4),
                "bias_dex": round(sum(errs)/len(errs),4),
                "rmse_dex": round(math.sqrt(sum(e*e for e in errs)/len(errs)),4),
                "hit_rate_factor2": round(100*sum(e <= math.log10(2.0) for e in ae)/len(ae),1),
                "hit_rate_factor3": round(100*sum(e <= math.log10(3.0) for e in ae)/len(ae),1),
            }
        else:
            m = {"nominal_lead_hours":lead,"count":0,"status":"learning"}
        leads.append(m); pairs[f"{lead}h"] = pp
    return {"nominal_lead_skill": leads, "lead_pairs": pairs}



def past_snapshots(archive: list[dict[str, Any]], now: datetime, model_id: str) -> dict[str, list[dict[str, Any]]]:
    issues = sorted({r.get("issued_at") for r in archive if r.get("model") == model_id and r.get("issued_at")})
    out = {"surface": [], "internal": [], "electron": []}
    for age in NOMINAL_LEADS:
        target_issue = now - timedelta(hours=age)
        if not issues:
            continue
        chosen = min(issues, key=lambda s: abs((parse_time(s) - target_issue).total_seconds()) if parse_time(s) else 1e99)
        dt = parse_time(chosen)
        if not dt or abs((dt - target_issue).total_seconds()) > 6 * 3600:
            continue
        rr = [r for r in archive if r.get("issued_at") == chosen and r.get("model") == model_id]
        rr.sort(key=lambda r: str(r.get("target_time")))
        out["surface"].append({"ageHours": age, "issued_at": chosen, "rows": [{"time": r.get("target_time"), "surface_kv": r.get("surface_kv")} for r in rr]})
        out["internal"].append({"ageHours": age, "issued_at": chosen, "rows": [{"time": r.get("target_time"), "internal_field_mvm": r.get("internal_field_mvm")} for r in rr]})
        out["electron"].append({"ageHours": age, "issued_at": chosen, "rows": [{"time": r.get("target_time"), "electron_flux_gt2mev": r.get("electron_flux_gt2mev")} for r in rr]})
    return out



def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def build_estimated_history(model: dict[str, Any], now: datetime, surface_plasma_rows: list[dict[str,Any]]) -> dict[str, Any]:
    """Archive observation-driven estimates separately from independent truth.

    UTC 3h samples; no future samples, forecast substitution or neutral fillers.
    Internal field requires eight valid trailing 3h electron samples (24h).
    Previously generated points remain fixed when coefficients change.
    """
    cutoff = now - timedelta(days=OBS_KEEP_DAYS)
    hk, hb, hw = load_history_drivers()
    def past(series):
        return [(t, v) for t, v in series if cutoff - timedelta(days=1) <= t <= now]
    hk, hb, hw = map(past, (hk, hb, hw))
    electron_path = OUT / "electron-history.json"
    old_e = load_json(electron_path, {}) or {}
    es = series_from(old_e, ["electron_flux_gt2mev"])
    electron_fetch_status = str(old_e.get("fetch_status") or "maintained by refresh_electron_histories")
    path = OUT / "estimated-history.json"
    existing = load_json(path, {}) or {}
    saved = {}
    for r in rows(existing, ("estimated",)):
        t = parse_time(r.get("time"))
        if t and cutoff <= t <= now:
            saved[iso_z(t)] = r
    def sample(series, t, max_age):
        # Most recent observation at or before target; never look ahead.
        pts = [(tt, v) for tt, v in series if tt <= t and (t-tt).total_seconds() <= max_age*3600]
        return max(pts, key=lambda x: x[0])[1] if pts else None
    t = cutoff.replace(hour=(cutoff.hour//3)*3, minute=0, second=0, microsecond=0)
    i = model["internal"]
    while t <= now:
        if t < cutoff:
            t += timedelta(hours=3)
            continue
        key = iso_z(t)
        kp, bz, wind = sample(hk,t,3.1), sample(hb,t,1.5), sample(hw,t,1.5)
        if all(v is not None for v in (kp,bz,wind)):
            r = saved.get(key, {"time":key, "data_kind":"estimate", "model":model["model_id"],
                "source":"Observed environment -> reference charging model; NOT charging telemetry",
                "kp":kp, "bz_min_nt":bz, "wind_kms":wind})
            if r.get("surface_kv") is None:
                p=_latest_plasma(surface_plasma_rows,t,max_age_h=6.0)
                if p:
                    plasma={
                        "electron_integral_flux_cm2_s":p.get("electron_integral_flux_cm2_s"),
                        "electron_characteristic_keV":p.get("electron_characteristic_keV"),
                        "ion_integral_flux_cm2_s":p.get("ion_integral_flux_cm2_s"),
                        "ion_characteristic_keV":p.get("ion_characteristic_keV"),
                        "sunlit_fraction":1.0,
                    }
                    v1,c1=solve_surface_potential_kv(plasma,SURFACE_SCENARIO)
                    v2,c2=solve_surface_potential_kv(plasma,DIFFERENTIAL_SURFACE_SCENARIO)
                    r["surface_kv"]=round(v1,3)
                    r["surface_secondary_kv"]=round(v2,3)
                    r["differential_kv"]=round(clamp(abs(v2-v1),0,35),3)
                    r["surface_model_mode"]="measured-mps-lo-current-balance"
            flux_now = sample(es,t,3.1)
            if r.get("internal_field_mvm") is None and flux_now is not None:
                # Historical estimate uses the same declared material scenario. A 7-day spin-up
                # is computed only from measurements available at or before target t.
                hobs=[(tt,v) for tt,v in es if t-timedelta(days=7) <= tt <= t]
                pseudo=[{"time":iso_z(t),"lead_hours":0,"electron_flux_gt2mev":flux_now}]
                field=material_internal_field_forecast(hobs,pseudo,t)[0]
                fluence,coverage=exact_fluence(hobs,t-timedelta(hours=24),t)
                r.update(internal_field_mvm=field.get("internal_field_mvm"),electron_flux_gt2mev=flux_now,
                    electron_fluence_24h=None if fluence is None else round(fluence,3),
                    fluence_coverage=round(coverage,3),internal_model=MATERIAL_SCENARIO["name"])
            saved[key]=r
        t += timedelta(hours=3)
    result={"updated_at":iso_z(now), "retention_days":OBS_KEEP_DAYS,
        "data_kind":"estimate", "verification_eligible":False,
        "note":"Observation-driven estimates, excluded from truth scoring and training. Missing drivers remain gaps.",
        "electron_fetch_status":electron_fetch_status,
        "driver_counts":{"kp":len(hk),"bz":len(hb),"wind":len(hw),"electron":len(es)},
        "estimated":[saved[k] for k in sorted(saved)]}
    atomic_json(path,result)
    print("Charging estimate history:",len(saved),"driver counts:",result["driver_counts"])
    return result


def main() -> None:
    now = utcnow()
    obs = observed_rows(now)
    persist_observed_rows(obs, now)

    e_hist = refresh_electron_histories(now)
    surface_plasma = refresh_surface_plasma_history(now)
    electron_model = build_electron_model(e_hist["research"], now)

    model = fit_coefficients(obs, now)
    estimated = build_estimated_history(model, now, surface_plasma["records"])
    fc = forecast_rows(model, now, e_hist["research"], electron_model, surface_plasma["records"])
    archive = update_archive(load_archive(), fc, electron_model["model_id"] + "|" + model["model_id"], now)

    verification = build_verification(archive, obs, electron_model["model_id"] + "|" + model["model_id"], now)
    electron_verification = build_electron_verification(archive, e_hist["research"], now)
    verification["electron"] = electron_verification
    snapshots = past_snapshots(archive, now, electron_model["model_id"] + "|" + model["model_id"])

    latest = {
        "updated_at": iso_z(now),
        "model": f"{electron_model['model_id']} + {model['model_id']}",
        "generation": {
            "family": "rb-charge",
            "current_model": f"{electron_model['model_id']} + {model['model_id']}",
        },
        "electron_model": electron_model,
        "reference_spacecraft": {
            "scope": "generic GEO reference spacecraft / dielectric material scenario",
            "surface_unit": "kV",
            "internal_unit": "MV/m",
            "warning": "Surface/internal values are scenario estimates unless independent charging telemetry exists. Electron flux is separately verified against measured GOES >2 MeV data.",
            "material_scenario": MATERIAL_SCENARIO,
            "surface_scenario": SURFACE_SCENARIO,
            "differential_surface_scenario": DIFFERENTIAL_SURFACE_SCENARIO,
        },
        "coefficients": model,
        "forecast": fc,
        "electron_observed_30d": [
            {"time": iso_z(t), "electron_flux_gt2mev": v}
            for t, v in e_hist["ui"]
        ],
        "estimated_history": estimated["estimated"],
        "history_status": {
            "driver_counts": estimated["driver_counts"],
            "estimate_count": len(estimated["estimated"]),
            "truth_count": len(obs),
            "electron_ui_count": len(e_hist["ui"]),
            "electron_research_count": len(e_hist["research"]),
            "electron_fetch_status": e_hist["fetch_status"],
            "surface_plasma_count": len(surface_plasma["records"]),
            "surface_electron_fetch_status": surface_plasma["electron_fetch_status"],
            "surface_ion_fetch_status": surface_plasma["ion_fetch_status"],
        },
        "observed": [{k: v for k, v in r.items() if k != "_t"} for r in obs[-1000:]],
        "verification": verification,
        "past_forecast_snapshots": snapshots,
        "methodology": {
            "electron": "Lagged direct log10 flux forecast using electron persistence + exponentially weighted solar-wind/geomagnetic history; optional AI corrects only time-ordered OOF residuals and is shrinkage weighted. No future observed driver is used.",
            "surface": "Reference current-balance solution Ji + Jph + Jse + Jbs - Je - Jleak = 0. GOES low-energy differential particle flux (<=30 keV) is used when available; future low-energy flux amplitudes use bounded persistence+storm scaling. If MPS-LO data are unavailable, a clearly labelled reference-plasma proxy is used. Independent spacecraft-potential observations remain required for validation.",
            "internal": "Reference dielectric state equation epsilon*dE/dt=J_eff-sigma*E with declared material/scenario parameters. J_eff is a transport-response scenario driven by the forecast electron environment, not a direct conversion of integral flux to hardware current.",
            "fluence": "Exact 24 h trapezoidal integration across 8 three-hour intervals / available measured+forecast trajectory; no 9-point x 3h overcount.",
            "forecast_step_hours": STEP_HOURS,
            "forecast_horizon_hours": FORECAST_HOURS,
            "charging_observed_retention_days": OBS_KEEP_DAYS,
            "electron_ui_retention_days": ELECTRON_UI_KEEP_DAYS,
            "electron_training_retention_days": ELECTRON_RESEARCH_KEEP_DAYS,
            "display_window_hours": {"past": 72, "future": 72},
        },
    }

    atomic_json(OUT / "latest.json", latest)
    atomic_json(OUT / "model.json", model)
    atomic_json(OUT / "forecast-archive.json", {"updated_at": iso_z(now), "items": archive})
    atomic_json(OUT / "verification.json", verification)
    print(f"SWIFT-RB model: {electron_model['model_id']}")
    print(f"SWIFT-CHARGE model: {model['model_id']}")
    print(f"forecast rows: {len(fc)} / independent charging targets: {len(obs)}")
    print("electron skill:", electron_verification["nominal_lead_skill"])


if __name__ == "__main__":
    main()
