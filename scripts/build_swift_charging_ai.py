#!/usr/bin/env python3
"""SWIFT-CHARGE: 72 h GEO charging forecast + verification + coefficient learning.

Inputs (all optional/tolerant):
  docs/data/swift-kp/latest.json
  docs/data/swift-bz/latest.json
  docs/data/swift-wind/latest.json
  docs/data/noaa/{kp,mag,wind}_history.json
  docs/data/satellite-risk/latest.json
  docs/data/swift-charging/observed.json

Outputs:
  docs/data/swift-charging/latest.json
  docs/data/swift-charging/model.json
  docs/data/swift-charging/forecast-archive.json
  docs/data/swift-charging/verification.json
  docs/data/swift-charging/observed.json  (pruned to latest 30 days)

Important scientific boundary:
- The forecast values are reference-spacecraft estimates, not direct GOES bus-potential telemetry.
- Surface charging is an engineering proxy until low-energy plasma / validated potential observations are supplied.
- Internal charging is an equivalent reference-dielectric field, not a measured internal field in GOES hardware.
- Verification is counted only against rows in observed.json. Model-generated values never score themselves.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
DATA = ROOT / "docs" / "data"
OUT = DATA / "swift-charging"
OUT.mkdir(parents=True, exist_ok=True)

FORECAST_HOURS = 72
STEP_HOURS = 3
ARCHIVE_DAYS = 45
OBS_KEEP_DAYS = 30
TRAIN_DAYS = 30
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
    except Exception:
        pass
    return default


def rows(obj: Any, keys: Iterable[str] = ("forecast", "records", "items", "history", "data")) -> list[dict[str, Any]]:
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
        t = parse_time(r.get("time") or r.get("target_time") or r.get("time_tag") or r.get("timestamp"))
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


def current_e2_flux() -> float:
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
    return 100.0  # neutral placeholder, explicitly marked below


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
        out.append({
            "time": iso_z(t), "_t": t,
            "surface_kv": num(r.get("surface_kv")),
            "differential_kv": num(r.get("differential_kv")),
            "internal_field_mvm": num(r.get("internal_field_mvm")),
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

    if len(sx) >= MIN_TRAIN:
        beta = ridge_fit(sx, sy)
        if beta:
            surface = {
                "intercept": clamp(beta[0], 0.0, 5.0),
                "kp": clamp(beta[1], 0.0, 2.0),
                "south_bz_nt": clamp(beta[2], 0.0, 0.8),
                "wind_excess_100kms": clamp(beta[3], 0.0, 2.0),
            }
            surface_status = f"ridge_learned_n={len(sx)}"

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
        "differential_ratio": DEFAULT_DIFFERENTIAL_RATIO,
        "training": {
            "window_days": TRAIN_DAYS,
            "surface_samples": len(sx),
            "internal_samples": len(ix),
            "surface_status": surface_status,
            "internal_status": internal_status,
        },
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    payload["model_id"] = f"SWIFT-CHARGE-v0.1-{digest}"
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


def forecast_rows(model: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    kp_s, bz_s, wind_s, raw = load_drivers()
    f0 = current_e2_flux()
    e_source = "GOES >2 MeV current/persistence proxy" if f0 != 100.0 else "neutral >2 MeV placeholder; satellite-risk flux unavailable"
    s_coef = model["surface"]
    i_coef = model["internal"]
    out = []
    # rolling 24h fluence proxy from forecast flux points; units are intentionally explicit proxy scaling.
    flux_history: list[tuple[datetime, float]] = []
    for lead in range(0, FORECAST_HOURS + 1, STEP_HOURS):
        t = now + timedelta(hours=lead)
        kp = clamp(driver_at(kp_s, t, 2.0), 0.0, 9.0)
        bz = clamp(driver_at(bz_s, t, -1.0), -80.0, 80.0)
        wind = clamp(driver_at(wind_s, t, 425.0), 200.0, 1800.0)
        geomag = 0.40 * max(0.0, kp - 3.0) + 0.05 * max(0.0, -bz - 2.0) + 0.30 * max(0.0, (wind - 450.0) / 150.0)
        # Persistence relaxes toward a driver-dependent log flux level.
        decay = math.exp(-lead / 42.0)
        target_log = clamp(2.0 + 0.42 * geomag, 1.7, 5.5)
        logf = decay * math.log10(max(1.0, f0)) + (1.0 - decay) * target_log
        flux = 10 ** clamp(logf, 0.0, 6.5)
        flux_history.append((t, flux))
        cutoff = t - timedelta(hours=24)
        pts = [(tt, ff) for tt, ff in flux_history if tt >= cutoff]
        # Approximate integral with STEP_HOURS sampling, reported as a scaled engineering proxy.
        fluence = sum(ff * STEP_HOURS * 3600.0 for _, ff in pts)
        fluence_1e8 = fluence / 1e8

        surface_abs = (
            s_coef["intercept"]
            + s_coef["kp"] * kp
            + s_coef["south_bz_nt"] * max(0.0, -bz)
            + s_coef["wind_excess_100kms"] * max(0.0, (wind - 400.0) / 100.0)
        )
        surface_kv = -clamp(surface_abs, 0.0, 30.0)
        differential_kv = clamp(abs(surface_kv) * float(model.get("differential_ratio", DEFAULT_DIFFERENTIAL_RATIO)), 0.0, 20.0)
        internal = (
            i_coef["intercept"]
            + i_coef["log10_e2mev_above_100"] * max(0.0, math.log10(max(1.0, flux)) - 2.0)
            + i_coef["kp_excess_3"] * max(0.0, kp - 3.0)
            + i_coef["wind_excess_100kms"] * max(0.0, (wind - 400.0) / 100.0)
            + i_coef["fluence_24h_1e8"] * fluence_1e8
        )
        internal = clamp(internal, 0.0, 5.0)

        out.append({
            "time": iso_z(t),
            "lead_hours": lead,
            "surface_kv": round(surface_kv, 3),
            "differential_kv": round(differential_kv, 3),
            "internal_field_mvm": round(internal, 4),
            "electron_flux_gt2mev": round(flux, 3),
            "electron_fluence_24h_proxy": round(fluence, 3),
            "kp": round(kp, 3),
            "bz_min_nt": round(bz, 3),
            "wind_kms": round(wind, 2),
            "source": "SWIFT-CHARGE reference GEO model + learned residual coefficients",
            "electron_source": e_source,
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
            "internal_field_mvm": r["internal_field_mvm"],
            "electron_flux_gt2mev": r["electron_flux_gt2mev"],
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


def past_snapshots(archive: list[dict[str, Any]], now: datetime, model_id: str) -> dict[str, list[dict[str, Any]]]:
    issues = sorted({r.get("issued_at") for r in archive if r.get("model") == model_id and r.get("issued_at")})
    out = {"surface": [], "internal": []}
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
    return out


def main() -> None:
    now = utcnow()
    obs = observed_rows(now)
    persist_observed_rows(obs, now)
    model = fit_coefficients(obs, now)
    fc = forecast_rows(model, now)
    archive = update_archive(load_archive(), fc, model["model_id"], now)
    verification = build_verification(archive, obs, model["model_id"], now)
    snapshots = past_snapshots(archive, now, model["model_id"])

    latest = {
        "updated_at": iso_z(now),
        "model": model["model_id"],
        "generation": {"family": "charging", "current_model": model["model_id"]},
        "reference_spacecraft": {
            "scope": "generic GEO reference spacecraft / dielectric model",
            "surface_unit": "kV",
            "internal_unit": "MV/m",
            "warning": "Values are estimates unless observed.json provides validated charging targets. Do not interpret as direct GOES hardware telemetry.",
        },
        "coefficients": model,
        "forecast": fc,
        "observed": [{k: v for k, v in r.items() if k != "_t"} for r in obs[-1000:]],
        "verification": verification,
        "past_forecast_snapshots": snapshots,
        "methodology": {
            "surface": "Reference potential magnitude from Kp + southward Bz + solar-wind enhancement, with ridge-updated coefficients when validated target data exist.",
            "internal": "Reference dielectric field from >2 MeV electron flux persistence/driver proxy + 24 h fluence + geomagnetic drivers; coefficients update only when validated target data exist.",
            "forecast_step_hours": STEP_HOURS,
            "forecast_horizon_hours": FORECAST_HOURS,
            "observed_retention_days": OBS_KEEP_DAYS,
            "display_window_hours": {"past": 72, "future": 72},
        },
    }

    (OUT / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "forecast-archive.json").write_text(json.dumps({"updated_at": iso_z(now), "items": archive}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SWIFT-CHARGE model: {model['model_id']}")
    print(f"forecast rows: {len(fc)} / observed targets: {len(obs)}")
    print("surface skill:", verification["nominal_lead_skill"]["surface"])
    print("internal skill:", verification["nominal_lead_skill"]["internal"])


if __name__ == "__main__":
    main()
