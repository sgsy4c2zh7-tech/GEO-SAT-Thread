#!/usr/bin/env python3
"""SWIFT Bz probabilistic forecast v2 with real forecast-archive verification.

This is a calibrated probabilistic model, not a deterministic prediction of ICME Bz.
It learns two online calibration terms from scored past forecasts:
- southward-probability bias
- Bz-min bias

Outputs:
- docs/data/swift-bz/latest.json
- docs/data/swift-bz/forecast.json
- docs/data/swift-bz/verification.json
- docs/data/swift-bz/forecast-archive.json
- docs/data/swift-bz/coefficients.json
"""
from __future__ import annotations

import json, math, statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
OUT = DATA / "swift-bz"
OUT.mkdir(parents=True, exist_ok=True)

LATEST = OUT / "latest.json"
FORECAST = OUT / "forecast.json"
VERIF = OUT / "verification.json"
ARCHIVE = OUT / "forecast-archive.json"
COEF = OUT / "coefficients.json"
INDEX = OUT / "index.json"

BIN_H = 3
FORECAST_DAYS = 5
RECURRENCE_DAYS = 27.27
MODEL = "SWIFT-Bz-v2.1-lead-calibration"
NOMINAL_LEADS = (24, 48, 72)
NOMINAL_TOLERANCE_H = 4.5
LEAD_CAL_WINDOW_DAYS = 30
CALIBRATION_VERSION = "bz-nominal-lead-bias-v1"


def utcnow():
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def iso_z(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(v):
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


def num(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def load(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def rows(obj):
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ("records", "items", "data", "history", "forecast", "arrivals"):
            if isinstance(obj.get(k), list):
                return [x for x in obj[k] if isinstance(x, dict)]
    return []


def load_mag():
    paths = [
        DATA / "noaa" / "mag_history.json",
        DATA / "noaa-imf" / "history.json",
        DATA / "noaa_imf_history.json",
    ]
    by = {}
    for p in paths:
        for r in rows(load(p, {})):
            t = parse_time(r.get("time") or r.get("time_tag"))
            bz = num(r.get("bz") if r.get("bz") is not None else r.get("bz_gsm"))
            bt = num(r.get("bt"))
            if t and bz is not None:
                by[iso_z(t)] = {"time": iso_z(t), "_dt": t, "bz": bz, "bt": bt}
    return sorted(by.values(), key=lambda x: x["_dt"])


def load_wind_forecast():
    obj = load(DATA / "swift-wind" / "latest.json", {}) or {}
    out = []
    for r in obj.get("forecast") or obj.get("records") or []:
        t = parse_time(r.get("time") or r.get("target_time") or r.get("start_time"))
        sp = num(r.get("speed") or r.get("swift_cme_enhanced_speed") or r.get("predicted_speed"))
        if t and sp is not None:
            out.append({"time": t, "speed": sp})
    return sorted(out, key=lambda x: x["time"])


def load_cmes():
    obj = load(DATA / "cme-arrivals" / "latest.json", {}) or {}
    out = []
    for r in rows(obj):
        t = parse_time(r.get("arrival_time") or r.get("arrival") or r.get("estimatedShockArrivalTime"))
        if t:
            q = dict(r); q["_arrival"] = t; out.append(q)
    return out


def nearest(series, t, hours=4):
    if not series:
        return None
    r = min(series, key=lambda x: abs((x["time"] - t).total_seconds()))
    return r if abs((r["time"] - t).total_seconds()) <= hours * 3600 else None


def window(hist, center, hours):
    lo, hi = center - timedelta(hours=hours), center + timedelta(hours=hours)
    return [r for r in hist if lo <= r["_dt"] <= hi]


def med(vals, default=0.0):
    vals = [x for x in vals if x is not None and math.isfinite(x)]
    return statistics.median(vals) if vals else default


def quantile(vals, q, default=0.0):
    vals = sorted(x for x in vals if x is not None and math.isfinite(x))
    if not vals:
        return default
    i = int(round((len(vals)-1)*q))
    return vals[max(0, min(len(vals)-1, i))]


def cme_score(cmes, t):
    best_score, best = 0.0, None
    for c in cmes:
        dh = abs((t - c["_arrival"]).total_seconds()) / 3600
        if dh > 24:
            continue
        cls = str(c.get("impact_class") or "").upper()
        base = 1.0 if ("CORE" in cls or "ENLIL" in cls) else 0.75 if "BODY" in cls else 0.45 if "FLANK" in cls else 0.25
        conf = num(c.get("confidence"), 0.5)
        geom = num(c.get("geometry_multiplier"), 0.5)
        shape = math.exp(-(dh/10.0)**2)
        score = base * conf * max(0.1, geom) * shape
        if score > best_score:
            best_score, best = score, c
    return clamp(best_score, 0, 1), best


def score_archive(hist, archive_items, now):
    out = []
    for a in archive_items:
        if a.get("scored"):
            out.append(a); continue
        tt = parse_time(a.get("target_time"))
        if not tt or tt > now - timedelta(hours=2):
            out.append(a); continue
        obs = window(hist, tt, 1.5)
        bz = [r["bz"] for r in obs if r.get("bz") is not None]
        if not bz:
            out.append(a); continue
        obs_min = min(bz)
        actual_south = any(x < 0 for x in bz)
        p = clamp(num(a.get("southward_bz_probability"), 50)/100, 0, 1)
        pred_min = num(a.get("bz_min_forecast"), 0)
        q = dict(a)
        q.update({
            "scored": True,
            "observed_bz_min": round(obs_min, 3),
            "observed_southward": actual_south,
            "brier": round((p - (1 if actual_south else 0))**2, 5),
            "error_bz_min": round(pred_min - obs_min, 3),
        })
        out.append(q)
    return out


def metrics(scored, now, days):
    r = [x for x in scored if x.get("scored") and parse_time(x.get("target_time")) and parse_time(x["target_time"]) >= now-timedelta(days=days)]
    if not r:
        return {"count": 0, "status": "learning"}
    brier = sum(num(x.get("brier"), 0) for x in r) / len(r)
    mae = sum(abs(num(x.get("error_bz_min"), 0)) for x in r) / len(r)
    pred_rate = sum(num(x.get("southward_bz_probability"), 50)/100 for x in r) / len(r)
    obs_rate = sum(1 if x.get("observed_southward") else 0 for x in r) / len(r)
    hit60 = sum(((num(x.get("southward_bz_probability"), 50) >= 60) == bool(x.get("observed_southward"))) for x in r)/len(r)*100
    return {
        "count": len(r),
        "brier_score": round(brier, 4),
        "direction_hit_rate_threshold60": round(hit60, 1),
        "bz_min_mae": round(mae, 2),
        "predicted_southward_rate": round(pred_rate*100, 1),
        "observed_southward_rate": round(obs_rate*100, 1),
    }


def readiness(ver30):
    n = int(ver30.get("count", 0) or 0)
    if n < 32:
        state = "learning"
    elif n < 100:
        state = "provisional"
    else:
        state = "calibrated"
    brier = num(ver30.get("brier_score"), 0.30)
    sample_factor = clamp(n/150, 0, 1)
    skill_factor = clamp((0.36-brier)/0.20, 0.2, 1.0)
    conf = 0.20 + 0.70*sample_factor*skill_factor
    if n < 32:
        conf = min(conf, 0.35)
    return {"state": state, "scored_forecasts": n, "kp_input_confidence": round(clamp(conf, 0.2, 0.9), 3)}


def current_generation_scored(scored, now):
    return [x for x in scored if x.get("scored") and x.get("model") == MODEL and parse_time(x.get("target_time")) and parse_time(x["target_time"]) >= now-timedelta(days=LEAD_CAL_WINDOW_DAYS)]


def robust_bias(vals):
    vals=[num(x) for x in vals]; vals=[x for x in vals if x is not None]
    if not vals:return 0.0
    med=statistics.median(vals); clipped=[clamp(x,med-3.0,med+3.0) for x in vals]
    return 0.65*med+0.35*(sum(clipped)/len(clipped))


def _adaptive_alpha(n):
    if n < 12:return 0.0
    if n < 30:return 0.15
    if n < 80:return 0.30
    return 0.45


def nominal_lead_scored(scored, now, nominal_h):
    best={}
    for x in current_generation_scored(scored, now):
        tt=parse_time(x.get("target_time")); it=parse_time(x.get("issued_at")); lh=num(x.get("lead_hours"))
        if not tt or not it or lh is None:continue
        delta=abs(lh-nominal_h)
        if delta>NOMINAL_TOLERANCE_H:continue
        key=iso_z(tt); score=(delta,-it.timestamp())
        if key not in best or score<best[key][0]:best[key]=(score,x)
    return [best[k][1] for k in sorted(best)]


def build_nominal_lead_calibration(scored, now):
    out={"version":CALIBRATION_VERSION,"model":MODEL,"window_days":LEAD_CAL_WINDOW_DAYS,"tolerance_hours":NOMINAL_TOLERANCE_H,"leads":{}}
    for h in NOMINAL_LEADS:
        rr=nominal_lead_scored(scored,now,h); n=len(rr); alpha=_adaptive_alpha(n)
        err=[num(x.get("error_bz_min"),0.0) for x in rr]
        bias=robust_bias(err)
        pred_rate=sum(num(x.get("southward_bz_probability"),50)/100 for x in rr)/n if n else 0.0
        obs_rate=sum(1 if x.get("observed_southward") else 0 for x in rr)/n if n else 0.0
        add_nt=clamp(-alpha*bias,-5.0,5.0)
        add_pp=clamp(alpha*(obs_rate-pred_rate)*100,-12.0,12.0)
        out["leads"][str(h)]={"nominal_lead_hours":h,"count":n,"bz_min_bias":round(bias,3),"alpha":round(alpha,3),"add_to_bz_min_nt":round(add_nt,3),"add_to_probability_pp":round(add_pp,3),"status":"active" if alpha>0 else "learning"}
    return out


def _interp_lead_cal(cal, lead_h, key):
    pts=[(0.0,0.0)]+[(float(h),num(cal["leads"][str(h)].get(key),0.0)) for h in NOMINAL_LEADS]
    x=clamp(float(lead_h),0.0,72.0)
    for (x0,y0),(x1,y1) in zip(pts,pts[1:]):
        if x<=x1:
            f=0.0 if x1==x0 else (x-x0)/(x1-x0); return y0+f*(y1-y0)
    return pts[-1][1]


def update_calibration(coef, scored, now):
    recent = current_generation_scored(scored, now)
    if len(recent) < 16:
        coef["calibration_generation"] = MODEL
        coef["calibration_count"] = len(recent)
        return coef
    pred_rate = sum(num(x.get("southward_bz_probability"), 50)/100 for x in recent)/len(recent)
    obs_rate = sum(1 if x.get("observed_southward") else 0 for x in recent)/len(recent)
    probability_bias_pp = clamp((obs_rate-pred_rate)*100, -20, 20)
    bz_bias = clamp(sum(-num(x.get("error_bz_min"), 0) for x in recent)/len(recent), -8, 8)
    old_p = num(coef.get("probability_bias_pp"), 0)
    old_b = num(coef.get("bz_min_bias_nt"), 0)
    coef["probability_bias_pp"] = round(0.8*old_p + 0.2*probability_bias_pp, 3)
    coef["bz_min_bias_nt"] = round(0.8*old_b + 0.2*bz_bias, 3)
    coef["calibration_count"] = len(recent)
    coef["calibration_generation"] = MODEL
    coef["updated_at"] = iso_z(now)
    return coef


def build(hist, wind_fc, cmes, coef, now, lead_cal):
    recent = [r for r in hist if r["_dt"] >= now-timedelta(hours=6)]
    recent_bz = med([r["bz"] for r in recent], 0.0)
    recent_bt = med([r.get("bt") for r in recent], 6.0)
    sigma = max(1.5, statistics.pstdev([r["bz"] for r in recent]) if len(recent) >= 3 else 2.5)
    clim_q10 = quantile([r["bz"] for r in hist[-6000:]], 0.10, -5.0)

    fc = []
    for i in range(FORECAST_DAYS*8):
        t = now + timedelta(hours=i*BIN_H)
        lead = i*BIN_H
        rec = window(hist, t-timedelta(days=RECURRENCE_DAYS), 2.0)
        rec_bz = med([r["bz"] for r in rec], None) if rec else None
        rec_bt = med([r.get("bt") for r in rec], None) if rec else None
        w_recent = max(0.15, 0.75*math.exp(-lead/36))
        w_rec = 0.35 if rec_bz is not None else 0.0
        w_clim = max(0.0, 1-w_recent-w_rec)
        base_bz = w_recent*recent_bz + (w_rec*rec_bz if rec_bz is not None else 0)
        base_bt = w_recent*recent_bt + w_clim*5.0 + (w_rec*(rec_bt or 5) if rec_bt is not None else 0)

        wr = nearest(wind_fc, t)
        vsw = wr["speed"] if wr else 400
        cscore, cme = cme_score(cmes, t)

        p = 0.32
        if rec:
            south = sum(1 for r in rec if r["bz"] < 0)/len(rec)
            strong = sum(1 for r in rec if r["bz"] <= -5)/len(rec)
            p += 0.32*(south-0.45) + 0.70*strong
        p += clamp((vsw-450)/700, 0, 0.16)
        if recent:
            recent_south = sum(1 for r in recent if r["bz"] < 0)/len(recent)
            p += w_recent*0.22*(recent_south-0.45)
        p += 0.35*cscore
        p += num(coef.get("probability_bias_pp"), 0)/100
        lead_prob_add_pp = _interp_lead_cal(lead_cal, lead, "add_to_probability_pp")
        p += lead_prob_add_pp/100
        p = clamp(p, 0.05, 0.92)

        bz_min = min(base_bz - 1.65*sigma - 8*cscore, clim_q10 - 4*cscore)
        bz_min += num(coef.get("bz_min_bias_nt"), 0)
        lead_bz_add_nt = _interp_lead_cal(lead_cal, lead, "add_to_bz_min_nt")
        bz_min += lead_bz_add_nt
        bt = clamp(base_bt + clamp((vsw-450)/350, 0, 4) + 7*cscore, 2, 40)

        conf = 0.28 + (0.20 if rec else 0) + 0.12 + (0.08 if cme else 0)
        conf -= min(0.18, lead/600)
        fc.append({
            "time": iso_z(t), "end_time": iso_z(t+timedelta(hours=BIN_H)), "lead_hours": lead,
            "bz_forecast": round(base_bz, 2),
            "bz_min_forecast": round(clamp(bz_min, -30, 5), 2),
            "bt_forecast": round(bt, 2),
            "southward_bz_probability": round(p*100, 1),
            "bz_risk": "HIGH" if p >= .75 else "MODERATE" if p >= .55 else "LOW-MODERATE" if p >= .35 else "LOW",
            "confidence": round(clamp(conf, .20, .82), 2),
            "cme_risk_score": round(cscore, 3),
            "cme_id": cme.get("id") if cme else None,
            "model_components": {
                "recent_bz": round(recent_bz, 2),
                "rotation27_bz": round(rec_bz, 2) if rec_bz is not None else None,
                "solar_wind_speed": round(vsw, 1),
                "probability_bias_pp": num(coef.get("probability_bias_pp"), 0),
                "bz_min_bias_nt": num(coef.get("bz_min_bias_nt"), 0),
                "lead_bias_add_nt": round(lead_bz_add_nt, 3),
                "lead_probability_add_pp": round(lead_prob_add_pp, 3),
                "lead_calibration_version": CALIBRATION_VERSION,
            },
        })
    return fc


def main():
    now = utcnow()
    hist = load_mag()
    wind_fc = load_wind_forecast()
    cmes = load_cmes()
    coef = load(COEF, {}) or {}
    coef.setdefault("version", MODEL)
    coef.setdefault("probability_bias_pp", 0.0)
    coef.setdefault("bz_min_bias_nt", 0.0)

    old_archive = (load(ARCHIVE, {}) or {}).get("items", [])
    scored = score_archive(hist, old_archive[-12000:], now)
    coef = update_calibration(coef, scored, now)
    lead_cal = build_nominal_lead_calibration(scored, now)
    coef["nominal_lead_bias_calibration"] = lead_cal

    forecast = build(hist, wind_fc, cmes, coef, now, lead_cal) if hist else []
    issued = iso_z(now)
    for r in forecast:
        scored.append({
            "issued_at": issued,
            "target_time": r["time"],
            "lead_hours": r["lead_hours"],
            "southward_bz_probability": r["southward_bz_probability"],
            "bz_min_forecast": r["bz_min_forecast"],
            "bt_forecast": r["bt_forecast"],
            "model": MODEL,
            "scored": False,
        })
    scored = scored[-12000:]

    current_scored = [x for x in scored if x.get("model") == MODEL]
    v7 = metrics(current_scored, now, 7)
    v30 = metrics(current_scored, now, 30)
    all30 = metrics(scored, now, 30)
    ready = readiness(v30)
    ver = {"updated_at": iso_z(now), "model": MODEL, "last_7d": v7, "last_30d": v30, "all_generations_30d": all30, "readiness": ready, "nominal_lead_bias_calibration": lead_cal}
    latest = {
        "updated_at": iso_z(now),
        "model": MODEL,
        "description": "Probabilistic southward-Bz model with current-generation archive scoring plus 24/48/72h residual bias calibration.",
        "forecast_days": FORECAST_DAYS,
        "cadence_hours": BIN_H,
        "inputs": {"imf_history_records": len(hist), "cme_arrivals": len(cmes), "uses_rotation27_days": RECURRENCE_DAYS},
        "readiness": ready,
        "current": forecast[0] if forecast else None,
        "max_risk": max(forecast, key=lambda r:r["southward_bz_probability"], default=None),
        "forecast": forecast,
        "verification": ver,
        "nominal_lead_bias_calibration": lead_cal,
        "coefficients": coef,
    }
    save(FORECAST, {"updated_at": iso_z(now), "items": forecast})
    save(LATEST, latest)
    save(VERIF, ver)
    save(ARCHIVE, {"updated_at": iso_z(now), "items": scored})
    save(COEF, coef)
    save(INDEX, {
        "updated_at": iso_z(now), "latest": "latest.json", "forecast": "forecast.json",
        "verification": "verification.json", "archive": "forecast-archive.json",
        "coefficients": "coefficients.json",
    })
    print(json.dumps({"readiness": ready, "verification_7d": v7, "forecast_count": len(forecast)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
