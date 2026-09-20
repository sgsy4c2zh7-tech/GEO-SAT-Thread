#!/usr/bin/env python3
"""Build one SWIFT Excel workbook per UTC month for GitHub Pages.

Output example:
- docs/reports/swift_space_weather_2026-09.xlsx
- docs/reports/index.json

The same YYYY-MM workbook is overwritten on each run during the month.
Past monthly workbooks are retained for 24 months, so there is exactly one Excel file per UTC month and two years remain available for research.

Workbook contents:
- Monthly_Summary: month KPI summary and daily summary table
- SolarWind_Month: hourly NOAA solar wind/IMF for the month
- Wind_Forecast_3d: next 3-day SWIFT Wind forecast
- Wind_Fcst_History: archived historical forecasts for forecast-vs-observation review
- Kp_Forecast, Accuracy, Bz_AI, CME_Arrivals, CME_Wind_Boost, Sources
"""
from __future__ import annotations

import json
import math
import shutil
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
OUT_DOCS = ROOT / "docs" / "reports"
OUT_DOCS.mkdir(parents=True, exist_ok=True)

MONTHLY_NAME = "swift_space_weather_{month}.xlsx"
BIN_HOURS = 1
FORECAST_DAYS = 3
HISTORY_DAYS = 40  # enough to cover a complete current UTC month
REPORT_RETENTION_MONTHS = 24


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


def records(obj: Any, keys: Iterable[str] = ("records", "items", "history", "data", "forecast", "arrivals")) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for key in keys:
            if isinstance(obj.get(key), list):
                return [r for r in obj[key] if isinstance(r, dict)]
    return []


def load_first(paths: Iterable[Path]) -> Any:
    for p in paths:
        obj = load_json(p)
        if obj is not None:
            return obj
    return None


def pick(r: dict[str, Any], names: Iterable[str]) -> Any:
    for n in names:
        if n in r and r[n] not in (None, ""):
            return r[n]
    return None


def load_wind_history() -> list[dict[str, Any]]:
    obj = load_first([
        DATA / "noaa" / "wind_history.json",
        DATA / "noaa_wind_history.json",
        DATA / "swift-wind" / "history.json",
    ])
    out = []
    for r in records(obj):
        t = parse_time(pick(r, ["time", "time_tag", "timestamp", "datetime", "date"]))
        speed = num(pick(r, ["speed", "observed_speed", "solar_wind_speed", "v", "Vsw"]))
        density = num(pick(r, ["density", "proton_density", "n"]))
        temperature = num(pick(r, ["temperature", "temp"]))
        if t and speed is not None:
            out.append({"time": iso_z(t), "_t": t, "speed": speed, "density": density, "temperature": temperature})
    return dedupe_time(out)


def load_mag_history() -> list[dict[str, Any]]:
    obj = load_first([
        DATA / "noaa" / "mag_history.json",
        DATA / "noaa_imf_history.json",
    ])
    out = []
    for r in records(obj):
        t = parse_time(pick(r, ["time", "time_tag", "timestamp", "datetime", "date"]))
        bz = num(pick(r, ["bz", "bz_gsm", "imf_bz"]))
        bt = num(pick(r, ["bt", "total_field"]))
        by = num(pick(r, ["by", "by_gsm"]))
        if t and bz is not None:
            out.append({"time": iso_z(t), "_t": t, "bz": bz, "bt": bt, "by": by})
    return dedupe_time(out)


def load_kp_history() -> list[dict[str, Any]]:
    obj = load_first([
        DATA / "noaa" / "kp_history.json",
        DATA / "noaa_kp_history.json",
    ])
    out = []
    for r in records(obj):
        t = parse_time(pick(r, ["time", "time_tag", "timestamp", "datetime", "date"]))
        kp = num(pick(r, ["kp", "kp_index", "estimated_kp", "Kp"]))
        if t and kp is not None:
            out.append({"time": iso_z(t), "_t": t, "kp": kp})
    return dedupe_time(out)


def dedupe_time(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    d = {}
    for r in items:
        if r.get("time"):
            d[r["time"]] = r
    return sorted(d.values(), key=lambda x: x["_t"])


def bin_hourly(items: list[dict[str, Any]], value_keys: list[str], days: int) -> list[dict[str, Any]]:
    now = utcnow()
    cutoff = now - timedelta(days=days)
    buckets: dict[datetime, dict[str, list[float]]] = {}
    for r in items:
        t = r.get("_t")
        if not isinstance(t, datetime) or t < cutoff:
            continue
        b = t.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(b, {k: [] for k in value_keys})
        for k in value_keys:
            v = num(r.get(k))
            if v is not None:
                buckets[b][k].append(v)
    rows = []
    for t in sorted(buckets.keys()):
        row = {"time": iso_z(t), "_t": t}
        for k in value_keys:
            vals = buckets[t][k]
            row[k] = round(statistics.median(vals), 3) if vals else None
        rows.append(row)
    return rows


def load_swift_wind() -> dict[str, Any] | None:
    index = load_json(DATA / "swift-wind" / "index.json")
    paths = []
    if isinstance(index, dict) and index.get("latest"):
        paths.append(DATA / "swift-wind" / str(index["latest"]))
    paths.extend([
        DATA / "swift-wind" / "latest.json",
        DATA / "swift_wind_ai.json",
    ])
    for p in paths:
        obj = load_json(p)
        if isinstance(obj, dict):
            return obj
    return None


def load_swift_kp() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    index = load_json(DATA / "swift-kp" / "index.json")
    latest_paths = []
    hist_paths = []
    if isinstance(index, dict):
        if index.get("latest"):
            latest_paths.append(DATA / "swift-kp" / str(index["latest"]))
        if index.get("history"):
            hist_paths.append(DATA / "swift-kp" / str(index["history"]))
    latest_paths.extend([DATA / "swift-kp" / "latest.json", DATA / "swift_kp_ai.json"])
    hist_paths.extend([DATA / "swift-kp" / "history.json"])
    latest = next((obj for obj in (load_json(p) for p in latest_paths) if isinstance(obj, dict)), None)
    hist = next((obj for obj in (load_json(p) for p in hist_paths) if isinstance(obj, dict)), None)
    return latest, hist


def load_bz_ai() -> dict[str, Any] | None:
    obj = load_json(DATA / "swift-bz" / "latest.json")
    return obj if isinstance(obj, dict) else None


def load_cme_arrivals() -> list[dict[str, Any]]:
    obj = load_json(DATA / "cme-arrivals" / "latest.json")
    out = []
    for r in records(obj):
        at = parse_time(pick(r, ["arrival_time", "arrival", "estimatedShockArrivalTime"]))
        t0 = parse_time(pick(r, ["t0", "activityStartTime", "startTime"]))
        if at:
            x = dict(r)
            x["_arrival"] = at
            x["_t0"] = t0
            out.append(x)
    return sorted(out, key=lambda x: x["_arrival"])


def load_cme_boost_model() -> dict[str, Any] | None:
    obj = load_json(DATA / "swift-wind" / "cme-boost-model.json")
    return obj if isinstance(obj, dict) else None


def g_scale(kp: float | None) -> str:
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


def median(vals: list[float], default: float | None = None) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return default
    return statistics.median(vals)


def nearest_value(rows: list[dict[str, Any]], target: datetime, key: str, max_hours: float = 2.0) -> float | None:
    best = None
    best_dt = float("inf")
    for r in rows:
        t = r.get("_t")
        if not isinstance(t, datetime):
            continue
        dt = abs((t - target).total_seconds()) / 3600.0
        if dt < best_dt:
            best_dt = dt
            best = r
    if best is None or best_dt > max_hours:
        return None
    return num(best.get(key))


def cme_boost_at(cmes: list[dict[str, Any]], t: datetime) -> tuple[float, str | None]:
    best = 0.0
    best_id = None
    for c in cmes:
        arrival = c.get("_arrival")
        if not isinstance(arrival, datetime):
            continue
        dt = abs((t - arrival).total_seconds()) / 3600.0
        if dt > 24:
            continue
        amp = num(c.get("expected_wind_speed_increase"), 0.0) or 0.0
        shape = math.exp(-(dt / 9.0) ** 2)
        v = amp * shape
        if v > best:
            best = v
            best_id = c.get("id")
    return round(best, 2), best_id


def wind_forecast_3d(swift_wind: dict[str, Any] | None, wind_hist: list[dict[str, Any]], cmes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utcnow().replace(minute=0, second=0)
    end = now + timedelta(days=FORECAST_DAYS)
    rows = []
    if swift_wind:
        for r in records(swift_wind):
            t = parse_time(pick(r, ["time", "start_time", "timestamp"]))
            if not t or t < now - timedelta(hours=3) or t > end:
                continue
            pred = num(pick(r, ["swift_cme_enhanced_speed", "predicted_speed", "forecast_speed", "speed"]))
            bg = num(pick(r, ["swift_background_speed", "background_speed", "wsa_speed"]))
            obs = num(pick(r, ["observed_speed", "actual_speed"]))
            cme_eff = num(pick(r, ["cme_effect_weighted", "cme_effect", "cme_boost"]))
            if pred is None:
                continue
            rows.append({"time": iso_z(t), "_t": t, "predicted_speed": pred, "background_speed": bg, "observed_speed": obs, "cme_boost": cme_eff, "source": "SWIFT Wind AI"})
    if rows:
        return sorted(rows, key=lambda x: x["_t"])

    recent = [r for r in wind_hist if r["_t"] >= now - timedelta(hours=12)]
    recent_v = median([num(r.get("speed")) for r in recent], 425.0) or 425.0
    for i in range(int(FORECAST_DAYS * 24 / 3) + 1):
        t = now + timedelta(hours=3 * i)
        rec = nearest_value(wind_hist, t - timedelta(days=27.27), "speed", max_hours=3.0)
        w_recent = max(0.15, 0.8 * math.exp(-i / 14.0))
        bg = w_recent * recent_v + (1 - w_recent) * (rec if rec is not None else 425.0)
        boost, cid = cme_boost_at(cmes, t)
        pred = max(250.0, min(950.0, bg + boost))
        rows.append({
            "time": iso_z(t),
            "_t": t,
            "predicted_speed": round(pred, 2),
            "background_speed": round(bg, 2),
            "observed_speed": None,
            "cme_boost": boost,
            "cme_id": cid,
            "source": "Report fallback: persistence + 27day + CME boost",
        })
    return rows


def kp_forecast_rows(swift_kp: dict[str, Any] | None, wind_fc: list[dict[str, Any]], bz_ai: dict[str, Any] | None) -> list[dict[str, Any]]:
    now = utcnow()
    out = []
    if swift_kp:
        for r in records(swift_kp, keys=("forecast", "records", "items")):
            t = parse_time(pick(r, ["start_time", "time", "timestamp"]))
            if not t or t < now - timedelta(hours=3) or t > now + timedelta(days=FORECAST_DAYS):
                continue
            kp = num(pick(r, ["kp", "predicted_kp", "forecast_kp"]))
            if kp is None:
                continue
            out.append({
                "time": iso_z(t), "_t": t, "kp": kp, "g_scale": r.get("g_scale") or g_scale(kp),
                "confidence": num(r.get("confidence")), "source": "SWIFT Kp AI",
            })
    if out:
        return sorted(out, key=lambda x: x["_t"])

    bz_rows = records(bz_ai, keys=("forecast",)) if bz_ai else []
    for wr in wind_fc:
        t = wr["_t"]
        if t.hour % 3 != 0:
            continue
        v = num(wr.get("predicted_speed"), 425.0) or 425.0
        boost = num(wr.get("cme_boost"), 0.0) or 0.0
        bz_min = None
        for br in bz_rows:
            bt = parse_time(br.get("time"))
            if bt and abs((bt - t).total_seconds()) <= 5400:
                bz_min = num(br.get("bz_min_forecast"))
                break
        south = max(0.0, -(bz_min or 0.0))
        kp = 1.0 + 0.008 * max(0, v - 350) + 0.006 * boost + 0.10 * south
        kp = max(0.0, min(9.0, kp))
        out.append({"time": iso_z(t), "_t": t, "kp": round(kp, 2), "g_scale": g_scale(kp), "confidence": 0.35, "source": "Report fallback"})
    return out


def wind_accuracy(swift_wind: dict[str, Any] | None, wind_hist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return true predicted-vs-observed pairs. Prefer the model's 30-day verification items."""
    pairs: list[dict[str, Any]] = []
    if swift_wind:
        verification = swift_wind.get("verification") if isinstance(swift_wind, dict) else None
        v30 = verification.get("last_30d") if isinstance(verification, dict) else None
        items = v30.get("items") if isinstance(v30, dict) else None
        if isinstance(items, list):
            for r in items:
                if not isinstance(r, dict):
                    continue
                t = parse_time(pick(r, ["time", "start_time", "timestamp"]))
                obs = num(pick(r, ["observed_speed", "observed", "actual_speed"]))
                pred = num(pick(r, ["predicted_speed", "predicted", "swift_cme_enhanced_speed"]))
                bg = num(pick(r, ["background_speed", "background", "swift_background_speed"]))
                if t and obs is not None and pred is not None:
                    pairs.append({"time": iso_z(t), "_t": t, "observed": obs, "predicted": pred, "background": bg, "error": pred - obs})
        if not pairs:
            for r in records(swift_wind):
                t = parse_time(pick(r, ["time", "start_time", "timestamp"]))
                obs = num(pick(r, ["observed_speed", "actual_speed"]))
                pred = num(pick(r, ["swift_cme_enhanced_speed", "predicted_speed", "forecast_speed"]))
                bg = num(pick(r, ["swift_background_speed", "background_speed", "wsa_speed"]))
                if t and obs is not None and pred is not None:
                    pairs.append({"time": iso_z(t), "_t": t, "observed": obs, "predicted": pred, "background": bg, "error": pred - obs})
    return dedupe_time(pairs) if pairs else []


def current_wind_accuracy(swift_wind: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(swift_wind, dict):
        return {"count": 0, "hit_rate_50kms": None, "mae": None, "bias": None, "rmse": None}
    v = swift_wind.get("verification", {}).get("last_24h", {}).get("enhanced", {})
    if not isinstance(v, dict):
        v = {}
    return {
        "count": int(num(v.get("count"), 0) or 0),
        "hit_rate_50kms": num(v.get("hit_rate_50kms")),
        "hit_rate_100kms": num(v.get("hit_rate_100kms")),
        "mae": num(v.get("mae")),
        "bias": num(v.get("bias")),
        "rmse": num(v.get("rmse")),
    }

def kp_accuracy(swift_kp_hist: dict[str, Any] | None) -> list[dict[str, Any]]:
    out = []
    if not swift_kp_hist:
        return out
    for r in records(swift_kp_hist, keys=("items", "history", "records", "data")):
        t = parse_time(pick(r, ["time", "start_time", "timestamp", "updated_at"]))
        pred = num(pick(r, ["predicted_kp", "prediction", "forecast_kp", "kp_predicted"]))
        obs = num(pick(r, ["observed_kp", "actual_kp", "kp_observed", "kp_actual"]))
        if t and pred is not None and obs is not None:
            out.append({
                "time": iso_z(t), "_t": t, "predicted": pred, "observed": obs, "error": pred - obs,
                "predicted_g": g_scale(pred), "observed_g": g_scale(obs),
            })
    return dedupe_time(out)


def metric_summary(pairs: list[dict[str, Any]], days: int, kind: str) -> dict[str, Any]:
    now = utcnow()
    rows = [r for r in pairs if isinstance(r.get("_t"), datetime) and r["_t"] >= now - timedelta(days=days)]
    if not rows:
        return {"kind": kind, "period": f"last_{days}d", "count": 0}
    errs = [num(r.get("error"), 0.0) or 0.0 for r in rows]
    abs_err = [abs(e) for e in errs]
    out = {
        "kind": kind,
        "period": f"last_{days}d",
        "count": len(rows),
        "mae": round(sum(abs_err) / len(abs_err), 3),
        "rmse": round(math.sqrt(sum(e * e for e in errs) / len(errs)), 3),
        "bias": round(sum(errs) / len(errs), 3),
    }
    if kind == "wind":
        out["hit_rate_within_50kms"] = round(100.0 * sum(1 for e in abs_err if e <= 50) / len(rows), 1)
        out["hit_rate_within_100kms"] = round(100.0 * sum(1 for e in abs_err if e <= 100) / len(rows), 1)
    if kind == "kp":
        out["hit_rate_within_067kp"] = round(100.0 * sum(1 for e in abs_err if e <= 0.67) / len(rows), 1)
        out["hit_rate_within_1kp"] = round(100.0 * sum(1 for e in abs_err if e <= 1.0) / len(rows), 1)
        out["g_scale_hit_rate"] = round(100.0 * sum(1 for r in rows if r.get("predicted_g") == r.get("observed_g")) / len(rows), 1)
    return out


def write_table(ws: Any, start_row: int, start_col: int, headers: list[str], data: list[dict[str, Any]], formats: dict[str, Any]) -> int:
    header_fmt = formats["header"]
    body_fmt = formats["body"]
    for c, h in enumerate(headers):
        ws.write(start_row, start_col + c, h, header_fmt)
    for r, row in enumerate(data, start_row + 1):
        for c, h in enumerate(headers):
            value = row.get(h)
            if isinstance(value, datetime):
                ws.write_datetime(r, start_col + c, value, formats["datetime"])
            else:
                ws.write(r, start_col + c, value, body_fmt)
    return start_row + 1 + len(data)


def prepare_rows_for_excel(rows: list[dict[str, Any]], mapping: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        x = {}
        for out_key, in_key in mapping:
            v = r.get(in_key)
            if in_key == "_t" and isinstance(v, datetime):
                x[out_key] = v.replace(tzinfo=None)
            else:
                x[out_key] = v
        out.append(x)
    return out


def add_line_chart(wb: Any, ws: Any, title: str, sheet: str, first_row: int, last_row: int, x_col: int, y_cols: list[tuple[int, str]], insert_cell: str, y_axis: str) -> None:
    if last_row <= first_row:
        return
    chart = wb.add_chart({"type": "line"})
    for col, name in y_cols:
        chart.add_series({
            "name": name,
            "categories": [sheet, first_row + 1, x_col, last_row, x_col],
            "values": [sheet, first_row + 1, col, last_row, col],
        })
    chart.set_title({"name": title})
    chart.set_x_axis({"name": "UTC", "date_axis": True, "num_format": "mm/dd hh:mm"})
    chart.set_y_axis({"name": y_axis})
    chart.set_legend({"position": "bottom"})
    chart.set_size({"width": 900, "height": 320})
    ws.insert_chart(insert_cell, chart)


def percentile(vals: list[float], p: float) -> float | None:
    vals = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    x = (len(vals) - 1) * p
    lo, hi = int(math.floor(x)), int(math.ceil(x))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (x - lo)


def month_bounds(now: datetime) -> tuple[datetime, datetime]:
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def in_range(rows: list[dict[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
    return [r for r in rows if isinstance(r.get("_t"), datetime) and start <= r["_t"] < end]


def daily_summary_rows(
    month_start: datetime,
    now: datetime,
    sw_rows: list[dict[str, Any]],
    kp_rows: list[dict[str, Any]],
    acc_rows: list[dict[str, Any]],
    cmes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    d = month_start
    today_end = min(now + timedelta(seconds=1), month_bounds(now)[1])
    while d < today_end:
        e = min(d + timedelta(days=1), today_end)
        sw = [r for r in sw_rows if d <= r["_t"] < e]
        kp = [r for r in kp_rows if d <= r["_t"] < e]
        acc = [r for r in acc_rows if d <= r["_t"] < e]
        cm = [r for r in cmes if isinstance(r.get("_arrival"), datetime) and d <= r["_arrival"] < e]
        speeds = [float(r["speed"]) for r in sw if num(r.get("speed")) is not None]
        kps = [float(r["kp"]) for r in kp if num(r.get("kp")) is not None]
        errs = [abs(float(r["error"])) for r in acc if num(r.get("error")) is not None]
        out.append({
            "Date": d.replace(tzinfo=None),
            "Wind_Avg_km_s": round(sum(speeds) / len(speeds), 1) if speeds else None,
            "Wind_Min_km_s": round(min(speeds), 1) if speeds else None,
            "Wind_Max_km_s": round(max(speeds), 1) if speeds else None,
            "Wind_P95_km_s": round(percentile(speeds, .95), 1) if speeds else None,
            "Kp_Max": round(max(kps), 2) if kps else None,
            "Accuracy_Count": len(errs),
            "Wind_Hit_50_pct": round(100.0 * sum(1 for x in errs if x <= 50) / len(errs), 1) if errs else None,
            "Wind_MAE_km_s": round(sum(errs) / len(errs), 1) if errs else None,
            "CME_Arrivals": len(cm),
        })
        d += timedelta(days=1)
    return out


def month_metric_rows(
    month_start: datetime,
    month_end: datetime,
    sw_rows: list[dict[str, Any]],
    mag_rows: list[dict[str, Any]],
    kp_rows: list[dict[str, Any]],
    acc_rows: list[dict[str, Any]],
    cmes: list[dict[str, Any]],
    current_acc: dict[str, Any],
) -> list[dict[str, Any]]:
    sw = in_range(sw_rows, month_start, month_end)
    mag = in_range(mag_rows, month_start, month_end)
    kp = in_range(kp_rows, month_start, month_end)
    acc = in_range(acc_rows, month_start, month_end)
    cm = [r for r in cmes if isinstance(r.get("_arrival"), datetime) and month_start <= r["_arrival"] < month_end]
    speeds = [float(r["speed"]) for r in sw if num(r.get("speed")) is not None]
    densities = [float(r["density"]) for r in sw if num(r.get("density")) is not None]
    bzs = [float(r["bz"]) for r in mag if num(r.get("bz")) is not None]
    kps = [float(r["kp"]) for r in kp if num(r.get("kp")) is not None]
    errs = [float(r["error"]) for r in acc if num(r.get("error")) is not None]
    abs_errs = [abs(x) for x in errs]
    return [
        {"Metric": "Month", "Value": month_start.strftime("%Y-%m"), "Unit": "UTC"},
        {"Metric": "Solar wind samples", "Value": len(speeds), "Unit": "records"},
        {"Metric": "Solar wind mean", "Value": round(sum(speeds) / len(speeds), 2) if speeds else None, "Unit": "km/s"},
        {"Metric": "Solar wind median", "Value": round(statistics.median(speeds), 2) if speeds else None, "Unit": "km/s"},
        {"Metric": "Solar wind min", "Value": round(min(speeds), 2) if speeds else None, "Unit": "km/s"},
        {"Metric": "Solar wind max", "Value": round(max(speeds), 2) if speeds else None, "Unit": "km/s"},
        {"Metric": "Solar wind P95", "Value": round(percentile(speeds, .95), 2) if speeds else None, "Unit": "km/s"},
        {"Metric": "Density mean", "Value": round(sum(densities) / len(densities), 2) if densities else None, "Unit": "cm^-3"},
        {"Metric": "Bz minimum", "Value": round(min(bzs), 2) if bzs else None, "Unit": "nT"},
        {"Metric": "Kp maximum", "Value": round(max(kps), 2) if kps else None, "Unit": "Kp"},
        {"Metric": "Kp mean", "Value": round(sum(kps) / len(kps), 2) if kps else None, "Unit": "Kp"},
        {"Metric": "Wind verification pairs", "Value": len(abs_errs), "Unit": "pairs"},
        {"Metric": "Monthly wind ±50 hit rate", "Value": round(100 * sum(1 for x in abs_errs if x <= 50) / len(abs_errs), 1) if abs_errs else None, "Unit": "%"},
        {"Metric": "Monthly wind MAE", "Value": round(sum(abs_errs) / len(abs_errs), 2) if abs_errs else None, "Unit": "km/s"},
        {"Metric": "Monthly wind bias", "Value": round(sum(errs) / len(errs), 2) if errs else None, "Unit": "km/s"},
        {"Metric": "Current ±50 hit rate (last 24h)", "Value": current_acc.get("hit_rate_50kms"), "Unit": "%"},
        {"Metric": "Current verification count (last 24h)", "Value": current_acc.get("count"), "Unit": "pairs"},
        {"Metric": "CME arrivals in month", "Value": len(cm), "Unit": "events"},
    ]



def load_swift_charging() -> dict[str, Any]:
    obj = load_json(DATA / "swift-charging" / "latest.json")
    return obj if isinstance(obj, dict) else {}


def load_charging_verification() -> dict[str, Any]:
    obj = load_json(DATA / "swift-charging" / "verification.json")
    return obj if isinstance(obj, dict) else {}


def load_charging_archive() -> list[dict[str, Any]]:
    obj = load_json(DATA / "swift-charging" / "forecast-archive.json") or {}
    return [r for r in records(obj, keys=("items", "records", "data")) if isinstance(r, dict)]


def _charging_skill_map(v: dict[str, Any], family: str) -> dict[int, dict[str, Any]]:
    rows_ = ((v.get("nominal_lead_skill") or {}).get(family) or []) if isinstance(v, dict) else []
    out = {}
    for x in rows_:
        if not isinstance(x, dict):
            continue
        try:
            out[int(x.get("nominal_lead_hours"))] = x
        except Exception:
            continue
    return out


def _charging_month_pairs(v: dict[str, Any], family: str, month_start: datetime, month_end: datetime) -> list[dict[str, Any]]:
    block = (((v.get("lead_pairs") or {}).get(family)) or {}) if isinstance(v, dict) else {}
    out = []
    for lead_key, items in block.items():
        try:
            nominal = int(str(lead_key).replace("h", ""))
        except Exception:
            nominal = None
        for r in items or []:
            if not isinstance(r, dict):
                continue
            tt = parse_time(r.get("target_time"))
            if tt and month_start <= tt < month_end:
                q = dict(r)
                q["nominal_lead_hours"] = nominal
                out.append(q)
    out.sort(key=lambda r: (parse_time(r.get("target_time")) or datetime.min.replace(tzinfo=timezone.utc), r.get("nominal_lead_hours") or 0))
    return out


def load_research_validation(month: str) -> dict[str, Any]:
    month_path = DATA / "validation" / "monthly" / f"{month}.json"
    obj = load_json(month_path)
    if isinstance(obj, dict):
        return obj
    obj = load_json(DATA / "validation" / "latest.json")
    return obj if isinstance(obj, dict) else {}


def load_enlil_latest() -> dict[str, Any]:
    obj = load_json(DATA / "enlil" / "latest.json")
    return obj if isinstance(obj, dict) else {}


def load_wsa_boundary() -> dict[str, Any]:
    obj = load_json(DATA / "enlil" / "wsa_boundary.json")
    return obj if isinstance(obj, dict) else {}


def metric_rows_from_validation(validation: dict[str, Any]) -> list[dict[str, Any]]:
    out=[]
    models=validation.get("models") or {}
    for model_name,block in models.items():
        if not isinstance(block,dict):
            continue
        for period in ("last_7d","last_30d","all_available"):
            m=block.get(period)
            if not isinstance(m,dict):
                continue
            row={"Model":model_name,"Period":period}
            row.update({k:v for k,v in m.items() if not isinstance(v,(dict,list))})
            out.append(row)
        for m in block.get("by_lead_day_30d") or []:
            if isinstance(m,dict):
                row={"Model":model_name,"Period":f"30d_lead_day_{m.get('lead_day')}"}
                row.update({k:v for k,v in m.items() if k!="lead_day" and not isinstance(v,(dict,list))})
                out.append(row)
    return out


def prune_monthly_reports(now: datetime) -> None:
    # Keep current month plus previous 23 UTC months.
    y,m=now.year,now.month
    idx=y*12+(m-1)
    cutoff_idx=idx-(REPORT_RETENTION_MONTHS-1)
    for path in OUT_DOCS.glob("swift_space_weather_????-??.xlsx"):
        try:
            ym=path.stem[-7:]; yy=int(ym[:4]); mm=int(ym[5:7]); pidx=yy*12+(mm-1)
            if pidx<cutoff_idx:
                path.unlink()
        except Exception:
            continue


def load_wind_forecast_archive() -> list[dict[str, Any]]:
    path = DATA / "swift-wind" / "forecast-archive.json"
    obj = load_json(path) or {}
    out = []
    for r in records(obj, keys=("items", "records", "data")):
        issued = parse_time(r.get("issued_at"))
        target = parse_time(r.get("target_time") or r.get("time"))
        pred = num(r.get("predicted_speed"))
        if issued and target and pred is not None:
            out.append({
                "_issued": issued, "issued_at": iso_z(issued),
                "_t": target, "time": iso_z(target),
                "predicted_speed": pred,
                "background_speed": num(r.get("background_speed")),
                "lead_hours": num(r.get("lead_hours")),
            })
    return sorted(out, key=lambda x: (x["_issued"], x["_t"]))


def select_forecast_snapshots(rows_in: list[dict[str, Any]], now: datetime, hours_list=(24,48,72)) -> list[dict[str, Any]]:
    if not rows_in:
        return []
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows_in:
        groups.setdefault(r["issued_at"], []).append(r)
    issued = [(parse_time(k), k) for k in groups]
    issued = [(d,k) for d,k in issued if d]
    out = []
    for h in hours_list:
        target_issue = now - timedelta(hours=h)
        if not issued:
            continue
        d,k = min(issued, key=lambda q: abs((q[0]-target_issue).total_seconds()))
        if abs((d-target_issue).total_seconds()) > 9*3600:
            continue
        for r in groups[k]:
            if now - timedelta(days=3) <= r["_t"] <= now + timedelta(days=3):
                q=dict(r); q["snapshot_hours_ago"]=h; out.append(q)
    return out



def bz_min_history_3h(mag_hist: list[dict[str, Any]], now: datetime, days: int = 7) -> list[dict[str, Any]]:
    cutoff = now - timedelta(days=days)
    buckets: dict[datetime, list[dict[str, Any]]] = {}
    for r in mag_hist:
        t = r.get("_t")
        bz = num(r.get("bz"))
        if not isinstance(t, datetime) or t < cutoff or bz is None:
            continue
        b = t.replace(hour=(t.hour // 3) * 3, minute=0, second=0, microsecond=0)
        buckets.setdefault(b, []).append(r)
    out = []
    for t in sorted(buckets):
        rr = buckets[t]
        bz_vals = [num(x.get("bz")) for x in rr if num(x.get("bz")) is not None]
        bt_vals = [num(x.get("bt")) for x in rr if num(x.get("bt")) is not None]
        if not bz_vals:
            continue
        out.append({
            "time": iso_z(t), "_t": t,
            "bz_min": round(min(bz_vals), 3),
            "bz_median": round(statistics.median(bz_vals), 3),
            "bt_median": round(statistics.median(bt_vals), 3) if bt_vals else None,
        })
    return out


def _validation_lead_pairs(validation: dict[str, Any], family: str) -> dict[str, list[dict[str, Any]]]:
    obj = (((validation.get("nominal_lead_skill") or {}).get("pairs") or {}).get(family) or {})
    return {str(k): [x for x in v if isinstance(x, dict)] for k, v in obj.items() if isinstance(v, list)}


def _kp_range_label(v: Any) -> str | None:
    x = num(v)
    if x is None:
        return None
    if 1.0 <= x < 5.0: return "Kp 1-4"
    if 5.0 <= x < 6.0: return "Kp 5-<6"
    if 6.0 <= x < 8.0: return "Kp 6-7"
    if 8.0 <= x <= 9.0: return "Kp 8-9"
    return None


def _kp_pair_metrics(rr: list[dict[str, Any]]) -> dict[str, Any]:
    if not rr:
        return {"count": 0, "mae": None, "bias": None, "hit_rate_067": None, "hit_rate_1": None}
    e = [num(x.get("error")) for x in rr]; e = [x for x in e if x is not None]
    if not e:
        return {"count": 0, "mae": None, "bias": None, "hit_rate_067": None, "hit_rate_1": None}
    return {
        "count": len(e),
        "mae": round(sum(abs(x) for x in e) / len(e), 3),
        "bias": round(sum(e) / len(e), 3),
        "hit_rate_067": round(100 * sum(abs(x) <= .67 for x in e) / len(e), 1),
        "hit_rate_1": round(100 * sum(abs(x) <= 1.0 for x in e) / len(e), 1),
    }


def _bz_pair_metrics(rr: list[dict[str, Any]]) -> dict[str, Any]:
    if not rr:
        return {"count": 0, "mae": None, "bias": None, "hit_rate_2nt": None, "hit_rate_3nt": None, "direction_hit_rate_threshold60": None, "brier_score": None}
    e = [num(x.get("error_bz_min")) for x in rr]; e = [x for x in e if x is not None]
    if not e:
        return {"count": 0, "mae": None, "bias": None, "hit_rate_2nt": None, "hit_rate_3nt": None, "direction_hit_rate_threshold60": None, "brier_score": None}
    valid = [x for x in rr if num(x.get("error_bz_min")) is not None]
    dh = sum((num(x.get("southward_probability"), 0) >= 60) == bool(x.get("observed_southward")) for x in valid)
    b = [num(x.get("brier")) for x in valid]; b = [x for x in b if x is not None]
    return {
        "count": len(valid),
        "mae": round(sum(abs(x) for x in e) / len(e), 3),
        "bias": round(sum(e) / len(e), 3),
        "hit_rate_2nt": round(100 * sum(abs(x) <= 2.0 for x in e) / len(e), 1),
        "hit_rate_3nt": round(100 * sum(abs(x) <= 3.0 for x in e) / len(e), 1),
        "direction_hit_rate_threshold60": round(100 * dh / len(valid), 1),
        "brier_score": round(sum(b) / len(b), 4) if b else None,
    }


def _monthly_pair_subset(rows: list[dict[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        t = parse_time(r.get("target_time"))
        if t and start <= t < end:
            out.append(r)
    return out

def main() -> None:
    now = utcnow()
    month_start, month_end = month_bounds(now)
    month = month_start.strftime("%Y-%m")
    monthly_name = MONTHLY_NAME.format(month=month)
    final_path = OUT_DOCS / monthly_name
    tmp_path = OUT_DOCS / f".{monthly_name}.tmp"

    wind_hist = load_wind_history()
    mag_hist = load_mag_history()
    kp_hist = load_kp_history()
    swift_wind = load_swift_wind()
    wind_archive = load_wind_forecast_archive()
    wind_snapshots = select_forecast_snapshots(wind_archive, now)
    swift_kp, swift_kp_hist = load_swift_kp()
    cmes = load_cme_arrivals()
    cme_model = load_cme_boost_model()
    bz_ai = load_bz_ai()
    validation = load_research_validation(month)
    charging = load_swift_charging()
    charging_verification = load_charging_verification() or (charging.get("verification") or {})
    charging_archive = load_charging_archive()
    validation_history_obj = load_json(DATA / 'validation' / 'history.json') or {}
    validation_history = [x for x in (validation_history_obj.get('history') or []) if isinstance(x, dict)]
    coronal_holes = load_json(DATA / 'coronal-holes' / 'latest.json') or {}
    coronal_hole_history = load_json(DATA / 'coronal-holes' / 'history.json') or {}
    enlil_latest = load_enlil_latest()
    wsa_boundary = load_wsa_boundary()

    wind_hourly_all = bin_hourly(wind_hist, ["speed", "density"], HISTORY_DAYS)
    mag_hourly_all = bin_hourly(mag_hist, ["bz", "bt"], HISTORY_DAYS)
    wind_hourly = in_range(wind_hourly_all, month_start, month_end)
    mag_hourly = in_range(mag_hourly_all, month_start, month_end)
    mag_by_time = {r["time"]: r for r in mag_hourly}
    sw_month = []
    for r in wind_hourly:
        m = mag_by_time.get(r["time"], {})
        x = dict(r)
        x["bz"] = m.get("bz")
        x["bt"] = m.get("bt")
        sw_month.append(x)

    wind_fc = wind_forecast_3d(swift_wind, wind_hist, cmes)
    kp_fc = kp_forecast_rows(swift_kp, wind_fc, bz_ai)
    bz_fc = []
    for r in records(bz_ai, keys=("forecast",)) if bz_ai else []:
        t = parse_time(r.get("time"))
        if t and t <= now + timedelta(days=FORECAST_DAYS):
            bz_fc.append({
                "_t": t, "time": iso_z(t), "bz_forecast": num(r.get("bz_forecast")),
                "bz_min_forecast": num(r.get("bz_min_forecast")), "bt_forecast": num(r.get("bt_forecast")),
                "southward_bz_probability": num(r.get("southward_bz_probability")),
                "bz_risk": r.get("bz_risk"), "cme_id": r.get("cme_id"),
            })
    bz_fc.sort(key=lambda x: x["_t"])

    wind_acc_rows = wind_accuracy(swift_wind, wind_hist)
    kp_acc_rows = kp_accuracy(swift_kp_hist)
    current_acc = current_wind_accuracy(swift_wind)
    wind_metrics = [metric_summary(wind_acc_rows, d, "wind") for d in (1, 7, 30)]
    kp_metrics = [metric_summary(kp_acc_rows, d, "kp") for d in (1, 7, 30)]
    daily_rows = daily_summary_rows(month_start, now, wind_hourly_all, kp_hist, wind_acc_rows, cmes)
    month_metrics = month_metric_rows(month_start, month_end, wind_hist, mag_hist, kp_hist, wind_acc_rows, cmes, current_acc)

    charging_fc = []
    for r in records(charging, keys=("forecast",)) if charging else []:
        t = parse_time(r.get("time"))
        if not t:
            continue
        charging_fc.append({
            "_t": t, "time": iso_z(t), "lead_hours": num(r.get("lead_hours")),
            "surface_kv": num(r.get("surface_kv")), "differential_kv": num(r.get("differential_kv")),
            "internal_field_mvm": num(r.get("internal_field_mvm")),
            "electron_flux_gt2mev": num(r.get("electron_flux_gt2mev")),
            "electron_fluence_24h_proxy": num(r.get("electron_fluence_24h_proxy")),
            "kp": num(r.get("kp")), "bz_min_nt": num(r.get("bz_min_nt")), "wind_kms": num(r.get("wind_kms")),
            "source": r.get("source"),
        })
    charging_fc.sort(key=lambda x: x["_t"])
    charging_obs = []
    for r in records(charging, keys=("observed",)) if charging else []:
        t = parse_time(r.get("time"))
        if t:
            charging_obs.append({"_t": t, "time": iso_z(t), "surface_kv": num(r.get("surface_kv")), "differential_kv": num(r.get("differential_kv")), "internal_field_mvm": num(r.get("internal_field_mvm")), "electron_flux_gt2mev": num(r.get("electron_flux_gt2mev")), "source": r.get("source")})
    charging_obs.sort(key=lambda x: x["_t"])
    surface_skill = _charging_skill_map(charging_verification, "surface")
    internal_skill = _charging_skill_map(charging_verification, "internal")
    surface_vals = [r["surface_kv"] for r in charging_fc if r.get("surface_kv") is not None]
    internal_vals = [r["internal_field_mvm"] for r in charging_fc if r.get("internal_field_mvm") is not None]
    if surface_vals:
        month_metrics.append({"Metric": "Charging forecast 72h most negative surface", "Value": round(min(surface_vals), 2), "Unit": "kV"})
    if internal_vals:
        month_metrics.append({"Metric": "Charging forecast 72h max internal field", "Value": round(max(internal_vals), 3), "Unit": "MV/m"})
    s24 = surface_skill.get(24, {})
    i24 = internal_skill.get(24, {})
    month_metrics.append({"Metric": "Surface charging 24h hit ±1 kV", "Value": s24.get("hit_rate_1kv"), "Unit": "%"})
    month_metrics.append({"Metric": "Internal charging 24h hit ±0.1 MV/m", "Value": i24.get("hit_rate_0_1mvm"), "Unit": "%"})

    # Companion JSON for the browser UI.  It is built from the exact same
    # arrays used below for the monthly Excel sheets, so Kp/Bz/Wind remain in sync.
    wind_hist_27d = [r for r in wind_hourly_all if r.get("_t") and r["_t"] >= now - timedelta(days=27)]
    bz_hist_3h = bz_min_history_3h(mag_hist, now, 7)
    nominal_skill = validation.get("nominal_lead_skill") or {}
    past_snapshots = validation.get("past_forecast_snapshots") or {}
    generation = validation.get("generation") or nominal_skill.get("generation") or {}
    calibration = validation.get("calibration") or {}
    skill_history_30d = [x for x in validation_history if parse_time(x.get("time")) and parse_time(x.get("time")) >= now - timedelta(days=30)][-400:]
    ui_payload = {
        "updated_at": iso_z(now),
        "month": month,
        "wind_history": [
            {"time": r["time"], "speed": r.get("speed"), "density": r.get("density"), "source": "NOAA/SWIFT history"}
            for r in wind_hist_27d
        ],
        "wind_forecast": [
            {"time": r["time"], "predicted_speed": r.get("predicted_speed"), "background_speed": r.get("background_speed"),
             "observed_speed": r.get("observed_speed"), "cme_boost": r.get("cme_boost"), "cme_id": r.get("cme_id"), "source": r.get("source")}
            for r in wind_fc
        ],
        "kp_history": [
            {"time": r["time"], "kp": r.get("kp"), "source": "NOAA observed"}
            for r in kp_hist if r.get("_t") and r["_t"] >= now - timedelta(days=7)
        ],
        "kp_forecast": [
            {"time": r["time"], "kp": r.get("kp"), "g_scale": r.get("g_scale"), "confidence": r.get("confidence"), "source": r.get("source")}
            for r in kp_fc
        ],
        "bz_history": [
            {"time": r["time"], "bz_min": r.get("bz_min"), "bz_median": r.get("bz_median"), "bt_median": r.get("bt_median"), "source": "NOAA IMF observed 3h bin"}
            for r in bz_hist_3h
        ],
        "bz_forecast": [
            {"time": r["time"], "bz_forecast": r.get("bz_forecast"), "bz_min_forecast": r.get("bz_min_forecast"),
             "bt_forecast": r.get("bt_forecast"), "southward_bz_probability": r.get("southward_bz_probability"),
             "bz_risk": r.get("bz_risk"), "cme_id": r.get("cme_id"), "source": "SWIFT Bz AI"}
            for r in bz_fc
        ],
        "wind_forecast_history": [
            {"issued_at": r["issued_at"], "time": r["time"], "predicted_speed": r.get("predicted_speed"),
             "background_speed": r.get("background_speed"), "lead_hours": r.get("lead_hours"),
             "snapshot_hours_ago": r.get("snapshot_hours_ago")}
            for r in wind_snapshots
        ],
        "kp_model": {
            "model": (swift_kp or {}).get("model"),
            "kp_floor": (swift_kp or {}).get("kp_floor"),
            "leadtime_skill": (swift_kp or {}).get("leadtime_skill"),
            "nominal_lead_bias_calibration": (swift_kp or {}).get("nominal_lead_bias_calibration") or ((swift_kp or {}).get("coefficients") or {}).get("nominal_lead_bias_calibration"),
            "current_observed_anchor": (swift_kp or {}).get("current_observed_anchor"),
            "recent_trend_3h": (swift_kp or {}).get("recent_trend_3h"),
        },
        "bz_model": {
            "model": (bz_ai or {}).get("model"),
            "readiness": (bz_ai or {}).get("readiness"),
            "nominal_lead_bias_calibration": (bz_ai or {}).get("nominal_lead_bias_calibration") or ((bz_ai or {}).get("coefficients") or {}).get("nominal_lead_bias_calibration"),
        },
        "charging": {
            "updated_at": charging.get("updated_at"),
            "model": charging.get("model"),
            "reference_spacecraft": charging.get("reference_spacecraft"),
            "coefficients": charging.get("coefficients"),
            "forecast": [
                {k: r.get(k) for k in ("time", "lead_hours", "surface_kv", "differential_kv", "internal_field_mvm", "electron_flux_gt2mev", "electron_fluence_24h_proxy", "kp", "bz_min_nt", "wind_kms", "source")}
                for r in charging_fc
            ],
            "observed": [
                {k: r.get(k) for k in ("time", "surface_kv", "differential_kv", "internal_field_mvm", "electron_flux_gt2mev", "source")}
                for r in charging_obs[-1000:]
            ],
            "verification": charging_verification,
            "past_forecast_snapshots": charging.get("past_forecast_snapshots") or {},
            "methodology": charging.get("methodology") or {},
        },
        "wind_accuracy_current": current_acc,
        "research_validation": {
            "models": validation.get("models", {}),
            "uncertainty": validation.get("uncertainty", {}),
            "generation": generation,
            "calibration": calibration,
            "nominal_lead_skill": nominal_skill,
            "past_forecast_snapshots": past_snapshots,
            "skill_history_30d": skill_history_30d,
            "methodology": validation.get("methodology", {}),
        },
        "enlil": {
            "updated_at": enlil_latest.get("updated_at"),
            "model": enlil_latest.get("model"),
            "forecast": enlil_latest.get("forecast", []),
            "wsa_boundary": {
                "updated_at": wsa_boundary.get("updated_at"),
                "source_file": wsa_boundary.get("source_file"),
                "stats": wsa_boundary.get("stats"),
            },
        },
        "excel_file": monthly_name,
    }
    (OUT_DOCS / "ui_forecast_latest.json").write_text(json.dumps(ui_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    wb = xlsxwriter.Workbook(str(tmp_path), {"nan_inf_to_errors": True})
    fmt = {
        "title": wb.add_format({"bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "align": "center", "valign": "vcenter"}),
        "section": wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#305496"}),
        "header": wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1, "align": "center", "valign": "vcenter"}),
        "body": wb.add_format({"border": 1, "valign": "vcenter"}),
        "datetime": wb.add_format({"num_format": "yyyy-mm-dd hh:mm", "border": 1}),
        "date": wb.add_format({"num_format": "yyyy-mm-dd", "border": 1}),
        "num1": wb.add_format({"num_format": "0.0", "border": 1}),
        "num2": wb.add_format({"num_format": "0.00", "border": 1}),
        "url": wb.add_format({"font_color": "blue", "underline": True}),
        "good": wb.add_format({"bg_color": "#E2F0D9", "border": 1}),
        "warn": wb.add_format({"bg_color": "#FFF2CC", "border": 1}),
        "bad": wb.add_format({"bg_color": "#F8CBAD", "border": 1}),
    }

    # Monthly summary + daily summary
    ws = wb.add_worksheet("Monthly_Summary")
    ws.merge_range("A1:H1", f"SWIFT Monthly Space Weather Summary — {month} UTC", fmt["title"])
    ws.write("A3", "Updated UTC", fmt["section"]); ws.write("B3", iso_z(now))
    ws.write("A4", "Workbook", fmt["section"]); ws.write("B4", monthly_name)
    ws.write("A6", "Monthly KPI", fmt["section"])
    write_table(ws, 6, 0, ["Metric", "Value", "Unit"], month_metrics, fmt)
    ws.write(6, 4, "Daily Summary", fmt["section"])
    write_table(ws, 7, 4, ["Date", "Wind_Avg_km_s", "Wind_Min_km_s", "Wind_Max_km_s", "Wind_P95_km_s", "Kp_Max", "Accuracy_Count", "Wind_Hit_50_pct", "Wind_MAE_km_s", "CME_Arrivals"], daily_rows, fmt)
    ws.set_column("A:A", 38); ws.set_column("B:B", 18); ws.set_column("C:C", 12); ws.set_column("E:N", 16)
    # Daily wind and accuracy charts use real daily values; raw hourly zig-zag is on the SolarWind sheet.
    if daily_rows:
        chart = wb.add_chart({"type": "line"})
        chart.add_series({"name": "Daily mean wind", "categories": ["Monthly_Summary", 8, 4, 7 + len(daily_rows), 4], "values": ["Monthly_Summary", 8, 5, 7 + len(daily_rows), 5], "marker": {"type": "circle", "size": 4}})
        chart.set_title({"name": "Daily mean solar wind"}); chart.set_y_axis({"name": "km/s"}); chart.set_legend({"none": True}); chart.set_size({"width": 760, "height": 280}); ws.insert_chart("A27", chart)
        chart2 = wb.add_chart({"type": "line"})
        chart2.add_series({"name": "±50 km/s hit rate", "categories": ["Monthly_Summary", 8, 4, 7 + len(daily_rows), 4], "values": ["Monthly_Summary", 8, 11, 7 + len(daily_rows), 11], "marker": {"type": "circle", "size": 4}})
        chart2.set_title({"name": "Daily wind forecast hit rate ±50 km/s"}); chart2.set_y_axis({"name": "%", "min": 0, "max": 100}); chart2.set_legend({"none": True}); chart2.set_size({"width": 760, "height": 280}); ws.insert_chart("A43", chart2)

    # Hourly observed month; no curve smoothing. This produces the actual zig-zag shape.
    ws = wb.add_worksheet("SolarWind_Month")
    sw_rows = prepare_rows_for_excel(sw_month, [("UTC", "_t"), ("Speed_km_s", "speed"), ("Density_cm3", "density"), ("Bz_nT", "bz"), ("Bt_nT", "bt")])
    write_table(ws, 0, 0, ["UTC", "Speed_km_s", "Density_cm3", "Bz_nT", "Bt_nT"], sw_rows, fmt)
    ws.set_column("A:A", 19); ws.set_column("B:E", 14)
    add_line_chart(wb, ws, f"{month} hourly solar wind speed — unsmoothed", "SolarWind_Month", 0, len(sw_rows), 0, [(1, "Speed km/s")], "G2", "km/s")
    add_line_chart(wb, ws, f"{month} hourly IMF Bz/Bt", "SolarWind_Month", 0, len(sw_rows), 0, [(3, "Bz nT"), (4, "Bt nT")], "G20", "nT")

    ws = wb.add_worksheet("Wind_Forecast_3d")
    wf_rows = prepare_rows_for_excel(wind_fc, [("UTC", "_t"), ("Predicted_Speed_km_s", "predicted_speed"), ("Background_Speed_km_s", "background_speed"), ("Observed_Speed_km_s", "observed_speed"), ("CME_Boost_km_s", "cme_boost"), ("CME_ID", "cme_id"), ("Source", "source")])
    write_table(ws, 0, 0, ["UTC", "Predicted_Speed_km_s", "Background_Speed_km_s", "Observed_Speed_km_s", "CME_Boost_km_s", "CME_ID", "Source"], wf_rows, fmt)
    ws.set_column("A:A", 19); ws.set_column("B:E", 18); ws.set_column("F:G", 28)
    add_line_chart(wb, ws, "Next 3 days solar wind forecast — hourly points", "Wind_Forecast_3d", 0, len(wf_rows), 0, [(1, "Predicted"), (2, "Background"), (4, "CME boost")], "I2", "km/s")

    ws = wb.add_worksheet("Wind_Fcst_History")
    hist_rows = []
    # Include recent archived forecast points and pair them with nearest observed hourly wind.
    obs_by_hour = {r["_t"].replace(minute=0, second=0, microsecond=0): r for r in wind_hourly_all if r.get("_t")}
    for r in wind_archive:
        if r["_t"] < month_start or r["_t"] >= month_end:
            continue
        oh = r["_t"].replace(minute=0, second=0, microsecond=0)
        obs = obs_by_hour.get(oh, {})
        observed = obs.get("speed")
        err = (r["predicted_speed"] - observed) if observed is not None else None
        hist_rows.append({
            "Issued_UTC": r["_issued"].replace(tzinfo=None), "Target_UTC": r["_t"].replace(tzinfo=None),
            "Lead_h": r.get("lead_hours"), "Predicted_km_s": r.get("predicted_speed"),
            "Observed_km_s": observed, "Error_km_s": err,
            "Within_50": (abs(err) <= 50) if err is not None else None,
        })
    write_table(ws, 0, 0, ["Issued_UTC", "Target_UTC", "Lead_h", "Predicted_km_s", "Observed_km_s", "Error_km_s", "Within_50"], hist_rows, fmt)
    ws.set_column("A:B", 19); ws.set_column("C:G", 16)
    if hist_rows:
        chart = wb.add_chart({"type":"line"})
        chart.add_series({"name":"Past forecast", "categories":["Wind_Fcst_History",1,1,len(hist_rows),1], "values":["Wind_Fcst_History",1,3,len(hist_rows),3]})
        chart.add_series({"name":"Observed", "categories":["Wind_Fcst_History",1,1,len(hist_rows),1], "values":["Wind_Fcst_History",1,4,len(hist_rows),4]})
        chart.set_title({"name":"Archived SWIFT forecast vs observed wind"}); chart.set_y_axis({"name":"km/s"}); chart.set_size({"width":840,"height":340}); ws.insert_chart("I2",chart)

    ws = wb.add_worksheet("Kp_Forecast")
    kpf_rows = prepare_rows_for_excel(kp_fc, [("UTC", "_t"), ("Kp", "kp"), ("G_Scale", "g_scale"), ("Confidence", "confidence"), ("Source", "source")])
    write_table(ws, 0, 0, ["UTC", "Kp", "G_Scale", "Confidence", "Source"], kpf_rows, fmt)
    ws.set_column("A:A", 19); ws.set_column("B:D", 14); ws.set_column("E:E", 28)
    add_line_chart(wb, ws, "Kp forecast — next 72 hours", "Kp_Forecast", 0, len(kpf_rows), 0, [(1, "Kp")], "G2", "Kp")

    ws = wb.add_worksheet("Accuracy")
    ws.merge_range("A1:H1", "Wind accuracy — primary target ±50 km/s", fmt["title"])
    write_table(ws, 2, 0, ["kind", "period", "count", "mae", "rmse", "bias", "hit_rate_within_50kms", "hit_rate_within_100kms"], wind_metrics, fmt)
    ws.write("A8", "Current ±50 km/s hit rate (last 24h)", fmt["section"]); ws.write("B8", current_acc.get("hit_rate_50kms"), fmt["body"]); ws.write("C8", current_acc.get("count"), fmt["body"])
    ws.merge_range("A10:I10", "Kp accuracy", fmt["title"])
    write_table(ws, 10, 0, ["kind", "period", "count", "mae", "rmse", "bias", "hit_rate_within_067kp", "hit_rate_within_1kp", "g_scale_hit_rate"], kp_metrics, fmt)
    wr = prepare_rows_for_excel(in_range(wind_acc_rows, month_start, month_end), [("UTC", "_t"), ("Observed", "observed"), ("Predicted", "predicted"), ("Background", "background"), ("Error", "error")])
    ws.write(16, 0, f"Wind detail — {month}", fmt["section"]); write_table(ws, 17, 0, ["UTC", "Observed", "Predicted", "Background", "Error"], wr, fmt)
    ws.set_column("A:B", 19); ws.set_column("C:I", 17)

    ws = wb.add_worksheet("Bz_AI")
    bz_rows = prepare_rows_for_excel(bz_fc, [("UTC", "_t"), ("Bz_Forecast_nT", "bz_forecast"), ("Bz_Min_Forecast_nT", "bz_min_forecast"), ("Bt_Forecast_nT", "bt_forecast"), ("Southward_Probability_pct", "southward_bz_probability"), ("Risk", "bz_risk"), ("CME_ID", "cme_id")])
    write_table(ws, 0, 0, ["UTC", "Bz_Forecast_nT", "Bz_Min_Forecast_nT", "Bt_Forecast_nT", "Southward_Probability_pct", "Risk", "CME_ID"], bz_rows, fmt)
    ws.set_column("A:A", 19); ws.set_column("B:E", 22); ws.set_column("F:G", 24)
    add_line_chart(wb, ws, "Bz AI forecast", "Bz_AI", 0, len(bz_rows), 0, [(1, "Bz forecast"), (2, "Bz min"), (3, "Bt")], "I2", "nT")

    ws = wb.add_worksheet("CME_Arrivals")
    cme_rows = []
    for c in cmes:
        at = c.get("_arrival")
        if not isinstance(at, datetime) or not (month_start <= at < month_end):
            continue
        t0 = c.get("_t0")
        cme_rows.append({
            "ID": c.get("id"), "Source": c.get("source"), "Impact_Class": c.get("impact_class"), "Earth_Candidate": c.get("earth_candidate"),
            "Start_UTC": t0.replace(tzinfo=None) if isinstance(t0, datetime) else None, "Arrival_UTC": at.replace(tzinfo=None),
            "Transit_h": c.get("transit_hours"), "Speed_km_s": c.get("speed"), "Effective_Speed_km_s": c.get("effective_speed"),
            "Expected_Wind_Increase_km_s": c.get("expected_wind_speed_increase"), "Longitude": c.get("longitude"), "Latitude": c.get("latitude"),
            "Angular_Separation_deg": c.get("angular_separation"), "Confidence": c.get("confidence"),
        })
    write_table(ws, 0, 0, ["ID", "Source", "Impact_Class", "Earth_Candidate", "Start_UTC", "Arrival_UTC", "Transit_h", "Speed_km_s", "Effective_Speed_km_s", "Expected_Wind_Increase_km_s", "Longitude", "Latitude", "Angular_Separation_deg", "Confidence"], cme_rows, fmt)
    ws.set_column("A:A", 35); ws.set_column("B:D", 16); ws.set_column("E:F", 19); ws.set_column("G:N", 16)

    ws = wb.add_worksheet("CME_Wind_Boost")
    model_rows = []
    if cme_model:
        for cls, item in (cme_model.get("by_impact_class") or {}).items():
            model_rows.append({"Impact_Class": cls, "Count": item.get("count"), "Median_Delta_V": item.get("median_delta_v"), "Mean_Delta_V": item.get("mean_delta_v"), "Default_Used": item.get("default_used")})
    write_table(ws, 0, 0, ["Impact_Class", "Count", "Median_Delta_V", "Mean_Delta_V", "Default_Used"], model_rows, fmt)
    samples = cme_model.get("samples", []) if cme_model else []
    sample_rows = []
    for s in samples[-300:]:
        at = parse_time(s.get("arrival_time"))
        sample_rows.append({"ID": s.get("id"), "Impact_Class": s.get("impact_class"), "Arrival_UTC": at.replace(tzinfo=None) if at else None, "Speed_km_s": s.get("speed"), "Baseline_Speed": s.get("baseline_speed"), "Post_Peak_Speed": s.get("post_peak_speed"), "Observed_Delta_V": s.get("observed_delta_v")})
    ws.write(9, 0, "Learning samples", fmt["section"]); write_table(ws, 10, 0, ["ID", "Impact_Class", "Arrival_UTC", "Speed_km_s", "Baseline_Speed", "Post_Peak_Speed", "Observed_Delta_V"], sample_rows, fmt)
    ws.set_column("A:A", 35); ws.set_column("B:G", 18)


    # Coronal-hole detections and HSS priors (two-year history retained in JSON).
    ws = wb.add_worksheet("Coronal_Holes")
    headers = ["Snapshot UTC","CH ID","Area frac","Latitude deg","CMD deg","Width deg proxy","DCHB deg proxy",
               "Unipolarity proxy","Earth score","Source strength","Pred peak km/s","Pred deltaV km/s",
               "Arrival UTC","Confidence","Method"]
    for c,h in enumerate(headers): ws.write(0,c,h,fmt["header"])
    rr=1
    for snap in (coronal_hole_history.get("items") or []):
        st=snap.get("time")
        impacts={str(x.get("source_id")):x for x in (snap.get("earth_impacts") or [])}
        for ch in (snap.get("coronal_holes") or []):
            im=impacts.get(str(ch.get("id")),{})
            vals=[st,ch.get("id"),ch.get("area_fraction_disk_deprojected"),ch.get("latitude_deg"),
                  ch.get("central_meridian_deg"),ch.get("width_deg_proxy"),ch.get("dchb_proxy_deg"),
                  ch.get("hmi_unipolarity_proxy"),ch.get("earth_facing_score"),ch.get("source_strength"),
                  ch.get("predicted_peak_speed_km_s_prior"),ch.get("predicted_delta_v_km_s_prior"),
                  im.get("arrival_time"),im.get("confidence"),im.get("method")]
            for c,v in enumerate(vals):ws.write(rr,c,v)
            rr+=1
    ws.freeze_panes(1,0);ws.autofilter(0,0,max(0,rr-1),len(headers)-1)
    ws.set_column(0,1,22);ws.set_column(2,13,14);ws.set_column(14,14,48)

    # Research-grade operational verification. These are issued forecasts scored only after observations arrive.
    pairs = validation.get("pairs") or {}
    ws = wb.add_worksheet("Research_Metrics")
    ws.merge_range("A1:L1", f"Operational forecast verification — {month} UTC", fmt["title"])
    metric_rows = metric_rows_from_validation(validation)
    metric_headers = ["Model","Period","count","mae","median_ae","rmse","bias","p90_ae","hit_rate_50","hit_rate_067","hit_rate_1","brier_score","direction_hit_rate_threshold60","pearson_r","mae_hours","bias_hours","within_6h","within_12h"]
    write_table(ws, 2, 0, metric_headers, metric_rows, fmt)
    ws.set_column("A:B", 22); ws.set_column("C:R", 14)
    ws.write(1, 0, "Primary metrics are true issued-forecast verification, not hindcast fit.", fmt["section"])

    ws = wb.add_worksheet("Wind_Verification")
    wv=[]
    for r in pairs.get("wind",[]):
        it=parse_time(r.get("issued_at"));tt=parse_time(r.get("target_time"))
        wv.append({"Issued_UTC":it.replace(tzinfo=None) if it else None,"Target_UTC":tt.replace(tzinfo=None) if tt else None,"Lead_h":r.get("lead_hours"),"Predicted_km_s":r.get("predicted"),"Observed_km_s":r.get("observed"),"Error_km_s":r.get("error"),"Abs_Error_km_s":r.get("abs_error"),"Model":r.get("model")})
    write_table(ws,0,0,["Issued_UTC","Target_UTC","Lead_h","Predicted_km_s","Observed_km_s","Error_km_s","Abs_Error_km_s","Model"],wv,fmt)
    ws.set_column("A:B",19);ws.set_column("C:G",16);ws.set_column("H:H",34)

    ws = wb.add_worksheet("ENLIL_Verification")
    ev=[]
    for r in pairs.get("enlil_wind",[]):
        it=parse_time(r.get("issued_at"));tt=parse_time(r.get("target_time"))
        ev.append({"Issued_UTC":it.replace(tzinfo=None) if it else None,"Target_UTC":tt.replace(tzinfo=None) if tt else None,"Lead_h":r.get("lead_hours"),"ENLIL_km_s":r.get("predicted"),"Observed_km_s":r.get("observed"),"Error_km_s":r.get("error"),"Abs_Error_km_s":r.get("abs_error")})
    write_table(ws,0,0,["Issued_UTC","Target_UTC","Lead_h","ENLIL_km_s","Observed_km_s","Error_km_s","Abs_Error_km_s"],ev,fmt)
    ws.set_column("A:B",19);ws.set_column("C:G",16)

    ws = wb.add_worksheet("Kp_Verification")
    kv=[]
    for r in pairs.get("kp",[]):
        it=parse_time(r.get("issued_at"));tt=parse_time(r.get("target_time"))
        kv.append({"Issued_UTC":it.replace(tzinfo=None) if it else None,"Target_UTC":tt.replace(tzinfo=None) if tt else None,"Lead_h":r.get("lead_hours"),"Predicted_Kp":r.get("predicted"),"Observed_Kp":r.get("observed"),"Error_Kp":r.get("error"),"Abs_Error_Kp":r.get("abs_error"),"Within_0.67":r.get("within_067"),"Within_1.0":r.get("within_1"),"Model":r.get("model")})
    write_table(ws,0,0,["Issued_UTC","Target_UTC","Lead_h","Predicted_Kp","Observed_Kp","Error_Kp","Abs_Error_Kp","Within_0.67","Within_1.0","Model"],kv,fmt)
    ws.set_column("A:B",19);ws.set_column("C:I",15);ws.set_column("J:J",34)

    ws = wb.add_worksheet("Bz_Verification")
    bv=[]
    for r in pairs.get("bz",[]):
        it=parse_time(r.get("issued_at"));tt=parse_time(r.get("target_time"))
        bv.append({"Issued_UTC":it.replace(tzinfo=None) if it else None,"Target_UTC":tt.replace(tzinfo=None) if tt else None,"Lead_h":r.get("lead_hours"),"Forecast_BzMin_nT":r.get("predicted_bz_min"),"Observed_BzMin_nT":r.get("observed_bz_min"),"Error_nT":r.get("error_bz_min"),"Southward_Prob_pct":r.get("southward_probability"),"Observed_Southward":r.get("observed_southward"),"Brier":r.get("brier")})
    write_table(ws,0,0,["Issued_UTC","Target_UTC","Lead_h","Forecast_BzMin_nT","Observed_BzMin_nT","Error_nT","Southward_Prob_pct","Observed_Southward","Brier"],bv,fmt)
    ws.set_column("A:B",19);ws.set_column("C:I",18)


    # Kp/Bz 24 h / 48 h / 72 h operational lead skill and past forecast history.
    lead_skill = validation.get("nominal_lead_skill") or {}
    kp_lead_pairs = _validation_lead_pairs(validation, "kp")
    bz_lead_pairs = _validation_lead_pairs(validation, "bz")

    ws = wb.add_worksheet("Model_Generations")
    gen_rows=[]
    for family in ("kp","bz"):
        g=(generation.get(family) or {}) if isinstance(generation,dict) else {}
        gen_rows.append({"Family":family.upper(),"Current_Model":g.get("current_model"),"Current_Pairs":g.get("current_pairs"),"Legacy_Pairs":g.get("legacy_pairs"),"By_Model_JSON":json.dumps(g.get("by_model") or {},ensure_ascii=False)})
    write_table(ws,0,0,["Family","Current_Model","Current_Pairs","Legacy_Pairs","By_Model_JSON"],gen_rows,fmt)
    ws.set_column("A:A",12);ws.set_column("B:B",42);ws.set_column("C:D",16);ws.set_column("E:E",80)
    ws.write(5,0,"Primary verification rule",fmt["section"]);ws.write(6,0,"Current-model-only: archive.model must exactly match latest.json model. Legacy/unversioned records stay available for historical comparison but are excluded from auto-calibration.",fmt["body"])

    ws = wb.add_worksheet("Auto_Calibration")
    cal_rows=[]
    for family in ("kp","bz"):
        block=(calibration.get(family) or {}) if isinstance(calibration,dict) else {}
        cal=block.get("nominal_lead_bias_calibration") or {}
        for h in (24,48,72):
            x=((cal.get("leads") or {}).get(str(h)) or {})
            cal_rows.append({"Family":family.upper(),"Model":block.get("model") or cal.get("model"),"Lead_h":h,"Count":x.get("count"),"Status":x.get("status"),"Bias":x.get("bias") if family=="kp" else x.get("bz_min_bias"),"Alpha":x.get("alpha"),"Kp_Subtract":x.get("subtract_from_kp"),"BzMin_Add_nT":x.get("add_to_bz_min_nt"),"Probability_Add_pp":x.get("add_to_probability_pp")})
    write_table(ws,0,0,["Family","Model","Lead_h","Count","Status","Bias","Alpha","Kp_Subtract","BzMin_Add_nT","Probability_Add_pp"],cal_rows,fmt)
    ws.set_column("A:A",12);ws.set_column("B:B",42);ws.set_column("C:J",18)

    ws = wb.add_worksheet("Forecast_Skill_History")
    hist_rows=[]
    for x in skill_history_30d:
        tt=parse_time(x.get("time"))
        k=x.get("kp_current_30d") or {};b=x.get("bz_current_30d") or {};w=x.get("swift_wind_30d") or {}
        hist_rows.append({"UTC":tt.replace(tzinfo=None) if tt else None,"Wind_Hit50_pct":w.get("hit_rate_50"),"Wind_MAE":w.get("mae"),"Kp_Hit1_pct":k.get("hit_rate_1"),"Kp_MAE":k.get("mae"),"Bz_Hit2nT_pct":b.get("hit_rate_2nt"),"Bz_MAE_nT":b.get("bz_min_mae"),"Kp_Model":x.get("kp_model"),"Bz_Model":x.get("bz_model")})
    write_table(ws,0,0,["UTC","Wind_Hit50_pct","Wind_MAE","Kp_Hit1_pct","Kp_MAE","Bz_Hit2nT_pct","Bz_MAE_nT","Kp_Model","Bz_Model"],hist_rows,fmt)
    ws.set_column("A:A",19);ws.set_column("B:G",18);ws.set_column("H:I",42)
    if hist_rows:
        add_line_chart(wb,ws,"Rolling 30d primary hit rates","Forecast_Skill_History",0,len(hist_rows),0,[(1,"Wind ±50 km/s"),(3,"Kp ±1.0"),(5,"Bz ±2 nT")],"K2","Hit rate %")

    ws = wb.add_worksheet("Kp_Lead_Skill")
    kp_lead_rows = []
    rolling_kp = {int(x.get("nominal_lead_hours")): x for x in (lead_skill.get("kp") or []) if isinstance(x, dict) and str(x.get("nominal_lead_hours", "")).isdigit()}
    for h in (24, 48, 72):
        all_rows = kp_lead_pairs.get(f"{h}h", [])
        mr = _monthly_pair_subset(all_rows, month_start, month_end)
        mm = _kp_pair_metrics(mr)
        rm = rolling_kp.get(h, {})
        kp_lead_rows.append({
            "Lead_h": h, "Window": "rolling 30d", "Count": rm.get("count"), "MAE_Kp": rm.get("mae"), "Bias_Kp": rm.get("bias"),
            "Hit_within_0.67_pct": rm.get("hit_rate_067"), "Hit_within_1.0_pct": rm.get("hit_rate_1")
        })
        kp_lead_rows.append({
            "Lead_h": h, "Window": month, "Count": mm.get("count"), "MAE_Kp": mm.get("mae"), "Bias_Kp": mm.get("bias"),
            "Hit_within_0.67_pct": mm.get("hit_rate_067"), "Hit_within_1.0_pct": mm.get("hit_rate_1")
        })
    write_table(ws, 0, 0, ["Lead_h","Window","Count","MAE_Kp","Bias_Kp","Hit_within_0.67_pct","Hit_within_1.0_pct"], kp_lead_rows, fmt)
    ws.set_column("A:A", 11); ws.set_column("B:B", 16); ws.set_column("C:G", 20)

    ws = wb.add_worksheet("Kp_Range_Skill")
    kp_range_rows = []
    requested_bands = ["Kp 1-4", "Kp 5-<6", "Kp 6-7", "Kp 8-9"]
    rolling_range = {(str(x.get("observed_range")), str(x.get("nominal_lead_hours"))): x for x in (lead_skill.get("kp_by_observed_range") or []) if isinstance(x, dict)}
    for band in requested_bands:
        for h in (24,48,72):
            rr = [x for x in _monthly_pair_subset(kp_lead_pairs.get(f"{h}h", []), month_start, month_end) if _kp_range_label(x.get("observed")) == band]
            mm = _kp_pair_metrics(rr); rm = rolling_range.get((band, str(h)), {})
            kp_range_rows.append({
                "Observed_Kp_Range": band, "Lead_h": h,
                "Rolling30d_Count": rm.get("count"), "Rolling30d_Hit_1.0_pct": rm.get("hit_rate_1"), "Rolling30d_Hit_0.67_pct": rm.get("hit_rate_067"),
                "Month_Count": mm.get("count"), "Month_Hit_1.0_pct": mm.get("hit_rate_1"), "Month_Hit_0.67_pct": mm.get("hit_rate_067"),
                "Month_MAE_Kp": mm.get("mae"), "Month_Bias_Kp": mm.get("bias")
            })
    write_table(ws, 0, 0, ["Observed_Kp_Range","Lead_h","Rolling30d_Count","Rolling30d_Hit_1.0_pct","Rolling30d_Hit_0.67_pct","Month_Count","Month_Hit_1.0_pct","Month_Hit_0.67_pct","Month_MAE_Kp","Month_Bias_Kp"], kp_range_rows, fmt)
    ws.set_column("A:A", 18); ws.set_column("B:J", 19)
    ws.write(len(kp_range_rows)+2, 0, "Band rule", fmt["section"])
    ws.write(len(kp_range_rows)+3, 0, "1<=Kp<5; 5<=Kp<6; 6<=Kp<8; 8<=Kp<=9. Kp<1 is excluded from this stratified table to match the requested bands.", fmt["body"])
    ws.set_column("A:A", 22)

    ws = wb.add_worksheet("Kp_Fcst_History")
    kh=[]
    for h in (24,48,72):
        for r in _monthly_pair_subset(kp_lead_pairs.get(f"{h}h", []), month_start, month_end):
            it=parse_time(r.get("issued_at"));tt=parse_time(r.get("target_time"))
            kh.append({"Nominal_Lead_h":h,"Issued_UTC":it.replace(tzinfo=None) if it else None,"Target_UTC":tt.replace(tzinfo=None) if tt else None,"Actual_Lead_h":r.get("lead_hours"),"Predicted_Kp":r.get("predicted"),"Observed_Kp":r.get("observed"),"Observed_Range":_kp_range_label(r.get("observed")),"Error_Kp":r.get("error"),"Within_0.67":r.get("within_067"),"Within_1.0":r.get("within_1"),"Model":r.get("model")})
    kh.sort(key=lambda x:(x.get("Target_UTC") or datetime.min, x.get("Nominal_Lead_h") or 0))
    write_table(ws,0,0,["Nominal_Lead_h","Issued_UTC","Target_UTC","Actual_Lead_h","Predicted_Kp","Observed_Kp","Observed_Range","Error_Kp","Within_0.67","Within_1.0","Model"],kh,fmt)
    ws.set_column("A:A",14);ws.set_column("B:C",19);ws.set_column("D:J",15);ws.set_column("K:K",36)
    kp_chart_map={}
    for r in kh:
        tt=r.get("Target_UTC")
        if tt is None:continue
        z=kp_chart_map.setdefault(tt,{"Target_UTC":tt,"Observed_Kp":r.get("Observed_Kp"),"Pred_24h":None,"Pred_48h":None,"Pred_72h":None})
        z[f"Pred_{int(r.get('Nominal_Lead_h'))}h"]=r.get("Predicted_Kp")
    kp_chart_rows=[kp_chart_map[k] for k in sorted(kp_chart_map)]
    write_table(ws,0,12,["Target_UTC","Observed_Kp","Pred_24h","Pred_48h","Pred_72h"],kp_chart_rows,fmt)
    ws.set_column("M:M",19);ws.set_column("N:Q",13)
    add_line_chart(wb,ws,"Kp observed vs 24/48/72 h lead forecasts","Kp_Fcst_History",0,len(kp_chart_rows),12,[(13,"Observed Kp"),(14,"24h forecast"),(15,"48h forecast"),(16,"72h forecast")],"S2","Kp")

    ws = wb.add_worksheet("Bz_Lead_Skill")
    bz_lead_rows=[]
    rolling_bz = {int(x.get("nominal_lead_hours")): x for x in (lead_skill.get("bz") or []) if isinstance(x, dict) and str(x.get("nominal_lead_hours", "")).isdigit()}
    for h in (24,48,72):
        all_rows=bz_lead_pairs.get(f"{h}h",[]);mr=_monthly_pair_subset(all_rows,month_start,month_end);mm=_bz_pair_metrics(mr);rm=rolling_bz.get(h,{})
        bz_lead_rows.append({"Lead_h":h,"Window":"rolling 30d","Count":rm.get("count"),"BzMin_MAE_nT":rm.get("bz_min_mae"),"BzMin_Bias_nT":rm.get("bz_min_bias"),"Hit_within_2nT_pct":rm.get("hit_rate_2nt"),"Hit_within_3nT_pct":rm.get("hit_rate_3nt"),"Southward_Direction_Hit_pct":rm.get("direction_hit_rate_threshold60"),"Brier":rm.get("brier_score")})
        bz_lead_rows.append({"Lead_h":h,"Window":month,"Count":mm.get("count"),"BzMin_MAE_nT":mm.get("mae"),"BzMin_Bias_nT":mm.get("bias"),"Hit_within_2nT_pct":mm.get("hit_rate_2nt"),"Hit_within_3nT_pct":mm.get("hit_rate_3nt"),"Southward_Direction_Hit_pct":mm.get("direction_hit_rate_threshold60"),"Brier":mm.get("brier_score")})
    write_table(ws,0,0,["Lead_h","Window","Count","BzMin_MAE_nT","BzMin_Bias_nT","Hit_within_2nT_pct","Hit_within_3nT_pct","Southward_Direction_Hit_pct","Brier"],bz_lead_rows,fmt)
    ws.set_column("A:A",11);ws.set_column("B:B",16);ws.set_column("C:I",22)

    ws = wb.add_worksheet("Bz_Fcst_History")
    bh=[]
    for h in (24,48,72):
        for r in _monthly_pair_subset(bz_lead_pairs.get(f"{h}h", []), month_start, month_end):
            it=parse_time(r.get("issued_at"));tt=parse_time(r.get("target_time"));err=num(r.get("error_bz_min"))
            bh.append({"Nominal_Lead_h":h,"Issued_UTC":it.replace(tzinfo=None) if it else None,"Target_UTC":tt.replace(tzinfo=None) if tt else None,"Actual_Lead_h":r.get("lead_hours"),"Forecast_BzMin_nT":r.get("predicted_bz_min"),"Observed_BzMin_nT":r.get("observed_bz_min"),"Error_nT":err,"Within_2nT":abs(err)<=2 if err is not None else None,"Within_3nT":abs(err)<=3 if err is not None else None,"Southward_Prob_pct":r.get("southward_probability"),"Observed_Southward":r.get("observed_southward"),"Brier":r.get("brier")})
    bh.sort(key=lambda x:(x.get("Target_UTC") or datetime.min, x.get("Nominal_Lead_h") or 0))
    write_table(ws,0,0,["Nominal_Lead_h","Issued_UTC","Target_UTC","Actual_Lead_h","Forecast_BzMin_nT","Observed_BzMin_nT","Error_nT","Within_2nT","Within_3nT","Southward_Prob_pct","Observed_Southward","Brier"],bh,fmt)
    ws.set_column("A:A",14);ws.set_column("B:C",19);ws.set_column("D:L",18)
    bz_chart_map={}
    for r in bh:
        tt=r.get("Target_UTC")
        if tt is None:continue
        z=bz_chart_map.setdefault(tt,{"Target_UTC":tt,"Observed_BzMin_nT":r.get("Observed_BzMin_nT"),"Pred_24h":None,"Pred_48h":None,"Pred_72h":None})
        z[f"Pred_{int(r.get('Nominal_Lead_h'))}h"]=r.get("Forecast_BzMin_nT")
    bz_chart_rows=[bz_chart_map[k] for k in sorted(bz_chart_map)]
    write_table(ws,0,13,["Target_UTC","Observed_BzMin_nT","Pred_24h","Pred_48h","Pred_72h"],bz_chart_rows,fmt)
    ws.set_column("N:N",19);ws.set_column("O:R",15)
    add_line_chart(wb,ws,"Bz minimum observed vs 24/48/72 h lead forecasts","Bz_Fcst_History",0,len(bz_chart_rows),13,[(14,"Observed Bz min"),(15,"24h forecast"),(16,"48h forecast"),(17,"72h forecast")],"T2","nT")

    # SWIFT-CHARGE forecast and verification
    ws = wb.add_worksheet("Surface_Charging")
    surface_rows = []
    obs_surface_by_time = {r["time"]: r for r in charging_obs if r.get("surface_kv") is not None}
    for r in charging_fc:
        surface_rows.append({
            "UTC": r["_t"].replace(tzinfo=None), "Lead_h": r.get("lead_hours"),
            "Surface_kV": r.get("surface_kv"), "Differential_kV": r.get("differential_kv"),
            "Observed_Surface_kV": (obs_surface_by_time.get(r["time"]) or {}).get("surface_kv"),
            "Kp": r.get("kp"), "BzMin_nT": r.get("bz_min_nt"), "Wind_km_s": r.get("wind_kms"), "Source": r.get("source")
        })
    write_table(ws,0,0,["UTC","Lead_h","Surface_kV","Differential_kV","Observed_Surface_kV","Kp","BzMin_nT","Wind_km_s","Source"],surface_rows,fmt)
    ws.set_column("A:A",19);ws.set_column("B:H",16);ws.set_column("I:I",48)
    add_line_chart(wb,ws,"SWIFT-CHARGE surface potential — next 72 h","Surface_Charging",0,len(surface_rows),0,[(2,"Surface kV"),(3,"Differential kV"),(4,"Observed surface kV")],"K2","kV")

    ws = wb.add_worksheet("Internal_Charging")
    internal_rows = []
    obs_internal_by_time = {r["time"]: r for r in charging_obs if r.get("internal_field_mvm") is not None}
    for r in charging_fc:
        internal_rows.append({
            "UTC": r["_t"].replace(tzinfo=None), "Lead_h": r.get("lead_hours"),
            "Internal_Field_MV_m": r.get("internal_field_mvm"),
            "Observed_Internal_MV_m": (obs_internal_by_time.get(r["time"]) or {}).get("internal_field_mvm"),
            "Electron_gt2MeV": r.get("electron_flux_gt2mev"), "Electron_Fluence24h_Proxy": r.get("electron_fluence_24h_proxy"),
            "Kp": r.get("kp"), "BzMin_nT": r.get("bz_min_nt"), "Wind_km_s": r.get("wind_kms"), "Source": r.get("source")
        })
    write_table(ws,0,0,["UTC","Lead_h","Internal_Field_MV_m","Observed_Internal_MV_m","Electron_gt2MeV","Electron_Fluence24h_Proxy","Kp","BzMin_nT","Wind_km_s","Source"],internal_rows,fmt)
    ws.set_column("A:A",19);ws.set_column("B:I",19);ws.set_column("J:J",48)
    add_line_chart(wb,ws,"SWIFT-CHARGE equivalent internal field — next 72 h","Internal_Charging",0,len(internal_rows),0,[(2,"Forecast MV/m"),(3,"Observed MV/m")],"L2","MV/m")

    ws = wb.add_worksheet("Charging_Lead_Skill")
    charging_skill_rows=[]
    for h in (24,48,72):
        s=surface_skill.get(h,{})
        i=internal_skill.get(h,{})
        charging_skill_rows.append({"Family":"SURFACE","Lead_h":h,"Count":s.get("count"),"Status":s.get("status"),"MAE":s.get("mae"),"Bias":s.get("bias"),"RMSE":s.get("rmse"),"Primary_Hit_pct":s.get("hit_rate_1kv"),"Secondary_Hit_pct":s.get("hit_rate_2kv"),"Primary_Threshold":"±1 kV","Secondary_Threshold":"±2 kV"})
        charging_skill_rows.append({"Family":"INTERNAL","Lead_h":h,"Count":i.get("count"),"Status":i.get("status"),"MAE":i.get("mae"),"Bias":i.get("bias"),"RMSE":i.get("rmse"),"Primary_Hit_pct":i.get("hit_rate_0_1mvm"),"Secondary_Hit_pct":i.get("hit_rate_0_2mvm"),"Primary_Threshold":"±0.1 MV/m","Secondary_Threshold":"±0.2 MV/m"})
    write_table(ws,0,0,["Family","Lead_h","Count","Status","MAE","Bias","RMSE","Primary_Hit_pct","Secondary_Hit_pct","Primary_Threshold","Secondary_Threshold"],charging_skill_rows,fmt)
    ws.set_column("A:A",14);ws.set_column("B:K",19)

    ws = wb.add_worksheet("Charging_Fcst_History")
    chist=[]
    for fam,key,unit in (("surface","surface_kv","kV"),("internal","internal_field_mvm","MV/m")):
        for r in _charging_month_pairs(charging_verification,fam,month_start,month_end):
            it=parse_time(r.get("issued_at"));tt=parse_time(r.get("target_time"));err=num(r.get("error"))
            primary=(abs(err)<=1.0 if fam=="surface" else abs(err)<=0.1) if err is not None else None
            secondary=(abs(err)<=2.0 if fam=="surface" else abs(err)<=0.2) if err is not None else None
            chist.append({"Family":fam.upper(),"Nominal_Lead_h":r.get("nominal_lead_hours"),"Issued_UTC":it.replace(tzinfo=None) if it else None,"Target_UTC":tt.replace(tzinfo=None) if tt else None,"Actual_Lead_h":r.get("lead_hours"),"Predicted":r.get("predicted"),"Observed":r.get("observed"),"Error":err,"Unit":unit,"Primary_Hit":primary,"Secondary_Hit":secondary,"Model":r.get("model")})
    write_table(ws,0,0,["Family","Nominal_Lead_h","Issued_UTC","Target_UTC","Actual_Lead_h","Predicted","Observed","Error","Unit","Primary_Hit","Secondary_Hit","Model"],chist,fmt)
    ws.set_column("A:B",15);ws.set_column("C:D",19);ws.set_column("E:K",15);ws.set_column("L:L",38)

    ws = wb.add_worksheet("Charging_Model")
    coeff=(charging.get("coefficients") or {}) if isinstance(charging,dict) else {}
    tr=(coeff.get("training") or {}) if isinstance(coeff,dict) else {}
    model_rows=[
        {"Item":"Model","Value":charging.get("model"),"Note":"Current SWIFT-CHARGE generation"},
        {"Item":"Reference scope","Value":((charging.get("reference_spacecraft") or {}).get("scope")),"Note":"Not direct GOES hardware telemetry"},
        {"Item":"Surface coefficients","Value":json.dumps(coeff.get("surface") or {},ensure_ascii=False),"Note":tr.get("surface_status")},
        {"Item":"Internal coefficients","Value":json.dumps(coeff.get("internal") or {},ensure_ascii=False),"Note":tr.get("internal_status")},
        {"Item":"Differential ratio","Value":coeff.get("differential_ratio"),"Note":"Reference material differential-potential proxy"},
        {"Item":"Surface training N","Value":tr.get("surface_samples"),"Note":"Validated target count used for coefficient update"},
        {"Item":"Internal training N","Value":tr.get("internal_samples"),"Note":"Validated target count used for coefficient update"},
        {"Item":"Verification rule","Value":charging_verification.get("observation_rule"),"Note":"Model-generated values never self-score"},
    ]
    write_table(ws,0,0,["Item","Value","Note"],model_rows,fmt);ws.set_column("A:A",24);ws.set_column("B:B",80);ws.set_column("C:C",60)

    ws = wb.add_worksheet("CME_Verification")
    cv=[]
    for r in pairs.get("cme",[]):
        st=parse_time(r.get("start_time"));pa=parse_time(r.get("predicted_arrival"));oa=parse_time(r.get("observed_arrival_proxy"))
        cv.append({"Event_ID":r.get("event_id"),"Start_UTC":st.replace(tzinfo=None) if st else None,"Predicted_Arrival_UTC":pa.replace(tzinfo=None) if pa else None,"Observed_Shock_Proxy_UTC":oa.replace(tzinfo=None) if oa else None,"ETA_Error_h":r.get("arrival_error_hours"),"Abs_ETA_Error_h":r.get("abs_arrival_error_hours"),"Shock_Proxy_Score":r.get("shock_proxy_score"),"Initial_Speed_km_s":r.get("initial_speed_km_s"),"Baseline_Wind_km_s":r.get("baseline_speed_km_s"),"Impact_Class":r.get("impact_class"),"Method":r.get("verification_method")})
    write_table(ws,0,0,["Event_ID","Start_UTC","Predicted_Arrival_UTC","Observed_Shock_Proxy_UTC","ETA_Error_h","Abs_ETA_Error_h","Shock_Proxy_Score","Initial_Speed_km_s","Baseline_Wind_km_s","Impact_Class","Method"],cv,fmt)
    ws.set_column("A:A",32);ws.set_column("B:D",20);ws.set_column("E:J",17);ws.set_column("K:K",65)

    ws = wb.add_worksheet("Uncertainty")
    ur=[]
    for family,vals in (validation.get("uncertainty") or {}).items():
        if not isinstance(vals,list):continue
        for x in vals:
            if isinstance(x,dict):ur.append({"Variable":family,**x})
    write_table(ws,0,0,["Variable","lead_day","count","status","p10","p50","p90"],ur,fmt)
    ws.set_column("A:A",26);ws.set_column("B:G",14)

    ws = wb.add_worksheet("Kp_Observed")
    ko=[]
    for r in kp_hist:
        t=r.get("_t")
        if isinstance(t,datetime) and month_start<=t<month_end:
            ko.append({"UTC":t.replace(tzinfo=None),"Kp":r.get("kp")})
    write_table(ws,0,0,["UTC","Kp"],ko,fmt);ws.set_column("A:A",19);ws.set_column("B:B",12)

    ws = wb.add_worksheet("ENLIL_Earth")
    er=[]
    for r in enlil_latest.get("records",[]):
        t=parse_time(r.get("time"))
        if t and month_start-timedelta(days=2)<=t<month_end+timedelta(days=4):
            er.append({"UTC":t.replace(tzinfo=None),"Vr_km_s":r.get("v_r"),"Density_cm3":r.get("earth_particles_per_cm3"),"Cloud":r.get("cloud")})
    write_table(ws,0,0,["UTC","Vr_km_s","Density_cm3","Cloud"],er,fmt);ws.set_column("A:A",19);ws.set_column("B:D",18)

    ws = wb.add_worksheet("Methods")
    method_rows=[
        {"Item":"Wind verification","Definition":"Issued SWIFT forecast vs NOAA RTSW; error = forecast - observed; ±50 km/s hit is primary short-lead metric."},
        {"Item":"Kp verification","Definition":"Issued 3-hour SWIFT Kp vs NOAA planetary K; primary hit = ±1.0 Kp, strict hit = ±0.67 Kp. 24/48/72 h skill selects one archived forecast per target nearest the nominal lead within ±4.5 h."},
        {"Item":"Kp range verification","Definition":"Observed-Kp strata are non-overlapping: 1<=Kp<5, 5<=Kp<6, 6<=Kp<8, 8<=Kp<=9. Kp<1 is excluded from this requested stratification."},
        {"Item":"Bz verification","Definition":"Forecast 3-hour Bz minimum vs observed 3-hour minimum; primary numeric hit = ±2 nT, secondary = ±3 nT; southward probability is also scored with 60% direction threshold and Brier score. 24/48/72 h skill uses ±4.5 h lead matching."},
        {"Item":"Model generation separation","Definition":"Primary Kp/Bz verification and calibration use only archive rows whose model field exactly matches the current latest.json model. Legacy and unversioned forecasts are preserved for historical comparison only."},
        {"Item":"24/48/72 h auto-calibration","Definition":"Current-generation residual bias is estimated separately near nominal 24/48/72 h leads over the previous 30 days. Correction strength ramps with sample count and is linearly interpolated by forecast lead; short-lead correction tends to zero."},
        {"Item":"Surface charging forecast","Definition":"SWIFT-CHARGE reference GEO spacecraft potential [kV] from Kp, southward Bz and solar-wind enhancement. Coefficients can be ridge-updated only when validated charging targets exist. It is not direct GOES bus-potential telemetry."},
        {"Item":"Internal charging forecast","Definition":"Equivalent reference-dielectric internal field [MV/m] from >2 MeV electron environment proxy, 24 h fluence and geomagnetic drivers. It is not a measured field inside GOES hardware."},
        {"Item":"Charging verification","Definition":"24/48/72 h surface primary hit = ±1 kV (secondary ±2 kV); internal primary hit = ±0.1 MV/m (secondary ±0.2 MV/m). Only independent observed/validated rows in docs/data/swift-charging/observed.json count as truth."},
        {"Item":"CME arrival","Definition":"NOAA speed/density shock proxy near predicted arrival. Keep this distinct from a manually adjudicated ICME boundary in publications."},
        {"Item":"Uncertainty","Definition":"Empirical p10/p50/p90 forecast-minus-observation residuals from previous 30 days, split by lead day."},
        {"Item":"DBM gamma","Definition":str(((validation.get("models") or {}).get("cme_arrival") or {}).get("dbm_gamma_fit"))},
        {"Item":"WSA-ENLIL","Definition":"Official NOAA SWPC L1 time series plus NCEP WSA velocity boundary at 21.5 R_sun. Full 3-D ENLIL grid is not stored in this workbook."},
        {"Item":"Retention","Definition":"Exactly one workbook per UTC month; current month overwritten each run; only latest 24 monthly workbooks retained."},
    ]
    write_table(ws,0,0,["Item","Definition"],method_rows,fmt);ws.set_column("A:A",24);ws.set_column("B:B",110)

    ws = wb.add_worksheet("Sources")
    source_rows = [
        {"Data": "NOAA RTSW solar wind", "Path_or_URL": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"},
        {"Data": "NOAA RTSW IMF", "Path_or_URL": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"},
        {"Data": "NOAA Planetary K-index", "Path_or_URL": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"},
        {"Data": "NOAA WSA-ENLIL Earth/L1 time series", "Path_or_URL": "https://services.swpc.noaa.gov/json/enlil_time_series.json"},
        {"Data": "NCEP WSA-Enlil operational model files", "Path_or_URL": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/wsa_enlil/prod/"},
        {"Data": "SWIFT Wind AI", "Path_or_URL": "docs/data/swift-wind/latest.json"},
        {"Data": "SWIFT Wind verification", "Path_or_URL": "docs/data/swift-wind/verification.json"},
        {"Data": "SWIFT Kp AI", "Path_or_URL": "docs/data/swift-kp/latest.json"},
        {"Data": "SWIFT Bz AI", "Path_or_URL": "docs/data/swift-bz/latest.json"},
        {"Data": "SWIFT Kp/Bz lead verification", "Path_or_URL": "docs/data/validation/latest.json"},
        {"Data": "SWIFT-CHARGE forecast", "Path_or_URL": "docs/data/swift-charging/latest.json"},
        {"Data": "SWIFT-CHARGE verification", "Path_or_URL": "docs/data/swift-charging/verification.json"},
        {"Data": "SWIFT-CHARGE forecast archive", "Path_or_URL": "docs/data/swift-charging/forecast-archive.json"},
        {"Data": "SWIFT-CHARGE independent targets", "Path_or_URL": "docs/data/swift-charging/observed.json"},
        {"Data": "SWIFT Kp forecast archive", "Path_or_URL": "docs/data/swift-kp/forecast-archive.json"},
        {"Data": "SWIFT Bz forecast archive", "Path_or_URL": "docs/data/swift-bz/forecast-archive.json"},
        {"Data": "CME arrivals", "Path_or_URL": "docs/data/cme-arrivals/latest.json"},
    ]
    write_table(ws, 0, 0, ["Data", "Path_or_URL"], source_rows, fmt); ws.set_column("A:A", 30); ws.set_column("B:B", 90)

    wb.close()
    tmp_path.replace(final_path)
    prune_monthly_reports(now)

    monthly_files = sorted(OUT_DOCS.glob("swift_space_weather_????-??.xlsx"), reverse=True)
    index_payload = {
        "updated_at": iso_z(now),
        "current_month": month,
        "current_month_file": monthly_name,
        "current_month_url": f"./reports/{monthly_name}",
        "monthly_latest_url": f"./reports/{monthly_name}",
        "ui_feed_url": "./reports/ui_forecast_latest.json",
        "monthly_files": [{"month": p.stem[-7:], "file": p.name, "url": f"./reports/{p.name}"} for p in monthly_files],
        "primary_wind_accuracy": {"definition": "|predicted - observed| <= 50 km/s", "window": "last_24h", **current_acc},
        "retention_months": REPORT_RETENTION_MONTHS,
        "validation_feed_url": "./data/validation/ui_validation_latest.json",
        "sheets": ["Monthly_Summary", "SolarWind_Month", "Wind_Forecast_3d", "Wind_Fcst_History", "Kp_Forecast", "Accuracy", "Bz_AI", "CME_Arrivals", "CME_Wind_Boost", "Coronal_Holes", "Research_Metrics", "Wind_Verification", "ENLIL_Verification", "Kp_Verification", "Bz_Verification", "Model_Generations", "Auto_Calibration", "Forecast_Skill_History", "Kp_Lead_Skill", "Kp_Range_Skill", "Kp_Fcst_History", "Bz_Lead_Skill", "Bz_Fcst_History", "Surface_Charging", "Internal_Charging", "Charging_Lead_Skill", "Charging_Fcst_History", "Charging_Model", "CME_Verification", "Uncertainty", "Kp_Observed", "ENLIL_Earth", "Methods", "Sources"],
    }
    (OUT_DOCS / "index.json").write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {final_path}")
    print(f"Current ±50 km/s hit rate (24h): {current_acc.get('hit_rate_50kms')}% / n={current_acc.get('count')}")


if __name__ == "__main__":
    main()
