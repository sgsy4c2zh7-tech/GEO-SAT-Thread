#!/usr/bin/env python3
"""Build one SWIFT Excel workbook per UTC month for GitHub Pages.

Output example:
- docs/reports/swift_space_weather_2026-09.xlsx
- docs/reports/index.json

The same YYYY-MM workbook is overwritten on each run during the month.
Past monthly workbooks are retained, so there is exactly one Excel file per month.

Workbook contents:
- Monthly_Summary: month KPI summary and daily summary table
- SolarWind_Month: hourly NOAA solar wind/IMF for the month
- Wind_Forecast_3d: next 3-day SWIFT Wind forecast
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
            if not t or t < now - timedelta(hours=3) or t > now + timedelta(days=5):
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
    swift_kp, swift_kp_hist = load_swift_kp()
    cmes = load_cme_arrivals()
    cme_model = load_cme_boost_model()
    bz_ai = load_bz_ai()

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

    ws = wb.add_worksheet("Kp_Forecast")
    kpf_rows = prepare_rows_for_excel(kp_fc, [("UTC", "_t"), ("Kp", "kp"), ("G_Scale", "g_scale"), ("Confidence", "confidence"), ("Source", "source")])
    write_table(ws, 0, 0, ["UTC", "Kp", "G_Scale", "Confidence", "Source"], kpf_rows, fmt)
    ws.set_column("A:A", 19); ws.set_column("B:D", 14); ws.set_column("E:E", 28)
    add_line_chart(wb, ws, "Kp forecast", "Kp_Forecast", 0, len(kpf_rows), 0, [(1, "Kp")], "G2", "Kp")

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

    ws = wb.add_worksheet("Sources")
    source_rows = [
        {"Data": "NOAA RTSW solar wind", "Path_or_URL": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"},
        {"Data": "NOAA RTSW IMF", "Path_or_URL": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"},
        {"Data": "NOAA Planetary K-index", "Path_or_URL": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"},
        {"Data": "SWIFT Wind AI", "Path_or_URL": "docs/data/swift-wind/latest.json"},
        {"Data": "SWIFT Wind verification", "Path_or_URL": "docs/data/swift-wind/verification.json"},
        {"Data": "SWIFT Kp AI", "Path_or_URL": "docs/data/swift-kp/latest.json"},
        {"Data": "SWIFT Bz AI", "Path_or_URL": "docs/data/swift-bz/latest.json"},
        {"Data": "CME arrivals", "Path_or_URL": "docs/data/cme-arrivals/latest.json"},
    ]
    write_table(ws, 0, 0, ["Data", "Path_or_URL"], source_rows, fmt); ws.set_column("A:A", 30); ws.set_column("B:B", 90)

    wb.close()
    tmp_path.replace(final_path)

    monthly_files = sorted(OUT_DOCS.glob("swift_space_weather_????-??.xlsx"), reverse=True)
    index_payload = {
        "updated_at": iso_z(now),
        "current_month": month,
        "current_month_file": monthly_name,
        "current_month_url": f"./reports/{monthly_name}",
        "monthly_latest_url": f"./reports/{monthly_name}",
        "monthly_files": [{"month": p.stem[-7:], "file": p.name, "url": f"./reports/{p.name}"} for p in monthly_files],
        "primary_wind_accuracy": {"definition": "|predicted - observed| <= 50 km/s", "window": "last_24h", **current_acc},
        "sheets": ["Monthly_Summary", "SolarWind_Month", "Wind_Forecast_3d", "Kp_Forecast", "Accuracy", "Bz_AI", "CME_Arrivals", "CME_Wind_Boost", "Sources"],
    }
    (OUT_DOCS / "index.json").write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {final_path}")
    print(f"Current ±50 km/s hit rate (24h): {current_acc.get('hit_rate_50kms')}% / n={current_acc.get('count')}")


if __name__ == "__main__":
    main()
