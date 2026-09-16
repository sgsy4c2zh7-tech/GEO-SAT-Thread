#!/usr/bin/env python3
"""Build SWIFT space-weather Excel report for GitHub Pages.

Outputs:
- reports/swift_space_weather_report_latest.xlsx
- docs/reports/swift_space_weather_report_latest.xlsx
- docs/reports/index.json
- reports/index.json

Report contents:
- Dashboard summary
- 27-day observed solar wind graph
- 3-day solar wind forecast graph
- Kp forecast graph
- Wind and Kp accuracy, separated
- Bz forecast AI graph
- CME arrival table with speed and predicted arrival time
- CME wind boost learning samples/model
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
OUT_ROOT = ROOT / "reports"
OUT_DOCS = ROOT / "docs" / "reports"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_DOCS.mkdir(parents=True, exist_ok=True)

LATEST_NAME = "swift_space_weather_report_latest.xlsx"
STAMP_NAME = "swift_space_weather_report_{stamp}.xlsx"

BIN_HOURS = 1
FORECAST_DAYS = 3
HISTORY_DAYS = 27


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
    pairs = []
    if swift_wind:
        for r in records(swift_wind):
            t = parse_time(pick(r, ["time", "start_time", "timestamp"]))
            obs = num(pick(r, ["observed_speed", "actual_speed"]))
            pred = num(pick(r, ["swift_cme_enhanced_speed", "predicted_speed", "forecast_speed"]))
            bg = num(pick(r, ["swift_background_speed", "background_speed", "wsa_speed"]))
            if t and obs is not None and pred is not None:
                pairs.append({"time": iso_z(t), "_t": t, "observed": obs, "predicted": pred, "background": bg, "error": pred - obs})
    # Deduplicate and keep newest.
    pairs = dedupe_time(pairs)
    if not pairs:
        return []
    return pairs


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


def main() -> None:
    now = utcnow()
    stamp = now.strftime("%Y%m%d_%H%M")
    tmp_path = OUT_ROOT / LATEST_NAME
    docs_path = OUT_DOCS / LATEST_NAME
    stamped_path = OUT_DOCS / STAMP_NAME.format(stamp=stamp)

    wind_hist = load_wind_history()
    mag_hist = load_mag_history()
    kp_hist = load_kp_history()
    swift_wind = load_swift_wind()
    swift_kp, swift_kp_hist = load_swift_kp()
    cmes = load_cme_arrivals()
    cme_model = load_cme_boost_model()
    bz_ai = load_bz_ai()

    wind_hourly = bin_hourly(wind_hist, ["speed", "density"], HISTORY_DAYS)
    mag_hourly = bin_hourly(mag_hist, ["bz", "bt"], HISTORY_DAYS)
    # Merge Bz into solar wind hourly table.
    mag_by_time = {r["time"]: r for r in mag_hourly}
    sw27 = []
    for r in wind_hourly:
        m = mag_by_time.get(r["time"], {})
        x = dict(r)
        x["bz"] = m.get("bz")
        x["bt"] = m.get("bt")
        sw27.append(x)

    wind_fc = wind_forecast_3d(swift_wind, wind_hist, cmes)
    kp_fc = kp_forecast_rows(swift_kp, wind_fc, bz_ai)
    bz_fc = []
    for r in records(bz_ai, keys=("forecast",)) if bz_ai else []:
        t = parse_time(r.get("time"))
        if t and t <= now + timedelta(days=FORECAST_DAYS):
            bz_fc.append({
                "_t": t,
                "time": iso_z(t),
                "bz_forecast": num(r.get("bz_forecast")),
                "bz_min_forecast": num(r.get("bz_min_forecast")),
                "bt_forecast": num(r.get("bt_forecast")),
                "southward_bz_probability": num(r.get("southward_bz_probability")),
                "bz_risk": r.get("bz_risk"),
                "cme_id": r.get("cme_id"),
            })
    bz_fc = sorted(bz_fc, key=lambda x: x["_t"])

    wind_acc_rows = wind_accuracy(swift_wind, wind_hist)
    kp_acc_rows = kp_accuracy(swift_kp_hist)
    wind_metrics = [metric_summary(wind_acc_rows, d, "wind") for d in (1, 7, 27)]
    kp_metrics = [metric_summary(kp_acc_rows, d, "kp") for d in (1, 7, 27)]

    wb = xlsxwriter.Workbook(str(tmp_path), {"nan_inf_to_errors": True})
    fmt = {
        "title": wb.add_format({"bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "align": "center", "valign": "vcenter"}),
        "section": wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#305496"}),
        "header": wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1, "align": "center", "valign": "vcenter"}),
        "body": wb.add_format({"border": 1, "valign": "vcenter"}),
        "datetime": wb.add_format({"num_format": "yyyy-mm-dd hh:mm", "border": 1}),
        "num1": wb.add_format({"num_format": "0.0", "border": 1}),
        "num2": wb.add_format({"num_format": "0.00", "border": 1}),
        "pct": wb.add_format({"num_format": "0.0%", "border": 1}),
        "url": wb.add_format({"font_color": "blue", "underline": True}),
        "good": wb.add_format({"bg_color": "#E2F0D9", "border": 1}),
        "warn": wb.add_format({"bg_color": "#FFF2CC", "border": 1}),
        "bad": wb.add_format({"bg_color": "#F8CBAD", "border": 1}),
    }

    # Dashboard
    ws = wb.add_worksheet("Dashboard")
    ws.merge_range("A1:H1", "SWIFT Space Weather Excel Report", fmt["title"])
    ws.write("A3", "Updated UTC", fmt["section"]); ws.write("B3", iso_z(now))
    ws.write("A4", "Main URL root", fmt["section"]); ws.write("B4", "./reports/swift_space_weather_report_latest.xlsx", fmt["url"])
    ws.write("A5", "Main URL docs", fmt["section"]); ws.write("B5", "./docs/reports/swift_space_weather_report_latest.xlsx", fmt["url"])
    summary_rows = [
        {"Item": "27-day solar wind rows", "Value": len(sw27)},
        {"Item": "NOAA wind raw records", "Value": len(wind_hist)},
        {"Item": "NOAA IMF raw records", "Value": len(mag_hist)},
        {"Item": "NOAA Kp raw records", "Value": len(kp_hist)},
        {"Item": "3-day wind forecast rows", "Value": len(wind_fc)},
        {"Item": "Kp forecast rows", "Value": len(kp_fc)},
        {"Item": "Bz AI forecast rows", "Value": len(bz_fc)},
        {"Item": "CME arrivals", "Value": len(cmes)},
        {"Item": "CME boost learning samples", "Value": cme_model.get("sample_count") if cme_model else 0},
    ]
    write_table(ws, 7, 0, ["Item", "Value"], summary_rows, fmt)
    # KPI metrics
    metrics = []
    for m in wind_metrics:
        metrics.append({
            "Metric": f"Wind {m['period']}", "Count": m.get("count", 0), "MAE": m.get("mae"), "Bias": m.get("bias"), "Hit1": m.get("hit_rate_within_50kms"), "Hit2": m.get("hit_rate_within_100kms"), "Unit": "km/s",
        })
    for m in kp_metrics:
        metrics.append({
            "Metric": f"Kp {m['period']}", "Count": m.get("count", 0), "MAE": m.get("mae"), "Bias": m.get("bias"), "Hit1": m.get("hit_rate_within_067kp"), "Hit2": m.get("hit_rate_within_1kp"), "Unit": "Kp",
        })
    write_table(ws, 7, 3, ["Metric", "Count", "MAE", "Bias", "Hit1", "Hit2", "Unit"], metrics, fmt)
    ws.write("D6", "Hit1 = Wind ±50 km/s / Kp ±0.67. Hit2 = Wind ±100 km/s / Kp ±1.0", fmt["section"])
    ws.set_column("A:A", 30); ws.set_column("B:B", 18); ws.set_column("D:D", 22); ws.set_column("E:H", 12); ws.set_column("I:J", 12)

    # Solar wind 27d
    ws = wb.add_worksheet("SolarWind_27d")
    sw_rows = prepare_rows_for_excel(sw27, [("UTC", "_t"), ("Speed_km_s", "speed"), ("Density_cm3", "density"), ("Bz_nT", "bz"), ("Bt_nT", "bt")])
    write_table(ws, 0, 0, ["UTC", "Speed_km_s", "Density_cm3", "Bz_nT", "Bt_nT"], sw_rows, fmt)
    ws.set_column("A:A", 19); ws.set_column("B:E", 14)
    last = len(sw_rows)
    add_line_chart(wb, ws, "Past 27 days solar wind speed", "SolarWind_27d", 0, last, 0, [(1, "Speed km/s")], "G2", "km/s")
    add_line_chart(wb, ws, "Past 27 days IMF Bz/Bt", "SolarWind_27d", 0, last, 0, [(3, "Bz nT"), (4, "Bt nT")], "G20", "nT")

    # Wind forecast 3d
    ws = wb.add_worksheet("Wind_Forecast_3d")
    wf_rows = prepare_rows_for_excel(wind_fc, [("UTC", "_t"), ("Predicted_Speed_km_s", "predicted_speed"), ("Background_Speed_km_s", "background_speed"), ("Observed_Speed_km_s", "observed_speed"), ("CME_Boost_km_s", "cme_boost"), ("CME_ID", "cme_id"), ("Source", "source")])
    write_table(ws, 0, 0, ["UTC", "Predicted_Speed_km_s", "Background_Speed_km_s", "Observed_Speed_km_s", "CME_Boost_km_s", "CME_ID", "Source"], wf_rows, fmt)
    ws.set_column("A:A", 19); ws.set_column("B:E", 18); ws.set_column("F:G", 28)
    add_line_chart(wb, ws, "Next 3 days solar wind forecast", "Wind_Forecast_3d", 0, len(wf_rows), 0, [(1, "Predicted"), (2, "Background"), (4, "CME boost")], "I2", "km/s")

    # Kp forecast
    ws = wb.add_worksheet("Kp_Forecast")
    kpf_rows = prepare_rows_for_excel(kp_fc, [("UTC", "_t"), ("Kp", "kp"), ("G_Scale", "g_scale"), ("Confidence", "confidence"), ("Source", "source")])
    write_table(ws, 0, 0, ["UTC", "Kp", "G_Scale", "Confidence", "Source"], kpf_rows, fmt)
    ws.set_column("A:A", 19); ws.set_column("B:D", 14); ws.set_column("E:E", 28)
    add_line_chart(wb, ws, "Kp forecast", "Kp_Forecast", 0, len(kpf_rows), 0, [(1, "Kp")], "G2", "Kp")

    # Accuracy separated
    ws = wb.add_worksheet("Accuracy")
    ws.merge_range("A1:G1", "Wind accuracy", fmt["title"])
    write_table(ws, 2, 0, ["kind", "period", "count", "mae", "rmse", "bias", "hit_rate_within_50kms", "hit_rate_within_100kms"], wind_metrics, fmt)
    ws.merge_range("A8:H8", "Kp accuracy", fmt["title"])
    write_table(ws, 9, 0, ["kind", "period", "count", "mae", "rmse", "bias", "hit_rate_within_067kp", "hit_rate_within_1kp", "g_scale_hit_rate"], kp_metrics, fmt)
    ws.set_column("A:B", 16); ws.set_column("C:I", 16)
    # Detail rows start lower to avoid making dashboard too heavy.
    wr = prepare_rows_for_excel(wind_acc_rows[-500:], [("UTC", "_t"), ("Observed", "observed"), ("Predicted", "predicted"), ("Background", "background"), ("Error", "error")])
    kr = prepare_rows_for_excel(kp_acc_rows[-500:], [("UTC", "_t"), ("Observed", "observed"), ("Predicted", "predicted"), ("Predicted_G", "predicted_g"), ("Observed_G", "observed_g"), ("Error", "error")])
    start = 16
    ws.write(start, 0, "Wind detail latest 500", fmt["section"])
    end = write_table(ws, start + 1, 0, ["UTC", "Observed", "Predicted", "Background", "Error"], wr, fmt)
    ws.write(start, 7, "Kp detail latest 500", fmt["section"])
    write_table(ws, start + 1, 7, ["UTC", "Observed", "Predicted", "Predicted_G", "Observed_G", "Error"], kr, fmt)

    # Bz AI
    ws = wb.add_worksheet("Bz_AI")
    bz_rows = prepare_rows_for_excel(bz_fc, [("UTC", "_t"), ("Bz_Forecast_nT", "bz_forecast"), ("Bz_Min_Forecast_nT", "bz_min_forecast"), ("Bt_Forecast_nT", "bt_forecast"), ("Southward_Probability_pct", "southward_bz_probability"), ("Risk", "bz_risk"), ("CME_ID", "cme_id")])
    write_table(ws, 0, 0, ["UTC", "Bz_Forecast_nT", "Bz_Min_Forecast_nT", "Bt_Forecast_nT", "Southward_Probability_pct", "Risk", "CME_ID"], bz_rows, fmt)
    ws.set_column("A:A", 19); ws.set_column("B:E", 22); ws.set_column("F:G", 24)
    add_line_chart(wb, ws, "Bz AI forecast", "Bz_AI", 0, len(bz_rows), 0, [(1, "Bz forecast"), (2, "Bz min"), (3, "Bt")], "I2", "nT")
    add_line_chart(wb, ws, "Southward Bz probability", "Bz_AI", 0, len(bz_rows), 0, [(4, "Probability %")], "I20", "%")

    # CME arrivals
    ws = wb.add_worksheet("CME_Arrivals")
    cme_rows = []
    now_minus = now - timedelta(days=7)
    now_plus = now + timedelta(days=5)
    for c in cmes:
        at = c.get("_arrival")
        if not isinstance(at, datetime) or not (now_minus <= at <= now_plus):
            continue
        t0 = c.get("_t0")
        cme_rows.append({
            "ID": c.get("id"),
            "Source": c.get("source"),
            "Impact_Class": c.get("impact_class"),
            "Earth_Candidate": c.get("earth_candidate"),
            "Start_UTC": t0.replace(tzinfo=None) if isinstance(t0, datetime) else None,
            "Arrival_UTC": at.replace(tzinfo=None),
            "Transit_h": c.get("transit_hours"),
            "Speed_km_s": c.get("speed"),
            "Effective_Speed_km_s": c.get("effective_speed"),
            "Expected_Wind_Increase_km_s": c.get("expected_wind_speed_increase"),
            "Longitude": c.get("longitude"),
            "Latitude": c.get("latitude"),
            "Angular_Separation_deg": c.get("angular_separation"),
            "Confidence": c.get("confidence"),
        })
    write_table(ws, 0, 0, ["ID", "Source", "Impact_Class", "Earth_Candidate", "Start_UTC", "Arrival_UTC", "Transit_h", "Speed_km_s", "Effective_Speed_km_s", "Expected_Wind_Increase_km_s", "Longitude", "Latitude", "Angular_Separation_deg", "Confidence"], cme_rows, fmt)
    ws.set_column("A:A", 35); ws.set_column("B:D", 16); ws.set_column("E:F", 19); ws.set_column("G:N", 16)

    # CME boost model/samples
    ws = wb.add_worksheet("CME_Wind_Boost")
    model_rows = []
    if cme_model:
        for cls, item in (cme_model.get("by_impact_class") or {}).items():
            model_rows.append({"Impact_Class": cls, "Count": item.get("count"), "Median_Delta_V": item.get("median_delta_v"), "Mean_Delta_V": item.get("mean_delta_v"), "Default_Used": item.get("default_used")})
    write_table(ws, 0, 0, ["Impact_Class", "Count", "Median_Delta_V", "Mean_Delta_V", "Default_Used"], model_rows, fmt)
    samples = cme_model.get("samples", []) if cme_model else []
    sample_rows = []
    for s in samples[-300:]:
        sample_rows.append({
            "ID": s.get("id"), "Impact_Class": s.get("impact_class"),
            "Arrival_UTC": parse_time(s.get("arrival_time")).replace(tzinfo=None) if parse_time(s.get("arrival_time")) else None,
            "Speed_km_s": s.get("speed"), "Baseline_Speed": s.get("baseline_speed"), "Post_Peak_Speed": s.get("post_peak_speed"), "Observed_Delta_V": s.get("observed_delta_v"),
        })
    ws.write(9, 0, "Learning samples", fmt["section"])
    write_table(ws, 10, 0, ["ID", "Impact_Class", "Arrival_UTC", "Speed_km_s", "Baseline_Speed", "Post_Peak_Speed", "Observed_Delta_V"], sample_rows, fmt)
    ws.set_column("A:A", 35); ws.set_column("B:G", 18)

    # Sources
    ws = wb.add_worksheet("Sources")
    source_rows = [
        {"Data": "NOAA RTSW solar wind", "Path_or_URL": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"},
        {"Data": "NOAA RTSW IMF", "Path_or_URL": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"},
        {"Data": "NOAA Planetary K-index", "Path_or_URL": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"},
        {"Data": "Local NOAA wind history", "Path_or_URL": "docs/data/noaa/wind_history.json"},
        {"Data": "Local NOAA IMF history", "Path_or_URL": "docs/data/noaa/mag_history.json"},
        {"Data": "Local NOAA Kp history", "Path_or_URL": "docs/data/noaa/kp_history.json"},
        {"Data": "SWIFT Wind AI", "Path_or_URL": "docs/data/swift-wind/latest.json"},
        {"Data": "SWIFT Kp AI", "Path_or_URL": "docs/data/swift-kp/latest.json"},
        {"Data": "SWIFT Bz AI", "Path_or_URL": "docs/data/swift-bz/latest.json"},
        {"Data": "CME arrivals", "Path_or_URL": "docs/data/cme-arrivals/latest.json"},
        {"Data": "CME wind boost model", "Path_or_URL": "docs/data/swift-wind/cme-boost-model.json"},
    ]
    write_table(ws, 0, 0, ["Data", "Path_or_URL"], source_rows, fmt)
    ws.set_column("A:A", 28); ws.set_column("B:B", 90)

    wb.close()

    shutil.copy2(tmp_path, docs_path)
    shutil.copy2(tmp_path, stamped_path)

    index_payload = {
        "updated_at": iso_z(now),
        "latest": LATEST_NAME,
        "latest_root_url": f"./reports/{LATEST_NAME}",
        "latest_docs_url": f"./docs/reports/{LATEST_NAME}",
        "timestamped_docs_url": f"./docs/reports/{stamped_path.name}",
        "sheets": ["Dashboard", "SolarWind_27d", "Wind_Forecast_3d", "Kp_Forecast", "Accuracy", "Bz_AI", "CME_Arrivals", "CME_Wind_Boost", "Sources"],
        "counts": {
            "solar_wind_27d_rows": len(sw27),
            "wind_forecast_rows": len(wind_fc),
            "kp_forecast_rows": len(kp_fc),
            "wind_accuracy_pairs": len(wind_acc_rows),
            "kp_accuracy_pairs": len(kp_acc_rows),
            "bz_forecast_rows": len(bz_fc),
            "cme_arrivals": len(cmes),
        },
    }
    (OUT_DOCS / "index.json").write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "index.json").write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {tmp_path}")
    print(f"Copied {docs_path}")


if __name__ == "__main__":
    main()
