#!/usr/bin/env python3
"""
Build monthly satellite-anomaly × space-weather Excel reports and retain 12 months.

Inputs
------
docs/data/satellite-risk/event-ledger.json
docs/data/satellite-risk/environment-history/YYYY-MM.json
docs/data/satellite-risk/ncei-legacy.json (optional reference)

Outputs
-------
docs/data/satellite-risk/monthly/YYYY-MM_Satellite_Anomaly_SpaceWeather.xlsx
docs/data/satellite-risk/monthly-index.json

The workbook explicitly separates public anomaly/outage reports from space-weather
exposure. Time coincidence is not treated as causal attribution.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs" / "data" / "satellite-risk"
MONTHLY = BASE / "monthly"
ENV_DIR = BASE / "environment-history"
LEDGER = BASE / "event-ledger.json"
NCEI = BASE / "ncei-legacy.json"
INDEX = BASE / "monthly-index.json"

CAUSE_COLORS = {
    "SURFACE_CHARGING_ESD": "#FF8C42",
    "INTERNAL_CHARGING": "#9B5DE5",
    "SEE_SEU": "#FF4FA3",
    "POWER_EPS": "#FFD34E",
    "ATTITUDE_GNC": "#36D7FF",
    "COMMUNICATIONS": "#4C8CFF",
    "COMMUNICATIONS_RFI": "#4C8CFF",
    "INSTRUMENT": "#FFB14E",
    "PRODUCT_DATA": "#E6A84C",
    "MISSION_CONTROL_SOFTWARE": "#A4B0BD",
    "UNKNOWN": "#FF5F63",
}


def load_json(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def parse_dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        s = str(v).replace("Z", "+00:00")
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def fnum(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def month_keys(now: datetime, n: int = 12) -> list[str]:
    out = []
    y, m = now.year, now.month
    for i in range(n):
        yy, mm = y, m - i
        while mm <= 0:
            yy -= 1
            mm += 12
        out.append(f"{yy:04d}-{mm:02d}")
    return out


def nearest_env(env_rows: list[dict[str, Any]], event_time: datetime, max_hours: float = 3.0) -> Optional[dict[str, Any]]:
    best = None
    bd = float("inf")
    for r in env_rows:
        t = parse_dt(r.get("time"))
        if not t:
            continue
        d = abs((t - event_time).total_seconds())
        if d < bd:
            bd, best = d, r
    return best if best is not None and bd <= max_hours * 3600 else None


def flatten_event(event: dict[str, Any], env_rows: list[dict[str, Any]]) -> list[Any]:
    t = parse_dt(event.get("issued_at"))
    sw = event.get("space_weather_windows", {}) if isinstance(event, dict) else {}
    at = sw.get("at_event", {}) if isinstance(sw, dict) else {}
    windows = sw.get("windows", {}) if isinstance(sw, dict) else {}
    e6 = windows.get("6", {}) if isinstance(windows, dict) else {}
    e24 = windows.get("24", {}) if isinstance(windows, dict) else {}
    e72 = windows.get("72", {}) if isinstance(windows, dict) else {}
    near = nearest_env(env_rows, t) if t else None
    electron = fnum((near or {}).get("electron_gt2mev"))
    proton = fnum((near or {}).get("proton_gt10mev"))
    return [
        event.get("issued_at"),
        ", ".join(event.get("satellites", []) or []),
        event.get("scope"),
        event.get("nature"),
        event.get("issue_category") or "UNKNOWN",
        event.get("cause_attribution") or "NOT ESTABLISHED",
        event.get("title"),
        fnum(at.get("kp")), fnum(at.get("bz_nt")), fnum(at.get("wind_kms")),
        electron, proton,
        fnum(e6.get("max_kp")), fnum(e6.get("min_bz_nt")), fnum(e6.get("max_wind_kms")),
        fnum(e24.get("max_kp")), fnum(e24.get("min_bz_nt")), fnum(e24.get("max_wind_kms")),
        fnum(e72.get("max_kp")), fnum(e72.get("min_bz_nt")), fnum(e72.get("max_wind_kms")),
        (event.get("space_weather_72h_before") or {}).get("association_score_0_100"),
        "Temporal association only; causality NOT established.",
        event.get("source"),
    ]


EVENT_HEADERS = [
    "Event UTC", "Satellite(s)", "Scope", "Nature", "Issue / Cause Category",
    "Cause Attribution", "Public Report",
    "Kp at Event", "Bz at Event (nT)", "Vsw at Event (km/s)",
    ">2 MeV Electron near Event", ">10 MeV Proton near Event",
    "6h Max Kp", "6h Min Bz (nT)", "6h Max Vsw (km/s)",
    "24h Max Kp", "24h Min Bz (nT)", "24h Max Vsw (km/s)",
    "72h Max Kp", "72h Min Bz (nT)", "72h Max Vsw (km/s)",
    "Association Score 0-100", "Interpretation", "Source",
]

ENV_HEADERS = [
    "UTC", "Kp Current", "Bz Current (nT)", "Vsw Current (km/s)",
    ">2 MeV Electron", ">10 MeV Proton",
    "72h Forecast Kp Max", "72h Forecast Bz Min (nT)", "72h Forecast Vsw Max (km/s)",
]


def make_report(month: str, events: list[dict[str, Any]], env_rows: list[dict[str, Any]], legacy_rows: list[dict[str, Any]]) -> Path:
    MONTHLY.mkdir(parents=True, exist_ok=True)
    path = MONTHLY / f"{month}_Satellite_Anomaly_SpaceWeather.xlsx"
    wb = xlsxwriter.Workbook(path)
    wb.set_properties({
        "title": f"SWIFT Satellite Anomaly x Space Weather {month}",
        "subject": "Public satellite anomaly/outage reports linked with space-weather observations",
        "author": "SWIFT",
        "comments": "Association does not establish causality.",
    })

    title_fmt = wb.add_format({"bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#24415D", "align": "left", "valign": "vcenter"})
    header_fmt = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#3B6785", "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True})
    text_fmt = wb.add_format({"valign": "top", "text_wrap": True, "border": 1, "border_color": "#DDE6EE"})
    num_fmt = wb.add_format({"valign": "top", "border": 1, "border_color": "#DDE6EE", "num_format": "0.00"})
    note_fmt = wb.add_format({"text_wrap": True, "font_color": "#6B5B23", "bg_color": "#FFF7DA", "border": 1, "border_color": "#ECDDA8"})
    small_fmt = wb.add_format({"font_size": 9, "font_color": "#60788C", "text_wrap": True})

    ws = wb.add_worksheet("Anomaly_Events")
    ws.freeze_panes(3, 0)
    ws.merge_range(0, 0, 0, len(EVENT_HEADERS)-1, f"SWIFT Satellite Anomaly × Space Weather — {month}", title_fmt)
    ws.merge_range(1, 0, 1, len(EVENT_HEADERS)-1, "Public anomaly/outage reports are kept separate from environmental exposure. Temporal coincidence does NOT establish a space-weather cause.", note_fmt)
    for c, h in enumerate(EVENT_HEADERS):
        ws.write(2, c, h, header_fmt)
    rows = [flatten_event(e, env_rows) for e in events]
    for r, vals in enumerate(rows, start=3):
        for c, v in enumerate(vals):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                ws.write_number(r, c, float(v), num_fmt)
            else:
                ws.write(r, c, v, text_fmt)
        cat = str(vals[4] or "UNKNOWN")
        color = CAUSE_COLORS.get(cat, CAUSE_COLORS["UNKNOWN"])
        ws.write(r, 4, cat, wb.add_format({"bg_color": color, "font_color": "#101820", "bold": True, "border": 1, "text_wrap": True}))
    ws.autofilter(2, 0, max(2, 2+len(rows)), len(EVENT_HEADERS)-1)
    widths = [20, 18, 18, 18, 24, 24, 58] + [13]*15 + [18, 48, 34]
    for i, w in enumerate(widths[:len(EVENT_HEADERS)]):
        ws.set_column(i, i, w)
    ws.set_row(0, 24)
    ws.set_row(1, 42)

    we = wb.add_worksheet("Environment_Log")
    we.freeze_panes(2, 0)
    we.merge_range(0, 0, 0, len(ENV_HEADERS)-1, f"Hourly GEO Space Environment Log — {month}", title_fmt)
    for c, h in enumerate(ENV_HEADERS):
        we.write(1, c, h, header_fmt)
    for r, x in enumerate(env_rows, start=2):
        fc = x.get("forecast_environment_72h") or {}
        vals = [
            x.get("time"), fnum(x.get("kp_current")), fnum(x.get("bz_current_nt")), fnum(x.get("wind_current_kms")),
            fnum(x.get("electron_gt2mev")), fnum(x.get("proton_gt10mev")),
            fnum(fc.get("kp_max_72h")), fnum(fc.get("bz_min_72h_nt")), fnum(fc.get("wind_max_72h_kms")),
        ]
        for c, v in enumerate(vals):
            if isinstance(v, (int, float)):
                we.write_number(r, c, float(v), num_fmt)
            else:
                we.write(r, c, v, text_fmt)
    we.autofilter(1, 0, max(1, 1+len(env_rows)), len(ENV_HEADERS)-1)
    we.set_column(0, 0, 21)
    we.set_column(1, len(ENV_HEADERS)-1, 18)

    wl = wb.add_worksheet("Legacy_NCEI_Reference")
    legacy_headers = ["Anomaly UTC", "Spacecraft", "Orbit", "Longitude", "Altitude km", "Anomaly Type", "Diagnosis", "Diagnosis Category", "Duration min", "Comment", "Source"]
    wl.merge_range(0, 0, 0, len(legacy_headers)-1, "NCEI Legacy Spacecraft Anomaly Reference", title_fmt)
    wl.write(1, 0, "These historical records are reference/training data and are not current satellite health status.", note_fmt)
    for c,h in enumerate(legacy_headers): wl.write(2,c,h,header_fmt)
    # Include records whose anomaly month matches this calendar month-of-year, capped for workbook size.
    month_num = int(month[-2:])
    selected = []
    for x in legacy_rows:
        d = parse_dt(x.get("anomaly_time"))
        if d and d.month == month_num:
            selected.append(x)
    selected = selected[-1500:]
    for r,x in enumerate(selected,start=3):
        vals = [x.get("anomaly_time"),x.get("spacecraft"),x.get("orbit_type"),x.get("longitude_deg"),x.get("altitude_km"),x.get("anomaly_type"),x.get("diagnosis"),x.get("diagnosis_category"),x.get("duration_minutes"),x.get("comment"),x.get("source")]
        for c,v in enumerate(vals):
            if isinstance(v,(int,float)) and not isinstance(v,bool): wl.write_number(r,c,float(v),num_fmt)
            else: wl.write(r,c,v,text_fmt)
        cat=str(x.get("diagnosis_category") or "UNKNOWN")
        wl.write(r,7,cat,wb.add_format({"bg_color":CAUSE_COLORS.get(cat,CAUSE_COLORS["UNKNOWN"]),"bold":True,"border":1,"text_wrap":True}))
    wl.set_column(0,0,20); wl.set_column(1,1,18); wl.set_column(2,8,16); wl.set_column(9,9,58); wl.set_column(10,10,42)
    wl.freeze_panes(3,0)

    wm = wb.add_worksheet("Method_Legend")
    wm.set_column(0,0,28); wm.set_column(1,1,68); wm.set_column(2,2,24)
    wm.merge_range("A1:C1", "Method / Scientific Boundary", title_fmt)
    method_rows = [
        ("Public health", "NOAA OSPO status and anomaly/outage reports. Can include product/data-flow events; not all are spacecraft hardware failures.", "Observed report"),
        ("At-event space weather", "Nearest SWIFT/NOAA Kp, Bz and solar-wind history plus nearest archived GOES electron/proton snapshot when available.", "Descriptive"),
        ("6/24/72 h windows", "Maximum Kp, minimum Bz and maximum Vsw before the event.", "Descriptive"),
        ("Association score", "Environmental-disturbance summary only. It is NOT a probability of failure and does not establish causality.", "Research metric"),
        ("TLE 3D position", "CelesTrak GEO TLE propagated with SGP4 by the hourly backend; Earth-fixed position is approximate for visualization.", "Current position"),
    ]
    for c,h in enumerate(["Item","Definition","Interpretation"]): wm.write(2,c,h,header_fmt)
    for r,row in enumerate(method_rows,start=3):
        for c,v in enumerate(row): wm.write(r,c,v,text_fmt)
    start=10
    wm.write(start,0,"Cause / issue category",header_fmt); wm.write(start,1,"Color",header_fmt); wm.write(start,2,"Meaning",header_fmt)
    meanings={
        "SURFACE_CHARGING_ESD":"Surface electrostatic discharge / charging",
        "INTERNAL_CHARGING":"Deep dielectric / internal charging",
        "SEE_SEU":"Single-event effect / upset",
        "POWER_EPS":"Power / battery / EPS category",
        "ATTITUDE_GNC":"Attitude / pointing / GNC / safehold category",
        "COMMUNICATIONS":"Telemetry / command / communications category",
        "INSTRUMENT":"Instrument-level public alert",
        "PRODUCT_DATA":"Product / data-delivery public alert",
        "UNKNOWN":"Unknown / not publicly attributed",
    }
    rr=start+1
    for cat,color in CAUSE_COLORS.items():
        if cat not in meanings: continue
        wm.write(rr,0,cat,text_fmt)
        wm.write(rr,1,color,wb.add_format({"bg_color":color,"border":1,"bold":True}))
        wm.write(rr,2,meanings[cat],text_fmt)
        rr+=1
    wm.write(rr+2,0,"Sources",header_fmt)
    sources=[
        "https://www.ospo.noaa.gov/operations/goes/status.html",
        "https://www.ospo.noaa.gov/operations/messages.html",
        "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json",
        "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json",
        "https://celestrak.org/NORAD/documentation/gp-data-formats.php",
        "https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/spacecraft-anomalies/data/anomalies.txt",
    ]
    for i,u in enumerate(sources, start=rr+3): wm.write(i,0,u,small_fmt)

    wb.close()
    return path


def main() -> None:
    MONTHLY.mkdir(parents=True, exist_ok=True)
    ledger = load_json(LEDGER, {"events": []})
    events = ledger.get("events", []) if isinstance(ledger, dict) else []
    legacy = load_json(NCEI, {"records": []})
    legacy_rows = legacy.get("records", []) if isinstance(legacy, dict) else []
    now = datetime.now(timezone.utc)
    keys = month_keys(now, 12)
    built = []
    for month in keys:
        env = load_json(ENV_DIR / f"{month}.json", {"records": []})
        env_rows = env.get("records", []) if isinstance(env, dict) else []
        month_events = []
        for e in events:
            d = parse_dt(e.get("issued_at")) if isinstance(e, dict) else None
            if d and d.strftime("%Y-%m") == month:
                month_events.append(e)
        p = make_report(month, month_events, env_rows, legacy_rows)
        built.append({
            "month": month,
            "file": f"satellite-risk/monthly/{p.name}",
            "events": len(month_events),
            "environment_records": len(env_rows),
        })

    keep = set(keys)
    for p in MONTHLY.glob("????-??_Satellite_Anomaly_SpaceWeather.xlsx"):
        if p.name[:7] not in keep:
            p.unlink(missing_ok=True)

    INDEX.write_text(json.dumps({
        "schema_version": "SWIFT-SATELLITE-MONTHLY-XLSX-v0.2",
        "generated_at": now.isoformat().replace("+00:00","Z"),
        "retention_months": 12,
        "reports": built,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"reports": len(built), "latest": built[0] if built else None}, indent=2))


if __name__ == "__main__":
    main()
