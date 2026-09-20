#!/usr/bin/env python3
"""
SWIFT GEO Satellite Anomaly / Space-Weather Risk Builder v0.2

Purpose
-------
Build a public-data research feed that keeps these concepts separate:

1) PUBLIC HEALTH / OUTAGE STATUS
   - NOAA OSPO GOES spacecraft/instrument status
   - NOAA OSPO recent satellite outage/anomaly messages
   These are reports of operational/product issues. They are NOT automatically
   evidence that space weather caused the issue.

2) SPACE ENVIRONMENT EXPOSURE
   - SWIFT/NOAA Kp, Bz, solar wind
   - GOES >2 MeV electron and >10 MeV proton integral flux
   - SWIFT 72 h forecasts

3) ASSOCIATION ANALYSIS
   - For each public GOES anomaly/outage timestamp, summarize the preceding
     72 h Kp/Bz/Vsw environment when local SWIFT history covers the event.
   - This is observational association, not causal attribution.

4) GEO RISK FORECAST
   - Research-oriented environment risk indicators for charging / SEE /
     geomagnetic disturbance / CME-sheath exposure.
   - This is NOT a satellite failure probability.

Output
------
docs/data/satellite-risk/latest.json
docs/data/satellite-risk/history/YYYY-MM-DDTHH.json
docs/data/satellite-risk/event-ledger.json

No credentials are required.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data"
OUT_DIR = DATA / "satellite-risk"
HISTORY_DIR = OUT_DIR / "history"
ENV_HISTORY_DIR = OUT_DIR / "environment-history"
NCEI_CACHE = OUT_DIR / "ncei-legacy.json"
TLE_CACHE = OUT_DIR / "geo-tle-latest.txt"
TLE_META = OUT_DIR / "geo-tle-meta.json"
TLE_REFRESH_HOURS = 23

UA = "SWIFT-GEO-Satellite-Risk/0.4 (+public research dashboard)"
TIMEOUT = 45

OSPO_STATUS = "https://www.ospo.noaa.gov/operations/goes/status.html"
OSPO_MESSAGES = "https://www.ospo.noaa.gov/operations/messages.html"
NCEI_ANOM_SUMMARY = (
    "https://www.ngdc.noaa.gov/stp/space-weather/"
    "satellite-data/spacecraft-anomalies/data/5jsumm.txt"
)
NCEI_ANOM_DOC = (
    "https://www.ngdc.noaa.gov/stp/space-weather/"
    "satellite-data/spacecraft-anomalies/data/anomalies.txt"
)
NCEI_ANOM_RAW = "https://www.ngdc.noaa.gov/stp/satellite/goes/doc/ANOM5JT.TXT"

SWPC_GOES_PRIMARY_E = (
    "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json"
)
SWPC_GOES_PRIMARY_P = (
    "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json"
)
SWPC_GOES_SOURCES = "https://services.swpc.noaa.gov/json/goes/instrument-sources.json"
SWPC_GOES_LONGITUDES = "https://services.swpc.noaa.gov/json/goes/satellite-longitudes.json"

CELESTRAK_GEO = (
    "https://celestrak.org/satcat/records.php?"
    "SPECIAL=gpz&FORMAT=JSON&ACTIVE=1&MAX=2000"
)
CELESTRAK_GEO_TLE = "https://celestrak.org/NORAD/elements/gp.php?GROUP=GEO&FORMAT=TLE"
CELESTRAK_GEO_2LE = "https://celestrak.org/NORAD/elements/gp.php?GROUP=GEO&FORMAT=2LE"

# These are intentionally transparent research bands, not operational failure
# thresholds. They are kept in the output JSON so the UI can display them.
RISK_BANDS = {
    "electron_gt2mev": {
        "metric": "GOES integral >2 MeV electron flux",
        "units": "pfu-like integral flux units from SWPC feed",
        "bands": [
            [0, 100, "LOW"],
            [100, 1000, "ELEVATED"],
            [1000, 10000, "HIGH"],
            [10000, None, "VERY HIGH"],
        ],
        "note": "Research exposure bands; not a satellite failure probability.",
    },
    "proton_gt10mev": {
        "metric": "GOES integral >10 MeV proton flux",
        "units": "pfu",
        "bands": [
            [0, 1, "LOW"],
            [1, 10, "ELEVATED"],
            [10, 100, "HIGH"],
            [100, None, "VERY HIGH"],
        ],
        "note": "10 pfu is the SWPC S1 threshold; risk translation here remains research-only.",
    },
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, (int, float)):
        try:
            # seconds or milliseconds
            x = float(v)
            if x > 1e12:
                x /= 1000.0
            return datetime.fromtimestamp(x, tz=timezone.utc)
        except Exception:
            return None
    s = str(v).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    for fmt in (
        None,
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y%m%d%H%M",
        "%Y%m%d",
    ):
        try:
            if fmt is None:
                d = datetime.fromisoformat(s)
            else:
                d = datetime.strptime(s, fmt)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def fnum(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fetch_text(url: str) -> str:
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": UA})
            r.raise_for_status()
            r.encoding = r.encoding or "utf-8"
            return r.text
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}: {last}")


def fetch_json(url: str) -> Any:
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": UA})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET JSON failed: {url}: {last}")


def safe_fetch_text(url: str, errors: list[str]) -> Optional[str]:
    try:
        return fetch_text(url)
    except Exception as e:
        errors.append(str(e))
        return None


def safe_fetch_json(url: str, errors: list[str]) -> Any:
    try:
        return fetch_json(url)
    except Exception as e:
        errors.append(str(e))
        return None


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.links: list[str] = []
        self._in_a = False
        self._a_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() in {"br", "p", "tr", "li", "div", "h1", "h2", "h3"}:
            self.parts.append("\n")
        if tag.lower() == "a":
            self._in_a = True
            self._a_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            s = " ".join("".join(self._a_parts).split())
            if s:
                self.links.append(unescape(s))
            self._in_a = False
            self._a_parts = []
        if tag.lower() in {"p", "tr", "li", "div"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
        if self._in_a:
            self._a_parts.append(data)

    def text(self) -> str:
        raw = unescape("".join(self.parts))
        return "\n".join(x.strip() for x in raw.splitlines() if x.strip())


def html_text(html: str) -> tuple[str, list[str]]:
    p = TextExtractor()
    p.feed(html)
    return p.text(), p.links


def load_json_candidates(paths: Iterable[Path]) -> Any:
    for p in paths:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def generic_rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in (
            "records", "items", "data", "history", "forecast",
            "kp_history", "wind_history", "bz_history",
            "kp_forecast", "wind_forecast", "bz_forecast",
        ):
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


TIME_KEYS = (
    "time_tag", "timestamp", "time", "datetime", "date_time",
    "target_time", "valid_time", "forecast_time", "observed_at",
)


def row_time(r: dict) -> Optional[datetime]:
    for k in TIME_KEYS:
        if k in r:
            d = parse_time(r.get(k))
            if d:
                return d
    return None


def first_num(r: dict, keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        if k in r:
            x = fnum(r.get(k))
            if x is not None:
                return x
    return None


def normalize_history(payload: Any, kind: str) -> list[tuple[datetime, float]]:
    rows = generic_rows(payload)
    out: list[tuple[datetime, float]] = []
    if kind == "kp":
        keys = ("kp", "Kp", "planetary_k_index", "kp_index")
    elif kind == "bz":
        keys = ("bz_gsm", "bz", "Bz", "bz_min", "bz_median")
    else:
        keys = ("speed", "wind_speed", "bulk_speed", "v", "V")
    for r in rows:
        t = row_time(r)
        v = first_num(r, keys)
        if t and v is not None:
            out.append((t, v))
    out.sort(key=lambda x: x[0])
    return out


def load_local_histories() -> dict[str, list[tuple[datetime, float]]]:
    wind = load_json_candidates([
        DATA / "noaa_wind_history.json",
        DATA / "noaa" / "wind_history.json",
        DATA / "noaa-wind" / "history.json",
    ])
    kp = load_json_candidates([
        DATA / "noaa_kp_history.json",
        DATA / "noaa" / "kp_history.json",
        DATA / "noaa-kp" / "history.json",
    ])
    imf = load_json_candidates([
        DATA / "noaa_imf_history.json",
        DATA / "noaa" / "imf_history.json",
        DATA / "noaa-imf" / "history.json",
    ])
    return {
        "wind": normalize_history(wind, "wind"),
        "kp": normalize_history(kp, "kp"),
        "bz": normalize_history(imf, "bz"),
    }


def load_forecasts() -> dict[str, Any]:
    return {
        "wind": load_json_candidates([DATA / "swift-wind" / "latest.json"]),
        "kp": load_json_candidates([DATA / "swift-kp" / "latest.json"]),
        "bz": load_json_candidates([DATA / "swift-bz" / "latest.json"]),
    }


def forecast_values(payload: Any, kind: str, hours: int = 72) -> list[tuple[datetime, float]]:
    rows = generic_rows(payload)
    now = now_utc()
    end = now + timedelta(hours=hours)
    if kind == "kp":
        keys = ("predicted_kp", "forecast_kp", "kp", "Kp")
    elif kind == "bz":
        keys = ("bz_min_forecast", "bz_forecast", "predicted_bz", "bz", "Bz")
    else:
        keys = (
            "swift_cme_enhanced_speed", "enhanced_speed", "forecast_speed",
            "predicted_speed", "swift_speed", "speed",
        )
    out = []
    for r in rows:
        t = row_time(r)
        v = first_num(r, keys)
        if t and v is not None and now - timedelta(hours=2) <= t <= end:
            out.append((t, v))
    out.sort(key=lambda x: x[0])
    return out


def parse_goes_status(html: str) -> dict[str, Any]:
    text, _ = html_text(html)
    sats: list[dict[str, Any]] = []
    for sat in ("GOES-19", "GOES-18", "GOES-16", "GOES-17"):
        i = text.find(sat)
        if i < 0:
            continue
        chunk = text[i:i+1500]
        role = None
        color = None
        for r in ("Operational East", "Operational West", "On-Orbit Storage", "GOES-East", "GOES-West"):
            if r.lower() in chunk.lower():
                role = r
                break
        for c in ("Green", "Yellow", "Orange", "Red", "Blue"):
            if re.search(rf"\b{c}\b", chunk, flags=re.I):
                color = c.title()
                break
        loc = None
        m = re.search(r"Spacecraft Location:\s*([0-9.]+\s*[EW])", chunk, flags=re.I)
        if m:
            loc = m.group(1).replace(" ", "")
        sats.append({
            "name": sat,
            "operational_status": role,
            "status_color": color,
            "location": loc,
            "public_status_source": "NOAA OSPO",
        })
    return {
        "satellites": sats,
        "note": "OSPO status page is generally updated monthly; recent outages are handled separately.",
    }


MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], 1
)}


def parse_issue_time(text: str) -> Optional[datetime]:
    s = " ".join(text.replace(",", " , ").split())
    # Month DD, YYYY HHMMZ
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2})\s*,?\s*(20\d{2}).{0,20}?(\d{4})\s*Z?",
        s, re.I
    )
    if m:
        try:
            return datetime(
                int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)),
                int(m.group(4)[:2]), int(m.group(4)[2:]), tzinfo=timezone.utc
            )
        except Exception:
            pass
    # Sep 08, 2026 1810Z
    short = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    }
    m = re.search(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*,?\s*(20\d{2}).{0,20}?(\d{4})\s*Z?", s)
    if m and m.group(1).lower() in short:
        try:
            return datetime(
                int(m.group(3)), short[m.group(1).lower()], int(m.group(2)),
                int(m.group(4)[:2]), int(m.group(4)[2:]), tzinfo=timezone.utc
            )
        except Exception:
            pass
    return None


def classify_public_event(text: str) -> dict[str, Any]:
    u = text.upper()
    sats = sorted(set(re.findall(r"GOES-\d{1,2}", u)))
    if any(k in u for k in ("SAFEHOLD", "SPACECRAFT", "ATTITUDE", "POWER", "BATTERY", "THRUSTER")):
        scope = "spacecraft/operations"
    elif any(k in u for k in ("ABI", "SEISS", "EXIS", "SUVI", "CCOR", "INSTRUMENT")):
        scope = "instrument/product"
    else:
        scope = "product/data"
    if "PLANNED" in u or "SCHEDULED" in u or "MAINTENANCE" in u:
        nature = "planned/administrative"
    elif "ANOMALY" in u or "OUTAGE" in u or "SAFEHOLD" in u:
        nature = "anomaly/outage"
    else:
        nature = "other"

    # This is an issue CATEGORY derived from the public text, not a root-cause claim.
    if any(k in u for k in ("ESD", "ELECTROSTATIC", "SURFACE CHARG")):
        issue_category = "SURFACE_CHARGING_ESD"
    elif any(k in u for k in ("ECEMP", "DEEP DIELECTRIC", "INTERNAL CHARG")):
        issue_category = "INTERNAL_CHARGING"
    elif any(k in u for k in ("SEU", "SINGLE EVENT", "RADIATION UPSET")):
        issue_category = "SEE_SEU"
    elif any(k in u for k in ("POWER", "BATTERY", "EPS")):
        issue_category = "POWER_EPS"
    elif any(k in u for k in ("ATTITUDE", "POINTING", "GNC", "SAFEHOLD", "THRUSTER")):
        issue_category = "ATTITUDE_GNC"
    elif any(k in u for k in ("TELEMETRY", "COMMAND", "COMMUNICATION", "COMMS", "RF")):
        issue_category = "COMMUNICATIONS"
    elif any(k in u for k in ("ABI", "SEISS", "EXIS", "SUVI", "CCOR", "INSTRUMENT")):
        issue_category = "INSTRUMENT"
    elif any(k in u for k in ("PRODUCT", "DATA", "DELIVERY", "NCCF")):
        issue_category = "PRODUCT_DATA"
    else:
        issue_category = "UNKNOWN"
    return {
        "satellites": sats, "scope": scope, "nature": nature,
        "issue_category": issue_category,
        "cause_attribution": "CATEGORY_FROM_PUBLIC_TEXT_NOT_ROOT_CAUSE",
    }


def parse_ospo_messages(html: str) -> list[dict[str, Any]]:
    text, links = html_text(html)
    candidates = links + [x.strip() for x in text.splitlines()]
    out = []
    seen = set()
    for s in candidates:
        if "GOES-" not in s.upper():
            continue
        if not any(k in s.upper() for k in ("ANOMALY", "OUTAGE", "SAFEHOLD", "FAILURE")):
            continue
        clean = " ".join(s.split())
        key = re.sub(r"\s+", " ", clean.upper())
        if key in seen:
            continue
        seen.add(key)
        d = parse_issue_time(clean)
        cls = classify_public_event(clean)
        out.append({
            "issued_at": iso(d),
            "title": clean[:700],
            **cls,
            "source": "NOAA OSPO satellite messages",
            "causal_attribution": "NOT ESTABLISHED",
        })
    out.sort(key=lambda x: x.get("issued_at") or "", reverse=True)
    return out


def parse_ncei_summary(txt: str) -> list[dict[str, Any]]:
    out = []
    pat = re.compile(
        r"^\s*\d+\s+(.+?)\s+(\d{2}/\d{2}/\d{2})\s+(\d{2}/\d{2}/\d{2})\s+(\d+)\s*$"
    )
    for line in txt.splitlines():
        m = pat.match(line)
        if not m:
            continue
        name = m.group(1).strip()
        out.append({
            "satellite": name,
            "first_report": m.group(2),
            "last_report": m.group(3),
            "count": int(m.group(4)),
        })
    out.sort(key=lambda x: x["count"], reverse=True)
    return out


def choose_integral_flux(payload: Any, particle: str, energy_token: str) -> dict[str, Any]:
    rows = generic_rows(payload)
    best = None
    for r in rows:
        energy = str(r.get("energy") or r.get("channel") or r.get("energy_label") or "")
        species = str(r.get("species") or r.get("particle") or "").lower()
        if particle.lower() not in species and particle.lower() not in str(r).lower():
            # SWPC rows often omit explicit species because endpoint already identifies it.
            pass
        if energy_token.lower().replace(" ", "") not in energy.lower().replace(" ", ""):
            continue
        t = row_time(r)
        flux = first_num(r, ("flux", "value", "integral_flux"))
        if t and flux is not None and (best is None or t > best[0]):
            best = (t, flux, r, energy)
    if best is None:
        # Fallback: most recent finite row; still label the actual energy string.
        for r in rows:
            t = row_time(r)
            flux = first_num(r, ("flux", "value", "integral_flux"))
            if t and flux is not None and (best is None or t > best[0]):
                best = (t, flux, r, str(r.get("energy") or r.get("channel") or "unknown"))
    if best is None:
        return {"time": None, "flux": None, "energy": energy_token, "satellite": None}
    t, flux, r, energy = best
    return {
        "time": iso(t),
        "flux": flux,
        "energy": energy,
        "satellite": r.get("satellite") or r.get("satellite_id") or r.get("satellite_name"),
    }


def window_values(series: list[tuple[datetime, float]], start: datetime, end: datetime) -> list[float]:
    return [v for t, v in series if start <= t <= end and math.isfinite(v)]


def summarize_event_environment(event: dict[str, Any], histories: dict[str, list[tuple[datetime,float]]]) -> dict[str, Any]:
    t = parse_time(event.get("issued_at"))
    if t is None:
        return {"covered": False, "reason": "event timestamp unavailable"}
    start = t - timedelta(hours=72)
    w = window_values(histories["wind"], start, t)
    k = window_values(histories["kp"], start, t)
    b = window_values(histories["bz"], start, t)
    metrics = {
        "window_start": iso(start),
        "window_end": iso(t),
        "max_wind_kms": max(w) if w else None,
        "max_kp": max(k) if k else None,
        "min_bz_nt": min(b) if b else None,
        "n_wind": len(w),
        "n_kp": len(k),
        "n_bz": len(b),
    }
    covered = bool(w or k or b)
    flags = {
        "wind_ge_500": metrics["max_wind_kms"] is not None and metrics["max_wind_kms"] >= 500,
        "kp_ge_5": metrics["max_kp"] is not None and metrics["max_kp"] >= 5,
        "bz_le_minus5": metrics["min_bz_nt"] is not None and metrics["min_bz_nt"] <= -5,
    }
    score = 0.0
    if metrics["max_wind_kms"] is not None:
        score += 30 * clamp((metrics["max_wind_kms"] - 350) / 350, 0, 1)
    if metrics["max_kp"] is not None:
        score += 35 * clamp((metrics["max_kp"] - 2) / 6, 0, 1)
    if metrics["min_bz_nt"] is not None:
        score += 35 * clamp((-metrics["min_bz_nt"] - 2) / 10, 0, 1)
    return {
        "covered": covered,
        "metrics": metrics,
        "flags": flags,
        "association_score_0_100": round(score, 1),
        "interpretation": (
            "Descriptive 72 h pre-event environment only; does not establish that space weather caused the anomaly."
        ),
    }


def label_from_score(score: float) -> str:
    if score >= 75:
        return "VERY HIGH"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MODERATE"
    return "LOW"


def current_forecast_environment(forecasts: dict[str, Any]) -> dict[str, Any]:
    wind = forecast_values(forecasts.get("wind"), "wind")
    kp = forecast_values(forecasts.get("kp"), "kp")
    bz = forecast_values(forecasts.get("bz"), "bz")
    return {
        "wind_max_72h_kms": max((v for _, v in wind), default=None),
        "kp_max_72h": max((v for _, v in kp), default=None),
        "bz_min_72h_nt": min((v for _, v in bz), default=None),
        "forecast_points": {"wind": len(wind), "kp": len(kp), "bz": len(bz)},
    }


def risk_components(env: dict[str, Any], electron: dict[str, Any], proton: dict[str, Any]) -> dict[str, Any]:
    e = fnum(electron.get("flux"))
    p = fnum(proton.get("flux"))
    kp = fnum(env.get("kp_max_72h"))
    bz = fnum(env.get("bz_min_72h_nt"))
    wind = fnum(env.get("wind_max_72h_kms"))

    # Transparent research scores 0..100.
    if e is None or e <= 0:
        internal = 0.0
    else:
        internal = clamp((math.log10(max(e, 1)) - 2.0) / 2.0 * 100, 0, 100)

    if p is None or p <= 0:
        see = 0.0
    else:
        see = clamp((math.log10(max(p, 0.1)) - 0.0) / 2.0 * 100, 0, 100)

    surface = 0.0
    if kp is not None:
        surface += 60 * clamp((kp - 3) / 5, 0, 1)
    if bz is not None:
        surface += 40 * clamp((-bz - 3) / 9, 0, 1)

    cme = 0.0
    if wind is not None:
        cme += 55 * clamp((wind - 450) / 350, 0, 1)
    if bz is not None:
        cme += 45 * clamp((-bz - 4) / 10, 0, 1)

    comps = {
        "internal_charging": round(clamp(internal, 0, 100), 1),
        "single_event_effect": round(clamp(see, 0, 100), 1),
        "surface_charging_geomagnetic": round(clamp(surface, 0, 100), 1),
        "cme_sheath_environment": round(clamp(cme, 0, 100), 1),
    }
    overall = max(comps.values()) if comps else 0.0
    return {
        "scores_0_100": comps,
        "overall_environment_risk_score": round(overall, 1),
        "overall_environment_risk_label": label_from_score(overall),
        "warning": "Environment exposure score only; NOT probability of satellite failure.",
    }



NCEI_DIAGNOSIS_LABELS = {
    "ECEMP": "INTERNAL_CHARGING",
    "ESD": "SURFACE_CHARGING_ESD",
    "SEU": "SEE_SEU",
    "MCP": "MISSION_CONTROL_SOFTWARE",
    "RFI": "COMMUNICATIONS_RFI",
    "UNK": "UNKNOWN",
}

def parse_ncei_fixed_width(txt: str) -> list[dict[str, Any]]:
    """Parse Version-5 fixed-width spacecraft anomaly records (152 columns)."""
    out: list[dict[str, Any]] = []
    for raw in txt.splitlines():
        line = raw.rstrip("\r\n")
        if len(line) < 69 or not line[:1].strip().isdigit():
            continue
        line = line.ljust(152)
        try:
            spacecraft = line[9:19].strip()
            date_s = line[19:27].strip()
            time_s = line[27:31].strip()
            if not spacecraft or not re.fullmatch(r"\d{8}", date_s):
                continue
            dt = datetime.strptime(date_s, "%Y%m%d").replace(tzinfo=timezone.utc)
            if time_s and time_s != "9999" and re.fullmatch(r"\d{4}", time_s):
                hh, mm = int(time_s[:2]), int(time_s[2:])
                if hh < 24 and mm < 60:
                    dt = dt.replace(hour=hh, minute=mm)
            lat = None
            if line[43:44].strip() in {"N", "S"} and line[44:46].strip().isdigit():
                lat = int(line[44:46].strip()) * (1 if line[43:44] == "N" else -1)
            lon = None
            if line[48:49].strip() in {"E", "W"} and line[49:52].strip().isdigit():
                lon = int(line[49:52].strip()) * (1 if line[48:49] == "E" else -1)
            alt = fnum(line[54:59].strip())
            anomaly_type = line[59:64].strip()
            diagnosis = line[64:69].strip()
            out.append({
                "spacecraft": spacecraft,
                "anomaly_time": iso(dt),
                "time_known": time_s != "9999",
                "uncertainty_minutes": fnum(line[31:34].strip()),
                "duration_minutes": fnum(line[34:38].strip()),
                "local_time_hhmm": line[38:42].strip() or None,
                "orbit_type": line[42:43].strip() or None,
                "latitude_deg": lat,
                "longitude_deg": lon,
                "altitude_km": alt,
                "anomaly_type": anomaly_type or None,
                "diagnosis": diagnosis or None,
                "diagnosis_category": NCEI_DIAGNOSIS_LABELS.get(diagnosis, "UNKNOWN"),
                "comment": line[69:148].strip() or None,
                "sun_vehicle_earth_angle_deg": fnum(line[148:151].strip()),
                "attitude_stabilization": line[151:152].strip() or None,
                "source": NCEI_ANOM_RAW,
            })
        except Exception:
            continue
    out.sort(key=lambda x: x.get("anomaly_time") or "")
    return out


def nearest_value(series: list[tuple[datetime, float]], t: datetime, max_gap_hours: float = 4.0) -> Optional[float]:
    if not series:
        return None
    best = min(series, key=lambda x: abs((x[0] - t).total_seconds()))
    if abs((best[0] - t).total_seconds()) > max_gap_hours * 3600:
        return None
    return best[1]


def summarize_event_windows(event: dict[str, Any], histories: dict[str, list[tuple[datetime,float]]]) -> dict[str, Any]:
    t = parse_time(event.get("issued_at") or event.get("anomaly_time"))
    if t is None:
        return {"covered": False, "reason": "event timestamp unavailable"}
    at = {
        "kp": nearest_value(histories["kp"], t, 4),
        "bz_nt": nearest_value(histories["bz"], t, 2),
        "wind_kms": nearest_value(histories["wind"], t, 2),
    }
    windows = {}
    for h in (6, 24, 72):
        start = t - timedelta(hours=h)
        w = window_values(histories["wind"], start, t)
        k = window_values(histories["kp"], start, t)
        b = window_values(histories["bz"], start, t)
        windows[str(h)] = {
            "max_wind_kms": max(w) if w else None,
            "max_kp": max(k) if k else None,
            "min_bz_nt": min(b) if b else None,
            "n_wind": len(w), "n_kp": len(k), "n_bz": len(b),
        }
    return {
        "covered": any(v is not None for v in at.values()) or any(
            any(x is not None for x in (m["max_wind_kms"], m["max_kp"], m["min_bz_nt"]))
            for m in windows.values()
        ),
        "at_event": at,
        "windows": windows,
        "interpretation": "Descriptive co-occurrence only; causal attribution is not automatic.",
    }


def parse_tle_sets(txt: str) -> list[dict[str, Any]]:
    lines = [x.rstrip() for x in txt.splitlines() if x.strip()]
    out = []
    i = 0
    while i < len(lines):
        if i + 2 < len(lines) and not lines[i].startswith("1 ") and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            name, l1, l2 = lines[i].strip(), lines[i+1], lines[i+2]
            i += 3
        elif i + 1 < len(lines) and lines[i].startswith("1 ") and lines[i+1].startswith("2 "):
            name, l1, l2 = f"NORAD {lines[i][2:7].strip()}", lines[i], lines[i+1]
            i += 2
        else:
            i += 1
            continue
        try:
            norad = int(l1[2:7].strip())
        except Exception:
            continue
        out.append({"name": name, "norad": norad, "tle1": l1, "tle2": l2})
    return out


def _valid_tle_text(txt: str) -> bool:
    """A valid GEO response must yield several actual two-line element sets."""
    try:
        return len(parse_tle_sets(txt)) >= 3
    except Exception:
        return False


def load_daily_tle_cache(errors: list[str]) -> tuple[str, dict[str, Any]]:
    """
    Keep ONE latest GEO TLE file only.

    The anomaly/risk workflow may run hourly, but CelesTrak is contacted only when
    the cached TLE is >= TLE_REFRESH_HOURS old (or the cache is missing).
    If the refresh fails, the last good cache is retained.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = now_utc()

    old_txt = ""
    if TLE_CACHE.exists():
        try:
            old_txt = TLE_CACHE.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"TLE cache read failed: {e}")

    meta = load_json_candidates([TLE_META]) or {}
    fetched_at = parse_time(meta.get("fetched_at")) if isinstance(meta, dict) else None
    fresh = bool(
        old_txt and _valid_tle_text(old_txt) and fetched_at
        and (now - fetched_at) < timedelta(hours=TLE_REFRESH_HOURS)
    )

    if fresh:
        return old_txt, {
            "mode": "cache",
            "fetched_at": iso(fetched_at),
            "refresh_hours": TLE_REFRESH_HOURS,
            "tle_sets": len(parse_tle_sets(old_txt)),
            "file": "docs/data/satellite-risk/geo-tle-latest.txt",
            "retention": "latest only",
        }

    refreshed = None
    used_url = None
    for url in (CELESTRAK_GEO_TLE, CELESTRAK_GEO_2LE):
        try:
            txt = fetch_text(url)
            if _valid_tle_text(txt):
                refreshed = txt
                used_url = url
                break
            errors.append(f"TLE response parsed zero/few records: {url}")
        except Exception as e:
            errors.append(f"TLE refresh failed {url}: {e}")

    if refreshed:
        TLE_CACHE.write_text(refreshed, encoding="utf-8")
        meta = {
            "fetched_at": iso(now),
            "source": used_url,
            "refresh_hours": TLE_REFRESH_HOURS,
            "retention": "latest only",
            "tle_sets": len(parse_tle_sets(refreshed)),
        }
        TLE_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return refreshed, {"mode": "refreshed", **meta, "file": "docs/data/satellite-risk/geo-tle-latest.txt"}

    if old_txt and _valid_tle_text(old_txt):
        errors.append("Using last good GEO TLE cache because daily refresh failed.")
        return old_txt, {
            "mode": "stale-cache",
            "fetched_at": iso(fetched_at),
            "refresh_hours": TLE_REFRESH_HOURS,
            "tle_sets": len(parse_tle_sets(old_txt)),
            "file": "docs/data/satellite-risk/geo-tle-latest.txt",
            "retention": "latest only",
        }

    return "", {
        "mode": "unavailable",
        "fetched_at": None,
        "refresh_hours": TLE_REFRESH_HOURS,
        "tle_sets": 0,
        "file": "docs/data/satellite-risk/geo-tle-latest.txt",
        "retention": "latest only",
    }


def swpc_goes_fallback_positions(longitude_json: Any, status_rows: list[dict[str, Any]], when: datetime) -> list[dict[str, Any]]:
    """
    TLE is primary. If no TLE rows can be propagated, show NOAA GOES satellites
    at SWPC-published geostationary longitudes so the 3-D view never silently
    becomes empty. SWPC longitude values are west longitudes in this feed.
    """
    raw = longitude_json if isinstance(longitude_json, list) else []
    status_by_name = {str(x.get("name", "")).upper(): x for x in status_rows}
    out = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        sat = r.get("satellite")
        lon_w = fnum(r.get("longitude"))
        if sat is None or lon_w is None:
            continue
        try:
            satn = int(sat)
        except Exception:
            continue
        name = f"GOES-{satn}"
        public = status_by_name.get(name.upper(), {})
        out.append({
            "name": name,
            "norad": None,
            "latitude_deg": 0.0,
            "longitude_deg": -abs(float(lon_w)),
            "altitude_km": 35786.0,
            "position_time": iso(when),
            "ops_status_code": None,
            "owner": "US",
            "launch_date": None,
            "public_health_label": public.get("public_health_label"),
            "public_status_color": public.get("status_color"),
            "recent_public_alert": None,
            "recent_public_alert_age_hours": None,
            "issue_category": "NONE",
            "issue_color": None,
            "position_basis": "NOAA SWPC published GOES longitude fallback; TLE unavailable.",
            "position_quality": "SWPC_LONGITUDE_FALLBACK",
        })
    return out


def gmst_rad(dt: datetime) -> float:
    jd = 2440587.5 + dt.timestamp() / 86400.0
    t = (jd - 2451545.0) / 36525.0
    deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t - (t ** 3) / 38710000.0
    return math.radians(deg % 360.0)


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    a = 6378.137
    f = 1.0 / 298.257223563
    e2 = f * (2 - f)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - e2))
    alt = 0.0
    for _ in range(6):
        sinp = math.sin(lat)
        n = a / math.sqrt(1 - e2 * sinp * sinp)
        alt = p / max(1e-9, math.cos(lat)) - n
        lat = math.atan2(z, p * (1 - e2 * n / max(1e-6, n + alt)))
    return math.degrees(lat), ((math.degrees(lon) + 180) % 360) - 180, alt


def propagate_tles(tle_sets: list[dict[str, Any]], when: datetime, errors: list[str]) -> list[dict[str, Any]]:
    try:
        from sgp4.api import Satrec, jday
    except Exception as e:
        errors.append(f"sgp4 unavailable: {e}")
        return []
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute, when.second + when.microsecond / 1e6)
    th = gmst_rad(when)
    ct, st = math.cos(th), math.sin(th)
    out = []
    for r in tle_sets:
        try:
            sat = Satrec.twoline2rv(r["tle1"], r["tle2"])
            err, pos, _vel = sat.sgp4(jd, fr)
            if err != 0:
                continue
            xi, yi, zi = pos
            # TEME -> approximate Earth-fixed rotation, sufficient for the dashboard placement.
            xe = ct * xi + st * yi
            ye = -st * xi + ct * yi
            ze = zi
            lat, lon, alt = ecef_to_geodetic(xe, ye, ze)
            row = dict(r)
            row.update({"latitude_deg": lat, "longitude_deg": lon, "altitude_km": alt, "position_time": iso(when)})
            out.append(row)
        except Exception:
            continue
    return out


def issue_color(category: str) -> str:
    return {
        "SURFACE_CHARGING_ESD": "#ff8c42",
        "INTERNAL_CHARGING": "#9b5de5",
        "SEE_SEU": "#ff4fa3",
        "POWER_EPS": "#ffd34e",
        "ATTITUDE_GNC": "#36d7ff",
        "COMMUNICATIONS": "#4c8cff",
        "COMMUNICATIONS_RFI": "#4c8cff",
        "INSTRUMENT": "#ffb14e",
        "PRODUCT_DATA": "#e6a84c",
        "MISSION_CONTROL_SOFTWARE": "#a4b0bd",
        "UNKNOWN": "#ff5f63",
    }.get(str(category or "UNKNOWN"), "#ff5f63")


def write_environment_snapshot(result: dict[str, Any]) -> None:
    ENV_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    now = parse_time(result.get("generated_at")) or now_utc()
    month = now.strftime("%Y-%m")
    p = ENV_HISTORY_DIR / f"{month}.json"
    old = load_json_candidates([p]) or {"month": month, "records": []}
    rows = old.get("records", []) if isinstance(old, dict) else []
    env = result.get("goes_space_environment_now", {})
    histories = load_local_histories()
    snap = {
        "time": iso(now),
        "kp_current": nearest_value(histories["kp"], now, 6),
        "bz_current_nt": nearest_value(histories["bz"], now, 2),
        "wind_current_kms": nearest_value(histories["wind"], now, 2),
        "electron_gt2mev": env.get("electron_gt2mev", {}).get("flux"),
        "proton_gt10mev": env.get("proton_gt10mev", {}).get("flux"),
        "forecast_environment_72h": result.get("forecast_environment_72h", {}),
    }
    by_hour = {str(x.get("time", ""))[:13]: x for x in rows if isinstance(x, dict)}
    by_hour[snap["time"][:13]] = snap
    vals = sorted(by_hour.values(), key=lambda x: x.get("time") or "")
    p.write_text(json.dumps({"month": month, "updated_at": iso(now), "records": vals}, ensure_ascii=False, indent=2), encoding="utf-8")
    # Rolling 12 calendar months.
    keep = set()
    y, m = now.year, now.month
    for i in range(12):
        yy, mm = y, m - i
        while mm <= 0:
            yy -= 1; mm += 12
        keep.add(f"{yy:04d}-{mm:02d}")
    for q in ENV_HISTORY_DIR.glob("????-??.json"):
        if q.stem not in keep:
            q.unlink(missing_ok=True)

def public_health_label(s: dict[str, Any]) -> str:
    c = str(s.get("status_color") or "").lower()
    if c == "green":
        return "OPERATIONAL"
    if c == "yellow":
        return "LIMITATION"
    if c == "orange":
        return "DEGRADED"
    if c == "red":
        return "NON-OPERATIONAL"
    if c == "blue":
        return "FUNCTIONAL / OFF OR STORAGE"
    return "PUBLIC STATUS UNKNOWN"


def build() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ENV_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    status_html = safe_fetch_text(OSPO_STATUS, errors) or ""
    msg_html = safe_fetch_text(OSPO_MESSAGES, errors) or ""
    # Current-year message archive gives more correlation opportunities than only
    # the 100 most recent messages.
    year = now_utc().year
    year_url = f"https://www.ospo.noaa.gov/data/messages/{year}/{year}_include.html"
    year_html = safe_fetch_text(year_url, errors) or ""

    ncei_txt = safe_fetch_text(NCEI_ANOM_SUMMARY, errors) or ""
    ncei_raw_txt = safe_fetch_text(NCEI_ANOM_RAW, errors) or ""
    tle_txt, tle_cache_info = load_daily_tle_cache(errors)
    electron_json = safe_fetch_json(SWPC_GOES_PRIMARY_E, errors)
    proton_json = safe_fetch_json(SWPC_GOES_PRIMARY_P, errors)
    source_json = safe_fetch_json(SWPC_GOES_SOURCES, errors)
    longitude_json = safe_fetch_json(SWPC_GOES_LONGITUDES, errors)
    geo_catalog = safe_fetch_json(CELESTRAK_GEO, errors)

    status = parse_goes_status(status_html) if status_html else {"satellites": [], "note": "status fetch failed"}
    events = parse_ospo_messages(msg_html + "\n" + year_html) if (msg_html or year_html) else []
    ncei_summary = parse_ncei_summary(ncei_txt) if ncei_txt else []
    ncei_events = parse_ncei_fixed_width(ncei_raw_txt) if ncei_raw_txt else []
    tle_sets = parse_tle_sets(tle_txt) if tle_txt else []

    electron = choose_integral_flux(electron_json, "electron", ">2 MeV")
    proton = choose_integral_flux(proton_json, "proton", ">10 MeV")

    histories = load_local_histories()
    forecasts = load_forecasts()
    env72 = current_forecast_environment(forecasts)
    risk = risk_components(env72, electron, proton)

    # Attach 72 h pre-event environment to recent public GOES anomaly/outage reports.
    corr = []
    for ev in events[:250]:
        row = dict(ev)
        row["space_weather_72h_before"] = summarize_event_environment(ev, histories)
        row["space_weather_windows"] = summarize_event_windows(ev, histories)
        corr.append(row)

    covered = [x for x in corr if x["space_weather_72h_before"].get("covered")]
    association_summary = {
        "public_events_total": len(corr),
        "events_with_local_environment_coverage": len(covered),
        "events_with_kp_ge_5": sum(
            1 for x in covered if x["space_weather_72h_before"].get("flags", {}).get("kp_ge_5")
        ),
        "events_with_wind_ge_500": sum(
            1 for x in covered if x["space_weather_72h_before"].get("flags", {}).get("wind_ge_500")
        ),
        "events_with_bz_le_minus5": sum(
            1 for x in covered if x["space_weather_72h_before"].get("flags", {}).get("bz_le_minus5")
        ),
        "interpretation": (
            "Counts show co-occurrence inside the 72 h pre-event window. "
            "They are not a causal test and include product/data outages."
        ),
    }

    # Same environment exposure applies broadly across GEO, while public health
    # status is satellite-specific. This first proof-of-concept prioritizes GOES.
    satellites = []
    for s in status.get("satellites", []):
        sat_events = [
            e for e in corr[:100]
            if s["name"].upper() in [x.upper() for x in e.get("satellites", [])]
        ]
        satellites.append({
            **s,
            "public_health_label": public_health_label(s),
            "latest_public_anomaly_or_outage": sat_events[0] if sat_events else None,
            "environment_forecast_72h": env72,
            "environment_risk": risk,
            "important_note": (
                "Public health state and environment risk are separate. "
                "A high environment score does not mean this satellite is failing."
            ),
        })

    # Top historic anomaly counts: useful context, not current operational status.
    historic_geo_like = [
        x for x in ncei_summary
        if any(k in x["satellite"].upper() for k in (
            "GOES", "GMS", "METEOSAT", "INTEL", "ARABSAT",
            "AUSSAT", "BRASIL", "PALAPA", "INSAT", "MARECS", "COMSTAR", "GSTAR"
        ))
    ][:40]

    # Keep only a compact subset of CelesTrak metadata to avoid a large output file.
    geo_rows = []
    if isinstance(geo_catalog, list):
        for r in geo_catalog[:2000]:
            if not isinstance(r, dict):
                continue
            geo_rows.append({
                "name": r.get("OBJECT_NAME"),
                "norad": r.get("NORAD_CAT_ID"),
                "ops_status_code": r.get("OPS_STATUS_CODE"),
                "owner": r.get("OWNER"),
                "launch_date": r.get("LAUNCH_DATE"),
                "period_min": r.get("PERIOD"),
                "apogee_km": r.get("APOGEE"),
                "perigee_km": r.get("PERIGEE"),
            })


    # TLE/SGP4-based approximate current GEO positions for the 3-D scene.
    satcat_by_norad = {}
    if isinstance(geo_catalog, list):
        for r in geo_catalog:
            if not isinstance(r, dict):
                continue
            try:
                satcat_by_norad[int(r.get("NORAD_CAT_ID"))] = r
            except Exception:
                pass
    propagated = propagate_tles(tle_sets, now_utc(), errors)
    status_by_name = {str(x.get("name", "")).upper(): x for x in satellites}
    recent_cutoff = now_utc() - timedelta(hours=72)
    recent_events_by_sat: dict[str, list[dict[str, Any]]] = {}
    for ev in corr:
        et = parse_time(ev.get("issued_at"))
        if not et or et < recent_cutoff or ev.get("nature") == "planned/administrative":
            continue
        for name in ev.get("satellites", []):
            recent_events_by_sat.setdefault(str(name).upper(), []).append(ev)
    geo_satellites_3d = []
    for r in propagated:
        meta = satcat_by_norad.get(int(r["norad"]), {})
        name = str(meta.get("OBJECT_NAME") or r.get("name") or f"NORAD {r['norad']}")
        up = name.upper().replace(" ", "-")
        public = None
        for k, v in status_by_name.items():
            if k.replace(" ", "-") in up or up in k.replace(" ", "-"):
                public = v; break
        alerts = []
        for k, vals in recent_events_by_sat.items():
            if k in up:
                alerts.extend(vals)
        latest_alert = sorted(alerts, key=lambda x: x.get("issued_at") or "", reverse=True)[0] if alerts else None
        cat = latest_alert.get("issue_category") if latest_alert else None
        alert_age_hours = None
        if latest_alert:
            aet = parse_time(latest_alert.get("issued_at"))
            if aet:
                alert_age_hours = round(max(0.0, (now_utc() - aet).total_seconds() / 3600.0), 2)
        geo_satellites_3d.append({
            "name": name,
            "norad": r["norad"],
            "latitude_deg": round(r["latitude_deg"], 4),
            "longitude_deg": round(r["longitude_deg"], 4),
            "altitude_km": round(r["altitude_km"], 1),
            "position_time": r["position_time"],
            "ops_status_code": meta.get("OPS_STATUS_CODE"),
            "owner": meta.get("OWNER"),
            "launch_date": meta.get("LAUNCH_DATE"),
            "public_health_label": public.get("public_health_label") if public else None,
            "public_status_color": public.get("status_color") if public else None,
            "recent_public_alert": latest_alert,
            "recent_public_alert_age_hours": alert_age_hours,
            "issue_category": cat or "NONE",
            "issue_color": issue_color(cat or "UNKNOWN") if latest_alert else None,
            "position_basis": "CelesTrak TLE + SGP4; Earth-fixed conversion is approximate for visualization.",
        })
    # Keep active-payload matches first, but preserve GOES/alerts even if SATCAT metadata is incomplete.
    geo_satellites_3d.sort(key=lambda x: (0 if (x.get("recent_public_alert") or "GOES" in x["name"].upper()) else 1, x["name"]))
    # Do not leave the 3-D layer blank. TLE remains primary; SWPC GOES longitude
    # is a transparent fallback only when no TLE could be propagated.
    if not geo_satellites_3d:
        geo_satellites_3d = swpc_goes_fallback_positions(longitude_json, satellites, now_utc())
        for row in geo_satellites_3d:
            alerts = recent_events_by_sat.get(str(row.get("name", "")).upper(), [])
            latest_alert = sorted(alerts, key=lambda x: x.get("issued_at") or "", reverse=True)[0] if alerts else None
            row["recent_public_alert"] = latest_alert
            if latest_alert:
                cat = latest_alert.get("issue_category") or "UNKNOWN"
                row["issue_category"] = cat
                row["issue_color"] = issue_color(cat)

    if ncei_events:
        NCEI_CACHE.write_text(json.dumps({
            "schema_version": "NCEI-SC-ANOMALY-V5-normalized-v0.2",
            "updated_at": iso(now_utc()),
            "source": NCEI_ANOM_RAW,
            "records": ncei_events,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "schema_version": "SWIFT-GEO-SAT-RISK-v0.4",
        "generated_at": iso(now_utc()),
        "purpose": (
            "Research dashboard for public satellite anomaly status, GEO space-environment exposure, "
            "and time-linked anomaly/environment analysis."
        ),
        "scientific_boundary": {
            "public_anomaly_status": "Observed/reported operational or product anomaly/outage.",
            "space_weather_risk": "Modeled environment exposure; not satellite health.",
            "causal_link": (
                "Never inferred automatically. Correlation engine only summarizes environment before a reported event."
            ),
        },
        "sources": {
            "ospo_status": OSPO_STATUS,
            "ospo_messages": OSPO_MESSAGES,
            "ospo_year_archive": year_url,
            "ncei_anomaly_summary": NCEI_ANOM_SUMMARY,
            "ncei_anomaly_raw": NCEI_ANOM_RAW,
            "swpc_goes_electrons": SWPC_GOES_PRIMARY_E,
            "swpc_goes_protons": SWPC_GOES_PRIMARY_P,
            "swpc_goes_instrument_sources": SWPC_GOES_SOURCES,
            "swpc_goes_longitudes": SWPC_GOES_LONGITUDES,
            "celestrak_geo": CELESTRAK_GEO,
            "celestrak_geo_tle": CELESTRAK_GEO_TLE,
        },
        "fetch_errors": errors,
        "goes_public_status": status,
        "satellites": satellites,
        "goes_space_environment_now": {
            "electron_gt2mev": electron,
            "proton_gt10mev": proton,
            "instrument_sources_raw": source_json,
            "satellite_longitudes_raw": longitude_json,
        },
        "forecast_environment_72h": env72,
        "environment_risk": risk,
        "risk_bands": RISK_BANDS,
        "recent_public_goes_events": corr[:120],
        "association_summary": association_summary,
        "ncei_historic_anomaly_summary_top_geo_like": historic_geo_like,
        "celestrak_active_geo_sample": geo_rows,
        "geo_satellites_3d": geo_satellites_3d,
        "tle_cache": tle_cache_info,
        "ncei_legacy_anomaly_records": len(ncei_events),
        "visual_cause_legend": {
            "SURFACE_CHARGING_ESD": "#ff8c42",
            "INTERNAL_CHARGING": "#9b5de5",
            "SEE_SEU": "#ff4fa3",
            "POWER_EPS": "#ffd34e",
            "ATTITUDE_GNC": "#36d7ff",
            "COMMUNICATIONS": "#4c8cff",
            "INSTRUMENT": "#ffb14e",
            "PRODUCT_DATA": "#e6a84c",
            "UNKNOWN": "#ff5f63"
        },
    }

    # Merge event ledger so events remain available even after OSPO rotates the page.
    ledger_path = OUT_DIR / "event-ledger.json"
    old = load_json_candidates([ledger_path]) or {}
    old_events = old.get("events", []) if isinstance(old, dict) else []
    by_key = {}
    for e in old_events + corr:
        if not isinstance(e, dict):
            continue
        key = (e.get("issued_at"), e.get("title"))
        by_key[key] = e
    ledger_events = sorted(
        by_key.values(),
        key=lambda x: x.get("issued_at") or "",
        reverse=True
    )[:2500]
    ledger = {
        "schema_version": "SWIFT-GEO-SAT-EVENT-LEDGER-v0.4",
        "updated_at": result["generated_at"],
        "events": ledger_events,
        "note": "Public OSPO anomaly/outage reports; not automatically space-weather-caused.",
    }
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")

    latest = OUT_DIR / "latest.json"
    latest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    stamp = now_utc().strftime("%Y-%m-%dT%H")
    snap = HISTORY_DIR / f"{stamp}.json"
    snap.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Retain hourly snapshots for 90 days.
    cutoff = now_utc() - timedelta(days=90)
    for p in HISTORY_DIR.glob("*.json"):
        try:
            d = datetime.strptime(p.stem, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            if d < cutoff:
                p.unlink()
        except Exception:
            pass

    write_environment_snapshot(result)

    return result


if __name__ == "__main__":
    data = build()
    print(
        json.dumps({
            "generated_at": data["generated_at"],
            "satellites": len(data["satellites"]),
            "recent_public_events": len(data["recent_public_goes_events"]),
            "risk_label": data["environment_risk"]["overall_environment_risk_label"],
            "fetch_errors": len(data["fetch_errors"]),
        }, indent=2)
    )
