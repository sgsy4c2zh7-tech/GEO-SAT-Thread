#!/usr/bin/env python3
"""Build SWIFT coronal-hole features from SDO/AIA + HMI via Helioviewer.

Operational purpose
-------------------
This is a lightweight, transparent, CHIMERA-inspired detector for SWIFT.
It is NOT a reproduction of the full CHIMERA implementation or a PFSS/WSA solver.

Inputs
------
Helioviewer screenshots closest to now:
- SDO/AIA 171 A
- SDO/AIA 193 A
- SDO/AIA 211 A
- SDO/HMI line-of-sight magnetogram

Method
------
1. Robustly normalize the three EUV channels inside the solar disk.
2. Identify multi-thermal dark regions using the 193/211 intensities and
   193:171 / 211:171 ratios (CHIMERA-inspired).
3. Clean the mask morphologically and retain sufficiently large connected regions.
4. Use the HMI image as a *polarity/unipolarity proxy* to reject obviously mixed
   dark filaments. Because Helioviewer screenshots are display products rather
   than calibrated HMI FITS magnetograms, this polarity value is explicitly
   labelled a proxy.
5. Convert each region to area, disk position, approximate latitude/CMD, and an
   Earth-facing score.
6. Produce a conservative HSS arrival window and residual speed prior. The Wind
   AI uses this as a feature/residual correction, not as a blind additive boost.

Outputs
-------
docs/data/coronal-holes/latest.json
docs/data/coronal-holes/history.json

Scientific references
---------------------
- Garton, Gallagher & Murray (2018), CHIMERA:
  https://arxiv.org/abs/1711.11476
- Helioviewer API:
  https://api.helioviewer.org/docs/v1/
- Riley, Linker & Arge (2015), DCHB:
  https://doi.org/10.1002/2014SW001144
"""
from __future__ import annotations

import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "coronal-holes"
OUT.mkdir(parents=True, exist_ok=True)
LATEST = OUT / "latest.json"
HISTORY = OUT / "history.json"

HV = "https://api.helioviewer.org/v1/takeScreenshot/"
IMAGE_SCALE = 2.5          # arcsec / px
SIZE = 1024
SOLAR_RADIUS_ARCSEC = 960.0
SOLAR_RADIUS_PX = SOLAR_RADIUS_ARCSEC / IMAGE_SCALE
RETENTION_DAYS = 30

CHANNELS = {
    "aia171": "[SDO,AIA,AIA,171,1,100]",
    "aia193": "[SDO,AIA,AIA,193,1,100]",
    "aia211": "[SDO,AIA,AIA,211,1,100]",
    "hmi": "[SDO,HMI,HMI,magnetogram,1,100]",
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_layer(date: datetime, layer: str) -> np.ndarray:
    params = {
        "date": iso(date),
        "imageScale": IMAGE_SCALE,
        "layers": layer,
        "events": "",
        "eventLabels": "false",
        "x0": 0,
        "y0": 0,
        "width": SIZE,
        "height": SIZE,
        "display": "true",
        "watermark": "false",
    }
    url = HV + "?" + urlencode(params)
    r = requests.get(
        url,
        timeout=90,
        headers={"User-Agent": "SWIFT-Coronal-Hole-AI/1.0 (+GitHub Actions)"},
    )
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def robust_norm(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = a[mask]
    if vals.size == 0:
        return a
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo + 1e-6:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def disk_geometry(shape: tuple[int, int]):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    x = (xx - cx) / SOLAR_RADIUS_PX
    y = -(yy - cy) / SOLAR_RADIUS_PX
    rr2 = x*x + y*y
    disk = rr2 <= 1.0
    mu = np.sqrt(np.clip(1.0 - rr2, 0, 1))
    return x, y, disk, mu


def hmi_unipolarity_proxy(hmi: np.ndarray, region: np.ndarray, disk: np.ndarray) -> tuple[float, float]:
    vals_all = hmi[disk]
    if vals_all.size < 100:
        return 0.0, 0.0
    mid = float(np.median(vals_all))
    scale = float(np.percentile(np.abs(vals_all - mid), 90)) + 1e-6
    v = (hmi[region] - mid) / scale
    if v.size < 10:
        return 0.0, 0.0
    strong = v[np.abs(v) > 0.12]
    if strong.size < 8:
        return 0.0, float(np.mean(v))
    pos = np.sum(strong > 0)
    neg = np.sum(strong < 0)
    unip = abs(pos-neg) / max(1, pos+neg)
    polarity = float(np.mean(np.sign(strong)))
    return float(unip), polarity


def region_to_features(label_id: int, labels: np.ndarray, x: np.ndarray, y: np.ndarray,
                       mu: np.ndarray, hmi: np.ndarray, disk: np.ndarray) -> dict[str, Any] | None:
    r = labels == label_id
    pix = int(np.sum(r))
    if pix < 180:
        return None

    weights = 1.0 / np.clip(mu[r], 0.25, 1.0)
    x0 = float(np.average(x[r], weights=weights))
    y0 = float(np.average(y[r], weights=weights))
    rho = math.sqrt(x0*x0 + y0*y0)
    if rho > 0.96:
        return None

    lat = math.degrees(math.asin(clamp(y0, -1, 1)))
    # Approximate central-meridian distance from projected x coordinate.
    coslat = max(0.25, math.cos(math.radians(lat)))
    cmd = math.degrees(math.asin(clamp(x0/coslat, -1, 1)))

    # Approximate deprojected disk area fraction.
    area_px_corr = float(np.sum(1.0/np.clip(mu[r], 0.25, 1.0)))
    disk_area_px = math.pi * SOLAR_RADIUS_PX**2
    area_fraction = area_px_corr / disk_area_px

    unip, polarity = hmi_unipolarity_proxy(hmi, r, disk)

    lat_weight = math.cos(math.radians(lat))**1.7
    center_weight = math.exp(-(cmd/38.0)**2)
    magnetic_weight = 0.55 + 0.45*unip
    # Score is dimensionless and intentionally not interpreted as probability.
    earth_score = area_fraction * lat_weight * center_weight * magnetic_weight

    # Width proxy and DCHB proxy. With image segmentation only, we do not have
    # field-line mapping; this is explicitly a geometric proxy.
    area_deg2_proxy = max(0.1, area_fraction * 4*math.pi * (180/math.pi)**2)
    width_deg = min(90.0, 2.0*math.sqrt(area_deg2_proxy/math.pi))
    dchb_proxy_deg = max(0.5, 0.23*width_deg)

    # Empirical CH/HSS speed prior: deep/large, low-latitude, unipolar holes tend
    # to produce faster wind. It is deliberately conservative because the Wind AI
    # later learns the residual against observations.
    source_strength = clamp(
        0.38*(1-math.exp(-area_fraction/0.018)) +
        0.22*lat_weight +
        0.20*center_weight +
        0.20*unip,
        0, 1
    )
    predicted_peak_speed = 350.0 + 360.0*source_strength
    predicted_delta_v = max(0.0, predicted_peak_speed - 390.0)

    return {
        "id": f"CH-{label_id}",
        "pixel_count": pix,
        "area_fraction_disk_deprojected": round(area_fraction, 6),
        "latitude_deg": round(lat, 2),
        "central_meridian_deg": round(cmd, 2),
        "width_deg_proxy": round(width_deg, 2),
        "dchb_proxy_deg": round(dchb_proxy_deg, 2),
        "hmi_unipolarity_proxy": round(unip, 3),
        "hmi_polarity_proxy": round(polarity, 3),
        "earth_facing_score": round(earth_score, 6),
        "source_strength": round(source_strength, 3),
        "predicted_peak_speed_km_s_prior": round(predicted_peak_speed, 1),
        "predicted_delta_v_km_s_prior": round(predicted_delta_v, 1),
    }


def impact_from_feature(f: dict[str, Any], t0: datetime) -> dict[str, Any]:
    v = float(f["predicted_peak_speed_km_s_prior"])
    cmd = float(f["central_meridian_deg"])
    # Ballistic propagation from ~0.1 AU to 1 AU plus a modest connection-time
    # correction for source longitude. Sign convention here follows image x:
    # westward sources (positive projected x) tend to connect sooner.
    au_km = 149_597_870.7
    travel_h = 0.90*au_km / max(300.0, v) / 3600.0
    connection_h = clamp(-cmd/13.2*24.0, -30.0, 42.0)
    arrival = t0 + timedelta(hours=travel_h + connection_h)

    confidence = clamp(
        0.28 +
        2.8*float(f["earth_facing_score"]) +
        0.22*float(f["hmi_unipolarity_proxy"]),
        0.18, 0.88
    )
    spread_h = 15.0 + (1-confidence)*28.0

    return {
        "source_id": f["id"],
        "arrival_time": iso(arrival),
        "window_start": iso(arrival-timedelta(hours=spread_h)),
        "window_end": iso(arrival+timedelta(hours=spread_h)),
        "confidence": round(confidence, 3),
        "predicted_peak_speed_km_s_prior": f["predicted_peak_speed_km_s_prior"],
        "predicted_delta_v_km_s_prior": f["predicted_delta_v_km_s_prior"],
        "earth_facing_score": f["earth_facing_score"],
        "latitude_deg": f["latitude_deg"],
        "central_meridian_deg": f["central_meridian_deg"],
        "area_fraction_disk_deprojected": f["area_fraction_disk_deprojected"],
        "hmi_unipolarity_proxy": f["hmi_unipolarity_proxy"],
        "method": "CHIMERA-inspired EUV segmentation + HMI polarity proxy + ballistic HSS prior",
    }


def main() -> None:
    now = now_utc()
    errors = []
    imgs: dict[str, np.ndarray] = {}
    for name, layer in CHANNELS.items():
        try:
            imgs[name] = fetch_layer(now, layer)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    required = {"aia171", "aia193", "aia211"}
    if not required.issubset(imgs):
        payload = {
            "updated_at": iso(now),
            "status": "fetch_failed",
            "errors": errors,
            "coronal_holes": [],
            "earth_impacts": [],
        }
        save(LATEST, payload)
        print(json.dumps(payload, indent=2))
        return

    shape = imgs["aia193"].shape
    x, y, disk, mu = disk_geometry(shape)
    n171 = robust_norm(imgs["aia171"], disk)
    n193 = robust_norm(imgs["aia193"], disk)
    n211 = robust_norm(imgs["aia211"], disk)
    hmi = imgs.get("hmi", np.full(shape, 0.5, dtype=np.float32))

    # CHIMERA-inspired multi-thermal dark segmentation.
    q193 = float(np.quantile(n193[disk], 0.31))
    q211 = float(np.quantile(n211[disk], 0.34))
    ratio193 = n193 / np.maximum(n171, 0.06)
    ratio211 = n211 / np.maximum(n171, 0.06)

    candidate = (
        disk &
        (mu > 0.28) &
        (n193 < min(0.48, q193*1.08)) &
        (n211 < min(0.50, q211*1.10)) &
        (ratio193 < 0.92) &
        (ratio211 < 1.00)
    )

    candidate = ndimage.binary_opening(candidate, structure=np.ones((3,3)), iterations=1)
    candidate = ndimage.binary_closing(candidate, structure=np.ones((5,5)), iterations=2)
    candidate = ndimage.binary_fill_holes(candidate)
    labels, nlab = ndimage.label(candidate)

    features = []
    for i in range(1, nlab+1):
        f = region_to_features(i, labels, x, y, mu, hmi, disk)
        if not f:
            continue
        # Image-dark regions with extremely mixed magnetic proxy are likely
        # filaments/quiet structures; retain only if very large.
        if f["hmi_unipolarity_proxy"] < 0.10 and f["area_fraction_disk_deprojected"] < 0.02:
            continue
        features.append(f)

    features.sort(key=lambda r: r["earth_facing_score"], reverse=True)
    impacts = [impact_from_feature(f, now) for f in features if abs(f["latitude_deg"]) <= 45]
    impacts.sort(key=lambda r: r["arrival_time"])

    payload = {
        "updated_at": iso(now),
        "model": "SWIFT-CH-v1.0-CHIMERA-inspired",
        "status": "ok",
        "methodology": {
            "euv": "AIA 171/193/211 robust multi-thermal dark segmentation",
            "magnetic": "HMI screenshot polarity/unipolarity proxy (not calibrated FITS field strength)",
            "forecast_use": "residual feature for SWIFT Wind AI; avoids blind double-counting with ENLIL/WSA",
            "helioviewer_api": "https://api.helioviewer.org/docs/v1/",
            "reference_chimera": "https://arxiv.org/abs/1711.11476",
            "reference_dchb": "https://doi.org/10.1002/2014SW001144",
        },
        "thresholds": {"q193": round(q193,4), "q211": round(q211,4)},
        "errors": errors,
        "coronal_holes": features,
        "earth_impacts": impacts,
    }
    save(LATEST, payload)

    hist = load(HISTORY, {"items":[]}) or {"items":[]}
    items = [x for x in hist.get("items",[]) if isinstance(x,dict)]
    # Store one compact snapshot per >= 2 h. history.json keeps only the latest 30 days.
    last_t = None
    if items:
        try:
            s = str(items[-1].get("time","")).replace("Z","+00:00")
            last_t = datetime.fromisoformat(s)
        except Exception:
            pass
    if last_t is None or now-last_t >= timedelta(hours=2):
        items.append({
            "time": iso(now),
            "model": payload["model"],
            "status": payload["status"],
            "coronal_holes": features,
            "earth_impacts": impacts,
        })
    # Keep only the latest 30 days of coronal-hole history.
    # A small future-time tolerance protects the file from malformed timestamps.
    cutoff = now - timedelta(days=RETENTION_DAYS)
    future_limit = now + timedelta(hours=6)

    kept = []
    for xh in items:
        try:
            t = datetime.fromisoformat(str(xh.get("time", "")).replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            t = t.astimezone(timezone.utc)

            if cutoff <= t <= future_limit:
                kept.append(xh)
        except Exception:
            continue

    # Ensure chronological order and prevent accidental unlimited growth.
    kept.sort(key=lambda x: str(x.get("time", "")))

    save(
        HISTORY,
        {
            "updated_at": iso(now),
            "retention_days": RETENTION_DAYS,
            "snapshot_interval_hours": 2,
            "item_count": len(kept),
            "items": kept,
        },
    )
    print(json.dumps({
        "model": payload["model"],
        "regions": len(features),
        "earth_impacts": len(impacts),
        "top": impacts[0] if impacts else None,
        "errors": errors,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
