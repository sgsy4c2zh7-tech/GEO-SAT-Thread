#!/usr/bin/env python3
"""Train Kp-band residual AI for SWIFT-TEC.

This script keeps the existing polynomial Kp response F(Kp) intact and learns
an additional grid-cell residual correction separately for four Kp bands:

    0-4 : 0 <= Kp < 5
    5-6 : 5 <= Kp < 7
    7-8 : 7 <= Kp < 9
    9   : Kp >= 9

The residual AI is deliberately simple and robust: for each month, grid cell
and Kp band it learns the systematic forecast residual left after the existing
Kp polynomial. The learned residual is shrunk toward zero, blended with the
previous saved value, and clipped.

NOAA model residual target:
    obs(t) - [obs(t-24h) - F(KpB) + F(Kp_actual(t))]

ISEE model residual target (matches current no-Base forecast):
    obs(t) - [weighted same-UTC-time mean of prior days + F(Kp_actual(t))]

Outputs are month-split to keep Git history small:
    docs/data/ai/kp_band_residual/MM.json
    docs/data/ai/isee_japan/kp_band_residual/MM.json

The runtime JS should add this residual to the existing polynomial Kp term.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from bisect import bisect_left
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

UTC = timezone.utc

KP_URLS = [
    os.environ.get("SWIFTTEC_KP_ACTUAL_1M_URL", "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"),
    os.environ.get("SWIFTTEC_KP_ACTUAL_URL", "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"),
]

TRAIN_DAYS = max(7, int(os.environ.get("SWIFTTEC_BAND_AI_TRAIN_DAYS", "30")))
PAIR_HOURS = float(os.environ.get("SWIFTTEC_BAND_AI_PAIR_HOURS", "24"))
PAIR_TOLERANCE_MIN = max(5, int(os.environ.get("SWIFTTEC_BAND_AI_PAIR_TOLERANCE_MIN", "20")))
ISEE_MEAN_DAYS = max(1, int(os.environ.get("SWIFTTEC_BAND_AI_ISEE_MEAN_DAYS", "10")))
ISEE_STEP_MIN = max(5, int(os.environ.get("SWIFTTEC_BAND_AI_ISEE_STEP_MIN", "30")))
BLEND_ALPHA = float(os.environ.get("SWIFTTEC_BAND_AI_BLEND_ALPHA", "0.25"))

BANDS = OrderedDict([
    ("0-4", {"lo": 0.0, "hi": 5.0, "min_learn": 12, "min_apply": 8, "shrink": 12.0, "clip": 8.0}),
    ("5-6", {"lo": 5.0, "hi": 7.0, "min_learn": 8,  "min_apply": 6, "shrink": 8.0,  "clip": 12.0}),
    ("7-8", {"lo": 7.0, "hi": 9.0, "min_learn": 4,  "min_apply": 3, "shrink": 6.0,  "clip": 16.0}),
    ("9",   {"lo": 9.0, "hi": 99.0,"min_learn": 2,  "min_apply": 2, "shrink": 6.0,  "clip": 20.0}),
])


@dataclass(frozen=True)
class SourceCfg:
    name: str
    tec_root: Path
    ai_root: Path
    kp_archive: Path
    model: str


SOURCES = {
    "noaa": SourceCfg(
        name="noaa",
        tec_root=Path("docs/data/tec"),
        ai_root=Path("docs/data/ai"),
        kp_archive=Path("docs/data/ai/kp_actual_archive_recent.json"),
        model="prev24h_base",
    ),
    "isee": SourceCfg(
        name="isee",
        tec_root=Path("docs/data/isee_tec"),
        ai_root=Path("docs/data/ai/isee_japan"),
        kp_archive=Path("docs/data/ai/isee_japan/kp_actual_archive.json"),
        model="same_utc_mean",
    ),
}


def iso(t: datetime) -> str:
    return t.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_maybe_gz(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return read_json(path)


def http_json(url: str):
    req = Request(url, headers={"User-Agent": "SWIFT-TEC/Kp-band-residual-ai-1.0"})
    with urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8", "replace"))


def parse_kp_rows(obj) -> list[tuple[datetime, float]]:
    out = []
    if not isinstance(obj, list):
        return out
    for row in obj:
        if isinstance(row, list) and len(row) >= 2:
            t = parse_time(row[0])
            raw = row[1]
        elif isinstance(row, dict):
            t = parse_time(row.get("time_tag") or row.get("time") or row.get("t") or row.get("time_utc"))
            raw = row.get("kp_index")
            if raw is None:
                raw = row.get("kp")
            if raw is None:
                raw = row.get("Kp")
        else:
            continue
        try:
            kp = float(raw)
        except Exception:
            continue
        if t and math.isfinite(kp):
            out.append((t, kp))
    return out


def load_kp_rows(cfg: SourceCfg) -> list[tuple[datetime, float]]:
    merged: dict[str, tuple[datetime, float]] = {}

    if cfg.kp_archive.exists():
        try:
            doc = read_json(cfg.kp_archive)
            for row in doc.get("rows", []):
                t = parse_time(row.get("time_utc") or row.get("time"))
                try:
                    kp = float(row.get("kp"))
                except Exception:
                    continue
                if t and math.isfinite(kp):
                    merged[iso(t)] = (t, kp)
        except Exception as exc:
            print(f"WARN {cfg.name}: Kp archive read failed: {exc}")

    for url in KP_URLS:
        try:
            rows = parse_kp_rows(http_json(url))
            for t, kp in rows:
                merged[iso(t)] = (t, kp)
            if rows:
                print(f"{cfg.name}: fetched {len(rows)} observed Kp rows from {url}")
        except Exception as exc:
            print(f"WARN {cfg.name}: Kp fetch failed {url}: {exc}")

    rows = sorted(merged.values(), key=lambda x: x[0])
    print(f"{cfg.name}: total observed Kp rows available={len(rows)}")
    return rows


def nearest_kp(t: datetime, rows: list[tuple[datetime, float]], max_hours: float = 2.0) -> float | None:
    if not rows:
        return None
    times = [r[0] for r in rows]
    pos = bisect_left(times, t)
    cand = []
    if pos < len(rows):
        cand.append(rows[pos])
    if pos > 0:
        cand.append(rows[pos - 1])
    if not cand:
        return None
    rt, kp = min(cand, key=lambda x: abs((x[0] - t).total_seconds()))
    if abs((rt - t).total_seconds()) > max_hours * 3600:
        return None
    return float(kp)


def band_label(kp: float) -> str:
    x = float(kp)
    if x < 5.0:
        return "0-4"
    if x < 7.0:
        return "5-6"
    if x < 9.0:
        return "7-8"
    return "9"


def load_frame_metas(cfg: SourceCfg) -> list[dict]:
    idx_path = cfg.tec_root / "index.json"
    if not idx_path.exists():
        return []
    idx = read_json(idx_path)
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=TRAIN_DAYS + 15)
    metas = []
    for f in idx.get("frames") or []:
        t = parse_time(f.get("time_utc") or f.get("time"))
        rel = f.get("file") or f.get("path")
        if not t or not rel or t < cutoff:
            continue
        path = cfg.tec_root / rel
        if path.exists():
            metas.append({"time": t, "path": path, "file": rel})
    metas.sort(key=lambda x: x["time"])
    return metas


def nearest_meta(metas: list[dict], target: datetime, tolerance_min: int = PAIR_TOLERANCE_MIN):
    if not metas:
        return None
    times = [m["time"] for m in metas]
    pos = bisect_left(times, target)
    cand = []
    if pos < len(metas):
        cand.append(metas[pos])
    if pos > 0:
        cand.append(metas[pos - 1])
    if not cand:
        return None
    best = min(cand, key=lambda x: abs((x["time"] - target).total_seconds()))
    if abs((best["time"] - target).total_seconds()) > tolerance_min * 60:
        return None
    return best


class FrameCache:
    def __init__(self, max_items: int = 240):
        self.max_items = max_items
        self._d: OrderedDict[str, tuple[dict, np.ndarray]] = OrderedDict()

    def get(self, meta: dict) -> tuple[dict, np.ndarray]:
        key = str(meta["path"])
        if key in self._d:
            value = self._d.pop(key)
            self._d[key] = value
            return value
        doc = read_json_maybe_gz(meta["path"])
        grid = np.asarray(doc.get("grid") or [], dtype=float)
        if grid.ndim != 2:
            raise ValueError(f"bad grid shape in {meta['path']}")
        value = (doc, grid)
        self._d[key] = value
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)
        return value


def load_coeff_doc(cfg: SourceCfg) -> dict:
    p = cfg.ai_root / "kp_grid_coefficients.json"
    if not p.exists():
        raise RuntimeError(f"{cfg.name}: missing {p}")
    return read_json(p)


def coeff_arrays(coeff_doc: dict, month: int, shape: tuple[int, int]):
    root = coeff_doc.get("coefficients_grid") or coeff_doc.get("grid_coefficients") or {}
    mg = root.get(str(month)) or root.get(month) or {}
    arrays = []
    for name in ("k0", "k1", "k2", "k3"):
        try:
            a = np.asarray(mg.get(name), dtype=float)
            if a.shape != shape:
                a = np.zeros(shape, dtype=float)
        except Exception:
            a = np.zeros(shape, dtype=float)
        arrays.append(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0))
    return arrays


def F_grid(coeff_doc: dict, month: int, kp: float, shape: tuple[int, int]):
    k0, k1, k2, k3 = coeff_arrays(coeff_doc, month, shape)
    x = float(kp) - 3.0
    out = k0 + k1 * x + k2 * x * x + k3 * x * x * x
    clip = float(coeff_doc.get("correction_clip_tecu") or 20.0)
    clip = max(1.0, min(60.0, clip))
    return np.clip(np.nan_to_num(out, nan=0.0), -clip, clip)


class Acc:
    def __init__(self, shape: tuple[int, int]):
        self.sum = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.int32)

    def add(self, residual: np.ndarray, valid: np.ndarray):
        v = np.asarray(residual, dtype=float)
        mask = np.asarray(valid, dtype=bool) & np.isfinite(v)
        if not np.any(mask):
            return
        self.sum[mask] += v[mask]
        self.count[mask] += 1


def weight(rank: int) -> float:
    return max(0.1, 1.0 - 0.1 * rank)


def train_noaa(cfg: SourceCfg, metas: list[dict], kp_rows, coeff_doc: dict):
    if not metas:
        return {}, None
    cache = FrameCache(max_items=160)
    first_doc, first_grid = cache.get(metas[0])
    shape = first_grid.shape
    lat_arr = first_doc.get("lat_arr") or first_doc.get("latArr") or []
    lon_arr = first_doc.get("lon_arr") or first_doc.get("lonArr") or []

    now = datetime.now(UTC)
    cutoff = now - timedelta(days=TRAIN_DAYS)
    acc: dict[int, dict[str, Acc]] = defaultdict(lambda: {b: Acc(shape) for b in BANDS})
    used_pairs = 0

    for mf in metas:
        if mf["time"] < cutoff:
            continue
        mb = nearest_meta(metas, mf["time"] - timedelta(hours=PAIR_HOURS))
        if not mb:
            continue
        kp_f = nearest_kp(mf["time"], kp_rows)
        kp_b = nearest_kp(mb["time"], kp_rows)
        if kp_f is None or kp_b is None:
            continue
        try:
            _, gf = cache.get(mf)
            _, gb = cache.get(mb)
        except Exception:
            continue
        if gf.shape != shape or gb.shape != shape:
            continue

        month = mf["time"].month
        pred = gb - F_grid(coeff_doc, month, kp_b, shape) + F_grid(coeff_doc, month, kp_f, shape)
        residual = gf - pred
        valid = np.isfinite(gf) & np.isfinite(gb) & np.isfinite(pred)
        acc[month][band_label(kp_f)].add(residual, valid)
        used_pairs += 1

    return acc, {
        "shape": shape,
        "lat_arr": list(lat_arr),
        "lon_arr": list(lon_arr),
        "used_pairs": used_pairs,
    }


def train_isee(cfg: SourceCfg, metas: list[dict], kp_rows, coeff_doc: dict):
    # Match the current production ISEE forecast step: 30-minute targets.
    metas = [m for m in metas if m["time"].minute % ISEE_STEP_MIN == 0]
    if not metas:
        return {}, None

    cache = FrameCache(max_items=220)
    first_doc, first_grid = cache.get(metas[0])
    shape = first_grid.shape
    lat_arr = first_doc.get("lat_arr") or first_doc.get("latArr") or []
    lon_arr = first_doc.get("lon_arr") or first_doc.get("lonArr") or []

    by_slot: dict[str, list[dict]] = defaultdict(list)
    for m in metas:
        by_slot[m["time"].strftime("%H%M")].append(m)
    for slot in by_slot:
        by_slot[slot].sort(key=lambda x: x["time"])

    now = datetime.now(UTC)
    cutoff = now - timedelta(days=TRAIN_DAYS)
    acc: dict[int, dict[str, Acc]] = defaultdict(lambda: {b: Acc(shape) for b in BANDS})
    used_pairs = 0

    for slot, seq in by_slot.items():
        loaded: list[tuple[dict, np.ndarray]] = []
        for m in seq:
            try:
                _, g = cache.get(m)
            except Exception:
                continue
            if g.shape == shape:
                loaded.append((m, g))

        for idx in range(1, len(loaded)):
            mf, gf = loaded[idx]
            if mf["time"] < cutoff:
                continue
            kp_f = nearest_kp(mf["time"], kp_rows)
            if kp_f is None:
                continue

            hist = loaded[max(0, idx - ISEE_MEAN_DAYS):idx]
            if not hist:
                continue

            num = np.zeros(shape, dtype=float)
            den = np.zeros(shape, dtype=float)
            for rank, (_, gh) in enumerate(reversed(hist)):
                w = weight(rank)
                mask = np.isfinite(gh)
                num[mask] += gh[mask] * w
                den[mask] += w
            with np.errstate(invalid="ignore", divide="ignore"):
                base = np.where(den > 0, num / den, np.nan)

            month = mf["time"].month
            pred = base + F_grid(coeff_doc, month, kp_f, shape)
            residual = gf - pred
            valid = np.isfinite(gf) & np.isfinite(base) & np.isfinite(pred)
            acc[month][band_label(kp_f)].add(residual, valid)
            used_pairs += 1

    return acc, {
        "shape": shape,
        "lat_arr": list(lat_arr),
        "lon_arr": list(lon_arr),
        "used_pairs": used_pairs,
    }


def month_path(cfg: SourceCfg, month: int) -> Path:
    return cfg.ai_root / "kp_band_residual" / f"{month:02d}.json"


def load_old_month(cfg: SourceCfg, month: int) -> dict:
    p = month_path(cfg, month)
    if not p.exists():
        return {}
    try:
        return read_json(p)
    except Exception:
        return {}


def old_arrays(old: dict, band: str, shape: tuple[int, int]):
    b = (old.get("bands") or {}).get(band) or {}
    try:
        r = np.asarray(b.get("residual"), dtype=float)
        if r.shape != shape:
            r = np.zeros(shape, dtype=float)
    except Exception:
        r = np.zeros(shape, dtype=float)
    try:
        n = np.asarray(b.get("sample_count"), dtype=int)
        if n.shape != shape:
            n = np.zeros(shape, dtype=int)
    except Exception:
        n = np.zeros(shape, dtype=int)
    return np.nan_to_num(r, nan=0.0), np.maximum(0, n)


def round_grid(a: np.ndarray, digits: int = 4):
    return np.round(a.astype(float), digits).tolist()


def int_grid(a: np.ndarray):
    return a.astype(int).tolist()


def write_month_docs(cfg: SourceCfg, acc_by_month, meta):
    if not meta:
        print(f"{cfg.name}: no usable TEC frames; no band residual files written")
        return 0
    shape = tuple(meta["shape"])
    out_dir = cfg.ai_root / "kp_band_residual"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    written = 0

    for month, by_band in sorted(acc_by_month.items()):
        # Do not rewrite a month if absolutely no new samples were seen.
        total_new = sum(int(x.count.sum()) for x in by_band.values())
        if total_new <= 0:
            continue

        old = load_old_month(cfg, month)
        bands_out = {}
        for band, params in BANDS.items():
            a = by_band[band]
            old_r, old_n = old_arrays(old, band, shape)
            new_n = a.count.astype(int)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_resid = np.where(new_n > 0, a.sum / np.maximum(new_n, 1), 0.0)

            shrink = float(params["shrink"])
            shrunk = mean_resid * (new_n / (new_n + shrink))
            shrunk = np.clip(shrunk, -float(params["clip"]), float(params["clip"]))

            update_mask = new_n >= int(params["min_learn"])
            first_mask = (old_n <= 0) & (new_n > 0)

            result = old_r.copy()
            # If no old value exists yet, allow even sub-minimum samples to seed a
            # heavily-shrunk value. Runtime min_apply still prevents premature use.
            result[first_mask] = shrunk[first_mask]
            result[update_mask] = (
                (1.0 - BLEND_ALPHA) * old_r[update_mask]
                + BLEND_ALPHA * shrunk[update_mask]
            )
            result = np.clip(result, -float(params["clip"]), float(params["clip"]))

            total_n = np.minimum(old_n.astype(np.int64) + new_n.astype(np.int64), 2_000_000)
            bands_out[band] = {
                "residual": round_grid(result, 4),
                "sample_count": int_grid(total_n),
                "new_sample_count": int_grid(new_n),
                "min_apply_samples": int(params["min_apply"]),
                "min_learn_samples": int(params["min_learn"]),
                "shrink_strength": shrink,
                "clip_tecu": float(params["clip"]),
            }

        doc = {
            "version": "swifttec-kp-band-residual-grid-v1",
            "source": cfg.name,
            "month": int(month),
            "updated_utc": iso(now),
            "model": (
                "existing polynomial F(Kp) + target-Kp-band residual AI; "
                "bands: 0-4, 5-6, 7-8, 9"
            ),
            "band_rule": {
                "0-4": "0 <= Kp < 5",
                "5-6": "5 <= Kp < 7",
                "7-8": "7 <= Kp < 9",
                "9": "Kp >= 9",
            },
            "fallback_order": {
                "9": ["9", "7-8", "5-6", "0-4"],
                "7-8": ["7-8", "5-6", "0-4"],
                "5-6": ["5-6", "0-4"],
                "0-4": ["0-4"],
            },
            "blend_alpha": BLEND_ALPHA,
            "train_days": TRAIN_DAYS,
            "lat_arr": meta["lat_arr"],
            "lon_arr": meta["lon_arr"],
            "n_lat": shape[0],
            "n_lon": shape[1],
            "training_cases": int(meta.get("used_pairs") or 0),
            "new_grid_samples": int(total_new),
            "bands": bands_out,
        }

        p = month_path(cfg, month)
        p.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"{cfg.name}: wrote {p} new_grid_samples={total_new}")
        written += 1

    return written


def train_one(cfg: SourceCfg) -> int:
    print(f"=== Train band residual AI: {cfg.name} ===")
    metas = load_frame_metas(cfg)
    print(f"{cfg.name}: TEC frame metadata={len(metas)}")
    if len(metas) < 2:
        print(f"WARN {cfg.name}: not enough TEC frames; skipping")
        return 0

    kp_rows = load_kp_rows(cfg)
    if not kp_rows:
        print(f"WARN {cfg.name}: no Kp rows; skipping")
        return 0

    coeff_doc = load_coeff_doc(cfg)
    if cfg.model == "prev24h_base":
        acc, meta = train_noaa(cfg, metas, kp_rows, coeff_doc)
    else:
        acc, meta = train_isee(cfg, metas, kp_rows, coeff_doc)

    if meta:
        print(f"{cfg.name}: training cases={meta.get('used_pairs')} grid={meta.get('shape')}")
    return write_month_docs(cfg, acc, meta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["noaa", "isee", "both"], default="both")
    args = ap.parse_args()

    selected = ["noaa", "isee"] if args.source == "both" else [args.source]
    total = 0
    for name in selected:
        try:
            total += train_one(SOURCES[name])
        except Exception as exc:
            print(f"ERROR {name}: {exc}")
            # Keep the other source running. This workflow is an enhancement,
            # not a reason to block the core TEC updater.
    print(f"Band residual AI complete: month files written={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
