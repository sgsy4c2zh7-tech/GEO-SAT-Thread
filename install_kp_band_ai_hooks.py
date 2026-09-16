#!/usr/bin/env python3
"""Patch the current SWIFT-TEC browser code to apply Kp-band residual AI.

Run once from the repository root after adding the trainer. The patch is
idempotent and targets the current v8.x code structure.

Files modified:
  docs/js/swifttec_isee_japan.js
  docs/js/swifttec_v4_archive_dop.js
"""
from __future__ import annotations

from pathlib import Path

ISEE = Path("docs/js/swifttec_isee_japan.js")
V4 = Path("docs/js/swifttec_v4_archive_dop.js")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly 1 match, got {count}")
    return text.replace(old, new, 1)


def patch_isee(text: str) -> str:
    text = replace_once(
        text,
        '  const JAPAN_GRID_COEFF_URL = "data/ai/isee_japan/kp_grid_coefficients.json";\n',
        '  const JAPAN_GRID_COEFF_URL = "data/ai/isee_japan/kp_grid_coefficients.json";\n'
        '  const JAPAN_BAND_RESIDUAL_DIR = "data/ai/isee_japan/kp_band_residual/";\n',
        "ISEE const",
    )
    text = replace_once(
        text,
        '  let gridCoeffCache = null;\n',
        '  let gridCoeffCache = null;\n  const bandResidualCache = new Map();\n',
        "ISEE cache",
    )

    anchor = '''  async function loadGridCoefficients() {
    gridCoeffCache = await getJson(JAPAN_GRID_COEFF_URL);
    return gridCoeffCache;
  }
'''
    addition = anchor + r'''
  function kpBandLabel(kp) {
    const x = Number(kp);
    if (!Number.isFinite(x) || x < 5) return "0-4";
    if (x < 7) return "5-6";
    if (x < 9) return "7-8";
    return "9";
  }

  function kpBandFallback(label) {
    if (label === "9") return ["9", "7-8", "5-6", "0-4"];
    if (label === "7-8") return ["7-8", "5-6", "0-4"];
    if (label === "5-6") return ["5-6", "0-4"];
    return ["0-4"];
  }

  async function loadBandResidualMonth(month) {
    const m = Number(month);
    if (bandResidualCache.has(m)) return bandResidualCache.get(m);
    const url = JAPAN_BAND_RESIDUAL_DIR + String(m).padStart(2, "0") + ".json";
    try {
      const doc = await getJson(url);
      bandResidualCache.set(m, doc);
      return doc;
    } catch (e) {
      console.warn("ISEE band residual not available:", url, e?.message || e);
      bandResidualCache.set(m, null);
      return null;
    }
  }

  function bandResidualAt(doc, kp, i, j) {
    if (!doc) return 0;
    const bands = doc.bands || {};
    const label = kpBandLabel(kp);
    for (const bname of kpBandFallback(label)) {
      const b = bands[bname];
      if (!b) continue;
      const n = Number(b.sample_count?.[i]?.[j] ?? 0);
      const minApply = Number(b.min_apply_samples ?? 1);
      const r = Number(b.residual?.[i]?.[j] ?? 0);
      if (Number.isFinite(n) && n >= minApply && Number.isFinite(r)) return r;
    }
    return 0;
  }
'''
    text = replace_once(text, anchor, addition, "ISEE helpers")

    old = '''    const [meanIndex, coeffDoc] = await Promise.all([
      loadMeanIndex(),
      loadGridCoefficients(),
    ]);
'''
    new = '''    const [meanIndex, coeffDoc] = await Promise.all([
      loadMeanIndex(),
      loadGridCoefficients(),
    ]);
'''
    # Keep existing promise block; load month-split band docs after start/end are known.
    if old not in text and new not in text:
        raise RuntimeError("ISEE Promise block not found")

    old2 = '''    const startUtc = roundedNowUtc30();
    const kpSeries = makeForecastKpSeries(startUtc);
    const nSteps = Math.round(FORECAST_HOURS * 60 / FORECAST_STEP_MIN);
'''
    new2 = '''    const startUtc = roundedNowUtc30();
    const kpSeries = makeForecastKpSeries(startUtc);
    const nSteps = Math.round(FORECAST_HOURS * 60 / FORECAST_STEP_MIN);

    // Month-split residual files keep Git history small. A 96h forecast can
    // cross at most one month boundary, so preload start/end months only.
    const endUtcForBand = new Date(startUtc.getTime() + FORECAST_HOURS * 3600000);
    const bandDocs = {};
    for (const m of new Set([startUtc.getUTCMonth() + 1, endUtcForBand.getUTCMonth() + 1])) {
      bandDocs[m] = await loadBandResidualMonth(m);
    }
'''
    text = replace_once(text, old2, new2, "ISEE preload band months")

    old3 = '''          let kpTerm = useAi ? F(cf, kpF) : 0;
          if (!Number.isFinite(kpTerm)) kpTerm = 0;
          kpTerm = Math.max(-clip, Math.min(clip, kpTerm));
'''
    new3 = '''          let kpTerm = useAi ? F(cf, kpF) : 0;
          if (useAi) {
            const bandDoc = bandDocs[t.getUTCMonth() + 1] || null;
            kpTerm += bandResidualAt(bandDoc, kpF, i, j);
          }
          if (!Number.isFinite(kpTerm)) kpTerm = 0;
          kpTerm = Math.max(-clip, Math.min(clip, kpTerm));
'''
    text = replace_once(text, old3, new3, "ISEE apply band residual")
    return text


def patch_v4(text: str) -> str:
    text = replace_once(
        text,
        '  const KP_AI_GRID_COEFF_URL = "data/ai/kp_grid_coefficients.json";\n',
        '  const KP_AI_GRID_COEFF_URL = "data/ai/kp_grid_coefficients.json";\n'
        '  const KP_AI_BAND_RESIDUAL_DIR = "data/ai/kp_band_residual/";\n',
        "V4 const",
    )
    text = replace_once(
        text,
        '  let kpAiGridCoefficients = null;\n',
        '  let kpAiGridCoefficients = null;\n  const kpBandResidualByMonth = new Map();\n',
        "V4 cache",
    )

    anchor = '''  function kpAiFValue(cf, kp) {
    const x = (isFinite(kp) ? kp : 3.0) - 3.0;
    return Number(cf.k0 || 0) + Number(cf.k1 || 0) * x + Number(cf.k2 || 0) * x * x + Number(cf.k3 || 0) * x * x * x;
  }
'''
    addition = anchor + r'''

  function kpBandLabel(kp) {
    const x = Number(kp);
    if (!Number.isFinite(x) || x < 5) return "0-4";
    if (x < 7) return "5-6";
    if (x < 9) return "7-8";
    return "9";
  }

  function kpBandFallback(label) {
    if (label === "9") return ["9", "7-8", "5-6", "0-4"];
    if (label === "7-8") return ["7-8", "5-6", "0-4"];
    if (label === "5-6") return ["5-6", "0-4"];
    return ["0-4"];
  }

  async function loadKpBandResidualMonth(month) {
    const m = Number(month);
    if (kpBandResidualByMonth.has(m)) return kpBandResidualByMonth.get(m);
    const url = KP_AI_BAND_RESIDUAL_DIR + String(m).padStart(2, "0") + ".json";
    try {
      const r = await fetch(url, { cache: "no-store" });
      const doc = r.ok ? await r.json() : null;
      kpBandResidualByMonth.set(m, doc);
      return doc;
    } catch (e) {
      console.warn("NOAA band residual not available:", url, e?.message || e);
      kpBandResidualByMonth.set(m, null);
      return null;
    }
  }

  async function loadKpBandResidualCurrentMonths() {
    const now = new Date();
    const end = new Date(now.getTime() + 5 * 86400000);
    const months = new Set([now.getUTCMonth() + 1, end.getUTCMonth() + 1]);
    await Promise.all([...months].map(loadKpBandResidualMonth));
  }

  function kpBandResidualAt(monthKey, kp, i, j) {
    const doc = kpBandResidualByMonth.get(Number(monthKey));
    if (!doc) return 0;
    const bands = doc.bands || {};
    const label = kpBandLabel(kp);
    for (const bname of kpBandFallback(label)) {
      const b = bands[bname];
      if (!b) continue;
      const n = Number(b.sample_count?.[i]?.[j] ?? 0);
      const minApply = Number(b.min_apply_samples ?? 1);
      const r = Number(b.residual?.[i]?.[j] ?? 0);
      if (Number.isFinite(n) && n >= minApply && Number.isFinite(r)) return r;
    }
    return 0;
  }
'''
    text = replace_once(text, anchor, addition, "V4 helpers")

    old_apply = '''        let corr = 0;
        if (cf && Number(cf.sample_count || 0) >= 4) {
          const y = kpAiFValue(cf, kpF) - kpAiFValue(cf, kpB);
          corr = isFinite(y) ? c(y, -lim, lim) : 0;
        }
        out[i][j] = Math.max(0, v + corr);
'''
    new_apply = '''        let corr = 0;
        if (cf && Number(cf.sample_count || 0) >= 4) {
          const y = kpAiFValue(cf, kpF) - kpAiFValue(cf, kpB);
          // ISEE forecast frames already contain their dedicated Japan Kp term;
          // apply this NOAA band residual only in the NOAA path.
          const band = kpAiUsingIseeJapan() ? 0 : kpBandResidualAt(mk, kpF, i, j);
          const total = y + band;
          corr = isFinite(total) ? c(total, -lim, lim) : 0;
        }
        out[i][j] = Math.max(0, v + corr);
'''
    text = replace_once(text, old_apply, new_apply, "V4 apply band residual")

    # Preload current/next month residuals whenever the normal Kp AI bundle loads.
    old_load = '''      kpAiGridCoefficients = gridCoeff;
      iseeKpAiCoefficients = iseeCoeff;
'''
    new_load = '''      kpAiGridCoefficients = gridCoeff;
      await loadKpBandResidualCurrentMonths();
      iseeKpAiCoefficients = iseeCoeff;
'''
    text = replace_once(text, old_load, new_load, "V4 load band residual")
    return text


def main():
    if not ISEE.exists() or not V4.exists():
        raise RuntimeError("Run from the repository root; expected docs/js files are missing")

    isee_text = ISEE.read_text(encoding="utf-8")
    v4_text = V4.read_text(encoding="utf-8")

    new_isee = patch_isee(isee_text)
    new_v4 = patch_v4(v4_text)

    if new_isee != isee_text:
        ISEE.write_text(new_isee, encoding="utf-8")
        print(f"PATCHED {ISEE}")
    else:
        print(f"ALREADY PATCHED {ISEE}")

    if new_v4 != v4_text:
        V4.write_text(new_v4, encoding="utf-8")
        print(f"PATCHED {V4}")
    else:
        print(f"ALREADY PATCHED {V4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
