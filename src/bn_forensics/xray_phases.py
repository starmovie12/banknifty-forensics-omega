"""Baseline X-ray Phases 1-8, AP01-AP36 runner and deterministic 2A.3 verdicts
(FILE-028; FR-07, FR-20; refactor of legacy 02/03 phase scripts).

Everything runs on Playground days only (Ω-S3). Every phase either produces its table or reports
``N/A — DATA MISSING`` naming the missing inputs (R10: no silent skipping). Multi-bar constructs
start at their confirmation bar; every percentile/tercile cut-off is discovery-frozen; full-day
H/L/C quantities are labelled ``DESCRIPTIVE-HINDSIGHT``. Option-structure cards evaluated on Era-A
days are tagged ``ERA-SPECIFIC`` (Ω-7).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from bn_forensics import NA_DATA_MISSING, InputError
from bn_forensics.bricks_compiler import (
    STATUS_ENV,
    BrickLibrary,
    CompiledCard,
    FeatureContext,
    compile_card,
    load_ap_cards,
    load_ap_meta,
    sklearn_available,
)
from bn_forensics.core_stats import (
    BarFrame,
    Ctx,
    EventResult,
    TestRunner,
    bh_qvalues,
    day_block_bootstrap,
    event_study,
    gate_adjusted_mde,
    horizon_cap,
    outcome,
    resample,
    write_csv,
)

GRADE_ORDER = {"Ω-A": 6, "A": 5, "Ω-B": 4, "B": 3, "Ω-C": 2, "C": 1, "D": 0}
TAG_ORDER = {"T1": 4, "T2": 3, "T?": 2, "T3": 1, "STRUCTURE-ONLY": 0}
VERDICT_11A_FIELDS = (
    "approach_id", "group", "name", "verdict", "best_grade", "best_test", "best_side", "best_tag",
    "nominal_mde_med", "gate_mde_med", "gate_mde_min", "gate_mde_max", "family_gate_mde_med",
    "needed_effect", "runnable_tests", "tests_n30", "label", "reason", "missing",
)
VERDICT_11B_FIELDS = ("group", "group_name", "members_S", "members_C", "members_I", "members_NA",
                      "best_evidence_grade", "verdict", "reason")
OPTION_STRUCTURE_TOKENS = ("opt_1m", "chain_1m", "chain_metrics_1m", "iv_greeks_1m", "expiry_calendar")


# =============================================================================================
# Inputs / outputs
# =============================================================================================
@dataclass
class XrayInputs:
    frame: BarFrame                          # Playground analysis frame (e.g. 3-minute)
    ctx: FeatureContext
    cfg: dict[str, Any]
    run_ctx: Ctx
    fit_day_mask: np.ndarray                 # True on every Playground day of ``frame``
    eras: dict[str, str]                     # ISO date -> era id
    costs: dict[str, float | None]           # matching-series costs in points
    day_limited: bool = False
    deflated: Callable[[EventResult, str], float | None] | None = None
    tables_dir: Path | None = None


@dataclass
class PhaseTable:
    phase: str
    title: str
    status: str                              # RUN | N/A — DATA MISSING | NOT-RUN(ENV)
    rows: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


@dataclass
class XrayResult:
    phases: list[PhaseTable]
    tests: list[dict[str, Any]]
    compiled: dict[str, CompiledCard]
    results: dict[str, list[EventResult]]
    verdicts_11a: list[dict[str, Any]]
    verdicts_11b: list[dict[str, Any]]
    era_tags: dict[str, str]
    interactions: list[dict[str, Any]]


def _nanmean(x: np.ndarray) -> float:
    """Mean of finite values, NaN for an all-missing cell (no RuntimeWarning)."""
    arr = np.asarray(x, float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _na(phase: str, title: str, missing: Sequence[str]) -> PhaseTable:
    return PhaseTable(phase, title, NA_DATA_MISSING, [], f"{NA_DATA_MISSING}: needs {list(missing)}")


def _q(x: np.ndarray, qs: Sequence[float]) -> list[float]:
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, q)) if x.size else float("nan") for q in qs]


def _describe(name: str, x: np.ndarray, units: str) -> dict[str, Any]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"metric": name, "units": units, "n": 0}
    p5, p25, p75, p95, p99 = _q(x, [0.05, 0.25, 0.75, 0.95, 0.99])
    return {"metric": name, "units": units, "n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p5": p5, "p25": p25, "p75": p75, "p95": p95, "p99": p99, "max": float(x.max())}


def _fwd(frame: BarFrame, h: int) -> np.ndarray:
    return outcome(frame, "T1", h)["atr"]


def _cap_h(frame: BarFrame, hs: Sequence[int], cfg: dict[str, Any]) -> list[int]:
    cap = horizon_cap(frame.bars_per_day, float(cfg["stats"]["horizon_cap_frac"]))
    return [h for h in hs if h <= cap]


def _day_ci(values: np.ndarray, frame: BarFrame, mask: np.ndarray, B: int, seed: int) -> dict[str, float]:
    ok = mask & np.isfinite(values)
    sums = np.bincount(frame.day_idx[ok], weights=values[ok], minlength=frame.n_days)
    cnts = np.bincount(frame.day_idx[ok], minlength=frame.n_days).astype(float)
    return day_block_bootstrap(sums, cnts, B, seed)


# =============================================================================================
# Phase 1 — Candle DNA
# =============================================================================================
def phase1(inp: XrayInputs) -> list[PhaseTable]:
    fr = inp.frame
    atr = fr.atr20
    body = np.abs(fr.c - fr.o)
    upper = fr.h - np.maximum(fr.o, fr.c)
    lower = np.minimum(fr.o, fr.c) - fr.l
    rng = fr.h - fr.l
    out = []
    rows = []
    for name, val in (("body", body), ("upper_wick", upper), ("lower_wick", lower), ("range", rng)):
        rows.append(_describe(f"{name}_points", val, "points"))
        rows.append(_describe(f"{name}_pct", 100 * val / fr.c, "%"))
        rows.append(_describe(f"{name}_atr20", val / atr, "ATR20"))
    with np.errstate(divide="ignore", invalid="ignore"):
        rows.append(_describe("body_pct_of_range", np.where(rng > 0, 100 * body / rng, np.nan), "%"))
    out.append(PhaseTable("1.1", "Candle anatomy", "RUN", rows))
    weekday = ((fr.dates.astype("datetime64[D]").astype(np.int64) + 3) % 7)
    heat = pd.DataFrame({"slot": fr.slot, "weekday": weekday, "range": rng}).dropna()
    piv = heat.groupby(["slot", "weekday"])["range"].median().unstack(fill_value=np.nan)
    rows = [{"slot": int(s), **{f"wd{int(w)}": float(piv.loc[s, w]) for w in piv.columns}} for s in piv.index]
    month = fr.dates.astype("datetime64[M]").astype(str)
    for m in sorted(set(month)):
        sel = month == m
        rows.append({"slot": -1, "month": m, "median_range": float(np.nanmedian(rng[sel]))})
    out.append(PhaseTable("1.2", "Slot x weekday median range (points); month splits", "RUN", rows))
    bf = inp.ctx.brick("body_frac")
    wr = inp.ctx.brick("wick_ratio")
    ra = inp.ctx.brick("range_atr")
    fit = inp.fit_day_mask[fr.day_idx]
    cut = {k: float(np.nanquantile(v[fit], q)) for k, v, q in (("bf90", bf, 0.9), ("bf10", bf, 0.1), ("bf25", bf, 0.25),
                                                                 ("wr10", wr, 0.1), ("wr90", wr, 0.9), ("ra90", ra, 0.9),
                                                                 ("ra10", ra, 0.1))}
    types = {"marubozu": bf >= cut["bf90"], "doji": bf <= cut["bf10"],
             "hammer_like": (wr <= cut["wr10"]) & (bf <= cut["bf25"]),
             "shooting_star_like": (wr >= cut["wr90"]) & (bf <= cut["bf25"]),
             "big_bar": ra >= cut["ra90"], "small_bar": ra <= cut["ra10"]}
    rows = []
    for name, mask in types.items():
        for h in _cap_h(fr, [1, 3, 5, 10], inp.cfg):
            fwd = _fwd(fr, h)
            ci = _day_ci(fwd, fr, mask, inp.run_ctx.bootstrap_B, inp.run_ctx.seed)
            rows.append({"type": name, "horizon": h, "frequency": float(np.mean(mask)), "n": int(mask.sum()),
                         "fwd_mean_atr20": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                         "cutoffs": "discovery_frozen"})
    out.append(PhaseTable("1.3", "Data-driven candle types and next 1/3/5/10-bar outcomes (h <= bars/day/4)", "RUN", rows))
    color = np.sign(fr.c - fr.o)
    streak = inp.ctx.brick("streak_len")
    nxt = np.r_[color[1:], np.nan]
    same_day = np.r_[fr.day_idx[1:] == fr.day_idx[:-1], False]
    base = float(np.mean((nxt == color)[same_day & np.isfinite(nxt) & (color != 0)]))
    rows = []
    for k in range(1, 9):
        sel = same_day & (np.abs(streak) == k) & np.isfinite(nxt)
        cont = float(np.mean(nxt[sel] == np.sign(streak[sel]))) if sel.any() else float("nan")
        rows.append({"streak_k": k, "n": int(sel.sum()), "p_continue": cont, "p_independence": base})
    out.append(PhaseTable("1.4", "Colour & streaks (continuation vs independence)", "RUN", rows))
    cl = inp.ctx.brick("close_loc")
    with np.errstate(divide="ignore", invalid="ignore"):
        ol = np.clip(np.ceil((fr.o - fr.l) / rng * 10), 1, 10)
    rows = [{"decile": d, "close_share": float(np.mean(cl[np.isfinite(cl)] == d)),
             "open_share": float(np.mean(ol[np.isfinite(ol)] == d))} for d in range(1, 11)]
    out.append(PhaseTable("1.5", "Close / open location deciles", "RUN", rows))
    gap = inp.ctx.brick("gap_atr")[fr.day_start]
    d = fr.day_ohlc()
    prev_close = np.r_[np.nan, d["close"][:-1]]
    rows = []
    for i in range(1, fr.n_days):
        s, e = fr.day_start[i], fr.day_end[i]
        g = d["open"][i] - prev_close[i]
        filled_at = None
        for j in range(s, e):
            if (g > 0 and fr.l[j] <= prev_close[i]) or (g < 0 and fr.h[j] >= prev_close[i]):
                filled_at = j - s
                break
        ret15 = _first_minutes_return(fr, i, 15)
        rows.append({"date": str(fr.days[i]), "gap_points": float(g), "gap_atr20": float(gap[i]),
                     "weekday": int((fr.days[i].astype(np.int64) + 3) % 7),
                     "filled_same_day": filled_at is not None, "bars_to_fill": filled_at,
                     "ret_first_15m_atr20": ret15, "ret_first_30m_atr20": _first_minutes_return(fr, i, 30),
                     "ret_first_60m_atr20": _first_minutes_return(fr, i, 60), "label": "DESCRIPTIVE-HINDSIGHT"})
    out.append(PhaseTable("1.6", "Gaps: size, weekday, fill probability & time-to-fill (DESCRIPTIVE-HINDSIGHT)", "RUN", rows))
    prev_h, prev_l = np.r_[np.nan, fr.h[:-1]], np.r_[np.nan, fr.l[:-1]]
    valid = np.r_[False, fr.day_idx[1:] == fr.day_idx[:-1]]
    inside = valid & (fr.h <= prev_h) & (fr.l >= prev_l)
    outside = valid & (fr.h >= prev_h) & (fr.l <= prev_l)
    larger = valid & (rng > np.r_[np.nan, rng[:-1]])
    seq2 = pd.Series([f"{int(a)}{int(b)}" for a, b in zip(np.r_[0, color[:-1]][valid], color[valid], strict=True)]).value_counts(normalize=True)
    rows = [{"relation": "inside", "share": float(inside[valid].mean())}, {"relation": "outside", "share": float(outside[valid].mean())},
            {"relation": "larger_than_prior", "share": float(larger[valid].mean())}]
    rows += [{"relation": f"colour_seq_{k}", "share": float(v)} for k, v in seq2.sort_index().items()]
    out.append(PhaseTable("1.7", "Bar-to-bar relations and 2-bar colour sequences", "RUN", rows))
    out.append(_compression_phase(inp, inside))
    rows = []
    for k in (5, 10):
        coarse = resample(fr, fr.interval_min * k) if fr.interval_min * k <= 60 else None
        if coarse is None:
            continue
        base_ret = np.r_[np.nan, np.diff(np.log(fr.c))]
        c_ret = np.r_[np.nan, np.diff(np.log(coarse.c))]
        rows.append({"scale_minutes": coarse.interval_min, "std_ratio": float(np.nanstd(c_ret) / np.nanstd(base_ret)),
                     "sqrt_k": math.sqrt(k), "skew": float(sps.skew(c_ret[np.isfinite(c_ret)])),
                     "kurtosis": float(sps.kurtosis(c_ret[np.isfinite(c_ret)]))})
    out.append(PhaseTable("1.9", "Scale consistency (higher-TF resampling)", "RUN", rows))
    rows = []
    closes = fr.c[np.isfinite(fr.c)]
    tick = float(np.min(np.diff(np.unique(np.round(closes, 2))))) if closes.size > 2 else float("nan")
    for step in (50, 100, 500):
        dist = np.abs(fr.c - np.round(fr.c / step) * step) / atr
        near = dist <= 0.1
        null_share = np.minimum(1.0, 0.2 * atr / step)
        rows.append({"round": step, "share_within_0.1atr": _nanmean(near),
                     "uniform_null_share": _nanmean(null_share), "tick_granularity": tick})
    out.append(PhaseTable("1.10", "Price granularity & round-number clustering vs uniform null", "RUN", rows))
    first, last = fr.is_first_bar, fr.is_last_bar
    rows = []
    for name, sel in (("first_bar", first), ("last_bar", last), ("other", ~first & ~last)):
        rows.append({"bars": name, "range_atr20_mean": _nanmean((rng / atr)[sel]),
                     "abs_ret_atr20_mean": _nanmean((np.abs(fr.c - fr.o) / atr)[sel])})
    out.append(PhaseTable("1.11", "First & last bars of day", "RUN", rows))
    return out


def _first_minutes_return(fr: BarFrame, day: int, minutes: int) -> float:
    s, e = fr.day_start[day], fr.day_end[day]
    open_min = fr.minute[s]
    idx = [j for j in range(s, e) if fr.minute[j] + fr.interval_min <= open_min + minutes]
    if not idx:
        return float("nan")
    return float((fr.c[idx[-1]] - fr.o[s]) / fr.atr20[idx[-1]])


def _compression_phase(inp: XrayInputs, inside: np.ndarray) -> PhaseTable:
    fr = inp.frame
    ra = inp.ctx.brick("range_atr")
    fit = inp.fit_day_mask[fr.day_idx]
    p25 = float(np.nanquantile(ra[fit], 0.25))
    small3 = (ra < p25) & (np.r_[np.nan, ra[:-1]] < p25) & (np.r_[np.nan, np.nan, ra[:-2]] < p25)
    bbq = inp.ctx.brick("bb_bandwidth_pct") <= 0.10
    nr7 = (inp.ctx.brick("nr_7") == 1) & fr.is_first_bar
    rows = []
    for name, mask, look in (("k3_small", small3, 3), ("inside_bar", inside, 1), ("bb_squeeze", bbq, 20), ("nr7_session", nr7, 1)):
        idx = np.flatnonzero(mask)
        broke, sizes = 0, []
        for i in idx:
            lo_i = max(fr.day_start[fr.day_idx[i]], i - look + 1)
            hi_lvl, lo_lvl = np.nanmax(fr.h[lo_i:i + 1]), np.nanmin(fr.l[lo_i:i + 1])
            end = min(fr.day_end[fr.day_idx[i]], i + 11)
            for j in range(i + 1, end):
                if fr.c[j] > hi_lvl or fr.c[j] < lo_lvl:
                    broke += 1
                    sizes.append(abs(fr.c[j] - fr.c[i]) / fr.atr20[i])
                    break
        n = idx.size
        rows.append({"pattern": name, "events": int(n), "break_within_10": broke / n if n else float("nan"),
                     "no_break_share": (n - broke) / n if n else float("nan"),
                     "median_break_size_atr20": float(np.median(sizes)) if sizes else float("nan")})
    return PhaseTable("1.8", "Compression-to-expansion (first break within 10 bars; NO-BREAK separate)", "RUN", rows)


# =============================================================================================
# Phase 2 — Price movement physics
# =============================================================================================
def acf(x: np.ndarray, lags: int) -> list[float]:
    x = x[np.isfinite(x)]
    x = x - x.mean()
    denom = float(np.sum(x * x))
    return [float(np.sum(x[k:] * x[:-k]) / denom) if denom > 0 and k < x.size else float("nan") for k in range(1, lags + 1)]


def ljung_box(rho: Sequence[float], n: int) -> tuple[float, float]:
    q = n * (n + 2) * sum(r * r / (n - k) for k, r in enumerate(rho, start=1) if np.isfinite(r))
    return float(q), float(sps.chi2.sf(q, len(rho)))


def variance_ratio(frame: BarFrame, h: int, B: int, seed: int) -> dict[str, float]:
    """Within-session VR(h) = Var(h-bar returns) / (h * Var(1-bar)) with a day-block CI."""
    lr = np.log(frame.c)
    one = np.r_[np.nan, np.diff(lr)]
    one[frame.day_start] = np.nan
    hret = lr - np.r_[np.full(h, np.nan), lr[:-h]]
    hret[np.r_[np.full(h, True), frame.day_idx[h:] != frame.day_idx[:-h]]] = np.nan
    vr = float(np.nanvar(hret) / (h * np.nanvar(one)))
    rng = np.random.default_rng(seed)
    days = np.arange(frame.n_days)
    boots = []
    for _ in range(min(B, 500)):
        pick = rng.choice(days, size=days.size, replace=True)
        sel = np.isin(frame.day_idx, pick)
        w = np.bincount(pick, minlength=frame.n_days)[frame.day_idx]
        a, b = hret[sel], one[sel]
        ww = w[sel]
        oka, okb = np.isfinite(a), np.isfinite(b)
        va = np.average((a[oka] - np.average(a[oka], weights=ww[oka])) ** 2, weights=ww[oka])
        vb = np.average((b[okb] - np.average(b[okb], weights=ww[okb])) ** 2, weights=ww[okb])
        boots.append(va / (h * vb))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"h": h, "vr": vr, "ci_lo": float(lo), "ci_hi": float(hi), "excludes_1": bool(lo > 1 or hi < 1)}


def hurst_rs(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    sizes = [s for s in (16, 32, 64, 128, 256) if s * 4 <= x.size]
    if len(sizes) < 2:
        return float("nan")
    rs = []
    for s in sizes:
        vals = []
        for k in range(x.size // s):
            seg = x[k * s:(k + 1) * s]
            dev = np.cumsum(seg - seg.mean())
            sd = seg.std()
            if sd > 0:
                vals.append((dev.max() - dev.min()) / sd)
        rs.append(np.mean(vals))
    slope = np.polyfit(np.log(sizes), np.log(rs), 1)[0]
    return float(slope)


def runs_test(signs: np.ndarray) -> dict[str, float]:
    s = signs[np.isfinite(signs) & (signs != 0)]
    n1, n2 = int((s > 0).sum()), int((s < 0).sum())
    if n1 == 0 or n2 == 0:
        return {"runs": float("nan"), "z": float("nan"), "p": float("nan")}
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / math.sqrt(var) if var > 0 else float("nan")
    return {"runs": float(runs), "z": float(z), "p": float(2 * sps.norm.sf(abs(z)))}


def phase2(inp: XrayInputs) -> list[PhaseTable]:
    fr = inp.frame
    out = []
    ret = np.r_[np.nan, np.diff(np.log(fr.c))]
    ret[fr.day_start] = np.nan
    r = ret[np.isfinite(ret)]
    sd = float(r.std())
    rows = [{"metric": "mean", "value": float(r.mean())}, {"metric": "std", "value": sd},
            {"metric": "skew", "value": float(sps.skew(r))}, {"metric": "excess_kurtosis", "value": float(sps.kurtosis(r))}]
    rows += [{"metric": f"tail_gt_{k}sd", "value": int(np.sum(np.abs(r) > k * sd))} for k in (2, 3, 4)]
    p1, p99 = _q(r, [0.01, 0.99])
    rows += [{"metric": "p1", "value": p1}, {"metric": "p99", "value": p99}]
    absr = np.abs(r)
    top = np.sort(absr)[::-1]
    rows.append({"metric": "share_abs_ret_top1pct_bars", "value": float(top[: max(1, top.size // 100)].sum() / absr.sum())})
    day_abs = np.bincount(fr.day_idx[np.isfinite(ret)], weights=np.abs(ret[np.isfinite(ret)]), minlength=fr.n_days)
    rows.append({"metric": "share_abs_ret_top5_days", "value": float(np.sort(day_abs)[::-1][:5].sum() / day_abs.sum())})
    out.append(PhaseTable("2.1", "Return distribution & tail concentration", "RUN", rows))
    lags = int(min(fr.bars_per_day // 2, 50))
    rho = acf(ret, lags)
    rho_abs = acf(np.abs(ret), lags)
    q, p = ljung_box(rho, r.size)
    rows = [{"lag": k + 1, "acf_ret": rho[k], "acf_abs_ret": rho_abs[k]} for k in range(lags)]
    rows.append({"lag": "ljung_box", "acf_ret": q, "acf_abs_ret": p})
    rows.append({"lag": "signed_volume_acf", "acf_ret": NA_DATA_MISSING, "acf_abs_ret": "needs trade-side volume"})
    for h in _cap_h(fr, [2, 4, 8, 16, 32], inp.cfg):
        rows.append({"lag": f"VR({h})", **variance_ratio(fr, h, inp.run_ctx.bootstrap_B, inp.run_ctx.seed)})
    rows.append({"lag": "hurst_rs", "acf_ret": hurst_rs(ret)})
    rows.append({"lag": "runs_test", **runs_test(np.sign(ret))})
    out.append(PhaseTable("2.2", "Intra-session serial dependence (ACF, Ljung-Box, VR, Hurst, runs)", "RUN", rows))
    d = fr.day_ohlc()
    with np.errstate(divide="ignore", invalid="ignore"):
        cc = np.abs(np.diff(np.log(d["close"])))
        park = np.sqrt((np.log(d["high"] / d["low"]) ** 2) / (4 * math.log(2)))
        gk = np.sqrt(0.5 * np.log(d["high"] / d["low"]) ** 2 - (2 * math.log(2) - 1) * np.log(d["close"] / d["open"]) ** 2)
    slot_abs = pd.Series(np.abs(ret)).groupby(fr.slot).mean()
    rng_day = d["high"] - d["low"]
    rows = [{"metric": "daily_close_to_close_abs_mean", "value": _nanmean(cc)},
            {"metric": "parkinson_daily_mean", "value": _nanmean(park)},
            {"metric": "garman_klass_daily_mean", "value": _nanmean(gk)},
            {"metric": "u_shape_first_over_mid", "value": float(slot_abs.iloc[0] / slot_abs.iloc[len(slot_abs) // 2])},
            {"metric": "abs_ret_acf_lag1", "value": rho_abs[0]},
            {"metric": "clustering_half_life_bars", "value": float(next((k + 1 for k, v in enumerate(rho_abs) if v < rho_abs[0] / 2), lags))},
            {"metric": "vol_of_vol_median", "value": float(np.nanmedian(inp.ctx.brick("vol_of_vol")))},
            {"metric": "yesterday_today_range_corr", "value": float(pd.Series(rng_day[:-1]).corr(pd.Series(rng_day[1:])))}]
    rows += [{"metric": f"atr20_p{int(q * 100)}", "value": v} for q, v in zip((0.1, 0.5, 0.9), _q(fr.atr20, [0.1, 0.5, 0.9]), strict=True)]
    out.append(PhaseTable("2.3", "Volatility dynamics", "RUN", rows))
    out.append(_day_structure(inp))
    out.append(_swing_phase(inp))
    out.append(_levels_phase(inp))
    rows = []
    for h in _cap_h(fr, [5, 10, 20], inp.cfg):
        o3 = outcome(fr, "T3", h)
        up, dn = o3["up"], o3["down"]
        rng_h = (up + dn)
        rows.append({"horizon": h, **{f"up_p{int(q * 100)}": v for q, v in zip((0.5, 0.75, 0.9, 0.95), _q(up, [0.5, 0.75, 0.9, 0.95]), strict=True)},
                     **{f"down_p{int(q * 100)}": v for q, v in zip((0.5, 0.75, 0.9, 0.95), _q(dn, [0.5, 0.75, 0.9, 0.95]), strict=True)},
                     **{f"share_range_ge_{x}": _nanmean(rng_h >= x) for x in inp.cfg["targets"]["expansion_x"]}})
    out.append(PhaseTable("2.7", "Direction-neutral excursion physics (zero first-touch)", "RUN", rows))
    return out


def _day_structure(inp: XrayInputs) -> PhaseTable:
    fr = inp.frame
    d = fr.day_ohlc()
    rows = []
    prev = {k: np.r_[np.nan, v[:-1]] for k, v in d.items()}
    for i in range(fr.n_days):
        s, e = fr.day_start[i], fr.day_end[i]
        hi_bar = s + int(np.nanargmax(fr.h[s:e])) if np.isfinite(fr.h[s:e]).any() else s
        lo_bar = s + int(np.nanargmin(fr.l[s:e])) if np.isfinite(fr.l[s:e]).any() else s
        rng = d["high"][i] - d["low"][i]
        rows.append({"date": str(fr.days[i]), "o_to_c_points": d["close"][i] - d["open"][i], "h_to_l_points": rng,
                     "close_in_range": (d["close"][i] - d["low"][i]) / rng if rng > 0 else np.nan,
                     "high_slot": int(fr.slot[hi_bar]), "low_slot": int(fr.slot[lo_bar]),
                     "touched_prev_high": bool(d["high"][i] >= prev["high"][i]), "touched_prev_low": bool(d["low"][i] <= prev["low"][i]),
                     "overnight_points": d["open"][i] - prev["close"][i], "intraday_points": d["close"][i] - d["open"][i],
                     "label": "DESCRIPTIVE-HINDSIGHT"})
    df = pd.DataFrame(rows)
    on, intra = df["overnight_points"].to_numpy(float), df["intraday_points"].to_numpy(float)
    tot = np.nanvar(on + intra)
    rows.append({"date": "SUMMARY", "overnight_var_share": float(np.nanvar(on) / tot) if tot > 0 else np.nan,
                 "intraday_var_share": float(np.nanvar(intra) / tot) if tot > 0 else np.nan,
                 "prev_high_touch_rate": float(df["touched_prev_high"].mean()), "prev_low_touch_rate": float(df["touched_prev_low"].mean())})
    return PhaseTable("2.4", "Day structure, H/L timing, prior-day levels, overnight vs intraday", "RUN", rows)


def zigzag_legs(c: np.ndarray, threshold: np.ndarray) -> list[tuple[int, int, float]]:
    """Confirmed zig-zag legs ``(start, confirm_bar, size)``; a pivot is known only at its confirmation bar."""
    legs = []
    piv_i, piv_p, direction = 0, c[0], 0
    ext_i, ext_p = 0, c[0]
    for i in range(1, c.size):
        if not np.isfinite(c[i]) or not np.isfinite(threshold[i]):
            continue
        if direction >= 0 and c[i] > ext_p:
            ext_i, ext_p = i, c[i]
        if direction <= 0 and c[i] < ext_p:
            ext_i, ext_p = i, c[i]
        move = c[i] - ext_p
        if (direction >= 0 and move <= -threshold[i]) or (direction <= 0 and move >= threshold[i]):
            legs.append((piv_i, i, float(ext_p - piv_p)))
            piv_i, piv_p = ext_i, ext_p
            direction = -1 if move < 0 else 1
            ext_i, ext_p = i, c[i]
    return legs


def _band(depth: np.ndarray, lo: float, hi: float) -> float:
    return float(np.mean((depth >= lo) & (depth <= hi))) if depth.size else float("nan")


def _swing_phase(inp: XrayInputs) -> PhaseTable:
    fr = inp.frame
    med_atr = float(np.nanmedian(fr.atr20))
    rows = []
    for k in (1.0, 2.0, 3.0):
        legs = zigzag_legs(fr.c, np.full(fr.n, k * med_atr))
        sizes = np.abs([lg[2] for lg in legs])
        depth = np.asarray([abs(b[2]) / abs(a[2]) for a, b in itertools.pairwise(legs) if abs(a[2]) > 0])
        rows.append({"k_median_atr": k, "legs": len(legs), "median_leg_atr20": float(np.median(sizes) / med_atr) if sizes.size else np.nan,
                     "pullback_38_50": _band(depth, 0.33, 0.43), "pullback_50_band": _band(depth, 0.45, 0.55),
                     "pullback_62_band": _band(depth, 0.57, 0.67),
                     "uniform_null_band_share": 0.10, "confirmation": "from CONFIRMATION BAR"})
    for n in (10, 20, 50):
        ext = inp.ctx.brick(f"nbar_extreme_{n}")
        rows.append({"k_median_atr": f"N={n}", "legs": int(np.nansum(np.abs(ext))), "median_leg_atr20": _nanmean(np.abs(ext))})
    return PhaseTable("2.5", "Swing/leg structure (discovery-frozen zig-zag) & N-bar breakouts", "RUN", rows)


def _levels_phase(inp: XrayInputs) -> PhaseTable:
    fr = inp.frame
    rows = []
    for step in (100, 500, 1000):
        dist = np.abs(fr.c - np.round(fr.c / step) * step) / fr.atr20
        near = dist <= 0.1
        nxt = np.abs(_fwd(fr, 3))
        rows.append({"level": f"round_{step}", "touch_share": _nanmean(near),
                     "abs_fwd3_near": _nanmean(nxt[near]) if near.any() else np.nan,
                     "abs_fwd3_far": _nanmean(nxt[~near])})
    for name in ("dist_pdh_atr", "dist_pdl_atr"):
        dist = np.abs(inp.ctx.brick(name))
        near = dist <= 0.1
        rows.append({"level": name, "touch_share": _nanmean(near),
                     "abs_fwd3_near": _nanmean(np.abs(_fwd(fr, 3))[near]) if near.any() else np.nan,
                     "abs_fwd3_far": _nanmean(np.abs(_fwd(fr, 3))[~near])})
    if "fut_volume" in inp.ctx.available:
        dist = np.abs(inp.ctx.brick("dist_vwap_atr"))
        rows.append({"level": "vwap (FUT-VOLUME)", "touch_share": _nanmean(dist <= 0.1)})
    else:
        rows.append({"level": "vwap", "touch_share": NA_DATA_MISSING, "note": "needs fut_volume or explicit VWAP (TWAP-PROXY never feeds verdicts)"})
    return PhaseTable("2.6", "Levels & magnets", "RUN", rows)


# =============================================================================================
# Phases 3-4 (data gated)
# =============================================================================================
def phase3(inp: XrayInputs) -> list[PhaseTable]:
    if "fut_volume" not in inp.ctx.available:
        return [_na(f"3.{i}", "Volume & participation (FUT-VOLUME)", ["fut_volume (index has_volume = false)"]) for i in range(1, 8)] + \
            ([_na("3.8", "Open interest classes", ["fut_oi"])] if "fut_oi" not in inp.ctx.available else [_oi_phase(inp)])
    fr = inp.frame
    rvol = inp.ctx.brick("rvol_slot")
    absr = np.abs(_fwd(fr, 1))
    rng_atr = (fr.h - fr.l) / fr.atr20
    rows = [{"slot": int(s), "median_rvol": float(np.nanmedian(rvol[fr.slot == s]))} for s in np.unique(fr.slot)]
    out = [PhaseTable("3.1", "RVOL slot profile (FUT-VOLUME)", "RUN", rows)]
    out.append(PhaseTable("3.2", "RVOL vs |ret| and range (FUT-VOLUME)", "RUN", [
        {"metric": "corr_rvol_absret", "value": float(pd.Series(rvol).corr(pd.Series(absr)))},
        {"metric": "corr_rvol_range", "value": float(pd.Series(rvol).corr(pd.Series(rng_atr)))}]))
    fit = inp.fit_day_mask[fr.day_idx]
    p95, p99 = np.nanquantile(rvol[fit], [0.95, 0.99])
    rows = [{"spike": name, "n": int(np.nansum(sel)), "fwd_abs_mean": _nanmean(np.abs(_fwd(fr, 5))[sel])}
            for name, sel in (("rvol>p95", rvol > p95), ("rvol>p99", rvol > p99))]
    out.append(PhaseTable("3.3", "RVOL spikes (FUT-VOLUME)", "RUN", rows))
    small = rng_atr < np.nanquantile(rng_atr[fit], 0.25)
    big = rng_atr > np.nanquantile(rng_atr[fit], 0.95)
    out.append(PhaseTable("3.4", "Climax & dry-up (FUT-VOLUME)", "RUN", [
        {"pattern": "climax", "n": int(np.nansum((rvol > p95) & big)), "fwd5_mean": _nanmean(_fwd(fr, 5)[(rvol > p95) & big])},
        {"pattern": "dry_up", "n": int(np.nansum(rvol < np.nanquantile(rvol[fit], 0.05))), "fwd5_mean": _nanmean(_fwd(fr, 5)[rvol < np.nanquantile(rvol[fit], 0.05)])}]))
    out.append(PhaseTable("3.5", "Effort vs result (FUT-VOLUME)", "RUN", [
        {"pattern": "absorption", "n": int(np.nansum((rvol > p95) & small))},
        {"pattern": "low_vol_big_range", "n": int(np.nansum((rvol < np.nanquantile(rvol[fit], 0.25)) & big))}]))
    ext = inp.ctx.brick("nbar_extreme_20")
    hi_bo = (np.abs(ext) == 1) & (rvol > np.nanmedian(rvol[fit]))
    lo_bo = (np.abs(ext) == 1) & (rvol <= np.nanmedian(rvol[fit]))
    out.append(PhaseTable("3.6", "High vs low RVOL breakouts (FUT-VOLUME)", "RUN", [
        {"breakout": "high_rvol", "n": int(hi_bo.sum()), "cont_fwd5": _nanmean((_fwd(fr, 5) * ext)[hi_bo])},
        {"breakout": "low_rvol", "n": int(lo_bo.sum()), "cont_fwd5": _nanmean((_fwd(fr, 5) * ext)[lo_bo])}]))
    poc = inp.ctx.brick("dist_prev_poc")
    out.append(PhaseTable("3.7", "Prior value area / POC & VWAP distance (FUT-VOLUME)", "RUN", [
        {"metric": "share_beyond_1atr_from_prev_poc", "value": _nanmean(np.abs(poc) > 1.0)}]))
    out.append(_oi_phase(inp) if "fut_oi" in inp.ctx.available else _na("3.8", "Open interest classes", ["fut_oi"]))
    return out


def _oi_phase(inp: XrayInputs) -> PhaseTable:
    fr = inp.frame
    cls = inp.ctx.brick("price_oi_class")
    fwd = _fwd(fr, 5)
    rows = [{"class": int(k), "label": ("long build-up", "short build-up", "short covering", "long unwinding")[int(k) - 1],
             "n": int(np.sum(cls == k)), "fwd5_mean_atr20": _nanmean(fwd[cls == k])} for k in (1, 2, 3, 4)]
    return PhaseTable("3.8", "Open interest 4 classes (labels only)", "RUN", rows)


def phase4(inp: XrayInputs) -> list[PhaseTable]:
    avail = inp.ctx.available
    out = []
    titles = {"4.1": "Premium anatomy", "4.2": "Empirical delta & gamma", "4.3": "Theta & decay curve",
              "4.4": "Premium path & option flow", "4.5": "IV surface, skew, IV vs RV", "4.6": "Bid-ask spread & liquidity",
              "4.7": "Futures basis, roll, lead-lag", "4.8": "Expiry-day physics & pinning", "4.9": "Chain OI walls, PCR, max-pain, signed GEX"}
    needs = {"4.1": ["opt_1m"], "4.2": ["opt_1m"], "4.3": ["opt_1m"], "4.4": ["opt_1m"], "4.5": ["iv_greeks_1m"],
             "4.6": ["recorder_quotes"], "4.7": ["fut_1m"], "4.8": ["expiry_calendar"], "4.9": ["chain_1m"]}
    for ph in sorted(titles, key=lambda p: float(p)):
        missing = [t for t in needs[ph] if t not in avail]
        if missing:
            out.append(_na(ph, titles[ph], missing))
            continue
        out.append(_phase4_item(inp, ph, titles[ph]))
    return out


def _phase4_item(inp: XrayInputs, ph: str, title: str) -> PhaseTable:
    fr = inp.frame
    ctx = inp.ctx
    if ph in ("4.1", "4.3", "4.4"):
        prem = ctx.brick("atm_straddle_pct")
        pr = ctx.brick("premium_ret_5")
        low_move = np.abs(_fwd(fr, 1)) < 0.25
        rows = [{"metric": "atm_straddle_pct_median", "value": float(np.nanmedian(prem))},
                {"metric": "premium_ret_5_mean", "value": _nanmean(pr)},
                {"metric": "premium_ret_5_low_move_bars", "value": _nanmean(pr[low_move])}]
        return PhaseTable(ph, title, "RUN", rows)
    if ph == "4.2":
        prem = ctx.aux.get("__straddle")
        if prem is None:
            ctx.brick("atm_straddle_pct")
            prem = ctx.aux["__straddle"]
        du = np.r_[np.nan, np.diff(fr.c)]
        dp = np.r_[np.nan, np.diff(prem)]
        ok = np.isfinite(du) & np.isfinite(dp) & ~((du != 0) & (dp == 0))
        if ok.sum() < 30:
            return PhaseTable(ph, title, "RUN", [{"metric": "insufficient non-stale bars", "value": int(ok.sum())}])
        x = np.column_stack([du[ok], 0.5 * du[ok] ** 2])
        coef = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), x]), dp[ok], rcond=None)[0]
        return PhaseTable(ph, title, "RUN", [{"metric": "straddle_delta", "value": float(coef[1])},
                                             {"metric": "straddle_gamma", "value": float(coef[2])},
                                             {"metric": "bs_comparison", "value": "PROXY only without IV/rates (r=0,q=0)"}])
    if ph == "4.5":
        iv = ctx.brick("atm_iv")
        sk = ctx.brick("skew")
        return PhaseTable(ph, title, "RUN", [{"metric": "atm_iv_median", "value": float(np.nanmedian(iv))},
                                             {"metric": "skew_median", "value": float(np.nanmedian(sk))}])
    if ph == "4.6":
        sp = ctx.brick("spread_pct")
        return PhaseTable(ph, title, "RUN", [{"metric": "spread_pct_median (real quotes only)", "value": float(np.nanmedian(sp))}])
    if ph == "4.7":
        basis = ctx.brick("fut_basis")
        return PhaseTable(ph, title, "RUN", [{"metric": "basis_atr20_median", "value": float(np.nanmedian(basis))},
                                             {"metric": "basis_atr20_p95", "value": _q(basis, [0.95])[0]}])
    if ph == "4.8":
        dte = ctx.brick("dte")
        rng = (fr.h - fr.l) / fr.atr20
        return PhaseTable(ph, title, "RUN", [{"dte": int(k), "n": int(np.sum(dte == k)), "range_atr20_mean": _nanmean(rng[dte == k])}
                                             for k in sorted({int(x) for x in dte[np.isfinite(dte)]})[:10]])
    pcr = ctx.brick("pcr")
    return PhaseTable(ph, title, "RUN", [{"metric": "pcr_median", "value": float(np.nanmedian(pcr))},
                                         {"metric": "signed_gex", "value": f"{NA_DATA_MISSING} (PROXY columns never feed verdicts)"}])


# =============================================================================================
# Phases 5-6 — time structure and regime atlas
# =============================================================================================
def phase5(inp: XrayInputs) -> list[PhaseTable]:
    fr = inp.frame
    ctx = inp.ctx
    fwd = _fwd(fr, 3)
    rng = (fr.h - fr.l) / fr.atr20
    rows = [{"slot": int(s), "mean_fwd3_atr20": _nanmean(fwd[fr.slot == s]), "mean_range_atr20": _nanmean(rng[fr.slot == s])}
            for s in np.unique(fr.slot)]
    for name in ("first_15m", "lunch_lull", "last_hour"):
        sel = ctx.window_mask(name)
        rows.append({"slot": name, "mean_fwd3_atr20": _nanmean(fwd[sel]), "mean_range_atr20": _nanmean(rng[sel]),
                     "reversal_rate": _nanmean((np.sign(fwd) != np.sign(np.r_[np.nan, np.diff(fr.c)]))[sel])})
    out = [PhaseTable("5.1", "Slot profile + named IST windows (era_calendar.yaml)", "RUN", rows)]
    d = fr.day_ohlc()
    day_ret = (d["close"] - d["open"]) / fr.atr20[fr.day_end - 1]
    wd = ((fr.days.astype(np.int64) + 3) % 7)
    dom = (fr.days - fr.days.astype("datetime64[M]")).astype(np.int64) + 1
    rows = [{"bucket": f"weekday_{k}", "n_days": int(np.sum(wd == k)), "mean_day_ret_atr20": _nanmean(day_ret[wd == k])} for k in range(5)]
    rows += [{"bucket": "month_end(day>=28)", "n_days": int(np.sum(dom >= 28)), "mean_day_ret_atr20": _nanmean(day_ret[dom >= 28])},
             {"bucket": "month_start(day<=3)", "n_days": int(np.sum(dom <= 3)), "mean_day_ret_atr20": _nanmean(day_ret[dom <= 3])}]
    wom = (dom - 1) // 7 + 1
    rows += [{"bucket": f"week_of_month_{k}", "n_days": int(np.sum(wom == k)), "mean_day_ret_atr20": _nanmean(day_ret[wom == k])} for k in range(1, 6)]
    if "expiry_calendar" in ctx.available:
        dte_day = ctx.brick("dte")[fr.day_start]
        rows.append({"bucket": "expiry_day", "n_days": int(np.sum(dte_day == 0)), "mean_day_ret_atr20": _nanmean(day_ret[dte_day == 0])})
    else:
        rows.append({"bucket": "expiry_cycle / pre-post holiday", "n_days": 0, "mean_day_ret_atr20": NA_DATA_MISSING})
    out.append(PhaseTable("5.2", "Weekday, week-of-month, month-end/start, expiry cycle", "RUN", rows))
    n_blocks = int(inp.cfg["stats"]["n_blocks"])
    blocks = np.array_split(np.arange(fr.n_days), n_blocks)
    coarse = False
    cells = []
    for s in np.unique(fr.slot):
        for k in range(5):
            sel = (fr.slot == s) & (wd[fr.day_idx] == k)
            per_block = [int(np.sum(np.isin(fr.day_idx[sel], b))) for b in blocks]
            coarse = coarse or min(per_block) < 30
            cells.append({"slot": int(s), "weekday": k, "n_days": int(sel.sum()), "mean_fwd3_atr20": _nanmean(fwd[sel])})
    out.append(PhaseTable("5.3", "slot x weekday heat-table" + (" (COARSE-ONLY: a n_blocks cell < 30 days)" if coarse else ""),
                          "RUN", cells, "COARSE-ONLY" if coarse else ""))
    top_days = np.argsort(np.abs(day_ret))[::-1][:5]
    share = float(np.nansum(np.abs(day_ret[top_days])) / np.nansum(np.abs(day_ret)))
    terc = np.digitize(fr.atr20[fr.day_end - 1], np.nanquantile(fr.atr20[fr.day_end - 1], [1 / 3, 2 / 3]))
    rows = [{"check": "top5_days_share_of_abs_day_ret", "value": share, "flag": "FEW-DAY-DEPENDENT" if share > 0.5 else ""}]
    rows += [{"check": f"vol_tercile_{t + 1}_mean_day_ret", "value": _nanmean(day_ret[terc == t])} for t in range(3)]
    out.append(PhaseTable("5.4", "Cause check (vol tercile control, FEW-DAY-DEPENDENT)", "RUN", rows))
    return out


def day_types(frame: BarFrame, fit_day_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Discovery-frozen day-type rules (DESCRIPTIVE-HINDSIGHT)."""
    d = frame.day_ohlc()
    rng = d["high"] - d["low"]
    body = d["close"] - d["open"]
    prev_c = np.r_[np.nan, d["close"][:-1]]
    gap = d["open"] - prev_c
    eff = np.abs(body) / rng
    fit = np.asarray(fit_day_mask, bool)
    cut = {"eff_hi": float(np.nanquantile(eff[fit], 0.67)), "eff_lo": float(np.nanquantile(eff[fit], 0.33)),
           "rng_hi": float(np.nanquantile(rng[fit], 0.80)), "rng_lo": float(np.nanquantile(rng[fit], 0.20)),
           "gap_hi": float(np.nanquantile(np.abs(gap[fit & np.isfinite(gap)]), 0.80)) if np.isfinite(gap[fit]).any() else float("inf")}
    types = np.full(frame.n_days, "other", dtype=object)
    prev_rng = np.r_[np.nan, rng[:-1]]
    for i in range(frame.n_days):
        if abs(gap[i]) >= cut["gap_hi"] and np.sign(body[i]) == np.sign(gap[i]):
            types[i] = "gap_and_go"
        elif abs(gap[i]) >= cut["gap_hi"]:
            types[i] = "gap_fill"
        elif eff[i] >= cut["eff_hi"]:
            types[i] = "trend_up" if body[i] > 0 else "trend_down"
        elif rng[i] >= cut["rng_hi"]:
            types[i] = "expansion"
        elif np.isfinite(prev_rng[i]) and rng[i] <= cut["rng_lo"]:
            types[i] = "inside_compressed"
        elif eff[i] <= cut["eff_lo"]:
            types[i] = "range_chop"
        else:
            types[i] = "reversal"
    return types, cut


def coarse_type(t: str) -> str:
    return "trend" if t in ("trend_up", "trend_down", "gap_and_go") else ("range_chop" if t in ("range_chop", "inside_compressed") else "other")


def phase6(inp: XrayInputs) -> list[PhaseTable]:
    fr = inp.frame
    types, cut = day_types(fr, inp.fit_day_mask)
    vc = pd.Series(types).value_counts()
    out = [PhaseTable("6.1", "Day-type rules (DESCRIPTIVE-HINDSIGHT, discovery-frozen)", "RUN",
                      [{"day_type": k, "n_days": int(v)} for k, v in sorted(vc.items())] + [{"day_type": "cutoffs", "n_days": 0, **cut}])]
    trans = pd.crosstab(pd.Series(types[:-1], name="t"), pd.Series(types[1:], name="t+1"))
    rows = [{"from": f, **{f"to_{t}": int(trans.loc[f, t]) for t in trans.columns}, "label": "ANECDOTAL" if trans.loc[f].sum() < 30 else ""} for f in trans.index]
    ct = pd.crosstab(pd.Series([coarse_type(t) for t in types[:-1]], name="t"), pd.Series([coarse_type(t) for t in types[1:]], name="t+1"))
    rows += [{"from": f"coarse:{f}", **{f"to_{t}": int(ct.loc[f, t]) for t in ct.columns}} for f in ct.index]
    out.append(PhaseTable("6.2", "day_type(t) -> day_type(t+1) matrix + 3-group coarse", "RUN", rows))
    out.append(_early_predictability(inp, types))
    d = fr.day_ohlc()
    rng = d["high"] - d["low"]
    roll = pd.Series(rng).rolling(20).mean()
    x = (rng - np.nanmean(rng)) / (np.nanstd(rng) or 1.0)
    k_ = float(inp.cfg["playground"]["cusum_k_sd"])
    h_ = float(inp.cfg["playground"]["cusum_h_sd"])
    pos = neg = 0.0
    changes = []
    for i, v in enumerate(np.nan_to_num(x)):
        pos, neg = max(0.0, pos + v - k_), min(0.0, neg + v + k_)
        if pos > h_ or neg < -h_:
            changes.append(str(fr.days[i]))
            pos = neg = 0.0
    half = fr.n_days // 2
    eras = pd.Series([inp.eras.get(str(dd), "UNKNOWN") for dd in fr.days])
    rows = [{"metric": "rolling20_range_last", "value": float(roll.iloc[-1]) if len(roll) else np.nan},
            {"metric": "cusum_change_points", "value": ";".join(changes[:20])},
            {"metric": "first_half_mean_range", "value": _nanmean(rng[:half])},
            {"metric": "second_half_mean_range", "value": _nanmean(rng[half:])}]
    rows += [{"metric": f"era_{e}_mean_range", "value": _nanmean(rng[(eras == e).to_numpy()])} for e in sorted(eras.unique())]
    out.append(PhaseTable("6.4", "Rolling vol/range, CUSUM change points, halves, era splits", "RUN", rows))
    out.append(_regime_flip(inp))
    return out


def _early_predictability(inp: XrayInputs, types: np.ndarray) -> PhaseTable:
    if not sklearn_available() or inp.run_ctx.profile == "lite":
        return PhaseTable("6.3", "Early predictability at checkpoints", STATUS_ENV, [], "NOT-RUN(ENV): scikit-learn unavailable (optional [models])")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    fr = inp.frame
    target = np.array([coarse_type(t) for t in types])
    rows = []
    gap = inp.ctx.brick("gap_atr")
    sofar = inp.ctx.brick("day_ret_so_far")
    rng_sofar = inp.ctx.brick("day_range_pos_so_far")
    for cp in sorted(inp.ctx.cal.get("checkpoints", {})):
        mask = inp.ctx.checkpoint_mask(cp)
        idx = np.flatnonzero(mask)
        if idx.size < 40:
            rows.append({"checkpoint": cp, "status": "ANECDOTAL", "n_days": int(idx.size)})
            continue
        x = np.column_stack([gap[idx], sofar[idx], rng_sofar[idx]])
        y = target[fr.day_idx[idx]]
        ok = np.isfinite(x).all(axis=1)
        x, y = x[ok], y[ok]
        n_train = int(0.6 * y.size)
        if np.unique(y[:n_train]).size < 2:
            rows.append({"checkpoint": cp, "status": "single-class discovery", "n_days": int(y.size)})
            continue
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(x[:n_train], y[:n_train])
        pred = model.predict(x[n_train:])
        acc = float(np.mean(pred == y[n_train:]))
        majority = pd.Series(y[:n_train]).mode().iloc[0]
        base = float(np.mean(y[n_train:] == majority))
        rng_b = np.random.default_rng(inp.run_ctx.seed)
        hits = (pred == y[n_train:]).astype(float)
        boots = [np.mean(hits[rng_b.integers(0, hits.size, hits.size)]) for _ in range(500)]
        rows.append({"checkpoint": cp, "n_days": int(y.size), "accuracy": acc, "majority_baseline": base,
                     "acc_ci_lo": float(np.quantile(boots, 0.025)), "acc_ci_hi": float(np.quantile(boots, 0.975)),
                     "macro_f1": float(f1_score(y[n_train:], pred, average="macro")), "target": "3-group coarse"})
    return PhaseTable("6.3", "Early predictability (fixed regularised logistic, seed 42, coarse target)", "RUN", rows)


def _regime_flip(inp: XrayInputs) -> PhaseTable:
    """3-condition regime-flip rule: both cells N>=30, CIs on opposite sides of 0, diff q<=0.05."""
    fr = inp.frame
    er = inp.ctx.brick("er_20")
    fit = inp.fit_day_mask[fr.day_idx]
    lo_cut, hi_cut = np.nanquantile(er[fit], [1 / 3, 2 / 3])
    orb = inp.ctx.brick("dist_or_high_atr")
    first_break = np.zeros(fr.n, dtype=bool)
    for d in range(fr.n_days):
        s, e = fr.day_start[d], fr.day_end[d]
        hits = np.flatnonzero(orb[s:e] > 0)
        if hits.size:
            first_break[s + hits[0]] = True
    fwd = _fwd(fr, 5)
    rows = []
    cells = {}
    for name, sel in (("low_ER", er <= lo_cut), ("high_ER", er >= hi_cut)):
        m = first_break & sel
        ci = _day_ci(fwd, fr, m, inp.run_ctx.bootstrap_B, inp.run_ctx.seed)
        n_days = int(np.unique(fr.day_idx[m & np.isfinite(fwd)]).size)
        cells[name] = (ci, n_days, fwd[m & np.isfinite(fwd)])
        rows.append({"cell": name, "n_days": n_days, "mean": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]})
    (c1, n1, v1), (c2, n2, v2) = cells["low_ER"], cells["high_ER"]
    opposite = bool(np.isfinite(c1["hi"]) and np.isfinite(c2["lo"]) and ((c1["hi"] < 0 < c2["lo"]) or (c2["hi"] < 0 < c1["lo"])))
    p_diff = float(sps.mannwhitneyu(v1, v2).pvalue) if v1.size and v2.size else 1.0
    verdict = "REGIME FLIP" if (n1 >= 30 and n2 >= 30 and opposite and p_diff <= 0.05) else "NO RELIABLE FLIP"
    rows.append({"cell": "verdict", "n_days": min(n1, n2), "mean": p_diff, "ci_lo": float(opposite), "ci_hi": float("nan"), "label": verdict})
    return PhaseTable("6.5", "Behaviour by regime & AP10 3-condition regime-flip rule", "RUN", rows)


# =============================================================================================
# Phase 7-8 — disciplined edge hunt and tradability
# =============================================================================================
def interaction_search(inp: XrayInputs, tests: Sequence[dict[str, Any]], compiled: dict[str, CompiledCard]) -> list[dict[str, Any]]:
    """7.3: top-10 ROBUST-D discovery parents by lower 95% CI -> <= 45 AND pairs, own BH family."""
    fr = inp.frame
    robust = [t for t in tests if t.get("stability") == "ROBUST" and t.get("status") == "EXECUTED"]
    robust.sort(key=lambda t: (-(t["ci_lo"] if t["sign"] > 0 else -t["ci_hi"]), t["test_id"]))
    parents = []
    seen = set()
    for t in robust:
        if t["test_id"] not in seen:
            parents.append(t)
            seen.add(t["test_id"])
        if len(parents) == 10:
            break
    rows = []
    masks = {}
    for t in parents:
        cc = compiled[t["test_id"]]
        m, d = cc.evaluate(inp.ctx)
        masks[t["test_id"]] = (m, d, t)
    ids = sorted(masks)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ma, da, ta = masks[a]
            mb, db, _tb = masks[b]
            m = ma & mb & (da == db)
            h = int(ta["horizon"])
            if not m.any():
                continue
            res = event_study(fr, m, direction=da, side=ta["side"], target=ta["target"], horizon=h,
                              day_mask=inp.fit_day_mask, test_id=f"{a}&{b}", ctx=inp.run_ctx)
            rows.append({"pair": f"{a} AND {b}", "horizon": h, "n_events": res.n_events, "effective_n": res.effective_n,
                         "lift_atr20": res.lift_atr, "p_val": res.p_val})
    if rows:
        q = bh_qvalues([r["p_val"] if np.isfinite(r["p_val"]) else 1.0 for r in rows])
        for r, qv in zip(rows, q, strict=True):
            r["q_pair"] = float(qv)
            r["passes_q_pair"] = bool(qv <= 0.05)
    return rows


def phase8(inp: XrayInputs, tests: Sequence[dict[str, Any]], compiled: dict[str, CompiledCard]) -> list[PhaseTable]:
    fr = inp.frame
    rng_pts = fr.h - fr.l
    d = fr.day_ohlc()
    hour = (fr.minute - fr.session_open_minute()[fr.day_idx]) // 60
    hour_rng = pd.DataFrame({"day": fr.day_idx, "hour": hour, "h": fr.h, "l": fr.l}).groupby(["day", "hour"]).agg(h=("h", "max"), l=("l", "min"))
    hr = (hour_rng["h"] - hour_rng["l"]).to_numpy()
    rows = [{"scale": "bar", "median_range_pts": float(np.nanmedian(rng_pts)), "p75_range_pts": _q(rng_pts, [0.75])[0]},
            {"scale": "hour", "median_range_pts": float(np.nanmedian(hr)), "p75_range_pts": _q(hr, [0.75])[0]},
            {"scale": "day", "median_range_pts": float(np.nanmedian(d["high"] - d["low"])), "p75_range_pts": _q(d["high"] - d["low"], [0.75])[0]}]
    out = [PhaseTable("8.1", "Noise floor tables (see 2.7 excursions)", "RUN", rows)]
    atr_pts = float(np.nanmedian(fr.atr20))
    rows = []
    for series, cost in sorted(inp.costs.items()):
        rows.append({"cost_series": series, "round_trip_cost_pts": cost if cost is not None else "USER TO CONFIRM",
                     "break_even_lift_atr20": (cost / atr_pts) if cost is not None and atr_pts > 0 else "UNKNOWN"})
    for t in sorted((t for t in tests if t.get("status") == "EXECUTED" and t.get("deflated_lift") is not None),
                    key=lambda t: -abs(t["deflated_lift"]))[:10]:
        rows.append({"cost_series": t["test_id"], "deflated_lift_atr20": t["deflated_lift"], "tag": t.get("t_tag")})
    out.append(PhaseTable("8.2", "Cost vs |deflated_lift| and break-even round-trip cost", "RUN", rows))
    rows = []
    for t in [t for t in tests if t.get("status") == "EXECUTED"][:50]:
        rows.append({"test_id": t["test_id"], "events_per_day": t["n_events"] / max(1, fr.n_days),
                     "event_days_share": t["effective_n"] / max(1, fr.n_days)})
    out.append(PhaseTable("8.3", "Opportunity density", "RUN", rows))
    if "opt_1m" in inp.ctx.available:
        decay = _phase4_item(inp, "4.3", "Option premium decay burden & break-even move")
        decay.phase = "8.4"
        out.append(decay)
    else:
        out.append(_na("8.4", "Option premium decay burden & break-even move", ["opt_1m"]))
    rows = []
    executed = sorted((t for t in tests if t.get("status") == "EXECUTED" and t["test_id"] in compiled and compiled[t["test_id"]].ok),
                      key=lambda t: -abs(t["lift_atr"] if np.isfinite(t["lift_atr"]) else 0))[:5]
    for t in executed:
        cc = compiled[t["test_id"]]
        m, dirn = cc.evaluate(inp.ctx)
        base = t["lift_atr"]
        for delay in (1, 2):
            md = np.r_[np.zeros(delay, bool), m[:-delay]] & np.r_[np.zeros(delay, bool), fr.day_idx[delay:] == fr.day_idx[:-delay]]
            dd = np.r_[np.ones(delay), dirn[:-delay]]
            res = event_study(fr, md, direction=dd, side=t["side"], target=t["target"], horizon=int(t["horizon"]),
                              day_mask=inp.fit_day_mask, ctx=inp.run_ctx, bootstrap=False)
            rows.append({"test_id": t["test_id"], "delay_bars": delay, "lift_retained": res.lift_atr / base if base else np.nan})
    out.append(PhaseTable("8.5", "1-bar & 2-bar delay lift retention", "RUN", rows))
    has_daily = any(t.get("target") == "T8" for t in tests)
    styles = []
    for style, cost_key in (("intraday option buying", "option_buyer"), ("option selling", "option"),
                            ("futures", "underlying"), ("positional", "underlying")):
        if style == "positional" and not has_daily:
            verdict = "NOT TESTED"
        elif inp.costs.get(cost_key) is None:
            verdict = "USER TO CONFIRM"
        else:
            best = max((abs(t["deflated_lift"]) for t in tests if t.get("deflated_lift") is not None), default=0.0)
            verdict = "NO EDGE FOUND" if best * atr_pts < inp.costs[cost_key] else "SEE FINDINGS"
        styles.append({"style": style, "verdict": verdict})
    out.append(PhaseTable("8.6", "Four style verdicts (diagnosis only)", "RUN", styles))
    return out


# =============================================================================================
# AP runner, era tags and 2A.3 verdicts
# =============================================================================================
def is_option_structure(card: dict[str, Any]) -> bool:
    return any(t in OPTION_STRUCTURE_TOKENS for t in card.get("data_needed", []))


def era_consistency(card: dict[str, Any], result: EventResult, eras: dict[str, str], cal: dict[str, Any]) -> str:
    """ERA-CONSISTENT / ERA-SPECIFIC / ERA-B-ONLY (Ω-7, FR-20)."""
    by_era: dict[str, list[float]] = {}
    for day, x in zip(result.event_days, result.x_d, strict=True):
        by_era.setdefault(eras.get(str(day), "UNKNOWN"), []).append(float(x))
    era_a = next((e for e in cal.get("eras", []) if e["id"] == "A"), {})
    if is_option_structure(card) and "A" in by_era and not era_a.get("option_structure_comparable", True):
        return "ERA-SPECIFIC"
    a_vals = by_era.get("A", [])
    b_vals = [v for k, vals in by_era.items() if k.startswith("B") for v in vals]
    if not a_vals:
        return "ERA-B-ONLY" if b_vals else "ERA-UNKNOWN"
    if not b_vals:
        return "ERA-A-ONLY"
    if len(a_vals) >= 30 and len(b_vals) >= 30 and np.sign(np.mean(a_vals)) != np.sign(np.mean(b_vals)):
        return "ERA-SPECIFIC"
    return "ERA-CONSISTENT"


def make_test_row(res: EventResult, card: dict[str, Any], meta: dict[str, Any], *, lane: str, approach_id: str) -> dict[str, Any]:
    return {**res.summary(), "status": "EXECUTED", "approach_id": approach_id, "lane": lane,
            "group": meta.get("group", ""), "phase": meta.get("phase", ""), "card_id": card["card_id"],
            "cost_series": card.get("cost_series"), "event_days_list": res.event_day_list(), "grade": "D",
            "skeptic_ran": 0, "label": "EXPLORATORY"}


def run_ap_lane(inp: XrayInputs, lib: BrickLibrary, *, ap_path: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, CompiledCard], dict[str, list[EventResult]]]:
    """Compile + fit + run every AP card on Playground days; N/A rows name missing inputs."""
    runner = TestRunner(inp.run_ctx, inp.frame, day_mask=inp.fit_day_mask)
    tests: list[dict[str, Any]] = []
    compiled: dict[str, CompiledCard] = {}
    results: dict[str, list[EventResult]] = {}
    for meta, card in load_ap_cards(ap_path):
        aid = meta["approach_id"]
        cc = compile_card(card, lib, available=inp.ctx.available, cal=inp.ctx.cal,
                          t7_enabled=bool(inp.cfg["targets"]["T7_enabled"]))
        compiled[card["card_id"]] = cc
        if not cc.ok:
            reason = cc.reason
            runner.run_batch([{"test_id": card["card_id"], "approach_id": aid, "family": "BASELINE",
                               "phase": meta.get("phase", ""), "target": card["target"], "horizons": card["horizons"],
                               "side": card["side"], "na_reason": reason}])
            for h in card["horizons"]:
                tests.append({"test_id": card["card_id"], "card_id": card["card_id"], "approach_id": aid,
                              "group": meta["group"], "status": cc.status, "reason": reason, "horizon": int(h),
                              "side": card["side"], "target": card["target"], "missing": list(cc.missing),
                              "lane": "AP", "grade": None})
            continue
        spec = cc.spec(inp.ctx, inp.fit_day_mask, approach_id=aid, phase=meta.get("phase", ""))
        spec["family"] = "BASELINE"
        res_list = runner.run_batch([spec])
        results[card["card_id"]] = res_list
        for res in res_list:
            row = make_test_row(res, card, meta, lane="AP", approach_id=aid)
            row["era_tag"] = era_consistency(card, res, inp.eras, inp.ctx.cal)
            tests.append(row)
        for h in card["horizons"]:
            if not any(r.horizon == h for r in res_list):
                tests.append({"test_id": card["card_id"], "card_id": card["card_id"], "approach_id": aid,
                              "group": meta["group"], "status": "SKIPPED", "reason": "horizon cap", "horizon": int(h),
                              "side": card["side"], "target": card["target"], "lane": "AP", "grade": None})
    executed = [t for t in tests if t["status"] == "EXECUTED"]
    q = bh_qvalues([t["p_val"] if np.isfinite(t["p_val"]) else 1.0 for t in executed])
    per_app: dict[str, int] = {}
    for t in executed:
        per_app[t["approach_id"]] = per_app.get(t["approach_id"], 0) + 1
    for t, qv in zip(executed, q, strict=True):
        t["q_disc"] = float(qv)
        m_app = per_app[t["approach_id"]]
        t["m_app"] = m_app
        t["gate_mde"] = gate_adjusted_mde(t["se"], float(inp.cfg["stats"]["q_disc"]), m_app) if np.isfinite(t["se"]) else float("nan")
        if inp.deflated is not None:
            res = next(r for r in results[t["card_id"]] if r.horizon == t["horizon"])
            t["deflated_lift"] = inp.deflated(res, t["target"])
    return tests, compiled, results


def needed_effect(costs: dict[str, float | None], cost_series: str, atr_points: float) -> float | None:
    cost = costs.get(cost_series) if cost_series != "structure_only" else None
    if cost is None or not math.isfinite(atr_points) or atr_points <= 0:
        return None
    return cost / atr_points


def verdict_2a3(approach: dict[str, Any], tests: Sequence[dict[str, Any]], *, costs: dict[str, float | None],
                atr_points: float, day_limited: bool, degraded: bool = False) -> dict[str, Any]:
    """Deterministic §9.2 2A.3 verdict for one approach (shared by FILE-028 and FILE-037)."""
    aid = approach["approach_id"]
    mine = [t for t in tests if t.get("approach_id") == aid]
    base = {"approach_id": aid, "group": approach["group"], "name": approach.get("name", "")}
    if degraded:
        return {**base, "verdict": "N/A DEGRADED", "reason": "MODE: DEGRADED", "label": ""}
    runnable = [t for t in mine if t.get("status") == "EXECUTED"]
    if not runnable:
        missing = sorted({m for t in mine for m in t.get("missing", [])})
        reason = f"{NA_DATA_MISSING}: {missing}" if missing else (mine[0]["reason"] if mine else "no tests")
        return {**base, "verdict": NA_DATA_MISSING, "reason": reason, "missing": ";".join(missing), "label": "",
                "runnable_tests": 0, "tests_n30": 0}

    def rank_key(t: dict[str, Any]) -> tuple[int, int, float]:
        return (GRADE_ORDER.get(t.get("grade") or "D", 0), TAG_ORDER.get(t.get("t_tag") or "T?", 2),
                -float(t.get("q_disc", 1.0)))

    best_by_side = {}
    for side in ("continuation", "reversal", "both"):
        side_tests = [t for t in runnable if t["side"] == side]
        if side_tests:
            best_by_side[side] = max(side_tests, key=rank_key)
    best = max(best_by_side.values(), key=rank_key)
    se = np.array([t["se"] for t in runnable], float)
    nominal_med = float(np.nanmedian(2.8 * se)) if np.isfinite(se).any() else float("nan")
    gated = [t["gate_mde"] for t in runnable if t["effective_n"] >= 30 and np.isfinite(t.get("gate_mde", np.nan))]
    gate_med = float(np.median(gated)) if gated else float("nan")
    cost_series = best.get("cost_series") or "option_buyer"
    need = needed_effect(costs, cost_series, atr_points)
    n30 = sum(1 for t in runnable if t["effective_n"] >= 30)
    out = {**base, "best_grade": best.get("grade") or "D", "best_test": best["test_id"], "best_side": best["side"],
           "best_tag": best.get("t_tag") or "T?", "nominal_mde_med": nominal_med, "gate_mde_med": gate_med,
           "gate_mde_min": float(np.min(gated)) if gated else float("nan"),
           "gate_mde_max": float(np.max(gated)) if gated else float("nan"),
           "family_gate_mde_med": gate_med, "needed_effect": need if need is not None else "UNKNOWN",
           "runnable_tests": len(runnable), "tests_n30": n30, "missing": "", "label": ""}
    grade = best.get("grade") or "D"
    if grade in ("A", "B", "Ω-A", "Ω-B") and int(best.get("skeptic_ran", 0)) >= 3 and not day_limited and best.get("t_tag") in ("T1", "T2", "T?", "STRUCTURE-ONLY"):
        return {**out, "verdict": "SUPPORTED FOR HYPOTHESIS", "reason": f"best {best['test_id']} grade {grade}, tag {best.get('t_tag')}"}
    if grade in ("A", "B", "Ω-A", "Ω-B") and best.get("t_tag") == "T3":
        return {**out, "verdict": "CONTRADICTED", "label": "REAL BUT UNTRADABLE",
                "reason": f"statistically real but below matching cost (gate MDE {gate_med:.3f} ATR20)"}
    no_abc = not any((t.get("grade") or "D") in ("A", "B", "C", "Ω-A", "Ω-B", "Ω-C") for t in runnable)
    if no_abc and n30 >= 0.5 * len(runnable):
        if need is None:
            return {**out, "verdict": "CONTRADICTED", "label": "STAT-ONLY",
                    "reason": f"0 A/B/C tests, {n30}/{len(runnable)} tests with effective N >= 30; gate-adjusted MDE {gate_med:.3f} ATR20; cost unknown (STAT-ONLY)"}
        if np.isfinite(gate_med) and gate_med <= need:
            return {**out, "verdict": "CONTRADICTED",
                    "reason": f"0 A/B/C tests; gate-adjusted MDE {gate_med:.3f} <= needed effect {need:.3f} ATR20"}
    reasons = []
    if day_limited:
        reasons.append("DAY-LIMITED (< 120 usable days)")
    if no_abc and n30 < 0.5 * len(runnable):
        reasons.append(f"only {n30}/{len(runnable)} tests have effective N >= 30")
    if need is not None and np.isfinite(gate_med) and gate_med > need:
        reasons.append(f"gate-adjusted MDE {gate_med:.3f} > needed effect {need:.3f} ATR20")
    if not no_abc:
        reasons.append(f"best grade {grade} lacks Skeptic coverage or tag")
    return {**out, "verdict": "INCONCLUSIVE", "reason": "; ".join(reasons) or "no rule matched"}


def rollup_verdict(approach: dict[str, Any], member_rows: Sequence[dict[str, Any]], vr_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """AP32 / AP33 roll-ups (0 tests): member side verdicts + Phase 2.2 variance-ratio evidence."""
    roll = approach["rollup"]
    side = roll["side"]
    members = [r for r in member_rows if r["approach_id"] in roll["members"]]
    vr_hits = [r for r in vr_rows if r.get("excludes_1") and ((r["vr"] > 1) if side == "continuation" else (r["vr"] < 1))]
    sup = [r for r in members if r["verdict"] == "SUPPORTED FOR HYPOTHESIS" and r.get("best_side") in (side, "both")]
    runnable = [r for r in members if r["verdict"] != NA_DATA_MISSING]
    if sup:
        verdict = "SUPPORTED FOR HYPOTHESIS"
    elif not runnable:
        verdict = NA_DATA_MISSING
    elif all(r["verdict"] == "CONTRADICTED" for r in runnable) and not vr_hits:
        verdict = "CONTRADICTED"
    else:
        verdict = "INCONCLUSIVE"
    return {"approach_id": approach["approach_id"], "group": approach["group"], "name": approach.get("name", ""),
            "verdict": verdict, "runnable_tests": 0, "tests_n30": 0, "best_grade": "", "label": "ROLL-UP (0 tests)",
            "reason": f"{len(sup)} supported {side} members; {len(vr_hits)} variance-ratio horizons with CI excluding 1",
            "missing": ""}


def group_verdicts(rows_11a: Sequence[dict[str, Any]], groups: dict[str, str]) -> list[dict[str, Any]]:
    """Table 11B per group G1-G10."""
    out = []
    for g in sorted(groups, key=lambda x: int(x[1:])):
        mem = [r for r in rows_11a if r["group"] == g]
        s = sum(1 for r in mem if r["verdict"] == "SUPPORTED FOR HYPOTHESIS")
        c = sum(1 for r in mem if r["verdict"] == "CONTRADICTED")
        na = sum(1 for r in mem if r["verdict"] in (NA_DATA_MISSING, "N/A DEGRADED"))
        i = len(mem) - s - c - na
        runnable = [r for r in mem if r["verdict"] not in (NA_DATA_MISSING, "N/A DEGRADED")]
        if s:
            verdict = "SUPPORTED"
        elif runnable and all(r["verdict"] == "CONTRADICTED" for r in runnable):
            verdict = "CONTRADICTED"
        elif not runnable:
            verdict = "N/A"
        else:
            verdict = "INCONCLUSIVE"
        grades = [r.get("best_grade") or "D" for r in mem if r.get("best_grade")]
        best = max(grades, key=lambda x: GRADE_ORDER.get(x, 0)) if grades else "—"
        out.append({"group": g, "group_name": groups[g], "members_S": s, "members_C": c, "members_I": i,
                    "members_NA": na, "best_evidence_grade": best, "verdict": verdict,
                    "reason": f"S/C/I/NA = {s}/{c}/{i}/{na}"})
    return out


def compute_verdicts(inp: XrayInputs, tests: Sequence[dict[str, Any]], vr_rows: Sequence[dict[str, Any]],
                     *, ap_path: str | Path | None = None, degraded: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tables 11A (36 rows) and 11B (10 groups) by the deterministic 2A.3 rule."""
    from bn_forensics import load_yaml

    doc = load_yaml(ap_path) if ap_path else None
    meta = load_ap_meta(ap_path)
    groups = doc["groups"] if doc else _groups_from_default()
    atr_points = float(np.nanmedian(inp.frame.atr20))
    rows = [verdict_2a3(m, tests, costs=inp.costs, atr_points=atr_points, day_limited=inp.day_limited, degraded=degraded)
            for m in meta if not m.get("rollup")]
    for m in meta:
        if m.get("rollup"):
            rows.append(rollup_verdict(m, rows, vr_rows))
    rows.sort(key=lambda r: r["approach_id"])
    return rows, group_verdicts(rows, groups)


def _groups_from_default() -> dict[str, str]:
    from bn_forensics import asset_dir, load_yaml

    return load_yaml(asset_dir("config") / "ap_cards.yaml")["groups"]


def run_xray(inp: XrayInputs, lib: BrickLibrary, *, ap_path: str | Path | None = None) -> XrayResult:
    """Phases 1-8 + AP01-AP36 + 2A.3 verdicts on Playground days (writes tables/*.csv)."""
    if not np.asarray(inp.fit_day_mask, bool).any():
        raise InputError("X-ray needs at least one Playground day")
    tests, compiled, results = run_ap_lane(inp, lib, ap_path=ap_path)
    phases: list[PhaseTable] = []
    phases += phase1(inp)
    p2 = phase2(inp)
    phases += p2
    phases += phase3(inp)
    phases += phase4(inp)
    phases += phase5(inp)
    phases += phase6(inp)
    interactions = interaction_search(inp, tests, compiled)
    phases.append(PhaseTable("7.3", "Pairwise interaction search (top-10 ROBUST-D parents, own BH family)", "RUN", interactions))
    if sklearn_available() and inp.run_ctx.profile != "lite":
        phases.append(PhaseTable("7.4", "Predictability ceiling & MODEL cards", "RUN",
                                 [{"note": "MODEL cards run through Hunt Mode H7 (fold-local cut-offs, frozen OOS)"}]))
    else:
        phases.append(PhaseTable("7.4", "Predictability ceiling & MODEL cards", STATUS_ENV, [], "NOT-RUN(ENV): scikit-learn unavailable"))
    leader = sorted((t for t in tests if t["status"] == "EXECUTED"), key=lambda t: -(t["ci_lo"] if t["sign"] > 0 else -t["ci_hi"]))[:50]
    phases.append(PhaseTable("7.5", "Leaderboard (discovery lower 95% CI bound)", "RUN",
                             [{"rank": i + 1, "test_id": t["test_id"], "horizon": t["horizon"], "lift_atr20": t["lift_atr"],
                               "ci_lo": t["ci_lo"], "ci_hi": t["ci_hi"], "stability": t["stability"]} for i, t in enumerate(leader)]))
    executed = [t for t in tests if t["status"] == "EXECUTED"]
    expected_fp = float(sum(min(1.0, t["q_disc"]) for t in executed if t["q_disc"] <= 0.05))
    phases.append(PhaseTable("7.6", "Chance expectation", "RUN",
                             [{"executed_tests": len(executed), "q_disc_le_0.05": sum(1 for t in executed if t["q_disc"] <= 0.05),
                               "expected_false_positives": expected_fp}]))
    phases += phase8(inp, tests, compiled)
    vr_rows = next((r for p in p2 if p.phase == "2.2" for r in [p.rows]), [])
    vr_rows = [r for r in vr_rows if str(r.get("lag", "")).startswith("VR(")]
    v11a, v11b = compute_verdicts(inp, tests, vr_rows, ap_path=ap_path)
    era_tags = {t["card_id"]: t["era_tag"] for t in tests if t.get("era_tag")}
    if inp.tables_dir is not None:
        write_xray_tables(inp.tables_dir, phases, v11a, v11b)
    return XrayResult(phases=phases, tests=tests, compiled=compiled, results=results, verdicts_11a=v11a,
                      verdicts_11b=v11b, era_tags=era_tags, interactions=interactions)


def write_xray_tables(tables_dir: str | Path, phases: Sequence[PhaseTable], v11a: Sequence[dict[str, Any]],
                      v11b: Sequence[dict[str, Any]]) -> list[Path]:
    """``tables/phase_<id>.csv`` + ``approach_verdicts_11a.csv`` + ``approach_verdicts_11b.csv``."""
    root = Path(tables_dir)
    written = []
    for p in phases:
        rows = p.rows or [{"status": p.status, "note": p.note}]
        fields = sorted({k for r in rows for k in r})
        written.append(write_csv(root / f"phase_{p.phase.replace('.', '_')}.csv", ["phase", "status", *fields],
                                 [{"phase": p.phase, "status": p.status, **r} for r in rows]))
    written.append(write_csv(root / "approach_verdicts_11a.csv", VERDICT_11A_FIELDS, v11a))
    written.append(write_csv(root / "approach_verdicts_11b.csv", VERDICT_11B_FIELDS, v11b))
    return written


__all__ = [
    "GRADE_ORDER",
    "TAG_ORDER",
    "VERDICT_11A_FIELDS",
    "VERDICT_11B_FIELDS",
    "PhaseTable",
    "XrayInputs",
    "XrayResult",
    "acf",
    "coarse_type",
    "compute_verdicts",
    "day_types",
    "era_consistency",
    "group_verdicts",
    "hurst_rs",
    "interaction_search",
    "is_option_structure",
    "ljung_box",
    "make_test_row",
    "needed_effect",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase8",
    "rollup_verdict",
    "run_ap_lane",
    "run_xray",
    "runs_test",
    "variance_ratio",
    "verdict_2a3",
    "write_xray_tables",
    "zigzag_legs",
]
