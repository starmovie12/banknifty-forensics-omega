"""Idea Card v2 compiler, Brick Leak Gate, Boundary Check and Proposal gates (FILE-029; FR-08, FR-21).

Pipeline for every card (RULE / MODEL / PROFILE, lanes AP / M / B / USER):

1. schema validation (FILE-040);
2. static Leak Gate: any ``t+k`` / future / full-day / EOD reference -> ``LEAK-REJECTED`` (R8);
3. Boundary Check: entry/exit/stop/target-price/sizing/win-rate/first-touch -> ``BOUNDARY-REJECTED``
   (NG-1, NG-2, FM-30);
4. availability gate: vendor values used before ``max(bar close, available_from_ts)``
   (``EOD_INCLUSIVE`` intraday, ``iv_daily`` before its check time, ``xcheck`` day-D) -> ``LEAK-REJECTED``;
5. data gate: missing data -> ``N/A — DATA MISSING`` naming the missing inputs (+ DRT upstream);
6. DSL parse + type check (no ``eval``; recursive-descent parser), max 3 bricks per condition;
7. dynamic +1-bar shift Leak Gate on every brick used (canary ``close_{t+1}`` must be caught).

Compiled cards freeze discovery cut-offs on Playground days (``fit``) and evaluate unchanged on
Court days. Proposals (BRICK / TARGET / PERSONA / MECHANISM) pass their gates before being written
to ``proposals/`` (symlink/traversal-safe) and ``bricks_pending.yaml``.
"""

from __future__ import annotations

import copy
import datetime as dt
import itertools
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bn_forensics import (
    NA_DATA_MISSING,
    InputError,
    asset_dir,
    atomic_write_yaml,
    load_yaml,
    parse_hhmm,
    safe_path,
    sha256_json,
)
from bn_forensics.core_stats import (
    BarFrame,
    ema,
    jaccard,
    lag_within_day,
    rolling_mean_strict,
    rolling_std_strict,
    wilder_smooth,
)
from bn_forensics.schemas_validator import sanitize_text, validate

LEAK_TOKENS = (
    r"\{\s*t\s*\+", r"\bt\s*\+\s*\d", r"\blead\s*\(", r"\bshift\s*\(\s*-", r"\bfuture", r"\bnext_",
    r"\bday_high\b", r"\bday_low\b", r"\bday_close\b", r"\bday_type\b", r"\bpoc_today\b", r"\beod_",
    r"EOD_INCLUSIVE", r"\bfinal_", r"\bvalidation_", r"\bcourt_", r"\bholdout_",
)
BOUNDARY_PATTERNS = (
    r"(?<!re-)\bentry\b", r"\benter\b", r"\bexit\b", r"\bstop[- ]?loss\b", r"\bstop\b", r"\btarget[- ]price\b",
    r"\btake[- ]profit\b", r"\bposition[- ]?siz", r"\bsizing\b", r"\blots?\b", r"\bwin[- ]?rate\b",
    r"\bfirst[- ]?touch\b", r"\bbarrier\b", r"\bpnl\b", r"\bp&l\b", r"\bequity curve\b", r"\bbacktest",
    r"\bbuy\b", r"\bsell\b", r"\btrailing\b", r"\bhit\b[^.]{0,40}\bbefore\b",
    r"[+-]?\d+(\.\d+)?\s*atr\w*\s+before\s+[+-]?\d", r"\breached\b[^.]{0,40}\bbefore\b",
)
FUNC_ARITY = {
    "AND": (2, 8), "OR": (2, 8), "NOT": (1, 1), "gt": (2, 2), "gte": (2, 2), "lt": (2, 2),
    "lte": (2, 2), "eq": (2, 2), "q10": (1, 1), "q25": (1, 1), "q75": (1, 1), "q90": (1, 1),
    "q95": (1, 1), "q99": (1, 1), "tercile": (1, 1), "sign": (1, 1), "abs": (1, 1), "lag": (2, 2),
    "A_then_B_within": (3, 3), "cross_above": (2, 2), "cross_below": (2, 2), "first_in_day": (1, 1),
    "in_window": (1, 1), "at_checkpoint": (1, 1), "DIR": (2, 2), "MODEL": (2, 9),
}
QUANTILE_LEVELS = {"q10": (0.10, "le"), "q25": (0.25, "le"), "q75": (0.75, "ge"),
                   "q90": (0.90, "ge"), "q95": (0.95, "ge"), "q99": (0.99, "ge")}
STATUS_OK = "OK"
STATUS_LEAK = "LEAK-REJECTED"
STATUS_BOUNDARY = "BOUNDARY-REJECTED"
STATUS_NA = NA_DATA_MISSING
STATUS_INVALID = "INVALID"
STATUS_ENV = "NOT-RUN(ENV)"


class CompileError(InputError):
    """A card or proposal failed to parse or type-check."""


# =============================================================================================
# Brick library
# =============================================================================================
@dataclass(frozen=True)
class BrickDef:
    brick_id: str
    family: str
    impl: str
    params: dict[str, Any]
    formula: str
    available_at: str
    data_needed: tuple[str, ...]
    kind: str
    pending: bool = False


@dataclass
class BrickLibrary:
    """Concrete bricks expanded from ``config/bricks.yaml`` (+ ``bricks_pending.yaml``)."""

    bricks: dict[str, BrickDef]
    library_id: str
    families: tuple[str, ...]
    max_bricks: int = 3

    @classmethod
    def load(cls, path: str | Path | None = None, pending_path: str | Path | None = None) -> BrickLibrary:
        doc = load_yaml(path or asset_dir("config") / "bricks.yaml")
        if not isinstance(doc, dict) or doc.get("schema_version") != "1.0":
            raise InputError("bricks.yaml must carry schema_version '1.0'")
        bricks: dict[str, BrickDef] = {}
        for raw in doc["bricks"]:
            for concrete in expand_brick(raw):
                if concrete.brick_id in bricks:
                    raise InputError(f"duplicate brick id {concrete.brick_id}")
                bricks[concrete.brick_id] = concrete
        if pending_path is not None and Path(pending_path).exists():
            pend = load_yaml(pending_path) or {}
            for raw in pend.get("bricks", []):
                bid = str(raw["brick_id"])
                if bid in bricks:
                    continue
                bricks[bid] = BrickDef(bid, "pending", "formula", {"formula": raw["formula"]},
                                       str(raw["formula"]), str(raw.get("available_at", "bar close")),
                                       tuple(raw.get("data_needed", ["spot_1m"])), "numeric", pending=True)
        max_b = int(doc.get("dsl", {}).get("max_bricks_per_condition", 3))
        return cls(bricks=bricks, library_id=str(doc.get("library_id", "")),
                   families=tuple(doc.get("families", [])), max_bricks=max_b)

    def __contains__(self, name: str) -> bool:
        return name in self.bricks


def expand_brick(raw: dict[str, Any]) -> list[BrickDef]:
    """Expand a parameterised brick (``id_template``) into concrete bricks."""
    params = raw.get("params") or {}
    keys = sorted(params)
    combos = list(itertools.product(*[params[k] for k in keys])) if keys else [()]
    template = raw.get("id_template", raw["brick_id"])
    out = []
    for combo in combos:
        values = dict(zip(keys, combo, strict=True))
        fmt = dict(values)
        fmt["field_suffix"] = "_x" if values.get("field") == "extreme" else ""
        bid = template.format(**fmt)
        out.append(BrickDef(bid, str(raw["family"]), str(raw["impl"]), values, str(raw["formula"]),
                            str(raw["available_at"]), tuple(raw["data_needed"]), str(raw.get("kind", "numeric"))))
    return out


# =============================================================================================
# Feature context and brick implementations
# =============================================================================================
@dataclass
class FeatureContext:
    """Analysis frame + aligned auxiliary data; computes brick series with caching."""

    frame: BarFrame
    lib: BrickLibrary
    cal: dict[str, Any]
    available: set[str]
    aux: dict[str, Any] = field(default_factory=dict)
    extra_bricks: dict[str, np.ndarray] = field(default_factory=dict)
    cache: dict[str, np.ndarray] = field(default_factory=dict)

    def brick(self, name: str) -> np.ndarray:
        if name in self.extra_bricks:
            return np.asarray(self.extra_bricks[name], float)
        if name in self.cache:
            return self.cache[name]
        if name not in self.lib.bricks:
            raise CompileError(f"unknown brick {name!r}")
        bdef = self.lib.bricks[name]
        missing = [d for d in bdef.data_needed if d not in self.available]
        if missing:
            raise CompileError(f"brick {name} needs missing data {missing}")
        impl = BRICK_IMPLS.get(bdef.impl)
        if impl is None:
            raise CompileError(f"brick {name}: implementation {bdef.impl!r} not registered")
        values = np.asarray(impl(self, bdef.params), dtype=float)
        if values.shape != (self.frame.n,):
            raise CompileError(f"brick {name} returned shape {values.shape}, expected ({self.frame.n},)")
        self.cache[name] = values
        return values

    # ---- shared derived series ---------------------------------------------------------
    def per_day_to_bars(self, per_day: np.ndarray) -> np.ndarray:
        return np.asarray(per_day, float)[self.frame.day_idx]

    def prev_day(self, per_day: np.ndarray) -> np.ndarray:
        shifted = np.r_[np.nan, np.asarray(per_day, float)[:-1]]
        return shifted[self.frame.day_idx]

    def so_far(self, x: np.ndarray, how: str) -> np.ndarray:
        """Running intraday max (``how='max'``) or min (``'min'``) up to and including each bar."""
        out = np.full(self.frame.n, np.nan)
        ufunc = np.maximum if how == "max" else np.minimum
        fill = -np.inf if how == "max" else np.inf
        for d in range(self.frame.n_days):
            s, e = self.frame.day_start[d], self.frame.day_end[d]
            acc = ufunc.accumulate(np.where(np.isnan(x[s:e]), fill, x[s:e]))
            out[s:e] = np.where(np.isfinite(acc), acc, np.nan)
        return out

    def window_mask(self, name: str) -> np.ndarray:
        windows = self.cal.get("named_windows", {})
        if name not in windows:
            raise CompileError(f"unknown named window {name!r} (era_calendar.yaml)")
        w = windows[name]
        fr = self.frame
        open_min = fr.session_open_minute()[fr.day_idx]
        close_min = fr.session_close_minute()[fr.day_idx]
        if w["anchor"] == "last_bar":
            return fr.is_last_bar
        if w["anchor"] == "open":
            rel = fr.minute - open_min
            return (rel >= int(w["start_min"])) & (rel < int(w["end_min"]))
        if w["anchor"] == "close":
            rel = close_min - fr.minute
            return (rel <= int(w["start_min"])) & (rel > int(w["end_min"]))
        if w["anchor"] == "clock":
            return (fr.minute >= parse_hhmm(w["start"])) & (fr.minute <= parse_hhmm(w["end"]))
        raise CompileError(f"window {name}: unknown anchor {w['anchor']!r}")

    def checkpoint_mask(self, name: str) -> np.ndarray:
        cps = self.cal.get("checkpoints", {})
        if name not in cps:
            raise CompileError(f"unknown checkpoint {name!r} (era_calendar.yaml)")
        return self.frame.close_minute == parse_hhmm(cps[name])

    def epoch_close_minutes(self) -> np.ndarray:
        day_int = self.frame.dates.astype("datetime64[D]").astype(np.int64)
        return day_int * 1440 + self.frame.close_minute

    def aligned(self, key: str) -> dict[str, np.ndarray]:
        """Aux bar series aligned to analysis bars by (date, bar-open minute bucket)."""
        cache_key = f"__aligned_{key}"
        if cache_key in self.aux:
            return self.aux[cache_key]
        other: BarFrame = self.aux[key]
        k = self.frame.interval_min
        own_key = self.frame.dates.astype("datetime64[D]").astype(np.int64) * 1440 + self.frame.minute
        oth_bucket = other.minute - (other.minute - other.session_open_minute()[other.day_idx]) % max(1, k)
        oth_key = other.dates.astype("datetime64[D]").astype(np.int64) * 1440 + oth_bucket
        df = pd.DataFrame({"key": oth_key, "c": other.c, "v": other.v, "oi": other.oi, "h": other.h, "l": other.l})
        agg = df.groupby("key", sort=True).agg(c=("c", "last"), v=("v", "sum"), oi=("oi", "last"),
                                                h=("h", "max"), l=("l", "min"))
        idx = agg.index.to_numpy()
        pos = np.searchsorted(idx, own_key)
        hit = (pos < idx.size) & (idx[np.minimum(pos, idx.size - 1)] == own_key)
        out = {}
        for col in ("c", "v", "oi", "h", "l"):
            vals = agg[col].to_numpy(dtype=float)
            arr = np.full(self.frame.n, np.nan)
            arr[hit] = vals[pos[hit]]
            out[col] = arr
        if np.all(np.isnan(other.v)):
            out["v"] = np.full(self.frame.n, np.nan)
        self.aux[cache_key] = out
        return out

    def asof_snapshot(self, key: str, value_fn: Callable[[pd.DataFrame], float]) -> np.ndarray:
        """Vendor snapshots (``available_from_ts``) reduced by ``value_fn`` and as-of joined so a
        value is used only at ``max(bar close, available_from_ts)`` (R8)."""
        df: pd.DataFrame = self.aux[key]
        avail = pd.to_datetime(df["available_from_ts"]).dt.tz_convert("Asia/Kolkata")
        avail_key = (avail.dt.date.to_numpy().astype("datetime64[D]").astype(np.int64) * 1440
                     + (avail.dt.hour * 60 + avail.dt.minute).to_numpy())
        work = df.assign(_ak=avail_key)
        snaps = work.groupby("_ak", sort=True)
        keys = np.array(sorted(snaps.groups), dtype=np.int64)
        values = np.array([value_fn(snaps.get_group(k)) for k in keys], dtype=float)
        bar_keys = self.epoch_close_minutes()
        pos = np.searchsorted(keys, bar_keys, side="right") - 1
        out = np.full(self.frame.n, np.nan)
        ok = pos >= 0
        same_day = np.zeros(self.frame.n, dtype=bool)
        same_day[ok] = (keys[pos[ok]] // 1440) == (bar_keys[ok] // 1440)
        out[same_day] = values[pos[same_day]]
        return out


def _atr(ctx: FeatureContext) -> np.ndarray:
    return ctx.frame.atr20


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise CompileError("lag k must be >= 1 (negative lags are look-ahead)")
    return np.r_[np.full(k, np.nan), x[:-k]] if k < x.size else np.full(x.size, np.nan)


def _rank_pct(values: np.ndarray, history: np.ndarray) -> float:
    hist = history[np.isfinite(history)]
    if hist.size == 0 or not np.isfinite(values):
        return float("nan")
    return float(np.mean(hist <= values))


def b_ret_k(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    c = ctx.frame.c
    return (c - _lag(c, int(p["k"]))) / _atr(ctx)


def b_day_ret_so_far(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return (ctx.frame.c - ctx.per_day_to_bars(ctx.frame.day_ohlc()["open"])) / _atr(ctx)


def b_gap_atr(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    d = ctx.frame.day_ohlc()
    gap = d["open"] - np.r_[np.nan, d["close"][:-1]]
    atr_first = ctx.frame.atr20[ctx.frame.day_start]
    return ctx.per_day_to_bars(gap / atr_first)


def b_overnight_ret(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    d = ctx.frame.day_ohlc()
    with np.errstate(divide="ignore", invalid="ignore"):
        val = 100.0 * np.log(d["open"] / np.r_[np.nan, d["close"][:-1]])
    return ctx.per_day_to_bars(val)


def _bar_range(ctx: FeatureContext) -> np.ndarray:
    return ctx.frame.h - ctx.frame.l


def b_close_loc(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    rng = _bar_range(ctx)
    with np.errstate(divide="ignore", invalid="ignore"):
        loc = (ctx.frame.c - ctx.frame.l) / rng
    dec = np.clip(np.ceil(loc * 10.0), 1, 10)
    return np.where(rng > 0, dec, np.nan)


def b_body_frac(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    rng = _bar_range(ctx)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rng > 0, np.abs(ctx.frame.c - ctx.frame.o) / rng, np.nan)


def b_wick_ratio(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    rng = _bar_range(ctx)
    upper = fr.h - np.maximum(fr.o, fr.c)
    lower = np.minimum(fr.o, fr.c) - fr.l
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rng > 0, (upper - lower) / rng, np.nan)


def b_range_atr(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return _bar_range(ctx) / _lag(_atr(ctx), 1)


def b_range_rank_slot(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    rng = _bar_range(ctx)
    n_slots = int(fr.slot.max()) + 1
    table = np.full((fr.n_days, n_slots), np.nan)
    table[fr.day_idx, fr.slot] = rng
    out = np.full(fr.n, np.nan)
    for i in range(fr.n):
        d, s = fr.day_idx[i], fr.slot[i]
        if d >= 1:
            out[i] = _rank_pct(rng[i], table[max(0, d - 20):d, s])
    return out


def b_er_n(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    n = int(p["n"])
    c = ctx.frame.c
    move = np.abs(c - _lag(c, n))
    path = rolling_mean_strict(np.abs(np.r_[np.nan, np.diff(c)]), n) * n
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(path > 0, move / path, np.nan)


def b_adx_n(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    n = int(p["n"])
    fr = ctx.frame
    up = fr.h - _lag(fr.h, 1)
    dn = _lag(fr.l, 1) - fr.l
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_dm[np.isnan(up) | np.isnan(dn)] = np.nan
    minus_dm[np.isnan(up) | np.isnan(dn)] = np.nan
    tr = wilder_smooth(fr.tr, n)
    pdi = 100 * wilder_smooth(plus_dm, n) / tr
    mdi = 100 * wilder_smooth(minus_dm, n) / tr
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return wilder_smooth(dx, n)


def b_rsi_n(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    n = int(p["n"])
    diff = np.r_[np.nan, np.diff(ctx.frame.c)]
    gain = wilder_smooth(np.where(np.isnan(diff), np.nan, np.clip(diff, 0, None)), n)
    loss = wilder_smooth(np.where(np.isnan(diff), np.nan, np.clip(-diff, 0, None)), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
    rsi = np.where((loss == 0) & (gain > 0), 100.0, rsi)
    return rsi


def b_ema_slope(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    e = ema(ctx.frame.c, 21)
    return (e - _lag(e, 5)) / _atr(ctx)


def b_dist_ema_atr(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return (ctx.frame.c - ema(ctx.frame.c, 21)) / _atr(ctx)


def b_zscore_close_n(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    n = int(p["n"])
    c = ctx.frame.c
    sd = rolling_std_strict(c, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (c - rolling_mean_strict(c, n)) / sd, np.nan)


def _rolling_prev_extreme(c: np.ndarray, n: int, fn: Callable[..., np.ndarray]) -> np.ndarray:
    out = np.full(c.size, np.nan)
    if c.size <= n:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(c, n)
    vals = fn(windows, axis=1)
    out[n:] = vals[:-1]
    return out


def b_nbar_extreme(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    n = int(p["N"])
    c = ctx.frame.c
    prev_max = _rolling_prev_extreme(c, n, np.max)
    prev_min = _rolling_prev_extreme(c, n, np.min)
    out = np.where(c > prev_max, 1.0, np.where(c < prev_min, -1.0, 0.0))
    out[np.isnan(prev_max) | np.isnan(c)] = np.nan
    return out


def b_bars_since_extreme(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    ext = b_nbar_extreme(ctx, {"N": 20})
    out = np.full(ext.size, np.nan)
    last = -1
    for i, val in enumerate(ext):
        if not np.isnan(val) and val != 0:
            last = i
        if last >= 0:
            out[i] = i - last
    return out


def b_streak_len(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    color = np.sign(fr.c - fr.o)
    out = np.full(fr.n, np.nan)
    run = 0.0
    for i in range(fr.n):
        col = color[i]
        if np.isnan(col) or col == 0:
            run = 0.0
        elif i > 0 and run != 0 and np.sign(run) == col:
            run += col
        else:
            run = col
        out[i] = run if not np.isnan(col) else np.nan
    return out


def _macd(ctx: FeatureContext) -> tuple[np.ndarray, np.ndarray]:
    c = ctx.frame.c
    line = ema(c, 12) - ema(c, 26)
    signal = ema(line, 9)
    return line, signal


def b_macd_hist(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    line, signal = _macd(ctx)
    return (line - signal) / _atr(ctx)


def b_macd_line(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return _macd(ctx)[0] / _atr(ctx)


def b_ema_cross_diff(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return (ema(ctx.frame.c, 9) - ema(ctx.frame.c, 21)) / _atr(ctx)


def b_bb_pctb(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    c = ctx.frame.c
    mid = rolling_mean_strict(c, 20)
    sd = rolling_std_strict(c, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (c - (mid - 2 * sd)) / (4 * sd), np.nan)


def b_supertrend_dir(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    atr10 = rolling_mean_strict(fr.tr, 10)
    mid = (fr.h + fr.l) / 2.0
    upper = mid + 3.0 * atr10
    lower = mid - 3.0 * atr10
    out = np.full(fr.n, np.nan)
    direction = 1.0
    f_up, f_lo = np.nan, np.nan
    for i in range(fr.n):
        if np.isnan(upper[i]) or np.isnan(fr.c[i]):
            continue
        prev_c = fr.c[i - 1] if i > 0 else np.nan
        f_up = upper[i] if (np.isnan(f_up) or upper[i] < f_up or (not np.isnan(prev_c) and prev_c > f_up)) else f_up
        f_lo = lower[i] if (np.isnan(f_lo) or lower[i] > f_lo or (not np.isnan(prev_c) and prev_c < f_lo)) else f_lo
        if fr.c[i] > f_up:
            direction = 1.0
        elif fr.c[i] < f_lo:
            direction = -1.0
        out[i] = direction
    return out


def b_dist_prev_day_level(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    d = ctx.frame.day_ohlc()
    level = ctx.prev_day(d["high"] if p["level"] == "high" else d["low"])
    if p.get("field") == "extreme":
        price = ctx.frame.h if p["level"] == "high" else ctx.frame.l
    else:
        price = ctx.frame.c
    return (price - level) / _atr(ctx)


def b_dist_or_level(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    width = int(ctx.cal.get("opening_ranges", {}).get("default", 30))
    open_min = fr.session_open_minute()[fr.day_idx]
    in_or = fr.close_minute <= open_min + width
    hi = np.where(in_or, fr.h, np.nan)
    lo = np.where(in_or, fr.l, np.nan)
    or_hi = np.full(fr.n_days, np.nan)
    or_lo = np.full(fr.n_days, np.nan)
    for d in range(fr.n_days):
        s, e = fr.day_start[d], fr.day_end[d]
        if np.isfinite(hi[s:e]).any():
            or_hi[d] = np.nanmax(hi[s:e])
            or_lo[d] = np.nanmin(lo[s:e])
    level = or_hi if p["level"] == "high" else or_lo
    val = (fr.c - level[fr.day_idx]) / _atr(ctx)
    val[fr.close_minute < open_min + width] = np.nan
    return val


def _fut_volume(ctx: FeatureContext) -> np.ndarray:
    vol = ctx.aligned("fut")["v"]
    if np.isfinite(vol).mean() < 0.5 or np.nanmax(np.nan_to_num(vol, nan=0.0)) <= 0:
        raise CompileError("FUT-VOLUME unusable (< 50% non-zero bars)")
    return vol


def b_dist_vwap_atr(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    vol = np.nan_to_num(_fut_volume(ctx), nan=0.0)
    tp = (fr.h + fr.l + fr.c) / 3.0
    out = np.full(fr.n, np.nan)
    for d in range(fr.n_days):
        s, e = fr.day_start[d], fr.day_end[d]
        pv = np.cumsum(np.nan_to_num(tp[s:e] * vol[s:e]))
        vv = np.cumsum(vol[s:e])
        with np.errstate(divide="ignore", invalid="ignore"):
            out[s:e] = np.where(vv > 0, pv / vv, np.nan)
    return (fr.c - out) / _atr(ctx)


def b_dist_round(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    step = float(p["step"])
    c = ctx.frame.c
    return (c - np.round(c / step) * step) / _atr(ctx)


def b_dist_prev_poc(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    vol = np.nan_to_num(_fut_volume(ctx), nan=0.0)
    poc = np.full(fr.n_days, np.nan)
    for d in range(fr.n_days):
        s, e = fr.day_start[d], fr.day_end[d]
        prices = np.round(fr.c[s:e] / 10.0) * 10.0
        ok = np.isfinite(prices)
        if not ok.any() or vol[s:e][ok].sum() <= 0:
            continue
        frame = pd.DataFrame({"p": prices[ok], "v": vol[s:e][ok]}).groupby("p", sort=True)["v"].sum()
        poc[d] = float(frame.idxmax())
    return (fr.c - ctx.prev_day(poc)) / _atr(ctx)


def b_day_range_pos_so_far(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    hi = ctx.so_far(ctx.frame.h, "max")
    lo = ctx.so_far(ctx.frame.l, "min")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hi > lo, (ctx.frame.c - lo) / (hi - lo), np.nan)


def b_rv_ratio(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    lr = np.log(ctx.frame.c / _lag(ctx.frame.c, 1))
    short = rolling_std_strict(lr, 10)
    long = rolling_std_strict(lr, 50)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(long > 0, short / long, np.nan)


def b_atr_pct_rank(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    eod = fr.atr20[fr.day_end - 1]
    out = np.full(fr.n, np.nan)
    for i in range(fr.n):
        d = fr.day_idx[i]
        if d >= 1:
            out[i] = _rank_pct(fr.atr20[i], eod[max(0, d - 60):d])
    return out


def b_vol_of_vol(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    atr = _atr(ctx)
    return rolling_std_strict(np.r_[np.nan, np.diff(atr)], 20) / atr


def b_nr_k(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    k = int(p["k"])
    d = ctx.frame.day_ohlc()
    rng = d["high"] - d["low"]
    flag = np.full(rng.size, np.nan)
    for i in range(k - 1, rng.size):
        window = rng[i - k + 1:i + 1]
        if np.isfinite(window).all():
            flag[i] = 1.0 if rng[i] <= window.min() else 0.0
    return ctx.prev_day(flag)


def b_bb_bandwidth_pct(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    c = ctx.frame.c
    with np.errstate(divide="ignore", invalid="ignore"):
        bw = 4 * rolling_std_strict(c, 20) / rolling_mean_strict(c, 20)
    out = np.full(c.size, np.nan)
    for i in range(100, c.size):
        out[i] = _rank_pct(bw[i], bw[i - 100:i])
    return out


def b_slot(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return ctx.frame.slot.astype(float)


def b_slot_group(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    out = np.zeros(ctx.frame.n)
    for code, name in ((1, "first_15m"), (2, "lunch_lull"), (3, "last_hour")):
        out = np.where((out == 0) & ctx.window_mask(name), code, out)
    return out


def b_weekday(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    days = ctx.frame.dates.astype("datetime64[D]").astype(np.int64)
    return ((days + 3) % 7).astype(float)


def b_min_since_open(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    return (fr.close_minute - fr.session_open_minute()[fr.day_idx]).astype(float)


def b_min_to_close(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    return (fr.session_close_minute()[fr.day_idx] - fr.close_minute).astype(float)


def _day_of_month(ctx: FeatureContext) -> np.ndarray:
    d = ctx.frame.dates.astype("datetime64[D]")
    return (d - d.astype("datetime64[M]")).astype(np.int64) + 1


def b_month_end(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return (_day_of_month(ctx) >= 28).astype(float)


def b_month_start(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return (_day_of_month(ctx) <= 3).astype(float)


def b_dte(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    expiries = sorted(ctx.aux.get("expiries", []))
    days = [dt.date.fromisoformat(str(d)) for d in ctx.frame.days]
    pos = {d: i for i, d in enumerate(days)}
    per_day = np.full(len(days), np.nan)
    for i, day in enumerate(days):
        nxt = next((e for e in expiries if e >= day), None)
        if nxt is not None and nxt in pos:
            per_day[i] = pos[nxt] - i
    return ctx.per_day_to_bars(per_day)


def b_rvol_slot(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    fr = ctx.frame
    vol = _fut_volume(ctx)
    n_slots = int(fr.slot.max()) + 1
    table = np.full((fr.n_days, n_slots), np.nan)
    table[fr.day_idx, fr.slot] = vol
    out = np.full(fr.n, np.nan)
    for i in range(fr.n):
        d, s = fr.day_idx[i], fr.slot[i]
        hist = table[max(0, d - 20):d, s]
        hist = hist[np.isfinite(hist)]
        if hist.size >= 10 and (hist > 0).mean() >= 0.5:
            med = np.median(hist)
            out[i] = vol[i] / med if med > 0 else np.nan
    return out


def b_vol_z(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    vol = _fut_volume(ctx)
    sd = rolling_std_strict(_lag(vol, 1), 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (vol - rolling_mean_strict(_lag(vol, 1), 20)) / sd, np.nan)


def _fut_oi(ctx: FeatureContext) -> np.ndarray:
    oi = ctx.aligned("fut")["oi"]
    if np.isfinite(oi).mean() < 0.5:
        raise CompileError("futures OI unusable")
    return oi


def b_oi_chg_pct(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    oi = _fut_oi(ctx)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (oi / lag_within_day(ctx.frame, oi, 5) - 1.0)


def b_price_oi_class(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    oi = _fut_oi(ctx)
    d_oi = oi - lag_within_day(ctx.frame, oi, 5)
    d_px = ctx.frame.c - lag_within_day(ctx.frame, ctx.frame.c, 5)
    cls = np.where((d_px > 0) & (d_oi > 0), 1, np.where((d_px < 0) & (d_oi > 0), 2,
                   np.where((d_px > 0) & (d_oi < 0), 3, np.where((d_px < 0) & (d_oi < 0), 4, 0))))
    out = cls.astype(float)
    out[np.isnan(d_oi) | np.isnan(d_px)] = np.nan
    return out


def b_fut_basis(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return (ctx.aligned("fut")["c"] - ctx.frame.c) / _atr(ctx)


def b_basis_chg(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    basis = ctx.aligned("fut")["c"] - ctx.frame.c
    return (basis - lag_within_day(ctx.frame, basis, 5)) / _atr(ctx)


def _atm_row(group: pd.DataFrame, spot: float) -> pd.DataFrame:
    strikes = group["strike"].to_numpy(dtype=float)
    atm = strikes[np.argmin(np.abs(strikes - spot))]
    return group[group["strike"] == atm]


def _spot_asof(ctx: FeatureContext) -> Callable[[pd.DataFrame], float]:
    fr = ctx.frame
    keys = ctx.epoch_close_minutes()

    def spot_for(group: pd.DataFrame) -> float:
        ts = pd.to_datetime(group["available_from_ts"]).dt.tz_convert("Asia/Kolkata").iloc[0]
        k = (np.datetime64(ts.date(), "D").astype(np.int64) * 1440 + ts.hour * 60 + ts.minute)
        pos = np.searchsorted(keys, k, side="right") - 1
        return float(fr.c[pos]) if pos >= 0 else float("nan")

    return spot_for


def b_atm_iv(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    spot_for = _spot_asof(ctx)

    def value(group: pd.DataFrame) -> float:
        ok = group[group["iv_status"].astype(str) == "OK"]
        if ok.empty:
            return float("nan")
        return float(_atm_row(ok, spot_for(group))["iv"].astype(float).mean())

    return ctx.asof_snapshot("iv", value)


def b_iv_chg_h(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    iv = ctx.brick("atm_iv")
    return iv - lag_within_day(ctx.frame, iv, int(p["h"]))


def b_skew(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    spot_for = _spot_asof(ctx)

    def value(group: pd.DataFrame) -> float:
        ok = group[group["iv_status"].astype(str) == "OK"]
        if ok.empty:
            return float("nan")
        spot = spot_for(group)
        strikes = np.sort(ok["strike"].astype(float).unique())
        if strikes.size < 3:
            return float("nan")
        step = float(np.median(np.diff(strikes)))
        put = ok[(ok["option_type"] == "PE") & np.isclose(ok["strike"].astype(float), spot - 2 * step, atol=step / 2)]
        call = ok[(ok["option_type"] == "CE") & np.isclose(ok["strike"].astype(float), spot + 2 * step, atol=step / 2)]
        if put.empty or call.empty:
            return float("nan")
        return float(put["iv"].astype(float).mean() - call["iv"].astype(float).mean())

    return ctx.asof_snapshot("iv", value)


def b_pcr(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    def value(group: pd.DataFrame) -> float:
        put = group.loc[group["option_type"] == "PE", "oi"].astype(float).sum()
        call = group.loc[group["option_type"] == "CE", "oi"].astype(float).sum()
        return float(put / call) if call > 0 else float("nan")

    return ctx.asof_snapshot("chain", value)


def b_oi_wall_dist(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    side = "CE" if p["side"] == "call" else "PE"

    def value(group: pd.DataFrame) -> float:
        sub = group[group["option_type"] == side]
        if sub.empty:
            return float("nan")
        return float(sub.loc[sub["oi"].astype(float).idxmax(), "strike"])

    wall = ctx.asof_snapshot("chain", value)
    return (ctx.frame.c - wall) / _atr(ctx)


def b_max_pain_dist(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    def value(group: pd.DataFrame) -> float:
        strikes = np.sort(group["strike"].astype(float).unique())
        if strikes.size == 0:
            return float("nan")
        ce = group[group["option_type"] == "CE"]
        pe = group[group["option_type"] == "PE"]
        pain = []
        for s in strikes:
            call_pay = (np.clip(s - ce["strike"].astype(float), 0, None) * ce["oi"].astype(float)).sum()
            put_pay = (np.clip(pe["strike"].astype(float) - s, 0, None) * pe["oi"].astype(float)).sum()
            pain.append(call_pay + put_pay)
        return float(strikes[int(np.argmin(pain))])

    mp = ctx.asof_snapshot("chain", value)
    return (ctx.frame.c - mp) / _atr(ctx)


def _atm_straddle(ctx: FeatureContext) -> np.ndarray:
    if "__straddle" in ctx.aux:
        return ctx.aux["__straddle"]
    opt: pd.DataFrame = ctx.aux["opt"]
    live = opt[~(opt["is_stale"].fillna(False) | opt["is_filled"].fillna(False))].copy()
    ts = pd.to_datetime(live["ts_open"]).dt.tz_convert("Asia/Kolkata")
    k = ctx.frame.interval_min
    minute = ts.dt.hour * 60 + ts.dt.minute
    live["_key"] = (ts.dt.date.to_numpy().astype("datetime64[D]").astype(np.int64) * 1440
                    + (minute - (minute % k)).to_numpy())
    own_key = ctx.frame.dates.astype("datetime64[D]").astype(np.int64) * 1440 + ctx.frame.minute
    spot = pd.Series(ctx.frame.c, index=own_key)
    out = np.full(ctx.frame.n, np.nan)
    grouped = live.groupby("_key", sort=True)
    for key, grp in grouped:
        if key not in spot.index:
            continue
        s = float(spot.loc[key]) if np.ndim(spot.loc[key]) == 0 else float(spot.loc[key].iloc[0])
        last = grp.sort_values("ts_open").groupby(["strike", "option_type"], sort=True)["c"].last().reset_index()
        atm = _atm_row(last, s)
        if set(atm["option_type"]) >= {"CE", "PE"}:
            out[np.flatnonzero(own_key == key)] = float(atm["c"].astype(float).sum())
    ctx.aux["__straddle"] = out
    return out


def b_atm_straddle_pct(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return 100.0 * _atm_straddle(ctx) / ctx.frame.c


def b_premium_ret_h(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    prem = _atm_straddle(ctx)
    with np.errstate(divide="ignore", invalid="ignore"):
        return prem / lag_within_day(ctx.frame, prem, int(p["h"])) - 1.0


def b_spread_pct(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    def value(group: pd.DataFrame) -> float:
        bid = group["bid"].astype(float)
        ask = group["ask"].astype(float)
        mid = (bid + ask) / 2.0
        ok = (bid > 0) & (ask > bid)
        return float(((ask - bid) / mid)[ok].median()) if ok.any() else float("nan")

    return ctx.asof_snapshot("quotes", value)


def b_cross_ret_k(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    al = ctx.aligned("nifty")
    atr_other = rolling_mean_strict(al["h"] - al["l"], 20)
    return (al["c"] - _lag(al["c"], int(p["k"]))) / atr_other


def b_banks_breadth(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    stocks: dict[str, BarFrame] = ctx.aux["stocks"]
    ups = []
    for sym in sorted(stocks):
        key = f"stock::{sym}"
        ctx.aux[key] = stocks[sym]
        al = ctx.aligned(key)
        day_open = np.full(ctx.frame.n_days, np.nan)
        for d in range(ctx.frame.n_days):
            seg = al["c"][ctx.frame.day_start[d]:ctx.frame.day_end[d]]
            ok = np.flatnonzero(np.isfinite(seg))
            if ok.size:
                day_open[d] = seg[ok[0]]
        ups.append(np.where(np.isnan(al["c"]), np.nan, (al["c"] > day_open[ctx.frame.day_idx]).astype(float)))
    if not ups:
        raise CompileError("no bank constituents available")
    return np.nanmean(np.vstack(ups), axis=0)


def b_vix_chg(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    al = ctx.aligned("vix")
    return al["c"] - lag_within_day(ctx.frame, al["c"], 5)


def b_global_ret(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    key = str(p["symbol"]).lower()
    al = ctx.aligned(key)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * np.log(al["c"] / _lag(al["c"], 5))


def b_giftnifty_gap(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    gift: BarFrame = ctx.aux["giftnifty"]
    nifty: BarFrame = ctx.aux["nifty"]
    open_min = ctx.frame.session_open_minute()
    per_day = np.full(ctx.frame.n_days, np.nan)
    nifty_close = {str(d): c for d, c in zip(nifty.days, nifty.day_ohlc()["close"], strict=True)}
    gift_keys = gift.dates.astype("datetime64[D]").astype(np.int64) * 1440 + gift.close_minute
    for i, day in enumerate(ctx.frame.days):
        if i == 0:
            continue
        prev = nifty_close.get(str(ctx.frame.days[i - 1]))
        cutoff = np.datetime64(day, "D").astype(np.int64) * 1440 + open_min[i]
        pos = np.searchsorted(gift_keys, cutoff, side="right") - 1
        if prev and pos >= 0 and np.isfinite(gift.c[pos]):
            per_day[i] = 100.0 * math.log(gift.c[pos] / prev)
    return ctx.per_day_to_bars(per_day)


def b_canary_lead_close(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    """Deliberately leaky canary ``close_{t+1}``: the Leak Gate MUST reject it."""
    c = ctx.frame.c
    return np.r_[c[1:], np.nan]


def b_formula(ctx: FeatureContext, p: dict[str, Any]) -> np.ndarray:
    return evaluate_formula(str(p["formula"]), ctx)


BRICK_IMPLS: dict[str, Callable[[FeatureContext, dict[str, Any]], np.ndarray]] = {
    "ret_k": b_ret_k, "day_ret_so_far": b_day_ret_so_far, "gap_atr": b_gap_atr,
    "overnight_ret": b_overnight_ret, "close_loc": b_close_loc, "body_frac": b_body_frac,
    "wick_ratio": b_wick_ratio, "range_atr": b_range_atr, "range_rank_slot": b_range_rank_slot,
    "er_n": b_er_n, "adx_n": b_adx_n, "rsi_n": b_rsi_n, "ema_slope": b_ema_slope,
    "dist_ema_atr": b_dist_ema_atr, "zscore_close_n": b_zscore_close_n, "nbar_extreme": b_nbar_extreme,
    "bars_since_extreme": b_bars_since_extreme, "streak_len": b_streak_len, "macd_hist": b_macd_hist,
    "macd_line": b_macd_line, "ema_cross_diff": b_ema_cross_diff, "bb_pctb": b_bb_pctb,
    "supertrend_dir": b_supertrend_dir, "dist_prev_day_level": b_dist_prev_day_level,
    "dist_or_level": b_dist_or_level, "dist_vwap_atr": b_dist_vwap_atr, "dist_round": b_dist_round,
    "dist_prev_poc": b_dist_prev_poc, "day_range_pos_so_far": b_day_range_pos_so_far,
    "rv_ratio": b_rv_ratio, "atr_pct_rank": b_atr_pct_rank, "vol_of_vol": b_vol_of_vol,
    "nr_k": b_nr_k, "bb_bandwidth_pct": b_bb_bandwidth_pct, "slot": b_slot, "slot_group": b_slot_group,
    "weekday": b_weekday, "min_since_open": b_min_since_open, "min_to_close": b_min_to_close,
    "month_end": b_month_end, "month_start": b_month_start, "dte": b_dte, "rvol_slot": b_rvol_slot,
    "vol_z": b_vol_z, "oi_chg_pct": b_oi_chg_pct, "price_oi_class": b_price_oi_class,
    "fut_basis": b_fut_basis, "basis_chg": b_basis_chg, "atm_iv": b_atm_iv, "iv_chg_h": b_iv_chg_h,
    "skew": b_skew, "pcr": b_pcr, "oi_wall_dist": b_oi_wall_dist, "max_pain_dist": b_max_pain_dist,
    "atm_straddle_pct": b_atm_straddle_pct, "premium_ret_h": b_premium_ret_h, "spread_pct": b_spread_pct,
    "cross_ret_k": b_cross_ret_k, "banks_breadth": b_banks_breadth, "vix_chg": b_vix_chg,
    "global_ret": b_global_ret, "giftnifty_gap": b_giftnifty_gap, "formula": b_formula,
    "canary_lead_close": b_canary_lead_close,
}


# =============================================================================================
# Safe formula language for BRICK proposals (no eval; FR-21)
# =============================================================================================
_FORMULA_TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*/(),]))")
PRIMITIVES = ("close", "open", "high", "low", "ATR20", "prev_day_high", "prev_day_low",
              "prev_day_close", "session_open", "prev_week_high", "prev_week_low")


def _formula_tokens(text: str) -> list[tuple[str, str]]:
    pos = 0
    out: list[tuple[str, str]] = []
    text = text.strip()
    while pos < len(text):
        m = _FORMULA_TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise CompileError(f"formula: unexpected character at {pos}: {text[pos:pos + 10]!r}")
        kind = m.lastgroup or ""
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


def _primitive(name: str, ctx: FeatureContext) -> np.ndarray:
    fr = ctx.frame
    d = fr.day_ohlc()
    if name in ("close", "open", "high", "low"):
        return {"close": fr.c, "open": fr.o, "high": fr.h, "low": fr.l}[name]
    if name == "ATR20":
        return fr.atr20
    if name == "prev_day_high":
        return ctx.prev_day(d["high"])
    if name == "prev_day_low":
        return ctx.prev_day(d["low"])
    if name == "prev_day_close":
        return ctx.prev_day(d["close"])
    if name == "session_open":
        return ctx.per_day_to_bars(d["open"])
    if name in ("prev_week_high", "prev_week_low"):
        weeks = fr.week_ids()
        uniq = np.unique(weeks)
        agg = {w: (np.nanmax(d["high"][weeks == w]) if name.endswith("high") else np.nanmin(d["low"][weeks == w])) for w in uniq}
        per_day = np.array([agg.get(w - 1, np.nan) for w in weeks])
        return ctx.per_day_to_bars(per_day)
    if name in ctx.lib.bricks or name in ctx.extra_bricks:
        return ctx.brick(name)
    raise CompileError(f"formula: unknown primitive or brick {name!r}")


def evaluate_formula(text: str, ctx: FeatureContext) -> np.ndarray:
    """Evaluate a proposal formula with + - * / unary-, lag(x,k), abs, log, sqrt (no ``eval``)."""
    tokens = _formula_tokens(text)
    pos = 0

    def peek() -> tuple[str, str] | None:
        return tokens[pos] if pos < len(tokens) else None

    def take(expected: str | None = None) -> tuple[str, str]:
        nonlocal pos
        tok = peek()
        if tok is None or (expected is not None and tok[1] != expected):
            raise CompileError(f"formula: expected {expected!r} in {text!r}")
        pos += 1
        return tok

    def atom() -> Any:
        tok = take()
        if tok[0] == "num":
            return float(tok[1])
        if tok[1] == "(":
            val = expr()
            take(")")
            return val
        if tok[1] == "-":
            return -atom()
        if tok[0] == "name":
            nxt = peek()
            if nxt is not None and nxt[1] == "(":
                take("(")
                args = [expr()]
                while peek() is not None and peek()[1] == ",":
                    take(",")
                    args.append(expr())
                take(")")
                fn = tok[1]
                if fn == "lag" and len(args) == 2:
                    k = int(args[1])
                    return _lag(np.asarray(args[0], float), k)
                if fn in ("abs", "log", "sqrt") and len(args) == 1:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return {"abs": np.abs, "log": np.log, "sqrt": np.sqrt}[fn](np.asarray(args[0], float))
                raise CompileError(f"formula: unsupported function {fn}/{len(args)}")
            return _primitive(tok[1], ctx)
        raise CompileError(f"formula: unexpected token {tok[1]!r}")

    def term() -> Any:
        val = atom()
        while peek() is not None and peek()[1] in "*/":
            op = take()[1]
            rhs = atom()
            with np.errstate(divide="ignore", invalid="ignore"):
                val = val * rhs if op == "*" else val / rhs
        return val

    def expr() -> Any:
        val = term()
        while peek() is not None and peek()[1] in "+-":
            op = take()[1]
            rhs = term()
            val = val + rhs if op == "+" else val - rhs
        return val

    result = expr()
    if pos != len(tokens):
        raise CompileError(f"formula: trailing tokens in {text!r}")
    out = np.asarray(result, float)
    if out.ndim == 0:
        out = np.full(ctx.frame.n, float(out))
    return out


# =============================================================================================
# DSL parser
# =============================================================================================
@dataclass
class Node:
    kind: str                      # "call" | "ref" | "num" | "name"
    value: Any
    args: list[Node] = field(default_factory=list)
    path: str = ""


_DSL_TOKEN = re.compile(r"\s*(?:(?P<num>-?\d+(?:\.\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[(),]))")


def parse_event(text: str) -> Node:
    """Recursive-descent parse of the Idea Card DSL into an AST (never ``eval``)."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        m = _DSL_TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise CompileError(f"DSL: unexpected character at {pos}: {text[pos:pos + 12]!r}")
        kind = m.lastgroup or ""
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    idx = 0

    def parse(path: str) -> Node:
        nonlocal idx
        if idx >= len(tokens):
            raise CompileError("DSL: unexpected end of expression")
        kind, val = tokens[idx]
        idx += 1
        if kind == "num":
            return Node("num", float(val), path=path)
        if kind != "name":
            raise CompileError(f"DSL: unexpected token {val!r}")
        if idx < len(tokens) and tokens[idx][1] == "(":
            idx += 1
            args: list[Node] = []
            if idx < len(tokens) and tokens[idx][1] == ")":
                idx += 1
            else:
                while True:
                    args.append(parse(f"{path}.{val}[{len(args)}]"))
                    if idx >= len(tokens):
                        raise CompileError("DSL: missing ')'")
                    sep = tokens[idx][1]
                    idx += 1
                    if sep == ")":
                        break
                    if sep != ",":
                        raise CompileError(f"DSL: expected ',' or ')' got {sep!r}")
            if val not in FUNC_ARITY:
                raise CompileError(f"DSL: unknown operator {val!r}")
            lo, hi = FUNC_ARITY[val]
            if not lo <= len(args) <= hi:
                raise CompileError(f"DSL: {val} takes {lo}..{hi} arguments, got {len(args)}")
            return Node("call", val, args, path=path)
        return Node("ref", val, path=path)

    node = parse("$")
    if idx != len(tokens):
        raise CompileError(f"DSL: trailing tokens after position {idx}")
    return node


def _bricks_in(node: Node) -> set[str]:
    if node.kind == "ref":
        return {node.value}
    out: set[str] = set()
    if node.kind == "call" and node.value in ("in_window", "at_checkpoint"):
        return out
    for arg in node.args:
        out |= _bricks_in(arg)
    return out


def _type_of(node: Node) -> str:
    """Static type check: 'num' or 'bool'; raises CompileError on mismatch."""
    if node.kind == "num":
        return "num"
    if node.kind == "ref":
        return "num"
    fn = node.value
    if fn in ("AND", "OR"):
        for a in node.args:
            if _type_of(a) != "bool":
                raise CompileError(f"{fn} needs boolean arguments")
        return "bool"
    if fn == "NOT":
        if _type_of(node.args[0]) != "bool":
            raise CompileError("NOT needs a boolean argument")
        return "bool"
    if fn in ("gt", "gte", "lt", "lte", "eq", "cross_above", "cross_below"):
        for a in node.args:
            if _type_of(a) != "num":
                raise CompileError(f"{fn} needs numeric arguments")
        return "bool"
    if fn in QUANTILE_LEVELS:
        if _type_of(node.args[0]) != "num":
            raise CompileError(f"{fn} needs a numeric argument")
        return "bool"
    if fn in ("tercile", "sign", "abs"):
        if _type_of(node.args[0]) != "num":
            raise CompileError(f"{fn} needs a numeric argument")
        return "num"
    if fn == "lag":
        if node.args[0].kind != "ref" or node.args[1].kind != "num":
            raise CompileError("lag(brick, k) needs a brick and an integer k")
        if node.args[1].value < 1 or node.args[1].value != int(node.args[1].value):
            raise CompileError("lag k must be a positive integer (negative lag is look-ahead)")
        return "num"
    if fn == "A_then_B_within":
        if _type_of(node.args[0]) != "bool" or _type_of(node.args[1]) != "bool" or node.args[2].kind != "num":
            raise CompileError("A_then_B_within(boolA, boolB, k)")
        return "bool"
    if fn == "first_in_day":
        if _type_of(node.args[0]) != "bool":
            raise CompileError("first_in_day needs a boolean argument")
        return "bool"
    if fn in ("in_window", "at_checkpoint"):
        if node.args[0].kind != "ref":
            raise CompileError(f"{fn}(name) needs a calendar name")
        return "bool"
    raise CompileError(f"{fn} is only allowed at the top level")


# =============================================================================================
# Static gates
# =============================================================================================
def leak_scan(text: str) -> list[str]:
    """Static look-ahead tokens in event / formula text (R8, F3.2)."""
    return [pat for pat in LEAK_TOKENS if re.search(pat, text, flags=re.IGNORECASE)]


def boundary_scan(text: str) -> list[str]:
    """Diagnosis-Only Boundary violations (NG-1, NG-2, tripwire #13); quoted claims excluded."""
    unquoted = re.sub(r"\"[^\"]*\"|'[^']*'", " ", text)
    cleaned = re.sub(r"\btarget\s+T\d+\b", " ", unquoted, flags=re.IGNORECASE)
    return [pat for pat in BOUNDARY_PATTERNS if re.search(pat, cleaned, flags=re.IGNORECASE)]


def availability_check(data_tokens: Iterable[str], available_at: str, cal: dict[str, Any]) -> list[str]:
    """Vendor availability gate (R8, FM-40). Returns leak reasons (empty when OK).

    * ``daily_metrics_DERIVED:EOD_INCLUSIVE`` is never usable intraday;
    * ``iv_daily_DERIVED`` is usable only from its first check time;
    * ``xcheck_csv`` (day-D values) is usable only from the next session.
    """
    avail_cfg = cal.get("availability", {})
    reasons = []
    when = available_at.lower()
    intraday = "next session" not in when and "next_session" not in when
    for token in data_tokens:
        if token.startswith("daily_metrics_DERIVED"):
            version = token.split(":", 1)[1] if ":" in token else "EOD_INCLUSIVE"
            rule = avail_cfg.get("daily_metrics_versions", {}).get(version, {"intraday": False})
            if intraday and not rule.get("intraday", False):
                reasons.append(f"{token} is {version}: usable only from {rule.get('usable_from', 'session end')}, never intraday")
        elif token.startswith("iv_daily_DERIVED"):
            checks = sorted(parse_hhmm(t) for t in avail_cfg.get("iv_daily_check_times", []))
            m = re.search(r"(\d{2}:\d{2})", available_at)
            used_from = parse_hhmm(m.group(1)) if m else None
            if intraday and (used_from is None or not checks or used_from < checks[0]):
                reasons.append(f"{token} used before its check_time {checks[0] // 60:02d}:{checks[0] % 60:02d}" if checks
                               else f"{token} has no declared check_time")
        elif token.startswith("xcheck") and intraday:
            reasons.append(f"{token}: day-D xcheck values are usable only from the next session")
    return reasons


def dynamic_leak_gate(brick: str, ctx: FeatureContext, *, cuts: Sequence[float] = (0.3, 0.6, 0.85),
                      seed: int = 42) -> tuple[bool, str]:
    """+1-bar shift test: perturbing bars after ``t`` must leave the brick unchanged at ``<= t``."""
    base = ctx.brick(brick)
    rng = np.random.default_rng(seed)
    fr = ctx.frame
    for frac in cuts:
        cut = int(fr.n * frac)
        after = np.arange(fr.n) > cut
        factor = np.ones(fr.n)
        factor[after] = np.exp(rng.normal(0.0, 0.02, int(after.sum())))
        widen = np.where(after, 1.001, 1.0)
        pert = BarFrame(dates=fr.dates, day_idx=fr.day_idx, slot=fr.slot, minute=fr.minute,
                        o=fr.o * factor, h=fr.h * factor * widen, l=fr.l * factor / widen, c=fr.c * factor,
                        v=fr.v, oi=fr.oi, masked=fr.masked, days=fr.days, day_start=fr.day_start,
                        day_end=fr.day_end, interval_min=fr.interval_min, bars_per_day=fr.bars_per_day,
                        meta=dict(fr.meta))
        pctx = FeatureContext(frame=pert, lib=ctx.lib, cal=ctx.cal, available=ctx.available,
                              aux={k: v for k, v in ctx.aux.items() if not str(k).startswith("__")},
                              extra_bricks=dict(ctx.extra_bricks))
        try:
            pert_vals = pctx.brick(brick)
        except CompileError as exc:
            return False, f"leak gate could not recompute {brick}: {exc}"
        a, b = base[: cut + 1], pert_vals[: cut + 1]
        same = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        if not same.all():
            first = int(np.flatnonzero(~same)[0])
            return False, f"{brick} changed at bar {first} <= t={cut} when only bars > t were perturbed (look-ahead)"
    return True, "pass"


# =============================================================================================
# Compiled card
# =============================================================================================
@dataclass
class CompiledCard:
    card: dict[str, Any]
    status: str
    reason: str = ""
    ast: Node | None = None
    bricks: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    cutoffs: dict[str, Any] = field(default_factory=dict)
    fitted: bool = False
    model: Any = None

    @property
    def card_id(self) -> str:
        return str(self.card.get("card_id", ""))

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def is_directional(self) -> bool:
        return bool(self.ast is not None and self.ast.kind == "call" and self.ast.value == "DIR")

    @property
    def is_model(self) -> bool:
        return bool(self.ast is not None and self.ast.kind == "call" and self.ast.value == "MODEL")

    def card_hash(self) -> str:
        """SHA-256 of the frozen hypothesis (card definition + frozen cut-offs)."""
        body = {k: v for k, v in self.card.items() if k not in ("frozen_at", "card_hash", "data_version")}
        return sha256_json({"card": body, "cutoffs": self.cutoffs})

    # ---- evaluation ---------------------------------------------------------------------
    def _eval(self, node: Node, ctx: FeatureContext, fit_mask: np.ndarray | None) -> np.ndarray:
        fr = ctx.frame
        if node.kind == "num":
            return np.full(fr.n, float(node.value))
        if node.kind == "ref":
            return ctx.brick(node.value)
        fn = node.value
        args = node.args
        if fn in ("AND", "OR"):
            vals = [self._eval(a, ctx, fit_mask).astype(bool) for a in args]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if fn == "AND" else (out | v)
            return out
        if fn == "NOT":
            return ~self._eval(args[0], ctx, fit_mask).astype(bool)
        if fn in ("gt", "gte", "lt", "lte", "eq"):
            a = self._eval(args[0], ctx, fit_mask)
            b = self._eval(args[1], ctx, fit_mask)
            with np.errstate(invalid="ignore"):
                res = {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b,
                       "eq": np.isclose(a, b, rtol=0, atol=1e-9)}[fn]
            return res & np.isfinite(a) & np.isfinite(b)
        if fn in QUANTILE_LEVELS:
            vals = self._eval(args[0], ctx, fit_mask)
            q, side = QUANTILE_LEVELS[fn]
            key = node.path
            if fit_mask is not None:
                use = fit_mask & np.isfinite(vals)
                if not use.any():
                    raise CompileError(f"{self.card_id}: no finite discovery values for {fn}")
                self.cutoffs[key] = float(np.quantile(vals[use], q))
            if key not in self.cutoffs:
                raise CompileError(f"{self.card_id}: cut-off {key} not frozen (fit on Playground first)")
            cut = self.cutoffs[key]
            with np.errstate(invalid="ignore"):
                res = vals <= cut if side == "le" else vals >= cut
            return res & np.isfinite(vals)
        if fn == "tercile":
            vals = self._eval(args[0], ctx, fit_mask)
            key = node.path
            if fit_mask is not None:
                use = fit_mask & np.isfinite(vals)
                if not use.any():
                    raise CompileError(f"{self.card_id}: no finite discovery values for tercile")
                self.cutoffs[key] = [float(x) for x in np.quantile(vals[use], [1 / 3, 2 / 3])]
            if key not in self.cutoffs:
                raise CompileError(f"{self.card_id}: tercile cut-off {key} not frozen")
            lo, hi = self.cutoffs[key]
            out = np.where(vals <= lo, 1.0, np.where(vals <= hi, 2.0, 3.0))
            out[~np.isfinite(vals)] = np.nan
            return out
        if fn == "sign":
            return np.sign(self._eval(args[0], ctx, fit_mask))
        if fn == "abs":
            return np.abs(self._eval(args[0], ctx, fit_mask))
        if fn == "lag":
            return _lag(ctx.brick(args[0].value), int(args[1].value))
        if fn in ("cross_above", "cross_below"):
            a = self._eval(args[0], ctx, fit_mask)
            b = self._eval(args[1], ctx, fit_mask)
            pa_, pb_ = lag_within_day(fr, a, 1), lag_within_day(fr, b, 1)
            with np.errstate(invalid="ignore"):
                res = (a > b) & (pa_ <= pb_) if fn == "cross_above" else (a < b) & (pa_ >= pb_)
            return res & np.isfinite(pa_) & np.isfinite(pb_)
        if fn == "A_then_B_within":
            a = self._eval(args[0], ctx, fit_mask).astype(bool)
            b = self._eval(args[1], ctx, fit_mask).astype(bool)
            k = int(args[2].value)
            prior = np.zeros(fr.n, dtype=bool)
            for lag_k in range(1, k + 1):
                prior |= np.nan_to_num(lag_within_day(fr, a.astype(float), lag_k), nan=0.0).astype(bool)
            return b & prior
        if fn == "first_in_day":
            cond = self._eval(args[0], ctx, fit_mask).astype(bool)
            out = np.zeros(fr.n, dtype=bool)
            for d in range(fr.n_days):
                s, e = fr.day_start[d], fr.day_end[d]
                hits = np.flatnonzero(cond[s:e])
                if hits.size:
                    out[s + hits[0]] = True
            return out
        if fn == "in_window":
            return ctx.window_mask(args[0].value)
        if fn == "at_checkpoint":
            return ctx.checkpoint_mask(args[0].value)
        raise CompileError(f"cannot evaluate {fn} here")

    def evaluate(self, ctx: FeatureContext, fit_day_mask: np.ndarray | None = None,
                 label: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Event mask and direction. ``fit_day_mask`` (Playground days) freezes cut-offs."""
        if not self.ok or self.ast is None:
            raise CompileError(f"{self.card_id} is {self.status}: {self.reason}")
        fit_bars = None if fit_day_mask is None else np.asarray(fit_day_mask, bool)[ctx.frame.day_idx]
        if fit_bars is None and not self.fitted:
            raise CompileError(f"{self.card_id}: evaluate on Court/forward days needs frozen cut-offs (fit first)")
        node = self.ast
        if self.is_model:
            mask = evaluate_model_card(self, ctx, fit_bars, label)
            direction = np.ones(ctx.frame.n)
        elif node.kind == "call" and node.value == "DIR":
            up = self._eval(node.args[0], ctx, fit_bars).astype(bool)
            dn = self._eval(node.args[1], ctx, fit_bars).astype(bool)
            mask = up | dn
            direction = np.where(up & ~dn, 1.0, np.where(dn & ~up, -1.0, 0.0))
            mask &= direction != 0
        else:
            mask = self._eval(node, ctx, fit_bars).astype(bool)
            direction = np.ones(ctx.frame.n)
        if fit_bars is not None:
            self.fitted = True
        return mask & ~ctx.frame.masked, direction

    def spec(self, ctx: FeatureContext, fit_day_mask: np.ndarray | None = None, *,
             approach_id: str = "", phase: str = "", label: np.ndarray | None = None) -> dict[str, Any]:
        """TestRunner spec (mask + direction computed now with frozen/fitted cut-offs)."""
        mask, direction = self.evaluate(ctx, fit_day_mask, label)
        card = self.card
        return {"test_id": self.card_id, "mask": mask, "direction": direction,
                "side": card.get("side", "both"), "target": card.get("target", "T1"),
                "horizons": list(card.get("horizons", [1])), "baseline": card.get("baseline", "slot_matched"),
                "family": card.get("target", "T1"), "phase": phase, "approach_id": approach_id}


_TARGET_MENU: tuple[str, ...] = ()


def target_menu() -> tuple[str, ...]:
    """Target ids of ``config/targets.yaml`` (the fixed-horizon outcome menu, §9.4)."""
    global _TARGET_MENU
    if not _TARGET_MENU:
        doc = load_yaml(asset_dir("config") / "targets.yaml")
        _TARGET_MENU = tuple(str(t.get("target_id") or t.get("id")) for t in doc["targets"])
    return _TARGET_MENU


def compile_card(card: dict[str, Any], lib: BrickLibrary, *, available: set[str], cal: dict[str, Any],
                 t7_enabled: bool = False, sklearn_ok: bool | None = None,
                 brick_leak_ctx: FeatureContext | None = None,
                 extra_bricks: Iterable[str] = ()) -> CompiledCard:
    """Compile one Idea Card v2 through every gate (order documented in the module docstring)."""
    card = copy.deepcopy(card)
    result = validate(card, "idea_card_v2")
    if not result.ok:
        return CompiledCard(card, STATUS_INVALID, "schema: " + "; ".join(f"{p}: {m}" for p, m in result.errors[:3]))
    event = str(card["event"])
    leaks = leak_scan(event)
    if leaks:
        return CompiledCard(card, STATUS_LEAK, f"look-ahead reference in event ({leaks[0]})")
    bounds = boundary_scan(f"{card['hypothesis']} || {event}")
    if bounds:
        return CompiledCard(card, STATUS_BOUNDARY, f"Diagnosis-Only Boundary violation ({bounds[0]})")
    if card["target"] == "T7" and not t7_enabled:
        return CompiledCard(card, STATUS_BOUNDARY, "T7 disabled (T7_enabled: false, CF-12, D-6)")
    if card["target"] not in target_menu():
        return CompiledCard(card, STATUS_INVALID, f"unknown target {card['target']!r}: not in config/targets.yaml "
                            "(T9+ needs an accepted TARGET proposal, FR-21)")
    try:
        ast = parse_event(event)
        top = ast.kind == "call" and ast.value in ("DIR", "MODEL")
        if top and ast.value == "DIR":
            for branch in ast.args:
                if _type_of(branch) != "bool":
                    raise CompileError("DIR branches must be boolean")
        elif top and ast.value == "MODEL":
            fam = ast.args[0]
            if fam.kind != "ref" or fam.value not in ("logistic", "ridge", "gbdt"):
                raise CompileError("MODEL(family, bricks...) family must be logistic | ridge | gbdt")
            if any(a.kind != "ref" for a in ast.args[1:]):
                raise CompileError("MODEL features must be brick ids")
        elif _type_of(ast) != "bool":
            raise CompileError("event must be boolean")
    except CompileError as exc:
        return CompiledCard(card, STATUS_INVALID, str(exc))
    if ast.kind == "call" and ast.value == "MODEL":
        bricks = {a.value for a in ast.args[1:]}
        branches = [bricks]
    elif ast.kind == "call" and ast.value == "DIR":
        branches = [_bricks_in(b) for b in ast.args]
        bricks = set().union(*branches)
    else:
        bricks = _bricks_in(ast)
        branches = [bricks]
    extra = set(extra_bricks) | (set(brick_leak_ctx.extra_bricks) if brick_leak_ctx is not None else set())
    unknown = sorted(b for b in bricks if b not in lib.bricks and b not in extra)
    data_tokens = set(card["data_needed"])
    for b in bricks:
        if b in lib.bricks:
            data_tokens |= set(lib.bricks[b].data_needed)
            avail_leaks = availability_check(lib.bricks[b].data_needed, lib.bricks[b].available_at, cal)
            if avail_leaks:
                return CompiledCard(card, STATUS_LEAK, f"availability: {avail_leaks[0]}", ast, tuple(sorted(bricks)))
    avail_leaks = availability_check(card["data_needed"], "bar close", cal)
    if avail_leaks:
        return CompiledCard(card, STATUS_LEAK, f"availability: {avail_leaks[0]}", ast, tuple(sorted(bricks)))
    missing = tuple(sorted(t for t in data_tokens if t not in available))
    if missing:
        return CompiledCard(card, STATUS_NA, f"{NA_DATA_MISSING}: needs {list(missing)}", ast, tuple(sorted(bricks)), missing)
    if unknown:
        return CompiledCard(card, STATUS_INVALID, f"unknown bricks {unknown} (file a BRICK proposal)", ast)
    if any(len(br) > lib.max_bricks for br in branches):
        return CompiledCard(card, STATUS_INVALID, f"more than {lib.max_bricks} bricks in one condition (PR-14)", ast)
    if ast.kind == "call" and ast.value == "MODEL":
        if sklearn_ok is None:
            sklearn_ok = sklearn_available()
        if not sklearn_ok:
            return CompiledCard(card, STATUS_ENV, "NOT-RUN(ENV): scikit-learn not installed (optional [models] extra)", ast, tuple(sorted(bricks)))
    if brick_leak_ctx is not None:
        for b in sorted(bricks):
            if b in lib.bricks:
                ok, why = dynamic_leak_gate(b, brick_leak_ctx)
                if not ok:
                    return CompiledCard(card, STATUS_LEAK, why, ast, tuple(sorted(bricks)))
    return CompiledCard(card, STATUS_OK, "", ast, tuple(sorted(bricks)))


def sklearn_available() -> bool:
    """True when the optional scikit-learn extra is importable."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def evaluate_model_card(cc: CompiledCard, ctx: FeatureContext, fit_bars: np.ndarray | None,
                        label: np.ndarray | None) -> np.ndarray:
    """MODEL card (H7): expanding discovery folds with fold-local cut-offs; frozen model on Court.

    Fit: 3 expanding day folds (min train 50% of discovery days), purge+embargo of ``max_h`` bars
    at each boundary, event = predicted probability >= the fold-local p(1 - top_frac) of the
    training predictions. Out-of-fold AUC (day-block CI) is stored in ``cc.cutoffs['oof_auc']``.
    Court: model refit on all discovery days and cut-off frozen from discovery predictions.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier

    fam = cc.ast.args[0].value
    feats = np.column_stack([ctx.brick(a.value) for a in cc.ast.args[1:]])
    fr = ctx.frame
    top_frac = 0.10

    def make_model() -> Any:
        if fam == "logistic":
            return LogisticRegression(C=1.0, max_iter=500)
        if fam == "ridge":
            return RidgeClassifier(alpha=1.0)
        return GradientBoostingClassifier(max_depth=2, n_estimators=100, learning_rate=0.1, random_state=42)

    def scores(model: Any, x: np.ndarray) -> np.ndarray:
        return model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)

    finite_x = np.isfinite(feats).all(axis=1)
    if fit_bars is not None:
        if label is None:
            raise CompileError("MODEL card fit needs an outcome label array")
        lab = np.asarray(label, float)
        usable = fit_bars & finite_x & np.isfinite(lab)
        disc_days = np.unique(fr.day_idx[fit_bars])
        n = disc_days.size
        max_h = max(int(h) for h in cc.card["horizons"])
        mask = np.zeros(fr.n, dtype=bool)
        oof = np.full(fr.n, np.nan)
        bounds = [int(n * f) for f in (0.5, 2 / 3, 5 / 6, 1.0)]
        for i in range(3):
            train_days = disc_days[: bounds[i]]
            test_days = disc_days[bounds[i]: bounds[i + 1]]
            if train_days.size < 5 or test_days.size == 0:
                continue
            tr_bars = usable & np.isin(fr.day_idx, train_days)
            tr_idx = np.flatnonzero(tr_bars)
            tr_idx = tr_idx[tr_idx < np.flatnonzero(np.isin(fr.day_idx, test_days))[0] - max_h]
            te_bars = usable & np.isin(fr.day_idx, test_days)
            y = (lab[tr_idx] > 0).astype(int)
            if tr_idx.size < 50 or np.unique(y).size < 2 or not te_bars.any():
                continue
            model = make_model().fit(feats[tr_idx], y)
            cut = float(np.quantile(scores(model, feats[tr_idx]), 1.0 - top_frac))
            te_scores = scores(model, feats[te_bars])
            oof[te_bars] = te_scores
            mask[np.flatnonzero(te_bars)[te_scores >= cut]] = True
            cc.cutoffs[f"fold{i}"] = cut
        both = np.isfinite(oof) & np.isfinite(lab)
        if both.any() and np.unique((lab[both] > 0).astype(int)).size == 2:
            cc.cutoffs["oof_auc"] = _auc_with_day_ci(oof[both], (lab[both] > 0).astype(int), fr.day_idx[both])
        all_idx = np.flatnonzero(usable)
        y_all = (lab[all_idx] > 0).astype(int)
        if all_idx.size >= 50 and np.unique(y_all).size == 2:
            cc.model = make_model().fit(feats[all_idx], y_all)
            cc.cutoffs["frozen"] = float(np.quantile(scores(cc.model, feats[all_idx]), 1.0 - top_frac))
        return mask
    if cc.model is None or "frozen" not in cc.cutoffs:
        raise CompileError(f"{cc.card_id}: MODEL card has no frozen model (fit on Playground first)")
    out = np.zeros(fr.n, dtype=bool)
    ok = finite_x
    if ok.any():
        out[np.flatnonzero(ok)] = scores(cc.model, feats[ok]) >= cc.cutoffs["frozen"]
    return out


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size)
    ranks[order] = np.arange(1, scores.size + 1)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _auc_with_day_ci(scores: np.ndarray, y: np.ndarray, days: np.ndarray, B: int = 500, seed: int = 42) -> dict[str, float]:
    auc = _auc(scores, y)
    uniq = np.unique(days)
    rng = np.random.default_rng(seed)
    boots = []
    groups = {d: np.flatnonzero(days == d) for d in uniq}
    for _ in range(B):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([groups[d] for d in pick])
        val = _auc(scores[idx], y[idx])
        if np.isfinite(val):
            boots.append(val)
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan))
    return {"auc": auc, "lo": float(lo), "hi": float(hi), "informative": bool(lo > 0.5)}


# =============================================================================================
# AP cards and proposal gates (FR-07, FR-21)
# =============================================================================================
def load_ap_cards(path: str | Path | None = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """``(approach_meta, merged Idea Card v2)`` for every AP01-AP36 card (defaults merged)."""
    doc = load_yaml(path or asset_dir("config") / "ap_cards.yaml")
    out = []
    for approach in doc["approaches"]:
        meta = {k: v for k, v in approach.items() if k != "cards"}
        meta["group_name"] = doc["groups"].get(approach["group"], "")
        for raw in approach["cards"]:
            card = {**copy.deepcopy(doc["card_defaults"]), **copy.deepcopy(raw)}
            card.setdefault("provenance", f"BASELINE:{approach['approach_id']}")
            out.append((meta, card))
    return out


def load_ap_meta(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Approach metadata (36 rows) including roll-ups with zero cards."""
    doc = load_yaml(path or asset_dir("config") / "ap_cards.yaml")
    return [{**{k: v for k, v in a.items() if k != "cards"}, "n_cards": len(a["cards"]),
             "group_name": doc["groups"].get(a["group"], "")} for a in doc["approaches"]]


@dataclass
class ProposalVerdict:
    proposal_type: str
    accepted: bool
    reason: str
    path: Path | None = None


def gate_brick_proposal(proposal: dict[str, Any], ctx: FeatureContext, cal: dict[str, Any]) -> ProposalVerdict:
    """BRICK proposal: schema, static leak scan, availability rules, dynamic +1-bar shift."""
    res = validate(proposal, "proposal")
    if not res.ok:
        return ProposalVerdict("BRICK", False, "schema: " + res.errors[0][1])
    formula = str(proposal["formula"])
    leaks = leak_scan(formula)
    if leaks:
        return ProposalVerdict("BRICK", False, f"LEAK-REJECTED: look-ahead token {leaks[0]}")
    avail = availability_check(proposal["data_needed"], str(proposal["available_at"]), cal)
    if avail:
        return ProposalVerdict("BRICK", False, f"LEAK-REJECTED: {avail[0]}")
    name = str(proposal["brick"])
    try:
        values = evaluate_formula(formula, ctx)
    except CompileError as exc:
        return ProposalVerdict("BRICK", False, f"formula error: {exc}")
    probe = FeatureContext(frame=ctx.frame, lib=ctx.lib, cal=cal, available=ctx.available, aux=ctx.aux,
                           extra_bricks=dict(ctx.extra_bricks))
    probe.lib = BrickLibrary(bricks={**ctx.lib.bricks, name: BrickDef(name, "pending", "formula", {"formula": formula},
                                                                     formula, str(proposal["available_at"]),
                                                                     tuple(proposal["data_needed"]), "numeric", True)},
                             library_id=ctx.lib.library_id, families=ctx.lib.families, max_bricks=ctx.lib.max_bricks)
    ok, why = dynamic_leak_gate(name, probe)
    if not ok:
        return ProposalVerdict("BRICK", False, f"LEAK-REJECTED: {why}")
    if not np.isfinite(values).any():
        return ProposalVerdict("BRICK", False, "brick is never finite on Playground data")
    return ProposalVerdict("BRICK", True, "leak gate passed (+1 bar shift, availability, static scan)")


def gate_target_proposal(proposal: dict[str, Any]) -> ProposalVerdict:
    res = validate(proposal, "proposal")
    if not res.ok:
        return ProposalVerdict("TARGET", False, "schema: " + res.errors[0][1])
    bad = boundary_scan(f"{proposal['metric']} {proposal['boundary_check']} {proposal['target']}".replace("zero hold/exit/first-touch", ""))
    if bad:
        return ProposalVerdict("TARGET", False, f"BOUNDARY-REJECTED: {bad[0]}")
    return ProposalVerdict("TARGET", True, "fixed-horizon boundary check passed; tagged STRUCTURE-ONLY")


def gate_persona_proposal(proposal: dict[str, Any], new_event_days: Sequence[Iterable[Any]],
                          existing: dict[str, Sequence[Iterable[Any]]], max_median: float = 0.30) -> ProposalVerdict:
    """Persona Foundry gate: median event-day Jaccard vs EVERY existing persona must be < 0.30."""
    res = validate(proposal, "proposal")
    if not res.ok:
        return ProposalVerdict("PERSONA", False, "schema: " + res.errors[0][1])
    worst_name, worst = "", 0.0
    for name, cards in sorted(existing.items()):
        sims = [jaccard(a, b) for a in new_event_days for b in cards]
        med = float(np.median(sims)) if sims else 0.0
        if med > worst:
            worst_name, worst = name, med
    if worst >= max_median:
        return ProposalVerdict("PERSONA", False, f"cosmetic rename: median Jaccard {worst:.2f} vs {worst_name} >= {max_median}")
    return ProposalVerdict("PERSONA", True, f"distinct persona: max median Jaccard {worst:.2f} < {max_median}")


def gate_mechanism_proposal(proposal: dict[str, Any], lib: BrickLibrary) -> ProposalVerdict:
    res = validate(proposal, "proposal")
    if not res.ok:
        return ProposalVerdict("MECHANISM", False, "schema: " + res.errors[0][1])
    if proposal["dose_variable"] not in lib.bricks:
        return ProposalVerdict("MECHANISM", False, f"dose_variable {proposal['dose_variable']} is not a library brick")
    return ProposalVerdict("MECHANISM", True, "dose variable resolvable; negative control declared")


def write_proposal(proposal: dict[str, Any], verdict: ProposalVerdict, proposals_dir: str | Path,
                   seq: int) -> Path:
    """Write ``proposals/<seq>_<type>.yaml`` (symlink/traversal-safe, exit 3 on violation)."""
    root = Path(proposals_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^a-z0-9_]+", "_", str(proposal.get("brick") or proposal.get("target") or proposal.get("persona")
                                         or proposal.get("mechanism") or "proposal").lower())[:40]
    target = safe_path(f"{seq:04d}_{proposal['proposal_type'].lower()}_{name}.yaml", root)
    payload = {**proposal, "gate": {"accepted": verdict.accepted, "reason": sanitize_text(verdict.reason, 300)}}
    atomic_write_yaml(target, payload)
    verdict.path = target
    return target


def write_bricks_pending(path: str | Path, accepted: Sequence[dict[str, Any]]) -> Path:
    """Current-run ``bricks_pending.yaml`` (Maintainer promotes them offline)."""
    doc = {"schema_version": "1.0", "bricks": [
        {"brick_id": p["brick"], "formula": p["formula"], "available_at": p["available_at"],
         "data_needed": list(p["data_needed"]), "leak_gate": "passed"} for p in accepted]}
    return atomic_write_yaml(path, doc)


__all__ = [
    "BOUNDARY_PATTERNS", "BRICK_IMPLS", "LEAK_TOKENS", "STATUS_BOUNDARY", "STATUS_ENV",
    "STATUS_INVALID", "STATUS_LEAK", "STATUS_NA", "STATUS_OK", "BrickDef", "BrickLibrary",
    "CompileError", "CompiledCard", "FeatureContext", "Node", "ProposalVerdict",
    "availability_check", "boundary_scan", "compile_card", "dynamic_leak_gate", "evaluate_formula",
    "evaluate_model_card", "expand_brick", "gate_brick_proposal", "gate_mechanism_proposal",
    "gate_persona_proposal", "gate_target_proposal", "leak_scan", "load_ap_cards", "load_ap_meta",
    "parse_event", "sklearn_available", "write_bricks_pending", "write_proposal",
]
