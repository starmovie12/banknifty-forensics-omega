"""Consolidated statistical engine (FILE-025, FR-06; refactor of legacy 02/03 Ctx/TestRunner).

One engine replaces the duplicated legacy ``Ctx`` / ``TestRunner.run_batch`` / ``horizon_list``
(L8) and fixes the silent horizon drop (CL4 / F7): every dropped horizon is logged in
``tables/test_registry.csv`` and ``planned == executed + logged_skipped`` is enforced.

Contents
* :class:`BarFrame` - chronological numpy bar frame (masked bars are NaN, never dropped/filled).
* Gap-free intraday ATR20 (§10.4 #1) and Wilder TR/ATR for daily files.
* Fixed-horizon outcomes for targets T1-T8 (zero first-touch, NG-2).
* Slot-matched and other-slots-vol-matched baselines (R5).
* Day-block / week-block bootstrap (B=5,000 or disclosed 2,000, seed 42), whole-day sign-flip
  p-values (R14), BH q-values, ROBUST/FRAGILE/NOISE stability (n_blocks = 4).
* MDE, Court alpha spending, ``B_court``, ``n_star`` with ATR20 -> sd conversion (§10.4 #2, #3, #7).
* :class:`TestRunner` with ``run_batch`` and the deterministic fork ``Process``+``Pipe`` map.
"""

from __future__ import annotations

import contextlib
import csv
import math
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from bn_forensics import (
    DEFAULT_SEED,
    IST,
    EventLog,
    InputError,
    InvalidPipeline,
    atomic_write_text,
    sha256_text,
)

SIGNED_TARGETS = frozenset({"T1", "T3", "T6", "T8"})
STRUCTURE_TARGETS = frozenset({"T2", "T3", "T4", "T5", "T7"})
REGISTRY_FIELDS = (
    "test_id", "approach_id", "family", "phase", "target", "horizon", "side", "status", "reason",
    "n_events", "effective_n",
)


# =============================================================================================
# Context
# =============================================================================================
@dataclass
class Ctx:
    """Run context shared by every statistical call (seed, profile, B, test registry)."""

    cfg: dict[str, Any]
    seed: int = DEFAULT_SEED
    profile: str = "full"
    workers: int = 1
    bootstrap_B: int = 5000
    registry: list[dict[str, Any]] = field(default_factory=list)
    log: EventLog | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], *, seed: int | None = None, profile: str = "full",
                    log: EventLog | None = None, workers: int | None = None) -> Ctx:
        if profile not in cfg.get("profiles", {}):
            raise InputError(f"unknown profile {profile!r}")
        prof = cfg["profiles"][profile]
        n_workers = int(prof["workers"]) if workers is None else int(workers)
        if profile == "lite":
            n_workers = min(n_workers, 2)
        return cls(cfg=cfg, seed=DEFAULT_SEED if seed is None else int(seed), profile=profile,
                   workers=max(1, n_workers), bootstrap_B=int(prof["bootstrap_B"]), log=log)

    def rng(self, offset: int = 0) -> np.random.Generator:
        """Fresh deterministic generator ``seed + offset`` (NFR-03)."""
        return np.random.default_rng(self.seed + offset)

    @property
    def stats_cfg(self) -> dict[str, Any]:
        return self.cfg["stats"]

    def disclosed_B(self) -> str:
        """Bootstrap disclosure string (R14)."""
        return f"B={self.bootstrap_B}" + (" (disclosed lite profile)" if self.profile == "lite" else "")

    def write_test_registry(self, path: str | Path) -> Path:
        """Write ``tables/test_registry.csv`` and assert ``planned == executed + logged_skipped``."""
        planned = len(self.registry)
        executed = sum(1 for r in self.registry if r["status"] == "EXECUTED")
        skipped = sum(1 for r in self.registry if r["status"] == "SKIPPED")
        if planned != executed + skipped:
            raise InvalidPipeline(
                f"test registry mismatch: planned {planned} != executed {executed} + skipped {skipped}",
                tripwire_id="TW-3")
        return write_csv(path, list(REGISTRY_FIELDS), self.registry)

    def registry_counts(self) -> dict[str, int]:
        planned = len(self.registry)
        executed = sum(1 for r in self.registry if r["status"] == "EXECUTED")
        return {"planned": planned, "executed": executed, "logged_skipped": planned - executed}


def write_csv(path: str | Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> Path:
    """Deterministic CSV writer (fixed float format, LF endings, atomic)."""
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: fmt_cell(row.get(k)) for k in fields})
    return atomic_write_text(path, buf.getvalue())


def fmt_cell(value: Any) -> str:
    """Stable text form for CSV cells (``%.10g`` floats, empty for None/NaN)."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "" if math.isnan(float(value)) else ("inf" if value > 0 else "-inf")
        text = f"{float(value):.10g}"
        return "0" if text == "-0" else text
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


# =============================================================================================
# Bar frame
# =============================================================================================
@dataclass
class BarFrame:
    """Chronological bar frame. Masked (stale/filled/iv not OK) bars hold NaN prices."""

    dates: np.ndarray          # datetime64[D] per bar
    day_idx: np.ndarray        # int index into ``days`` per bar
    slot: np.ndarray           # 0-based bar index within the observed session
    minute: np.ndarray         # IST minute-of-day of the bar OPEN
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray              # noqa: E741 - OHLC naming
    c: np.ndarray
    v: np.ndarray
    oi: np.ndarray
    masked: np.ndarray
    days: np.ndarray           # unique datetime64[D]
    day_start: np.ndarray
    day_end: np.ndarray
    interval_min: int
    bars_per_day: int
    meta: dict[str, Any] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return int(self.c.shape[0])

    @property
    def n_days(self) -> int:
        return int(self.days.shape[0])

    @property
    def close_minute(self) -> np.ndarray:
        return self.minute + self.interval_min

    @property
    def is_first_bar(self) -> np.ndarray:
        first = np.zeros(self.n, dtype=bool)
        first[self.day_start] = True
        return first

    @property
    def is_last_bar(self) -> np.ndarray:
        last = np.zeros(self.n, dtype=bool)
        last[self.day_end - 1] = True
        return last

    def session_open_minute(self) -> np.ndarray:
        """Observed first-bar open minute per day."""
        return self.minute[self.day_start]

    def session_close_minute(self) -> np.ndarray:
        """Observed last-bar close minute per day."""
        return self.minute[self.day_end - 1] + self.interval_min

    @property
    def tr(self) -> np.ndarray:
        if "tr" not in self._cache:
            self._cache["tr"] = true_range(self.o, self.h, self.l, self.c, self.day_start)
        return self._cache["tr"]

    @property
    def atr20(self) -> np.ndarray:
        if "atr20" not in self._cache:
            self._cache["atr20"] = rolling_mean_strict(self.tr, 20)
        return self._cache["atr20"]

    def day_mask_to_bar_mask(self, day_mask: np.ndarray | None) -> np.ndarray:
        if day_mask is None:
            return np.ones(self.n, dtype=bool)
        return np.asarray(day_mask, dtype=bool)[self.day_idx]

    def subset_days(self, keep_days: np.ndarray) -> BarFrame:
        """New frame restricted to the given dates (datetime64[D] array), order preserved."""
        keep_days = np.asarray(keep_days, dtype="datetime64[D]")
        day_keep = np.isin(self.days, keep_days)
        bar_keep = day_keep[self.day_idx]
        return _frame_from_arrays(
            self.dates[bar_keep], self.slot[bar_keep], self.minute[bar_keep], self.o[bar_keep],
            self.h[bar_keep], self.l[bar_keep], self.c[bar_keep], self.v[bar_keep],
            self.oi[bar_keep], self.masked[bar_keep], self.interval_min, dict(self.meta))

    def day_ohlc(self) -> dict[str, np.ndarray]:
        """Per-day open/high/low/close from non-masked bars (NaN if a day is fully masked)."""
        if "day_ohlc" not in self._cache:
            starts = self.day_start
            hi = np.fmax.reduceat(np.where(np.isnan(self.h), -np.inf, self.h), starts)
            lo = np.fmin.reduceat(np.where(np.isnan(self.l), np.inf, self.l), starts)
            hi = np.where(np.isfinite(hi), hi, np.nan)
            lo = np.where(np.isfinite(lo), lo, np.nan)
            opens = np.full(self.n_days, np.nan)
            closes = np.full(self.n_days, np.nan)
            for d in range(self.n_days):
                seg_o = self.o[self.day_start[d]:self.day_end[d]]
                seg_c = self.c[self.day_start[d]:self.day_end[d]]
                ok_o = np.flatnonzero(~np.isnan(seg_o))
                ok_c = np.flatnonzero(~np.isnan(seg_c))
                if ok_o.size:
                    opens[d] = seg_o[ok_o[0]]
                if ok_c.size:
                    closes[d] = seg_c[ok_c[-1]]
            self._cache["day_ohlc"] = {"open": opens, "high": hi, "low": lo, "close": closes}
        return self._cache["day_ohlc"]

    def week_ids(self) -> np.ndarray:
        """ISO-week id per day (Monday-based calendar week number since epoch)."""
        days_int = self.days.astype("datetime64[D]").astype(np.int64)
        return (days_int + 3) // 7  # 1970-01-01 is a Thursday -> shift to Monday start


def _frame_from_arrays(dates: np.ndarray, slot: np.ndarray, minute: np.ndarray, o: np.ndarray,
                       h: np.ndarray, lo: np.ndarray, c: np.ndarray, v: np.ndarray,
                       oi: np.ndarray, masked: np.ndarray, interval_min: int,
                       meta: dict[str, Any]) -> BarFrame:
    dates = np.asarray(dates, dtype="datetime64[D]")
    if dates.size == 0:
        raise InputError("empty bar frame")
    days, first_idx, day_idx = np.unique(dates, return_index=True, return_inverse=True)
    order_ok = np.all(np.diff(first_idx) > 0)
    if not order_ok:
        raise InputError("bar frame is not chronologically ordered by day")
    counts = np.bincount(day_idx, minlength=days.size)
    day_start = first_idx.astype(np.int64)
    day_end = day_start + counts
    bpd = int(np.bincount(counts).argmax()) if counts.size else 0
    return BarFrame(dates=dates, day_idx=day_idx.astype(np.int64), slot=np.asarray(slot, np.int64),
                    minute=np.asarray(minute, np.int64), o=np.asarray(o, float), h=np.asarray(h, float),
                    l=np.asarray(lo, float), c=np.asarray(c, float), v=np.asarray(v, float),
                    oi=np.asarray(oi, float), masked=np.asarray(masked, bool), days=days,
                    day_start=day_start, day_end=day_end, interval_min=int(interval_min),
                    bars_per_day=bpd, meta=meta)


def to_ist_minutes(ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """tz-aware timestamps (us or ns, any +05:30 tz) -> (IST date datetime64[D], minute-of-day)."""
    series = pd.Series(ts)
    if getattr(series.dt, "tz", None) is None:
        raise InputError("timestamps must be timezone-aware (+05:30)")
    local = series.dt.tz_convert(IST)
    dates = local.dt.date.to_numpy().astype("datetime64[D]")
    minutes = (local.dt.hour * 60 + local.dt.minute).to_numpy(dtype=np.int64)
    return dates, minutes


def frame_from_table(df: pd.DataFrame, *, interval_min: int | None = None,
                     meta: dict[str, Any] | None = None) -> BarFrame:
    """Build a :class:`BarFrame` from a canonical BASE_COLS DataFrame (one instrument).

    Stale/filled rows are masked to NaN (X11), never dropped or forward-filled. Slots are the
    position of the bar inside its observed session grid.
    """
    required = {"ts_open", "o", "h", "l", "c"}
    missing = required - set(df.columns)
    if missing:
        raise InputError(f"frame_from_table missing columns: {sorted(missing)}")
    work = df.sort_values("ts_open", kind="mergesort").reset_index(drop=True)
    dates, minutes = to_ist_minutes(work["ts_open"])
    if interval_min is None:
        diffs = np.diff(minutes)
        positive = diffs[diffs > 0]
        interval_min = int(np.bincount(positive).argmax()) if positive.size else 1
    masked = np.zeros(len(work), dtype=bool)
    for flag in ("is_stale", "is_filled"):
        if flag in work.columns:
            masked |= work[flag].fillna(False).to_numpy(dtype=bool)
    if "iv_status" in work.columns:
        masked |= work["iv_status"].astype(str).to_numpy() != "OK"
    o = work["o"].to_numpy(dtype=float).copy()
    h = work["h"].to_numpy(dtype=float).copy()
    lo = work["l"].to_numpy(dtype=float).copy()
    c = work["c"].to_numpy(dtype=float).copy()
    for arr in (o, h, lo, c):
        arr[masked] = np.nan
    v = work["v"].to_numpy(dtype=float).copy() if "v" in work.columns else np.full(len(work), np.nan)
    oi = work["oi"].to_numpy(dtype=float).copy() if "oi" in work.columns else np.full(len(work), np.nan)
    if "has_volume" in work.columns and not bool(work["has_volume"].fillna(False).any()):
        v[:] = np.nan
    v[masked] = np.nan
    _, first_idx, day_idx = np.unique(dates, return_index=True, return_inverse=True)
    slot = np.zeros(len(work), dtype=np.int64)
    open_by_day = minutes[first_idx]
    slot = (minutes - open_by_day[day_idx]) // max(1, interval_min)
    return _frame_from_arrays(dates, slot, minutes, o, h, lo, c, v, oi, masked, interval_min,
                              dict(meta or {}))


def resample(frame: BarFrame, minutes: int) -> BarFrame:
    """Rebuild ``minutes``-bars from a finer frame (Lab rebuild of bars_3m / bars_15m).

    OHLC come from non-masked constituents; a bar is masked only when every constituent is.
    """
    if minutes % frame.interval_min:
        raise InputError(f"cannot resample {frame.interval_min}m bars into {minutes}m bars")
    k = minutes // frame.interval_min
    bucket = frame.slot // k
    key = frame.day_idx * 100_000 + bucket
    if np.any(np.diff(key) < 0):
        raise InputError("resample needs bars ordered by day and slot")
    starts = np.flatnonzero(np.r_[True, np.diff(key) != 0])
    n_out = starts.size
    o = np.full(n_out, np.nan)
    c = np.full(n_out, np.nan)
    for g_idx in range(n_out):
        lo_i = starts[g_idx]
        hi_i = starts[g_idx + 1] if g_idx + 1 < n_out else frame.n
        seg_o = frame.o[lo_i:hi_i]
        seg_c = frame.c[lo_i:hi_i]
        ok_o = np.flatnonzero(~np.isnan(seg_o))
        ok_c = np.flatnonzero(~np.isnan(seg_c))
        if ok_o.size:
            o[g_idx] = seg_o[ok_o[0]]
        if ok_c.size:
            c[g_idx] = seg_c[ok_c[-1]]
    h = np.fmax.reduceat(np.where(np.isnan(frame.h), -np.inf, frame.h), starts)
    lo_arr = np.fmin.reduceat(np.where(np.isnan(frame.l), np.inf, frame.l), starts)
    h = np.where(np.isfinite(h), h, np.nan)
    lo_arr = np.where(np.isfinite(lo_arr), lo_arr, np.nan)
    v = np.add.reduceat(np.nan_to_num(frame.v, nan=0.0), starts)
    if np.all(np.isnan(frame.v)):
        v = np.full(n_out, np.nan)
    oi = frame.oi[np.r_[starts[1:] - 1, frame.n - 1]]
    masked = np.logical_and.reduceat(frame.masked, starts)
    first_bar = starts
    dates = frame.dates[first_bar]
    minute = frame.minute[first_bar]
    slot = frame.slot[first_bar] // k
    return _frame_from_arrays(dates, slot, minute, o, h, lo_arr, c, v, oi, masked, minutes,
                              dict(frame.meta, resampled_from=frame.interval_min))


# =============================================================================================
# ATR and rolling helpers
# =============================================================================================
def true_range(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray,
               day_start: np.ndarray) -> np.ndarray:
    """Gap-free intraday TR (§10.4 #1): ``TR_1 = H_1 - L_1`` on every session's first bar."""
    prev_c = np.r_[np.nan, c[:-1]]
    prev_c[np.asarray(day_start, dtype=np.int64)] = np.nan
    hl = h - lo
    gap_terms = np.fmax(np.abs(h - prev_c), np.abs(lo - prev_c))
    tr = np.fmax(hl, gap_terms)
    tr[np.isnan(hl)] = np.nan
    return tr


def wilder_true_range(h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Daily-file Wilder TR on all bars (gap included)."""
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.fmax(h - lo, np.fmax(np.abs(h - prev_c), np.abs(lo - prev_c)))
    tr[np.isnan(h - lo)] = np.nan
    return tr


def wilder_smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder smoothing (RMA) with NaN-skipping seed."""
    out = np.full(x.shape[0], np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if valid.size < n:
        return out
    start = valid[n - 1]
    seed_vals = x[valid[:n]]
    acc = float(np.mean(seed_vals))
    out[start] = acc
    for i in range(start + 1, x.shape[0]):
        if not np.isnan(x[i]):
            acc = acc + (x[i] - acc) / n
        out[i] = acc
    return out


def rolling_mean_strict(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean that is NaN whenever any value inside the window is NaN."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < window:
        return out
    isnan = np.isnan(x)
    vals = np.where(isnan, 0.0, x)
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    cnan = np.concatenate([[0], np.cumsum(isnan.astype(np.int64))])
    idx = np.arange(window - 1, n)
    sums = csum[idx + 1] - csum[idx + 1 - window]
    nans = cnan[idx + 1] - cnan[idx + 1 - window]
    out[idx] = np.where(nans == 0, sums / window, np.nan)
    return out


def rolling_std_strict(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling sample std (ddof=1), NaN if any NaN in the window."""
    mean = rolling_mean_strict(x, window)
    mean_sq = rolling_mean_strict(x * x, window)
    var = (mean_sq - mean * mean) * window / max(window - 1, 1)
    return np.sqrt(np.clip(var, 0.0, None))


def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Causal EMA (alpha = 2/(span+1)); NaNs carry the previous EMA without updating."""
    alpha = 2.0 / (span + 1.0)
    out = np.full(x.shape[0], np.nan)
    acc = np.nan
    for i, val in enumerate(x):
        if np.isnan(val):
            out[i] = acc
            continue
        acc = val if np.isnan(acc) else acc + alpha * (val - acc)
        out[i] = acc
    return out


def daily_atr(frame: BarFrame, n: int = 20) -> np.ndarray:
    """Wilder ATR(n) on daily OHLC built from the frame (daily-file convention)."""
    d = frame.day_ohlc()
    tr = wilder_true_range(d["high"], d["low"], d["close"])
    return wilder_smooth(tr, n)


# =============================================================================================
# Horizon cap (R24, CL4 / F7)
# =============================================================================================
def horizon_cap(bars_per_day: int, cap_frac: float = 0.25) -> int:
    """Largest allowed intraday horizon: ``floor(bars_per_day * 1/4)``."""
    return math.floor(bars_per_day * cap_frac)


def horizon_list(requested: Iterable[int], bars_per_day: int,
                 cap_frac: float = 0.25) -> tuple[list[int], list[tuple[int, str]]]:
    """Split requested horizons into kept and (horizon, reason) skipped - never silently."""
    cap = horizon_cap(bars_per_day, cap_frac)
    kept: list[int] = []
    skipped: list[tuple[int, str]] = []
    for h in requested:
        h_int = int(h)
        if h_int < 1:
            skipped.append((h_int, "horizon < 1"))
        elif h_int > cap:
            skipped.append((h_int, f"exceeds bars_per_day/4 (cap {cap} at {bars_per_day} bars/day)"))
        else:
            kept.append(h_int)
    return kept, skipped


# =============================================================================================
# Outcomes (fixed horizon only; NG-2)
# =============================================================================================
def _shift_within_day(frame: BarFrame, x: np.ndarray, h: int) -> np.ndarray:
    """``x[t+h]`` when ``t+h`` lies in the same session, else NaN."""
    out = np.full(frame.n, np.nan)
    if h <= 0 or h >= frame.n:
        return out
    idx = np.arange(frame.n - h)
    same = frame.day_idx[idx + h] == frame.day_idx[idx]
    out[idx[same]] = x[idx[same] + h]
    return out


def _window_extreme(frame: BarFrame, x: np.ndarray, h: int, fn: Callable[..., np.ndarray]) -> np.ndarray:
    stacked = np.vstack([_shift_within_day(frame, x, k) for k in range(1, h + 1)])
    valid = ~np.isnan(stacked).any(axis=0)
    out = np.full(frame.n, np.nan)
    if valid.any():
        out[valid] = fn(stacked[:, valid], axis=0)
    return out


def outcome(frame: BarFrame, target: str, h: int, *, x_threshold: float = 1.0,
            premium: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-bar fixed-horizon outcome for ``target`` in ATR20 (and points where meaningful).

    Returns ``{"atr": ..., "points": ..., "hit": ...}`` arrays (NaN where undefined).
    """
    atr = frame.atr20
    c = frame.c
    if target == "T1":
        fwd = _shift_within_day(frame, c, h)
        pts = fwd - c
        return {"atr": pts / atr, "points": pts, "hit": np.where(np.isnan(pts), np.nan, (pts > 0).astype(float))}
    if target == "T2":
        fwd = _shift_within_day(frame, c, h)
        pts = np.abs(fwd - c)
        return {"atr": pts / atr, "points": pts, "hit": np.full(frame.n, np.nan)}
    if target == "T3":
        hi = _window_extreme(frame, frame.h, h, np.max)
        lo = _window_extreme(frame, frame.l, h, np.min)
        up = (hi - c) / atr
        down = (c - lo) / atr
        return {"atr": up - down, "points": (hi - c) - (c - lo), "hit": np.full(frame.n, np.nan),
                "up": up, "down": down}
    if target == "T4":
        hi = _window_extreme(frame, frame.h, h, np.max)
        lo = _window_extreme(frame, frame.l, h, np.min)
        rng = (hi - lo) / atr
        ind = np.where(np.isnan(rng), np.nan, (rng >= x_threshold).astype(float))
        return {"atr": ind, "points": np.full(frame.n, np.nan), "hit": ind, "range_atr": rng}
    if target == "T5":
        logret = np.log(c / np.r_[np.nan, c[:-1]])
        logret[frame.day_start] = np.nan
        sq = logret * logret
        future = np.vstack([_shift_within_day(frame, sq, k) for k in range(1, h + 1)])
        past = np.vstack([_lag_within_day(frame, sq, k) for k in range(0, h)])
        rv_next = np.sqrt(np.sum(future, axis=0))
        rv_prev = np.sqrt(np.sum(past, axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.log(rv_next / rv_prev)
        val[~np.isfinite(val)] = np.nan
        return {"atr": val, "points": np.full(frame.n, np.nan), "hit": np.full(frame.n, np.nan)}
    if target == "T6":
        if premium is None:
            raise InputError("T6 needs an ATM premium series (opt_1m)")
        fwd = _shift_within_day(frame, premium, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (fwd - premium) / premium
        return {"atr": ret, "points": fwd - premium, "hit": np.where(np.isnan(ret), np.nan, (ret > 0).astype(float))}
    if target == "T7":
        raise InputError("T7 is disabled (T7_enabled: false, D-6)")
    if target == "T8":
        out_pts = np.full(frame.n, np.nan)
        last = frame.day_end - 1
        nxt_first = frame.day_start[1:]
        out_pts[last[:-1]] = frame.o[nxt_first] - frame.c[last[:-1]]
        return {"atr": out_pts / atr, "points": out_pts,
                "hit": np.where(np.isnan(out_pts), np.nan, (out_pts > 0).astype(float))}
    raise InputError(f"unknown target {target!r}")


def _lag_within_day(frame: BarFrame, x: np.ndarray, k: int) -> np.ndarray:
    """``x[t-k]`` when inside the same session, else NaN."""
    if k == 0:
        return x.copy()
    out = np.full(frame.n, np.nan)
    if k >= frame.n:
        return out
    idx = np.arange(k, frame.n)
    same = frame.day_idx[idx - k] == frame.day_idx[idx]
    out[idx[same]] = x[idx[same] - k]
    return out


def lag_within_day(frame: BarFrame, x: np.ndarray, k: int) -> np.ndarray:
    """Public causal lag inside the session."""
    return _lag_within_day(frame, x, k)


# =============================================================================================
# Baselines (R5)
# =============================================================================================
def slot_means(values: np.ndarray, slot: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Mean of finite ``values`` per slot over bars where ``use`` is True (array indexed by slot)."""
    ok = use & np.isfinite(values)
    n_slots = int(slot.max()) + 1 if slot.size else 0
    sums = np.bincount(slot[ok], weights=values[ok], minlength=n_slots)
    cnts = np.bincount(slot[ok], minlength=n_slots)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)


def vol_terciles(frame: BarFrame, cutoffs: tuple[float, float] | None = None,
                 fit_mask: np.ndarray | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """Causal prior-bar ATR20 tercile (1..3) per bar with discovery-frozen cut-offs."""
    prior = np.r_[np.nan, frame.atr20[:-1]]
    if cutoffs is None:
        use = np.isfinite(prior) & (fit_mask if fit_mask is not None else True)
        if not use.any():
            return np.zeros(frame.n, dtype=np.int64), (np.nan, np.nan)
        q1, q2 = np.quantile(prior[use], [1 / 3, 2 / 3])
        cutoffs = (float(q1), float(q2))
    terc = np.where(np.isnan(prior), 0, np.where(prior <= cutoffs[0], 1, np.where(prior <= cutoffs[1], 2, 3)))
    return terc.astype(np.int64), cutoffs


# =============================================================================================
# Resampling statistics
# =============================================================================================
def day_aggregate(values: np.ndarray, day_idx: np.ndarray, n_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-day sums and counts of finite values."""
    ok = np.isfinite(values)
    sums = np.bincount(day_idx[ok], weights=values[ok], minlength=n_days)
    cnts = np.bincount(day_idx[ok], minlength=n_days).astype(float)
    return sums, cnts


def day_block_bootstrap(sums: np.ndarray, counts: np.ndarray, B: int, seed: int,
                        ci: float = 0.95) -> dict[str, float]:
    """Day-block bootstrap of the event-weighted mean ``sum(sums)/sum(counts)``.

    Whole event-days are resampled with replacement (R14). Returns mean, se, lo, hi.
    """
    keep = counts > 0
    s = sums[keep]
    n = counts[keep]
    if s.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mean = float(s.sum() / n.sum())
    if s.size < 2:
        return {"mean": mean, "se": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, s.size, size=(B, s.size))
    boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return {"mean": mean, "se": float(np.std(boot, ddof=1)), "lo": float(lo), "hi": float(hi)}


def week_block_bootstrap(x_d: np.ndarray, week_ids: np.ndarray, B: int, seed: int,
                         ci: float = 0.95) -> dict[str, float]:
    """Bootstrap the mean of day values resampling whole calendar-week blocks (7th Skeptic)."""
    x_d = np.asarray(x_d, float)
    week_ids = np.asarray(week_ids)
    weeks, inv = np.unique(week_ids, return_inverse=True)
    sums = np.bincount(inv, weights=x_d, minlength=weeks.size)
    cnts = np.bincount(inv, minlength=weeks.size).astype(float)
    res = day_block_bootstrap(sums, cnts, B, seed, ci)
    res["n_weeks"] = float(weeks.size)
    return res


def sign_flip_pvalue(x: np.ndarray, sign: int, B: int, seed: int, *, two_sided: bool = False,
                     weights: np.ndarray | None = None, chunk: int = 10_000) -> float:
    """Whole-day sign-flip randomisation p-value ``(1 + #{T* >= T}) / (B + 1)`` (§10.4 #7).

    ``T = sign * sum(w_d x_d) / sum(w_d)``; under H0 each day value's sign is flipped
    independently. One-sided in the frozen ``sign`` unless ``two_sided``.
    """
    x = np.asarray(x, float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    keep = np.isfinite(x) & (w > 0)
    x = x[keep]
    w = w[keep]
    if x.size == 0:
        return 1.0
    wx = w * x
    denom = float(w.sum())
    t_obs = float(wx.sum() / denom)
    stat_obs = abs(t_obs) if two_sided else sign * t_obs
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = int(B)
    tol = 1e-12 * max(1.0, abs(stat_obs))
    while remaining > 0:
        m = min(chunk, remaining)
        flips = rng.integers(0, 2, size=(m, x.size), dtype=np.int8).astype(float) * 2.0 - 1.0
        t_star = (flips * wx).sum(axis=1) / denom
        stat_star = np.abs(t_star) if two_sided else sign * t_star
        exceed += int(np.count_nonzero(stat_star >= stat_obs - tol))
        remaining -= m
    return (1.0 + exceed) / (B + 1.0)


def bh_qvalues(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def stability_label(x_d: np.ndarray, ci_lo: float, ci_hi: float, *, n_blocks: int = 4,
                    min_block_days: int = 10, effective_n_min: int = 30) -> dict[str, Any]:
    """ROBUST / FRAGILE / NOISE / ANECDOTAL (Appendix B ``ROBUST`` definition, R6)."""
    x_d = np.asarray(x_d, float)
    eff_n = int(np.isfinite(x_d).sum())
    if eff_n < effective_n_min:
        return {"label": "ANECDOTAL", "blocks_same_sign": 0, "block_means": []}
    overall = float(np.nanmean(x_d))
    sgn = 1 if overall >= 0 else -1
    blocks = np.array_split(x_d[np.isfinite(x_d)], n_blocks)
    means = [float(np.mean(b)) if b.size else float("nan") for b in blocks]
    enough = all(b.size >= min_block_days for b in blocks)
    same = sum(1 for mval in means if np.isfinite(mval) and np.sign(mval) == sgn)
    ci_excludes = bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_lo > 0 or ci_hi < 0))
    if ci_excludes and enough and same >= n_blocks - 1:
        label = "ROBUST"
    elif ci_excludes:
        label = "FRAGILE"
    else:
        label = "NOISE"
    return {"label": label, "blocks_same_sign": same, "block_means": means}


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    """Symmetric Jaccard overlap of two event-day sets (§10.4 #5)."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def event_days_hash(days: Iterable[Any]) -> str:
    """SHA-256 of the sorted distinct ISO event-day list."""
    iso = sorted({str(np.datetime64(d, "D")) for d in days})
    return sha256_text("\n".join(iso))


# =============================================================================================
# MDE, alpha spending, N* (§10.4 #2, #3, #7)
# =============================================================================================
def z_upper(p: float) -> float:
    """Upper-tail standard-normal quantile ``z_p`` with ``P(Z > z_p) = p``."""
    return float(sps.norm.isf(p))


def alpha_k(k: int, alpha_total: float = 0.05) -> float:
    """Court session alpha ``alpha_k = alpha * 6 / (pi^2 k^2)`` (k >= 1)."""
    if k < 1:
        raise InputError("Court session index k must be >= 1")
    return alpha_total * 6.0 / (math.pi ** 2 * k * k)


def b_court(alpha_session: float, K: int, b_min: int = 10_000, factor: int = 10) -> int:
    """``B_court = max(10,000, ceil(10 / (alpha_k / K)))`` (§10.4 #7, PR-15)."""
    return max(int(b_min), math.ceil(factor / (alpha_session / K)))


def nominal_mde(se: float, factor: float = 2.8) -> float:
    """``Nominal_MDE = 2.8 * SE_day_block`` (80% power at 5%)."""
    return factor * se


def gate_adjusted_mde(se_full: float, q: float, m_app: int, power: float = 0.80) -> float:
    """``(z_{q/(2*m_app)} + z_{power}) * SE_full``."""
    return (z_upper(q / (2.0 * max(1, m_app))) + z_upper(1.0 - power)) * se_full


def court_mde_sd(alpha_session: float, K: int, n_events: float, power: float = 0.80) -> float:
    """``Court_MDE = (z_{alpha_k/K} + z_{0.80}) / sqrt(n_events)`` in outcome-sd units."""
    if n_events <= 0:
        return float("inf")
    return (z_upper(alpha_session / K) + z_upper(1.0 - power)) / math.sqrt(n_events)


def n_star(alpha_session: float, K: int, mde_target_atr20: float, sd_outcome_atr20: float,
           event_days_per_court_day: float, power: float = 0.80) -> dict[str, float]:
    """Event-days and Court days needed for ``MDE_target`` (ATR20 converted to sd first, X08)."""
    if sd_outcome_atr20 <= 0 or not math.isfinite(sd_outcome_atr20):
        raise InputError("sd_outcome_atr20 must be positive to convert MDE_target to sd units")
    z_sum = z_upper(alpha_session / K) + z_upper(1.0 - power)
    mde_sd = mde_target_atr20 / sd_outcome_atr20
    needed = math.ceil((z_sum / mde_sd) ** 2)
    rate = event_days_per_court_day if event_days_per_court_day > 0 else float("nan")
    court_days = math.ceil(needed / rate) if math.isfinite(rate) else needed
    return {"mde_target_atr20": mde_target_atr20, "mde_target_sd": mde_sd, "z_sum": z_sum,
            "event_days_needed": needed, "court_days_needed": court_days}


def n_new(n_old: float, mde_current: float, mde_target: float) -> float:
    """Digital-Twin planning ``N_new = N_old * (MDE_current / MDE_target)^2``."""
    return n_old * (mde_current / mde_target) ** 2


# =============================================================================================
# Event study
# =============================================================================================
@dataclass
class EventResult:
    """Result of one (card, target, horizon) event study."""

    test_id: str
    target: str
    horizon: int
    side: str
    n_events: int
    effective_n: int
    event_value: float
    baseline: float
    lift_atr: float
    lift_points: float
    median_lift: float
    hit_rate_lift_pp: float
    ci_lo: float
    ci_hi: float
    se: float
    p_val: float
    sign: int
    nominal_mde: float
    stability: str
    blocks_same_sign: int
    outcome_sd: float
    event_days: np.ndarray
    x_d: np.ndarray
    day_counts: np.ndarray
    event_index: np.ndarray
    residuals: np.ndarray
    signed_outcome: np.ndarray
    baseline_values: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    def event_day_list(self) -> list[str]:
        return [str(d) for d in self.event_days]

    def summary(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id, "target": self.target, "horizon": self.horizon,
            "side": self.side, "n_events": self.n_events, "effective_n": self.effective_n,
            "event_value": self.event_value, "baseline": self.baseline, "lift_atr": self.lift_atr,
            "lift_points": self.lift_points, "median_lift": self.median_lift,
            "hit_rate_lift_pp": self.hit_rate_lift_pp, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
            "se": self.se, "p_val": self.p_val, "sign": self.sign,
            "nominal_mde": self.nominal_mde, "stability": self.stability,
            "outcome_sd": self.outcome_sd,
        }


def event_study(frame: BarFrame, mask: np.ndarray, *, direction: np.ndarray | None = None,
                side: str = "both", target: str = "T1", horizon: int = 5,
                baseline: str = "slot_matched", day_mask: np.ndarray | None = None,
                test_id: str = "", ctx: Ctx | None = None, premium: np.ndarray | None = None,
                vol_cutoffs: tuple[float, float] | None = None,
                x_threshold: float = 1.0, bootstrap: bool = True) -> EventResult:
    """Fixed-horizon event study of ``mask`` against a matched baseline on ``day_mask`` days."""
    seed = ctx.seed if ctx is not None else DEFAULT_SEED
    B = ctx.bootstrap_B if ctx is not None else 5000
    scfg = ctx.stats_cfg if ctx is not None else {"n_blocks": 4, "min_block_event_days": 10,
                                                  "effective_n_min": 30, "ci_level": 0.95}
    use_bars = frame.day_mask_to_bar_mask(day_mask)
    out = outcome(frame, target, horizon, premium=premium, x_threshold=x_threshold)
    raw = out["atr"]
    pts = out["points"]
    dir_arr = np.ones(frame.n) if direction is None else np.asarray(direction, float)
    if side == "reversal":
        dir_arr = -dir_arr
    signed_target = target in SIGNED_TARGETS
    eff_dir = dir_arr if signed_target else np.ones(frame.n)
    signed = raw * eff_dir
    signed_pts = pts * eff_dir
    event = np.asarray(mask, bool) & use_bars & np.isfinite(signed)
    if signed_target:
        pos_mean = slot_means(raw, frame.slot, use_bars & ~np.asarray(mask, bool))
        pos_pts = slot_means(pts, frame.slot, use_bars & ~np.asarray(mask, bool))
        base_vals = pos_mean[frame.slot] * eff_dir
        base_pts = pos_pts[frame.slot] * eff_dir
    else:
        sm = slot_means(raw, frame.slot, use_bars & ~np.asarray(mask, bool))
        sm_pts = slot_means(pts, frame.slot, use_bars & ~np.asarray(mask, bool))
        base_vals = sm[frame.slot]
        base_pts = sm_pts[frame.slot]
    if baseline == "other_slots_vol_matched":
        terc, _ = vol_terciles(frame, vol_cutoffs, use_bars)
        other = use_bars & ~np.asarray(mask, bool) & np.isfinite(raw)
        for t in (1, 2, 3):
            sel = other & (terc == t)
            mean_t = float(np.mean(raw[sel])) if sel.any() else np.nan
            mean_p = float(np.nanmean(pts[sel])) if sel.any() and np.isfinite(pts[sel]).any() else np.nan
            in_t = terc == t
            base_vals = np.where(in_t, mean_t * eff_dir, base_vals)
            base_pts = np.where(in_t, mean_p * eff_dir, base_pts)
    hit_base = np.full(frame.n, np.nan)
    if target in ("T1", "T6", "T8"):
        up = slot_means((raw > 0).astype(float), frame.slot, use_bars & np.isfinite(raw) & ~np.asarray(mask, bool))
        dn = slot_means((raw < 0).astype(float), frame.slot, use_bars & np.isfinite(raw) & ~np.asarray(mask, bool))
        hit_base = np.where(eff_dir >= 0, up[frame.slot], dn[frame.slot])
    event &= np.isfinite(base_vals)
    idx = np.flatnonzero(event)
    resid = signed[idx] - base_vals[idx]
    resid_pts = signed_pts[idx] - base_pts[idx]
    n_ev = int(idx.size)
    days_of = frame.day_idx[idx]
    sums, cnts = day_aggregate(resid, days_of, frame.n_days) if n_ev else (np.zeros(frame.n_days), np.zeros(frame.n_days))
    ev_day_mask = cnts > 0
    x_d = np.where(ev_day_mask, sums / np.maximum(cnts, 1), np.nan)[ev_day_mask]
    eff_n = int(ev_day_mask.sum())
    if n_ev:
        lift = float(np.mean(resid))
        med = float(np.median(resid))
        ev_val = float(np.mean(signed[idx]))
        base_mean = float(np.mean(base_vals[idx]))
        lift_pts = float(np.nanmean(resid_pts)) if np.isfinite(resid_pts).any() else float("nan")
        hit_lift = float("nan")
        if target in ("T1", "T6", "T8"):
            hits = (signed[idx] > 0).astype(float)
            hit_lift = float(100.0 * (np.mean(hits) - np.nanmean(hit_base[idx])))
        sd_out = float(np.std(signed[idx], ddof=1)) if n_ev > 1 else float("nan")
    else:
        lift = med = ev_val = base_mean = lift_pts = hit_lift = sd_out = float("nan")
    if bootstrap and eff_n >= 2:
        bt = day_block_bootstrap(sums, cnts, B, seed, scfg.get("ci_level", 0.95))
        p_val = sign_flip_pvalue(sums[ev_day_mask] / cnts[ev_day_mask], 1, B, seed,
                                 two_sided=True, weights=cnts[ev_day_mask])
    else:
        bt = {"mean": lift, "se": float("nan"), "lo": float("nan"), "hi": float("nan")}
        p_val = float("nan")
    stab = stability_label(x_d, bt["lo"], bt["hi"], n_blocks=int(scfg.get("n_blocks", 4)),
                           min_block_days=int(scfg.get("min_block_event_days", 10)),
                           effective_n_min=int(scfg.get("effective_n_min", 30)))
    sign = 1 if (np.isfinite(lift) and lift >= 0) else -1
    return EventResult(
        test_id=test_id, target=target, horizon=int(horizon), side=side, n_events=n_ev,
        effective_n=eff_n, event_value=ev_val, baseline=base_mean, lift_atr=lift,
        lift_points=lift_pts, median_lift=med, hit_rate_lift_pp=hit_lift, ci_lo=bt["lo"],
        ci_hi=bt["hi"], se=bt["se"], p_val=p_val, sign=sign,
        nominal_mde=nominal_mde(bt["se"]) if np.isfinite(bt["se"]) else float("nan"),
        stability=stab["label"], blocks_same_sign=stab["blocks_same_sign"], outcome_sd=sd_out,
        event_days=frame.days[ev_day_mask], x_d=x_d, day_counts=cnts[ev_day_mask], event_index=idx,
        residuals=resid, signed_outcome=signed[idx], baseline_values=base_vals[idx],
        extras={"block_means": stab["block_means"], "resid_points": resid_pts},
    )


# =============================================================================================
# TestRunner
# =============================================================================================
class TestRunner:
    """Runs batches of compiled specs and logs every planned / executed / skipped test.

    A spec is a dict with keys ``test_id``, ``mask`` (bool array) or ``mask_fn``
    (``frame -> (mask, direction)``), optional ``direction``, ``side``, ``target``,
    ``horizons``, ``baseline``, ``family``, ``phase``, ``approach_id``, ``na_reason``.
    """

    __test__ = False  # not a pytest collection target

    def __init__(self, ctx: Ctx, frame: BarFrame, *, day_mask: np.ndarray | None = None) -> None:
        self.ctx = ctx
        self.frame = frame
        self.day_mask = day_mask
        cap_frac = float(ctx.stats_cfg.get("horizon_cap_frac", 0.25))
        self.cap_frac = cap_frac
        self.t7_enabled = bool(ctx.cfg.get("targets", {}).get("T7_enabled", False))

    def _log(self, spec: dict[str, Any], horizon: int, status: str, reason: str,
             result: EventResult | None) -> None:
        self.ctx.registry.append({
            "test_id": spec.get("test_id", ""), "approach_id": spec.get("approach_id", ""),
            "family": spec.get("family", ""), "phase": spec.get("phase", ""),
            "target": spec.get("target", "T1"), "horizon": int(horizon),
            "side": spec.get("side", "both"), "status": status, "reason": reason,
            "n_events": None if result is None else result.n_events,
            "effective_n": None if result is None else result.effective_n,
        })

    def run_one(self, spec: dict[str, Any]) -> list[EventResult]:
        target = spec.get("target", "T1")
        horizons = list(spec.get("horizons", [1]))
        kept, skipped = horizon_list(horizons, self.frame.bars_per_day, self.cap_frac)
        for h, reason in skipped:
            self._log(spec, h, "SKIPPED", reason, None)
        if spec.get("na_reason"):
            for h in kept:
                self._log(spec, h, "SKIPPED", str(spec["na_reason"]), None)
            return []
        if target == "T7" and not self.t7_enabled:
            for h in kept:
                self._log(spec, h, "SKIPPED", "T7 disabled (T7_enabled: false, D-6)", None)
            return []
        if "mask_fn" in spec:
            mask, direction = spec["mask_fn"](self.frame)
        else:
            mask, direction = spec["mask"], spec.get("direction")
        results: list[EventResult] = []
        for h in kept:
            res = event_study(self.frame, mask, direction=direction, side=spec.get("side", "both"),
                              target=target, horizon=h, baseline=spec.get("baseline", "slot_matched"),
                              day_mask=self.day_mask, test_id=spec.get("test_id", ""), ctx=self.ctx,
                              premium=spec.get("premium"), vol_cutoffs=spec.get("vol_cutoffs"),
                              x_threshold=float(spec.get("x_threshold", 1.0)))
            self._log(spec, h, "EXECUTED", "", res)
            results.append(res)
        return results

    def run_batch(self, specs: Sequence[dict[str, Any]]) -> list[EventResult]:
        """Run all specs; enforce ``planned == executed + logged_skipped`` for this batch."""
        before = len(self.ctx.registry)
        planned = sum(len(list(s.get("horizons", [1]))) for s in specs)
        out: list[EventResult] = []
        for spec in specs:
            out.extend(self.run_one(spec))
        logged = len(self.ctx.registry) - before
        if logged != planned:
            raise InvalidPipeline(f"run_batch logged {logged} rows for {planned} planned tests (F7)",
                                  tripwire_id="TW-3")
        return out


# =============================================================================================
# Deterministic fork Process + Pipe map (Termux-safe, §7.7)
# =============================================================================================
def _worker(func: Callable[[Any], Any], chunk: list[tuple[int, Any]], conn: Any) -> None:
    try:
        results = [(i, func(item)) for i, item in chunk]
        conn.send(("ok", results))
    except BaseException:  # report any worker failure to the parent
        conn.send(("err", traceback.format_exc()))
    finally:
        conn.close()


def parallel_map(func: Callable[[Any], Any], items: Sequence[Any], workers: int = 1) -> list[Any]:
    """Map ``func`` over ``items`` with ``fork`` ``Process`` + ``Pipe`` only.

    Results are merged in input order (deterministic). Never uses Pool/Queue/Lock (no
    ``sem_open`` on Android). Falls back to serial execution on ``OSError``.
    """
    items = list(items)
    if workers <= 1 or len(items) < 2:
        return [func(x) for x in items]
    try:
        import multiprocessing as mp

        mp_ctx = mp.get_context("fork")
    except (ValueError, OSError):
        return [func(x) for x in items]
    indexed = list(enumerate(items))
    chunks = [indexed[w::workers] for w in range(workers)]
    procs: list[Any] = []
    conns: list[Any] = []
    try:
        for chunk in chunks:
            if not chunk:
                continue
            parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
            proc = mp_ctx.Process(target=_worker, args=(func, chunk, child_conn), daemon=True)
            proc.start()
            child_conn.close()
            procs.append(proc)
            conns.append(parent_conn)
    except OSError:
        for proc in procs:
            with contextlib.suppress(OSError):
                proc.terminate()
        return [func(x) for x in items]
    merged: dict[int, Any] = {}
    errors: list[str] = []
    for conn in conns:
        status, payload = conn.recv()
        conn.close()
        if status == "ok":
            merged.update(dict(payload))
        else:
            errors.append(str(payload))
    for proc in procs:
        proc.join()
    if errors:
        raise RuntimeError("parallel worker failed:\n" + errors[0])
    return [merged[i] for i in range(len(items))]


__all__ = [
    "REGISTRY_FIELDS", "SIGNED_TARGETS", "STRUCTURE_TARGETS", "BarFrame", "Ctx", "EventResult",
    "TestRunner", "alpha_k", "b_court", "bh_qvalues", "court_mde_sd", "daily_atr",
    "day_aggregate", "day_block_bootstrap", "ema", "event_days_hash", "event_study", "fmt_cell",
    "frame_from_table", "gate_adjusted_mde", "horizon_cap", "horizon_list", "jaccard",
    "lag_within_day", "n_new", "n_star", "nominal_mde", "outcome", "parallel_map", "resample",
    "rolling_mean_strict", "rolling_std_strict", "sign_flip_pvalue", "slot_means",
    "stability_label", "to_ist_minutes", "true_range", "vol_terciles", "week_block_bootstrap",
    "wilder_smooth", "wilder_true_range", "wilson_ci", "write_csv", "z_upper",
]
