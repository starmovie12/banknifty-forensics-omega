"""Independent twin implementation (FILE-036, FR-05, F5).

Plain-Python, loop-based re-implementation of the event-study arithmetic. It shares ZERO code with
``core_stats.py`` and imports no data loader of this package (an AST test enforces this, F5):

* bar construction from canonical Parquet (``pyarrow`` directly), masking stale/filled bars,
  re-sampling 1-minute bars into ``k``-minute bars;
* gap-free intraday ATR20 (``TR_1 = H_1 - L_1`` on each session's first bar, §10.4 #1);
* fixed-horizon outcomes T1/T2/T3/T4/T5/T8 in ATR20 units;
* slot-matched and other-slots-vol-matched baselines, per-event residuals, event-weighted lift.

``compare_events`` checks parity with the main engine: counts exact, floats within ``1e-6``
absolute or ``0.1%`` relative. Run as ``python -m bn_forensics.twin_impl --help``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IST_OFFSET = dt.timedelta(hours=5, minutes=30)
SIGNED = ("T1", "T3", "T6", "T8")


@dataclass
class Bar:
    date: str
    minute: int
    o: float
    h: float
    lo: float
    c: float
    masked: bool


def _nan() -> float:
    return float("nan")


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


# =============================================================================================
# Bars
# =============================================================================================
def read_partition_bars(paths: Sequence[str | Path]) -> list[Bar]:
    """Read canonical partitions with pyarrow directly and build 1-minute bars (masked -> NaN)."""
    import pyarrow.parquet as pq

    rows: list[tuple[int, Bar]] = []
    for path in sorted(str(p) for p in paths):
        table = pq.read_table(path, columns=["ts_open", "o", "h", "l", "c", "is_stale", "is_filled"])
        ts_type = table.schema.field("ts_open").type
        ints = table.column("ts_open").cast("int64").to_pylist()
        div = 1000 if ts_type.unit == "ns" else 1
        cols = {name: table.column(name).to_pylist() for name in ("o", "h", "l", "c", "is_stale", "is_filled")}
        for i, raw in enumerate(ints):
            micros = raw // div
            local = dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=micros) + IST_OFFSET
            masked = bool(cols["is_stale"][i]) or bool(cols["is_filled"][i])
            vals = [_nan() if masked else float(cols[k][i]) for k in ("o", "h", "l", "c")]
            rows.append((micros, Bar(local.date().isoformat(), local.hour * 60 + local.minute, vals[0], vals[1], vals[2], vals[3], masked)))
    rows.sort(key=lambda r: r[0])
    return [b for _, b in rows]


def resample_bars(bars: Sequence[Bar], minutes: int, base_minutes: int = 1) -> list[Bar]:
    """Group consecutive same-day bars into ``minutes`` buckets counted from the session's first bar."""
    k = minutes // base_minutes
    out: list[Bar] = []
    day_open: dict[str, int] = {}
    groups: dict[tuple[str, int], list[Bar]] = {}
    order: list[tuple[str, int]] = []
    for b in bars:
        day_open.setdefault(b.date, b.minute)
        slot = (b.minute - day_open[b.date]) // base_minutes
        key = (b.date, slot // k)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(b)
    for key in order:
        grp = groups[key]
        live = [b for b in grp if _finite(b.c)]
        o = next((b.o for b in grp if _finite(b.o)), _nan())
        c = live[-1].c if live else _nan()
        highs = [b.h for b in grp if _finite(b.h)]
        lows = [b.lo for b in grp if _finite(b.lo)]
        out.append(Bar(key[0], grp[0].minute, o, max(highs) if highs else _nan(), min(lows) if lows else _nan(), c,
                       all(b.masked for b in grp)))
    return out


def day_starts(bars: Sequence[Bar]) -> list[bool]:
    return [i == 0 or bars[i].date != bars[i - 1].date for i in range(len(bars))]


def slots(bars: Sequence[Bar]) -> list[int]:
    out: list[int] = []
    counter = 0
    for i, b in enumerate(bars):
        counter = 0 if (i == 0 or b.date != bars[i - 1].date) else counter + 1
        out.append(counter)
    return out


def twin_atr20(bars: Sequence[Bar]) -> list[float]:
    """Gap-free intraday ATR20 with plain loops (NaN unless all 20 TRs are finite)."""
    first = day_starts(bars)
    tr: list[float] = []
    for i, b in enumerate(bars):
        hl = b.h - b.lo
        if not _finite(hl):
            tr.append(_nan())
            continue
        best = hl
        if not first[i] and _finite(bars[i - 1].c):
            best = max(best, abs(b.h - bars[i - 1].c), abs(b.lo - bars[i - 1].c))
        tr.append(best)
    out: list[float] = []
    for i in range(len(tr)):
        if i < 19:
            out.append(_nan())
            continue
        window = tr[i - 19:i + 1]
        out.append(sum(window) / 20.0 if all(_finite(w) for w in window) else _nan())
    return out


def _ahead(bars: Sequence[Bar], i: int, k: int) -> int | None:
    j = i + k
    return j if j < len(bars) and bars[j].date == bars[i].date else None


def twin_outcome(bars: Sequence[Bar], atr: Sequence[float], i: int, target: str, h: int) -> float:
    """One bar's fixed-horizon outcome in ATR20 units (NaN when undefined)."""
    a = atr[i]
    c = bars[i].c
    if target in ("T1", "T2"):
        j = _ahead(bars, i, h)
        if j is None:
            return _nan()
        move = bars[j].c - c
        return (move if target == "T1" else abs(move)) / a
    if target in ("T3", "T4"):
        idx = [_ahead(bars, i, k) for k in range(1, h + 1)]
        if any(j is None for j in idx):
            return _nan()
        his = [bars[j].h for j in idx]
        los = [bars[j].lo for j in idx]
        if not all(_finite(x) for x in his + los):
            return _nan()
        if target == "T3":
            return ((max(his) - c) - (c - min(los))) / a
        rng = (max(his) - min(los)) / a
        return _nan() if not _finite(rng) else (1.0 if rng >= 1.0 else 0.0)
    if target == "T5":
        def logret(j: int) -> float:
            if j <= 0 or bars[j].date != bars[j - 1].date:
                return _nan()
            return math.log(bars[j].c / bars[j - 1].c)
        fut = [_ahead(bars, i, k) for k in range(1, h + 1)]
        past = [i - k if i - k >= 0 and bars[i - k].date == bars[i].date else None for k in range(0, h)]
        if any(j is None for j in fut + past):
            return _nan()
        f = [logret(j) for j in fut]
        p = [logret(j) for j in past]
        if not all(_finite(x) for x in f + p):
            return _nan()
        rv_next, rv_prev = math.sqrt(sum(x * x for x in f)), math.sqrt(sum(x * x for x in p))
        if rv_prev == 0 or rv_next == 0:
            return _nan()
        return math.log(rv_next / rv_prev)
    if target == "T8":
        if i + 1 >= len(bars) or bars[i + 1].date == bars[i].date:
            return _nan()
        return (bars[i + 1].o - c) / a
    raise ValueError(f"twin does not implement target {target}")


def _quantile(values: list[float], q: float) -> float:
    """numpy-compatible linear-interpolation quantile written with plain loops."""
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


@dataclass
class EventSpec:
    index: int
    direction: float


def twin_event_study(bars: Sequence[Bar], events: Sequence[EventSpec], *, target: str, h: int, side: str,
                     baseline: str = "slot_matched", day_set: Iterable[str] | None = None,
                     event_mask_all: Iterable[int] | None = None) -> dict[str, Any]:
    """Per-event residuals ``signed outcome - matched baseline`` and the event-weighted lift.

    ``event_mask_all`` lists every bar index flagged by the card (excluded from the baseline
    pool); ``events`` are the bars to score.
    """
    atr = twin_atr20(bars)
    days = set(day_set) if day_set is not None else {b.date for b in bars}
    flagged = set(event_mask_all) if event_mask_all is not None else {e.index for e in events}
    slot_of = slots(bars)
    raw = [twin_outcome(bars, atr, i, target, h) if _finite(atr[i]) or target == "T5" else _nan() for i in range(len(bars))]
    sums: dict[int, float] = {}
    cnts: dict[int, int] = {}
    for i, val in enumerate(raw):
        if bars[i].date in days and i not in flagged and _finite(val):
            sums[slot_of[i]] = sums.get(slot_of[i], 0.0) + val
            cnts[slot_of[i]] = cnts.get(slot_of[i], 0) + 1
    tercile_mean: dict[int, float] = {}
    terc: list[int] = []
    if baseline == "other_slots_vol_matched":
        prior = [_nan(), *atr[:-1]]
        pool = [prior[i] for i in range(len(bars)) if bars[i].date in days and _finite(prior[i])]
        q1, q2 = _quantile(pool, 1 / 3), _quantile(pool, 2 / 3)
        terc = [0 if not _finite(p) else (1 if p <= q1 else (2 if p <= q2 else 3)) for p in prior]
        for t in (1, 2, 3):
            vals = [raw[i] for i in range(len(bars)) if bars[i].date in days and i not in flagged and terc[i] == t and _finite(raw[i])]
            tercile_mean[t] = sum(vals) / len(vals) if vals else _nan()
    signed_target = target in SIGNED
    rows = []
    for ev in events:
        i = ev.index
        if bars[i].date not in days:
            continue
        d = ev.direction * (-1.0 if side == "reversal" else 1.0) if signed_target else 1.0
        val = raw[i]
        if not _finite(val):
            continue
        if baseline == "other_slots_vol_matched" and terc and terc[i] in tercile_mean:
            base = tercile_mean[terc[i]] * d
        else:
            s = slot_of[i]
            base = (sums[s] / cnts[s]) * d if cnts.get(s) else _nan()
        if not _finite(base):
            continue
        rows.append({"index": i, "date": bars[i].date, "minute": bars[i].minute, "signed": val * d,
                     "baseline": base, "residual": val * d - base, "atr20": atr[i]})
    n = len(rows)
    lift = sum(r["residual"] for r in rows) / n if n else _nan()
    ev_val = sum(r["signed"] for r in rows) / n if n else _nan()
    base_mean = sum(r["baseline"] for r in rows) / n if n else _nan()
    return {"n_events": n, "effective_n": len({r["date"] for r in rows}), "event_value": ev_val,
            "baseline": base_mean, "lift_atr": lift, "rows": rows}


def compare_values(a: float, b: float, tol_abs: float = 1e-6, tol_rel: float = 1e-3) -> bool:
    """Parity rule: both NaN, or ``|a-b| <= tol_abs`` or ``|a-b| <= tol_rel * |b|``."""
    if not _finite(a) and not _finite(b):
        return True
    if not (_finite(a) and _finite(b)):
        return False
    diff = abs(a - b)
    return diff <= tol_abs or diff <= tol_rel * abs(b)


def compare_events(twin_rows: Sequence[dict[str, Any]], engine_rows: Sequence[dict[str, Any]], *,
                   tol_abs: float = 1e-6, tol_rel: float = 1e-3) -> dict[str, Any]:
    """Counts must match exactly; every compared float within tolerance (F5)."""
    mismatches = []
    if len(twin_rows) != len(engine_rows):
        mismatches.append({"field": "count", "twin": len(twin_rows), "engine": len(engine_rows)})
    for t, e in zip(twin_rows, engine_rows, strict=False):
        for key in ("signed", "baseline", "residual", "atr20"):
            if key in t and key in e and not compare_values(float(t[key]), float(e[key]), tol_abs, tol_rel):
                mismatches.append({"index": t.get("index"), "field": key, "twin": t[key], "engine": e[key]})
    return {"compared": min(len(twin_rows), len(engine_rows)), "mismatches": mismatches, "pass": not mismatches}


def bars_from_rows(rows: Iterable[dict[str, Any]]) -> list[Bar]:
    """Bars from plain dict rows (``date, minute, o, h, l, c, masked``) - used by CLI and tests."""
    out = []
    for r in rows:
        masked = str(r.get("masked", "false")).lower() in ("1", "true")
        vals = [_nan() if masked else float(r[k]) for k in ("o", "h", "l", "c")]
        out.append(Bar(str(r["date"]), int(r["minute"]), vals[0], vals[1], vals[2], vals[3], masked))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: recompute event residuals for an events CSV against Parquet partitions or a bars CSV."""
    parser = argparse.ArgumentParser(prog="python -m bn_forensics.twin_impl", description=__doc__.splitlines()[0])
    parser.add_argument("--bars-csv", help="CSV with date,minute,o,h,l,c,masked")
    parser.add_argument("--parquet", nargs="*", default=[], help="canonical partition files")
    parser.add_argument("--resample", type=int, default=1)
    parser.add_argument("--events", required=True, help="CSV with index,direction")
    parser.add_argument("--target", default="T1")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--side", default="both")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.bars_csv:
        with open(args.bars_csv, encoding="utf-8") as fh:
            bars = bars_from_rows(csv.DictReader(fh))
    else:
        bars = read_partition_bars(args.parquet)
    if args.resample > 1:
        bars = resample_bars(bars, args.resample)
    with open(args.events, encoding="utf-8") as fh:
        events = [EventSpec(int(r["index"]), float(r["direction"])) for r in csv.DictReader(fh)]
    res = twin_event_study(bars, events, target=args.target, h=args.horizon, side=args.side)
    Path(args.out).write_text(json.dumps({k: v for k, v in res.items() if k != "rows"} | {"rows": res["rows"]}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
