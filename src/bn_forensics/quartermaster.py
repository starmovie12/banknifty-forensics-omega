"""Data Quartermaster (FILE-026, FR-01, FR-02, FR-03; refactor of legacy 01_passport_plan.py).

* Canonical Store Adapter: identity adapter for ``updl`` stores (``[week_*/store/<split>/]<table>/
  underlying=<X>/trading_date=<D>/part-*.parquet``) and a pure CSV adapter for broker files, both
  emitting exact ``BASE_COLS`` (§10.1) with a row-conservation report.
* 13-point Quarantine Gate ``Q1``-``Q13`` (§10.1b). Stale/filled bars and ``iv_status != OK`` are
  masked, never dropped or forward-filled; exact ``rows_in == rows_out + rows_dropped_logged``.
* Sample-First CP0 probe with identity echo-back, data-derived session grids and expiry calendar
  (R29), era tags, Source Passports, ``data_passport.md``.
* VoI-ranked broker-neutral Data Request Tickets (``tickets/open_tickets.yaml``).

``raw/`` and the source store are protected by the guarded writer and verified by SHA-256
manifests before and after intake (NFR-06). Text inside data files is data only (R26, F28).
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bn_forensics import (
    IST,
    IST_OFFSET_MINUTES,
    NA_DATA_MISSING,
    EventLog,
    InputError,
    PathViolation,
    RowConservationError,
    atomic_write_bytes,
    atomic_write_text,
    atomic_write_yaml,
    iso_ist,
    load_yaml,
    parse_hhmm,
    protect_root,
    safe_path,
    sha256_bytes,
    sha256_file,
    sha256_json,
    tree_manifest,
)
from bn_forensics.schemas_validator import sanitize_text, scan_injection, validate_or_raise

# =============================================================================================
# Store contract (§10.1)
# =============================================================================================
TS_TYPE = pa.timestamp("us", tz="+05:30")
BASE_COLS: tuple[tuple[str, pa.DataType], ...] = (
    ("instrument_id", pa.string()),
    ("instrument_key", pa.string()),
    ("underlying", pa.string()),
    ("segment", pa.string()),
    ("expiry_date", pa.date32()),
    ("strike", pa.float64()),
    ("option_type", pa.string()),
    ("interval", pa.string()),
    ("ts_open", TS_TYPE),
    ("ts_close", TS_TYPE),
    ("o", pa.float64()),
    ("h", pa.float64()),
    ("l", pa.float64()),
    ("c", pa.float64()),
    ("v", pa.int64()),
    ("oi", pa.int64()),
    ("has_volume", pa.bool_()),
    ("is_stale", pa.bool_()),
    ("is_filled", pa.bool_()),
    ("session_type", pa.string()),
    ("trading_date", pa.date32()),
    ("adjusted_status", pa.string()),
    ("source", pa.string()),
    ("fetch_run_id", pa.string()),
    ("raw_sha256", pa.string()),
)
BASE_NAMES: tuple[str, ...] = tuple(name for name, _ in BASE_COLS)
BASE_SCHEMA = pa.schema(list(BASE_COLS))
# DEF-18: §10.1 names derived tables but gives no per-table column lists; these are the minimal
# identity + availability columns observed in S7 (o_DERIVED..., iv_status, version, check_time).
DERIVED_REQUIRED: dict[str, tuple[str, ...]] = {
    "bars_3m": ("underlying", "trading_date", "ts_open", "ts_close", "o_DERIVED", "h_DERIVED",
                "l_DERIVED", "c_DERIVED", "available_from_ts"),
    "bars_15m": ("underlying", "trading_date", "ts_open", "ts_close", "o_DERIVED", "h_DERIVED",
                 "l_DERIVED", "c_DERIVED", "available_from_ts"),
    "chain_1m": ("underlying", "trading_date", "ts_open", "strike", "option_type", "oi",
                 "available_from_ts"),
    "chain_metrics_1m": ("underlying", "trading_date", "ts_open", "available_from_ts"),
    "iv_greeks_1m": ("underlying", "trading_date", "ts_open", "strike", "option_type", "iv",
                     "iv_status", "available_from_ts"),
    "iv_daily_DERIVED": ("underlying", "trading_date", "check_time", "available_from_ts"),
    "daily_metrics_DERIVED": ("underlying", "trading_date", "version", "available_from_ts"),
}
SEGMENT_MAP = {"NSE_INDEX": "INDEX", "NSE_EQ": "EQ", "BSE_INDEX": "INDEX", "INDEX": "INDEX",
               "FUT": "FUT", "OPT": "OPT", "EQ": "EQ", "GLOBAL": "GLOBAL"}
STORE_RE = re.compile(
    r"(?:^|/)(?:(?P<week>week_[^/]+)/)?(?:store/)?(?:(?P<split>dev|holdout|lab)/)?"
    r"(?P<table>[A-Za-z0-9_]+)/underlying=(?P<underlying>[^/]+)/"
    r"trading_date=(?P<date>\d{4}-\d{2}-\d{2})/(?P<file>[^/]+\.parquet)$")
CSV_ALIASES = {
    "timestamp": ("timestamp", "datetime", "ts", "time", "date_time", "ts_open"),
    "o": ("open", "o"), "h": ("high", "h"), "l": ("low", "l"), "c": ("close", "c", "ltp"),
    "v": ("volume", "v", "vol"), "oi": ("oi", "open_interest"),
    "symbol": ("symbol", "underlying", "tradingsymbol", "instrument"),
    "interval": ("interval", "timeframe"),
}
CHECK_IDS = tuple(f"Q{i}" for i in range(1, 14))
FAMILY_TABLES = {
    "spot": ("spot_1m",), "futures": ("fut_1m",), "options": ("opt_1m", "opt_wide_1m", "chain_1m",
                                                               "chain_metrics_1m", "iv_greeks_1m"),
    "vix": ("vix_1m", "vix_1d"), "cross_asset": ("stock_1m", "global_1m"),
    "daily": ("spot_1d", "iv_daily_DERIVED", "daily_metrics_DERIVED"),
}


def canonical_segment(segment: str, option_type: str | None) -> str:
    """Derived canonical segment (vendor value is kept unchanged in ``segment``)."""
    seg = str(segment)
    if seg == "NSE_FO":
        return "OPT" if option_type in ("CE", "PE") else "FUT"
    return SEGMENT_MAP.get(seg, "GLOBAL")


# =============================================================================================
# Era calendar helpers (R29)
# =============================================================================================
def load_era_calendar(path: str | Path | None = None) -> dict[str, Any]:
    """Load FILE-016 ``era_calendar.yaml``."""
    from bn_forensics import asset_dir

    cal = load_yaml(path or asset_dir("config") / "era_calendar.yaml")
    if not isinstance(cal, dict) or cal.get("schema_version") != "1.0":
        raise InputError("era_calendar.yaml must carry schema_version '1.0'")
    return cal


def _d(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def era_of(day: dt.date, cal: dict[str, Any]) -> str:
    """Era id (A, B1, B2, B3) of a date from ``era_calendar.yaml``."""
    for era in cal["eras"]:
        lo, hi = _d(era.get("from")), _d(era.get("to"))
        if (lo is None or day >= lo) and (hi is None or day <= hi):
            return str(era["id"])
    return "UNKNOWN"


def table_family(table: str, cal: dict[str, Any]) -> str:
    return str(cal.get("table_family", {}).get(table, "index"))


def expected_grid(table: str, day: dt.date, cal: dict[str, Any]) -> tuple[int, int] | None:
    """Expected ``(open_minute, bars)`` for a table on a date, or None (foreign/global)."""
    fam = table_family(table, cal)
    grids = cal.get("session_grids", {}).get(fam)
    if not grids:
        return None
    for g in grids:
        lo, hi = _d(g.get("from")), _d(g.get("to"))
        if (lo is None or day >= lo) and (hi is None or day <= hi):
            return parse_hhmm(g["open"]), int(g["bars"])
    return None


def exposure_ranges(cal: dict[str, Any]) -> list[tuple[dt.date, dt.date]]:
    return [(_d(r["from"]), _d(r["to"])) for r in cal.get("exposure_calendar", {}).get("ranges", [])]


def is_exposed(day: dt.date, ranges: Sequence[tuple[dt.date, dt.date]]) -> bool:
    return any(lo <= day <= hi for lo, hi in ranges)


def expected_expiry(year: int, month: int, era_id: str, cal: dict[str, Any],
                    trading_days: set[dt.date] | None = None) -> dt.date | None:
    """Monthly expiry predicted by the era rule (last <weekday>, holiday -> previous trading day)."""
    era = next((e for e in cal["eras"] if e["id"] == era_id), None)
    if era is None:
        return None
    rule = str(era.get("expiry_rule", {}).get("monthly", ""))
    match = re.search(r"last (\w+)", rule)
    if not match or match.group(1) not in cal.get("weekday_index", {}):
        return None
    target_wd = int(cal["weekday_index"][match.group(1)])
    nxt = dt.date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    day = nxt - dt.timedelta(days=1)
    while day.weekday() != target_wd:
        day -= dt.timedelta(days=1)
    holidays = {_d(h) for h in cal.get("holidays", [])}
    while day in holidays or (trading_days is not None and trading_days and day not in trading_days
                              and min(trading_days) <= day <= max(trading_days)):
        day -= dt.timedelta(days=1)
    return day


# =============================================================================================
# Store index and reader (identity adapter for updl stores)
# =============================================================================================
@dataclass
class Partition:
    table: str
    underlying: str
    trading_date: dt.date
    paths: list[Path]
    sha256: str
    split: str | None = None
    week: str | None = None


@dataclass
class StoreIndex:
    """Index of canonical partitions under one or more store roots (weeks merged, Q12)."""

    roots: list[Path]
    partitions: dict[tuple[str, str, dt.date], Partition] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    duplicates: int = 0
    side_files: dict[str, list[Path]] = field(default_factory=dict)

    @classmethod
    def scan(cls, roots: Iterable[str | Path]) -> StoreIndex:
        index = cls(roots=[Path(r) for r in roots])
        found: dict[tuple[str, str, dt.date], list[tuple[Path, str, str | None, str | None]]] = defaultdict(list)
        for root in index.roots:
            if not root.exists():
                continue
            if root.is_symlink():
                raise PathViolation(f"store root is a symlink: {root}")
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise PathViolation(f"symlink inside store rejected: {path}")
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                if path.name in ("contracts.csv", "strategy_ready_days.csv"):
                    index.side_files.setdefault(path.name, []).append(path)
                    continue
                m = STORE_RE.search(rel)
                if not m:
                    continue
                key = (m.group("table"), m.group("underlying"), dt.date.fromisoformat(m.group("date")))
                found[key].append((path, sha256_file(path), m.group("split"), m.group("week")))
        for key in sorted(found):
            entries = found[key]
            digests = sorted({e[1] for e in entries})
            paths = sorted({e[0] for e in entries})
            if len(digests) > 1:
                frames = [pq.read_table(p) for p in paths]
                if _tables_equal_content(frames):
                    index.duplicates += len(paths) - 1
                    paths = paths[:1]
                else:
                    index.conflicts.append({"table": key[0], "underlying": key[1],
                                            "trading_date": key[2].isoformat(),
                                            "paths": [p.as_posix() for p in paths],
                                            "sha256": digests})
                    continue
            elif len(paths) > 1:
                index.duplicates += len(paths) - 1
                paths = paths[:1]
            first = entries[0]
            index.partitions[key] = Partition(key[0], key[1], key[2], paths, digests[0], first[2], first[3])
        return index

    def tables(self) -> list[str]:
        return sorted({k[0] for k in self.partitions})

    def underlyings(self, table: str) -> list[str]:
        return sorted({k[1] for k in self.partitions if k[0] == table})

    def dates(self, table: str, underlying: str | None = None) -> list[dt.date]:
        return sorted({k[2] for k in self.partitions if k[0] == table and (underlying is None or k[1] == underlying)})

    def data_version(self) -> str:
        """SHA-256 over every partition key + content digest (``data_version``)."""
        rows = [[k[0], k[1], k[2].isoformat(), p.sha256] for k, p in sorted(self.partitions.items())]
        return sha256_json(rows)


def _tables_equal_content(tables: Sequence[pa.Table]) -> bool:
    """True when partitions differ only by bytes/metadata (identical rows on BASE columns)."""
    first = tables[0]
    cols = [c for c in first.column_names if c not in ("fetch_run_id", "source")]
    ref = first.select(cols).sort_by([(c, "ascending") for c in cols if c in ("instrument_key", "ts_open")])
    for other in tables[1:]:
        if set(other.column_names) != set(first.column_names):
            return False
        cand = other.select(cols).sort_by([(c, "ascending") for c in cols if c in ("instrument_key", "ts_open")])
        if not ref.equals(cand):
            return False
    return True


PartitionGuard = Callable[[Partition], None]


class StoreReader:
    """Raw data loader over a :class:`StoreIndex`. Every partition read passes ``guard`` first.

    ``split_seal.SealedReader`` supplies the guard that blocks Court partitions during Playground
    (F31). This class is the only reader of store Parquet in the package.
    """

    def __init__(self, index: StoreIndex, guard: PartitionGuard | None = None) -> None:
        self.index = index
        self.guard = guard
        self.reads: list[tuple[str, str, str]] = []

    def read(self, table: str, underlying: str, dates: Iterable[dt.date] | None = None,
             columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Read partitions as a pandas DataFrame (Arrow types preserved until conversion)."""
        arrow = self.read_arrow(table, underlying, dates, columns)
        if arrow is None:
            return pd.DataFrame(columns=list(columns) if columns else list(BASE_NAMES))
        return arrow.to_pandas()

    def read_arrow(self, table: str, underlying: str, dates: Iterable[dt.date] | None = None,
                   columns: Sequence[str] | None = None) -> pa.Table | None:
        """Read partitions as one Arrow table (exact store types), or None when nothing matches."""
        wanted = None if dates is None else {_d(d) for d in dates}
        frames: list[pa.Table] = []
        for (tbl, und, day), part in sorted(self.index.partitions.items()):
            if tbl != table or und != underlying or (wanted is not None and day not in wanted):
                continue
            if self.guard is not None:
                self.guard(part)
            for path in part.paths:
                frames.append(pq.read_table(path, columns=list(columns) if columns else None))
            self.reads.append((tbl, und, day.isoformat()))
        if not frames:
            return None
        return pa.concat_tables(frames, promote_options="default")


# =============================================================================================
# CSV adapter (pure function + row-conservation report)
# =============================================================================================
@dataclass
class AdapterReport:
    source_file: str
    sha256: str
    rows_in: int
    rows_out: int
    dropped: dict[str, int]
    symbol: str
    interval: str
    ts_convention: str
    notes: list[str] = field(default_factory=list)

    @property
    def conserved(self) -> bool:
        return self.rows_in == self.rows_out + sum(self.dropped.values())


def _pick(columns: Sequence[str], key: str) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for alias in CSV_ALIASES[key]:
        if alias in lower:
            return lower[alias]
    return None


def probe_ts_convention(first_minutes: Sequence[int], expected_open: int, interval: int) -> str:
    """3-session probe (Q7): START if sessions begin at the grid open, END if one interval later."""
    votes = Counter()
    for m in first_minutes:
        if m == expected_open:
            votes["START"] += 1
        elif m == expected_open + interval:
            votes["END"] += 1
        else:
            votes["AMBIGUOUS"] += 1
    if votes["START"] and not votes["END"] and not votes["AMBIGUOUS"]:
        return "START"
    if votes["END"] and not votes["START"] and not votes["AMBIGUOUS"]:
        return "END"
    return "AMBIGUOUS"


def adapt_csv(path: str | Path, cal: dict[str, Any], *, table: str = "spot_1m",
              source: str = "broker_csv", declared_ts_convention: str = "START") -> tuple[pa.Table, AdapterReport]:
    """Adapt a broker CSV into exact ``BASE_COLS`` (pure function, Q10 conservation report)."""
    src = Path(path)
    raw_bytes = src.read_bytes()
    digest = sha256_bytes(raw_bytes)
    try:
        df = pd.read_csv(src, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise InputError(f"unreadable CSV {src.name}: {exc}") from exc
    rows_in = len(df)
    ts_col = _pick(df.columns, "timestamp")
    needed = {k: _pick(df.columns, k) for k in ("o", "h", "l", "c")}
    if ts_col is None or any(v is None for v in needed.values()):
        raise InputError(f"{src.name}: missing timestamp/open/high/low/close columns (Q2)")
    dropped: Counter[str] = Counter()
    ts = pd.to_datetime(df[ts_col], errors="coerce", format="ISO8601")
    bad_ts = ts.isna()
    if getattr(ts.dt, "tz", None) is None:
        raise InputError(f"{src.name}: timestamps carry no UTC offset; +05:30 is required (Q6)")
    offsets = df[ts_col].str.extract(r"([+-]\d{2}:\d{2}|Z)$")[0]
    if (offsets.dropna() != "+05:30").any():
        raise InputError(f"{src.name}: timestamp UTC offset is not +05:30 on every row (Q6)")
    dropped["unparseable_timestamp"] += int(bad_ts.sum())
    keep = ~bad_ts.to_numpy()
    work = df.loc[keep].reset_index(drop=True)
    ts = ts[keep].reset_index(drop=True).dt.tz_convert(IST)
    nums = {}
    for key, col in needed.items():
        nums[key] = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
    vol_col, oi_col = _pick(work.columns, "v"), _pick(work.columns, "oi")
    v = pd.to_numeric(work[vol_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64) if vol_col else np.zeros(len(work), np.int64)
    oi = pd.to_numeric(work[oi_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64) if oi_col else np.zeros(len(work), np.int64)
    sym_col, int_col = _pick(work.columns, "symbol"), _pick(work.columns, "interval")
    symbol = sanitize_text(work[sym_col].iloc[0], 40) if sym_col and len(work) else "UNKNOWN"
    minutes = (ts.dt.hour * 60 + ts.dt.minute).to_numpy(dtype=np.int64)
    dates = ts.dt.date.to_numpy()
    if int_col and len(work):
        interval_txt = sanitize_text(work[int_col].iloc[0], 8)
        interval_min = int(re.sub(r"\D", "", interval_txt) or "1")
    else:
        diffs = np.diff(minutes)
        positive = diffs[diffs > 0]
        interval_min = int(np.bincount(positive).argmax()) if positive.size else 1
        interval_txt = f"{interval_min}m"
    first_minutes = []
    seen: set[dt.date] = set()
    for d_val, m_val in zip(dates, minutes, strict=True):
        if d_val not in seen:
            seen.add(d_val)
            first_minutes.append(int(m_val))
    probe_days = sorted(seen)[:3]
    grid = expected_grid(table, probe_days[0], cal) if probe_days else None
    convention = probe_ts_convention(first_minutes[:3], grid[0], interval_min) if grid else "AMBIGUOUS"
    notes: list[str] = []
    if convention == "AMBIGUOUS":
        notes.append("A1: 3-session timestamp probe AMBIGUOUS -> treated as START (Q7)")
        convention_used = "START"
    else:
        convention_used = convention
    if convention_used != declared_ts_convention:
        raise InputError(
            f"{src.name}: CP0 timestamp convention mismatch: probe={convention} declared={declared_ts_convention} "
            f"(first bars at {first_minutes[:3]} vs grid open {grid[0] if grid else 'n/a'}) (Q7, F8)")
    ts_open_us = ts.dt.as_unit("us").astype("int64").to_numpy(dtype=np.int64)
    ts_close_us = ts_open_us + interval_min * 60_000_000
    for text_col in work.columns:
        hits = scan_injection(" ".join(map(str, work[text_col].head(50))))
        if hits:
            notes.append(f"F28: instruction-like text in column {sanitize_text(text_col, 30)!r} treated as data: {hits[:2]}")
    n = len(work)
    table_out = pa.table({
        "instrument_id": pa.array([symbol] * n, pa.string()),
        "instrument_key": pa.array([f"CSV|{symbol}"] * n, pa.string()),
        "underlying": pa.array([symbol] * n, pa.string()),
        "segment": pa.array(["NSE_INDEX"] * n, pa.string()),
        "expiry_date": pa.array([None] * n, pa.date32()),
        "strike": pa.array([None] * n, pa.float64()),
        "option_type": pa.array([None] * n, pa.string()),
        "interval": pa.array([interval_txt] * n, pa.string()),
        "ts_open": pa.array(ts_open_us, TS_TYPE),
        "ts_close": pa.array(ts_close_us, TS_TYPE),
        "o": pa.array(nums["o"], pa.float64()),
        "h": pa.array(nums["h"], pa.float64()),
        "l": pa.array(nums["l"], pa.float64()),
        "c": pa.array(nums["c"], pa.float64()),
        "v": pa.array(v, pa.int64()),
        "oi": pa.array(oi, pa.int64()),
        "has_volume": pa.array([bool(v.any())] * n, pa.bool_()),
        "is_stale": pa.array([False] * n, pa.bool_()),
        "is_filled": pa.array([False] * n, pa.bool_()),
        "session_type": pa.array(["NORMAL"] * n, pa.string()),
        "trading_date": pa.array(list(dates), pa.date32()),
        "adjusted_status": pa.array(["UNVERIFIED"] * n, pa.string()),
        "source": pa.array([source] * n, pa.string()),
        "fetch_run_id": pa.array([f"csv:{digest[:12]}"] * n, pa.string()),
        "raw_sha256": pa.array([digest] * n, pa.string()),
    }, schema=BASE_SCHEMA)
    report = AdapterReport(source_file=src.name, sha256=digest, rows_in=rows_in, rows_out=n,
                           dropped=dict(dropped), symbol=symbol, interval=interval_txt,
                           ts_convention=convention, notes=notes)
    if not report.conserved:
        raise RowConservationError(f"adapter row loss in {src.name}: {rows_in} != {n} + {sum(dropped.values())}")
    return table_out, report


# =============================================================================================
# Quarantine Gate
# =============================================================================================
@dataclass
class CheckResult:
    check: str
    status: str   # PASS | FLAG | FAIL | N/A
    detail: str
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class QuarantineReport:
    """All 13 checks + the row ledger for one table/underlying."""

    table: str
    underlying: str
    checks: dict[str, CheckResult]
    rows_in: int
    rows_out: int
    dropped: dict[str, int]
    masked: int
    excluded_days: dict[str, str]
    clean_days: list[dt.date]
    era_counts: dict[str, int]
    ts_convention: str
    echo: str
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def row_ledger_ok(self) -> bool:
        return self.rows_in == self.rows_out + sum(self.dropped.values())

    def failed(self) -> list[str]:
        return [k for k, v in self.checks.items() if v.status == "FAIL"]


def _ts_series_us(table: pa.Table, col: str) -> np.ndarray:
    arr = table.column(col)
    unit = arr.type.unit
    ints = arr.cast(pa.int64()).to_numpy(zero_copy_only=False)
    factor = {"s": 1_000_000, "ms": 1000, "us": 1, "ns": 1}[unit]
    return ints // 1000 if unit == "ns" else ints * factor


def _offset_ok(tz: str | None) -> bool:
    if tz is None:
        return False
    if tz in ("+05:30", "Asia/Kolkata", "Asia/Calcutta"):
        return True
    try:
        from zoneinfo import ZoneInfo

        offset = dt.datetime(2026, 1, 1, tzinfo=ZoneInfo(tz)).utcoffset()
    except (KeyError, ValueError):
        return False
    return offset is not None and int(offset.total_seconds() // 60) == IST_OFFSET_MINUTES


def quarantine_base_table(table: pa.Table, *, table_name: str, cal: dict[str, Any],
                          declared_ts_convention: str = "START") -> tuple[pa.Table, QuarantineReport]:
    """Run Q2-Q8, Q10-Q13 on one BASE_COLS table (Q1 at discovery, Q9 flag-only at Ω-S1).

    Returns the cleaned table (Tier-1 rows removed and logged; stale/filled rows KEPT and masked
    downstream) and the report. Raises on Q6 offset mismatch (exit 3) and Q10 loss (exit 4).
    """
    checks: dict[str, CheckResult] = {}
    rows_in = table.num_rows
    dropped: Counter[str] = Counter()
    missing = [c for c in BASE_NAMES if c not in table.column_names]
    if missing:
        raise InputError(f"{table_name}: missing BASE_COLS {missing} (Q2)")
    checks["Q1"] = CheckResult("Q1", "PASS", "paths resolved inside allowed root, 0 symlinks")
    checks["Q2"] = CheckResult("Q2", "PASS", f"all {len(BASE_NAMES)} BASE_COLS present")
    type_problems = []
    for name, typ in BASE_COLS:
        actual = table.schema.field(name).type
        if name in ("ts_open", "ts_close"):
            if not pa.types.is_timestamp(actual) or actual.unit not in ("us", "ns"):
                type_problems.append(f"{name}:{actual}")
        elif not actual.equals(typ) and not (pa.types.is_large_string(actual) and pa.types.is_string(typ)):
            type_problems.append(f"{name}:{actual}!={typ}")
    if type_problems:
        raise InputError(f"{table_name}: Arrow type mismatch {type_problems} (Q3)")
    for name in ("ts_open", "ts_close"):
        if not _offset_ok(table.schema.field(name).type.tz):
            raise InputError(f"{table_name}: {name} tz {table.schema.field(name).type.tz!r} is not +05:30 (Q6)")
    df = table.to_pandas()
    df["_ts_open_us"] = _ts_series_us(table, "ts_open")
    df["_ts_close_us"] = _ts_series_us(table, "ts_close")
    stale = df["is_stale"].fillna(False).to_numpy(bool) | df["is_filled"].fillna(False).to_numpy(bool)
    ohlc = df[["o", "h", "l", "c"]].to_numpy(dtype=float)
    nonfinite = (~np.isfinite(ohlc)).any(axis=1) & ~stale
    dropped["Q3_nonfinite_ohlc_live"] = int(nonfinite.sum())
    checks["Q3"] = CheckResult("Q3", "PASS" if not nonfinite.any() else "FLAG",
                               f"types ok; {int(nonfinite.sum())} live rows with NaN/Inf OHLC Tier-1 dropped",
                               {"tier1_nonfinite": int(nonfinite.sum())})
    o, h, lo, c = (ohlc[:, i] for i in range(4))
    with np.errstate(invalid="ignore"):
        bad_ohlc = ~((h >= np.maximum(o, c)) & (np.maximum(o, c) >= np.minimum(o, c)) &
                     (np.minimum(o, c) >= lo) & (lo > 0)) & ~stale & ~nonfinite
    dropped["Q4_ohlc_integrity"] = int(bad_ohlc.sum())
    checks["Q4"] = CheckResult("Q4", "PASS" if not bad_ohlc.any() else "FLAG",
                               f"{int(bad_ohlc.sum())} rows violate h>=max(o,c)>=min(o,c)>=l>0 (Tier-1 removed)",
                               {"tier1_ohlc": int(bad_ohlc.sum())})
    keep = ~(nonfinite | bad_ohlc)
    df = df.loc[keep].copy()
    dup = df.duplicated(subset=["instrument_key", "_ts_open_us"], keep="first").to_numpy()
    dropped["Q5_duplicate_ts"] = int(dup.sum())
    df = df.loc[~dup].sort_values(["instrument_key", "_ts_open_us"], kind="mergesort")
    mono = all(np.all(np.diff(g["_ts_open_us"].to_numpy()) > 0) for _, g in df.groupby("instrument_key", sort=True))
    checks["Q5"] = CheckResult("Q5", "PASS" if mono else "FAIL",
                               f"{int(dup.sum())} duplicate (instrument_key, ts_open) rows Tier-1 deduplicated; strictly monotonic={mono}",
                               {"tier1_duplicates": int(dup.sum())})
    local_open = pd.to_datetime(df["_ts_open_us"], unit="us", utc=True).dt.tz_convert(IST)
    df["_minute"] = (local_open.dt.hour * 60 + local_open.dt.minute).to_numpy()
    df["_date"] = df["trading_date"].to_numpy()
    excluded: dict[str, str] = {}
    grid_rows = []
    for day, grp in df.groupby("_date", sort=True):
        stypes = set(grp["session_type"].astype(str))
        day_d = _d(day)
        if stypes != {"NORMAL"}:
            excluded[day_d.isoformat()] = f"session_type {sorted(stypes)} (non-NORMAL, Q6)"
            continue
        exp = expected_grid(table_name, day_d, cal)
        per_inst = grp.groupby("instrument_key")["_minute"].agg(["min", "count", "max"])
        observed_bars = int(per_inst["count"].max())
        observed_open = int(per_inst["min"].min())
        grid_rows.append((day_d, observed_open, observed_bars))
        grid_checked = exp is not None and canonical_segment(
            str(grp["segment"].iloc[0]), grp["option_type"].iloc[0]) in ("INDEX", "FUT", "EQ")
        if grid_checked and (observed_open != exp[0] or observed_bars != exp[1]):
            excluded[day_d.isoformat()] = (f"grid mismatch: observed open {observed_open // 60:02d}:{observed_open % 60:02d} "
                                           f"bars {observed_bars} vs expected {exp[0] // 60:02d}:{exp[0] % 60:02d} bars {exp[1]} (Q6, R29)")
    checks["Q6"] = CheckResult("Q6", "PASS" if not excluded else "FLAG",
                               f"offset +05:30 ok; {len(excluded)} day(s) excluded (non-NORMAL or grid mismatch)",
                               {"excluded_days": len(excluded)})
    probe_days = sorted({d for d, _, _ in grid_rows})[:3]
    first_mins = [o_min for d, o_min, _ in grid_rows if d in probe_days]
    exp0 = expected_grid(table_name, probe_days[0], cal) if probe_days else None
    interval = 1
    if len(df) > 1:
        diffs = np.diff(df["_ts_open_us"].to_numpy()) // 60_000_000
        positive = diffs[diffs > 0]
        interval = int(np.bincount(positive).argmax()) if positive.size else 1
    convention = probe_ts_convention(first_mins, exp0[0], interval) if exp0 else "AMBIGUOUS"
    guard_ok = bool(np.all(df["_ts_close_us"].to_numpy() > df["_ts_open_us"].to_numpy()))
    notes: list[str] = []
    if convention == "AMBIGUOUS":
        notes.append("A1: 3-session timestamp probe ambiguous -> START assumed (Q7)")
    if convention not in ("AMBIGUOUS", declared_ts_convention):
        raise InputError(f"{table_name}: timestamp convention probe {convention} != declared {declared_ts_convention} (Q7, F8)")
    checks["Q7"] = CheckResult("Q7", "PASS" if guard_ok else "FAIL",
                               f"ts_convention={convention if convention != 'AMBIGUOUS' else 'START (A1)'}; ts_close > ts_open on 100% rows={guard_ok}")
    masked_n = int(stale[keep][~dup].sum()) if len(stale) else 0
    missing_bars = 0
    for day_d, _o_min, obs in grid_rows:
        exp = expected_grid(table_name, day_d, cal)
        if exp is not None:
            missing_bars += max(0, exp[1] - obs)
    checks["Q8"] = CheckResult("Q8", "PASS", f"{masked_n} stale/filled bars masked to NaN (never dropped/filled); {missing_bars} missing bars vs grid (never filled)",
                               {"masked": masked_n, "missing_vs_grid": missing_bars})
    checks["Q9"] = CheckResult("Q9", "N/A", "Tier-2 MAD spike flags are computed on Playground days only at Ω-S1 (flag-only)")
    rows_out = len(df)
    ledger_ok = rows_in == rows_out + sum(dropped.values())
    checks["Q10"] = CheckResult("Q10", "PASS" if ledger_ok else "FAIL",
                                f"rows_in {rows_in} == rows_out {rows_out} + dropped_logged {sum(dropped.values())}: {ledger_ok}",
                                {"rows_in": rows_in, "rows_out": rows_out, "dropped": sum(dropped.values())})
    if not ledger_ok:
        raise RowConservationError(f"{table_name}: row conservation failed (Q10)")
    adj = sorted(set(df["adjusted_status"].astype(str)))
    tags = []
    if canonical_segment(str(df["segment"].iloc[0]) if len(df) else "", None) == "FUT" and adj != ["UNADJUSTED"]:
        tags.append("LEVELS-NEED-UNADJUSTED")
    checks["Q11"] = CheckResult("Q11", "PASS" if not tags else "FLAG",
                                f"adjusted_status values {adj}; level features need per-contract unadjusted series" + (" -> flagged" if tags else ""))
    era_counts = Counter(era_of(d, cal) for d, _, _ in grid_rows if d.isoformat() not in excluded)
    checks["Q12"] = CheckResult("Q12", "PASS", f"single source for this table; era counts {dict(sorted(era_counts.items()))}")
    clean_days = sorted(d for d, _, _ in grid_rows if d.isoformat() not in excluded)
    underlying = str(df["underlying"].iloc[0]) if len(df) else "UNKNOWN"
    echo = (f"Received: symbol={underlying}, interval={df['interval'].iloc[0] if len(df) else 'n/a'}, "
            f"dates={clean_days[0].isoformat() if clean_days else 'n/a'}..{clean_days[-1].isoformat() if clean_days else 'n/a'}, "
            f"rows={rows_in}, ts_convention={'START' if convention == 'AMBIGUOUS' else convention}")
    checks["Q13"] = CheckResult("Q13", "PASS", echo)
    out = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    out_table = pa.Table.from_pandas(out, schema=_schema_like(table), preserve_index=False)
    report = QuarantineReport(table=table_name, underlying=underlying, checks=checks, rows_in=rows_in,
                              rows_out=rows_out, dropped=dict(dropped), masked=masked_n,
                              excluded_days=excluded, clean_days=clean_days,
                              era_counts=dict(sorted(era_counts.items())),
                              ts_convention="START" if convention == "AMBIGUOUS" else convention,
                              echo=echo, tags=tags, notes=notes)
    return out_table, report


def _schema_like(table: pa.Table) -> pa.Schema:
    fields = []
    for name in BASE_NAMES:
        fld = table.schema.field(name)
        typ = pa.string() if pa.types.is_large_string(fld.type) else fld.type
        fields.append(pa.field(name, typ))
    return pa.schema(fields)


def check_derived_table(table: pa.Table, table_name: str) -> CheckResult:
    """Q2/Q7 on a vendor derived table: required columns and ``available_from_ts >= ts_close``."""
    required = DERIVED_REQUIRED.get(table_name, ("available_from_ts",))
    missing = [c for c in required if c not in table.column_names]
    if missing:
        return CheckResult("Q2", "FAIL", f"{table_name} missing {missing} -> families N/A + DRT")
    if "ts_close" in table.column_names and "available_from_ts" in table.column_names:
        avail = _ts_series_us(table, "available_from_ts")
        close = _ts_series_us(table, "ts_close")
        bad = int(np.sum(avail < close))
        if bad:
            return CheckResult("Q7", "FAIL", f"{table_name}: {bad} rows with available_from_ts < ts_close -> quarantined + DATA_REPAIR")
    return CheckResult("Q7", "PASS", f"{table_name}: availability guard ok")


def reconcile_derived_bars(spot: pa.Table, vendor: pa.Table, minutes: int) -> dict[str, Any]:
    """Rebuild ``minutes``-bars from ``spot_1m`` and compare with the vendor ``bars_*`` copy.

    Any difference raises a ``DATA_REPAIR`` DRT (§10.1); the Lab always uses its own rebuild.
    """
    sdf = spot.to_pandas()
    sdf = sdf[~(sdf["is_stale"].fillna(False) | sdf["is_filled"].fillna(False))]
    ts_us = _ts_series_us(pa.Table.from_pandas(sdf[["ts_open"]], preserve_index=False), "ts_open")
    bucket = (ts_us // (minutes * 60_000_000)) * (minutes * 60_000_000)
    rebuilt = pd.DataFrame({"bucket": bucket, "o": sdf["o"].to_numpy(), "h": sdf["h"].to_numpy(),
                            "l": sdf["l"].to_numpy(), "c": sdf["c"].to_numpy()})
    agg = rebuilt.groupby("bucket", sort=True).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    vdf = vendor.to_pandas()
    vdf["bucket"] = _ts_series_us(vendor, "ts_open")
    merged = agg.reset_index().merge(vdf[["bucket", "o_DERIVED", "h_DERIVED", "l_DERIVED", "c_DERIVED"]], on="bucket")
    if merged.empty:
        return {"rows": 0, "max_abs_diff": None, "repair": True, "reason": "no overlapping bars"}
    diff = np.abs(merged[["o", "h", "l", "c"]].to_numpy() - merged[["o_DERIVED", "h_DERIVED", "l_DERIVED", "c_DERIVED"]].to_numpy())
    max_diff = float(np.nanmax(diff))
    return {"rows": len(merged), "max_abs_diff": max_diff, "repair": max_diff > 1e-9,
            "reason": "vendor derived bars differ from the Lab rebuild" if max_diff > 1e-9 else "identical"}


def reconcile_sources(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    """Q12 dual-source reconciliation on overlapping ``ts_open``: any OHLC diff > 0 blocks merge."""
    key = "ts_open"
    merged = a[[key, "o", "h", "l", "c"]].merge(b[[key, "o", "h", "l", "c"]], on=key, suffixes=("_a", "_b"))
    if merged.empty:
        ts_a = pd.Series(a[key]).sort_values()
        ts_b = pd.Series(b[key]).sort_values()
        shift = None
        if len(ts_a) and len(ts_b):
            shift = (ts_b.iloc[0] - ts_a.iloc[0]).total_seconds() / 60.0
        return {"overlap_rows": 0, "max_abs_diff": None, "blocked": True,
                "reason": f"no common timestamps (first-bar offset {shift} min) -> silent merge blocked"}
    diffs = np.abs(merged[["o_a", "h_a", "l_a", "c_a"]].to_numpy() - merged[["o_b", "h_b", "l_b", "c_b"]].to_numpy())
    max_diff = float(np.max(diffs))
    return {"overlap_rows": len(merged), "max_abs_diff": max_diff, "blocked": max_diff > 0.0,
            "reason": "OHLC differ on overlap -> silent merge blocked" if max_diff > 0 else "identical"}


def mad_spike_flags(c: np.ndarray, slot: np.ndarray, day_idx: np.ndarray, playground_day: np.ndarray,
                    *, k: float = 8.0, retrace: float = 0.80) -> dict[str, np.ndarray]:
    """Q9 Tier-2 flags: ``abs(ret) > k * MAD`` per slot from Playground days; flag-only.

    ``retrace_flag`` marks bars whose next bar retraces >= 80% - computed for audit only and never
    used to delete (no future information deletes a bar, R8).
    """
    ret = np.r_[np.nan, np.diff(c)]
    ret[np.r_[True, day_idx[1:] != day_idx[:-1]]] = np.nan
    use = np.asarray(playground_day, bool)[day_idx] & np.isfinite(ret)
    spike = np.zeros(c.shape[0], dtype=bool)
    for s in np.unique(slot):
        sel = use & (slot == s)
        if sel.sum() < 5:
            continue
        r = ret[sel]
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad == 0:
            q75, q25 = np.percentile(r, [75, 25])
            mad = (q75 - q25) / 1.349
        if mad <= 0:
            continue
        in_slot = (slot == s) & np.isfinite(ret)
        spike[in_slot] = np.abs(ret[in_slot] - med) > k * mad
    nxt = np.r_[ret[1:], np.nan]
    with np.errstate(divide="ignore", invalid="ignore"):
        retr = spike & np.isfinite(nxt) & (np.sign(nxt) == -np.sign(ret)) & (np.abs(nxt) >= retrace * np.abs(ret))
    return {"spike_flag": spike, "retrace_flag": retr}


# =============================================================================================
# Expiry calendar from data (R29)
# =============================================================================================
def expiry_calendar_from_data(index: StoreIndex, underlying: str, cal: dict[str, Any],
                              trading_days: Sequence[dt.date]) -> dict[str, Any]:
    """Expiry dates and lot sizes read from ``contracts.csv`` / ``expiry_date`` and cross-checked."""
    expiries: set[dt.date] = set()
    lot_sizes: dict[str, int] = {}
    sources: list[str] = []
    for path in index.side_files.get("contracts.csv", []):
        if path.parent.name != underlying:
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        sources.append(path.as_posix())
        for _, row in frame.iterrows():
            exp = row.get("expiry", "")
            if exp:
                expiries.add(dt.date.fromisoformat(exp[:10]))
                if row.get("lot_size", ""):
                    lot_sizes[exp[:10]] = int(float(row["lot_size"]))
    for table in ("fut_1m", "opt_1m"):
        if underlying in index.underlyings(table):
            reader = StoreReader(index)
            df = reader.read(table, underlying, columns=["expiry_date"])
            for val in pd.Series(df["expiry_date"]).dropna().unique():
                expiries.add(_d(val))
            sources.append(f"{table}.expiry_date")
    tdays = set(trading_days)
    disagreements = []
    for exp in sorted(expiries):
        rule = expected_expiry(exp.year, exp.month, era_of(exp, cal), cal, tdays)
        if rule is not None and rule != exp:
            disagreements.append({"expiry": exp.isoformat(), "era_rule": rule.isoformat(),
                                  "note": "data wins (R29); logged, never silently resolved"})
    return {"expiries": sorted(expiries), "lot_sizes": lot_sizes, "sources": sources,
            "disagreements": disagreements, "available": bool(expiries)}


# =============================================================================================
# DRT emitter (FR-03, VoI ranked)
# =============================================================================================
PRIORITY_WEIGHT = {"BLOCKER": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


@dataclass
class DrtBuilder:
    """Builds broker-neutral DRTs and ranks them by Value of Information."""

    tickets: list[dict[str, Any]] = field(default_factory=list)
    next_id: int = 1
    adaptive: bool = False

    def load_existing(self, existing: Sequence[dict[str, Any]]) -> None:
        """Keep open tickets (ids preserved); new ids continue after the largest id ever issued."""
        self.tickets = [dict(t) for t in existing if t["state"] not in ("ACCEPTED", "REJECTED")]
        self.next_id = 1 + max([int(t["ticket_id"][2:]) for t in existing] or [0])

    def add(self, *, trigger: str, why: str, what: str, instruments: Sequence[str],
            interval: str = "1m", from_: str = "UNKNOWN", to: str = "UNKNOWN",
            fields: Sequence[str] = (), priority: str = "MEDIUM", mde_ratio: float | str = "n/a",
            cells_unlocked: int = 0, cards_unblocked: int = 0, effort: str = "unknown",
            substitutes: Sequence[str] = (), questions: Sequence[str] = (),
            if_not: str = "", other: str = "", raised_by: str = "Quartermaster",
            families: Sequence[str] = (), card_ids: Sequence[str] = ()) -> dict[str, Any]:
        for existing in self.tickets:
            if existing["trigger"] == trigger and existing["need"]["what"] == what:
                existing["card_ids"] = sorted(set(existing.get("card_ids", [])) | set(card_ids))
                existing["expected_gain"]["cards_unblocked"] = max(existing["expected_gain"]["cards_unblocked"], cards_unblocked)
                return existing
        ticket_id = f"D-{self.next_id:04d}"
        self.next_id += 1
        ticket = {
            "schema_version": "1.0",
            "ticket_id": ticket_id,
            "raised_by": raised_by,
            "trigger": trigger,
            "why": sanitize_text(why, 1000),
            "need": {"what": what, "instruments": list(instruments) or ["UNKNOWN"], "interval": interval,
                     "from": from_, "to": to, "fields": list(fields), "session": "regular",
                     "adjusted": "unadjusted per contract" if trigger == "DATA_REPAIR" else "n/a",
                     "other": other},
            "priority": priority,
            "expected_gain": {"mde_ratio": (round(float(mde_ratio), 4) if isinstance(mde_ratio, (int, float)) and math.isfinite(float(mde_ratio)) else "n/a"),
                              "cells_unlocked": int(cells_unlocked), "cards_unblocked": int(cards_unblocked)},
            "effort": effort,
            "acceptable_substitutes": list(substitutes),
            "capability_questions": list(questions) or ["How many trading days of history can your source deliver for this field?"],
            "sample_first": True,
            "sample_spec": "2-3 trading days inside one expiry week (CP0)",
            "delivery_format": "canonical_store",
            "acceptance_checks": ["schema", "gaps", "row_conservation", "ts_convention", "reconciliation_if_overlap"],
            "if_not_delivered": if_not or "affected items stay N/A — DATA MISSING; the rest of the run continues",
            "adaptive_data": bool(self.adaptive),
            "state": "OPEN",
            "families": list(families),
            "card_ids": sorted(set(card_ids)),
        }
        self.tickets.append(ticket)
        return ticket

    def ranked(self) -> list[dict[str, Any]]:
        """VoI order: priority weight, then cards unblocked, cells unlocked, MDE ratio; stable by id."""
        def score(t: dict[str, Any]) -> float:
            gain = t["expected_gain"]
            mde = gain["mde_ratio"] if isinstance(gain["mde_ratio"], (int, float)) else 1.0
            return PRIORITY_WEIGHT[t["priority"]] * 100.0 + gain["cards_unblocked"] * 2.0 + gain["cells_unlocked"] + max(0.0, mde - 1.0)
        ordered = sorted(self.tickets, key=lambda t: (-score(t), t["ticket_id"]))
        for rank, t in enumerate(ordered, start=1):
            t["voi_rank"] = rank
            t["voi_score"] = round(score(t), 4)
        return ordered

    def write(self, path: str | Path) -> Path:
        """Validate every ticket (FILE-019) and write ``open_tickets.yaml`` atomically."""
        doc = {"schema_version": "1.0", "tickets": self.ranked()}
        validate_or_raise(doc, "ticket_file", what="tickets/open_tickets.yaml")
        return atomic_write_yaml(path, doc)


def load_tickets(path: str | Path) -> list[dict[str, Any]]:
    """Read an existing tickets file (validated); missing file -> empty list."""
    target = Path(path)
    if not target.exists():
        return []
    doc = load_yaml(target)
    validate_or_raise(doc, "ticket_file", what=str(target))
    return list(doc["tickets"])


def power_audit(usable_days: int, cfg: dict[str, Any], *, sd_outcome_atr20: float,
                k: int = 1) -> dict[str, Any]:
    """Projected Court MDE vs ``MDE_target`` -> ``mde_ratio`` for the POWER_SHORTFALL DRT."""
    from scipy.stats import norm

    court_cfg = cfg["court"]
    alpha_session = court_cfg["alpha_total"] * 6.0 / (math.pi ** 2 * k * k)
    K = int(court_cfg["K_default"])
    court_days = usable_days * float(court_cfg["court_frac_reference"])
    n_events = court_days * float(court_cfg["event_rate_reference"])
    z_sum = float(norm.isf(alpha_session / K) + norm.isf(1.0 - cfg["stats"]["power"]))
    mde_sd = z_sum / math.sqrt(n_events) if n_events > 0 else float("inf")
    mde_atr = mde_sd * sd_outcome_atr20
    target = float(cfg["targets"]["mde_target_atr20"])
    return {"usable_days": usable_days, "projected_court_days": court_days,
            "projected_event_days": n_events, "court_mde_sd": mde_sd, "court_mde_atr20": mde_atr,
            "mde_target_atr20": target, "mde_ratio": mde_atr / target if target > 0 else float("inf"),
            "shortfall": mde_atr > target, "sd_outcome_atr20": sd_outcome_atr20}


def quick_outcome_sd(df: pd.DataFrame, horizon: int = 5) -> float:
    """Independent quick estimate of the h-bar forward-return sd in ATR20 units (DRT sizing only)."""
    close = df["c"].to_numpy(dtype=float)
    hi = df["h"].to_numpy(dtype=float)
    lo = df["l"].to_numpy(dtype=float)
    if close.size < horizon + 25:
        return float("nan")
    rng_bar = np.nanmean(hi - lo)
    fwd = close[horizon:] - close[:-horizon]
    sd = float(np.nanstd(fwd, ddof=1))
    return sd / rng_bar if rng_bar > 0 else float("nan")


# =============================================================================================
# Intake orchestration (Ω-S0)
# =============================================================================================
@dataclass
class IntakeResult:
    store_dir: Path
    reports: list[QuarantineReport]
    adapter_reports: list[AdapterReport]
    passports: list[dict[str, Any]]
    tickets: list[dict[str, Any]]
    data_version: str
    clean_days: list[dt.date]
    era_counts: dict[str, int]
    reconciliation: list[dict[str, Any]]
    derived_checks: list[CheckResult]
    expiry: dict[str, Any]
    missing_families: list[str]
    echo_lines: list[str]
    raw_manifest_ok: bool
    sample_first: dict[str, Any]
    row_ledger_ok: bool
    passport_path: Path | None = None

    def summary(self) -> dict[str, Any]:
        return {"rows_in": sum(r.rows_in for r in self.reports),
                "rows_out": sum(r.rows_out for r in self.reports),
                "rows_dropped": sum(sum(r.dropped.values()) for r in self.reports),
                "row_ledger_ok": self.row_ledger_ok, "era_counts": self.era_counts,
                "clean_days": len(self.clean_days), "data_version": self.data_version}


def write_partition(table: pa.Table, store_dir: Path, table_name: str, underlying: str,
                    day: dt.date) -> Path:
    """Atomically write one canonical partition ``<store>/lab/<table>/underlying=<X>/trading_date=<D>``."""
    out = safe_path(Path("lab") / table_name / f"underlying={underlying}" / f"trading_date={day.isoformat()}" / "part-0.parquet", store_dir)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    atomic_write_bytes(out, sink.getvalue().to_pybytes())
    return out


def discover_incoming(incoming_dir: Path) -> tuple[list[Path], list[Path]]:
    """Q1: list CSV files and Parquet stores under ``incoming/`` (symlinks/traversal rejected)."""
    root = Path(incoming_dir)
    if not root.exists():
        raise InputError(f"incoming directory not found: {root}")
    if root.is_symlink():
        raise PathViolation(f"incoming directory is a symlink: {root}")
    csvs: list[Path] = []
    has_parquet = False
    for path in sorted(root.rglob("*")):
        safe_path(path, root)
        if path.is_symlink():
            raise PathViolation(f"symlink rejected in incoming/: {path}")
        if path.is_file() and path.suffix.lower() == ".csv" and path.name not in ("contracts.csv", "strategy_ready_days.csv"):
            csvs.append(path)
        elif path.is_file() and path.suffix.lower() == ".parquet":
            has_parquet = True
    return csvs, ([root] if has_parquet else [])


def run_intake(incoming_dir: str | Path, store_dir: str | Path, *, cfg: dict[str, Any],
               cal: dict[str, Any], run_dir: str | Path | None = None,
               tickets_path: str | Path | None = None, raw_dir: str | Path | None = None,
               sample_first: bool = True, declared_ts_convention: str = "START",
               log: EventLog | None = None, adaptive: bool = False) -> IntakeResult:
    """Ω-S0: Sample-First CP0, 13-point gate, canonical store write, passport, DRTs."""
    incoming = Path(incoming_dir).absolute()
    store = Path(store_dir).absolute()
    store.mkdir(parents=True, exist_ok=True)
    raw_before: dict[str, str] = {}
    if raw_dir is not None and Path(raw_dir).exists():
        protect_root(raw_dir)
        raw_before = tree_manifest(raw_dir)
    csvs, stores = discover_incoming(incoming)
    for root in stores:
        protect_root(root)
    primary = cfg["store"]["primary_underlying"]
    reports: list[QuarantineReport] = []
    adapters: list[AdapterReport] = []
    passports: list[dict[str, Any]] = []
    derived_checks: list[CheckResult] = []
    reconciliation: list[dict[str, Any]] = []
    echo_lines: list[str] = []
    accepted: dict[tuple[str, str], pa.Table] = {}
    sample_info: dict[str, Any] = {"enabled": sample_first, "days": [], "status": "SKIPPED"}

    def _gate(tbl: pa.Table, name: str) -> None:
        if sample_first and not sample_info["days"]:
            days = sorted(set(tbl.column("trading_date").to_pylist()))[: int(cfg["quarantine"]["sample_first_days"])]
            sample_tbl = tbl.filter(pc.is_in(tbl.column("trading_date"), value_set=pa.array(days, pa.date32())))
            _, sample_report = quarantine_base_table(sample_tbl, table_name=name, cal=cal,
                                                     declared_ts_convention=declared_ts_convention)
            sample_info.update({"days": [d.isoformat() for d in days], "status": "PASS",
                                "echo": sample_report.echo})
            echo_lines.append("CP0 " + sample_report.echo)
        clean, rep = quarantine_base_table(tbl, table_name=name, cal=cal,
                                           declared_ts_convention=declared_ts_convention)
        reports.append(rep)
        echo_lines.append(rep.echo)
        key = (name, rep.underlying)
        if key in accepted:
            prev = accepted[key].to_pandas()
            recon = reconcile_sources(prev, clean.to_pandas())
            recon.update({"table": name, "underlying": rep.underlying})
            reconciliation.append(recon)
            if recon["blocked"]:
                return
            merged = pa.concat_tables([accepted[key], clean]).to_pandas()
            merged = merged.drop_duplicates(subset=["instrument_key", "ts_open"], keep="first")
            accepted[key] = pa.Table.from_pandas(merged, schema=clean.schema, preserve_index=False)
        else:
            accepted[key] = clean

    for csv_path in csvs:
        name = "spot_1m"
        tbl, rep = adapt_csv(csv_path, cal, table=name, declared_ts_convention=declared_ts_convention)
        adapters.append(rep)
        passports.append({"source_id": f"SRC-{len(passports) + 1:03d}", "file": rep.source_file,
                          "sha256": rep.sha256, "rows": rep.rows_in, "symbol": rep.symbol,
                          "interval": rep.interval, "ts_convention": rep.ts_convention,
                          "adapter": "csv", "notes": rep.notes})
        _gate(tbl, name)
    for root in stores:
        index = StoreIndex.scan([root])
        for conflict in index.conflicts:
            reconciliation.append({**conflict, "blocked": True, "reason": "differing partitions for one key (Q12)"})
        reader = StoreReader(index)
        for table_name in index.tables():
            for und in index.underlyings(table_name):
                part_tbl = reader.read_arrow(table_name, und)
                if part_tbl is None:
                    continue
                if table_name in cfg["store"]["derived_tables"]:
                    derived_checks.append(check_derived_table(part_tbl, table_name))
                    rebuild_min = cfg["analysis"]["rebuild_tables"].get(table_name)
                    spot_tbl = reader.read_arrow(cfg["store"]["mandatory_table"], und)
                    if rebuild_min and spot_tbl is not None and "c_DERIVED" in part_tbl.column_names:
                        recon = reconcile_derived_bars(spot_tbl, part_tbl, int(rebuild_min))
                        derived_checks.append(CheckResult(
                            "Q12", "FAIL" if recon["repair"] else "PASS",
                            f"{table_name}/{und} vs Lab rebuild from spot_1m: {recon['reason']} "
                            f"(rows {recon['rows']}, max diff {recon['max_abs_diff']})"))
                    continue
                if table_name not in cfg["store"]["base_tables"]:
                    continue
                cast_tbl = part_tbl.select(list(BASE_NAMES)) if set(BASE_NAMES) <= set(part_tbl.column_names) else part_tbl
                passports.append({"source_id": f"SRC-{len(passports) + 1:03d}", "file": f"{root.name}:{table_name}/{und}",
                                  "sha256": sha256_json(sorted(p.sha256 for k, p in index.partitions.items() if k[0] == table_name and k[1] == und)),
                                  "rows": cast_tbl.num_rows, "symbol": und, "interval": "1m",
                                  "ts_convention": "START", "adapter": "identity (updl)", "notes": []})
                _gate(cast_tbl, table_name)
        if "contracts.csv" in index.side_files:
            for src in index.side_files["contracts.csv"]:
                dst = safe_path(Path("lab") / src.parent.name / "contracts.csv", store)
                atomic_write_bytes(dst, src.read_bytes())
    if not any(k[0] == cfg["store"]["mandatory_table"] for k in accepted):
        raise InputError(f"mandatory table {cfg['store']['mandatory_table']} is missing or unreadable (Q2)")
    for (name, und), tbl in sorted(accepted.items()):
        days = sorted(set(tbl.column("trading_date").to_pylist()))
        for day in days:
            part = tbl.filter(pc.equal(tbl.column("trading_date"), pa.scalar(day, pa.date32())))
            write_partition(part, store, name, und, day)
    index = StoreIndex.scan([store])
    spot_reports = [r for r in reports if r.table == cfg["store"]["mandatory_table"] and r.underlying == primary] \
        or [r for r in reports if r.table == cfg["store"]["mandatory_table"]]
    clean_days = sorted(set().union(*[set(r.clean_days) for r in spot_reports])) if spot_reports else []
    era_counts = Counter(era_of(d, cal) for d in clean_days)
    expiry = expiry_calendar_from_data(index, primary, cal, clean_days)
    present_tables = set(index.tables())
    missing_families = sorted(fam for fam, tbls in FAMILY_TABLES.items() if not any(t in present_tables for t in tbls))
    builder = DrtBuilder(adaptive=adaptive)
    if tickets_path is not None:
        builder.load_existing(load_tickets(tickets_path))
    date_from = clean_days[0].isoformat() if clean_days else "UNKNOWN"
    date_to = clean_days[-1].isoformat() if clean_days else "UNKNOWN"
    for fam in missing_families:
        builder.add(trigger="COVERAGE_HOLE", why=f"Data family '{fam}' is absent; dependent phases/approaches are {NA_DATA_MISSING}",
                    what=f"{fam} family tables {list(FAMILY_TABLES[fam])}", instruments=[f"{primary} {fam}"],
                    from_=date_from, to=date_to, priority="HIGH" if fam in ("futures", "options") else "MEDIUM",
                    cells_unlocked=len(FAMILY_TABLES[fam]) * 3, families=[fam],
                    questions=[f"Can your source deliver {fam} data at 1-minute resolution for {date_from}..{date_to}?"])
    if not expiry["available"]:
        builder.add(trigger="CALENDAR", why="No contracts.csv / expiry_date found: expiry, DTE and holiday items are N/A (R29, D-4)",
                    what="expiry calendar (contracts.csv with expiry, lot_size)", instruments=[f"{primary} contracts"],
                    from_=date_from, to=date_to, priority="HIGH", cells_unlocked=6, cards_unblocked=4,
                    fields=["instrument_key", "expiry", "strike", "option_type", "lot_size"], families=["calendar"],
                    questions=["Does your source provide contract master files with expiry and lot size?"])
    for chk in derived_checks:
        if chk.status == "FAIL":
            builder.add(trigger="DATA_REPAIR", why=chk.detail, what="vendor derived table repair",
                        instruments=[primary], priority="MEDIUM", cells_unlocked=2)
    for recon in reconciliation:
        if recon.get("blocked"):
            builder.add(trigger="RECONCILIATION", why=f"Q12: {recon.get('reason')}", what=f"source decision for {recon.get('table')}",
                        instruments=[str(recon.get("underlying", primary))], priority="HIGH", cells_unlocked=1)
    usable_est = max(0, len(clean_days) - int(cfg["split"]["warmup_days"]) - int(cfg["split"]["embargo_long"]))
    spot_key = next((k for k in sorted(accepted) if k[0] == cfg["store"]["mandatory_table"]), None)
    sd_est = quick_outcome_sd(accepted[spot_key].to_pandas()) if spot_key else float("nan")
    audit = power_audit(usable_est, cfg, sd_outcome_atr20=sd_est if math.isfinite(sd_est) else 1.0)
    if audit["shortfall"]:
        builder.add(trigger="POWER_SHORTFALL",
                    why=(f"Projected Court MDE {audit['court_mde_atr20']:.3f} ATR20 on {usable_est} usable days exceeds "
                         f"MDE_target {audit['mde_target_atr20']:.2f} ATR20 (ratio {audit['mde_ratio']:.2f})"),
                    what="additional history (Era-A backfill or forward recorder days)", instruments=[f"{primary} spot+futures"],
                    to=date_from, priority="HIGH", mde_ratio=audit["mde_ratio"], cells_unlocked=10,
                    questions=["How many months of 1-minute index and futures history can your source fetch?"],
                    families=["power"])
    tickets = builder.ranked()
    if tickets_path is not None:
        builder.write(tickets_path)
    raw_ok = True
    if raw_dir is not None and Path(raw_dir).exists():
        raw_ok = tree_manifest(raw_dir) == raw_before
        if not raw_ok:
            raise RowConservationError("raw/ SHA-256 manifest changed during intake (NFR-06)")
    ledger_ok = all(r.row_ledger_ok for r in reports)
    result = IntakeResult(store_dir=store, reports=reports, adapter_reports=adapters, passports=passports,
                          tickets=tickets, data_version=index.data_version(), clean_days=clean_days,
                          era_counts=dict(sorted(era_counts.items())), reconciliation=reconciliation,
                          derived_checks=derived_checks, expiry=expiry, missing_families=missing_families,
                          echo_lines=echo_lines, raw_manifest_ok=raw_ok, sample_first=sample_info,
                          row_ledger_ok=ledger_ok)
    if run_dir is not None:
        result.passport_path = write_data_passport(Path(run_dir) / "data_passport.md", result, audit)
    if log is not None:
        log.emit("quarantine_completed", {**result.summary(), "era_counts": result.era_counts}, stage="Ω-S0", role="AG-04")
    return result


def write_data_passport(path: Path, result: IntakeResult, audit: dict[str, Any]) -> Path:
    """Render ``data_passport.md`` (identity echo, Source Passports, row ledger, coverage)."""
    lines = ["# Data Passport", "", f"Generated: {iso_ist()}", f"data_version: `{result.data_version}`", "",
             "## Identity echo-back (Q13)"]
    lines += [f"- {sanitize_text(e, 300)}" for e in result.echo_lines]
    lines += ["", "## Source Passports", "| Source ID | File | SHA-256 | Rows | Symbol | Interval | ts_convention | Adapter |",
              "|---|---|---|---|---|---|---|---|"]
    for p in result.passports:
        lines.append(f"| {p['source_id']} | {sanitize_text(p['file'], 80)} | `{p['sha256'][:16]}` | {p['rows']} | "
                     f"{sanitize_text(p['symbol'], 30)} | {p['interval']} | {p['ts_convention']} | {p['adapter']} |")
    lines += ["", "## Quarantine Gate (Q1-Q13)"]
    for rep in result.reports:
        lines.append(f"### {rep.table} / {sanitize_text(rep.underlying, 30)}")
        for cid in CHECK_IDS:
            chk = rep.checks.get(cid)
            if chk is not None:
                lines.append(f"- {cid} {chk.status}: {sanitize_text(chk.detail, 300)}")
        lines.append(f"- Row ledger: rows_in={rep.rows_in} rows_out={rep.rows_out} dropped_logged={sum(rep.dropped.values())} "
                     f"({', '.join(f'{k}={v}' for k, v in sorted(rep.dropped.items()))}) -> row_ledger_ok={rep.row_ledger_ok}")
        lines.append(f"- masked (stale/filled -> NaN, never dropped): {rep.masked}")
        for day, why in sorted(rep.excluded_days.items()):
            lines.append(f"- excluded {day}: {why}")
        lines += [f"- {n}" for n in rep.notes]
    lines += ["", "## Coverage Statement",
              f"- Clean NORMAL days: {len(result.clean_days)} ({result.clean_days[0] if result.clean_days else 'n/a'} .. "
              f"{result.clean_days[-1] if result.clean_days else 'n/a'})",
              f"- Era counts (Treasury): {result.era_counts}",
              f"- Missing data families: {result.missing_families or 'none'}",
              f"- Expiry calendar from data: {'yes' if result.expiry['available'] else 'NO -> CALENDAR DRT'}; "
              f"rule disagreements logged: {len(result.expiry['disagreements'])}",
              f"- Power audit: projected Court MDE {audit['court_mde_atr20']:.3f} ATR20 vs target {audit['mde_target_atr20']} "
              f"(mde_ratio {audit['mde_ratio']:.2f})",
              f"- Sample-First CP0: {result.sample_first.get('status')} on {result.sample_first.get('days')}",
              f"- raw/ SHA-256 unchanged: {result.raw_manifest_ok}", "",
              "## USER TO CONFIRM",
              "- Round-trip costs (D-1) and MDE_target (D-2)",
              "- NSE holiday list for DTE where contract files are absent (D-4)",
              "- Exposure Calendar ranges (D-12, pre-flight item 13)", "",
              "## Reconciliation (Q12)"]
    lines += [f"- {sanitize_text(str(r), 300)}" for r in result.reconciliation] or ["- none"]
    lines += ["", "## Open Data Request Tickets (VoI ranked)"]
    lines += [f"- {t['ticket_id']} [{t['priority']}] {t['trigger']}: {sanitize_text(t['why'], 200)}" for t in result.tickets] or ["- none"]
    return atomic_write_text(path, "\n".join(lines) + "\n")


__all__ = [
    "BASE_COLS", "BASE_NAMES", "BASE_SCHEMA", "CHECK_IDS", "DERIVED_REQUIRED", "FAMILY_TABLES",
    "STORE_RE", "AdapterReport", "CheckResult", "DrtBuilder", "IntakeResult", "Partition",
    "QuarantineReport", "StoreIndex", "StoreReader", "adapt_csv", "canonical_segment",
    "check_derived_table", "discover_incoming", "era_of", "expected_expiry", "expected_grid",
    "expiry_calendar_from_data", "exposure_ranges", "is_exposed", "load_era_calendar",
    "load_tickets", "mad_spike_flags", "power_audit", "probe_ts_convention",
    "quarantine_base_table", "quick_outcome_sd", "reconcile_derived_bars", "reconcile_sources",
    "run_intake",
    "table_family", "write_data_passport", "write_partition",
]
