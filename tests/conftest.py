"""Shared pytest fixtures (FILE-042).

Every dataset is synthetic and generated at test time into ``tmp_path`` (seed 42). Nothing here
reads real ``updl`` data. The Arrow column contract below is the tests' *independent* statement of
PRD §10.1 ``BASE_COLS``; the production adapter (FILE-026) is checked against it.

Autouse guards:
* ``offline_guard``: any outbound AF_INET/AF_INET6 ``connect`` raises (NFR-08, TRAP-08).
* ``isolated_env``: broker tokens and ``*TOKEN*``/``*SECRET*`` variables are removed, and
  ``BNF_LOCK_DIR`` points into ``tmp_path`` so lock tests never collide (NFR-05, §8.3).
"""

from __future__ import annotations

import datetime as dt
import os
import socket
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TZ = "+05:30"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

# Independent statement of §10.1 BASE_COLS (exact names and Arrow types observed in S7).
BASE_SCHEMA = pa.schema([
    ("instrument_id", pa.string()),
    ("instrument_key", pa.string()),
    ("underlying", pa.string()),
    ("segment", pa.string()),
    ("expiry_date", pa.date32()),
    ("strike", pa.float64()),
    ("option_type", pa.string()),
    ("interval", pa.string()),
    ("ts_open", pa.timestamp("us", tz=TZ)),
    ("ts_close", pa.timestamp("us", tz=TZ)),
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
])

# S7 session facts used only to *generate* synthetic data (tests may hold literals; src may not).
INDEX_OPEN_MIN = 9 * 60 + 15
INDEX_BARS = 375
FNO_BARS_FROM_20260803 = 385
MUHURAT_BARS = 70


def weekdays(start: dt.date, n: int) -> list[dt.date]:
    """``n`` consecutive Mon-Fri dates from ``start``."""
    out: list[dt.date] = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def _epoch_us(day: dt.date, minute_of_day: int) -> int:
    moment = dt.datetime(day.year, day.month, day.day, minute_of_day // 60, minute_of_day % 60, tzinfo=IST)
    return int(moment.timestamp() * 1_000_000)


def build_bars(
    days: Iterable[dt.date],
    *,
    rng: np.random.Generator,
    bars_per_day: int | Callable[[dt.date], int] = INDEX_BARS,
    open_min: int = INDEX_OPEN_MIN,
    start_price: float = 52000.0,
    sigma: float = 0.0005,
    gap_sigma: float = 0.004,
    interval_min: int = 1,
    underlying: str = "BANKNIFTY",
    segment: str = "NSE_INDEX",
    instrument_key: str = "NSE_INDEX|Nifty Bank",
    has_volume: bool = False,
    session_type: str | Callable[[dt.date], str] = "NORMAL",
    stale_frac: float = 0.0,
    filled_frac: float = 0.0,
    expiry: dt.date | None = None,
    strike: float | None = None,
    option_type: str | None = None,
    source: str = "synthetic",
    adjusted_status: str = "UNVERIFIED",
) -> pa.Table:
    """Random-walk OHLC bars with the exact BASE_COLS Arrow schema (ts_open = candle START)."""
    cols: dict[str, list[Any]] = {name: [] for name in BASE_SCHEMA.names}
    price = start_price
    for day in days:
        n = bars_per_day(day) if callable(bars_per_day) else bars_per_day
        stype = session_type(day) if callable(session_type) else session_type
        price *= float(np.exp(rng.normal(0.0, gap_sigma)))
        rets = rng.normal(0.0, sigma, size=n)
        wick = np.abs(rng.normal(0.0, sigma * start_price * 0.4, size=(n, 2)))
        stale = rng.random(n) < stale_frac
        filled = (rng.random(n) < filled_frac) & ~stale
        for i in range(n):
            o = price
            c = o * float(np.exp(rets[i]))
            h = max(o, c) + float(wick[i, 0])
            lo = min(o, c) - float(wick[i, 1])
            if stale[i] or filled[i]:
                c, h, lo = o, o, o
            minute = open_min + i * interval_min
            ts = _epoch_us(day, minute)
            cols["instrument_id"].append(instrument_key.split("|")[-1])
            cols["instrument_key"].append(instrument_key)
            cols["underlying"].append(underlying)
            cols["segment"].append(segment)
            cols["expiry_date"].append(expiry)
            cols["strike"].append(strike)
            cols["option_type"].append(option_type)
            cols["interval"].append(f"{interval_min}m")
            cols["ts_open"].append(ts)
            cols["ts_close"].append(ts + interval_min * 60_000_000)
            cols["o"].append(round(o, 2))
            cols["h"].append(round(h, 2))
            cols["l"].append(round(lo, 2))
            cols["c"].append(round(c, 2))
            cols["v"].append(int(rng.integers(100, 5000)) if has_volume else 0)
            cols["oi"].append(int(rng.integers(10_000, 90_000)) if has_volume else 0)
            cols["has_volume"].append(has_volume)
            cols["is_stale"].append(bool(stale[i]))
            cols["is_filled"].append(bool(filled[i]))
            cols["session_type"].append(stype)
            cols["trading_date"].append(day)
            cols["adjusted_status"].append(adjusted_status)
            cols["source"].append(source)
            cols["fetch_run_id"].append("synthetic-run-0001")
            cols["raw_sha256"].append("")
            price = c
    table = pa.table(cols, schema=BASE_SCHEMA)
    return table


def derived_bars(base: pa.Table, minutes: int) -> pa.Table:
    """Vendor-style derived bars with ``o_DERIVED``... columns and ``available_from_ts``."""
    df = base.to_pandas()
    df["ts_open_us"] = base.column("ts_open").cast(pa.int64()).to_numpy()
    df["bucket"] = (df["ts_open_us"] // (minutes * 60_000_000)) * (minutes * 60_000_000)
    grouped = df.groupby(["trading_date", "bucket"], sort=True)
    agg = grouped.agg(o_DERIVED=("o", "first"), h_DERIVED=("h", "max"), l_DERIVED=("l", "min"),
                      c_DERIVED=("c", "last"), v_DERIVED=("v", "sum"), underlying=("underlying", "first"))
    agg = agg.reset_index()
    bucket = agg["bucket"].to_numpy(dtype=np.int64)
    return pa.table({
        "underlying": pa.array(agg["underlying"].astype(str).tolist(), pa.string()),
        "trading_date": pa.array(agg["trading_date"].tolist(), pa.date32()),
        "interval": pa.array([f"{minutes}m"] * len(agg), pa.string()),
        "ts_open": pa.array(bucket, pa.timestamp("us", tz=TZ)),
        "ts_close": pa.array(bucket + minutes * 60_000_000, pa.timestamp("us", tz=TZ)),
        "o_DERIVED": pa.array(agg["o_DERIVED"].to_numpy(float), pa.float64()),
        "h_DERIVED": pa.array(agg["h_DERIVED"].to_numpy(float), pa.float64()),
        "l_DERIVED": pa.array(agg["l_DERIVED"].to_numpy(float), pa.float64()),
        "c_DERIVED": pa.array(agg["c_DERIVED"].to_numpy(float), pa.float64()),
        "v_DERIVED": pa.array(agg["v_DERIVED"].to_numpy(np.int64), pa.int64()),
        "available_from_ts": pa.array(bucket + minutes * 60_000_000, pa.timestamp("us", tz=TZ)),
    })


def write_partitions(table: pa.Table, root: Path, table_name: str, *, split: str = "holdout",
                     week: str | None = "week_01") -> list[Path]:
    """Write ``[week_*/store/<split>/]<table>/underlying=<X>/trading_date=<D>/part-0.parquet``."""
    base = root / week / "store" / split if week else root
    written: list[Path] = []
    dates = table.column("trading_date").to_pylist()
    unders = table.column("underlying").to_pylist()
    keys = sorted(set(zip(unders, dates, strict=True)))
    for underlying, day in keys:
        mask = pc.and_(pc.equal(table.column("underlying"), underlying),
                       pc.equal(table.column("trading_date"), pa.scalar(day, pa.date32())))
        part = table.filter(mask)
        out = base / table_name / f"underlying={underlying}" / f"trading_date={day.isoformat()}" / "part-0.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(part, out)
        written.append(out)
    return written


@pytest.fixture(autouse=True)
def offline_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly on any outbound network connection (NFR-08)."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _blocked(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(f"outbound network connection blocked by test guard: {address!r}")
        return real_connect(self, address)

    def _blocked_ex(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(f"outbound network connection blocked by test guard: {address!r}")
        return real_connect_ex(self, address)

    def _no_create(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("socket.create_connection blocked by test guard")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_ex)
    monkeypatch.setattr(socket, "create_connection", _no_create)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Scrub secrets, pin the seed, and isolate the app-private lock directory per test."""
    for name in list(os.environ):
        upper = name.upper()
        if upper == "UPSTOX_RO_TOKEN" or "TOKEN" in upper or "SECRET" in upper:
            monkeypatch.delenv(name, raising=False)
    locks = tmp_path / "locks"
    monkeypatch.setenv("BNF_LOCK_DIR", str(locks))
    monkeypatch.setenv("BNF_SEED", "42")
    return locks


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator (seed 42, NFR-03)."""
    return np.random.default_rng(42)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def sample_csv_path() -> Path:
    """FILE-049: 3-day synthetic 1-min fixture."""
    path = REPO_ROOT / "tests" / "fixtures" / "sample_spot_3d.csv"
    assert path.is_file()
    return path


@pytest.fixture
def base_schema() -> pa.Schema:
    return BASE_SCHEMA


@pytest.fixture
def make_bars() -> Callable[..., pa.Table]:
    """Factory for BASE_COLS bar tables (see :func:`build_bars`)."""
    return build_bars


@pytest.fixture
def make_derived() -> Callable[[pa.Table, int], pa.Table]:
    return derived_bars


@pytest.fixture
def store_writer() -> Callable[..., list[Path]]:
    return write_partitions


@pytest.fixture
def make_weekdays() -> Callable[[dt.date, int], list[dt.date]]:
    return weekdays


@pytest.fixture
def white_noise_table(rng: np.random.Generator) -> pa.Table:
    """250 clean trading days of 1-min white-noise spot bars (FR-04 / T-12 shapes)."""
    days = weekdays(dt.date(2027, 1, 4), 250)
    return build_bars(days, rng=rng)


@pytest.fixture
def corrupt_ohlc_table(rng: np.random.Generator) -> pa.Table:
    """3 days with injected Tier-1 corruption: h<l, non-positive price, duplicate timestamp."""
    days = weekdays(dt.date(2027, 1, 4), 3)
    table = build_bars(days, rng=rng)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    cols["h"][10], cols["l"][10] = cols["l"][10] - 5.0, cols["h"][10] + 5.0   # h < l
    cols["o"][20] = -1.0                                                     # non-positive
    for name in cols:                                                        # duplicate row
        cols[name].insert(31, cols[name][30])
    return pa.table(cols, schema=BASE_SCHEMA)


@pytest.fixture
def multi_era_days() -> list[dt.date]:
    """Weekdays spanning Era A, B1, B2 and B3 boundaries (2024-10 .. 2025-10)."""
    return weekdays(dt.date(2024, 10, 1), 280)


@pytest.fixture
def stacked_option_table(rng: np.random.Generator) -> pa.Table:
    """CE/PE stacked option bars on 3 strikes with 25% stale bars (S7 quality facts)."""
    days = weekdays(dt.date(2026, 11, 9), 3)
    tables = []
    for k, strike in enumerate((51800.0, 52000.0, 52200.0)):
        for otype in ("CE", "PE"):
            tables.append(build_bars(
                days, rng=np.random.default_rng(100 + k * 2 + (otype == "PE")),
                bars_per_day=FNO_BARS_FROM_20260803, start_price=400.0, sigma=0.01,
                segment="NSE_FO", instrument_key=f"NSE_FO|{int(strike)}{otype}",
                has_volume=True, stale_frac=0.25, expiry=dt.date(2026, 11, 23), strike=strike,
                option_type=otype))
    return pa.concat_tables(tables)


@pytest.fixture
def updl_store(tmp_path: Path, rng: np.random.Generator) -> Path:
    """An ``updl``-schema store with exact S7 types (T-02).

    * ``spot_1m``: NSE_INDEX, 375 bars/day, plus a 70-bar MUHURAT Sunday (2026-11-08);
    * ``fut_1m``: NSE_FO, 385 bars/day (post 2026-08-03 grid), lot size in ``contracts.csv``;
    * ``bars_3m``: vendor derived bars with ``o_DERIVED`` columns and ``available_from_ts``;
    * ``opt_1m``: CE/PE with 25% stale bars;
    * side files: ``BANKNIFTY/contracts.csv`` and ``strategy_ready_days.csv``.
    """
    root = tmp_path / "updl"
    days = weekdays(dt.date(2026, 11, 9), 5)
    muhurat = dt.date(2026, 11, 8)
    all_days = [muhurat, *days]

    def spot_bars(day: dt.date) -> int:
        return MUHURAT_BARS if day == muhurat else INDEX_BARS

    def spot_session(day: dt.date) -> str:
        return "MUHURAT" if day == muhurat else "NORMAL"

    spot = build_bars(all_days, rng=rng, bars_per_day=spot_bars, session_type=spot_session)
    fut = build_bars(days, rng=np.random.default_rng(7), bars_per_day=FNO_BARS_FROM_20260803,
                     segment="NSE_FO", instrument_key="NSE_FO|62147", has_volume=True,
                     expiry=dt.date(2026, 11, 23), start_price=52100.0)
    opt = build_bars(days, rng=np.random.default_rng(9), bars_per_day=FNO_BARS_FROM_20260803,
                     segment="NSE_FO", instrument_key="NSE_FO|52000CE", has_volume=True,
                     stale_frac=0.25, expiry=dt.date(2026, 11, 23), strike=52000.0,
                     option_type="CE", start_price=400.0, sigma=0.01)
    normal_spot = spot.filter(pc.equal(spot.column("session_type"), "NORMAL"))
    write_partitions(spot, root, "spot_1m")
    write_partitions(fut, root, "fut_1m")
    write_partitions(opt, root, "opt_1m")
    write_partitions(derived_bars(normal_spot, 3), root, "bars_3m")
    index_dir = root / "week_01" / "BANKNIFTY"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "contracts.csv").write_text(
        "instrument_key,expiry,strike,option_type,lot_size,first_day,last_day,minutes_with_data\n"
        "NSE_FO|62147,2026-11-23,,FUT,30,2026-10-01,2026-11-23,9625\n"
        "NSE_FO|52000CE,2026-11-23,52000,CE,30,2026-10-01,2026-11-23,7210\n",
        encoding="utf-8")
    (root / "week_01" / "strategy_ready_days.csv").write_text(
        "trading_date,status\n" + "".join(f"{d.isoformat()},PRICE_READY\n" for d in days),
        encoding="utf-8")
    return root


@pytest.fixture
def day_series() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Factory: day-level outcome series ``x_d`` (ATR20) with an optional planted effect.

    Returns ``(x, is_event)`` for ``n_days``; events occur on a ``rate`` share of days and carry
    ``effect`` (in units of the noise sd) added to their outcome.
    """

    def _make(n_days: int, *, rate: float = 0.25, effect: float = 0.0, seed: int = 42,
              sd: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        gen = np.random.default_rng(seed)
        is_event = gen.random(n_days) < rate
        x = gen.normal(0.0, sd, size=n_days)
        x[is_event] += effect * sd
        return x, is_event

    return _make
