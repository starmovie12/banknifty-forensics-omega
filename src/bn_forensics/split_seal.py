"""Split & file-level Court seal (FILE-027, FR-04; refactor of legacy ``--seal`` / ``--include-holdout``).

* Deterministic embargo: 5 if ``disable_daily_close == NO`` and ``trading_days >= 145``;
  ``max(5, max_tested_daily_h)`` on daily files; else 1.
* ``usable_days = clean_days - warmup_days (20) - embargo``; chronological 60% Playground,
  embargo, 40% Court in the configured split mode (``D-11``: ``era_b_court`` default or ``global``).
* Every Court day is tagged ``VIRGIN`` or ``EXPOSED`` from the Exposure Calendar (R28, ``D-12``).
* ``court_days_hash`` = SHA-256 of the sorted Court date list only (never data).
* :class:`CourtSeal` is a file-level seal: every store partition read goes through its guard;
  a Playground read of a Court partition is logged and raises :class:`SealViolation` (F31, exit 4).
  Court partitions open only after the frozen docket verifies (F32 otherwise).
* Court -> Playground graduation, ``N_min`` fresh-day blocks for session ``k+1`` and Era-A /
  backfill handling as ``PRE-SAMPLE`` (the discovery split never shifts silently, F16).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bn_forensics import (
    EventLog,
    InputError,
    InvalidPipeline,
    SealViolation,
    append_line,
    atomic_write_json,
    canonical_json,
    iso_ist,
    load_json,
    sha256_text,
)
from bn_forensics.quartermaster import (
    Partition,
    StoreIndex,
    StoreReader,
    era_of,
    exposure_ranges,
    is_exposed,
    mad_spike_flags,
)

SPLIT_MODES = ("era_b_court", "global")
EXPOSURE_POLICIES = ("cap", "exclude", "ignore")


def compute_embargo(trading_days: int, cfg: dict[str, Any], *, daily_files: bool = False,
                    max_tested_daily_h: int = 0, disable_daily_close: str | None = None) -> int:
    """Deterministic embargo in days (FR-04)."""
    split = cfg["split"]
    long_e, short_e = int(split["embargo_long"]), int(split["embargo_short"])
    flag = str(disable_daily_close if disable_daily_close is not None else split["disable_daily_close"]).upper()
    if daily_files:
        return max(long_e, int(max_tested_daily_h))
    if flag == "NO" and trading_days >= int(split["daily_close_min_trading_days"]):
        return long_e
    return short_e


def court_days_hash(days: Iterable[dt.date]) -> str:
    """SHA-256 of the sorted ISO date list only (``court_days_hash``)."""
    return sha256_text("\n".join(sorted(d.isoformat() for d in days)))


@dataclass
class SplitManifest:
    """Frozen split for one ``data_version`` (persisted as ``split_manifest.json``)."""

    schema_version: str
    data_version: str
    mode: str
    exposure_policy: str
    clean_days: list[str]
    warmup_days: list[str]
    playground_days: list[str]
    embargo_days: list[str]
    court_days: list[str]
    excluded_exposed_days: list[str]
    pre_sample_days: list[str]
    usable_days: int
    embargo: int
    day_limited: bool
    court_days_hash: str
    exposure: dict[str, str]
    eras: dict[str, str]
    session_k: int = 0
    graduated: list[list[str]] = field(default_factory=list)
    frozen_at: str = ""

    # ---- helpers ------------------------------------------------------------------------
    def dates(self, key: str) -> list[dt.date]:
        return [dt.date.fromisoformat(d) for d in getattr(self, key)]

    @property
    def virgin_court_days(self) -> int:
        return sum(1 for d in self.court_days if self.exposure.get(d) == "VIRGIN")

    @property
    def exposed_court_days(self) -> int:
        return sum(1 for d in self.court_days if self.exposure.get(d) == "EXPOSED")

    def era_counts(self, keys: Sequence[str] = ("playground_days", "court_days")) -> dict[str, int]:
        counts: dict[str, int] = {}
        for key in keys:
            for d in getattr(self, key):
                era = self.eras.get(d, "UNKNOWN")
                counts[era] = counts.get(era, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def verify(self) -> None:
        """Re-derive ``court_days_hash`` and check partition disjointness (F16 / F31 guard)."""
        if court_days_hash(self.dates("court_days")) != self.court_days_hash:
            raise InvalidPipeline("court_days_hash mismatch: Court block changed after sealing", tripwire_id="F32")
        pg, cd, em = set(self.playground_days), set(self.court_days), set(self.embargo_days)
        if pg & cd or pg & em or cd & em:
            raise InvalidPipeline("split blocks overlap (Playground/embargo/Court)", tripwire_id="F16")
        if self.court_days and self.playground_days and max(self.playground_days) >= min(self.court_days):
            graduated = {d for block in self.graduated for d in block}
            late = [d for d in self.playground_days if d >= min(self.court_days) and d not in graduated]
            if late:
                raise InvalidPipeline("Playground day after Court start that is not graduated", tripwire_id="F16")


def tag_exposure(days: Iterable[dt.date], cal: dict[str, Any]) -> dict[str, str]:
    """``VIRGIN`` / ``EXPOSED`` per day from the Exposure Calendar (R28)."""
    ranges = exposure_ranges(cal)
    return {d.isoformat(): ("EXPOSED" if is_exposed(d, ranges) else "VIRGIN") for d in days}


def make_split(clean_days: Sequence[dt.date], cfg: dict[str, Any], cal: dict[str, Any], *,
               data_version: str, mode: str | None = None, exposure_policy: str | None = None,
               daily_files: bool = False, max_tested_daily_h: int = 0,
               log: EventLog | None = None) -> SplitManifest:
    """Ω-S1: warm-up, embargo, chronological 60/40 split, exposure tags, ``court_days_hash``."""
    split = cfg["split"]
    mode = mode or split["mode"]
    policy = exposure_policy or cfg["exposure"]["policy"]
    if mode not in SPLIT_MODES:
        raise InputError(f"unknown split mode {mode!r} (D-11): {SPLIT_MODES}")
    if policy not in EXPOSURE_POLICIES:
        raise InputError(f"unknown exposure policy {policy!r} (D-12): {EXPOSURE_POLICIES}")
    days = sorted(set(clean_days))
    warm_n = int(split["warmup_days"])
    if len(days) <= warm_n:
        raise InputError(f"only {len(days)} clean days; warm-up alone needs {warm_n}")
    embargo = compute_embargo(len(days), cfg, daily_files=daily_files, max_tested_daily_h=max_tested_daily_h)
    warmup = days[:warm_n]
    rest = days[warm_n:]
    eras = {d.isoformat(): era_of(d, cal) for d in days}
    frac = float(split["playground_frac"])
    if mode == "global" or not any(eras[d.isoformat()] == "A" for d in rest):
        usable = max(0, len(rest) - embargo)
        n_play = round(frac * usable)
        playground = rest[:n_play]
        embargo_days = rest[n_play:n_play + embargo]
        court = rest[n_play + embargo:]
    else:
        era_b = [d for d in rest if eras[d.isoformat()] != "A"]
        era_a = [d for d in rest if eras[d.isoformat()] == "A"]
        usable_b = max(0, len(era_b) - embargo)
        n_play_b = round(frac * usable_b)
        court = era_b[n_play_b + embargo:]
        embargo_days = era_b[n_play_b:n_play_b + embargo]
        playground = sorted(era_a + era_b[:n_play_b])
        usable = len(rest) - embargo
    exposure = tag_exposure(court, cal)
    excluded: list[dt.date] = []
    if policy == "exclude":
        excluded = [d for d in court if exposure[d.isoformat()] == "EXPOSED"]
        court = [d for d in court if exposure[d.isoformat()] == "VIRGIN"]
        exposure = {k: v for k, v in exposure.items() if v == "VIRGIN"}
    manifest = SplitManifest(
        schema_version="1.0", data_version=data_version, mode=mode, exposure_policy=policy,
        clean_days=[d.isoformat() for d in days], warmup_days=[d.isoformat() for d in warmup],
        playground_days=[d.isoformat() for d in playground],
        embargo_days=[d.isoformat() for d in embargo_days], court_days=[d.isoformat() for d in court],
        excluded_exposed_days=[d.isoformat() for d in excluded], pre_sample_days=[],
        usable_days=int(usable), embargo=int(embargo),
        day_limited=bool(usable < int(split["day_limited_floor"])),
        court_days_hash=court_days_hash(court), exposure=exposure, eras=eras, frozen_at=iso_ist())
    manifest.verify()
    if log is not None:
        log.emit("exposure_tagged", {"virgin_days": manifest.virgin_court_days,
                                     "exposed_days": manifest.exposed_court_days,
                                     "exposed_share_per_card": {}}, stage="Ω-S1", role="AG-04")
    return manifest


def write_manifest(path: str | Path, manifest: SplitManifest) -> Path:
    """Persist the split manifest atomically (deterministic JSON)."""
    return atomic_write_json(path, manifest.to_dict())


def load_manifest(path: str | Path) -> SplitManifest:
    """Load and verify a split manifest."""
    raw = load_json(path)
    try:
        manifest = SplitManifest(**raw)
    except TypeError as exc:
        raise InputError(f"malformed split manifest {path}: {exc}") from exc
    manifest.verify()
    return manifest


# =============================================================================================
# File-level Court seal (Ω-4, F31)
# =============================================================================================
class CourtSeal:
    """Guards every store partition read; Court partitions are sealed until the docket verifies."""

    def __init__(self, manifest: SplitManifest, *, audit_path: str | Path | None = None) -> None:
        self.manifest = manifest
        self.court_dates = {dt.date.fromisoformat(d) for d in manifest.court_days}
        self.playground_dates = {dt.date.fromisoformat(d) for d in manifest.playground_days}
        self.mode = "PLAYGROUND"
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.audit: list[dict[str, Any]] = []
        self.violations: list[dict[str, Any]] = []
        self.sealed_paths: set[str] = set()

    def bind_index(self, index: StoreIndex) -> None:
        """Record the exact Court partition files (file-level seal)."""
        self.sealed_paths = {p.as_posix() for key, part in index.partitions.items()
                             if key[2] in self.court_dates for p in part.paths}

    def _record(self, part: Partition, allowed: bool) -> None:
        entry = {"mode": self.mode, "table": part.table, "underlying": part.underlying,
                 "trading_date": part.trading_date.isoformat(), "allowed": allowed,
                 "court": part.trading_date in self.court_dates}
        self.audit.append(entry)
        if self.audit_path is not None and entry["court"]:
            append_line(self.audit_path, canonical_json(entry))

    def guard(self, part: Partition) -> None:
        """Partition guard for :class:`StoreReader` (raises SealViolation on F31)."""
        is_court = part.trading_date in self.court_dates or any(p.as_posix() in self.sealed_paths for p in part.paths)
        if is_court and self.mode != "COURT":
            self._record(part, False)
            self.violations.append({"table": part.table, "trading_date": part.trading_date.isoformat()})
            raise SealViolation(
                f"Court partition {part.table}/{part.underlying}/{part.trading_date} read during {self.mode} (F31, Ω-4)")
        self._record(part, True)

    def playground_reader(self, index: StoreIndex) -> StoreReader:
        """Reader that can only ever return non-Court partitions."""
        self.bind_index(index)
        return StoreReader(index, guard=self.guard)

    def open_court(self, verify_docket: Callable[[], None]) -> None:
        """Unseal after the frozen docket verifies (raises DocketTamper/InvalidPipeline otherwise)."""
        self.manifest.verify()
        verify_docket()
        self.mode = "COURT"

    def court_reader(self, index: StoreIndex, verify_docket: Callable[[], None]) -> StoreReader:
        self.bind_index(index)
        self.open_court(verify_docket)
        return StoreReader(index, guard=self.guard)

    def close(self) -> None:
        self.mode = "PLAYGROUND"

    def court_reads_during_playground(self) -> int:
        """Ω-4 metric: number of Court partition accesses attempted outside COURT mode."""
        return sum(1 for e in self.audit if e["court"] and e["mode"] != "COURT")


# =============================================================================================
# Graduation, fresh blocks and backfill (A-5, F16, F17, F37)
# =============================================================================================
def graduate(manifest: SplitManifest, *, log: EventLog | None = None) -> SplitManifest:
    """After a Court session: Court days join the Playground; ``session_k`` increments."""
    if not manifest.court_days:
        raise InputError("no Court block to graduate")
    block = list(manifest.court_days)
    if block in manifest.graduated:
        raise InvalidPipeline("Court block already graduated: re-running it is a burn breach (F37)", tripwire_id="F37")
    manifest.graduated.append(block)
    manifest.playground_days = sorted(set(manifest.playground_days) | set(manifest.embargo_days) | set(block))
    manifest.embargo_days = []
    manifest.court_days = []
    manifest.court_days_hash = court_days_hash([])
    manifest.session_k += 1
    if log is not None:
        log.emit("court_block_graduated", {"session_k": manifest.session_k, "days": len(block)}, stage="Ω-S5", role="AG-02")
    return manifest


def next_court_block(manifest: SplitManifest, new_clean_days: Sequence[dt.date], cfg: dict[str, Any],
                     cal: dict[str, Any]) -> dict[str, Any]:
    """Fresh days for session ``k+1``: after every used day, embargo applied, ``N_min >= 60``."""
    used = set(manifest.clean_days) | {d for block in manifest.graduated for d in block}
    last_used = max(used) if used else ""
    fresh = sorted(d for d in set(new_clean_days) if d.isoformat() > last_used)
    embargo = int(cfg["split"]["embargo_short"]) if len(fresh) < int(cfg["split"]["daily_close_min_trading_days"]) else int(cfg["split"]["embargo_long"])
    candidate = fresh[embargo:]
    n_min = int(cfg["split"]["n_min_fresh_days"])
    ready = len(candidate) >= n_min
    return {"fresh_days": [d.isoformat() for d in fresh], "embargo_days": [d.isoformat() for d in fresh[:embargo]],
            "court_days": [d.isoformat() for d in candidate] if ready else [],
            "ready": ready, "n_min": n_min, "available": len(candidate),
            "exposure": tag_exposure(candidate, cal) if ready else {},
            "reason": "" if ready else f"only {len(candidate)} fresh days < N_min {n_min} (WAIT)"}


def open_session_block(manifest: SplitManifest, block: dict[str, Any]) -> SplitManifest:
    """Install a ready fresh block as the sealed Court for session ``k+1``."""
    if not block.get("ready"):
        raise InputError(block.get("reason") or "fresh block not ready")
    if manifest.court_days:
        raise InvalidPipeline("current Court block not graduated yet", tripwire_id="F37")
    manifest.embargo_days = list(block["embargo_days"])
    manifest.court_days = list(block["court_days"])
    manifest.court_days_hash = court_days_hash([dt.date.fromisoformat(d) for d in block["court_days"]])
    manifest.exposure = dict(block["exposure"])
    manifest.clean_days = sorted(set(manifest.clean_days) | set(block["fresh_days"]))
    manifest.verify()
    return manifest


def apply_backfill(manifest: SplitManifest, prepended_days: Sequence[dt.date], cal: dict[str, Any], *,
                   new_data_version: str) -> SplitManifest:
    """Backfill earlier than the first warm-up day becomes ``PRE-SAMPLE`` Playground (V5-9).

    The existing Playground / embargo / Court dates are unchanged; only ``data_version`` and the
    PRE-SAMPLE list change. Days that overlap the existing range are rejected (F16).
    """
    first = min(manifest.clean_days)
    pre = sorted({d.isoformat() for d in prepended_days})
    late = [d for d in pre if d >= first]
    if late:
        raise InvalidPipeline(f"backfill overlaps the sealed range: {late[:3]} (F16)", tripwire_id="F16")
    manifest.pre_sample_days = sorted(set(manifest.pre_sample_days) | set(pre))
    manifest.playground_days = sorted(set(manifest.playground_days) | set(pre))
    manifest.clean_days = sorted(set(manifest.clean_days) | set(pre))
    manifest.eras.update({d: era_of(dt.date.fromisoformat(d), cal) for d in pre})
    manifest.data_version = new_data_version
    manifest.verify()
    return manifest


def tier2_flags(c: np.ndarray, slot: np.ndarray, day_idx: np.ndarray, frame_days: Sequence[dt.date],
                manifest: SplitManifest, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Q9 Tier-2 MAD spike flags computed from Playground days only (flag-only, never deletes)."""
    pg = {dt.date.fromisoformat(d) for d in manifest.playground_days}
    playground_day = np.array([d in pg for d in frame_days], dtype=bool)
    q = cfg["quarantine"]
    return mad_spike_flags(c, slot, day_idx, playground_day, k=float(q["mad_k"]), retrace=float(q["retrace_frac"]))


__all__ = [
    "EXPOSURE_POLICIES", "SPLIT_MODES", "CourtSeal", "SplitManifest", "apply_backfill",
    "compute_embargo", "court_days_hash", "graduate", "load_manifest", "make_split",
    "next_court_block", "open_session_block", "tag_exposure", "tier2_flags", "write_manifest",
]
