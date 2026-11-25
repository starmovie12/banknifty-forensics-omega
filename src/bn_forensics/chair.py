"""Chair orchestrator: Ω-S0..Ω-S9, run_state, tripwires, budgets, resume (FILE-038, FR-14, FR-15).

The Chair (AG-06) runs the deterministic pipeline and owns ``runs/<run_id>/run_state.json``.

* **Stage entry.** At every stage entry it re-hashes ``prompts/brain.md`` and the brick/target/AP
  libraries. A changed hash fires tripwire #2 and the run ends ``MODE: INVALID`` (exit 4).
* **Digests.** Each stage writes a digest ``digests/S<k>.md`` of at most 8 lines.
* **Tripwires.** It monitors the 17 V5 tripwires and the 8 Omega tripwires (F31-F38).
* **Budgets.** It enforces ``--max-wall-min``, ``--max-ideas`` and the optional token/USD caps with a
  deterministic ``PARTIAL`` checkpoint (exit 2). The checkpoint writes the ``PARTIAL`` marker plus
  ``OWNER_BRIEF.md`` and ``AI_HANDOFF.md``.
* **Never-Idle.** It keeps ``next_actions.yaml`` at 3 or more actions. It emits
  ``futility_certificate.yaml`` whenever no card is ``COURT-CONFIRMED``, and rejects a bare
  "nothing found" ending (Ω-12).
* **Isolation.** The Skeptic (FILE-034) and Step F verify (FILE-037) run as isolated subprocesses.

Exit codes follow §10.3. A run ends 0 (``COMPLETE``) or 5 (``COMPLETE_WITH_GAPS``). Otherwise it ends
2 (``PARTIAL``), 3 (input/schema error) or 4 (``MODE: INVALID``).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bn_forensics import (
    EXIT_GAPS,
    EXIT_INPUT,
    EXIT_INVALID,
    EXIT_OK,
    EXIT_PARTIAL,
    STAGE_STATES,
    STAGES,
    BnfError,
    EventLog,
    InputError,
    InvalidPipeline,
    LockContention,
    PartialCheckpoint,
    RowConservationError,
    acquire_locks,
    asset_dir,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    iso_ist,
    load_config,
    load_json,
    load_yaml,
    monotonic_minutes,
    safe_path,
    scrub_secret_env,
    sha256_file,
    sha256_json,
    validate_run_id,
)
from bn_forensics import quartermaster as qm
from bn_forensics.bricks_compiler import BrickLibrary, FeatureContext
from bn_forensics.core_stats import (
    BarFrame,
    Ctx,
    alpha_k,
    court_mde_sd,
    event_days_hash,
    frame_from_table,
    gate_adjusted_mde,
    horizon_cap,
    jaccard,
    n_star,
    outcome,
    resample,
    write_csv,
    z_upper,
)
from bn_forensics.court import freeze_docket, rank_pool, run_session, verify_docket
from bn_forensics.ledger import Ledger, series_from_days
from bn_forensics.null_lab import NullLabResult, run_null_lab
from bn_forensics.playground import PlaygroundEngine, PlaygroundResult
from bn_forensics.schemas_validator import (
    sanitize_text,
    validate,
    validate_digest,
    validate_or_raise,
)
from bn_forensics.shrinkage_basis import (
    Candidate,
    fit_families,
    orthogonal_basis,
    overlap_rows,
    series_costs,
    tradability_tag,
    write_basis_tables,
)
from bn_forensics.shrinkage_basis import deflated_lift as apply_fit
from bn_forensics.split_seal import (
    CourtSeal,
    SplitManifest,
    graduate,
    load_manifest,
    make_split,
    write_manifest,
)
from bn_forensics.verify_checks import (
    APPROACH_TEST_FIELDS,
    GRAVEYARD_FIELDS,
    REGISTRY_EVENT_FIELDS,
    REGISTRY_FIELDS,
)
from bn_forensics.xray_phases import (
    VERDICT_11A_FIELDS,
    VERDICT_11B_FIELDS,
    XrayInputs,
    XrayResult,
    compute_verdicts,
    run_xray,
)

PROMPT_FILES = ("brain.md", "generator_agent.md", "quartermaster_agent.md", "handoff_evaluator.md")
LIBRARY_FILES = ("bricks.yaml", "targets.yaml", "ap_cards.yaml")
TRIPWIRES_V5 = {
    "TW-1": "row-conservation mismatch", "TW-2": "brain_hash mismatch",
    "TW-3": "planned != executed + logged_skipped", "TW-4": "ambiguous 3-session timestamp convention",
    "TW-5": "cross-week schema drift", "TW-6": "Null-A FPR > 5%", "TW-7": "planted-edge recall > 25% off analytic MDE",
    "TW-8": "leak canary missed", "TW-9": "metamorphic test fail", "TW-10": "twin_impl parity mismatch",
    "TW-11": "Court block reuse / burn breach", "TW-12": "ADAPTIVE-DATA unlogged", "TW-13": "forbidden strategy words",
    "TW-14": "Hinglish factual line missing Finding/Card ID", "TW-15": "*_DERIVED without recipe graded A/Ω-A",
    "TW-16": "unapproved Court re-run before N_min fresh days (F37)", "TW-17": "compute/token budget exceeded",
}
TRIPWIRES_OMEGA = {
    "F31": "Court partition read during Playground", "F32": "docket tamper", "F33": "p-value peeking path",
    "F34": "clone docket (Jaccard > 0.50)", "F35": "bare 'nothing found' ending", "F36": "Lane M gradient/control missing",
    "F37": "graduated Court block re-used", "F38": "EXPOSED-COURT graded Ω-A without virgin/Ledger confirmation",
}
CRITICAL_TRIPWIRES = {"F6": "TW-1", "F8": "TW-8", "F5": "TW-10", "F6-NULL": "TW-6", "F31": "F31", "F32": "F32", "F37": "F37",
                      "TW-3": "TW-3", "F16": "F31", "F2.5": "TW-3"}
NOTHING_FOUND_PHRASES = ("nothing found", "no edge exists", "kuch nahi mila")
SELFTEST_EVENT = "AND(gt(feature3, 0.5), gt(feature7, 0.5))"
FeatureProvider = Callable[[BarFrame], dict[str, np.ndarray]]


# =============================================================================================
# Hashes (brain / prompts / library / config)
# =============================================================================================
def prompts_dir_default() -> Path:
    return asset_dir("prompts")


def brain_hash(prompts_dir: Path | None = None) -> str:
    path = Path(prompts_dir or prompts_dir_default()) / "brain.md"
    if not path.exists():
        raise InputError(f"prompts/brain.md not found at {path}")
    return sha256_file(path)


def prompts_hash(prompts_dir: Path | None = None) -> str:
    root = Path(prompts_dir or prompts_dir_default())
    return sha256_json({name: sha256_file(root / name) for name in PROMPT_FILES if (root / name).exists()})


def library_hash() -> str:
    root = asset_dir("config")
    return sha256_json({name: sha256_file(root / name) for name in LIBRARY_FILES})


# =============================================================================================
# Options / outcome
# =============================================================================================
@dataclass
class RunOptions:
    command: str = "run"
    config: str | None = None
    store_dir: Path = Path("store")
    runs_root: Path = Path("runs")
    incoming_dir: Path | None = None
    run_id: str | None = None
    seed: int = 42
    profile: str = "full"
    max_ideas: int = 0
    max_wall_min: float = 240.0
    k_docket: int | None = None
    cards_dir: Path | None = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    sample_first: bool = True
    json_logs: bool = False
    until: str = "Ω-S9"
    prompts_dir: Path | None = None
    null_worlds: int | None = None
    recall_worlds: int | None = None
    features: FeatureProvider | None = None
    ai_fallback: dict[str, list[dict[str, Any]]] | None = None
    cfg_overrides: dict[str, Any] = field(default_factory=dict)
    cal_overrides: dict[str, Any] = field(default_factory=dict)
    stage_hook: Callable[[str, Chair], None] | None = None
    clock: Callable[[float], float] = monotonic_minutes
    echo: Callable[[str], None] | None = None


@dataclass
class RunOutcome:
    run_id: str
    run_dir: Path
    exit_code: int
    state: str
    mode_line: str
    status_lines: list[str]
    tripwires: list[dict[str, Any]]
    summary: dict[str, Any] = field(default_factory=dict)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "n/a" if not math.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


# =============================================================================================
# Selftest world (T-16): synthetic Bank-Nifty-like store with a planted edge
# =============================================================================================
@dataclass
class SelftestWorld:
    incoming: Path
    days: list[dt.date]
    features: dict[tuple[str, int], tuple[float, float]]
    planted_events: int
    planted_atr20: float
    card: dict[str, Any]

    def provider(self) -> FeatureProvider:
        def build(frame: BarFrame) -> dict[str, np.ndarray]:
            pairs = np.array([self.features.get((str(d), int(m)), (0.0, 0.0)) for d, m in zip(frame.dates, frame.minute, strict=True)],
                             dtype=float).reshape(-1, 2)
            return {"feature3": pairs[:, 0], "feature7": pairs[:, 1]}
        return build


def selftest_card() -> dict[str, Any]:
    return {"schema_version": "2.0", "card_id": "C-0001", "type": "RULE", "lane": "B", "provenance": "HUNT:H1", "round": 1,
            "hypothesis": "feature3 and feature7 together precede a fixed-horizon forward-return lift",
            "event": SELFTEST_EVENT, "side": "both", "target": "T1", "horizons": [3], "baseline": "slot_matched",
            "cutoffs": "discovery_frozen", "data_needed": ["spot_1m"], "cost_series": "underlying", "prior_weight": 1.0,
            "expected_sign": "positive"}


def make_selftest_world(root: Path, cal: dict[str, Any], *, seed: int = 42, n_days: int = 450, start: dt.date = dt.date(2027, 1, 4),
                        bar_minutes: int = 3, event_day_rate: float = 0.25, effect_atr20: float = 0.8,
                        underlying: str = "BANKNIFTY", structure: bool = True) -> SelftestWorld:
    """450 synthetic trading days of 1-minute spot bars (``updl`` store layout) with a planted edge.

    The tape has day-level volatility clustering, an intraday U-shaped volatility profile, a 1-minute
    bounce reversal and first-hour opening-drive persistence. On about ``event_day_rate`` of days the
    synthetic bricks ``feature3`` and ``feature7`` are both 1 on four analysis bars, and the next
    ``3`` analysis bars drift by ``effect_atr20`` x the median ATR20. Session grids come from
    ``era_calendar.yaml``; synthetic dates are 2027+ (VIRGIN). ``structure=False`` gives a pure
    white-noise tape (T-11 futility worlds).
    """
    rng = np.random.default_rng(seed)
    days: list[dt.date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    open_min, n1 = qm.expected_grid("spot_1m", days[0], cal)
    n3 = n1 // bar_minutes
    nd = len(days)
    z = rng.normal(0.0, 1.0, (nd, n1))
    lv = np.zeros(nd)
    for i in range(1, nd):
        lv[i] = 0.9 * lv[i - 1] + rng.normal(0.0, 0.12)
    u = np.linspace(-1.0, 1.0, n1)
    sig = 26.0 * np.exp(lv)[:, None] * (1.0 + 0.35 * u ** 2)[None, :]
    if not structure:
        sig = np.full((nd, n1), 26.0)
    reversal = 0.30 if structure else 0.0
    inc = np.empty((nd, n1))
    prev = np.zeros(nd)
    for t in range(n1):
        inc[:, t] = sig[:, t] * z[:, t] - reversal * prev
        prev = inc[:, t]
    drive = rng.normal(0.0, 6.0, nd) * (1.0 if structure else 0.0)
    first_hour = 60 // max(1, bar_minutes) * bar_minutes
    inc[:, : first_hour // 2] += drive[:, None]
    inc[:, first_hour // 2: first_hour] += 0.5 * drive[:, None]
    gap = rng.normal(0.0, 0.004 * 52000.0, nd)
    ev_day = rng.random(nd) < event_day_rate
    f3 = np.zeros((nd, n3))
    f7 = np.zeros((nd, n3))
    slots = np.array([n3 // 4, n3 * 13 // 30, n3 * 3 // 5, n3 * 23 // 30])
    events: list[tuple[int, int]] = []
    for i in range(nd):
        noise = rng.random(n3)
        f3[i, noise < 0.15] = 1.0
        f7[i, (noise >= 0.15) & (noise < 0.30)] = 1.0
        if ev_day[i]:
            for s in slots + rng.integers(0, 6, slots.size):
                f3[i, s] = 1.0
                f7[i, s] = 1.0
                events.append((i, int(s)))
    wick = np.abs(rng.normal(0.0, 8.0, (nd, n1, 2)))

    def ohlc(increments: np.ndarray) -> tuple[np.ndarray, ...]:
        c = np.cumsum(increments, axis=1) + (52000.0 + np.cumsum(gap))[:, None]
        o = np.concatenate([c[:, :1] - increments[:, :1], c[:, :-1]], axis=1)
        return o, np.maximum(o, c) + wick[..., 0], np.minimum(o, c) - wick[..., 1], c

    o, h, lo, c = ohlc(inc)
    k = bar_minutes
    usable = n3 * k
    atr = float(np.median(h[:, :usable].reshape(nd, n3, k).max(2) - lo[:, :usable].reshape(nd, n3, k).min(2))) * 1.05
    drift = effect_atr20 * atr
    span = 3 * k
    for i, s in events:
        a = k * s + k
        inc[i, a: a + span] += drift / span
    o, h, lo, c = ohlc(inc)
    incoming = Path(root) / "incoming"
    base_dir = incoming / "week_01" / "store" / "holdout" / "spot_1m" / f"underlying={underlying}"
    for i, day in enumerate(days):
        mins = open_min + np.arange(n1)
        midnight_utc_us = int((dt.datetime(day.year, day.month, day.day) - dt.datetime(1970, 1, 1)).total_seconds() * 1_000_000)
        ts = midnight_utc_us - 330 * 60_000_000 + mins * 60_000_000
        cols: dict[str, Any] = {
            "instrument_id": ["Nifty Bank"] * n1, "instrument_key": ["NSE_INDEX|Nifty Bank"] * n1, "underlying": [underlying] * n1,
            "segment": ["NSE_INDEX"] * n1, "expiry_date": [None] * n1, "strike": [None] * n1, "option_type": [None] * n1,
            "interval": ["1m"] * n1, "ts_open": ts, "ts_close": ts + 60_000_000, "o": np.round(o[i], 2), "h": np.round(h[i], 2),
            "l": np.round(lo[i], 2), "c": np.round(c[i], 2), "v": np.zeros(n1, np.int64), "oi": np.zeros(n1, np.int64),
            "has_volume": np.zeros(n1, bool), "is_stale": np.zeros(n1, bool), "is_filled": np.zeros(n1, bool),
            "session_type": ["NORMAL"] * n1, "trading_date": [day] * n1, "adjusted_status": ["UNVERIFIED"] * n1,
            "source": ["selftest"] * n1, "fetch_run_id": ["selftest-0001"] * n1, "raw_sha256": [""] * n1}
        arrays = [pa.array(cols[name], pa.int64()).cast(typ) if name in ("ts_open", "ts_close") else pa.array(cols[name], typ)
                  for name, typ in qm.BASE_COLS]
        out = base_dir / f"trading_date={day.isoformat()}" / "part-0.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_arrays(arrays, schema=qm.BASE_SCHEMA), out)
    features = {(day.isoformat(), open_min + k * s): (float(f3[i, s]), float(f7[i, s]))
                for i, day in enumerate(days) for s in range(n3) if f3[i, s] or f7[i, s]}
    return SelftestWorld(incoming=incoming, days=days, features=features, planted_events=len(events), planted_atr20=atr,
                         card=selftest_card())


# =============================================================================================
# Chair
# =============================================================================================
class Chair:
    """One pipeline run (``bnf run`` / ``intake`` / ``null-lab`` / ... / ``selftest`` worlds)."""

    MUTATING = ("intake", "playground", "court", "ledger-tick", "report", "resume", "run", "selftest")

    def __init__(self, opts: RunOptions) -> None:
        self.opts = opts
        removed = scrub_secret_env()
        self.cfg = _deep_merge(load_config(opts.config), opts.cfg_overrides)
        self.cfg["budget"]["max_wall_min"] = float(opts.max_wall_min)
        self.cal = _deep_merge(qm.load_era_calendar(), opts.cal_overrides)
        self.lib = BrickLibrary.load()
        self.runs_root = Path(opts.runs_root).absolute()
        self.store = Path(opts.store_dir).absolute()
        self.run_id = self._resolve_run_id(opts.run_id)
        self.run_dir = safe_path(Path(self.run_id), self.runs_root) if self.runs_root.exists() else self.runs_root / self.run_id
        self.tables = self.run_dir / "tables"
        self.t0 = time.monotonic()
        self.status_lines: list[str] = []
        self.tripwires: list[dict[str, Any]] = []
        self.digests: list[str] = []
        self.deliverables: dict[str, str] = {}
        self.bundle: dict[str, Any] = {"run_id": self.run_id, "secrets_scrubbed": removed}
        self.state = "INIT"
        self.stage = "INIT"
        self.used_ideas = 0
        self.lock_mode = "flock"
        self.log: EventLog | None = None
        self.brain0 = ""
        self.library0 = ""

    # ---- small helpers -----------------------------------------------------------------------
    def _resolve_run_id(self, run_id: str | None) -> str:
        if run_id in (None, "", "latest"):
            if self.opts.command in ("report", "resume", "ledger-tick", "court", "playground", "null-lab") and (self.runs_root / "LATEST").exists():
                return validate_run_id((self.runs_root / "LATEST").read_text(encoding="utf-8").strip())
            if run_id == "latest" and self.opts.command in ("report", "resume", "ledger-tick"):
                raise InputError(f"no previous run under {self.runs_root} (LATEST missing)")
            stamp = iso_ist().replace(":", "").replace("-", "").replace("+0530", "")[:15]
            return validate_run_id(f"run_{stamp}_{self.opts.seed}")
        return validate_run_id(run_id)

    def say(self, line: str) -> None:
        self.status_lines.append(line)
        if self.opts.echo is not None:
            self.opts.echo(line)

    def elapsed_min(self) -> float:
        return float(self.opts.clock(self.t0))

    def trip(self, tid: str, severity: str, action: str, detail: str) -> None:
        entry = {"tripwire_id": tid, "severity": severity, "stage": self.stage, "action_taken": action, "detail": sanitize_text(detail, 300)}
        self.tripwires.append(entry)
        if self.log is not None:
            self.log.emit("tripwire_fired", {"tripwire_id": tid, "severity": severity, "stage": self.stage, "action_taken": action},
                          level="CRITICAL" if severity == "CRITICAL" else "WARN", stage=self.stage, role="AG-06")

    def digest(self, k: int, lines: Sequence[str]) -> None:
        text = "\n".join(lines[: int(self.cfg["report"]["digest_max_lines"])]) + "\n"
        problems = validate_digest(text, int(self.cfg["report"]["digest_max_lines"]))
        if problems:
            raise InputError(f"digest S{k}: {problems}")
        rel = f"digests/S{k}.md"
        atomic_write_text(self.run_dir / rel, text)
        if rel not in self.digests:
            self.digests.append(rel)

    def enter(self, stage: str) -> None:
        """Stage entry: hash re-check (tripwire #2), budget check, run_state flush."""
        self.stage = stage
        self.state = STAGE_STATES.get(stage, self.state)
        if brain_hash(self.opts.prompts_dir) != self.brain0:
            self.trip("TW-2", "CRITICAL", "MODE_INVALID", "prompts/brain.md changed mid-run")
            raise InvalidPipeline("brain_hash mismatch at stage entry (tripwire #2)", tripwire_id="TW-2")
        if library_hash() != self.library0:
            self.trip("TW-2", "CRITICAL", "MODE_INVALID", "brick/target/AP library changed mid-run")
            raise InvalidPipeline("library_hash mismatch at stage entry", tripwire_id="TW-2")
        self.check_budget(stage)
        self.write_state()
        if self.opts.stage_hook is not None:
            self.opts.stage_hook(stage, self)

    def check_budget(self, stage: str) -> None:
        b = self.cfg["budget"]
        used = self.elapsed_min()
        if used >= float(self.opts.max_wall_min):
            raise PartialCheckpoint(f"wall-clock budget {self.opts.max_wall_min} min reached ({used:.2f} min)", stage=stage,
                                    reason="max_wall_min")
        if b.get("token_cap") is not None and self.opts.tokens_used >= int(b["token_cap"]):
            raise PartialCheckpoint(f"token cap {b['token_cap']} reached ({self.opts.tokens_used})", stage=stage, reason="token_cap")
        if b.get("usd_cap") is not None and self.opts.cost_usd >= float(b["usd_cap"]):
            raise PartialCheckpoint(f"USD cap {b['usd_cap']} reached ({self.opts.cost_usd})", stage=stage, reason="usd_cap")

    def write_state(self, *, exit_code: int | None = None, partial_reason: str | None = None) -> dict[str, Any]:
        manifest: SplitManifest | None = self.bundle.get("manifest_obj")
        try:
            import pandas as pd
            versions = {"pandas": pd.__version__, "numpy": np.__version__}
        except ImportError:
            versions = {"numpy": np.__version__}
        state = {
            "schema_version": "1.0", "stage": self.stage, "prompt_version": "V5.0+Omega1.0+V3", "model_id": "host-agent",
            "brain_hash": self.brain0, "prompts_hash": prompts_hash(self.opts.prompts_dir), "library_hash": self.library0,
            "data_version": self.bundle.get("data_version", "0" * 64), "mode": self.bundle.get("run_mode", "STANDARD"),
            "court_session_k": int(manifest.session_k) if manifest else 0, "m_playground_total": int(self.used_ideas),
            "budget": {"max_wall_min": float(self.opts.max_wall_min), "max_ideas": int(self.opts.max_ideas),
                       "used_wall_min": round(self.elapsed_min(), 4), "used_ideas": int(self.used_ideas),
                       "token_cap": self.cfg["budget"].get("token_cap"), "usd_cap": self.cfg["budget"].get("usd_cap"),
                       "tokens_used": int(self.opts.tokens_used), "cost_usd": float(self.opts.cost_usd)},
            "env": {"python": platform.python_version(), "profile": self.opts.profile, "lock_mode": self.lock_mode,
                    "run_env": os.environ.get("BNF_RUN_ENV", "local_cli"), **versions},
            "open_tickets": [t["ticket_id"] for t in self.bundle.get("tickets", []) if t.get("state") not in ("ACCEPTED", "REJECTED")],
            "row_ledger_ok": bool(self.bundle.get("row_ledger_ok", True)), "tripwires": self.tripwires, "digests": self.digests,
            "run_id": self.run_id, "state": self.state, "seed": int(self.opts.seed),
            "stages_completed": list(self.bundle.get("stages_completed", [])), "deliverables": self.deliverables,
            "config_hash": sha256_json(self.cfg),
            "paths": {"store": str(self.store), "runs_root": str(self.runs_root), "incoming": str(self.opts.incoming_dir or ""),
                      "cards_dir": str(self.opts.cards_dir or "")},
        }
        if exit_code is not None:
            state["exit_code"] = int(exit_code)
        if partial_reason is not None:
            state["partial_reason"] = partial_reason
            state["resume_from"] = self.stage
        if self.bundle.get("data_gaps"):
            state["data_gaps"] = list(self.bundle["data_gaps"])
        if manifest is not None:
            state["split"] = {"usable_days": manifest.usable_days, "playground": len(manifest.playground_days),
                              "court": len(manifest.court_days), "virgin": manifest.virgin_court_days,
                              "exposed": manifest.exposed_court_days, "day_limited": manifest.day_limited}
        validate_or_raise(state, "run_state", what="run_state.json")
        atomic_write_json(self.run_dir / "run_state.json", state)
        return state

    def done(self, stage: str) -> None:
        self.bundle.setdefault("stages_completed", [])
        if stage not in self.bundle["stages_completed"]:
            self.bundle["stages_completed"].append(stage)

    def until_reached(self, stage: str) -> bool:
        return STAGES.index(stage) >= STAGES.index(self.opts.until)

    # ---- frames --------------------------------------------------------------------------------
    def _frame(self, reader: qm.StoreReader, days: Sequence[dt.date]) -> BarFrame:
        table = self.cfg["analysis"]["source_table"]
        und = self.cfg["store"]["primary_underlying"]
        df = reader.read(table, und, days)
        if df.empty:
            raise InputError(f"no {table} rows for the requested days")
        fr = frame_from_table(df)
        minutes = int(self.cfg["analysis"]["bar_minutes"])
        return resample(fr, minutes) if minutes > fr.interval_min else fr

    def _ctx(self, frame: BarFrame) -> FeatureContext:
        extra = self.opts.features(frame) if self.opts.features is not None else {}
        return FeatureContext(frame=frame, lib=self.lib, cal=self.cal, available=set(self.bundle["available"]), extra_bricks=extra)

    # ==========================================================================================
    # Pipeline
    # ==========================================================================================
    def run(self) -> RunOutcome:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = EventLog(self.run_dir / "pipeline.jsonl", run_id=self.run_id, echo_json=self.opts.json_logs)
        self.brain0 = brain_hash(self.opts.prompts_dir)
        self.library0 = library_hash()
        resources = [self.run_dir, self.store, self.runs_root / "ledger"]
        exit_code, mode_line = EXIT_OK, "MODE: FULL (code executed)"
        try:
            with acquire_locks(resources) as locks:
                self.lock_mode = "o_excl" if any(getattr(lk, "mode", "flock") != "flock" for lk in locks) else "flock"
                atomic_write_text(self.runs_root / "LATEST", self.run_id + "\n")
                if self.opts.command not in ("report", "ledger-tick"):
                    self.write_state()
                exit_code = self._pipeline()
                mode_line = self.bundle.get("mode_line", mode_line)
        except PartialCheckpoint as exc:
            exit_code = EXIT_PARTIAL
            self.state = "PARTIAL_CHECKPOINT"
            if exc.reason in ("max_wall_min", "max_ideas", "token_cap", "usd_cap"):
                self.trip("TW-17", "HIGH", "PARTIAL", str(exc))
            self.bundle["run_mode"] = "PARTIAL"
            mode_line = self._mode_line(partial=True)
            self.bundle["mode_line"] = mode_line
            atomic_write_text(self.run_dir / "PARTIAL", f"{mode_line}\nreason: {exc}\nresume: bnf resume --run-id {self.run_id}\n")
            self.write_state(exit_code=exit_code, partial_reason=str(exc))
            self._render_partial(str(exc))
            self.say(f"[{self.stage}] PARTIAL: {exc}")
            self.say(f"bnf resume --run-id {self.run_id}")
        except InvalidPipeline as exc:
            exit_code = EXIT_INVALID
            self.state = "INVALID_PIPELINE"
            tid = exc.tripwire_id or "INVALID"
            if not any(t["tripwire_id"] == tid for t in self.tripwires):
                self.trip(tid, "CRITICAL", "MODE_INVALID", str(exc))
            self.bundle["run_mode"] = "INVALID"
            mode_line = f"MODE: INVALID — tripwire {tid}: {sanitize_text(str(exc), 200)}"
            atomic_write_text(self.run_dir / "BANKNIFTY_DATA_FORENSICS_REPORT.md", mode_line + "\n\nThe run halted on a critical tripwire. "
                              "No finding from this run may be used.\n")
            self.write_state(exit_code=exit_code)
            self.say(mode_line)
        except LockContention as exc:
            exit_code = EXIT_INPUT
            self.state = "ABORT_INPUT_ERROR"
            mode_line = f"ERROR (exit 3): {sanitize_text(str(exc), 300)}"
            self.say(mode_line)
            self.log = None
        except BnfError as exc:
            exit_code = int(getattr(exc, "exit_code", EXIT_INPUT))
            if exit_code == EXIT_INVALID:
                self.bundle["run_mode"] = "INVALID"
                mode_line = f"MODE: INVALID — {sanitize_text(str(exc), 200)}"
            else:
                self.state = "ABORT_INPUT_ERROR"
                mode_line = f"ERROR (exit {exit_code}): {sanitize_text(str(exc), 300)}"
            try:
                self.write_state(exit_code=exit_code)
            except BnfError:
                atomic_write_text(self.run_dir / "ERROR", mode_line + "\n")
            self.say(mode_line)
        if self.log is not None:
            self.log.emit("run_finalized", {"exit_code": exit_code, "state": self.state, "stage": self.stage,
                                            "deliverables": len(self.deliverables)}, stage=self.stage, role="AG-06")
        return RunOutcome(run_id=self.run_id, run_dir=self.run_dir, exit_code=exit_code, state=self.state, mode_line=mode_line,
                          status_lines=self.status_lines, tripwires=self.tripwires, summary=self.bundle.get("summary", {}))

    def _pipeline(self) -> int:
        if self.opts.command == "report" or self.opts.command == "ledger-tick":
            return self._rerender(tick=self.opts.command == "ledger-tick")
        self.stage_s0()
        if not self.until_reached("Ω-S0"):
            self.stage_s1()
        if not self.until_reached("Ω-S1"):
            self.stage_s2()
        if not self.until_reached("Ω-S2"):
            self.stage_s3()
        if not self.until_reached("Ω-S3"):
            self.stage_s4_s5()
        if not self.until_reached("Ω-S5"):
            self.stage_s6()
        if not self.until_reached("Ω-S6"):
            self.stage_s7()
        if not self.until_reached("Ω-S7"):
            self.stage_s8()
        if not self.until_reached("Ω-S8"):
            return self.stage_s9()
        self.write_state(exit_code=EXIT_OK)
        return EXIT_OK

    # ---- Ω-S0 intake --------------------------------------------------------------------------
    def stage_s0(self) -> None:
        self.enter("Ω-S0")
        incoming = self.opts.incoming_dir
        if incoming is None or not Path(incoming).exists():
            raise InputError(f"incoming directory not found: {incoming}")
        try:
            res = qm.run_intake(incoming, self.store, cfg=self.cfg, cal=self.cal, run_dir=self.run_dir,
                                tickets_path=self.run_dir / "tickets" / "open_tickets.yaml", sample_first=self.opts.sample_first,
                                log=self.log)
        except RowConservationError as exc:
            self.trip("TW-1", "CRITICAL", "MODE_INVALID", str(exc))
            raise
        for line in res.echo_lines:
            self.say(line)
        s = res.summary()
        atomic_write_json(self.run_dir / "intake_summary.json", {"schema_version": "1.0", **s, "missing_families": list(res.missing_families),
                                                                 "echo": list(res.echo_lines), "sample_first": res.sample_first})
        self.bundle.update({"intake": res, "data_version": res.data_version, "row_ledger_ok": res.row_ledger_ok,
                            "tickets": list(res.tickets), "missing_families": list(res.missing_families),
                            "available": sorted({rep.table for rep in res.reports} | {"spot_1m"})})
        if not res.row_ledger_ok:
            self.trip("TW-1", "CRITICAL", "MODE_INVALID", "row ledger not conserved")
            raise InvalidPipeline("row conservation failed (F6)", tripwire_id="F6")
        self.say(f"[Ω-S0] INTAKE PASS: rows_in={s['rows_in']}, rows_out={s['rows_out']}, dropped={s['rows_dropped']}, "
                 f"clean_days={s['clean_days']}, row_ledger_ok={str(s['row_ledger_ok']).lower()}")
        self.digest(0, [f"[FACT] rows_in {s['rows_in']}, rows_out {s['rows_out']}, dropped {s['rows_dropped']} (row ledger OK).",
                        f"[FACT] clean days {s['clean_days']}; eras {s['era_counts']}.",
                        f"[FACT] missing families: {', '.join(res.missing_families) or 'none'}.",
                        f"[FACT] open tickets: {', '.join(t['ticket_id'] for t in res.tickets[:6]) or 'none'}."])
        self.done("Ω-S0")

    # ---- Ω-S1 split & seal --------------------------------------------------------------------
    def stage_s1(self) -> None:
        self.enter("Ω-S1")
        res: qm.IntakeResult = self.bundle["intake"]
        path = self.run_dir / "split_manifest.json"
        manifest = None
        if path.exists():
            prior = load_manifest(path)
            if prior.data_version == res.data_version:
                manifest = prior
        if manifest is None:
            manifest = make_split(res.clean_days, self.cfg, self.cal, data_version=res.data_version, log=self.log)
            write_manifest(path, manifest)
        manifest.verify()
        self.bundle["manifest_obj"] = manifest
        self.bundle["run_mode"] = "SCOUT" if manifest.mode == "SCOUT" else "STANDARD"
        self.bundle["mode_line"] = self._mode_line()
        self.say(f"[Ω-S1] SPLIT SEALED: usable={manifest.usable_days}, playground={len(manifest.playground_days)}, "
                 f"court={len(manifest.court_days)} (virgin {manifest.virgin_court_days}, exposed {manifest.exposed_court_days}), "
                 f"embargo={manifest.embargo}{', DAY-LIMITED' if manifest.day_limited else ''}")
        self.digest(1, [f"[FACT] usable days {manifest.usable_days}; embargo {manifest.embargo}.",
                        f"[FACT] Playground {len(manifest.playground_days)} days, Court {len(manifest.court_days)} days sealed (hash {manifest.court_days_hash[:12]}).",
                        f"[FACT] Court exposure: virgin {manifest.virgin_court_days}, exposed {manifest.exposed_court_days}.",
                        f"[FACT] DAY-LIMITED: {str(manifest.day_limited).lower()}; mode {manifest.mode}."])
        self.done("Ω-S1")

    def _mode_line(self, *, partial: bool = False) -> str:
        manifest: SplitManifest | None = self.bundle.get("manifest_obj")
        base = "MODE: SCOUT" if (manifest is not None and manifest.mode == "SCOUT") else "MODE: FULL (code executed)"
        parts = [base]
        if manifest is not None and manifest.day_limited:
            parts.append("DAY-LIMITED")
        if partial:
            parts.append("PARTIAL")
        if self.bundle.get("data_gaps"):
            parts.append("DATA-GAPS: " + ", ".join(self.bundle["data_gaps"]))
        return " · ".join(parts)

    # ---- Ω-S2 Null Lab ------------------------------------------------------------------------
    def stage_s2(self) -> None:
        self.enter("Ω-S2")
        manifest: SplitManifest = self.bundle["manifest_obj"]
        self.run_ctx = Ctx.from_config(self.cfg, seed=self.opts.seed, profile=self.opts.profile, log=self.log)
        seal = CourtSeal(manifest, audit_path=self.run_dir / "seal_audit.jsonl")
        index = qm.StoreIndex.scan([self.store])
        reader = seal.playground_reader(index)
        pg_dates = manifest.dates("warmup_days") + manifest.dates("playground_days")
        full = self._frame(reader, pg_dates)
        pg_frame = full.subset_days(np.array(manifest.playground_days, dtype="datetime64[D]"))
        self.bundle.update({"seal": seal, "index": index, "pg_frame": pg_frame})
        ctx = self._ctx(pg_frame)
        self.bundle["pg_ctx"] = ctx
        day_vals = self._day_values(pg_frame)
        nl_res = run_null_lab(n_playground_days=len(manifest.playground_days), n_court_days=len(manifest.court_days), cfg=self.cfg,
                              run_ctx=self.run_ctx, lib=self.lib, cal=self.cal, ctx=ctx, real_day_values=day_vals,
                              worlds=self.opts.null_worlds, recall_worlds=self.opts.recall_worlds, tables_dir=self.tables,
                              log=self.log, raise_on_fail=False)
        self.bundle["null_lab"] = nl_res
        atomic_write_json(self.run_dir / "null_lab.json", {"schema_version": "1.0", **self._null_dict(nl_res)})
        if nl_res.calibration_warning:
            self.trip("TW-7", "MEDIUM", "LOGGED", "planted-edge recall diverges > 25% from analytic power (CALIBRATION WARNING)")
        if nl_res.failures:
            failures = set(nl_res.failures)
            tid = "TW-8" if "Leak Canary rejection" in failures else ("TW-9" if "Metamorphic tests" in failures else
                                                                       ("TW-10" if any("twin" in f for f in failures) else "TW-6"))
            self.trip(tid, "CRITICAL", "MODE_INVALID", f"Null Lab failed: {sorted(failures)}")
            raise InvalidPipeline(f"MODE: INVALID PIPELINE - Null Lab failed: {sorted(failures)}", tripwire_id=tid)
        self.say(nl_res.status_line())
        self.digest(2, [f"[FACT] Court null FPR {nl_res.fpr_court:.3f}, Ledger {nl_res.fpr_ledger:.3f}, union {nl_res.fpr_union:.3f} (<= 0.05).",
                        f"[FACT] Peeking trap FPR {nl_res.trap_fpr:.3f}; uncapped-V5 {nl_res.uncapped_fpr:.3f} (both must exceed 0.05).",
                        f"[FACT] Leak canary {nl_res.canary_caught:.0%}; metamorphic {nl_res.meta_pass}/4; twin parity {'PASS' if nl_res.twin_pass else 'FAIL'}.",
                        "[FACT] Planted recall " + ", ".join(f"d={k}: {v:.2f}" for k, v in nl_res.recall_at_delta.items()) + ".",
                        f"[FACT] Haircut ratio {nl_res.haircut:.3f}; calibration warning {str(nl_res.calibration_warning).lower()}."])
        self.done("Ω-S2")

    def _day_values(self, frame: BarFrame) -> np.ndarray:
        out = outcome(frame, "T1", 1)["atr"]
        vals = []
        for d in range(frame.n_days):
            seg = out[frame.day_start[d]: frame.day_end[d]]
            seg = seg[~np.isnan(seg)]
            if seg.size:
                vals.append(float(seg.mean()))
        return np.asarray(vals, float)

    @staticmethod
    def _null_dict(r: NullLabResult) -> dict[str, Any]:
        return {"verdict": r.verdict, "fpr_court": r.fpr_court, "fpr_ledger": r.fpr_ledger, "fpr_union": r.fpr_union,
                "fpr_null_b": r.fpr_null_b, "recall_at_delta": r.recall_at_delta, "analytic_power": r.analytic_power,
                "calibration_warning": r.calibration_warning, "haircut": r.haircut, "canary_caught": r.canary_caught,
                "meta_pass": r.meta_pass, "twin_pass": r.twin_pass, "trap_fpr": r.trap_fpr, "uncapped_fpr": r.uncapped_fpr,
                "ledger_alpha": r.ledger_alpha, "ledger_alpha_halved": r.ledger_alpha_halved, "digital_twin": r.digital_twin,
                "rows": r.rows, "status_line": r.status_line()}

    # ---- Ω-S3 X-ray + Playground ----------------------------------------------------------------
    def costs(self) -> dict[str, float | None]:
        manifest: SplitManifest = self.bundle["manifest_obj"]
        last = manifest.dates("playground_days")[-1] if manifest.playground_days else dt.date.today()
        return series_costs(self.cfg, lot_size=None, on=last)

    def stage_s3(self) -> None:
        self.enter("Ω-S3")
        manifest: SplitManifest = self.bundle["manifest_obj"]
        pg_frame: BarFrame = self.bundle["pg_frame"]
        ctx: FeatureContext = self.bundle["pg_ctx"]
        costs = self.costs()
        self.bundle["costs"] = costs
        xin = XrayInputs(frame=pg_frame, ctx=ctx, cfg=self.cfg, run_ctx=self.run_ctx, fit_day_mask=np.ones(pg_frame.n_days, bool),
                         eras=manifest.eras, costs=costs, day_limited=manifest.day_limited, tables_dir=self.tables)
        xr = run_xray(xin, self.lib)
        self.bundle.update({"xin": xin, "xray": xr})
        haircut = self.bundle["null_lab"].haircut if self.bundle.get("null_lab") else None

        def checkpoint(info: dict[str, Any]) -> None:
            self.used_ideas = int(info.get("used_ideas", self.used_ideas))
            self.check_budget("Ω-S3")

        eng = PlaygroundEngine(ctx, self.lib, self.cfg, self.run_ctx, manifest, run_dir=self.run_dir, cards_dir=self.opts.cards_dir,
                               ai_fallback=self.opts.ai_fallback, checkpoint_callback=checkpoint, haircut=haircut, log=self.log,
                               eras=manifest.eras)
        if self.bundle["seal"].court_reads_during_playground():
            raise InvalidPipeline("Court partition read during Playground (F31, Ω-4)", tripwire_id="F31")
        pres = eng.run(max_ideas=int(self.opts.max_ideas) or None)
        self.used_ideas = pres.m_playground
        self.bundle["playground"] = pres
        self.say(f"[Ω-S3] PLAYGROUND DONE: ideas={pres.m_playground}, clusters={pres.clusters}, pool={len(pres.pool)}, "
                 f"saturated={len(pres.modes_saturated())}/13, court_reads=0")
        top = rank_pool(pres.pool)[:3]
        self.digest(3, [f"[FACT] Playground ideas {pres.m_playground} in {pres.clusters} clusters; pool {len(pres.pool)} cards.",
                        f"[FACT] Hunt Modes saturated: {', '.join(pres.modes_saturated()) or 'none'}.",
                        f"[FACT] Hunt Modes open: {', '.join(pres.modes_open()) or 'none'}.",
                        *[f"[FACT] [{c.card_id}] deflated lift {_fmt(c.deflated_lift)} ATR20 ({c.target} h={c.horizon})." for c in top],
                        "[FACT] Court partitions read during Playground: 0 (Ω-4)."])
        self.done("Ω-S3")

    # ---- Ω-S4 + Ω-S5 docket & Court -------------------------------------------------------------
    def stage_s4_s5(self) -> None:
        self.enter("Ω-S4")
        manifest: SplitManifest = self.bundle["manifest_obj"]
        pres: PlaygroundResult = self.bundle["playground"]
        docket_path = self.run_dir / "court_docket.json"
        docket = freeze_docket(pres.pool, manifest, self.cfg, K=self.opts.k_docket, run_id=self.run_id, path=docket_path, log=self.log)
        compiled = {c.card_id: c.compiled for c in pres.pool}
        self.bundle.update({"docket": docket, "compiled": compiled, "verdicts": []})
        self.digest(4, [f"[FACT] Docket session {docket['session']}: {len(docket['cards'])} of K={docket['K']} seats, Jaccard <= {docket['jaccard_max']}.",
                        f"[FACT] alpha_k {docket['alpha_k']:.6f}, threshold alpha_k/K {docket['threshold']:.7f}, B_court {docket['B_court']}.",
                        f"[FACT] Power pre-check median event-days {docket['power_precheck']['median_event_days']:.1f}, MDE {_fmt(docket['power_precheck']['projected_mde_sd'])} sd: {docket['status']}.",
                        f"[FACT] docket_hash {docket['docket_hash'][:16]}."])
        self.done("Ω-S4")
        self.enter("Ω-S5")
        if docket["status"] != "FROZEN":
            from bn_forensics.court import EVENT_LEDGER_FIELDS, VERDICT_FIELDS

            write_csv(self.tables / "court_verdicts.csv", VERDICT_FIELDS, [])
            write_csv(self.tables / "event_ledger.csv", EVENT_LEDGER_FIELDS, [])
            self.say(f"[Ω-S5] COURT POWER-DEFERRED: median event-days {docket['power_precheck']['median_event_days']:.1f}, no alpha spent")
            self.digest(5, ["[FACT] Court POWER-DEFERRED: no Court file opened, k unchanged, no alpha spent.", "[FACT] WAIT action added to next_actions.yaml."])
            self.done("Ω-S5")
            return
        seal: CourtSeal = self.bundle["seal"]
        index: qm.StoreIndex = self.bundle["index"]
        reader = seal.court_reader(index, lambda: verify_docket(docket_path, compiled, manifest))
        court_frame = self._frame(reader, manifest.dates("court_days"))
        court_ctx = self._ctx(court_frame)
        pg_frame: BarFrame = self.bundle["pg_frame"]
        from bn_forensics.core_stats import vol_terciles

        _, cutoffs = vol_terciles(pg_frame, None)
        verdicts = run_session(docket, compiled, court_ctx, manifest, self.cfg, self.run_ctx, playground_vol_cutoffs=cutoffs,
                               expiries=[], raw_frame=court_frame, tables_dir=self.tables, log=self.log)
        seal.close()
        graduate(manifest, log=self.log)
        write_manifest(self.run_dir / "split_manifest.json", manifest)
        self.bundle.update({"verdicts": verdicts, "court_frame": court_frame, "court_ctx": court_ctx})
        conf = [v for v in verdicts if v.verdict == "COURT-CONFIRMED"]
        self.say(f"[Ω-S5] COURT SESSION k={docket['session']}: confirmed={len(conf)}, "
                 f"rejected={sum(v.verdict == 'COURT-REJECTED' for v in verdicts)}, "
                 f"underpowered={sum(v.verdict == 'COURT-UNDERPOWERED' for v in verdicts)}, threshold={docket['threshold']:.7f}")
        self.digest(5, [f"[FACT] Court session {docket['session']} on {len(manifest.court_days)} sealed days; threshold {docket['threshold']:.7f}.",
                        *[f"[FACT] [{v.card_id}] {v.verdict}, p {v.p:.5f}, event-days {v.event_days}, {v.exposure}." for v in conf[:5]],
                        f"[FACT] Court block graduated to Playground (session_k {manifest.session_k})."])
        self.done("Ω-S5")

    # ---- Ω-S6 Skeptic, grades, verify -------------------------------------------------------------
    def stage_s6(self) -> None:
        self.enter("Ω-S6")
        self.tables.mkdir(parents=True, exist_ok=True)
        pres: PlaygroundResult = self.bundle["playground"]
        xr: XrayResult = self.bundle["xray"]
        costs = self.bundle["costs"]
        pg_frame: BarFrame = self.bundle["pg_frame"]
        atr_pts = float(np.nanmedian(pg_frame.atr20))
        self.bundle["atr_pts"] = atr_pts
        registry, events, ev_rows = self._registry_rows(atr_pts, costs)
        write_csv(self.run_dir / "findings_registry.csv", REGISTRY_FIELDS, registry)
        write_csv(self.tables / "registry_events.csv", REGISTRY_EVENT_FIELDS, ev_rows)
        self._skeptic_inputs()
        self._test_plan(atr_pts, costs)
        self._run_isolated("bn_forensics.skeptic_checks", ["--run-dir", str(self.run_dir)])
        skeptic = {r["card_id"]: r for r in _read_csv(self.tables / "skeptic_results.csv")}
        registry = _read_csv(self.run_dir / "findings_registry.csv")
        tests = self._grade_tests(xr, skeptic, atr_pts, costs)
        v11a, v11b = compute_verdicts(self.bundle["xin"], tests, self._vr_rows(xr))
        write_csv(self.tables / "approach_verdicts_11a.csv", VERDICT_11A_FIELDS, v11a)
        write_csv(self.tables / "approach_verdicts_11b.csv", VERDICT_11B_FIELDS, v11b)
        write_csv(self.tables / "approach_tests.csv", APPROACH_TEST_FIELDS, [self._test_row(t) for t in tests])
        self._graveyard(pres, registry)
        basis, n_indep = self._basis(registry, events)
        self.run_ctx.write_test_registry(self.tables / "test_registry.csv")
        proc = self._run_isolated("bn_forensics.verify_checks", ["--run-dir", str(self.run_dir), "--store", str(self.store),
                                                                 *(["--config", str(self.opts.config)] if self.opts.config else [])],
                                  allow=(EXIT_OK, EXIT_PARTIAL))
        vlog = load_json(self.tables / "verification_log.json")
        if self.log is not None:
            self.log.emit("verification_completed", {"sample_rows": len(vlog["checked"]), "mismatches": len(vlog["mismatches"]),
                                                     "demoted_count": len(vlog["demoted"]), "verify_pass": vlog["verify_pass"]},
                          stage="Ω-S6", role="AG-06")
        registry = _read_csv(self.run_dir / "findings_registry.csv")
        self.bundle.update({"registry": registry, "skeptic": skeptic, "tests": tests, "v11a": v11a, "v11b": v11b,
                            "n_indep": n_indep, "basis": basis, "verify": vlog})
        grades = {g: sum(1 for r in registry if r["Evidence grade"] == g) for g in ("Ω-A", "Ω-B", "Ω-C")}
        self.say(f"[Ω-S6] SKEPTIC+VERIFY: Ω-A={grades['Ω-A']}, Ω-B={grades['Ω-B']}, Ω-C={grades['Ω-C']}, N_indep={n_indep}, "
                 f"verify={'PASS' if vlog['verify_pass'] else 'PARTIAL'} ({len(vlog['demoted'])} demoted)")
        self.digest(6, [f"[FACT] Ω grades: Ω-A {grades['Ω-A']}, Ω-B {grades['Ω-B']}, Ω-C {grades['Ω-C']} (isolated Skeptic, SELF-REVIEW).",
                        f"[FACT] Independent effects N_indep {n_indep}; effects are not additive.",
                        f"[FACT] Step F verify: {len(vlog['checked'])} rows checked, {len(vlog['demoted'])} demoted, LIVE {vlog['live']['pass']}/{vlog['live']['total']}.",
                        "[FACT] Tables 11A/11B recomputed by the 2A.3 rule."])
        if proc.returncode == EXIT_PARTIAL:
            raise PartialCheckpoint("Step F verify demoted unresolved rows after 2 fix rounds", stage="Ω-S6", reason="verify")
        self.done("Ω-S6")

    def _run_isolated(self, module: str, args: list[str], *, allow: tuple[int, ...] = (EXIT_OK,)) -> subprocess.CompletedProcess[str]:
        """FILE-034 / FILE-037 run as isolated ``python3 -m`` subprocesses launched by FILE-024 (§8.1 Ω-S6)."""
        from bn_forensics.cli import run_isolated

        proc = run_isolated(module, args)
        if proc.returncode == EXIT_INVALID:
            raise InvalidPipeline(f"{module} reported MODE: INVALID: {proc.stderr.strip()[-300:]}", tripwire_id="TW-3")
        if proc.returncode not in allow:
            raise InputError(f"{module} exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
        for line in proc.stdout.splitlines():
            if line.strip():
                self.say(line.strip())
        return proc

    def _cand_by_id(self) -> dict[str, Candidate]:
        return {c.card_id: c for c in self.bundle["playground"].pool}

    def _registry_rows(self, atr_pts: float, costs: dict[str, float | None]) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
        docket = self.bundle["docket"]
        verdicts = {v.card_id: v for v in self.bundle["verdicts"]}
        pres: PlaygroundResult = self.bundle["playground"]
        cands = self._cand_by_id()
        q = self.cfg["stats"]["q_disc"]
        top3 = list(rank_pool(pres.pool)[:3])
        rows, events, ev_rows = [], {}, []
        court_frame: BarFrame | None = self.bundle.get("court_frame")
        for entry in docket["cards"]:
            cid = entry["card_id"]
            cand = cands[cid]
            ev = pres.evaluated.get(cid)
            pg_res = ev.result if ev is not None else None
            v = verdicts.get(cid)
            # Court evidence when the session ran; otherwise (POWER-DEFERRED) the Playground numbers,
            # so every registry row stays recomputable by Step F (FR-16).
            if v is not None and v.result is not None and court_frame is not None:
                res, frame, fctx, phase = v.result, court_frame, self.bundle["court_ctx"], "Ω-S5"
            else:
                res, frame, fctx, phase = pg_res, self.bundle["pg_frame"], self.bundle["pg_ctx"], "Ω-S3"
            target, h = entry["target"], int(entry["horizon"])
            rid = f"{cid}@{target}h{h}"
            dl = cand.deflated_lift
            tag = tradability_tag(target, dl, atr_pts, cand.card.get("cost_series", "underlying"), costs, self.cfg)["tag"]
            label = [v.verdict if v else "POWER-DEFERRED", v.exposure if v else ""]
            if self.bundle.get("missing_families") and str(cand.card.get("lane")) == "M":
                label.append("PARTIAL DATA")
            ev_days = [str(d) for d in res.event_days] if res is not None else sorted(cand.event_days)
            events[rid] = ev_days
            overlap = max((jaccard(cand.event_days, t.event_days) for t in top3 if t.card_id != cid), default=0.0)
            pg_se = pg_res.se if pg_res is not None else float("nan")
            row = {
                "ID": rid, "Phase": phase, "Family_m": f"{str(cand.card.get('provenance', 'HUNT')).split(':')[-1]}/m={pres.m_playground}",
                "Label": ";".join(x for x in label if x), "Statement": f"[{cid}] " + sanitize_text(cand.card.get("hypothesis", ""), 200),
                "Metric": f"{target} fixed-horizon outcome (ATR20)", "Horizon": h, "Side": cand.card.get("side", "both"),
                "Event N": res.n_events if res else 0, "Effective N": res.effective_n if res else 0,
                "Event value": res.event_value if res else float("nan"), "Baseline": res.baseline if res else float("nan"),
                "Lift (points)": res.lift_points if res else float("nan"), "Lift (ATR20)": res.lift_atr if res else float("nan"),
                "Hit-rate lift (pp)": res.hit_rate_lift_pp if res else float("nan"),
                "95% CI": f"[{_fmt(res.ci_lo, 4)}, {_fmt(res.ci_hi, 4)}]" if res else "n/a",
                "Nominal MDE": res.nominal_mde if res else (pg_res.nominal_mde if pg_res else float("nan")),
                "SE_disc": pg_se, "Gate-adjusted MDE": gate_adjusted_mde(pg_se, float(q), max(1, len(docket["cards"]))) if math.isfinite(pg_se) else float("nan"),
                "q_disc": min(1.0, pg_res.p_val * max(1, pres.m_playground)) if pg_res is not None and math.isfinite(pg_res.p_val) else float("nan"),
                "p_val": v.p if v else float("nan"), "Discovery lift": pg_res.lift_atr if pg_res else float("nan"),
                "Validation lift": res.lift_atr if (res is not None and v is not None) else float("nan"),
                "Stability": res.stability if res else "NOT RUN",
                "Outlier test": "see Skeptic re-test 4", "Skeptic verdict": "NOT RUN", "Evidence grade": "D",
                "Tradability tag": tag, "Regime": self.bundle["xray"].era_tags.get(cid, "ERA-B-ONLY" if self._era_b_only() else "ERA-CONSISTENT"),
                "Event-day hash": event_days_hash(ev_days), "Overlap with disc-top-3": round(overlap, 4),
                "Approach ID": cid.split("-")[0] if cid.startswith("AP") else "—", "Lane": cand.lane,
                "Deflated lift": dl if dl is not None else float("nan"), "Mechanism status": "NOT TESTED" if cand.lane != "M" else "MECHANISM-CONSISTENT",
                "Burn count": int(docket["session"]) if v else 0, "Independent effect ID": "",
            }
            rows.append(row)
            if res is not None:
                cc = self.bundle["compiled"][cid]
                _, direction = cc.evaluate(fctx)
                spec = {"target": target, "horizon": h, "side": cand.card.get("side", "both"), "baseline": cand.card.get("baseline", "slot_matched")}
                for i in res.event_index:
                    ev_rows.append({"ID": rid, "kind": "event", "date": str(frame.days[frame.day_idx[i]]),
                                    "minute": int(frame.minute[i]), "direction": float(direction[i]), **spec})
                for kind in ("day", "frame"):
                    ev_rows += [{"ID": rid, "kind": kind, "date": str(d), "minute": "", "direction": "", **spec} for d in frame.days]
        return rows, events, ev_rows

    def _era_b_only(self) -> bool:
        eras = set(self.bundle["manifest_obj"].eras.values())
        return bool(eras) and all(e.startswith("B") for e in eras)

    def _skeptic_inputs(self) -> None:
        cands = self._cand_by_id()
        rows = []
        for entry in self.bundle["docket"]["cards"]:
            c = cands[entry["card_id"]]
            derived = any(str(d).endswith("_DERIVED") or "_DERIVED:" in str(d) for d in c.card.get("data_needed", []))
            rows.append({"card_id": c.card_id, "era_tag": self.bundle["xray"].era_tags.get(c.card_id, "ERA-B-ONLY" if self._era_b_only() else "ERA-CONSISTENT"),
                         "unverified_causal": str(derived).lower(), "mechanism_status": "" if c.lane != "M" else "MECHANISM-CONSISTENT",
                         "ledger_state": "", "lane": c.lane})
        write_csv(self.tables / "skeptic_inputs.csv", ("card_id", "era_tag", "unverified_causal", "mechanism_status", "ledger_state", "lane"), rows)

    def _test_plan(self, atr_pts: float, costs: dict[str, float | None]) -> None:
        manifest: SplitManifest = self.bundle["manifest_obj"]
        lines = ["# Test plan (frozen before the Skeptic and Step F)", "",
                 f"- bootstrap_b: {self.run_ctx.bootstrap_B}", f"- seed: {self.opts.seed}", f"- exposure_policy: {manifest.exposure_policy}",
                 f"- day_limited: {str(manifest.day_limited).lower()}", f"- atr20_median_pts: {atr_pts!r}",
                 *[f"- cost_pts_{k}: {v!r}" for k, v in sorted(costs.items()) if v is not None],
                 f"- court_session_k: {self.bundle['docket']['session']}", f"- alpha_k: {self.bundle['docket']['alpha_k']}",
                 f"- K: {self.bundle['docket']['K']}", f"- data_version: {manifest.data_version}"]
        atomic_write_text(self.run_dir / "test_plan.md", "\n".join(lines) + "\n")

    def _grade_tests(self, xr: XrayResult, skeptic: dict[str, dict[str, str]], atr_pts: float,
                     costs: dict[str, float | None]) -> list[dict[str, Any]]:
        executed = [t for t in xr.tests if t.get("status") == "EXECUTED"]
        fits = fit_families([{"family": t["target"], "lift": t["lift_atr"], "se": t["se"]} for t in executed
                             if math.isfinite(t.get("lift_atr", np.nan)) and math.isfinite(t.get("se", np.nan))],
                            haircut=self.bundle["null_lab"].haircut if self.bundle.get("null_lab") else None)
        docket_h = {e["card_id"]: int(e["horizon"]) for e in self.bundle["docket"]["cards"]}
        out = []
        for t in xr.tests:
            row = dict(t)
            if row.get("status") == "EXECUTED":
                fit = fits.get(row["target"])
                dl = apply_fit(float(row["lift_atr"]), float(row["se"]), fit) if fit is not None and math.isfinite(row.get("se", np.nan)) else None
                row["deflated_lift"] = dl
                row["t_tag"] = tradability_tag(row["target"], dl, atr_pts, row.get("cost_series") or "underlying", costs, self.cfg)["tag"]
                sk = skeptic.get(row["card_id"])
                if sk is not None and docket_h.get(row["card_id"]) == int(row["horizon"]) and sk.get("omega_grade") in ("Ω-A", "Ω-B", "Ω-C"):
                    row["grade"] = sk["omega_grade"]
                    row["skeptic_ran"] = int(float(sk.get("ran") or 0))
            out.append(row)
        return out

    @staticmethod
    def _vr_rows(xr: XrayResult) -> list[dict[str, Any]]:
        return [r for p in xr.phases if p.phase == "2.2" for r in p.rows if str(r.get("lag", "")).startswith("VR(")]

    @staticmethod
    def _test_row(t: dict[str, Any]) -> dict[str, Any]:
        missing = t.get("missing") or []
        return {"approach_id": t.get("approach_id", ""), "test_id": t.get("test_id", ""), "status": t.get("status", ""),
                "side": t.get("side", ""), "grade": t.get("grade") or "", "t_tag": t.get("t_tag") or "",
                "q_disc": t.get("q_disc"), "se": t.get("se"), "gate_mde": t.get("gate_mde"), "effective_n": t.get("effective_n"),
                "skeptic_ran": t.get("skeptic_ran", 0) if t.get("status") == "EXECUTED" else "",
                "cost_series": t.get("cost_series") or "", "missing": ";".join(missing) if isinstance(missing, (list, tuple)) else str(missing),
                "reason": t.get("reason", "")}

    def _graveyard(self, pres: PlaygroundResult, registry: list[dict[str, str]]) -> None:
        seated = {r["ID"].split("@")[0] for r in registry}
        rows = []
        ranked = rank_pool(pres.pool)
        rank_of = {c.card_id: i + 1 for i, c in enumerate(ranked)}
        for cid, ev in sorted(pres.evaluated.items()):
            if cid in seated or ev.status not in ("EVALUATED", "DISCARDED") or ev.result is None:
                continue
            res = ev.result
            row = dict.fromkeys(GRAVEYARD_FIELDS, "")
            row.update({"ID": f"{cid}@{res.target}h{res.horizon}", "Phase": "Ω-S3", "Label": "NOISE" if res.stability in ("NOISE", "ANECDOTAL") else res.stability,
                        "Statement": f"[{cid}] " + sanitize_text(ev.card.get("hypothesis", ""), 160), "Horizon": res.horizon,
                        "Side": ev.card.get("side", "both"), "Event N": res.n_events, "Effective N": res.effective_n,
                        "Lift (ATR20)": res.lift_atr, "Nominal MDE": res.nominal_mde if math.isfinite(res.nominal_mde) else 2.8,
                        "SE_disc": res.se, "Discovery lift": res.lift_atr, "Stability": res.stability, "Evidence grade": "D",
                        "Lane": ev.card.get("lane", ""), "Approach ID": cid.split("-")[0] if cid.startswith("AP") else "—",
                        "graveyard_reason": f"not seated (pool rank {rank_of.get(cid, 'n/a')}; {ev.status})"})
            rows.append(row)
        def closeness(r: dict[str, Any]) -> float:
            lift, mde = r["Lift (ATR20)"], r["Nominal MDE"]
            ok = isinstance(lift, float) and isinstance(mde, float) and math.isfinite(lift) and math.isfinite(mde) and mde > 0
            return abs(lift) / mde if ok else 0.0

        rows.sort(key=lambda r: (-closeness(r), r["ID"]))   # closest misses first (|lift| / MDE)
        write_csv(self.tables / "graveyard.csv", GRAVEYARD_FIELDS, rows)
        self.bundle["graveyard"] = rows

    def _basis(self, registry: list[dict[str, str]], events: dict[str, list[str]]) -> tuple[list[dict[str, Any]], int]:
        findings = [{"finding_id": r["ID"], "event_days": set(events.get(r["ID"], [])), "x_by_day": {}} for r in registry]
        if not findings:
            write_basis_tables(self.tables, [], [])
            return [], 0
        basis, n_indep = orthogonal_basis(findings)
        ov = overlap_rows({r["ID"]: events.get(r["ID"], []) for r in registry})
        write_basis_tables(self.tables, basis, ov)
        eff = {b["finding_id"]: b["independent_effect_id"] for b in basis}
        rows = _read_csv(self.run_dir / "findings_registry.csv")
        for r in rows:
            r["Independent effect ID"] = eff.get(r["ID"], "")
        write_csv(self.run_dir / "findings_registry.csv", REGISTRY_FIELDS, rows)
        return basis, n_indep

    # ---- Ω-S7 Ledger ----------------------------------------------------------------------------
    def stage_s7(self) -> None:
        self.enter("Ω-S7")
        led = Ledger(self.runs_root / "ledger", self.cfg, log=self.log)
        led.backup()
        nl_res = self.bundle.get("null_lab")
        if nl_res is not None and nl_res.ledger_alpha_halved:
            led.apply_union_fpr(nl_res.fpr_union, float(self.cfg["null_lab"]["fpr_max"]))
        manifest: SplitManifest = self.bundle["manifest_obj"]
        cands = self._cand_by_id()
        pres: PlaygroundResult = self.bundle["playground"]
        verdicts = {v.card_id: v for v in self.bundle["verdicts"]}
        states = {}
        for entry in self.bundle["docket"]["cards"]:
            cid = entry["card_id"]
            x_by: dict[str, list[float]] = {}
            b_by: dict[str, list[float]] = {}
            pairs = [(pres.evaluated[cid].result if cid in pres.evaluated else None, self.bundle["pg_frame"]),
                     (verdicts[cid].result if cid in verdicts else None, self.bundle.get("court_frame"))]
            for res, fr in pairs:
                if res is None or fr is None:
                    continue
                for p, i in enumerate(res.event_index):
                    day = str(fr.days[fr.day_idx[i]])
                    x_by.setdefault(day, []).append(float(res.signed_outcome[p]))
                    b_by.setdefault(day, []).append(float(res.baseline_values[p]))
            hist = series_from_days(manifest.clean_days, {d: float(np.mean(v)) for d, v in x_by.items()},
                                    {d: float(np.mean(v)) for d, v in b_by.items()})
            states[cid] = led.register(cid, card_hash=entry["card_hash"], sign=int(entry["sign"]), history=hist,
                                       provenance=str(cands[cid].card.get("provenance", "")), frozen_at=self.bundle["docket"]["frozen_at"],
                                       data_version=manifest.data_version, round_=int(cands[cid].card.get("round", 1)),
                                       target=entry["target"], horizon=int(entry["horizon"]))
        tick = led.tick()
        led.save()
        counts = led.counts()
        next_tick = led.next_tick_date(manifest.dates("clean_days"))
        self.bundle.update({"ledger_counts": counts, "ledger_rows": led.rows(), "ledger_next_tick": next_tick, "ledger_tick": tick})
        self.say(f"[Ω-S7] LEDGER: registered={sum(1 for s in states.values() if s == 'FWD-WATCH')}, fwd_watch={counts['fwd_watch']}, "
                 f"confirmed={counts['confirmed']}, decayed={counts['decayed']}, lifetime={counts['registered_lifetime']}/{counts['M']}")
        self.digest(7, [f"[FACT] Ledger FWD-WATCH {counts['fwd_watch']}, LEDGER-CONFIRMED {counts['confirmed']}, FWD-DECAYED {counts['decayed']}.",
                        f"[FACT] lifetime registrations {counts['registered_lifetime']} of M={counts['M']}; next tick {next_tick}.",
                        "[FACT] e-values are baseline-centred, sign-gated Gaussian mixtures (c=1.10); p-value peeking is banned."])
        self.done("Ω-S7")

    # ---- Ω-S8 futility, next actions, tickets -----------------------------------------------------
    def stage_s8(self) -> None:
        self.enter("Ω-S8")
        manifest: SplitManifest = self.bundle["manifest_obj"]
        confirmed = [v for v in self.bundle["verdicts"] if v.verdict == "COURT-CONFIRMED"]
        tickets_path = self.run_dir / "tickets" / "open_tickets.yaml"
        builder = qm.DrtBuilder()
        builder.load_existing(qm.load_tickets(tickets_path))
        known = {tuple(t.get("card_ids", [])) for t in builder.tickets}
        for t in self.bundle.get("tests", []):
            miss = t.get("missing") or []
            if t.get("status") not in ("EXECUTED", "SKIPPED") and miss and (t["card_id"],) not in known:
                known.add((t["card_id"],))
                builder.add(trigger="BLOCKED_CARD", why=f"[{t['card_id']}] needs {', '.join(miss)} ({t.get('approach_id')} is N/A — DATA MISSING)",
                            what=", ".join(miss), instruments=[self.cfg["store"]["primary_underlying"]], priority="MEDIUM",
                            cards_unblocked=1, cells_unlocked=1, raised_by="Judge", families=list(miss), card_ids=[t["card_id"]])
        builder.write(tickets_path)
        tickets = builder.ranked()
        self.bundle["tickets"] = tickets
        futility = None
        if not confirmed:
            futility = self._futility(manifest, tickets)
            validate_or_raise(futility, "futility_certificate", what="futility_certificate.yaml")
            atomic_write_yaml(self.run_dir / "futility_certificate.yaml", futility)
        self.bundle["futility"] = futility
        actions = self._next_actions(tickets)
        validate_or_raise(actions, "next_actions", what="next_actions.yaml")
        if len(actions["actions"]) < 3:
            raise InvalidPipeline("Never-Idle rule: next_actions.yaml has fewer than 3 actions (Ω-6)", tripwire_id="F35")
        atomic_write_yaml(self.run_dir / "next_actions.yaml", actions)
        self.bundle["next_actions"] = actions
        blockers = [t for t in tickets if t["priority"] == "BLOCKER" and t["state"] not in ("ACCEPTED", "REJECTED")]
        gaps = list(self.bundle.get("missing_families", []))
        self.bundle["data_gaps"] = gaps if gaps or blockers else []
        self.say(f"[Ω-S8] NEVER-IDLE: next_actions={len(actions['actions'])}, open_tickets={len(tickets)}, "
                 f"futility_certificate={'WRITTEN' if futility else 'not needed'}")
        self.digest(8, [f"[FACT] COURT-CONFIRMED cards: {len(confirmed)}; futility certificate {'written' if futility else 'not required'}.",
                        f"[FACT] next_actions.yaml holds {len(actions['actions'])} typed actions (>= 3).",
                        f"[FACT] open tickets {len(tickets)}: {', '.join(t['ticket_id'] for t in tickets[:5])}.",
                        f"[FACT] data gaps: {', '.join(gaps) or 'none'}."])
        self.done("Ω-S8")

    def _futility(self, manifest: SplitManifest, tickets: list[dict[str, Any]]) -> dict[str, Any]:
        pres: PlaygroundResult | None = self.bundle.get("playground")
        nl_res: NullLabResult | None = self.bundle.get("null_lab")
        docket = self.bundle.get("docket") or {"K": int(self.cfg["court"]["K_default"]), "session": 1, "cards": []}
        K = int(docket["K"])
        k = int(docket.get("session", 1))
        a_k = alpha_k(k, float(self.cfg["court"]["alpha_total"]))
        proj = [float(c.get("projected_court_event_days", 0.0)) for c in docket.get("cards", [])]
        n_med = float(np.median(proj)) if proj else float(self.cfg["court"]["event_rate_reference"]) * len(manifest.court_days)
        sd_out = self._outcome_sd()
        mde_sd = court_mde_sd(a_k, K, max(n_med, 1.0))
        target_atr = float(self.cfg["targets"]["mde_target_atr20"])
        rate = n_med / max(1, len(manifest.court_days))
        ns = n_star(a_k, K, target_atr, sd_out if sd_out > 0 else 1.0,
                    rate if rate > 0 else float(self.cfg["court"]["event_rate_reference"]))
        counts = self.bundle.get("ledger_counts") or {"fwd_watch": 0, "confirmed": 0, "decayed": 0, "registered_lifetime": 0,
                                                      "M": int(self.cfg["ledger"]["M"])}
        caveats = ["Futility != no edge exists", "Re-measure synthetic [SIM] numbers via Null Lab at real n",
                   "Effects are not additive; N_indep counts independent effects"]
        if manifest.exposed_court_days:
            caveats.append("EXPOSED-COURT: these Court days were seen by earlier research (R28)")
        return {
            "schema_version": "1.0", "data_version": manifest.data_version,
            "usable_days": {"playground": len(manifest.playground_days), "court": len(manifest.court_days),
                            "court_virgin": manifest.virgin_court_days, "court_exposed": manifest.exposed_court_days,
                            "era_A": sum(1 for e in manifest.eras.values() if e == "A"),
                            "era_B": sum(1 for e in manifest.eras.values() if e.startswith("B"))},
            "search": {"m_playground": int(pres.m_playground) if pres else 0, "clusters": int(pres.clusters) if pres else 0,
                       "modes_saturated": pres.modes_saturated() if pres else [], "modes_open": pres.modes_open() if pres else []},
            "court_mde": {"K": K, "alpha_k": round(a_k, 6), "n_events_median": round(n_med, 3), "mde_sd": round(mde_sd, 4),
                          "mde_atr20": round(mde_sd * sd_out, 4), "outcome_sd_atr20": round(sd_out, 4)},
            "null_lab": {"fpr_court": nl_res.fpr_court if nl_res else 0.0, "fpr_ledger": nl_res.fpr_ledger if nl_res else 0.0,
                         "fpr_union": nl_res.fpr_union if nl_res else 0.0,
                         "recall_at_delta": nl_res.recall_at_delta if nl_res else {}},
            "n_star": {"mde_target_atr20": target_atr, "mde_target_sd": round(float(ns["mde_target_sd"]), 4),
                       "event_days_needed": int(ns["event_days_needed"]), "court_days_needed": int(ns["court_days_needed"])},
            "cheapest_next_data": [{"ticket": t["ticket_id"], "expected_n_star_reduction": "n/a"} for t in tickets[:3]],
            "ledger": {"fwd_watch": int(counts["fwd_watch"]), "confirmed": int(counts["confirmed"]), "decayed": int(counts["decayed"]),
                       "registered_lifetime": int(counts["registered_lifetime"]), "M": int(counts["M"]),
                       "next_tick": str(self.bundle.get("ledger_next_tick", "PENDING-DATA"))},
            "must_carry_caveats": caveats,
        }

    def _outcome_sd(self) -> float:
        fr: BarFrame | None = self.bundle.get("pg_frame")
        if fr is None:
            return 1.0
        h = max(1, min(3, horizon_cap(fr.bars_per_day)))
        vals = outcome(fr, "T1", h)["atr"]
        vals = vals[~np.isnan(vals)]
        return float(np.std(vals, ddof=1)) if vals.size > 2 else 1.0

    def _next_actions(self, tickets: list[dict[str, Any]]) -> dict[str, Any]:
        pres: PlaygroundResult | None = self.bundle.get("playground")
        docket = self.bundle.get("docket") or {}
        actions: list[dict[str, Any]] = []
        for mode in (pres.modes_open() if pres else [])[:2]:
            actions.append({"id": f"A-{len(actions) + 1:03d}", "type": "HUNT", "mode": mode, "cost": "low",
                            "expected_gain": "new idea clusters", "blocked": False})
        for t in tickets[:2]:
            actions.append({"id": f"A-{len(actions) + 1:03d}", "type": "DATA", "ticket": t["ticket_id"], "cost": "medium",
                            "expected_gain": "unblock N/A cells or cut N*", "blocked": False})
        actions.append({"id": f"A-{len(actions) + 1:03d}", "type": "WAIT",
                        "until": "fresh Court days >= N_min and power pre-check pass" if docket.get("status") == "POWER-DEFERRED" else "next 10-day Ledger tick",
                        "expected_gain": "Ledger e-value update", "blocked": False})
        actions.append({"id": f"A-{len(actions) + 1:03d}", "type": "NULL-LAB", "cost": "low",
                        "expected_gain": "re-measure FPR and recall at the new n", "blocked": False})
        return {"schema_version": "1.0", "actions": actions}

    # ---- Ω-S9 reports -----------------------------------------------------------------------------
    def stage_s9(self) -> int:
        self.enter("Ω-S9")
        from bn_forensics import reporter

        exit_code = EXIT_GAPS if self.bundle.get("data_gaps") else EXIT_OK
        self.bundle["mode_line"] = self._mode_line()
        self.state = "COMPLETE_WITH_GAPS" if exit_code == EXIT_GAPS else "COMPLETE"
        self.bundle["terminal_state"] = self.state
        paths = reporter.render_all(self)
        reject_bare_nothing_found(self.run_dir, confirmed=sum(1 for v in self.bundle["verdicts"] if v.verdict == "COURT-CONFIRMED"))
        self.deliverables = {str(p.relative_to(self.run_dir)): sha256_file(p) for p in sorted(paths) if p.exists()}
        for rel in ("findings_registry.csv", "court_docket.json", "next_actions.yaml", "tickets/open_tickets.yaml", "split_manifest.json"):
            p = self.run_dir / rel
            if p.exists():
                self.deliverables[rel] = sha256_file(p)
        for p in sorted(self.tables.glob("*.csv")):
            self.deliverables[str(p.relative_to(self.run_dir))] = sha256_file(p)
        self.done("Ω-S9")
        self.bundle["summary"] = self.summary()
        atomic_write_json(self.run_dir / "bundle_summary.json", {"schema_version": "1.0", **self.bundle["summary"]})
        self.write_state(exit_code=exit_code)
        self.say(f"[Ω-S9] REPORT WRITTEN: {len(self.deliverables)} deliverables hashed; {self.bundle['mode_line']}")
        return exit_code

    def summary(self) -> dict[str, Any]:
        reg = self.bundle.get("registry", [])
        verdicts = self.bundle.get("verdicts", [])
        return {"run_id": self.run_id, "mode_line": self.bundle.get("mode_line", ""), "state": self.state,
                "data_version": self.bundle.get("data_version", ""),
                "confirmed": [v.card_id for v in verdicts if v.verdict == "COURT-CONFIRMED"],
                "grades": {r["ID"]: r["Evidence grade"] for r in reg}, "n_indep": self.bundle.get("n_indep", 0),
                "verify_pass": bool(self.bundle.get("verify", {}).get("verify_pass", False)),
                "null_lab": self._null_dict(self.bundle["null_lab"]) if self.bundle.get("null_lab") else {},
                "docket_status": (self.bundle.get("docket") or {}).get("status", ""), "deliverables": dict(self.deliverables),
                "tripwires": [t["tripwire_id"] for t in self.tripwires]}

    def _render_partial(self, reason: str) -> None:
        from bn_forensics import reporter

        try:
            paths = reporter.render_partial(self, reason)
            self.deliverables.update({str(p.relative_to(self.run_dir)): sha256_file(p) for p in paths})
            self.write_state(exit_code=EXIT_PARTIAL, partial_reason=reason)
        except BnfError as exc:
            atomic_write_text(self.run_dir / "OWNER_BRIEF.md", f"{self.bundle.get('mode_line', 'MODE: PARTIAL')}\n[NOTE] partial render failed: {exc}\n")

    def _rerender(self, *, tick: bool) -> int:
        """``bnf report`` / ``bnf ledger-tick`` on an existing run (no Court re-run, F37)."""
        state_path = self.run_dir / "run_state.json"
        if not state_path.exists():
            raise InputError(f"no run_state.json under {self.run_dir}")
        prior = load_json(state_path)
        validate_or_raise(prior, "run_state", what=str(state_path))
        if prior["brain_hash"] != self.brain0 or prior["library_hash"] != self.library0:
            self.trip("TW-2", "CRITICAL", "MODE_INVALID", "brain/library hash differs from run_state.json")
            raise InvalidPipeline("brain_hash/library_hash mismatch vs run_state.json (tripwire #2)", tripwire_id="TW-2")
        self.stage = "Ω-S7" if tick else "Ω-S9"
        led = Ledger(self.runs_root / "ledger", self.cfg, log=self.log)
        if tick:
            led.backup()
            led.tick()
            led.save()
        self.say(f"[{self.stage}] {'LEDGER TICK' if tick else 'REPORT'}: run {self.run_id} ({prior.get('state', '')}); "
                 f"ledger fwd_watch={led.counts()['fwd_watch']}, confirmed={led.counts()['confirmed']}")
        self.bundle.update({"data_version": prior["data_version"], "stages_completed": prior.get("stages_completed", []),
                            "data_gaps": prior.get("data_gaps", []), "row_ledger_ok": prior.get("row_ledger_ok", True)})
        self.digests = list(prior.get("digests", []))
        self.deliverables = dict(prior.get("deliverables", {}))
        self.used_ideas = int(prior.get("m_playground_total", 0))
        if (self.run_dir / "split_manifest.json").exists():
            self.bundle["manifest_obj"] = load_manifest(self.run_dir / "split_manifest.json")
        exit_code = EXIT_GAPS if self.bundle["data_gaps"] else EXIT_OK
        if (self.run_dir / "findings_registry.csv").exists() and (self.tables / "verification_log.json").exists():
            from bn_forensics import reporter

            if not (self.run_dir / "next_actions.yaml").exists():
                actions = self._next_actions(qm.load_tickets(self.run_dir / "tickets" / "open_tickets.yaml"))
                validate_or_raise(actions, "next_actions", what="next_actions.yaml")
                atomic_write_yaml(self.run_dir / "next_actions.yaml", actions)
            self.state = "COMPLETE_WITH_GAPS" if exit_code == EXIT_GAPS else "COMPLETE"
            self.bundle["mode_line"] = self._mode_line()
            paths = reporter.render_all(self)
            self.deliverables.update({str(p.relative_to(self.run_dir)): sha256_file(p) for p in paths if p.exists()})
            self.done("Ω-S9")
        else:
            self.state = prior.get("state", "COMPLETE")
            exit_code = int(prior.get("exit_code", EXIT_OK)) if prior.get("exit_code") in (EXIT_OK, EXIT_GAPS) else EXIT_OK
        self.write_state(exit_code=exit_code)
        return exit_code


# =============================================================================================
# Guards and helpers
# =============================================================================================
def reject_bare_nothing_found(run_dir: Path, *, confirmed: int, texts: Sequence[str] | None = None) -> None:
    """Ω-12: a run with 0 COURT-CONFIRMED cards must carry a valid Futility Certificate.

    Any "nothing found" style ending without one is rejected (F35, exit 4).
    """
    run_dir = Path(run_dir)
    bodies = list(texts or [])
    for name in ("OWNER_BRIEF.md", "AI_HANDOFF.md", "BANKNIFTY_DATA_FORENSICS_REPORT.md"):
        p = run_dir / name
        if p.exists():
            bodies.append(p.read_text(encoding="utf-8"))
    if confirmed:
        return
    cert = run_dir / "futility_certificate.yaml"
    ok = cert.exists() and validate(load_yaml(cert), "futility_certificate").ok
    bare = any(ph in body.lower() for body in bodies for ph in NOTHING_FOUND_PHRASES)
    if not ok or (bare and not ok):
        raise InvalidPipeline("bare 'nothing found' ending without a valid futility_certificate.yaml (Ω-12)", tripwire_id="F35")


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def csv_digest(run_dir: Path) -> str:
    """SHA-256 over every CSV deliverable (selftest determinism check, T-16)."""
    root = Path(run_dir)
    return sha256_json({str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*.csv"))})


def reset_workspace(path: Path, marker: str = ".bnf_selftest_workspace") -> None:
    """Remove a previous selftest workspace, only if it carries the selftest marker file."""
    path = Path(path)
    if path.exists():
        if not (path / marker).exists():
            raise InputError(f"refusing to clear {path}: not a selftest workspace (marker missing)")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / marker).write_text("bnf selftest workspace\n", encoding="utf-8")


def summarize_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


__all__ = [
    "CRITICAL_TRIPWIRES", "LIBRARY_FILES", "NOTHING_FOUND_PHRASES", "PROMPT_FILES", "SELFTEST_EVENT", "TRIPWIRES_OMEGA", "TRIPWIRES_V5",
    "Chair", "RunOptions", "RunOutcome", "SelftestWorld", "brain_hash", "csv_digest", "library_hash", "make_selftest_world",
    "prompts_dir_default", "prompts_hash", "reject_bare_nothing_found", "reset_workspace", "selftest_card", "summarize_json", "z_upper",
]
