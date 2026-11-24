"""Unlimited Playground — Never-Stop engine (FILE-030, FR-09; §9.5, PR-09; Ω-S3).

* Runs on **Playground days only**: it receives an already-built :class:`FeatureContext` whose
  frame must contain no sealed Court day (checked against the split manifest, F31). This module
  has no store reader and never imports ``chair``; budget checkpoints arrive through the injected
  ``checkpoint_callback`` (RT-002).
* 13 Hunt Modes: deterministic code enumerators ``H2 H3 H4 H5 H9 H10 H11 H12`` (zero AI tokens) and
  host-agent modes ``H1 H6 H7 H8 H13`` reading Idea Cards v2 from ``--cards-dir`` (or the
  deterministic selftest fallback). Zero idea or round cap; zero claims.
* Every generated, evaluated, discarded or rejected idea is hashed into
  ``runs/<run_id>/idea_ledger.jsonl`` (``m_lifetime``) and feeds the Empirical-Bayes pool.
* Saturation per mode on distinct clusters (event-day Jaccard < 0.8): 3 consecutive batches with
  ``>= 300`` (deterministic) / ``>= 50`` (AI) novel ideas adding 0 new top-50 shrunk-score cards;
  a fully enumerated finite grid is recorded as saturated by exhaustion.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bn_forensics import (
    HUNT_MODES,
    EventLog,
    InputError,
    PartialCheckpoint,
    SealViolation,
    append_line,
    atomic_write_json,
    canonical_json,
    iso_ist,
    load_json,
    load_yaml,
    sha256_json,
    truncate_incomplete_jsonl,
)
from bn_forensics.bricks_compiler import (
    BrickLibrary,
    CompiledCard,
    FeatureContext,
    Node,
    compile_card,
    load_ap_cards,
    parse_event,
)
from bn_forensics.core_stats import (
    Ctx,
    EventResult,
    TestRunner,
    event_days_hash,
    horizon_cap,
    jaccard,
    outcome,
)
from bn_forensics.schemas_validator import validate, validate_or_raise
from bn_forensics.shrinkage_basis import Candidate, ShrinkageFit, deflated_lift, fit_families
from bn_forensics.split_seal import SplitManifest

CheckpointCallback = Callable[[dict[str, Any]], None]


# =============================================================================================
# Idea Ledger (runs/<run_id>/idea_ledger.jsonl)
# =============================================================================================
class IdeaLedger:
    """Append-only JSONL Idea Ledger; incomplete trailing lines are truncated on resume (NFR-04)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        truncate_incomplete_jsonl(self.path)
        self.records: list[dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = load_json_line(line)
                    validate_or_raise(rec, "idea_ledger_record", what="idea_ledger.jsonl")
                    self.records.append(rec)
        self.m_lifetime = max((r["m_lifetime"] for r in self.records), default=0)
        self.seen = {r["card_id"] for r in self.records}

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.m_lifetime += 1
        rec = {"schema_version": "1.0", "m_lifetime": self.m_lifetime, **record}
        res = validate(rec, "idea_ledger_record")
        if not res.ok:
            raise InputError("idea ledger record invalid: " + res.summary())
        append_line(self.path, canonical_json(rec))
        self.records.append(rec)
        self.seen.add(rec["card_id"])
        return rec


def load_json_line(line: str) -> dict[str, Any]:
    import json

    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise InputError("idea ledger line is not an object")
    return obj


# =============================================================================================
# DSL rendering and deterministic generators
# =============================================================================================
def to_dsl(node: Node) -> str:
    """Render an AST back to the canonical DSL string."""
    if node.kind == "num":
        val = float(node.value)
        return str(int(val)) if val == int(val) else f"{val:g}"
    if node.kind == "ref":
        return str(node.value)
    return f"{node.value}({', '.join(to_dsl(a) for a in node.args)})"


LADDER = ("q10", "q25", "q75", "q90", "q95", "q99")


def _mutations(node: Node, grid: Sequence[float]) -> Iterator[Node]:
    """One-step literal mutations (H2): numbers on the multiplicative grid, quantile ladder steps."""
    if node.kind == "call" and node.value in LADDER:
        i = LADDER.index(node.value)
        for j in (i - 1, i + 1):
            if 0 <= j < len(LADDER) and (LADDER[j] < "q50") == (node.value < "q50"):
                yield Node("call", LADDER[j], copy.deepcopy(node.args), node.path)
    if node.kind == "call" and node.value in ("gt", "gte", "lt", "lte") and node.args[1].kind == "num":
        base = float(node.args[1].value)
        for g in grid:
            if g == 1.0 or base == 0:
                continue
            new = copy.deepcopy(node)
            new.args[1] = Node("num", round(base * g, 6), path=node.args[1].path)
            yield new
    if node.kind == "call":
        for i, arg in enumerate(node.args):
            if node.value in ("in_window", "at_checkpoint", "lag", "A_then_B_within") and arg.kind != "call":
                continue
            for mutated in _mutations(arg, grid):
                new = copy.deepcopy(node)
                new.args[i] = mutated
                yield new


def _card_id(mode: str, body: dict[str, Any]) -> str:
    return f"{mode}-{sha256_json({k: v for k, v in body.items() if k != 'card_id'})[:12]}"


def _make_card(base: dict[str, Any], mode: str, **over: Any) -> dict[str, Any]:
    card = copy.deepcopy(base)
    card.update(over)
    card["provenance"] = f"HUNT:{mode}"
    if card.get("lane") == "M" and card.get("mechanism"):
        card["lane"] = "M"
    elif card.get("lane") == "USER":
        card["lane"] = "USER"
    else:
        card["lane"] = "B"
        card.pop("mechanism", None)
    card.setdefault("round", 1)
    card["card_id"] = _card_id(mode, card)
    return card


def gen_h2(leaders: Sequence[dict[str, Any]], grid: Sequence[float]) -> Iterator[dict[str, Any]]:
    for lead in leaders:
        ast = parse_event(lead["event"])
        for mutated in _mutations(ast, grid):
            yield _make_card(lead, "H2", event=to_dsl(mutated), hypothesis=f"Leader mutation of {lead['card_id']}: {to_dsl(mutated)}")


def gen_h3(leaders: Sequence[dict[str, Any]], targets: Sequence[str]) -> Iterator[dict[str, Any]]:
    for lead in leaders:
        for t in targets:
            if t == lead["target"]:
                continue
            structure = t in ("T2", "T3", "T4", "T5", "T7")
            yield _make_card(lead, "H3", target=t, cost_series="structure_only" if structure else lead.get("cost_series", "underlying"),
                             hypothesis=f"Target switch of {lead['card_id']} to {t}")


def gen_h4(leaders: Sequence[dict[str, Any]], cap: int) -> Iterator[dict[str, Any]]:
    profile = [h for h in (1, 2, 3, 5, 8, 13, 21) if h <= cap]
    for lead in leaders:
        if profile and sorted(profile) != sorted(lead["horizons"]):
            yield _make_card(lead, "H4", type="PROFILE", horizons=profile, hypothesis=f"Horizon profile of {lead['card_id']}")


def gen_h5(leaders: Sequence[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    swap = {"gt": "lt", "lt": "gt", "gte": "lte", "lte": "gte"}
    for lead in leaders:
        ast = parse_event(lead["event"])
        if ast.kind == "call" and ast.value == "DIR":
            other = "reversal" if lead["side"] == "continuation" else "continuation"
            yield _make_card(lead, "H5", side=other, hypothesis=f"Mirror ({other}) of {lead['card_id']}")
        elif ast.kind == "call" and ast.value in swap:
            yield _make_card(lead, "H5", event=to_dsl(Node("call", swap[ast.value], ast.args)), hypothesis=f"Mirror of {lead['card_id']}")
        elif ast.kind == "call" and ast.value != "MODEL":
            yield _make_card(lead, "H5", event=f"NOT({lead['event']})", hypothesis=f"Negation of {lead['card_id']}")


def gen_h9(template: dict[str, Any], available: set[str]) -> Iterator[dict[str, Any]]:
    for m in (15, 60, 120, 180):
        yield _make_card(template, "H9", event=f"AND(q90(range_atr), gt(min_since_open, {m}))",
                         hypothesis=f"Range-clock bar: top-decile range after {m} minutes")
    for g in (1, 2, 3):
        yield _make_card(template, "H9", event=f"AND(q75(range_rank_slot), eq(slot_group, {g}))",
                         hypothesis=f"Range-clock: top-quartile slot range in window group {g}")
    if "fut_volume" in available:
        for m in (15, 120):
            yield _make_card(template, "H9", event=f"AND(q95(rvol_slot), gt(min_since_open, {m}))", data_needed=["spot_1m", "fut_volume"],
                             hypothesis=f"Volume-clock bar: p95 RVOL after {m} minutes (FUT-VOLUME)")


def gen_h10(template: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for k, x in itertools.product((3, 5, 10), (1.0, 1.5, 2.0, 2.5)):
        for side in ("continuation", "reversal"):
            yield _make_card(template, "H10", event=f"DIR(gt(ret_{k}, {x}), lt(ret_{k}, -{x}))", side=side,
                             hypothesis=f"CUSUM-style sampling: {k}-bar move beyond {x} ATR20 ({side})")


def gen_h11(leaders: Sequence[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for lead in leaders:
        ast = parse_event(lead["event"])
        if ast.kind == "call" and ast.value in ("DIR", "MODEL"):
            branches = ast.args if ast.value == "DIR" else []
            if not branches:
                continue
            for t in (1, 2, 3):
                cond = f"eq(tercile(atr_pct_rank), {t})"
                new_args = [f"AND({to_dsl(b)}, {cond})" for b in branches]
                yield _make_card(lead, "H11", event=f"DIR({new_args[0]}, {new_args[1]})", hypothesis=f"{lead['card_id']} in vol tercile {t}")
            continue
        for t in (1, 2, 3):
            yield _make_card(lead, "H11", event=f"AND({lead['event']}, eq(tercile(atr_pct_rank), {t}))",
                             hypothesis=f"{lead['card_id']} conditional on prior-day ATR tercile {t}")
        for g in (1, 2, 3):
            yield _make_card(lead, "H11", event=f"AND({lead['event']}, eq(slot_group, {g}))",
                             hypothesis=f"{lead['card_id']} conditional on slot group {g}")


# =============================================================================================
# Engine
# =============================================================================================
@dataclass
class ModeStatus:
    mode: str
    kind: str
    status: str = "OPEN"
    batches: int = 0
    ideas: int = 0
    novel_clusters: int = 0
    stagnant_streak: int = 0
    reason: str = ""


@dataclass
class Evaluated:
    card: dict[str, Any]
    compiled: CompiledCard
    result: EventResult
    status: str
    mode: str
    lift: float
    se: float
    sign: int
    event_days: frozenset[str]


@dataclass
class PlaygroundResult:
    pool: list[Candidate]
    top50: list[str]
    modes: dict[str, ModeStatus]
    m_playground: int
    clusters: int
    evaluated: dict[str, Evaluated] = field(default_factory=dict)
    fits: dict[str, ShrinkageFit] = field(default_factory=dict)

    def modes_saturated(self) -> list[str]:
        return sorted((m for m, s in self.modes.items() if s.status == "SATURATED"), key=lambda x: int(x[1:]))

    def modes_open(self) -> list[str]:
        return sorted((m for m, s in self.modes.items() if s.status != "SATURATED"), key=lambda x: int(x[1:]))


class PlaygroundEngine:
    """Ω-S3 engine. ``ctx.frame`` must hold Playground days only (the seal is re-checked here)."""

    def __init__(self, ctx: FeatureContext, lib: BrickLibrary, cfg: dict[str, Any], run_ctx: Ctx,
                 manifest: SplitManifest, *, run_dir: str | Path, cards_dir: str | Path | None = None,
                 ai_fallback: dict[str, list[dict[str, Any]]] | None = None,
                 checkpoint_callback: CheckpointCallback | None = None, haircut: float | None = None,
                 log: EventLog | None = None, eras: dict[str, str] | None = None) -> None:
        frame_days = {str(d) for d in ctx.frame.days}
        sealed = frame_days & set(manifest.court_days)
        if sealed:
            raise SealViolation(f"Playground frame contains {len(sealed)} sealed Court day(s), e.g. {sorted(sealed)[0]}")
        self.ctx = ctx
        self.lib = lib
        self.cfg = cfg
        self.pcfg = cfg["playground"]
        self.run_ctx = run_ctx
        self.manifest = manifest
        self.run_dir = Path(run_dir)
        self.cards_dir = Path(cards_dir) if cards_dir is not None else None
        self.ai_fallback = ai_fallback or {}
        self.checkpoint = checkpoint_callback
        self.haircut = haircut
        self.log = log
        self.eras = eras or {}
        self.max_ideas: int | None = None
        self.ledger = IdeaLedger(self.run_dir / "idea_ledger.jsonl")
        self.state_path = self.run_dir / "playground_state.json"
        self.fit_mask = np.ones(ctx.frame.n_days, dtype=bool)
        self.evaluated: dict[str, Evaluated] = {}
        self.clusters: list[frozenset[str]] = []
        self.top: list[str] = []
        self.fits: dict[str, ShrinkageFit] = {}
        det, ai = set(self.pcfg["deterministic_modes"]), set(self.pcfg["ai_modes"])
        if det | ai != set(HUNT_MODES) or det & ai:
            raise InputError("config playground modes must partition H1..H13")
        self.modes = {m: ModeStatus(m, "deterministic" if m in det else "ai") for m in HUNT_MODES}
        self.cap = horizon_cap(ctx.frame.bars_per_day, float(cfg["stats"]["horizon_cap_frac"]))
        self._label_cache: dict[int, np.ndarray] = {}
        if self.state_path.exists():
            saved = load_json(self.state_path)
            for m, st in saved.get("modes", {}).items():
                if m in self.modes:
                    self.modes[m] = ModeStatus(**st)

    # ---- evaluation ---------------------------------------------------------------------------
    def _label(self, h: int) -> np.ndarray:
        if h not in self._label_cache:
            self._label_cache[h] = outcome(self.ctx.frame, "T1", h)["atr"]
        return self._label_cache[h]

    def evaluate_card(self, card: dict[str, Any], mode: str) -> Evaluated | None:
        """Compile, fit on Playground, run fixed-horizon tests, log to the Idea Ledger."""
        cid = card["card_id"]
        if cid in self.ledger.seen:
            return self.evaluated.get(cid)
        cc = compile_card(card, self.lib, available=self.ctx.available, cal=self.ctx.cal,
                          t7_enabled=bool(self.cfg["targets"]["T7_enabled"]), extra_bricks=set(self.ctx.extra_bricks))
        hunt = mode if mode in HUNT_MODES else "BASELINE"
        if not cc.ok:
            self.ledger.append({"card_id": cid, "card_hash": sha256_json(card), "hunt_mode": hunt, "status": "REJECTED",
                                "event_days_hash": event_days_hash([]), "event_days_count": 0, "deflated_lift": None,
                                "timestamp_ist": iso_ist(), "target": card.get("target", "T1"),
                                "family": card.get("target", "T1"), "reason": f"{cc.status}: {cc.reason}"[:300],
                                "lane": card.get("lane", "B")})
            return None
        try:
            label = self._label(int(card["horizons"][0])) if cc.is_model else None
            spec = cc.spec(self.ctx, self.fit_mask, label=label)
        except (InputError, ValueError) as exc:
            self.ledger.append({"card_id": cid, "card_hash": sha256_json(card), "hunt_mode": hunt, "status": "REJECTED",
                                "event_days_hash": event_days_hash([]), "event_days_count": 0, "deflated_lift": None,
                                "timestamp_ist": iso_ist(), "target": card["target"], "family": card["target"],
                                "reason": f"evaluation error: {exc}"[:300], "lane": card.get("lane", "B")})
            return None
        spec["family"] = "PLAYGROUND"
        spec["horizons"] = [h for h in spec["horizons"] if h <= self.cap] or spec["horizons"][:1]
        results = TestRunner(self.run_ctx, self.ctx.frame, day_mask=self.fit_mask).run_batch([spec])
        usable = [r for r in results if np.isfinite(r.se) and r.se > 0 and np.isfinite(r.lift_atr)]
        best = max(usable, key=lambda r: abs(r.lift_atr) / r.se) if usable else (results[0] if results else None)
        if best is None:
            return None
        days = frozenset(best.event_day_list())
        min_n = int(self.cfg["stats"]["effective_n_min"])
        status = "EVALUATED" if best.effective_n >= min_n else "DISCARDED"
        reason = "" if status == "EVALUATED" else f"effective N {best.effective_n} < {min_n} (ANECDOTAL)"
        if status == "EVALUATED":
            for other in self.evaluated.values():
                if other.status == "EVALUATED" and other.card["target"] == card["target"] and \
                        jaccard(days, other.event_days) >= float(self.pcfg["merge_jaccard"]) and \
                        other.card["side"] == card["side"] and abs(other.lift) / other.se >= abs(best.lift_atr) / best.se:
                    status, reason = "DISCARDED", f"duplicate of {other.card['card_id']} (event-day Jaccard >= {self.pcfg['merge_jaccard']})"
                    break
        fam = card["target"]
        dl = deflated_lift(best.lift_atr, best.se, self.fits.get(fam)) if self.fits else None
        ev = Evaluated(card=card, compiled=cc, result=best, status=status, mode=hunt, lift=best.lift_atr, se=best.se,
                       sign=1 if best.lift_atr >= 0 else -1, event_days=days)
        self.evaluated[cid] = ev
        self.ledger.append({"card_id": cid, "card_hash": cc.card_hash(), "hunt_mode": hunt, "status": status,
                            "event_days_hash": event_days_hash(days), "event_days_count": len(days),
                            "deflated_lift": None if dl is None or not np.isfinite(dl) else float(dl),
                            "timestamp_ist": iso_ist(), "target": fam, "family": fam,
                            "lift": float(best.lift_atr) if np.isfinite(best.lift_atr) else None,
                            "se": float(best.se) if np.isfinite(best.se) else None, "sign": ev.sign,
                            "reason": reason, "lane": card.get("lane", "B"),
                            "prior_weight": float(card.get("prior_weight", 1.0))})
        return ev

    # ---- ranking and saturation ---------------------------------------------------------------
    def refit(self) -> None:
        self.fits = fit_families(self.ledger.records, min_cards=int(self.cfg["shrinkage"]["min_cards"]), haircut=self.haircut)

    def shrunk(self, ev: Evaluated) -> float:
        dl = deflated_lift(ev.lift, ev.se, self.fits.get(ev.card["target"]))
        return 0.0 if dl is None or not np.isfinite(dl) else abs(dl) * float(ev.card.get("prior_weight", 1.0))

    def top_n(self) -> list[str]:
        n = int(self.pcfg["saturation"]["top_n"])
        live = [e for e in self.evaluated.values() if e.status == "EVALUATED"]
        live.sort(key=lambda e: (-self.shrunk(e), e.compiled.card_hash()))
        return [e.card["card_id"] for e in live[:n]]

    def _novel(self, days: frozenset[str]) -> bool:
        thr = float(self.pcfg["saturation"]["cluster_jaccard"])
        if all(jaccard(days, c) < thr for c in self.clusters):
            self.clusters.append(days)
            return True
        return False

    def run_batch(self, mode: str, cards: Sequence[dict[str, Any]]) -> dict[str, Any]:
        st = self.modes.get(mode)
        before = set(self.top)
        novel = 0
        n = 0
        for card in cards:
            ev = self.evaluate_card(card, mode)
            n += 1
            if ev is not None and ev.status == "EVALUATED" and self._novel(ev.event_days):
                novel += 1
        self.refit()
        self.top = self.top_n()
        added = len(set(self.top) - before)
        if st is not None:
            st.batches += 1
            st.ideas += n
            st.novel_clusters += novel
            sat = self.pcfg["saturation"]
            needed = int(sat["det_batch_min"] if st.kind == "deterministic" else sat["ai_batch_min"])
            st.stagnant_streak = st.stagnant_streak + 1 if (novel >= needed and added == 0) else 0
            if st.stagnant_streak >= int(sat["consecutive_batches"]):
                st.status, st.reason = "SATURATED", f"{st.stagnant_streak} consecutive batches of >= {needed} novel ideas added 0 new top-{sat['top_n']} cards (PR-09)"
        payload = {"hunt_mode": mode, "batch_size": n, "clusters_added": novel,
                   "saturated_modes": [m for m, s in self.modes.items() if s.status == "SATURATED"],
                   "tokens_used": 0, "top_added": added, "m_lifetime": self.ledger.m_lifetime}
        self._save()
        if self.log is not None:
            self.log.emit("playground_batch_logged", payload, stage="Ω-S3", role="AG-02")
        if self.checkpoint is not None:
            self.checkpoint({"used_ideas": self.ledger.m_lifetime, "mode": mode, "stage": "Ω-S3"})
        if self.max_ideas and self.ledger.m_lifetime >= self.max_ideas:
            raise PartialCheckpoint(f"idea budget {self.max_ideas} reached at m_lifetime {self.ledger.m_lifetime}",
                                    stage="Ω-S3", reason="max_ideas")
        return payload

    def _save(self) -> None:
        atomic_write_json(self.state_path, {"schema_version": "1.0", "modes": {m: asdict(s) for m, s in self.modes.items()},
                                            "m_lifetime": self.ledger.m_lifetime, "clusters": len(self.clusters),
                                            "top": self.top})

    def _exhaust(self, mode: str, generated: int, reason: str = "") -> None:
        st = self.modes[mode]
        if st.status != "SATURATED":
            st.status = "SATURATED"
            st.reason = reason or f"EXHAUSTED: finite grid fully enumerated ({generated} ideas); no novel ideas remain"

    def _drain(self, mode: str, generator: Iterable[dict[str, Any]]) -> None:
        """Feed a generator in fixed-size batches until it is exhausted or the mode saturates."""
        st = self.modes[mode]
        if st.status == "SATURATED":
            return
        size = int(self.pcfg["saturation"]["det_batch_min"] if st.kind == "deterministic" else self.pcfg["saturation"]["ai_batch_min"])
        it = iter(generator)
        total = 0
        while st.status != "SATURATED":
            batch = list(itertools.islice(it, size))
            if not batch:
                self._exhaust(mode, total)
                break
            total += len(batch)
            self.run_batch(mode, batch)

    # ---- host-agent cards -------------------------------------------------------------------
    def host_cards(self) -> dict[str, list[dict[str, Any]]]:
        """Idea Cards v2 from ``--cards-dir`` (validated; invalid files quarantined in the ledger)."""
        out: dict[str, list[dict[str, Any]]] = {m: [] for m in self.pcfg["ai_modes"]}
        if self.cards_dir is None or not self.cards_dir.exists():
            return out
        from bn_forensics import safe_path

        for path in sorted(self.cards_dir.iterdir()):
            safe_path(path, self.cards_dir)
            if path.suffix.lower() not in (".yaml", ".yml", ".json"):
                continue
            doc = load_yaml(path) if path.suffix.lower() != ".json" else load_json(path)
            cards = doc if isinstance(doc, list) else (doc.get("cards", [doc]) if isinstance(doc, dict) else [])
            for card in cards:
                mode = str(card.get("provenance", "")).split(":")[-1] if str(card.get("provenance", "")).startswith("HUNT:") else "H1"
                out.setdefault(mode if mode in out else "H1", []).append(card)
        return out

    def run(self, *, max_ideas: int | None = None, include_baseline: bool = True) -> PlaygroundResult:
        """Run every Hunt Mode until saturation, exhaustion or a budget checkpoint."""
        self.max_ideas = max_ideas or None
        if include_baseline:
            base_cards = [card for _, card in load_ap_cards()]
            self.run_batch("BASELINE", base_cards)
        host = self.host_cards()
        for mode in self.pcfg["ai_modes"]:
            cards = host.get(mode) or self.ai_fallback.get(mode, [])
            if not cards:
                self.modes[mode].reason = "OPEN: no host-agent cards in --cards-dir"
                continue
            self._drain(mode, cards)
        leaders = self._leader_cards(10)
        template = {"schema_version": "2.0", "type": "RULE", "lane": "B", "round": 1, "side": "both", "target": "T1",
                    "horizons": [h for h in (1, 3, 5, 10) if h <= self.cap] or [1], "baseline": "slot_matched",
                    "cutoffs": "discovery_frozen", "data_needed": ["spot_1m"], "cost_series": "underlying",
                    "prior_weight": 1.0, "expected_sign": "none", "hypothesis": "deterministic hunt template",
                    "event": "q90(range_atr)", "card_id": "TEMPLATE", "provenance": "HUNT:template"}
        targets = [t for t in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
                   if not (t == "T7" and not self.cfg["targets"]["T7_enabled"]) and not (t == "T6" and "opt_1m" not in self.ctx.available)]
        generators: dict[str, Iterable[dict[str, Any]]] = {
            "H2": gen_h2(leaders, self.pcfg["mutation_grid"]), "H3": gen_h3(leaders, targets), "H4": gen_h4(leaders, self.cap),
            "H5": gen_h5(leaders), "H9": gen_h9(template, self.ctx.available), "H10": gen_h10(template),
            "H11": gen_h11(leaders),
        }
        for mode in ("H2", "H3", "H4", "H5", "H9", "H10", "H11"):
            self._drain(mode, generators[mode])
        self._era_sweep(leaders)
        self.refit()
        self.top = self.top_n()
        self._save()
        return PlaygroundResult(pool=self.pool(), top50=self.top, modes=self.modes, m_playground=self.ledger.m_lifetime,
                                clusters=len(self.clusters), evaluated=self.evaluated, fits=self.fits)

    def _leader_cards(self, n: int) -> list[dict[str, Any]]:
        self.refit()
        live = sorted((e for e in self.evaluated.values() if e.status == "EVALUATED" and not e.compiled.is_model),
                      key=lambda e: (-self.shrunk(e), e.compiled.card_hash()))
        return [e.card for e in live[:n]]

    def _era_sweep(self, leaders: Sequence[dict[str, Any]]) -> None:
        """H12: leaders re-evaluated per era; a single-era Playground cannot be swept."""
        eras_present = sorted({self.eras.get(str(d), "UNKNOWN") for d in self.ctx.frame.days})
        if len(eras_present) < 2:
            self._exhaust("H12", 0, f"EXHAUSTED: single era {eras_present} in Playground (no era sweep possible)")
            return
        st = self.modes["H12"]
        n = 0
        for lead in leaders:
            ev = self.evaluated.get(lead["card_id"])
            if ev is None:
                continue
            for era in eras_present:
                day_mask = np.array([self.eras.get(str(d)) == era for d in self.ctx.frame.days], dtype=bool)
                spec = ev.compiled.spec(self.ctx)
                spec["family"] = "PLAYGROUND-H12"
                spec["test_id"] = f"{lead['card_id']}@{era}"
                spec["horizons"] = [ev.result.horizon]
                res = TestRunner(self.run_ctx, self.ctx.frame, day_mask=day_mask).run_batch([spec])
                n += 1
                if res:
                    days = frozenset(res[0].event_day_list())
                    self.ledger.append({"card_id": f"H12-{lead['card_id']}-{era}"[:64],
                                        "card_hash": sha256_json({"lead": ev.compiled.card_hash(), "era": era}),
                                        "hunt_mode": "H12", "status": "EVALUATED" if res[0].effective_n >= 30 else "DISCARDED",
                                        "event_days_hash": event_days_hash(days), "event_days_count": len(days),
                                        "deflated_lift": None, "timestamp_ist": iso_ist(), "target": lead["target"],
                                        "family": lead["target"],
                                        "lift": float(res[0].lift_atr) if np.isfinite(res[0].lift_atr) else None,
                                        "se": float(res[0].se) if np.isfinite(res[0].se) else None,
                                        "reason": f"era sweep {era}", "lane": lead.get("lane", "B")})
        st.ideas += n
        st.batches += 1
        self._exhaust("H12", n)

    def pool(self) -> list[Candidate]:
        """Court-eligible candidates (EVALUATED, effective N >= 30) with EB-deflated lifts."""
        out = []
        n_days = self.ctx.frame.n_days
        for ev in self.evaluated.values():
            if ev.status != "EVALUATED" or (ev.compiled.is_model and ev.compiled.model is None):
                continue
            dl = deflated_lift(ev.lift, ev.se, self.fits.get(ev.card["target"]))
            out.append(Candidate(card_id=ev.card["card_id"], card_hash=ev.compiled.card_hash(), lane=ev.card.get("lane", "B"),
                                 sign=ev.sign, target=ev.card["target"], horizon=int(ev.result.horizon),
                                 prior_weight=float(ev.card.get("prior_weight", 1.0)),
                                 deflated_lift=None if dl is None or not np.isfinite(dl) else float(dl),
                                 event_days=ev.event_days, playground_days=n_days, compiled=ev.compiled, card=ev.card))
        return out


__all__ = [
    "LADDER", "CheckpointCallback", "Evaluated", "IdeaLedger", "ModeStatus", "PlaygroundEngine", "PlaygroundResult",
    "gen_h2", "gen_h3", "gen_h4", "gen_h5", "gen_h9", "gen_h10", "gen_h11", "load_json_line", "to_dsl",
]
