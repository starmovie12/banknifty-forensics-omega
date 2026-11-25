"""Standalone Step F verification pass (FILE-037, FR-16, R18; App Y F2.5, V5-15).

Runs after ``findings_registry.csv`` is frozen, as an isolated subprocess
(``python3 -m bn_forensics.verify_checks --run-dir runs/<id> --store <store>``). Imports only the
package core (FILE-012) and the independent twin (FILE-036). It never touches ``core_stats``.

1. **Row recompute.** The rows checked are a seed-42 stratified sample of ``verify.sample_rows`` (10)
   rows by Evidence grade, every ``Ω-A`` / Grade ``A`` row, and the ``val-top-3`` (largest
   ``|Validation lift|``). Each is recomputed with ``twin_impl`` from the canonical Parquet store,
   using the event specs in ``tables/registry_events.csv``. ``Event N`` must match exactly.
   ``Event value``, ``Baseline`` and ``Lift (ATR20)`` must agree within ``verify.rel_tol`` (1%).
2. **Fix rounds.** At most ``verify.max_fix_rounds`` (2). An optional in-process ``fixer`` repairs
   inputs between rounds, and every round re-reads the run files from disk. A mismatch that persists
   is demoted: the row leaves the registry for ``tables/graveyard.csv`` as ``D UNVERIFIED``. The
   result is ``PARTIAL`` (exit 2). A demoted row still present afterwards is ``MODE: INVALID`` (exit 4).
3. **Tables 11A/11B.** An independent re-implementation of the deterministic 2A.3 rule runs on
   ``tables/approach_tests.csv``. On disagreement or demotion, both tables are rebuilt.
4. **Appendix Y LIVE assertions** are written to ``tables/live_assertions.csv``.

Outputs are ``tables/verification_log.json`` (``schema_version`` 1.0), ``tables/live_assertions.csv``
and the event ``verification_completed`` (M-1).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bn_forensics import (
    EXIT_INPUT,
    EXIT_INVALID,
    EXIT_OK,
    EXIT_PARTIAL,
    BnfError,
    EventLog,
    InputError,
    InvalidPipeline,
    asset_dir,
    atomic_write_json,
    atomic_write_text,
    load_config,
    load_yaml,
    sha256_file,
)
from bn_forensics.twin_impl import (
    Bar,
    EventSpec,
    compare_values,
    read_partition_bars,
    resample_bars,
    twin_event_study,
)

# 37-column Findings Registry (§10.2, S4 §9#8, K-043): 32 V3.4 columns + 5 V5 columns.
REGISTRY_FIELDS = (
    "ID", "Phase", "Family_m", "Label", "Statement", "Metric", "Horizon", "Side", "Event N", "Effective N",
    "Event value", "Baseline", "Lift (points)", "Lift (ATR20)", "Hit-rate lift (pp)", "95% CI", "Nominal MDE",
    "SE_disc", "Gate-adjusted MDE", "q_disc", "p_val", "Discovery lift", "Validation lift", "Stability",
    "Outlier test", "Skeptic verdict", "Evidence grade", "Tradability tag", "Regime", "Event-day hash",
    "Overlap with disc-top-3", "Approach ID",
    "Lane", "Deflated lift", "Mechanism status", "Burn count", "Independent effect ID",
)
GRAVEYARD_FIELDS = (*REGISTRY_FIELDS, "graveyard_reason")
REGISTRY_EVENT_FIELDS = ("ID", "kind", "date", "minute", "direction", "target", "horizon", "side", "baseline")
APPROACH_TEST_FIELDS = ("approach_id", "test_id", "status", "side", "grade", "t_tag", "q_disc", "se", "gate_mde",
                        "effective_n", "skeptic_ran", "cost_series", "missing", "reason")
VERDICT_11A_FIELDS = (
    "approach_id", "group", "name", "verdict", "best_grade", "best_test", "best_side", "best_tag",
    "nominal_mde_med", "gate_mde_med", "gate_mde_min", "gate_mde_max", "family_gate_mde_med",
    "needed_effect", "runnable_tests", "tests_n30", "label", "reason", "missing",
)
VERDICT_11B_FIELDS = ("group", "group_name", "members_S", "members_C", "members_I", "members_NA",
                      "best_evidence_grade", "verdict", "reason")
LIVE_FIELDS = ("assertion_id", "assertion", "status", "detail")
MISMATCH_FIELDS = ("ID", "field", "registry", "recomputed", "reason")
COMPARED_FIELDS = (("Event value", "event_value"), ("Baseline", "baseline"), ("Lift (ATR20)", "lift_atr"))
GRADE_RANK = {"Ω-A": 6, "A": 5, "Ω-B": 4, "B": 3, "Ω-C": 2, "C": 1, "D": 0}
TAG_RANK = {"T1": 4, "T2": 3, "T?": 2, "T3": 1, "STRUCTURE-ONLY": 0}
TAGS = frozenset(TAG_RANK)
VERDICTS_11A = frozenset({"SUPPORTED FOR HYPOTHESIS", "CONTRADICTED", "INCONCLUSIVE", "N/A — DATA MISSING", "N/A DEGRADED"})
NA_MISSING = "N/A — DATA MISSING"
PARTITION_RE = re.compile(r"(?P<table>[A-Za-z0-9_]+)/underlying=(?P<underlying>[^/]+)/trading_date=(?P<date>\d{4}-\d{2}-\d{2})/[^/]+\.parquet$")
LOG_SCHEMA_VERSION = "1.0"


# =============================================================================================
# Small IO helpers
# =============================================================================================
def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        text = f"{value:.10g}"
        return "0" if text == "-0" else text
    return str(value)


def write_rows(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _cell(row.get(k)) for k in fields})
    atomic_write_text(path, buf.getvalue())
    return path


def num(text: Any) -> float:
    """Parse a CSV cell into a float (blank / unparsable -> NaN)."""
    if text is None:
        return float("nan")
    if isinstance(text, (int, float)):
        return float(text)
    try:
        return float(str(text).strip())
    except ValueError:
        return float("nan")


def _ok(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def read_plan(path: Path) -> dict[str, str]:
    """``key: value`` lines of ``test_plan.md`` (day_limited, atr20_median_pts, cost_pts_<series>)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lstrip("-* ").strip()
        if ":" in stripped and not stripped.startswith("#"):
            key, val = stripped.split(":", 1)
            out[key.strip().strip("`").lower()] = val.strip().strip("`")
    return out


# =============================================================================================
# Store access (twin bars straight from canonical Parquet)
# =============================================================================================
def find_partitions(store: Path, table: str, underlying: str, days: set[str] | None = None) -> dict[str, Path]:
    """One partition file per trading date. A date served from two roots raises (no silent merge)."""
    found: dict[str, list[Path]] = {}
    for path in sorted(Path(store).rglob("*.parquet")):
        m = PARTITION_RE.search(path.as_posix())
        if not m or m["table"] != table or m["underlying"] != underlying:
            continue
        if days is not None and m["date"] not in days:
            continue
        found.setdefault(m["date"], []).append(path)
    out: dict[str, Path] = {}
    for day, paths in found.items():
        dirs = {p.parent for p in paths}
        if len(dirs) > 1:
            raise InputError(f"ambiguous store: trading_date={day} served by {len(dirs)} partitions (F9, no silent merge)")
        if len(paths) > 1:
            raise InputError(f"trading_date={day}: {len(paths)} part files in one partition")
        out[day] = paths[0]
    return out


def load_bars(store: Path, table: str, underlying: str, bar_minutes: int, days: set[str] | None = None) -> list[Bar]:
    parts = find_partitions(store, table, underlying, days)
    if not parts:
        raise InputError(f"no {table} partitions for {underlying} under {store}")
    bars = read_partition_bars([parts[d] for d in sorted(parts)])
    return resample_bars(bars, bar_minutes) if bar_minutes > 1 else bars


# =============================================================================================
# Row selection and recompute
# =============================================================================================
def select_rows(registry: Sequence[dict[str, str]], *, seed: int = 42, sample_rows: int = 10,
                val_top: int = 3) -> tuple[list[str], dict[str, list[str]]]:
    """Seed-42 stratified sample (by Evidence grade) + every Ω-A/A row + val-top-3."""
    rng = np.random.default_rng(seed)
    strata: dict[str, list[str]] = {}
    for row in registry:
        strata.setdefault(row.get("Evidence grade", "") or "D", []).append(row["ID"])
    queues = {g: [ids[i] for i in rng.permutation(len(ids))] for g, ids in sorted(strata.items())}
    sample: list[str] = []
    while len(sample) < min(sample_rows, len(registry)):
        for g in sorted(queues):
            if queues[g] and len(sample) < sample_rows:
                sample.append(queues[g].pop(0))
    grade_a = [r["ID"] for r in registry if r.get("Evidence grade") in ("Ω-A", "A")]
    scored = [(abs(num(r.get("Validation lift"))), r["ID"]) for r in registry if _ok(num(r.get("Validation lift")))]
    top = [rid for _, rid in sorted(scored, key=lambda t: (-t[0], t[1]))[:val_top]]
    reasons: dict[str, list[str]] = {}
    for rid in sample:
        reasons.setdefault(rid, []).append("seed42-sample")
    for rid in grade_a:
        reasons.setdefault(rid, []).append("grade-A")
    for rid in top:
        reasons.setdefault(rid, []).append("val-top")
    order = [rid for rid in (r["ID"] for r in registry) if rid in reasons]
    return order, reasons


@dataclass
class RowSpec:
    target: str
    horizon: int
    side: str
    baseline: str
    events: list[tuple[str, int, float]]
    days: set[str]
    frame_days: set[str]


def load_specs(path: Path) -> dict[str, RowSpec]:
    specs: dict[str, RowSpec] = {}
    for r in read_rows(path):
        rid = r["ID"]
        spec = specs.get(rid)
        if spec is None:
            spec = specs[rid] = RowSpec(r.get("target") or "T1", int(num(r.get("horizon")) if _ok(num(r.get("horizon"))) else 1),
                                        r.get("side") or "both", r.get("baseline") or "slot_matched", [], set(), set())
        kind = r.get("kind", "event")
        if kind == "event":
            spec.events.append((r["date"], int(num(r["minute"])), num(r.get("direction")) if _ok(num(r.get("direction"))) else 1.0))
        elif kind == "day":
            spec.days.add(r["date"])
        elif kind == "frame":
            spec.frame_days.add(r["date"])
        else:
            raise InputError(f"registry_events.csv: unknown kind {kind!r} for {rid}")
    return specs


def recompute(spec: RowSpec, bars: Sequence[Bar]) -> dict[str, Any]:
    """Twin recompute of one registry row on the frame days the engine used."""
    frame_days = spec.frame_days or {b.date for b in bars}
    sub = [b for b in bars if b.date in frame_days]
    index = {(b.date, b.minute): i for i, b in enumerate(sub)}
    events, unknown = [], []
    for date, minute, direction in spec.events:
        i = index.get((date, minute))
        if i is None:
            unknown.append(f"{date} {minute}")
        else:
            events.append(EventSpec(i, direction))
    flagged = [e.index for e in events]
    tw = twin_event_study(sub, events, target=spec.target, h=spec.horizon, side=spec.side, baseline=spec.baseline,
                          day_set=spec.days or frame_days, event_mask_all=flagged)
    return {**tw, "unknown_events": unknown}


def check_row(row: dict[str, str], spec: RowSpec | None, bars: Sequence[Bar], rel_tol: float) -> list[dict[str, Any]]:
    rid = row["ID"]
    if spec is None or not spec.events:
        return [{"ID": rid, "field": "*", "registry": "", "recomputed": "", "reason": "NO EVENT SPEC (cannot verify)"}]
    tw = recompute(spec, bars)
    out = []
    if tw["unknown_events"]:
        out.append({"ID": rid, "field": "events", "registry": len(spec.events), "recomputed": len(spec.events) - len(tw["unknown_events"]),
                    "reason": f"event bars absent from store: {tw['unknown_events'][:3]}"})
    reg_n = num(row.get("Event N"))
    if not _ok(reg_n) or int(reg_n) != int(tw["n_events"]):
        out.append({"ID": rid, "field": "Event N", "registry": row.get("Event N", ""), "recomputed": tw["n_events"], "reason": "count differs"})
    for reg_field, tw_field in COMPARED_FIELDS:
        a, b = float(tw[tw_field]), num(row.get(reg_field))
        if not compare_values(a, b, tol_abs=1e-6, tol_rel=rel_tol):
            out.append({"ID": rid, "field": reg_field, "registry": row.get(reg_field, ""), "recomputed": a,
                        "reason": f"differs by more than {rel_tol:.0%}"})
    return out


# =============================================================================================
# Independent 2A.3 verdict rule (Tables 11A / 11B)
# =============================================================================================
def _needed(costs: dict[str, float | None], series: str, atr_pts: float) -> float | None:
    cost = None if series == "structure_only" else costs.get(series)
    if cost is None or not _ok(atr_pts) or atr_pts <= 0:
        return None
    return cost / atr_pts


def _test_key(t: dict[str, Any]) -> tuple[int, int, float]:
    q = num(t.get("q_disc"))
    return (GRADE_RANK.get(t.get("grade") or "D", 0), TAG_RANK.get(t.get("t_tag") or "T?", 2), -(q if _ok(q) else 1.0))


def rule_2a3(approach: dict[str, Any], tests: Sequence[dict[str, Any]], *, costs: dict[str, float | None], atr_pts: float,
             day_limited: bool) -> dict[str, Any]:
    aid = approach["approach_id"]
    mine = [t for t in tests if t.get("approach_id") == aid]
    head = {"approach_id": aid, "group": approach["group"], "name": approach.get("name", "")}
    runnable = [t for t in mine if t.get("status") == "EXECUTED"]
    if not runnable:
        missing = sorted({m for t in mine for m in str(t.get("missing") or "").split(";") if m})
        reason = f"{NA_MISSING}: {missing}" if missing else (mine[0].get("reason", "") if mine else "no tests")
        return {**head, "verdict": NA_MISSING, "reason": reason, "missing": ";".join(missing), "label": "",
                "runnable_tests": 0, "tests_n30": 0}
    best = None
    for side in ("continuation", "reversal", "both"):
        side_tests = [t for t in runnable if t.get("side") == side]
        if side_tests:
            cand = max(side_tests, key=_test_key)
            best = cand if best is None or _test_key(cand) > _test_key(best) else best
    if best is None:
        best = max(runnable, key=_test_key)
    se = np.array([num(t.get("se")) for t in runnable], float)
    nominal = float(np.nanmedian(2.8 * se)) if (~np.isnan(se)).any() else float("nan")
    gated = [num(t.get("gate_mde")) for t in runnable if num(t.get("effective_n")) >= 30 and _ok(num(t.get("gate_mde")))]
    gate_med = float(np.median(gated)) if gated else float("nan")
    need = _needed(costs, best.get("cost_series") or "option_buyer", atr_pts)
    n30 = sum(1 for t in runnable if num(t.get("effective_n")) >= 30)
    grade = best.get("grade") or "D"
    tag = best.get("t_tag") or "T?"
    out = {**head, "best_grade": grade, "best_test": best.get("test_id", ""), "best_side": best.get("side", ""), "best_tag": tag,
           "nominal_mde_med": nominal, "gate_mde_med": gate_med, "gate_mde_min": min(gated) if gated else float("nan"),
           "gate_mde_max": max(gated) if gated else float("nan"), "family_gate_mde_med": gate_med,
           "needed_effect": need if need is not None else "UNKNOWN", "runnable_tests": len(runnable), "tests_n30": n30,
           "missing": "", "label": ""}
    strong = grade in ("A", "B", "Ω-A", "Ω-B")
    raw_tag = best.get("t_tag") or ""          # an untagged best test is never SUPPORTED (same as FILE-028)
    if strong and int(num(best.get("skeptic_ran")) if _ok(num(best.get("skeptic_ran"))) else 0) >= 3 and not day_limited \
            and raw_tag in ("T1", "T2", "T?", "STRUCTURE-ONLY"):
        return {**out, "verdict": "SUPPORTED FOR HYPOTHESIS", "reason": f"best {best.get('test_id', '')} grade {grade}, tag {tag}"}
    if strong and raw_tag == "T3":
        return {**out, "verdict": "CONTRADICTED", "label": "REAL BUT UNTRADABLE",
                "reason": f"statistically real but below matching cost (gate MDE {gate_med:.3f} ATR20)"}
    no_abc = not any((t.get("grade") or "D") in ("A", "B", "C", "Ω-A", "Ω-B", "Ω-C") for t in runnable)
    if no_abc and n30 >= 0.5 * len(runnable):
        if need is None:
            return {**out, "verdict": "CONTRADICTED", "label": "STAT-ONLY",
                    "reason": f"0 A/B/C tests, {n30}/{len(runnable)} tests with effective N >= 30; gate-adjusted MDE {gate_med:.3f} ATR20; cost unknown (STAT-ONLY)"}
        if _ok(gate_med) and gate_med <= need:
            return {**out, "verdict": "CONTRADICTED",
                    "reason": f"0 A/B/C tests; gate-adjusted MDE {gate_med:.3f} <= needed effect {need:.3f} ATR20"}
    why = []
    if day_limited:
        why.append("DAY-LIMITED (< 120 usable days)")
    if no_abc and n30 < 0.5 * len(runnable):
        why.append(f"only {n30}/{len(runnable)} tests have effective N >= 30")
    if need is not None and _ok(gate_med) and gate_med > need:
        why.append(f"gate-adjusted MDE {gate_med:.3f} > needed effect {need:.3f} ATR20")
    if not no_abc:
        why.append(f"best grade {grade} lacks Skeptic coverage or tag")
    return {**out, "verdict": "INCONCLUSIVE", "reason": "; ".join(why) or "no rule matched"}


def rule_rollup(approach: dict[str, Any], member_rows: Sequence[dict[str, Any]], vr_rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    roll = approach["rollup"]
    side = roll["side"]
    members = [r for r in member_rows if r["approach_id"] in roll["members"]]
    hits = [r for r in vr_rows if str(r.get("excludes_1", "")).lower() == "true" and _ok(num(r.get("vr")))
            and ((num(r["vr"]) > 1) if side == "continuation" else (num(r["vr"]) < 1))]
    sup = [r for r in members if r["verdict"] == "SUPPORTED FOR HYPOTHESIS" and r.get("best_side") in (side, "both")]
    runnable = [r for r in members if r["verdict"] != NA_MISSING]
    if sup:
        verdict = "SUPPORTED FOR HYPOTHESIS"
    elif not runnable:
        verdict = NA_MISSING
    elif all(r["verdict"] == "CONTRADICTED" for r in runnable) and not hits:
        verdict = "CONTRADICTED"
    else:
        verdict = "INCONCLUSIVE"
    return {"approach_id": approach["approach_id"], "group": approach["group"], "name": approach.get("name", ""),
            "verdict": verdict, "runnable_tests": 0, "tests_n30": 0, "best_grade": "", "label": "ROLL-UP (0 tests)",
            "reason": f"{len(sup)} supported {side} members; {len(hits)} variance-ratio horizons with CI excluding 1", "missing": ""}


def rule_11b(rows_11a: Sequence[dict[str, Any]], groups: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for g in sorted(groups, key=lambda x: int(x[1:])):
        mem = [r for r in rows_11a if r["group"] == g]
        s = sum(1 for r in mem if r["verdict"] == "SUPPORTED FOR HYPOTHESIS")
        c = sum(1 for r in mem if r["verdict"] == "CONTRADICTED")
        na = sum(1 for r in mem if r["verdict"] in (NA_MISSING, "N/A DEGRADED"))
        i = len(mem) - s - c - na
        runnable = [r for r in mem if r["verdict"] not in (NA_MISSING, "N/A DEGRADED")]
        if s:
            verdict = "SUPPORTED"
        elif runnable and all(r["verdict"] == "CONTRADICTED" for r in runnable):
            verdict = "CONTRADICTED"
        elif not runnable:
            verdict = "N/A"
        else:
            verdict = "INCONCLUSIVE"
        grades = [r.get("best_grade") for r in mem if r.get("best_grade")]
        best = max(grades, key=lambda x: GRADE_RANK.get(x, 0)) if grades else "—"
        out.append({"group": g, "group_name": groups[g], "members_S": s, "members_C": c, "members_I": i, "members_NA": na,
                    "best_evidence_grade": best, "verdict": verdict, "reason": f"S/C/I/NA = {s}/{c}/{i}/{na}"})
    return out


def recompute_tables(tests: Sequence[dict[str, Any]], plan: dict[str, str], vr_rows: Sequence[dict[str, str]],
                     ap_path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    doc = load_yaml(ap_path or asset_dir("config") / "ap_cards.yaml")
    costs = {k[len("cost_pts_"):]: (num(v) if _ok(num(v)) else None) for k, v in plan.items() if k.startswith("cost_pts_")}
    atr_pts = num(plan.get("atr20_median_pts"))
    day_limited = str(plan.get("day_limited", "false")).lower() == "true"
    plain = [a for a in doc["approaches"] if not a.get("rollup")]
    rows = [rule_2a3(a, tests, costs=costs, atr_pts=atr_pts, day_limited=day_limited) for a in plain]
    for a in doc["approaches"]:
        if a.get("rollup"):
            rows.append(rule_rollup(a, rows, vr_rows))
    rows.sort(key=lambda r: r["approach_id"])
    return rows, rule_11b(rows, doc["groups"])


def diff_tables(published: Sequence[dict[str, str]], recomputed: Sequence[dict[str, Any]], key: str,
                fields: Sequence[str]) -> list[dict[str, Any]]:
    pub = {r.get(key, ""): r for r in published}
    out = []
    for r in recomputed:
        p = pub.get(r[key])
        if p is None:
            out.append({"ID": r[key], "field": key, "registry": "", "recomputed": r[key], "reason": "row missing from published table"})
            continue
        for f in fields:
            if str(p.get(f, "")) != _cell(r.get(f, "")):
                out.append({"ID": r[key], "field": f, "registry": p.get(f, ""), "recomputed": _cell(r.get(f, "")), "reason": "2A.3 recompute differs"})
    extra = sorted(set(pub) - {r[key] for r in recomputed})
    out += [{"ID": k, "field": key, "registry": k, "recomputed": "", "reason": "unexpected row in published table"} for k in extra]
    return out


# =============================================================================================
# Appendix Y LIVE assertions
# =============================================================================================
def live_assertions(root: Path) -> list[dict[str, Any]]:
    tables = root / "tables"
    reg_path = root / "findings_registry.csv"
    header: list[str] = []
    if reg_path.exists():
        with open(reg_path, encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh), [])
    reg = read_rows(reg_path)
    gy_path = tables / "graveyard.csv"
    gy = read_rows(gy_path)
    v11a = read_rows(tables / "approach_verdicts_11a.csv")
    v11b = read_rows(tables / "approach_verdicts_11b.csv")
    test_reg = read_rows(tables / "test_registry.csv")
    rows: list[dict[str, Any]] = []

    def add(aid: str, text: str, ok: bool, detail: str) -> None:
        rows.append({"assertion_id": aid, "assertion": text, "status": "PASS" if ok else "FAIL", "detail": detail})

    add("LIVE-01", "findings_registry.csv has the 37 §10.2 columns in order", tuple(header) == REGISTRY_FIELDS,
        f"{len(header)} columns")
    bad_grade = [r["ID"] for r in reg if r.get("Evidence grade") not in GRADE_RANK]
    add("LIVE-02", "every Evidence grade is one of Ω-A/Ω-B/Ω-C/A/B/C/D", not bad_grade, f"bad: {bad_grade[:5]}")
    exposed_a = [r["ID"] for r in reg if r.get("Evidence grade") == "Ω-A" and "EXPOSED-COURT" in (r.get("Label") or "")]
    add("LIVE-03", "no Ω-A finding carries EXPOSED-COURT (R28)", not exposed_a, f"offenders: {exposed_a[:5]}")
    bad_tag = [r["ID"] for r in reg if r.get("Evidence grade") in ("Ω-A", "Ω-B", "Ω-C", "A", "B") and r.get("Tradability tag") not in TAGS]
    add("LIVE-04", "graded findings carry a T-tag from deflated_lift (T1/T2/T3/T?/STRUCTURE-ONLY)", not bad_tag, f"bad: {bad_tag[:5]}")
    bad_burn = []
    for r in reg:
        b = num(r.get("Burn count"))
        if not _ok(b) or b < 0 or b != int(b) or (r.get("Evidence grade") in ("Ω-A", "Ω-B") and b < 1):
            bad_burn.append(r["ID"])
    add("LIVE-05", "Burn count = Court session k >= 1 for Court grades, 0 only if Ledger-only (CF-02)", not bad_burn, f"bad: {bad_burn[:5]}")
    ap_ids = sorted(r.get("approach_id", "") for r in v11a)
    groups = sorted((r.get("group", "") for r in v11b), key=lambda g: int(g[1:]) if g[1:].isdigit() else 0)
    add("LIVE-06", "Table 11A lists AP01-AP36 and Table 11B lists G1-G10",
        ap_ids == [f"AP{i:02d}" for i in range(1, 37)] and groups == [f"G{i}" for i in range(1, 11)],
        f"11A rows {len(ap_ids)}, 11B rows {len(groups)}")
    bad_v = [r.get("approach_id") for r in v11a if r.get("verdict") not in VERDICTS_11A or "PROXY" in (r.get("verdict") or "")]
    # AP32/AP33 roll-ups carry 0 tests; their verdict is derived from member approaches (exempt)
    bad_c = [r.get("approach_id") for r in v11a if r.get("verdict") == "CONTRADICTED"
             and not str(r.get("label", "")).startswith("ROLL-UP")
             and not (_ok(num(r.get("gate_mde_med"))) or r.get("label") in ("STAT-ONLY", "REAL BUT UNTRADABLE"))]
    add("LIVE-07", "11A verdicts in the 2A.3 set, CONTRADICTED carries gate MDE / STAT-ONLY / REAL BUT UNTRADABLE, 0 PROXY",
        not bad_v and not bad_c, f"bad verdicts {bad_v[:5]}, bare CONTRADICTED {bad_c[:5]}")
    statuses = {r.get("status") for r in test_reg}
    add("LIVE-08", "test_registry.csv reconciles planned == executed + logged_skipped", bool(test_reg) and statuses <= {"EXECUTED", "SKIPPED"},
        f"{len(test_reg)} planned; statuses {sorted(s or '' for s in statuses)}")
    add("LIVE-09", "tables/graveyard.csv saved (SHA-256 recorded)", gy_path.exists(),
        f"sha256 {sha256_file(gy_path)[:16]}" if gy_path.exists() else "missing")
    no_mde = [r.get("ID", "") for r in gy if not _ok(num(r.get("Nominal MDE")))]
    add("LIVE-10", "every Graveyard line carries its MDE", not no_mde, f"missing MDE: {no_mde[:5]}")
    no_sk = [r["ID"] for r in reg if r.get("Evidence grade") in ("Ω-A", "Ω-B", "Ω-C") and "SELF-REVIEW" not in (r.get("Skeptic verdict") or "")]
    add("LIVE-11", "every Ω-graded finding carries the isolated Skeptic verdict (SELF-REVIEW)", not no_sk, f"missing: {no_sk[:5]}")
    bad_lift = [r["ID"] for r in reg if num(r.get("Event N")) >= 1 and not _ok(num(r.get("Lift (ATR20)")))]
    add("LIVE-12", "every registry row with events has a finite Lift (ATR20)", not bad_lift, f"bad: {bad_lift[:5]}")
    return rows


# =============================================================================================
# Orchestration
# =============================================================================================
@dataclass
class VerifyResult:
    exit_code: int
    checked: list[str]
    reasons: dict[str, list[str]]
    mismatches: list[dict[str, Any]]
    demoted: list[str]
    table_mismatches: list[dict[str, Any]]
    tables_rebuilt: bool
    live: list[dict[str, Any]]
    rounds: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verify_pass(self) -> bool:
        return self.exit_code == EXIT_OK

    def status_line(self) -> str:
        live_ok = sum(1 for r in self.live if r["status"] == "PASS")
        mode = "PASS" if self.verify_pass else ("PARTIAL" if self.exit_code == EXIT_PARTIAL else "INVALID")
        return (f"[Ω-S6] VERIFY {mode}: checked={len(self.checked)}, mismatches={len({m['ID'] for m in self.mismatches})}, "
                f"demoted={len(self.demoted)}, tables={'REBUILT' if self.tables_rebuilt else 'OK'}, live={live_ok}/{len(self.live)}")


def _demote(root: Path, ids: set[str], mismatches: Sequence[dict[str, Any]]) -> None:
    reg_path = root / "findings_registry.csv"
    reg = read_rows(reg_path)
    keep = [r for r in reg if r["ID"] not in ids]
    gone = [r for r in reg if r["ID"] in ids]
    gy_path = root / "tables" / "graveyard.csv"
    gy = read_rows(gy_path)
    for r in gone:
        why = "; ".join(f"{m['field']}: {m['reason']}" for m in mismatches if m["ID"] == r["ID"])
        label = ";".join(x for x in (r.get("Label") or "").split(";") if x)
        gy.append({**r, "Evidence grade": "D", "Label": f"{label};UNVERIFIED" if label else "UNVERIFIED",
                   "graveyard_reason": f"D UNVERIFIED: {why}"})
    write_rows(gy_path, GRAVEYARD_FIELDS, gy)
    write_rows(reg_path, REGISTRY_FIELDS, keep)
    tests_path = root / "tables" / "approach_tests.csv"
    tests = read_rows(tests_path)
    if tests:
        card_ids = {i.split("@")[0] for i in ids}
        for t in tests:
            if t.get("test_id") in ids or t.get("test_id", "").split("@")[0] in card_ids:
                t["grade"] = "D"
        write_rows(tests_path, APPROACH_TEST_FIELDS, tests)


def run_verify(run_dir: str | Path, *, bars: Sequence[Bar] | None = None, store: str | Path | None = None,
               cfg: dict[str, Any] | None = None, table: str | None = None, underlying: str | None = None,
               bar_minutes: int | None = None, fixer: Callable[[list[dict[str, Any]]], None] | None = None,
               log: EventLog | None = None, ap_path: str | Path | None = None) -> VerifyResult:
    """Step F: recompute, fix rounds, demotion, 11A/11B rebuild, LIVE assertions."""
    cfg = cfg or load_config()
    vc = cfg["verify"]
    root = Path(run_dir)
    tables = root / "tables"
    reg_path = root / "findings_registry.csv"
    if not reg_path.exists():
        raise InputError(f"findings_registry.csv not found under {root}")
    plan = read_plan(root / "test_plan.md")
    seed = int(num(plan.get("seed", "42"))) if _ok(num(plan.get("seed", "42"))) else 42
    registry = read_rows(reg_path)
    checked, reasons = select_rows(registry, seed=seed, sample_rows=int(vc["sample_rows"]), val_top=int(vc["val_top"]))
    if bars is None:
        if store is None:
            raise InputError("verify needs --store (canonical Parquet) or in-memory bars")
        bars = load_bars(Path(store), table or cfg["analysis"]["source_table"], underlying or cfg["store"]["primary_underlying"],
                         int(bar_minutes or cfg["analysis"]["bar_minutes"]))
    rel_tol = float(vc["rel_tol"])
    max_rounds = int(vc["max_fix_rounds"])

    def check_all() -> list[dict[str, Any]]:
        reg = {r["ID"]: r for r in read_rows(reg_path)}
        specs = load_specs(tables / "registry_events.csv")
        out: list[dict[str, Any]] = []
        for rid in checked:
            if rid in reg:
                out += check_row(reg[rid], specs.get(rid), bars, rel_tol)
        return out

    rounds = []
    mismatches = check_all()
    rounds.append({"round": 0, "mismatched_ids": sorted({m["ID"] for m in mismatches})})
    for k in range(1, max_rounds + 1):
        if not mismatches:
            break
        if fixer is not None:
            fixer(list(mismatches))
        mismatches = check_all()
        rounds.append({"round": k, "fixer": fixer is not None, "mismatched_ids": sorted({m["ID"] for m in mismatches})})
    demoted = sorted({m["ID"] for m in mismatches})
    ap = Path(ap_path) if ap_path else None
    vr_rows = [r for r in read_rows(tables / "phase_2_2.csv") if r.get("vr")]
    table_mismatches: list[dict[str, Any]] = []
    rebuilt = False
    if read_rows(tables / "approach_tests.csv"):
        # published tables vs the independent 2A.3 recompute on the frozen (pre-demotion) inputs
        r11a, r11b = recompute_tables(read_rows(tables / "approach_tests.csv"), plan, vr_rows, ap)
        table_mismatches = diff_tables(read_rows(tables / "approach_verdicts_11a.csv"), r11a, "approach_id", ("verdict", "label", "best_grade")) + \
            diff_tables(read_rows(tables / "approach_verdicts_11b.csv"), r11b, "group", ("verdict", "best_evidence_grade"))
        if demoted:
            _demote(root, set(demoted), mismatches)
            r11a, r11b = recompute_tables(read_rows(tables / "approach_tests.csv"), plan, vr_rows, ap)
        if demoted or table_mismatches:
            write_rows(tables / "approach_verdicts_11a.csv", VERDICT_11A_FIELDS, r11a)
            write_rows(tables / "approach_verdicts_11b.csv", VERDICT_11B_FIELDS, r11b)
            rebuilt = True
    else:
        if demoted:
            _demote(root, set(demoted), mismatches)
        table_mismatches = [{"ID": "11A/11B", "field": "approach_tests.csv", "registry": "", "recomputed": "",
                             "reason": "tables/approach_tests.csv missing: 2A.3 recompute not possible"}]
    live = live_assertions(root)
    write_rows(tables / "live_assertions.csv", LIVE_FIELDS, live)
    still = {r["ID"] for r in read_rows(reg_path)} & set(demoted)
    if still:
        exit_code = EXIT_INVALID
    elif demoted or table_mismatches or any(r["status"] == "FAIL" for r in live):
        exit_code = EXIT_PARTIAL
    else:
        exit_code = EXIT_OK
    result = VerifyResult(exit_code=exit_code, checked=checked, reasons=reasons, mismatches=mismatches, demoted=demoted,
                          table_mismatches=table_mismatches, tables_rebuilt=rebuilt, live=live, rounds=rounds)
    gy = tables / "graveyard.csv"
    atomic_write_json(tables / "verification_log.json", {
        "schema_version": LOG_SCHEMA_VERSION, "verify_pass": result.verify_pass, "exit_code": exit_code,
        "sample_rows": int(vc["sample_rows"]), "checked": [{"ID": i, "why": reasons[i]} for i in checked],
        "rel_tol": rel_tol, "max_fix_rounds": max_rounds, "rounds": rounds, "mismatches": mismatches, "demoted": demoted,
        "table_mismatches": table_mismatches, "tables_rebuilt": rebuilt,
        "live": {"pass": sum(1 for r in live if r["status"] == "PASS"), "total": len(live)},
        "graveyard_sha256": sha256_file(gy) if gy.exists() else "", "status_line": result.status_line(),
    })
    if log is not None:
        log.emit("verification_completed", {"sample_rows": len(checked), "mismatches": len(mismatches), "demoted_count": len(demoted),
                                            "verify_pass": result.verify_pass}, stage="Ω-S6", role="AG-06",
                 level="INFO" if result.verify_pass else "WARN")
    if still:
        raise InvalidPipeline(f"MODE: INVALID - unverified rows still in findings_registry.csv: {sorted(still)}", tripwire_id="F2.5")
    return result


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 3 (§10.3), never argparse's default 2."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        raise SystemExit(EXIT_INPUT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(prog="python -m bn_forensics.verify_checks", description="Standalone Step F verifier (FILE-037)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--store", required=True, help="canonical Parquet store root")
    parser.add_argument("--config", default=None)
    parser.add_argument("--table", default=None)
    parser.add_argument("--underlying", default=None)
    parser.add_argument("--bar-minutes", type=int, default=None)
    parser.add_argument("--log", default=None, help="JSONL event log to append verification_completed to")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if not Path(args.run_dir).is_dir():
            raise InputError(f"run dir not found: {args.run_dir}")
        log = EventLog(args.log) if args.log else None
        res = run_verify(args.run_dir, store=args.store, cfg=cfg, table=args.table, underlying=args.underlying,
                         bar_minutes=args.bar_minutes, log=log)
    except BnfError as exc:
        sys.stderr.write(f"verify_checks: {exc}\n")
        return int(exc.exit_code)
    sys.stdout.write(res.status_line() + "\n")
    return res.exit_code


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "APPROACH_TEST_FIELDS",
    "GRAVEYARD_FIELDS",
    "LIVE_FIELDS",
    "REGISTRY_EVENT_FIELDS",
    "REGISTRY_FIELDS",
    "VERDICT_11A_FIELDS",
    "VERDICT_11B_FIELDS",
    "RowSpec",
    "VerifyResult",
    "check_row",
    "diff_tables",
    "find_partitions",
    "live_assertions",
    "load_bars",
    "load_specs",
    "main",
    "read_plan",
    "recompute",
    "recompute_tables",
    "rule_2a3",
    "rule_11b",
    "rule_rollup",
    "run_verify",
    "select_rows",
]
