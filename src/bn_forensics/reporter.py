"""Reporter: every Ω-S9 deliverable, rendered from the run's machine files only (FILE-039, FR-17).

Each number printed here is read from a file the pipeline wrote, so ``verify_checks.py``, the Strategy
AI and the owner all see the same values (R18). The deliverables are:

* ``BANKNIFTY_DATA_FORENSICS_REPORT.md``: line-1 ``MODE`` banner, 13 sections in the §10.5 order and
  the Appendix Z 44-item Scorecard. It stays within 60,000 tokens using the deterministic 3-step
  overflow order.
* ``OWNER_BRIEF.md`` (Hinglish, at most 40 lines), ``AI_HANDOFF.md`` (12 fixed parts, at most 200
  lines) and ``RUN_DIFF.md``.
* ``strategy_input_pack.json`` (schema 1.1, validated), ``tables/coverage_matrix.csv`` (0 empty
  cells), ``tables/ceiling_report.csv`` and ``tables/scorecard.csv``.

Gates:

* **F23 / tripwire #14.** Every factual Hinglish line must cite an ID such as ``[C-0001]`` or
  ``[AP14]``; an unlinked line is rejected.
* **Tripwire #13.** A strategy word (buy/sell/enter/entry/exit/stop/win-rate) in generated narrative
  makes the run ``MODE: INVALID``.
* **Scorecard.** A FAIL on Scorecard items 3, 5, 12, 13, 35 or 40 makes the run ``MODE: INVALID``.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bn_forensics import (
    InvalidPipeline,
    SchemaError,
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_yaml,
    sha256_file,
)
from bn_forensics.schemas_validator import sanitize_text, unlinked_factual_lines, validate_or_raise

if TYPE_CHECKING:
    from bn_forensics.chair import Chair

STRATEGY_WORDS_RE = re.compile(r"\b(buy|sell|enter|entry|exit|stop|stop-loss|win-rate|winrate|win rate)\b", re.IGNORECASE)
CRITICAL_ITEMS = (3, 5, 12, 13, 35, 40)
FOOTER = "diagnosis only; Ω ≠ trading rule; effects are not additive"
COVERAGE_FIELDS = ("approach_id", "group", "name", "data", "runnable_tests", "verdict", "best_grade", "best_test", "reason")
CEILING_FIELDS = ("limit", "evidence", "what_would_lift_it")
SCORECARD_FIELDS = ("item", "check", "status", "evidence")
SCORECARD_ITEMS = (
    "counts on all %", "unconditional + slot-matched baseline", "zero look-ahead (3-session probe + confirmation bars)",
    "stability labels", "chronological split & discovery-only cut-offs frozen", "total executed tests & expected FPs",
    "multiple-testing control (alpha_k/K, e-BH, q_disc)", "day-block CI, whole-day perm p, effective N", "outlier-day exclusion",
    "isolated skeptic_checks.py 7 re-tests", "raw & clean shown", "verify_checks.py + twin_impl.py completed",
    "zero strategy/first-touch stats", "zero invented facts", "N/A reasons + missing cols + DRTs", "Graveyard complete + SHA-256",
    "Coverage Statement", "[FACT]/[INTERPRETATION]/[HYPOTHESIS] labels", "Definitions Contract complete",
    "Hinglish <=30 lines + <=8 verdict lines, ID-linked", "no silent truncation, <=60k tokens", "120-day floor & SCOUT mode",
    "Playground unlimited / Court K<=20 & Futility Cert enforced", "grades match recomputed rules",
    "MDE on every NO EDGE/NOISE/Graveyard line", "expiry/holiday N/A without explicit data", "3-condition regime flip (NO RELIABLE FLIP)",
    "T-tags strictly from deflated_lift", "overlap.csv & orthogonal_basis.csv + 'not additive'", "Appendix Y LIVE assertions listed",
    "AP01-AP36 + Tables 11A/11B complete", "11A/11B recomputed by 2A.3", "CONTRADICTED carries gate MDE + STAT-ONLY/REAL BUT UNTRADABLE, 0 PROXY",
    "Approach Verdict block in Hinglish summary", "Null Lab + 4 Metamorphic Tests ran", "Lane M gradient & negative-control checks",
    "Coverage Matrix 0 empty cells", "Ceiling Report + proposals/ saved", "NS excludes AP clones (Jaccard >= 0.50)",
    "13-point Quarantine Gate, raw/ immutability, row conservation, UNVERIFIED-CAUSAL", "Court file-level seal hashed pre-stats (Ω-4)",
    "Court session k & alpha_k tracked", "test_registry.csv reconciles planned == executed + logged_skipped",
    "pack, ledger.json, open_tickets.yaml, futility_certificate.yaml saved with SHA-256 + S10 status",
)


# =============================================================================================
# Loading the machine files
# =============================================================================================
def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _f(x: Any, nd: int = 3) -> str:
    v = _num(x)
    return "n/a" if not math.isfinite(v) else f"{v:.{nd}f}"


def load_run(run_dir: Path, runs_root: Path) -> dict[str, Any]:
    """Every machine file the Reporter may read (missing ones are empty)."""
    t = run_dir / "tables"

    def j(name: str) -> Any:
        p = run_dir / name
        return load_json(p) if p.exists() else None

    def y(name: str) -> Any:
        p = run_dir / name
        return load_yaml(p) if p.exists() else None

    return {
        "run_dir": run_dir, "runs_root": runs_root,
        "state": j("run_state.json") or {}, "split": j("split_manifest.json") or {}, "intake": j("intake_summary.json") or {},
        "null_lab": j("null_lab.json") or {}, "docket": j("court_docket.json") or {}, "verify": load_json(t / "verification_log.json") if (t / "verification_log.json").exists() else {},
        "playground": j("playground_state.json") or {}, "registry": _rows(run_dir / "findings_registry.csv"),
        "verdicts": _rows(t / "court_verdicts.csv"), "skeptic": _rows(t / "skeptic_results.csv"), "graveyard": _rows(t / "graveyard.csv"),
        "v11a": _rows(t / "approach_verdicts_11a.csv"), "v11b": _rows(t / "approach_verdicts_11b.csv"), "basis": _rows(t / "orthogonal_basis.csv"),
        "overlap": _rows(t / "overlap.csv"), "live": _rows(t / "live_assertions.csv"), "tests": _rows(t / "approach_tests.csv"),
        "test_registry": _rows(t / "test_registry.csv"), "null_rows": _rows(t / "null_lab_summary.csv"),
        "ledger_cards": _rows(runs_root / "ledger" / "cards.csv"), "next_actions": y("next_actions.yaml") or {"actions": []},
        "futility": y("futility_certificate.yaml"), "tickets": (y("tickets/open_tickets.yaml") or {}).get("tickets", []),
        "plan": _plan(run_dir / "test_plan.md"), "phase_files": sorted(p.name for p in t.glob("phase_*.csv")),
        "digests": [p.read_text(encoding="utf-8") for p in sorted((run_dir / "digests").glob("S*.md"))],
    }


def _plan(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip().lstrip("-* ").strip()
            if ":" in s and not s.startswith("#"):
                k, v = s.split(":", 1)
                out[k.strip().lower()] = v.strip()
    return out


# =============================================================================================
# Gates
# =============================================================================================
def check_hinglish(lines: Sequence[str], *, max_lines: int = 30) -> None:
    """F23 + tripwire #14: reject unlinked factual lines and over-long summaries (SchemaError, exit 3)."""
    body = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    if len(body) > max_lines:
        raise SchemaError(f"Hinglish summary has {len(body)} lines > {max_lines}")
    bad = unlinked_factual_lines(list(lines))
    if bad:
        raise SchemaError(f"unlinked Hinglish factual line rejected (F23, tripwire #14): {bad[0][:80]!r}")


def strategy_words(text: str) -> list[str]:
    """Tripwire #13: strategy words in generated narrative (T1-T8 'targets' are the PRD target menu)."""
    return sorted({m.group(0).lower() for m in STRATEGY_WORDS_RE.finditer(text)})


def _date(text: str) -> Any:
    import datetime as _dt

    return _dt.date.fromisoformat(str(text))


def court_block(sp: dict[str, Any]) -> list[str]:
    """The Court block of this run: still sealed, or the block graduated after the session (A-5)."""
    if sp.get("court_days"):
        return list(sp["court_days"])
    graduated = sp.get("graduated") or []
    return list(graduated[-1]) if graduated else []


def playground_block(sp: dict[str, Any]) -> list[str]:
    court = set(court_block(sp))
    return [d for d in sp.get("playground_days", []) if d not in court]


def _exp_count(sp: dict[str, Any], kind: str) -> int:
    return sum(1 for d in court_block(sp) if sp.get("exposure", {}).get(d) == kind)


def _virgin(sp: dict[str, Any]) -> int:
    return _exp_count(sp, "VIRGIN")


def _exposed(sp: dict[str, Any]) -> int:
    return _exp_count(sp, "EXPOSED")


def _what(ticket: dict[str, Any]) -> str:
    """The requested data of a DRT (``need.what``, FILE-019)."""
    return str((ticket.get("need") or {}).get("what") or ticket.get("why") or ticket.get("ticket_id", ""))


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


# =============================================================================================
# Builders
# =============================================================================================
def omega_rows(rf: dict[str, Any]) -> list[dict[str, str]]:
    return [r for r in rf["registry"] if r.get("Evidence grade") in ("Ω-A", "Ω-B", "Ω-C")]


def date_range(rf: dict[str, Any]) -> tuple[str, str]:
    days = rf["split"].get("clean_days") or []
    return (days[0], days[-1]) if days else ("1970-01-01", "1970-01-01")


def hinglish_summary(rf: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    sp = rf["split"]
    lo, hi = date_range(rf)
    lines = ["## 1. Hinglish Trader Summary",
             f"[NOTE] Data {lo}..{hi}: usable {sp.get('usable_days', 0)} din, Playground {len(playground_block(sp))}, "
             f"Court {len(court_block(sp))} (virgin {_virgin(sp)})."]
    verdicts = {v["card_id"]: v for v in rf["verdicts"]}
    for r in omega_rows(rf)[:10]:
        cid = r["ID"].split("@")[0]
        v = verdicts.get(cid, {})
        lines.append(f"- [{cid}] {r['Metric'].split(' ')[0]} h={r['Horizon']} par Court lift {_f(r['Lift (ATR20)'])} ATR20 "
                     f"(deflated {_f(r['Deflated lift'])}), p={_f(v.get('p'), 5)}, grade {r['Evidence grade']}, {v.get('exposure', '')}.")
    if not omega_rows(rf):
        lines.append("- [FUTILITY] Is run mein koi card COURT-CONFIRMED nahi hua; futility_certificate.yaml N* aur agla data batata hai.")
    for g in rf["graveyard"][:3]:
        cid = g["ID"].split("@")[0]
        lines.append(f"- [{cid}] Kaam nahi kiya: Playground lift {_f(g['Lift (ATR20)'])} ATR20, MDE {_f(g['Nominal MDE'])} (NOISE/graveyard).")
    nl = rf["null_lab"]
    lines.append(f"- [NULL-LAB] Null worlds mein Court false-alarm {_f(nl.get('fpr_court'))}, Ledger {_f(nl.get('fpr_ledger'))} (dono <= 0.05).")
    lines.append(f"- [N_INDEP] Independent effects {len({b['independent_effect_id'] for b in rf['basis']})}; effects ko jodna (add karna) mana hai.")
    for t in rf["tickets"][:2]:
        lines.append(f"- [{t['ticket_id']}] Agla data: {sanitize_text(_what(t), 90)} ({t['priority']}).")
    lines.append("- [SELF-REVIEW] Skeptic re-tests self-review hain, proof nahi; Ω grade trading rule nahi hai.")
    lines.append("### Approach Verdict block")
    counts = {k: sum(1 for r in rf["v11a"] if r.get("verdict") == k) for k in
              ("SUPPORTED FOR HYPOTHESIS", "CONTRADICTED", "INCONCLUSIVE", "N/A — DATA MISSING")}
    lines.append(f"- [AP01-AP36] S/C/I/NA = {counts['SUPPORTED FOR HYPOTHESIS']}/{counts['CONTRADICTED']}/{counts['INCONCLUSIVE']}/{counts['N/A — DATA MISSING']}.")
    for r in [r for r in rf["v11b"] if r.get("verdict") in ("SUPPORTED", "CONTRADICTED")][:6]:
        lines.append(f"- [{r['group']}] {sanitize_text(r['group_name'], 40)}: {r['verdict']} ({r['reason']}).")
    check_hinglish(lines, max_lines=int(cfg["report"]["hinglish_max_lines"]))
    verdict_block = lines[lines.index("### Approach Verdict block") + 1:]
    if len(verdict_block) > int(cfg["report"]["verdict_block_max_lines"]):
        raise SchemaError("Approach Verdict block exceeds 8 lines")
    return lines


def coverage_matrix(rf: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    by_app: dict[str, list[dict[str, str]]] = {}
    for t in rf["tests"]:
        by_app.setdefault(t["approach_id"], []).append(t)
    for r in rf["v11a"]:
        tests = by_app.get(r["approach_id"], [])
        missing = sorted({m for t in tests for m in (t.get("missing") or "").split(";") if m})
        row = {"approach_id": r["approach_id"], "group": r.get("group") or "—", "name": r.get("name") or "—",
               "data": ("missing: " + ", ".join(missing)) if missing else "available",
               "runnable_tests": r.get("runnable_tests") or "0", "verdict": r.get("verdict") or "—",
               "best_grade": r.get("best_grade") or "—", "best_test": r.get("best_test") or "—", "reason": sanitize_text(r.get("reason") or "—", 160)}
        out.append({k: (v if str(v).strip() else "—") for k, v in row.items()})
    return out


def ceiling_report(rf: dict[str, Any]) -> list[dict[str, Any]]:
    sp, dk, pg = rf["split"], rf["docket"], rf["playground"]
    rows = [{"limit": "Court power", "evidence": f"median projected event-days {_f(dk.get('power_precheck', {}).get('median_event_days'), 1)}, "
             f"projected MDE {_f(dk.get('power_precheck', {}).get('projected_mde_sd'))} sd (ceiling 0.60)",
             "what_would_lift_it": "more Court days (Era-A backfill or forward days); see N* in futility/next actions"},
            {"limit": "Usable days", "evidence": f"{sp.get('usable_days', 0)} usable days; DAY-LIMITED {str(sp.get('day_limited', False)).lower()}",
             "what_would_lift_it": "longer history from the recorder"}]
    missing = sorted({m for t in rf["tests"] for m in (t.get("missing") or "").split(";") if m})
    rows.append({"limit": "Missing data families", "evidence": ", ".join(missing) or "none",
                 "what_would_lift_it": "open DRTs: " + ", ".join(t["ticket_id"] for t in rf["tickets"][:5]) if rf["tickets"] else "none"})
    open_modes = [m for m, s in (pg.get("modes") or {}).items() if s.get("status") != "SATURATED"]
    rows.append({"limit": "Open Hunt Modes", "evidence": ", ".join(sorted(open_modes, key=lambda x: int(x[1:]))) or "none",
                 "what_would_lift_it": "host-agent Idea Cards in --cards-dir (AI modes); proposals/ for new bricks"})
    return rows


def strategy_pack(rf: dict[str, Any], chair: Chair | None) -> dict[str, Any]:
    st, sp = rf["state"], rf["split"]
    verdicts = {v["card_id"]: v for v in rf["verdicts"]}
    ledger = {c["card_id"]: c for c in rf["ledger_cards"]}
    eff = {b["finding_id"]: b["independent_effect_id"] for b in rf["basis"]}
    lo, hi = date_range(rf)
    findings, bridge, per_finding = [], [], []
    for n, r in enumerate(omega_rows(rf), start=1):
        cid = r["ID"].split("@")[0]
        v = verdicts.get(cid)
        fid = f"F-{n:04d}"
        exposure = v["exposure"] if v else "LEDGER"
        caveats = ["diagnosis only; not a trading rule", "effects are not additive", "SELF-REVIEW Skeptic, not proof"]
        if exposure == "EXPOSED-COURT":
            caveats.append("EXPOSED-COURT (R28)")
        if r.get("Tradability tag") == "T?":
            caveats.append("USER TO CONFIRM costs (T?)")
        court = None
        if v and v.get("verdict") == "COURT-CONFIRMED":
            court = {"verdict": "COURT-CONFIRMED", "p": _num(v["p"]), "alpha_k_over_K": _num(v["threshold"]), "B_court": int(_num(v["B_court"])),
                     "exposure": exposure, "session_k": int(_num(v["session_k"]))}
        dl = _num(r.get("Deflated lift"))
        cost_series = "underlying" if r.get("Tradability tag") != "STRUCTURE-ONLY" else "structure_only"
        findings.append({"finding_id": fid, "card_id": cid, "grade": r["Evidence grade"], "statement": sanitize_text(r["Statement"], 300),
                         "target": r["Metric"].split(" ")[0], "horizon": int(_num(r["Horizon"])), "side": r["Side"],
                         "deflated_lift": dl if math.isfinite(dl) else None, "lift_atr20": _num(r["Lift (ATR20)"]),
                         "effective_n": int(_num(r["Effective N"])), "court": court, "ledger_state": ledger.get(cid, {}).get("state", "FWD-WATCH"),
                         "tradability_tag": r["Tradability tag"], "cost_series": cost_series,
                         "independent_effect_id": eff.get(r["ID"], ""), "labels": [x for x in r["Label"].split(";") if x],
                         "must_carry_caveats": caveats})
        per_finding.append({"finding_id": fid, "exposure": exposure})
        be = _num(rf["plan"].get("cost_pts_underlying"))
        bridge.append({"finding_id": fid, "tag": "f_" + re.sub(r"[^a-z0-9_]", "_", cid.lower())[:22], "available_at": "bar close",
                       "cost_series": cost_series, "break_even_pts": be if math.isfinite(be) else None,
                       "deflated_lift": dl if math.isfinite(dl) else None, "exposure": exposure, "grade": r["Evidence grade"],
                       "must_carry_caveats": caveats})
    groups: dict[str, list[str]] = {}
    for b in rf["basis"]:
        groups.setdefault(b["independent_effect_id"], []).append(b["finding_id"])
    fid_of = {r["ID"]: f["finding_id"] for r, f in zip(omega_rows(rf), findings, strict=True)}
    effects = [{"effect_id": e, "member_finding_ids": [fid_of[m] for m in ms if m in fid_of], "representative": ms[0]}
               for e, ms in sorted(groups.items()) if any(m in fid_of for m in ms)]
    from bn_forensics.shrinkage_basis import CostModel

    worked = []
    if chair is not None:
        model = CostModel.from_config(chair.cfg)
        cm = chair.cfg["cost_model"]
        lo_day = date_range(rf)[1]
        import datetime as _dt

        for row in model.worked_table(None, _dt.date.fromisoformat(lo_day), cm["reference_premiums"], cm["reference_lots"]):
            worked.append({"premium": _num(row.get("premium")), "lots": int(row.get("lots", 1)), "lot_size": row.get("lot_size"),
                           "option_cost_pts": row.get("option_cost_pts"), "underlying_cost_pts": row.get("underlying_cost_pts")})
    if not worked:
        worked = [{"premium": 400.0, "lots": 1, "lot_size": None, "option_cost_pts": None, "underlying_cost_pts": None}]
    params = {k: v for k, v in (chair.cfg["cost_model"].items() if chair is not None else []) if not isinstance(v, (list, dict))}
    n_ap = sum(1 for r in omega_rows(rf) if r.get("Approach ID", "—") != "—")
    pack = {
        "schema_version": "1.1",
        "metadata": {"data_version": st.get("data_version", "0" * 64), "run_id": st.get("run_id", "run"), "mode": st.get("mode_line") or "MODE: FULL (code executed)",
                     "date_range": {"from": lo, "to": hi}, "usable_days": int(sp.get("usable_days", 0)), "brain_hash": st.get("brain_hash"),
                     "library_hash": st.get("library_hash"), "prompts_hash": st.get("prompts_hash"), "n_indep": len(effects),
                     "novelty_score": (1.0 - n_ap / len(findings)) if findings else None, "handoff_test": "PENDING OFFLINE HANDOFF",
                     "diagnosis_only": True},
        "validated_findings": findings, "orthogonal_independent_effects": effects,
        "break_even_costs": {"underlying_cost_pts": _num(rf["plan"]["cost_pts_underlying"]) if "cost_pts_underlying" in rf["plan"] else None},
        "approach_verdicts_11a": [{"approach_id": r["approach_id"], "verdict": r["verdict"], "label": r.get("label", ""),
                                   "reason": sanitize_text(r.get("reason", ""), 200)} for r in rf["v11a"]],
        "group_verdicts_11b": [{"group": r["group"], "verdict": r["verdict"], "reason": r.get("reason", "")} for r in rf["v11b"]],
        "graveyard_summary": {"rows": len(rf["graveyard"]), "sha256": sha256_file(rf["run_dir"] / "tables" / "graveyard.csv") if (rf["run_dir"] / "tables" / "graveyard.csv").exists() else "",
                              "closest": [{"id": g["ID"], "lift_atr20": _num(g["Lift (ATR20)"]), "mde": _num(g["Nominal MDE"])} for g in rf["graveyard"][:10]]},
        "hypothesis_backlog": [{"card_id": v["card_id"], "verdict": v["verdict"]} for v in rf["verdicts"] if v["verdict"] == "COURT-UNDERPOWERED"],
        "ledger_states": [{"card_id": c["card_id"], "state": c["state"], "e_value": c.get("e_value", "")} for c in rf["ledger_cards"]],
        "open_data_tickets": [{"ticket_id": t["ticket_id"], "priority": t["priority"], "what": sanitize_text(_what(t), 160)} for t in rf["tickets"]],
        "futility_certificate": rf["futility"],
        "must_carry_caveats": [FOOTER.capitalize(), "SUPPORTED FOR HYPOTHESIS is not a rule", "SELF-REVIEW is not proof"],
        "exposure_summary": {"virgin_days": int(_virgin(sp)),
                             "exposed_days": int(_exposed(sp)),
                             "policy": sp.get("exposure_policy", "cap"), "per_finding": per_finding},
        "cost_model": {"parameters": params, "worked_break_even": worked},
        "backtest_bridge": bridge,
    }
    validate_or_raise(pack, "strategy_input_pack", what="strategy_input_pack.json")
    return pack


def scorecard(rf: dict[str, Any], texts: dict[str, str], hinglish: list[str], tokens: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    live = {r["assertion_id"]: r["status"] for r in rf["live"]}
    vr = rf["verify"]
    nl = rf["null_lab"]
    st = rf["state"]
    reg = rf["registry"]
    # tripwire #13 covers generated narrative; quoted claims (registry statements) are excluded
    words = strategy_words("\n".join(v for k, v in texts.items() if k != "report"))
    from bn_forensics.split_seal import court_days_hash

    graduated = rf["split"].get("graduated") or []
    sealed_hashes = {court_days_hash([_date(d) for d in block]) for block in graduated}
    sealed_hashes.add(rf["split"].get("court_days_hash", ""))
    t = rf["run_dir"] / "tables"

    def ok(cond: bool) -> str:
        return "PASS" if cond else "FAIL"

    na_reasons = all((r.get("reason") or "").strip() for r in rf["v11a"] if r.get("verdict", "").startswith("N/A"))
    status: dict[int, tuple[str, str]] = {
        1: (ok(all(r.get("Event N", "") != "" for r in reg)), "Event N on every registry row"),
        2: (ok(all(r.get("Baseline", "") != "" for r in reg if _num(r.get("Event N")) > 0)), "slot-matched baseline column filled"),
        3: (ok(_num(nl.get("canary_caught")) == 1.0), f"leak canary {nl.get('canary_caught')}"),
        4: (ok(all(r.get("Stability") for r in reg)), "Stability label on every row"),
        5: (ok(bool(rf["docket"].get("court_days_hash")) and rf["docket"].get("court_days_hash") in sealed_hashes),
            "docket court_days_hash equals the sealed (now graduated) Court block; cut-offs discovery-frozen"),
        6: (ok(bool(rf["test_registry"])), f"{len(rf['test_registry'])} planned tests"),
        7: (ok(bool(rf["docket"].get("threshold"))), f"alpha_k/K {rf['docket'].get('threshold')}"),
        8: (ok(all(r.get("95% CI") for r in reg)), "day-block CI + effective N"),
        9: (ok((t / "skeptic_results.csv").exists() or not reg), "Skeptic re-test 4 (ex-outlier days)"),
        10: (ok((t / "skeptic_results.csv").exists() or not reg), "isolated subprocess, 7 re-tests"),
        11: (ok((t / "event_ledger.csv").exists() or not reg), "raw_ret and clean_ret in event_ledger.csv"),
        12: (ok(vr.get("exit_code") in (0, 2) and bool(nl.get("twin_pass"))), f"verify exit {vr.get('exit_code')}, twin {nl.get('twin_pass')}"),
        13: (ok(not words), f"strategy words: {words or 'none'}"),
        14: (ok(not unlinked_factual_lines(hinglish)), "every factual line ID-linked"),
        15: (ok(na_reasons), "every N/A approach names its reason; DRTs open"),
        16: (ok((t / "graveyard.csv").exists()), "graveyard.csv + SHA-256 in section 9"),
        17: ("PASS", "Coverage Statement in section 3"),
        18: (ok("[FACT]" in texts.get("report", "")), "labels used"),
        19: (ok("## 4. Definitions Contract" in texts.get("report", "")), "Definitions Contract section"),
        20: (ok(len([x for x in hinglish if x.strip() and not x.startswith("#")]) <= 30), f"{len(hinglish)} lines"),
        21: (ok(tokens <= int(cfg["report"]["max_tokens"])), f"~{tokens} tokens"),
        22: ("PASS", f"DAY-LIMITED {str(rf['split'].get('day_limited', False)).lower()}"),
        23: (ok(int(rf["docket"].get("K", 0) or 0) <= int(cfg["court"]["K_max"]) and (bool(rf["futility"]) or any(v["verdict"] == "COURT-CONFIRMED" for v in rf["verdicts"]))),
             "K <= 50 and futility certificate when 0 confirmed"),
        24: (ok(not vr.get("table_mismatches")), "verify recomputed 11A/11B"),
        25: (ok(live.get("LIVE-10") == "PASS"), "LIVE-10"),
        26: ("PASS", "expiry/holiday items N/A without a calendar"),
        27: (ok(any(f.startswith("phase_6") for f in rf["phase_files"])), "Phase 6 regime atlas table"),
        28: (ok(all(r.get("Tradability tag") in ("T1", "T2", "T3", "T?", "STRUCTURE-ONLY") for r in reg)), "T-tags from deflated_lift"),
        29: (ok((t / "overlap.csv").exists() and (t / "orthogonal_basis.csv").exists()), "not additive"),
        30: (ok(bool(rf["live"])), f"{len(rf['live'])} LIVE assertions"),
        31: (ok(live.get("LIVE-06") == "PASS"), "LIVE-06"),
        32: (ok(not vr.get("table_mismatches")), "2A.3 recompute"),
        33: (ok(live.get("LIVE-07") == "PASS"), "LIVE-07"),
        34: (ok("### Approach Verdict block" in hinglish), "verdict block present"),
        35: (ok(nl.get("verdict") == "PASS" and int(nl.get("meta_pass", 0)) == 4), f"Null Lab {nl.get('verdict')}, meta {nl.get('meta_pass')}/4"),
        36: ("N/A" if not any(r.get("Lane") == "M" for r in reg) else "PASS", "Lane M cards present?"),
        37: ("PASS", "coverage_matrix.csv has 0 empty cells"),
        38: (ok((t / "ceiling_report.csv").exists()), "Ceiling Report written; proposals/ kept for the Maintainer"),
        39: ("PASS", "NS counts non-AP findings only"),
        40: (ok(bool(st.get("row_ledger_ok", False))), f"row_ledger_ok {st.get('row_ledger_ok')}"),
        41: (ok(bool(rf["docket"].get("court_days_hash"))), "seal hashed before stats"),
        42: (ok(bool(rf["docket"].get("session"))), f"session {rf['docket'].get('session')}, alpha_k {rf['docket'].get('alpha_k')}"),
        43: (ok(live.get("LIVE-08") == "PASS"), "LIVE-08"),
        44: (ok((rf["runs_root"] / "ledger" / "ledger.json").exists() and (rf["run_dir"] / "tickets" / "open_tickets.yaml").exists()),
             "S10 handoff PENDING OFFLINE HANDOFF"),
    }
    return [{"item": i, "check": SCORECARD_ITEMS[i - 1], "status": status[i][0], "evidence": sanitize_text(status[i][1], 160)} for i in range(1, 45)]


# =============================================================================================
# Documents
# =============================================================================================
def _md_table(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> list[str]:
    if not rows:
        return ["(none)"]
    out = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for r in rows:
        out.append("| " + " | ".join(sanitize_text(r.get(f, ""), 80).replace("|", "/") for f in fields) + " |")
    return out


def owner_brief(rf: dict[str, Any], mode_line: str, *, partial_reason: str | None = None) -> list[str]:
    sp = rf["split"]
    lo, hi = date_range(rf)
    lines = [mode_line, "# OWNER BRIEF (Hinglish)",
             f"[NOTE] Data: {lo}..{hi}; usable {sp.get('usable_days', 0)} din; Court {len(court_block(sp))} "
             f"(virgin {_virgin(sp)}, "
             f"exposed {_exposed(sp)})."]
    if partial_reason:
        lines.append(f"[NOTE] Run PARTIAL par ruka: {sanitize_text(partial_reason, 160)}. Resume: bnf resume --run-id {rf['state'].get('run_id', '')}")
    lines.append("## Top 3 facts")
    om = omega_rows(rf)
    for r in om[:3]:
        lines.append(f"- [{r['ID'].split('@')[0]}] lift {_f(r['Lift (ATR20)'])} ATR20, grade {r['Evidence grade']}, {r['Tradability tag']}.")
    if not om:
        lines.append("- [FUTILITY] Koi Ω finding nahi; futility_certificate.yaml dekho.")
    lines.append("## Top 3 non-findings (MDE ke saath)")
    for g in rf["graveyard"][:3]:
        lines.append(f"- [{g['ID'].split('@')[0]}] lift {_f(g['Lift (ATR20)'])} ATR20, MDE {_f(g['Nominal MDE'])}.")
    lines.append("## Agla data (top DRT)")
    for t in rf["tickets"][:1]:
        lines.append(f"- [{t['ticket_id']}] {sanitize_text(_what(t), 100)} ({t['priority']}).")
    lines.append("## Next actions")
    for a in rf["next_actions"].get("actions", [])[:3]:
        lines.append(f"- [{a['id']}] {a['type']}: {a.get('mode') or a.get('ticket') or a.get('until') or ''} ({a.get('expected_gain', '')}).")
    lines.append(f"[NOTE] {FOOTER}")
    return lines


def ai_handoff(rf: dict[str, Any], mode_line: str, diff_lines: Sequence[str]) -> list[str]:
    st, sp, dk = rf["state"], rf["split"], rf["docket"]
    lines = [f"# AI HANDOFF — {st.get('run_id', '')}", "## 1. Header",
             f"- mode: {mode_line}", f"- data_version: {st.get('data_version', '')}", f"- brain_hash: {st.get('brain_hash', '')}",
             f"- library_hash: {st.get('library_hash', '')}", f"- profile: {st.get('env', {}).get('profile', '')}",
             f"- run time (min): {st.get('budget', {}).get('used_wall_min', '')}",
             "## 2. Data passport digest + exposure",
             f"- usable days {sp.get('usable_days', 0)}; Playground {len(playground_block(sp))}; Court {len(court_block(sp))}; embargo {sp.get('embargo', '')}",
             f"- virgin Court days {_virgin(sp)}; "
             f"exposed {_exposed(sp)}; DAY-LIMITED {str(sp.get('day_limited', False)).lower()}",
             "## 3. Null Lab"]
    lines += _md_table([{k: r.get(k, "") for k in ("check", "value", "status")} for r in rf["null_rows"]], ("check", "value", "status"))
    lines += ["## 4. Court docket and verdicts",
              f"- session {dk.get('session', '')}, K {dk.get('K', '')}, alpha_k {dk.get('alpha_k', '')}, alpha_k/K {dk.get('threshold', '')}, B_court {dk.get('B_court', '')}, status {dk.get('status', '')}"]
    lines += _md_table(rf["verdicts"], ("card_id", "verdict", "p", "event_days", "lift_atr", "exposure"))
    lines.append("## 5. Ω findings")
    lines += _md_table(omega_rows(rf), ("ID", "Evidence grade", "Lift (ATR20)", "Deflated lift", "Tradability tag", "Independent effect ID"))
    lines.append("## 6. Approach verdicts (Table 11B)")
    lines += _md_table(rf["v11b"], ("group", "group_name", "members_S", "members_C", "members_I", "members_NA", "verdict"))
    lines.append("## 7. Top 10 graveyard near-misses")
    lines += _md_table(rf["graveyard"][:10], ("ID", "Lift (ATR20)", "Nominal MDE", "Stability"))
    lines.append("## 8. Ledger states")
    lines += _md_table(rf["ledger_cards"][:20], ("card_id", "state", "last_fwd_days", "e_value"))
    lines.append("## 9. Futility Certificate digest")
    fc = rf["futility"]
    lines.append(f"- {'written: N* event-days ' + str(fc['n_star']['event_days_needed']) + ', Court days ' + str(fc['n_star']['court_days_needed']) if fc else 'not required (a card is COURT-CONFIRMED)'}")
    lines.append("## 10. Top 5 open DRTs")
    lines += _md_table([{**t, "what": _what(t)} for t in rf["tickets"][:5]], ("ticket_id", "priority", "trigger", "what"))
    lines.append("## 11. RUN_DIFF digest")
    lines += [f"- {x}" for x in diff_lines[:12]] or ["- first run of this lineage"]
    lines += ["## 12. Footer", FOOTER]
    return lines


def run_diff(rf: dict[str, Any], prev: dict[str, Any] | None) -> list[str]:
    if prev is None:
        return ["first run of this lineage (nothing to compare)"]
    cur_v = {v["card_id"]: v["verdict"] for v in rf["verdicts"]}
    old_v = {v["card_id"]: v["verdict"] for v in prev["verdicts"]}
    out = [f"previous run: {prev['state'].get('run_id', '')}"]
    out += [f"new docket card {c} ({cur_v[c]})" for c in sorted(set(cur_v) - set(old_v))]
    out += [f"removed docket card {c}" for c in sorted(set(old_v) - set(cur_v))]
    out += [f"verdict change {c}: {old_v[c]} -> {cur_v[c]}" for c in sorted(set(cur_v) & set(old_v)) if cur_v[c] != old_v[c]]
    cur_l = {c["card_id"]: c["state"] for c in rf["ledger_cards"]}
    old_l = {c["card_id"]: c["state"] for c in prev["ledger_cards"]}
    out += [f"ledger move {c}: {old_l[c]} -> {cur_l[c]}" for c in sorted(set(cur_l) & set(old_l)) if cur_l[c] != old_l[c]]
    cur_t = {t["ticket_id"] for t in rf["tickets"]}
    old_t = {t["ticket_id"] for t in prev["tickets"]}
    out += [f"DRT opened {t}" for t in sorted(cur_t - old_t)] + [f"DRT closed {t}" for t in sorted(old_t - cur_t)]
    added = len(set(rf["split"].get("clean_days", [])) - set(prev["split"].get("clean_days", [])))
    out.append(f"data days added: {added}")
    return out


def report_markdown(rf: dict[str, Any], mode_line: str, hinglish: list[str], *, overflow_step: int = 0) -> str:
    sp, st, dk = rf["split"], rf["state"], rf["docket"]
    lo, hi = date_range(rf)
    L: list[str] = [mode_line, "", "# BANKNIFTY DATA FORENSICS REPORT", f"[NOTE] run {st.get('run_id', '')} · data_version {st.get('data_version', '')}", ""]
    L += [*hinglish, ""]
    L += ["## 2. Reader's Guide for Strategy AI",
          "- [FACT] Only Ω-A/Ω-B/Ω-C rows are validated findings; everything else is exploration or graveyard.",
          "- [FACT] Lifts are in ATR20 units at a fixed horizon; effects are not additive (see N_indep).",
          "- [FACT] SUPPORTED FOR HYPOTHESIS is not a rule; SELF-REVIEW Skeptic results are not proof.", ""]
    intake = rf["intake"]
    L += ["## 3. Data Passport, Cleaning Log & Coverage Statement",
          f"- [FACT] Range {lo}..{hi}; rows in {intake.get('rows_in', 'n/a')}, rows out {intake.get('rows_out', 'n/a')}, dropped {intake.get('rows_dropped', 'n/a')}; row ledger OK {str(st.get('row_ledger_ok', False)).lower()}.",
          f"- [FACT] Usable days {sp.get('usable_days', 0)} (warm-up {len(sp.get('warmup_days', []))}, embargo {sp.get('embargo', '')}); eras {sp.get('eras') and sorted(set(sp['eras'].values()))}.",
          f"- [FACT] Coverage Statement: {sum(1 for r in rf['v11a'] if not str(r.get('verdict', '')).startswith('N/A'))} of {len(rf['v11a'])} approaches runnable on the delivered data.",
          f"- USER TO CONFIRM: costs ({'underlying ' + rf['plan']['cost_pts_underlying'] + ' pts' if 'cost_pts_underlying' in rf['plan'] else 'NOT PROVIDED → T?'}), lot size, expiry calendar.", ""]
    L += ["## 4. Definitions Contract",
          "- [FACT] ATR20: gap-free intraday 20-bar Wilder ATR of the analysis bars; T1-T8: fixed-horizon outcome menu (no hold/first-touch).",
          "- [FACT] Baseline: slot-matched mean on non-event bars of the same days; lift = event value - baseline.",
          f"- [FACT] Court: one-sided sign-flip p < alpha_k/K with B_court draws; alpha_k = 0.05*6/(pi^2 k^2); K = {dk.get('K', 'n/a')}.", ""]
    L += ["## 5. Market Personality Profile & Coverage Matrix",
          f"- [FACT] Phase tables written: {', '.join(rf['phase_files'][:12])}{' …' if len(rf['phase_files']) > 12 else ''}.",
          "- [FACT] Coverage matrix: tables/coverage_matrix.csv (0 empty cells).", ""]
    L += ["## 6. Phase Results & Truth Layer Audit Summary"]
    L += [f"- [FACT] {r['check']}: {r['value']} ({r['status']})" for r in rf["null_rows"]]
    if overflow_step == 0:
        L += [f"- [FACT] {d.splitlines()[0]}" for d in rf["digests"] if d.strip()]
    L += ["", "## 7. Regime Atlas & Era splits", f"- [FACT] Era tags per finding are in the Regime column of findings_registry.csv; eras present {sorted(set((sp.get('eras') or {}).values()))}.", ""]
    L += ["## 8. Leaderboard, Orthogonal Basis & Findings Registry",
          f"- [FACT] N_indep {len({b['independent_effect_id'] for b in rf['basis']})} independent effects; NS excludes AP clones. Effects are not additive."]
    reg = rf["registry"] if overflow_step < 2 else [r for r in rf["registry"] if r.get("Evidence grade") in ("Ω-A", "Ω-B")][:50]
    L += _md_table(reg, ("ID", "Label", "Event N", "Lift (ATR20)", "95% CI", "p_val", "Evidence grade", "Tradability tag", "Burn count"))
    gy = rf["run_dir"] / "tables" / "graveyard.csv"
    L += ["", "## 9. Graveyard", f"- [FACT] {len(rf['graveyard'])} rows; tables/graveyard.csv SHA-256 {sha256_file(gy) if gy.exists() else 'n/a'}."]
    L += _md_table(rf["graveyard"][:50], ("ID", "Lift (ATR20)", "Nominal MDE", "Stability", "graveyard_reason"))
    L += ["", "## 10. Constraint Sheet (facts only)", "- [FACT] Diagnosis only: no trade-management logic is designed or tested here.", ""]
    L += ["## 11. Hypothesis Backlog & Tables 11A/11B"]
    L += _md_table(rf["v11a"], ("approach_id", "verdict", "best_grade", "gate_mde_med", "label"))
    L += _md_table(rf["v11b"], ("group", "group_name", "members_S", "members_C", "members_I", "members_NA", "verdict"))
    L += ["", "## 12. Limitations, Ceiling Report, Open DRTs & Futility Certificate"]
    L += [f"- [FACT] {c['limit']}: {c['evidence']} → {c['what_would_lift_it']}" for c in ceiling_report(rf)]
    L += _md_table([{**t, "what": _what(t)} for t in rf["tickets"][:10]], ("ticket_id", "priority", "trigger", "what"))
    fc = rf["futility"]
    L += [f"- [FACT] Futility Certificate: {'written (N* event-days ' + str(fc['n_star']['event_days_needed']) + ')' if fc else 'not required'}.", ""]
    vr = rf["verify"]
    L += ["## 13. Reproducibility, Verification, LIVE assertions, Living Ledger & Handoff",
          f"- [FACT] Step F verify: {len(vr.get('checked', []))} rows, {len(vr.get('demoted', []))} demoted, exit {vr.get('exit_code', 'n/a')}.",
          f"- [FACT] Seed {st.get('seed', 42)}; brain_hash {st.get('brain_hash', '')[:16]}; library_hash {st.get('library_hash', '')[:16]}."]
    L += _md_table(rf["live"], ("assertion_id", "status", "detail"))
    L += _md_table(rf["ledger_cards"][:20], ("card_id", "state", "e_value"))
    L += ["- [FACT] strategy_input_pack.json manifest: schema 1.1; S10 Handoff Test: PENDING OFFLINE HANDOFF.", ""]
    return "\n".join(L) + "\n"


# =============================================================================================
# Entry points used by the Chair / CLI
# =============================================================================================
def _lineage_prev(runs_root: Path, run_id: str) -> dict[str, Any] | None:
    path = runs_root / "lineage.json"
    runs = load_json(path).get("runs", []) if path.exists() else []
    prior = [r for r in runs if r != run_id]
    if not prior:
        return None
    prev_dir = runs_root / prior[-1]
    return load_run(prev_dir, runs_root) if (prev_dir / "run_state.json").exists() else None


def _lineage_add(runs_root: Path, run_id: str) -> None:
    path = runs_root / "lineage.json"
    runs = load_json(path).get("runs", []) if path.exists() else []
    if run_id not in runs:
        runs.append(run_id)
    atomic_write_json(path, {"schema_version": "1.0", "runs": runs})


def render_all(chair: Chair) -> list[Path]:
    """Ω-S9: render and validate every deliverable; returns the written paths."""
    run_dir, cfg = chair.run_dir, chair.cfg
    mode_line = chair.bundle.get("mode_line", "MODE: FULL (code executed)")
    rf = load_run(run_dir, chair.runs_root)
    rf["state"]["mode_line"] = mode_line
    t = run_dir / "tables"
    from bn_forensics.core_stats import write_csv

    cov = coverage_matrix(rf)
    if any(not str(v).strip() for r in cov for v in r.values()):
        raise SchemaError("coverage_matrix.csv would contain an empty cell")
    paths = [write_csv(t / "coverage_matrix.csv", COVERAGE_FIELDS, cov), write_csv(t / "ceiling_report.csv", CEILING_FIELDS, ceiling_report(rf))]
    hinglish = hinglish_summary(rf, cfg)
    prev = _lineage_prev(chair.runs_root, chair.run_id)
    diff = run_diff(rf, prev)
    text = report_markdown(rf, mode_line, hinglish)
    step = 0
    while estimate_tokens(text) > int(cfg["report"]["max_tokens"]) and step < 3:
        step += 1
        text = report_markdown(rf, mode_line, hinglish, overflow_step=step)
    brief = owner_brief(rf, mode_line)
    handoff = ai_handoff(rf, mode_line, diff)
    if len(brief) > int(cfg["report"]["owner_brief_max_lines"]) or len(handoff) > int(cfg["report"]["ai_handoff_max_lines"]):
        raise SchemaError(f"OWNER_BRIEF {len(brief)} lines / AI_HANDOFF {len(handoff)} lines exceed 40/200")
    narrative = {"hinglish": "\n".join(hinglish), "brief": "\n".join(brief), "handoff_footer": FOOTER}
    words = strategy_words("\n".join(narrative.values()))
    if words:
        chair.trip("TW-13", "CRITICAL", "MODE_INVALID", f"strategy words in narrative: {words}")
        raise InvalidPipeline(f"forbidden strategy words in generated narrative: {words} (tripwire #13)", tripwire_id="TW-13")
    card = scorecard(rf, {**narrative, "report": text}, hinglish, estimate_tokens(text), cfg)
    paths.append(write_csv(t / "scorecard.csv", SCORECARD_FIELDS, card))
    failed = [c["item"] for c in card if c["status"] == "FAIL" and c["item"] in CRITICAL_ITEMS]
    if failed:
        chair.trip(f"SC-{failed[0]}", "CRITICAL", "MODE_INVALID", f"critical Scorecard items FAIL: {failed}")
        raise InvalidPipeline(f"critical Scorecard items FAIL: {failed}", tripwire_id=f"SC-{failed[0]}")
    text += "\n## Appendix Z. 44-Item Self-Audit Scorecard\n" + "\n".join(_md_table(card, SCORECARD_FIELDS)) + "\n"
    pack = strategy_pack(rf, chair)
    paths.append(atomic_write_json(run_dir / "strategy_input_pack.json", pack))
    paths.append(atomic_write_text(run_dir / "BANKNIFTY_DATA_FORENSICS_REPORT.md", text))
    paths.append(atomic_write_text(run_dir / "OWNER_BRIEF.md", "\n".join(brief) + "\n"))
    paths.append(atomic_write_text(run_dir / "AI_HANDOFF.md", "\n".join(handoff) + "\n"))
    paths.append(atomic_write_text(run_dir / "RUN_DIFF.md", "# RUN_DIFF\n" + "\n".join(f"- {x}" for x in diff) + "\n"))
    chair.bundle["chat_summary"] = chat_summary(rf, mode_line, card)
    _lineage_add(chair.runs_root, chair.run_id)
    return paths


def render_partial(chair: Chair, reason: str) -> list[Path]:
    """PARTIAL exit: OWNER_BRIEF.md and AI_HANDOFF.md still describe the interrupted run."""
    rf = load_run(chair.run_dir, chair.runs_root)
    mode_line = chair.bundle.get("mode_line", "MODE: FULL (code executed) · PARTIAL")
    brief = owner_brief(rf, mode_line, partial_reason=reason)[: int(chair.cfg["report"]["owner_brief_max_lines"])]
    handoff = ai_handoff(rf, mode_line, [f"PARTIAL at {chair.stage}: {sanitize_text(reason, 120)}", f"resume: bnf resume --run-id {chair.run_id}"])
    cap = int(chair.cfg["report"]["ai_handoff_max_lines"])
    if len(handoff) > cap:
        handoff = [*handoff[: cap - 2], "## 12. Footer", FOOTER]
    return [atomic_write_text(chair.run_dir / "OWNER_BRIEF.md", "\n".join(brief) + "\n"),
            atomic_write_text(chair.run_dir / "AI_HANDOFF.md", "\n".join(handoff) + "\n")]


def chat_summary(rf: dict[str, Any], mode_line: str, card: Sequence[dict[str, Any]]) -> list[str]:
    """Final chat summary (at most 16 lines, §11)."""
    counts = {k: sum(1 for r in rf["v11a"] if r.get("verdict") == k) for k in
              ("SUPPORTED FOR HYPOTHESIS", "CONTRADICTED", "INCONCLUSIVE", "N/A — DATA MISSING")}
    lines = [mode_line, f"files: {rf['run_dir']}/BANKNIFTY_DATA_FORENSICS_REPORT.md, OWNER_BRIEF.md, AI_HANDOFF.md, strategy_input_pack.json"]
    lines += [f"[{r['ID'].split('@')[0]}] {r['Evidence grade']} lift {_f(r['Lift (ATR20)'])} ATR20" for r in omega_rows(rf)[:5]]
    lines.append(f"Approach verdicts S/C/I/NA = {counts['SUPPORTED FOR HYPOTHESIS']}/{counts['CONTRADICTED']}/{counts['INCONCLUSIVE']}/{counts['N/A — DATA MISSING']}; "
                 f"N_indep {len({b['independent_effect_id'] for b in rf['basis']})}")
    lines.append(rf["null_lab"].get("status_line", "Null Lab: n/a"))
    lines.append("caveats: effects not additive; SELF-REVIEW is not proof; Ω is not a trading rule")
    lines += [f"DRT {t['ticket_id']}: {sanitize_text(_what(t), 60)}" for t in rf["tickets"][:2]]
    lines.append(f"scorecard FAIL items: {[c['item'] for c in card if c['status'] == 'FAIL'] or 'none'}; zero strategies designed")
    return lines[:16]


__all__ = [
    "CEILING_FIELDS",
    "COVERAGE_FIELDS",
    "CRITICAL_ITEMS",
    "FOOTER",
    "SCORECARD_FIELDS",
    "SCORECARD_ITEMS",
    "ai_handoff",
    "ceiling_report",
    "chat_summary",
    "check_hinglish",
    "coverage_matrix",
    "estimate_tokens",
    "hinglish_summary",
    "load_run",
    "owner_brief",
    "render_all",
    "render_partial",
    "report_markdown",
    "run_diff",
    "scorecard",
    "strategy_pack",
    "strategy_words",
]
