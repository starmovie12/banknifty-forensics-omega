"""Code-side JSON/YAML schema validator (FILE-040, FR-18).

Every AI-generated artifact (Idea Cards v2, DRTs, proposals, digests, ``next_actions.yaml``) and
every machine-readable pipeline artifact is validated against the versioned schemas in
``schemas/`` before downstream use. Failures carry the exact field path and block ingestion
(:class:`bn_forensics.SchemaError`, exit 3).

``self_check_schemas()`` is the ST-1 exit gate. It checks that all 6 schema documents are valid
Draft 2020-12, carry ``schema_version``, accept the PRD §10 example instances and reject known-bad
mutations (Lane M card without ``dose_variable``, DRT without ``need.other``).
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JsonSchemaDocumentError

from bn_forensics import SchemaError, asset_dir, load_json, load_yaml, sha256_text

SCHEMA_FILES: dict[str, str] = {
    "idea_card_v2": "idea_card_v2.schema.json",
    "court_docket": "court_docket.schema.json",
    "drt": "drt.schema.json",
    "run_state": "run_state.schema.json",
    "futility_certificate": "futility_certificate.schema.json",
    "strategy_input_pack": "strategy_input_pack.schema.json",
}
# Named sub-contracts stored as $defs inside the 6 schema files (DEF-08).
SUB_CONTRACTS: dict[str, tuple[str, str]] = {
    "proposal": ("idea_card_v2", "proposal"),
    "idea_ledger_record": ("idea_card_v2", "idea_ledger_record"),
    "next_actions": ("futility_certificate", "next_actions"),
    "ticket_file": ("drt", "ticket_file"),
}
SCHEMA_VERSIONS: dict[str, str] = {
    "idea_card_v2": "2.0",
    "court_docket": "1.0",
    "drt": "1.0",
    "run_state": "1.0",
    "futility_certificate": "1.0",
    "strategy_input_pack": "1.0",
}

ID_LINK_RE = re.compile(r"\[(C-[0-9]{3,6}|AP[0-9]{2}(?:-[A-Z0-9]+)?|F-[0-9]{3,6}|[A-Z][A-Za-z0-9_.-]{1,40})\]")
_INJECTION_RE = re.compile(
    r"(ignore (all |any )?(previous|prior|above) (instructions|rules)|disregard (the )?(rules|instructions)"
    r"|you are now|system prompt|<\s*/?\s*system\s*>|act as|new instructions|override (the )?rules"
    r"|execute (this|the following)|run this command|rm -rf|import os|__import__|eval\(|exec\()",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of one validation: ``ok`` plus ``(json_path, message)`` errors."""

    schema: str
    ok: bool
    errors: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def summary(self) -> str:
        if self.ok:
            return f"{self.schema}: PASS"
        lines = [f"{self.schema}: FAIL ({len(self.errors)} error(s))"]
        lines.extend(f"  at {path}: {msg}" for path, msg in self.errors)
        return "\n".join(lines)


def schema_dir() -> Path:
    """Directory holding the 6 ``*.schema.json`` files."""
    return asset_dir("schemas")


@cache
def load_schema(name: str) -> dict[str, Any]:
    """Load and Draft-2020-12-check one schema document by short name."""
    if name not in SCHEMA_FILES:
        raise SchemaError(f"unknown schema {name!r}; known: {sorted(SCHEMA_FILES)}")
    path = schema_dir() / SCHEMA_FILES[name]
    doc = load_json(path)
    try:
        Draft202012Validator.check_schema(doc)
    except JsonSchemaDocumentError as exc:
        raise SchemaError(f"schema document {path.name} is not valid Draft 2020-12: {exc.message}") from exc
    return doc


def _validator_for(name: str) -> tuple[Draft202012Validator, str]:
    if name in SUB_CONTRACTS:
        parent, definition = SUB_CONTRACTS[name]
        doc = load_schema(parent)
        if definition not in doc.get("$defs", {}):
            raise SchemaError(f"$defs.{definition} missing in {SCHEMA_FILES[parent]}")
        wrapper = {key: value for key, value in doc.items()
                   if key in ("$schema", "$id", "$defs")}
        wrapper["$ref"] = f"#/$defs/{definition}"
        return Draft202012Validator(wrapper), name
    return Draft202012Validator(load_schema(name)), name


def _json_path(error: Any) -> str:
    parts = ["$"]
    for item in error.absolute_path:
        parts.append(f"[{item}]" if isinstance(item, int) else f".{item}")
    return "".join(parts)


def validate(instance: Any, schema_name: str) -> ValidationResult:
    """Validate ``instance`` against a schema or named sub-contract; never raises on content."""
    validator, label = _validator_for(schema_name)
    errors = sorted(validator.iter_errors(instance), key=lambda e: (list(map(str, e.absolute_path)), e.message))
    collected = tuple((_json_path(err), err.message) for err in errors)
    return ValidationResult(schema=label, ok=not collected, errors=collected)


def validate_or_raise(instance: Any, schema_name: str, *, what: str = "") -> None:
    """Validate and raise :class:`SchemaError` (exit 3) with exact field paths on failure."""
    result = validate(instance, schema_name)
    if not result.ok:
        label = f"{what}: " if what else ""
        raise SchemaError(label + result.summary())


def load_artifact(path: str | Path) -> Any:
    """Load a YAML or JSON artifact with safe loaders (text inside is data only, R26)."""
    target = Path(path)
    if target.suffix.lower() == ".json":
        return load_json(target)
    if target.suffix.lower() in (".yaml", ".yml"):
        return load_yaml(target)
    raise SchemaError(f"unsupported artifact type: {target.name}")


def validate_file(path: str | Path, schema_name: str) -> ValidationResult:
    """Load an artifact file and validate it."""
    return validate(load_artifact(path), schema_name)


def validate_digest(text: str, max_lines: int = 8) -> list[str]:
    """Digest Contract (F4): at most ``max_lines`` non-empty lines. Returns problems."""
    lines = [line for line in text.splitlines() if line.strip()]
    problems: list[str] = []
    if len(lines) > max_lines:
        problems.append(f"digest has {len(lines)} lines > {max_lines}")
    if not lines:
        problems.append("digest is empty")
    return problems


def unlinked_factual_lines(lines: list[str]) -> list[str]:
    """F23 gate: factual Hinglish lines must cite a Finding/Card/Approach ID like ``[C-0031]``.

    Headings (``#``), blank lines and lines explicitly marked ``[NOTE]`` carry no facts.
    """
    bad: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("[NOTE]") or set(line) <= {"-", "|", " "}:
            continue
        if not ID_LINK_RE.search(line):
            bad.append(line)
    return bad


def scan_injection(text: str) -> list[str]:
    """Instruction-like strings inside data (F28). They are reported, never obeyed."""
    return [m.group(0) for m in _INJECTION_RE.finditer(text)]


def sanitize_text(value: Any, max_len: int = 500) -> str:
    """Render untrusted data text as inert, bounded, control-free text for reports."""
    text = _CONTROL_RE.sub("", str(value)).replace("\r", " ")
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


# ---------------------------------------------------------------------------------------------
# ST-1 self check
# ---------------------------------------------------------------------------------------------
def _h(label: str) -> str:
    return sha256_text(label)


def example_instances() -> dict[str, Any]:
    """Valid example instances built from the PRD §10.2 / §10.5 examples (placeholders hashed)."""
    card = {
        "schema_version": "2.0", "card_id": "C-0031", "type": "RULE", "lane": "M",
        "provenance": "PERSONA:options-greeks", "round": 1,
        "hypothesis": "Near-expiry close within 0.5 ATR20 of max-OI wall increases 5-bar stall rate",
        "event": "AND(lt(oi_wall_dist_call, 0.5), lte(dte, 1))", "side": "both", "target": "T3",
        "horizons": [3, 5, 10], "baseline": "slot_matched", "cutoffs": "discovery_frozen",
        "data_needed": ["chain_metrics_1m", "expiry_calendar"], "cost_series": "underlying",
        "mechanism": {
            "actors": "expiry gamma hedgers", "dose_variable": "oi_wall_dist_call",
            "expected_gradient": "smaller distance -> higher stall rate",
            "negative_control": "non-expiry days (dte >= 4), effect ~ 0",
            "falsifier": "flat dose gradient or equal effect on non-expiry days",
        },
        "prior_weight": 1.5, "expected_sign": "positive", "frozen_at": "2026-11-22T15:50:00+05:30",
        "card_hash": _h("card"), "data_version": _h("data"),
    }
    docket = {
        "schema_version": "1.0", "session": 1, "alpha_k": 0.030396, "K": 20, "jaccard_max": 0.5,
        "frozen_at": "2026-11-22T16:00:00+05:30", "data_version": _h("data"),
        "court_days_hash": _h("days"), "B_court": 10000,
        "power_precheck": {"median_event_days": 49, "projected_mde_sd": 0.544, "ceiling_sd": 0.60,
                           "passed": True},
        "exposure": {"virgin_days": 0, "exposed_days": 196},
        "cards": [{"card_id": "C-0412", "card_hash": _h("c412"), "sign": 1,
                   "jaccard_max_vs_chosen": 0.31, "lane": "M"}],
    }
    drt = {
        "schema_version": "1.0", "ticket_id": "D-0004", "raised_by": "Quartermaster",
        "trigger": "MECHANISM_DOSE",
        "why": "Card C-0031 requires multi-strike option OI snapshots and an explicit expiry calendar",
        "need": {"what": "multi-strike option OI snapshots",
                 "instruments": ["BANKNIFTY options, ATM +/- 10 strikes"], "interval": "1m",
                 "from": "2026-09-01", "to": "2026-11-20",
                 "fields": ["oi", "ltp", "bid", "ask", "iv_if_available"], "session": "regular",
                 "adjusted": "n/a", "other": ""},
        "priority": "HIGH", "expected_gain": {"mde_ratio": "n/a", "cells_unlocked": 6, "cards_unblocked": 3},
        "effort": "unknown", "acceptable_substitutes": ["5m interval", "ATM +/- 5 strikes"],
        "capability_questions": ["How many months of expired option-chain OI history can your source fetch?"],
        "sample_first": True, "sample_spec": "2 trading days inside 1 expiry week",
        "delivery_format": "canonical_store",
        "acceptance_checks": ["schema", "gaps", "row_conservation", "ts_convention", "reconciliation_if_overlap"],
        "if_not_delivered": "C-0031 stays N/A — DATA MISSING; rest of run continues",
        "adaptive_data": False, "state": "OPEN",
    }
    run_state = {
        "schema_version": "1.0", "stage": "Ω-S7", "prompt_version": "V5.0+Omega1.0",
        "model_id": "host-agent", "brain_hash": _h("brain"), "prompts_hash": _h("prompts"),
        "library_hash": _h("library"), "data_version": _h("data"), "mode": "STANDARD",
        "court_session_k": 1, "m_playground_total": 12400,
        "budget": {"max_wall_min": 240, "max_ideas": 20000, "used_wall_min": 0, "used_ideas": 0},
        "env": {"python": "3.13.18", "profile": "full", "lock_mode": "flock"},
        "open_tickets": ["D-0001", "D-0004"], "row_ledger_ok": True, "tripwires": [],
        "digests": ["digests/S0.md", "digests/S2.md", "digests/S5.md"],
    }
    futility = {
        "schema_version": "1.0", "data_version": _h("data"),
        "usable_days": {"playground": 294, "court": 196, "court_virgin": 0, "court_exposed": 196,
                        "era_A": 0, "era_B": 490},
        "search": {"m_playground": 12400, "clusters": 318,
                   "modes_saturated": [f"H{i}" for i in range(1, 14)], "modes_open": []},
        "court_mde": {"K": 20, "alpha_k": 0.030396, "n_events_median": 49, "mde_sd": 0.544,
                      "mde_atr20": 0.524, "outcome_sd_atr20": 0.963},
        "null_lab": {"fpr_court": 0.023, "fpr_ledger": 0.0, "fpr_union": 0.023,
                     "recall_at_delta": {"0.1": 0.01, "0.2": 0.04, "0.3": 0.09, "0.5": 0.25, "0.8": 0.93}},
        "n_star": {"mde_target_atr20": 0.25, "mde_target_sd": 0.260, "event_days_needed": 215,
                   "court_days_needed": 860},
        "cheapest_next_data": [{"ticket": "D-0001", "expected_n_star_reduction": "58%"}],
        "ledger": {"fwd_watch": 20, "confirmed": 0, "decayed": 0, "registered_lifetime": 20,
                   "M": 20000, "next_tick": "2026-12-07"},
        "must_carry_caveats": ["Futility != no edge exists",
                               "Re-measure synthetic [SIM] numbers via Null Lab at real n",
                               "EXPOSED-COURT: these Court days were seen by earlier research (R28)"],
    }
    next_actions = {
        "schema_version": "1.0",
        "actions": [
            {"id": "A-001", "type": "HUNT", "mode": "H6", "cost": "low", "expected_gain": "new clusters", "blocked": False},
            {"id": "A-002", "type": "DATA", "ticket": "D-0007", "cost": "medium", "expected_gain": "N* -35%", "blocked": False},
            {"id": "A-003", "type": "WAIT", "until": "next 10-day tick", "expected_gain": "ledger update", "blocked": False},
        ],
    }
    pack = {
        "schema_version": "1.1",
        "metadata": {"data_version": _h("data"), "run_id": "run_example", "mode": "MODE: FULL (code executed)",
                     "date_range": {"from": "2024-11-21", "to": "2026-11-20"}, "usable_days": 490,
                     "diagnosis_only": True},
        "validated_findings": [{
            "finding_id": "F-0001", "card_id": "C-0412", "grade": "Ω-B",
            "statement": "[C-0412] fixed-horizon forward return lift after the event",
            "target": "T1", "horizon": 5, "side": "continuation", "deflated_lift": 0.21,
            "lift_atr20": 0.30, "effective_n": 49,
            "court": {"verdict": "COURT-CONFIRMED", "p": 0.0004, "alpha_k_over_K": 0.0015198,
                      "B_court": 10000, "exposure": "EXPOSED-COURT", "session_k": 1},
            "ledger_state": "FWD-WATCH", "tradability_tag": "T?", "cost_series": "option_buyer",
            "independent_effect_id": "E-01", "must_carry_caveats": ["EXPOSED-COURT (R28)"],
        }],
        "orthogonal_independent_effects": [{"effect_id": "E-01", "member_finding_ids": ["F-0001"]}],
        "break_even_costs": {"underlying_cost_pts": None},
        "approach_verdicts_11a": [{"approach_id": "AP14", "verdict": "INCONCLUSIVE"}],
        "group_verdicts_11b": [{"group": "G6", "verdict": "INCONCLUSIVE"}],
        "graveyard_summary": {}, "hypothesis_backlog": [], "ledger_states": [],
        "open_data_tickets": [], "futility_certificate": None,
        "must_carry_caveats": ["Diagnosis only; Ω != trading rule; effects are not additive"],
        "exposure_summary": {"virgin_days": 0, "exposed_days": 196, "policy": "cap",
                             "per_finding": [{"finding_id": "F-0001", "exposure": "EXPOSED-COURT"}]},
        "cost_model": {"parameters": {"brokerage_per_order": "30"},
                       "worked_break_even": [{"premium": 400, "lots": 1, "lot_size": 30,
                                              "option_cost_pts": 5.71, "underlying_cost_pts": 11.42}]},
        "backtest_bridge": [{"finding_id": "F-0001", "tag": "f_c0412", "available_at": "bar close",
                             "cost_series": "option_buyer", "break_even_pts": 11.42,
                             "deflated_lift": 0.21, "exposure": "EXPOSED-COURT", "grade": "Ω-B",
                             "must_carry_caveats": ["EXPOSED-COURT (R28)"]}],
    }
    proposal = {"schema_version": "1.0", "proposal_type": "BRICK", "brick": "dist_prev_week_high_atr",
                "formula": "(close - prev_week_high) / ATR20", "available_at": "bar close",
                "data_needed": ["spot_1m"], "why": "Weekly level reaction", "leak_gate": "pending"}
    ledger_record = {"schema_version": "1.0", "m_lifetime": 1, "card_id": "C-0001",
                     "card_hash": _h("c1"), "hunt_mode": "H1", "status": "EVALUATED",
                     "event_days_hash": _h("days1"), "event_days_count": 64, "deflated_lift": 0.12,
                     "timestamp_ist": "2026-11-22T16:05:00+05:30"}
    return {
        "idea_card_v2": card, "court_docket": docket, "drt": drt, "run_state": run_state,
        "futility_certificate": futility, "next_actions": next_actions,
        "strategy_input_pack": pack, "proposal": proposal, "idea_ledger_record": ledger_record,
        "ticket_file": {"schema_version": "1.0", "tickets": [drt]},
    }


def negative_instances() -> list[tuple[str, str, Any]]:
    """Known-bad mutations that MUST fail (T-01, FR-18 acceptance)."""
    good = example_instances()
    cases: list[tuple[str, str, Any]] = []

    card = copy.deepcopy(good["idea_card_v2"])
    del card["mechanism"]["dose_variable"]
    cases.append(("lane M card missing mechanism.dose_variable", "idea_card_v2", card))
    card = copy.deepcopy(good["idea_card_v2"])
    del card["mechanism"]
    cases.append(("lane M card without mechanism", "idea_card_v2", card))
    card = copy.deepcopy(good["idea_card_v2"])
    card["schema_version"] = "1.0"
    cases.append(("idea card wrong schema_version", "idea_card_v2", card))
    drt = copy.deepcopy(good["drt"])
    del drt["need"]["other"]
    cases.append(("DRT missing need.other", "drt", drt))
    docket = copy.deepcopy(good["court_docket"])
    docket["cards"][0]["jaccard_max_vs_chosen"] = 0.85
    cases.append(("docket card above Jaccard 0.5", "court_docket", docket))
    state = copy.deepcopy(good["run_state"])
    del state["brain_hash"]
    cases.append(("run_state without brain_hash", "run_state", state))
    fut = copy.deepcopy(good["futility_certificate"])
    fut["must_carry_caveats"] = ["no caveat"]
    cases.append(("futility certificate without futility caveat", "futility_certificate", fut))
    actions = copy.deepcopy(good["next_actions"])
    actions["actions"] = actions["actions"][:2]
    cases.append(("next_actions with < 3 actions", "next_actions", actions))
    pack = copy.deepcopy(good["strategy_input_pack"])
    pack["validated_findings"][0]["grade"] = "COURT-UNDERPOWERED"
    cases.append(("pack with non-Ω finding", "strategy_input_pack", pack))
    pack = copy.deepcopy(good["strategy_input_pack"])
    del pack["backtest_bridge"]
    cases.append(("pack 1.1 without backtest_bridge", "strategy_input_pack", pack))
    pack = copy.deepcopy(good["strategy_input_pack"])
    pack["validated_findings"][0]["court"]["verdict"] = "COURT-UNDERPOWERED"
    cases.append(("pack Ω-B finding from underpowered Court", "strategy_input_pack", pack))
    return cases


def self_check_schemas() -> dict[str, Any]:
    """ST-1 gate: all 6 schemas valid, versioned, accept PRD examples, reject bad mutations.

    Returns a summary dict; raises :class:`SchemaError` on any failure.
    """
    failures: list[str] = []
    for name, filename in sorted(SCHEMA_FILES.items()):
        doc = load_schema(name)
        if doc.get("schema_version") != SCHEMA_VERSIONS[name]:
            failures.append(f"{filename}: schema_version {doc.get('schema_version')!r} != {SCHEMA_VERSIONS[name]!r}")
    positives = example_instances()
    for name, instance in sorted(positives.items()):
        result = validate(instance, name)
        if not result.ok:
            failures.append("example rejected: " + result.summary())
    negatives = negative_instances()
    for label, name, instance in negatives:
        if validate(instance, name).ok:
            failures.append(f"bad instance accepted: {label}")
    if failures:
        raise SchemaError("self_check_schemas FAILED:\n" + "\n".join(failures))
    return {
        "schemas": len(SCHEMA_FILES),
        "positive_examples": len(positives),
        "negative_examples": len(negatives),
        "status": "PASS",
    }


def describe(instance: Mapping[str, Any]) -> str:
    """Short deterministic description of an artifact for logs."""
    return json.dumps({k: instance[k] for k in sorted(instance)[:4]}, sort_keys=True, default=str)[:200]


__all__ = [
    "ID_LINK_RE", "SCHEMA_FILES", "SCHEMA_VERSIONS", "SUB_CONTRACTS", "ValidationResult",
    "describe", "example_instances", "load_artifact", "load_schema", "negative_instances",
    "sanitize_text", "scan_injection", "schema_dir", "self_check_schemas",
    "unlinked_factual_lines", "validate", "validate_digest", "validate_file", "validate_or_raise",
]
