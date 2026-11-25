"""Unified ``bnf`` CLI (FILE-024, FR-19, §10.3).

``bnf <intake|null-lab|playground|court|ledger-tick|report|resume|run|selftest> [flags]``

Exit codes: 0 OK, 2 PARTIAL (resumable, prints ``bnf resume --run-id <id>``), 3 input/schema/CLI
usage error (argparse's default 2 is overridden, X14), 4 ``MODE: INVALID``, 5 COMPLETE_WITH_GAPS.

Every run starts with ``UPSTOX_RO_TOKEN`` and all ``*TOKEN*``/``*SECRET*`` variables unset (NFR-05).
The isolated Skeptic (FILE-034) and Step F verifier (FILE-037) are launched from here as
``python3 -m`` subprocesses with the same scrubbed environment (§8.1 Ω-S6).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bn_forensics import (
    EXIT_GAPS,
    EXIT_INPUT,
    EXIT_INVALID,
    EXIT_OK,
    IST,
    BnfError,
    InputError,
    asset_dir,
    load_json,
    load_yaml,
    scrub_secret_env,
    set_fixed_clock,
    validate_run_id,
)

ISOLATED_MODULES = ("bn_forensics.skeptic_checks", "bn_forensics.verify_checks", "bn_forensics.twin_impl")
SUBCOMMANDS = ("intake", "null-lab", "playground", "court", "ledger-tick", "report", "resume", "run", "selftest")
UNTIL = {"intake": "Ω-S0", "null-lab": "Ω-S2", "playground": "Ω-S3", "court": "Ω-S6", "run": "Ω-S9", "resume": "Ω-S9", "selftest": "Ω-S9"}
SELFTEST_CLOCK = dt.datetime(2029, 1, 1, 18, 0, 0, tzinfo=IST)


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 3, never argparse's default 2, so a typo never looks like a PARTIAL (X14)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        raise SystemExit(EXIT_INPUT)


def run_isolated(module: str, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run FILE-034 / FILE-037 / FILE-036 as an isolated ``python3 -m`` subprocess (scrubbed env)."""
    if module not in ISOLATED_MODULES:
        raise InputError(f"module {module!r} is not an isolated FILE-034/036/037 entry point")
    env = dict(os.environ)
    scrub_secret_env(env)
    return subprocess.run([sys.executable, "-m", module, *[str(a) for a in args]], capture_output=True, text=True, env=env, check=False)


def _bool(text: str) -> bool:
    if str(text).lower() in ("1", "true", "yes", "on"):
        return True
    if str(text).lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


def _run_id(text: str) -> str:
    try:
        return validate_run_id(text)
    except InputError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> _Parser:
    common = _Parser(add_help=False)
    common.add_argument("--config", default="config/default_config.yaml")
    common.add_argument("--store-dir", default=os.environ.get("BNF_STORE_DIR", "./store"))
    common.add_argument("--run-dir", default="./runs", help="runs root (one sub-directory per run_id)")
    common.add_argument("--seed", type=int, default=int(os.environ.get("BNF_SEED", "42")))
    common.add_argument("--profile", choices=("full", "lite"), default="full")
    common.add_argument("--json-logs", action="store_true")
    pipeline = _Parser(add_help=False)
    pipeline.add_argument("--incoming-dir", default="./incoming")
    pipeline.add_argument("--sample-first", type=_bool, default=True)
    pipeline.add_argument("--run-id", type=_run_id, default=None)
    budget = _Parser(add_help=False)
    budget.add_argument("--max-ideas", type=int, default=0)
    budget.add_argument("--cards-dir", default=None)
    budget.add_argument("--tokens-used", type=int, default=0)
    budget.add_argument("--cost-usd", type=float, default=0.0)
    budget.add_argument("--max-wall-min", type=float, default=240.0)
    parser = _Parser(prog="bnf", description="Bank Nifty Deep Data Forensics Lab (diagnosis only)")
    sub = parser.add_subparsers(dest="command", parser_class=_Parser)
    sub.add_parser("intake", parents=[common, pipeline], help="Ω-S0 Quartermaster intake")
    sub.add_parser("null-lab", parents=[common, pipeline], help="Ω-S0..S2 Null Lab audit")
    sub.add_parser("playground", parents=[common, pipeline, budget], help="Ω-S0..S3 X-ray + Playground")
    court = sub.add_parser("court", parents=[common, pipeline, budget], help="Ω-S0..S6 docket, Court, Skeptic, verify")
    court.add_argument("--k-docket", type=int, default=None)
    run = sub.add_parser("run", parents=[common, pipeline, budget], help="full pipeline Ω-S0..S9")
    run.add_argument("--k-docket", type=int, default=None)
    for name in ("ledger-tick", "report", "resume"):
        p = sub.add_parser(name, parents=[common, budget] if name == "resume" else [common])
        p.add_argument("--run-id", type=_run_id, default="latest")
    sub.add_parser("selftest", parents=[common], help="offline end-to-end acceptance on synthetic worlds (T-16)")
    return parser


def _echo(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _options(args: argparse.Namespace, command: str, **extra: Any) -> Any:
    from bn_forensics.chair import RunOptions

    k_docket = getattr(args, "k_docket", None)
    if k_docket is not None and not 1 <= int(k_docket) <= 50:
        raise InputError("--k-docket must be within 1..50")
    return RunOptions(
        command=command, config=args.config, store_dir=Path(args.store_dir), runs_root=Path(args.run_dir),
        incoming_dir=Path(getattr(args, "incoming_dir", "") or ".") if getattr(args, "incoming_dir", None) else None,
        run_id=getattr(args, "run_id", None), seed=int(args.seed), profile=args.profile, max_ideas=int(getattr(args, "max_ideas", 0) or 0),
        max_wall_min=float(getattr(args, "max_wall_min", 240.0)), k_docket=k_docket,
        cards_dir=Path(args.cards_dir) if getattr(args, "cards_dir", None) else None, tokens_used=int(getattr(args, "tokens_used", 0) or 0),
        cost_usd=float(getattr(args, "cost_usd", 0.0) or 0.0), sample_first=bool(getattr(args, "sample_first", True)),
        json_logs=bool(args.json_logs), until=UNTIL.get(command, "Ω-S9"), echo=_echo, **extra)


def _finish(outcome: Any) -> int:
    chat = outcome.summary and outcome.summary.get("chat")
    for line in chat or []:
        _echo(line)
    if outcome.exit_code == 2:
        _echo(f"bnf resume --run-id {outcome.run_id}")
    return int(outcome.exit_code)


def cmd_pipeline(args: argparse.Namespace) -> int:
    from bn_forensics.chair import Chair

    opts = _options(args, args.command)
    if args.command == "resume":
        opts = _resume_options(args, opts)
    chair = Chair(opts)
    outcome = chair.run()
    outcome.summary["chat"] = chair.bundle.get("chat_summary", [])
    return _finish(outcome)


def _resume_options(args: argparse.Namespace, opts: Any) -> Any:
    from bn_forensics.chair import Chair, brain_hash, library_hash

    probe = Chair(opts)
    state_path = probe.run_dir / "run_state.json"
    if not state_path.exists():
        raise InputError(f"no run_state.json for run {probe.run_id}")
    state = load_json(state_path)
    if state["brain_hash"] != brain_hash(opts.prompts_dir) or state["library_hash"] != library_hash():
        from bn_forensics import InvalidPipeline

        raise InvalidPipeline("brain_hash/library_hash changed since the PARTIAL checkpoint (tripwire #2)", tripwire_id="TW-2")
    paths = state.get("paths", {})
    opts.run_id = probe.run_id
    opts.incoming_dir = Path(paths["incoming"]) if paths.get("incoming") else None
    opts.store_dir = Path(paths.get("store") or opts.store_dir)
    opts.cards_dir = Path(paths["cards_dir"]) if paths.get("cards_dir") else opts.cards_dir
    if "Ω-S5" in state.get("stages_completed", []):
        opts.command = "report"
    return opts


# =============================================================================================
# selftest (T-16)
# =============================================================================================
def cmd_selftest(args: argparse.Namespace) -> int:
    from bn_forensics import quartermaster as qm
    from bn_forensics.chair import Chair, csv_digest, make_selftest_world, reset_workspace

    set_fixed_clock(SELFTEST_CLOCK)
    failures: list[str] = []
    try:
        ws = Path(args.run_dir) / "selftest_ws"
        reset_workspace(ws)
        cal = qm.load_era_calendar()
        cfg_path = args.config
        # (1) fixture CSV through the Sample-First CP0 gate (identity echo-back, row conservation)
        fx_in = ws / "fixture_incoming"
        fx_in.mkdir(parents=True)
        shutil.copyfile(asset_dir("fixtures") / "sample_spot_3d.csv", fx_in / "sample_spot_3d.csv")
        from bn_forensics import load_config

        cfg = load_config(cfg_path)
        fx = qm.run_intake(fx_in, ws / "fixture_store", cfg=cfg, cal=cal, run_dir=ws / "fixture_run")
        _echo(next((ln for ln in fx.echo_lines if ln.startswith("Received:")), fx.echo_lines[0] if fx.echo_lines else "Received: n/a"))
        if not fx.row_ledger_ok:
            failures.append("fixture row conservation")
        # (2) the 450-day synthetic world with a planted +0.8 ATR20 edge, a leaky brick and no option OI
        world = make_selftest_world(ws / "world", cal, seed=int(args.seed))
        leaky = dict(world.card, card_id="C-0002", hypothesis="Leaky card that peeks at the next close", event="gt(close_{t+1}, 0)",
                     expected_sign="none")
        common = {"features": world.provider(), "ai_fallback": {"H1": [world.card, leaky]},
                  "cfg_overrides": {"cost_model": {"underlying_cost_pts": 2.0}, "targets": {"mde_target_atr20": 0.25}}}
        args.store_dir, args.run_dir, args.incoming_dir = str(ws / "store"), str(ws / "runs"), str(world.incoming)
        args.run_id, args.sample_first = "selftest", True
        chair = Chair(_options(args, "selftest", **common))
        out = chair.run()
        failures += _assert_selftest(chair, out, expect_grade="Ω-A")
        digest = csv_digest(chair.run_dir)
        # (3) the same world with its dates listed in the Exposure Calendar -> capped at Ω-B (R28)
        args.store_dir, args.run_dir, args.run_id = str(ws / "store_exposed"), str(ws / "runs_exposed"), "selftest_exposed"
        exposure = {"exposure_calendar": {"ranges": [{"from": world.days[0].isoformat(), "to": world.days[-1].isoformat(),
                                                      "label": "selftest exposure variant", "source": "selftest"}]}}
        chair_b = Chair(_options(args, "selftest", **common, cal_overrides=exposure, null_worlds=100, recall_worlds=20))
        out_b = chair_b.run()
        failures += [f"exposed variant: {f}" for f in _assert_selftest(chair_b, out_b, expect_grade="Ω-B")]
        for line in chair.bundle.get("chat_summary", [])[:16]:
            _echo(line)
        _echo(f"SELFTEST_CSV_SHA256 {digest}")
    finally:
        set_fixed_clock(None)
    if failures:
        _echo("SELFTEST FAIL: " + "; ".join(failures))
        return EXIT_INVALID
    _echo("SELFTEST PASS: planted edge COURT-CONFIRMED and graded Ω-A (VIRGIN) / Ω-B (EXPOSED); verify PASS; pack valid")
    return EXIT_OK


def _assert_selftest(chair: Any, out: Any, *, expect_grade: str) -> list[str]:
    import csv

    fails: list[str] = []
    if out.exit_code not in (EXIT_OK, EXIT_GAPS):
        return [f"pipeline exit {out.exit_code}: {out.mode_line}"]
    run_dir = Path(out.run_dir)
    state = load_json(run_dir / "run_state.json")
    if not state.get("row_ledger_ok"):
        fails.append("row_ledger_ok false")
    nl = load_json(run_dir / "null_lab.json")
    if nl["canary_caught"] != 1.0 or nl["fpr_court"] > 0.05 or nl["verdict"] != "PASS":
        fails.append(f"Null Lab {nl['verdict']} canary {nl['canary_caught']} fpr {nl['fpr_court']}")
    if chair.bundle["seal"].court_reads_during_playground():
        fails.append("Court files read during Playground (Ω-4)")
    with open(run_dir / "tables" / "court_verdicts.csv", encoding="utf-8", newline="") as fh:
        verdicts = {r["card_id"]: r for r in csv.DictReader(fh)}
    planted = [cid for cid, c in chair.bundle["compiled"].items()
               if "feature3" in str(c.card.get("event", "")) and "feature7" in str(c.card.get("event", "")) and cid in verdicts]
    confirmed = [cid for cid in planted if verdicts[cid]["verdict"] == "COURT-CONFIRMED"]
    if not confirmed:
        fails.append(f"planted edge not COURT-CONFIRMED (docket planted cards: {planted})")
    grades = {r["ID"].split("@")[0]: r["Evidence grade"] for r in chair.bundle.get("registry", [])}
    if not any(grades.get(cid) == expect_grade for cid in confirmed):
        fails.append(f"planted edge grade {[grades.get(c) for c in confirmed]} != {expect_grade}")
    vlog = load_json(run_dir / "tables" / "verification_log.json")
    if not vlog["verify_pass"]:
        fails.append(f"verify_checks not PASS: {vlog['status_line']}")
    pack = load_json(run_dir / "strategy_input_pack.json")
    from bn_forensics.schemas_validator import validate

    if not validate(pack, "strategy_input_pack").ok:
        fails.append("strategy_input_pack.json invalid")
    finding = [f for f in pack["validated_findings"] if f["card_id"] in confirmed and f["grade"] == expect_grade]
    if not finding or finding[0]["deflated_lift"] is None or not finding[0]["must_carry_caveats"]:
        fails.append("pack lacks the planted finding with deflated_lift + must_carry_caveats")
    with open(run_dir / "tables" / "scorecard.csv", encoding="utf-8", newline="") as fh:
        critical = [r for r in csv.DictReader(fh) if int(r["item"]) in (3, 5, 12, 13, 35, 40) and r["status"] == "FAIL"]
    if critical:
        fails.append(f"critical scorecard FAIL {[r['item'] for r in critical]}")
    ledger_lines = (run_dir / "idea_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    if not any('"C-0002"' in ln and "LEAK-REJECTED" in ln for ln in ledger_lines):
        fails.append("leaky close_{t+1} card was not LEAK-REJECTED")
    tickets = (load_yaml(run_dir / "tickets" / "open_tickets.yaml") or {}).get("tickets", [])
    if not any(t["trigger"] == "BLOCKED_CARD" for t in tickets):
        fails.append("no BLOCKED_CARD DRT for the missing option OI")
    for name, cap in (("OWNER_BRIEF.md", 40), ("AI_HANDOFF.md", 200)):
        if len((run_dir / name).read_text(encoding="utf-8").splitlines()) > cap:
            fails.append(f"{name} longer than {cap} lines")
    return fails


# =============================================================================================
# main
# =============================================================================================
def main(argv: Sequence[str] | None = None) -> int:
    scrub_secret_env()
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command is None:
        parser.print_usage(sys.stderr)
        sys.stderr.write("bnf: error: a subcommand is required\n")
        return EXIT_INPUT
    try:
        if args.command == "selftest":
            return cmd_selftest(args)
        return cmd_pipeline(args)
    except BnfError as exc:
        code = int(exc.exit_code)
        prefix = f"MODE: INVALID — tripwire {exc.tripwire_id}: " if code == EXIT_INVALID else f"error (exit {code}): "
        sys.stderr.write(prefix + str(exc) + "\n")
        return code
    except KeyboardInterrupt:
        sys.stderr.write("interrupted (no PARTIAL marker written; state files are atomic)\n")
        return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ISOLATED_MODULES", "SUBCOMMANDS", "build_parser", "cmd_pipeline", "cmd_selftest", "main", "run_isolated"]
