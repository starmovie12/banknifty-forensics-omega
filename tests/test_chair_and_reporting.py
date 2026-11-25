"""Chair, Reporter, prompts and CLI: T-11, T-12, T-14 (static), T-16 pieces, NFR-04/09/11.

Synthetic worlds only (``chair.make_selftest_world``). The 450-day planted world is run once for the
module; 100-day white-noise worlds cover DAY-LIMITED, futility, budgets, resume and determinism.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

import bn_forensics as bnf
from bn_forensics import chair as ch
from bn_forensics import cli
from bn_forensics import quartermaster as qm
from bn_forensics import reporter as rp
from bn_forensics.schemas_validator import validate

REPO = Path(__file__).resolve().parents[1]
CLOCK = dt.datetime(2029, 1, 1, 18, 0, 0, tzinfo=bnf.IST)


@pytest.fixture(autouse=True)
def fixed_clock() -> Any:
    bnf.set_fixed_clock(CLOCK)
    yield
    bnf.set_fixed_clock(None)


def _opts(root: Path, world: ch.SelftestWorld, run_id: str, **kw: Any) -> ch.RunOptions:
    base: dict[str, Any] = {"command": "run", "store_dir": root / "store", "runs_root": root / "runs", "incoming_dir": world.incoming,
                            "run_id": run_id, "features": world.provider(), "ai_fallback": {"H1": [world.card]},
                            "cfg_overrides": {"cost_model": {"underlying_cost_pts": 2.0}}, "null_worlds": 80, "recall_worlds": 10}
    base.update(kw)
    return ch.RunOptions(**base)


@pytest.fixture(scope="module")
def cal() -> dict[str, Any]:
    return qm.load_era_calendar()


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory, cal: dict[str, Any]) -> dict[str, Any]:
    """The T-16 world (450 days, planted +0.8 ATR20 on feature3 AND feature7) through Ω-S0..Ω-S9."""
    bnf.set_fixed_clock(CLOCK)
    root = tmp_path_factory.mktemp("planted")
    world = ch.make_selftest_world(root / "world", cal)
    leaky = dict(world.card, card_id="C-0002", hypothesis="Leaky card that peeks at the next close", event="gt(close_{t+1}, 0)",
                 expected_sign="none")
    chair = ch.Chair(_opts(root, world, "planted", ai_fallback={"H1": [world.card, leaky]}))
    out = chair.run()
    bnf.set_fixed_clock(None)
    return {"root": root, "world": world, "chair": chair, "out": out, "run": out.run_dir}


@pytest.fixture(scope="module")
def noise_world(tmp_path_factory: pytest.TempPathFactory, cal: dict[str, Any]) -> ch.SelftestWorld:
    """100 white-noise days: DAY-LIMITED, no planted edge (T-11 / T-12)."""
    return ch.make_selftest_world(tmp_path_factory.mktemp("noise") / "world", cal, n_days=100, event_day_rate=0.0, structure=False)


@pytest.fixture(scope="module")
def noise_run(tmp_path_factory: pytest.TempPathFactory, noise_world: ch.SelftestWorld) -> dict[str, Any]:
    bnf.set_fixed_clock(CLOCK)
    root = tmp_path_factory.mktemp("noise_run")
    chair = ch.Chair(_opts(root, noise_world, "noise", cfg_overrides={}, null_worlds=60))
    out = chair.run()
    bnf.set_fixed_clock(None)
    return {"root": root, "chair": chair, "out": out, "run": out.run_dir}


# ============================================================================================
# T-16 (pieces): planted edge end to end
# ============================================================================================
def test_t16_planted_world_passes_every_selftest_assertion(planted: dict[str, Any]) -> None:
    out = planted["out"]
    assert out.exit_code == bnf.EXIT_GAPS, out.mode_line           # spot only: other families are DATA-GAPS (DEF-06)
    assert out.state == "COMPLETE_WITH_GAPS"
    assert cli._assert_selftest(planted["chair"], out, expect_grade="Ω-A") == []


def test_t16_stage_status_lines_and_state_machine(planted: dict[str, Any]) -> None:
    lines = planted["out"].status_lines
    for stage in ("[Ω-S0] INTAKE PASS", "[Ω-S1] SPLIT SEALED", "[Ω-S2] NULL_LAB PASS", "[Ω-S3] PLAYGROUND DONE",
                  "[Ω-S5] COURT SESSION k=1", "[Ω-S6] SKEPTIC+VERIFY", "[Ω-S7] LEDGER", "[Ω-S8] NEVER-IDLE", "[Ω-S9] REPORT WRITTEN"):
        assert any(ln.startswith(stage) for ln in lines), stage
    assert any(ln.startswith("Received: symbol=BANKNIFTY, interval=1m") for ln in lines)
    state = bnf.load_json(planted["run"] / "run_state.json")
    assert validate(state, "run_state").ok
    assert state["stage"] == "Ω-S9"
    assert state["state"] == "COMPLETE_WITH_GAPS"
    assert state["stages_completed"] == [f"Ω-S{i}" for i in range(10)]
    assert state["digests"] == [f"digests/S{i}.md" for i in range(9)]
    assert state["data_gaps"] == ["cross_asset", "daily", "futures", "options", "vix"]


def test_t16_digests_are_at_most_8_lines(planted: dict[str, Any]) -> None:
    for path in sorted((planted["run"] / "digests").glob("S*.md")):
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert 1 <= len(lines) <= 8, path.name


def test_nfr09_every_deliverable_is_hashed_and_logged(planted: dict[str, Any]) -> None:
    state = bnf.load_json(planted["run"] / "run_state.json")
    deliverables = state["deliverables"]
    for rel in ("BANKNIFTY_DATA_FORENSICS_REPORT.md", "OWNER_BRIEF.md", "AI_HANDOFF.md", "RUN_DIFF.md", "strategy_input_pack.json",
                "findings_registry.csv", "tables/coverage_matrix.csv", "tables/scorecard.csv", "court_docket.json"):
        assert rel in deliverables, rel
        assert deliverables[rel] == bnf.sha256_file(planted["run"] / rel)
    events = [json.loads(x) for x in (planted["run"] / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()]
    names = {e["event"] for e in events}
    assert {"null_lab_completed", "court_verdict_issued", "exposure_tagged", "verification_completed", "run_finalized"} <= names
    assert all(set(e) >= {"timestamp_ist", "level", "run_id", "stage", "role", "data_version", "event", "payload"} for e in events)
    assert all(e["run_id"] == "planted" for e in events)


# ============================================================================================
# T-12: reports, limits, ID links, coverage matrix, pack
# ============================================================================================
def test_t12_report_line1_sections_and_token_limit(planted: dict[str, Any], cfg_full: dict[str, Any]) -> None:
    text = (planted["run"] / "BANKNIFTY_DATA_FORENSICS_REPORT.md").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "MODE: FULL (code executed) · DATA-GAPS: cross_asset, daily, futures, options, vix"
    heads = [ln for ln in text.splitlines() if re.match(r"^## \d+\. ", ln)]
    assert [int(h.split(".")[0][3:]) for h in heads] == list(range(1, 14))
    assert "## Appendix Z. 44-Item Self-Audit Scorecard" in text
    assert rp.estimate_tokens(text) <= cfg_full["report"]["max_tokens"]


@pytest.fixture(scope="module")
def cfg_full() -> dict[str, Any]:
    return bnf.load_config()


def test_t12_owner_brief_and_ai_handoff_limits_and_parts(planted: dict[str, Any]) -> None:
    brief = (planted["run"] / "OWNER_BRIEF.md").read_text(encoding="utf-8").splitlines()
    assert len(brief) <= 40
    assert brief[0].startswith("MODE: FULL (code executed)")
    handoff = (planted["run"] / "AI_HANDOFF.md").read_text(encoding="utf-8").splitlines()
    assert len(handoff) <= 200
    parts = [int(m.group(1)) for ln in handoff if (m := re.match(r"^## (\d+)\. ", ln))]
    assert parts == list(range(1, 13))
    assert handoff[-1] == "diagnosis only; Ω ≠ trading rule; effects are not additive"


def test_t12_every_handoff_number_equals_the_machine_files(planted: dict[str, Any]) -> None:
    run = planted["run"]
    handoff = (run / "AI_HANDOFF.md").read_text(encoding="utf-8")
    import csv

    with open(run / "tables" / "court_verdicts.csv", encoding="utf-8", newline="") as fh:
        for v in csv.DictReader(fh):
            assert f"| {v['card_id']} | {v['verdict']} | {v['p']} | {v['event_days']} | {v['lift_atr']} |" in handoff
    with open(run / "findings_registry.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["Evidence grade"].startswith("Ω"):
                assert f"| {r['ID']} | {r['Evidence grade']} | {r['Lift (ATR20)']} | {r['Deflated lift']} |" in handoff
    nl = bnf.load_json(run / "null_lab.json")
    assert f"court_fpr={nl['fpr_court']:.3f}" in nl["status_line"]


def test_t12_hinglish_lines_are_id_linked_and_bounded(planted: dict[str, Any], cfg_full: dict[str, Any]) -> None:
    rf = rp.load_run(planted["run"], planted["root"] / "runs")
    lines = rp.hinglish_summary(rf, cfg_full)
    body = [x for x in lines if x.strip() and not x.startswith("#")]
    assert len(body) <= 30
    block = lines[lines.index("### Approach Verdict block") + 1:]
    assert 1 <= len(block) <= 8
    from bn_forensics.schemas_validator import unlinked_factual_lines

    assert unlinked_factual_lines(lines) == []
    assert rp.strategy_words("\n".join(lines)) == []


def test_t12_unlinked_hinglish_claim_is_rejected() -> None:
    with pytest.raises(bnf.SchemaError, match="F23"):
        rp.check_hinglish(["## 1. Hinglish Trader Summary", "- Market subah upar jaata hai."])
    rp.check_hinglish(["## 1. Hinglish Trader Summary", "- [C-0001] Market subah upar jaata hai."])
    with pytest.raises(bnf.SchemaError, match="lines"):
        rp.check_hinglish([f"- [C-{i:04d}] fact" for i in range(31)])


def test_t12_coverage_matrix_has_zero_empty_cells_and_36_rows(planted: dict[str, Any]) -> None:
    import csv

    with open(planted["run"] / "tables" / "coverage_matrix.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 36
    assert all(str(v).strip() for r in rows for v in r.values())
    assert sum(1 for r in rows if r["verdict"] == bnf.NA_DATA_MISSING) >= 1


def test_t12_strategy_pack_valid_with_caveats_and_bridge(planted: dict[str, Any]) -> None:
    pack = bnf.load_json(planted["run"] / "strategy_input_pack.json")
    assert validate(pack, "strategy_input_pack").ok
    assert pack["schema_version"] == "1.1"
    assert {f["grade"] for f in pack["validated_findings"]} <= {"Ω-A", "Ω-B", "Ω-C"}
    assert all(f["must_carry_caveats"] for f in pack["validated_findings"])
    assert len(pack["backtest_bridge"]) == len(pack["validated_findings"])
    assert pack["exposure_summary"]["virgin_days"] == 170
    assert pack["metadata"]["diagnosis_only"] is True
    assert pack["break_even_costs"]["underlying_cost_pts"] == 2.0


def test_t12_scorecard_has_44_items_and_no_critical_fail(planted: dict[str, Any]) -> None:
    import csv

    with open(planted["run"] / "tables" / "scorecard.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["item"]) for r in rows] == list(range(1, 45))
    assert not [r for r in rows if int(r["item"]) in rp.CRITICAL_ITEMS and r["status"] == "FAIL"]


def test_t12_day_limited_world_has_zero_supported_and_t_question_without_costs(noise_run: dict[str, Any]) -> None:
    run = noise_run["run"]
    assert noise_run["out"].exit_code in (bnf.EXIT_OK, bnf.EXIT_GAPS), noise_run["out"].mode_line
    split = bnf.load_json(run / "split_manifest.json")
    assert split["day_limited"] is True
    assert "DAY-LIMITED" in (run / "BANKNIFTY_DATA_FORENSICS_REPORT.md").read_text(encoding="utf-8").splitlines()[0]
    import csv

    with open(run / "tables" / "approach_verdicts_11a.csv", encoding="utf-8", newline="") as fh:
        assert not [r for r in csv.DictReader(fh) if r["verdict"] == "SUPPORTED FOR HYPOTHESIS"]
    with open(run / "findings_registry.csv", encoding="utf-8", newline="") as fh:
        tags = {r["Tradability tag"] for r in csv.DictReader(fh)}
    assert tags <= {"T?", "STRUCTURE-ONLY"}
    assert "USER TO CONFIRM" in (run / "BANKNIFTY_DATA_FORENSICS_REPORT.md").read_text(encoding="utf-8")


def test_t12_run_diff_lists_verdict_changes() -> None:
    def rf(verdicts: dict[str, str], tickets: list[str], days: list[str], ledger: dict[str, str]) -> dict[str, Any]:
        return {"verdicts": [{"card_id": k, "verdict": v} for k, v in verdicts.items()], "tickets": [{"ticket_id": t} for t in tickets],
                "split": {"clean_days": days}, "ledger_cards": [{"card_id": k, "state": v} for k, v in ledger.items()],
                "state": {"run_id": "run_old"}}
    old = rf({"C-0001": "COURT-REJECTED", "C-0002": "COURT-CONFIRMED"}, ["D-0001"], ["2027-01-04"], {"C-0002": "FWD-WATCH"})
    new = rf({"C-0001": "COURT-CONFIRMED", "C-0003": "COURT-REJECTED"}, ["D-0002"], ["2027-01-04", "2027-01-05"], {"C-0002": "LEDGER-CONFIRMED"})
    diff = rp.run_diff(new, old)
    assert "verdict change C-0001: COURT-REJECTED -> COURT-CONFIRMED" in diff
    assert "new docket card C-0003 (COURT-REJECTED)" in diff
    assert "removed docket card C-0002" in diff
    assert "ledger move C-0002: FWD-WATCH -> LEDGER-CONFIRMED" in diff
    assert "DRT opened D-0002" in diff
    assert "DRT closed D-0001" in diff
    assert "data days added: 1" in diff
    assert rp.run_diff(new, None) == ["first run of this lineage (nothing to compare)"]


def test_t12_report_rerender_reproduces_the_same_deliverables(planted: dict[str, Any]) -> None:
    run = planted["run"]
    before = {p: bnf.sha256_file(run / p) for p in ("BANKNIFTY_DATA_FORENSICS_REPORT.md", "OWNER_BRIEF.md", "strategy_input_pack.json")}
    bnf.set_fixed_clock(CLOCK)
    chair = ch.Chair(ch.RunOptions(command="report", store_dir=planted["root"] / "store", runs_root=planted["root"] / "runs",
                                   run_id="planted", cfg_overrides={"cost_model": {"underlying_cost_pts": 2.0}}))
    out = chair.run()
    assert out.exit_code == bnf.EXIT_GAPS
    after = {p: bnf.sha256_file(run / p) for p in before}
    assert after["strategy_input_pack.json"] == before["strategy_input_pack.json"]
    assert after["BANKNIFTY_DATA_FORENSICS_REPORT.md"] == before["BANKNIFTY_DATA_FORENSICS_REPORT.md"]


def test_t13_strategy_words_in_narrative_make_the_run_invalid(planted: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    assert rp.strategy_words("Buy above the high and exit at the stop") == ["buy", "exit", "stop"]
    assert rp.strategy_words("T1 fixed-horizon target menu, diagnosis only") == []
    monkeypatch.setattr(rp, "owner_brief", lambda *a, **k: ["MODE: FULL (code executed)", "- [C-0001] buy here"])
    with pytest.raises(bnf.InvalidPipeline) as exc:
        rp.render_all(planted["chair"])
    assert exc.value.tripwire_id == "TW-13"


# ============================================================================================
# T-11: futility, Never-Idle, Ω-12, budgets, brain tamper, resume
# ============================================================================================
def test_t11_white_noise_emits_futility_certificate_and_next_actions(noise_run: dict[str, Any]) -> None:
    run = noise_run["run"]
    import csv

    with open(run / "tables" / "court_verdicts.csv", encoding="utf-8", newline="") as fh:
        assert not [v for v in csv.DictReader(fh) if v["verdict"] == "COURT-CONFIRMED"]
    cert = bnf.load_yaml(run / "futility_certificate.yaml")
    assert validate(cert, "futility_certificate").ok
    assert cert["n_star"]["event_days_needed"] > 0
    assert cert["n_star"]["court_days_needed"] >= cert["n_star"]["event_days_needed"]
    assert "Futility != no edge exists" in cert["must_carry_caveats"]
    actions = bnf.load_yaml(run / "next_actions.yaml")
    assert validate(actions, "next_actions").ok
    assert len(actions["actions"]) >= 3
    assert {a["type"] for a in actions["actions"]} >= {"HUNT", "DATA", "WAIT"}
    pack = bnf.load_json(run / "strategy_input_pack.json")
    assert pack["futility_certificate"] == cert
    assert pack["validated_findings"] == []


def test_t11_bare_nothing_found_is_rejected(tmp_path: Path, noise_run: dict[str, Any]) -> None:
    (tmp_path / "OWNER_BRIEF.md").write_text("MODE: FULL (code executed)\nNothing found.\n", encoding="utf-8")
    with pytest.raises(bnf.InvalidPipeline) as exc:
        ch.reject_bare_nothing_found(tmp_path, confirmed=0)
    assert exc.value.tripwire_id == "F35"
    assert exc.value.exit_code == 4
    shutil.copyfile(noise_run["run"] / "futility_certificate.yaml", tmp_path / "futility_certificate.yaml")
    ch.reject_bare_nothing_found(tmp_path, confirmed=0)
    ch.reject_bare_nothing_found(tmp_path / "missing", confirmed=2)


def test_t11_wall_clock_budget_partial_then_resume(tmp_path: Path, noise_world: ch.SelftestWorld) -> None:
    def clock(start: float) -> float:
        return 999.0 if clock.armed else 0.0
    clock.armed = False  # type: ignore[attr-defined]

    def hook(stage: str, chair: ch.Chair) -> None:
        if stage == "Ω-S2":
            clock.armed = True  # type: ignore[attr-defined]

    opts = _opts(tmp_path, noise_world, "budget", cfg_overrides={}, null_worlds=40, max_wall_min=1.0, clock=clock, stage_hook=hook)
    out = ch.Chair(opts).run()
    run = out.run_dir
    assert out.exit_code == bnf.EXIT_PARTIAL
    assert out.state == "PARTIAL_CHECKPOINT"
    assert (run / "PARTIAL").exists()
    assert (run / "OWNER_BRIEF.md").exists()
    assert (run / "AI_HANDOFF.md").exists()
    assert "PARTIAL" in (run / "OWNER_BRIEF.md").read_text(encoding="utf-8").splitlines()[0]
    state = bnf.load_json(run / "run_state.json")
    assert validate(state, "run_state").ok
    assert state["exit_code"] == 2
    assert state["resume_from"] == "Ω-S3"
    assert any(t["tripwire_id"] == "TW-17" for t in state["tripwires"])
    assert "bnf resume --run-id budget" in out.status_lines
    code = cli.main(["resume", "--run-id", "budget", "--run-dir", str(tmp_path / "runs"), "--store-dir", str(tmp_path / "store")])
    assert code in (bnf.EXIT_OK, bnf.EXIT_GAPS)
    assert bnf.load_json(run / "run_state.json")["stage"] == "Ω-S9"


def test_t11_idea_budget_is_a_partial_checkpoint(tmp_path: Path, noise_world: ch.SelftestWorld) -> None:
    out = ch.Chair(_opts(tmp_path, noise_world, "ideas", cfg_overrides={}, null_worlds=40, max_ideas=5)).run()
    assert out.exit_code == bnf.EXIT_PARTIAL
    assert bnf.load_json(out.run_dir / "run_state.json")["partial_reason"].startswith("idea budget 5")


def test_t11_brain_md_tamper_mid_run_is_mode_invalid(tmp_path: Path, noise_world: ch.SelftestWorld) -> None:
    prompts = tmp_path / "prompts"
    shutil.copytree(REPO / "prompts", prompts)

    def hook(stage: str, chair: ch.Chair) -> None:
        if stage == "Ω-S1":
            with open(prompts / "brain.md", "a", encoding="utf-8") as fh:
                fh.write("\n- injected rule\n")

    out = ch.Chair(_opts(tmp_path, noise_world, "tamper", cfg_overrides={}, prompts_dir=prompts, stage_hook=hook)).run()
    assert out.exit_code == bnf.EXIT_INVALID
    assert out.mode_line.startswith("MODE: INVALID — tripwire TW-2")
    state = bnf.load_json(out.run_dir / "run_state.json")
    assert state["tripwires"][0]["tripwire_id"] == "TW-2"
    assert (out.run_dir / "BANKNIFTY_DATA_FORENSICS_REPORT.md").read_text(encoding="utf-8").startswith("MODE: INVALID")


def test_t16_second_identical_run_matches_csv_sha256(tmp_path: Path, noise_world: ch.SelftestWorld) -> None:
    digests = []
    for name in ("a", "b"):
        bnf.set_fixed_clock(CLOCK)
        out = ch.Chair(_opts(tmp_path / name, noise_world, "same", cfg_overrides={}, null_worlds=40)).run()
        assert out.exit_code in (bnf.EXIT_OK, bnf.EXIT_GAPS)
        digests.append(ch.csv_digest(out.run_dir))
    assert digests[0] == digests[1]


# ============================================================================================
# CLI contract, locks, prompts (T-14 static, FR-24)
# ============================================================================================
def test_t12_cli_usage_errors_exit_3(tmp_path: Path) -> None:
    for argv in (["--no-such-flag"], [], ["run", "--run-id", "../escape"], ["court", "--k-docket", "abc"], ["selftest", "--bogus"]):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = int(exc.code)
        assert code == 3, argv
    assert cli.main(["report", "--run-id", "latest", "--run-dir", str(tmp_path / "empty")]) == 3
    with pytest.raises(bnf.InputError):
        cli.run_isolated("bn_forensics.chair", [])
    assert set(cli.SUBCOMMANDS) == {"intake", "null-lab", "playground", "court", "ledger-tick", "report", "resume", "run", "selftest"}


def test_nfr04_lock_contention_exits_3(tmp_path: Path, noise_world: ch.SelftestWorld) -> None:
    opts = _opts(tmp_path, noise_world, "locked", cfg_overrides={})
    run_dir = tmp_path / "runs" / "locked"
    run_dir.mkdir(parents=True)
    with bnf.AppLock(run_dir.absolute()):
        out = ch.Chair(opts).run()
    assert out.exit_code == bnf.EXIT_INPUT


def test_t16_intake_subcommand_prints_echo_back(tmp_path: Path, noise_world: ch.SelftestWorld, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["intake", "--incoming-dir", str(noise_world.incoming), "--store-dir", str(tmp_path / "store"),
                     "--run-dir", str(tmp_path / "runs"), "--run-id", "intake_only"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Received: symbol=BANKNIFTY, interval=1m" in out
    assert "[Ω-S0] INTAKE PASS" in out
    assert bnf.load_json(tmp_path / "runs" / "intake_only" / "run_state.json")["stages_completed"] == ["Ω-S0"]


PROMPTS = ("brain.md", "generator_agent.md", "quartermaster_agent.md")
SECTIONS = ("Version Header", "Role", "Objective", "Inputs", "Output Schema", "Forbidden Actions", "Escalation Rule", "Worked Examples")


@pytest.mark.parametrize("name", PROMPTS)
def test_t14_prompts_carry_the_8_sections_and_a_version_header(name: str) -> None:
    text = (REPO / "prompts" / name).read_text(encoding="utf-8")
    heads = [m.group(1).strip() for m in re.finditer(r"^# \d+\. (.+)$", text, re.MULTILINE)]
    assert heads == list(SECTIONS)
    assert re.search(r"^prompt_id: \S+", text, re.MULTILINE)
    assert re.search(r"^sha256_tracked: true$", text, re.MULTILINE)
    assert text.count("## Example") >= 2
    assert "schemas/" in text


def test_t14_brain_md_is_short_and_defines_degraded_mode() -> None:
    text = (REPO / "prompts" / "brain.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 150
    assert "MODE: DEGRADED (no code, approximate)" in text
    assert "N/A DEGRADED" in text
    assert re.search(r"^max_lines: 150$", text, re.MULTILINE)


def test_brain_hash_tracks_every_byte(tmp_path: Path) -> None:
    shutil.copytree(REPO / "prompts", tmp_path / "p")
    h0 = ch.brain_hash(tmp_path / "p")
    assert h0 == bnf.sha256_file(REPO / "prompts" / "brain.md")
    (tmp_path / "p" / "brain.md").write_text((tmp_path / "p" / "brain.md").read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert ch.brain_hash(tmp_path / "p") != h0
    assert len(ch.library_hash()) == 64
