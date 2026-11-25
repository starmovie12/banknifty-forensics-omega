"""Truth layer: Null Lab, omega_sim, metamorphic + twin parity, shrinkage, Skeptic, Step F verify.

Covers T-07 (Ω-1, Ω-2, Ω-9, Ω-10), T-09 (V5-5, V5-12, F5, FM-20, FM-21) and T-10 (F1.2, F2.5, F4.4,
V5-15). Every world here is synthetic (seeded); no real market data is used.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bn_forensics as bnf
from bn_forensics import bricks_compiler as bc
from bn_forensics import core_stats as cs
from bn_forensics import null_lab as nl
from bn_forensics import omega_sim as om
from bn_forensics import quartermaster as qm
from bn_forensics import shrinkage_basis as sb
from bn_forensics import skeptic_checks as sk
from bn_forensics import twin_impl as tw
from bn_forensics import verify_checks as vc
from bn_forensics import xray_phases as xp
from bn_forensics.court import EVENT_LEDGER_FIELDS
from conftest import build_bars, weekdays, write_partitions

SRC = Path(__file__).resolve().parents[1] / "src" / "bn_forensics"


# ============================================================================================
# Shared synthetic frame
# ============================================================================================
@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return bnf.load_config()


@pytest.fixture(scope="module")
def spot_table():
    rng = np.random.default_rng(20240)
    return build_bars(weekdays(dt.date(2027, 1, 4), 40), rng=rng, stale_frac=0.02)


@pytest.fixture(scope="module")
def frame(spot_table) -> cs.BarFrame:
    return cs.resample(cs.frame_from_table(spot_table.to_pandas()), 3)


@pytest.fixture(scope="module")
def run_ctx(cfg) -> cs.Ctx:
    return cs.Ctx.from_config(cfg)


@pytest.fixture(scope="module")
def feature_ctx(frame) -> bc.FeatureContext:
    return bc.FeatureContext(frame=frame, lib=bc.BrickLibrary.load(), cal=qm.load_era_calendar(), available={"spot_1m"})


@pytest.fixture(scope="module")
def null_result(cfg, run_ctx, feature_ctx, tmp_path_factory) -> tuple[nl.NullLabResult, Path]:
    """Ω-S2 at a real-sized split (225 Playground / 135 Court days) over 300 white-noise worlds."""
    tables = tmp_path_factory.mktemp("null_lab") / "tables"
    res = nl.run_null_lab(n_playground_days=225, n_court_days=135, cfg=cfg, run_ctx=run_ctx, lib=feature_ctx.lib,
                          cal=feature_ctx.cal, ctx=feature_ctx, real_day_values=np.random.default_rng(3).normal(0, 1, 250),
                          worlds=300, recall_worlds=40, tables_dir=tables)
    return res, tables


# ============================================================================================
# T-07  Null Lab (Ω-1, Ω-2, Ω-3, Ω-9) and omega_sim (Ω-10)
# ============================================================================================
def test_t07_court_and_union_null_fpr_at_most_5_percent_over_300_worlds(null_result) -> None:
    res, tables = null_result
    assert res.verdict == "PASS"
    assert res.failures == []
    assert res.fpr_court <= 0.05
    assert res.fpr_ledger <= 0.05
    assert res.fpr_union <= 0.05
    assert res.fpr_null_b is not None
    assert res.fpr_null_b <= 0.05
    rows = vc.read_rows(tables / "null_lab_summary.csv")
    assert tuple(rows[0].keys()) == nl.SUMMARY_FIELDS
    assert any(r["check"].startswith("Null-A Court FPR") and r["status"] == "PASS" for r in rows)
    assert res.status_line().startswith("[Ω-S2] NULL_LAB PASS: court_fpr=")
    assert "canary=100%" in res.status_line()
    assert "meta=4/4" in res.status_line()
    assert "twin=PASS" in res.status_line()


def test_t07_peeking_trap_and_uncapped_v5_regression_detect_inflation(null_result) -> None:
    res, _ = null_result
    assert res.trap_fpr > 0.05          # Ω-3: p-value peek-and-stop must FAIL the Null Lab
    assert res.uncapped_fpr > 0.05      # Ω-9: the uncapped-V5 regression shows the harness is sensitive
    assert res.trap_fpr > res.fpr_ledger


def test_t07_planted_recall_grid_and_calibration_flag(null_result, cfg) -> None:
    res, _ = null_result
    keys = [f"{d}" for d in cfg["null_lab"]["deltas"]]
    assert keys == ["0.1", "0.2", "0.3", "0.5", "0.8"]
    assert set(res.recall_at_delta) == set(keys)
    assert res.recall_at_delta["0.8"] > res.recall_at_delta["0.1"]
    assert res.recall_at_delta["0.8"] >= 0.5
    powers = [res.analytic_power[k] for k in keys]
    assert powers == sorted(powers)
    divergence = max(abs(res.recall_at_delta[k] - res.analytic_power[k]) for k in keys)
    assert res.calibration_warning == (divergence > 0.25)
    assert 0.0 <= res.haircut <= 1.0
    assert res.digital_twin["n_new_court_days_for_target"] > 0


def test_t07_canary_rejected_and_leak_effect_is_large(null_result) -> None:
    res, _ = null_result
    assert res.canary_caught == 1.0
    assert res.canary_effect_if_uncaught is not None
    assert res.canary_effect_if_uncaught > 0.5   # the leak would have looked like a huge edge


def test_t07_broken_leak_gate_makes_pipeline_invalid(cfg, run_ctx, feature_ctx, monkeypatch) -> None:
    class _Accepted:
        status = "COMPILED"

    monkeypatch.setattr(nl, "compile_card", lambda *a, **k: _Accepted())
    monkeypatch.setattr(nl, "dynamic_leak_gate", lambda *a, **k: (True, "accepted"))
    with pytest.raises(bnf.InvalidPipeline) as exc:
        nl.run_null_lab(n_playground_days=120, n_court_days=100, cfg=cfg, run_ctx=run_ctx, lib=feature_ctx.lib,
                        cal=feature_ctx.cal, ctx=feature_ctx, worlds=12, recall_worlds=4)
    assert exc.value.exit_code == 4
    assert exc.value.tripwire_id == "F8"


def test_t07_union_fpr_above_5_percent_halves_ledger_alpha(cfg, run_ctx, feature_ctx, monkeypatch) -> None:
    calls = {"n": 0}
    real = nl._court_any

    def every_tenth(*args: Any, **kwargs: Any) -> tuple[bool, set[int]]:
        calls["n"] += 1
        if calls["n"] % 10 == 0:
            return True, {0}
        return real(*args, **kwargs)

    alphas: list[float] = []
    real_ledger = nl._ledger_any

    def spy_ledger(*args: Any, **kwargs: Any) -> bool:
        alphas.append(kwargs["alpha"])
        return real_ledger(*args, **kwargs)

    monkeypatch.setattr(nl, "_court_any", every_tenth)
    monkeypatch.setattr(nl, "_ledger_any", spy_ledger)
    res = nl.run_null_lab(n_playground_days=150, n_court_days=120, cfg=cfg, run_ctx=run_ctx, lib=feature_ctx.lib,
                          cal=feature_ctx.cal, ctx=feature_ctx, worlds=20, recall_worlds=4, raise_on_fail=False)
    assert res.fpr_union > 0.05
    assert res.ledger_alpha_halved
    assert res.ledger_alpha == cfg["ledger"]["alpha_halved"] == 0.025
    assert set(alphas) == {0.05, 0.025}
    assert res.verdict == "INVALID"
    union_row = next(r for r in res.rows if r["check"].startswith("Union"))
    assert "halved" in union_row["detail"]


def test_t07_court_decision_is_exact_with_early_termination() -> None:
    rng = np.random.default_rng(11)
    x_null = rng.normal(0, 1, 60)
    x_edge = rng.normal(0.9, 1, 60)
    thr = cs.alpha_k(1) / 20
    B = 10_000
    for x, expect in ((x_null, False), (x_edge, True)):
        gen = np.random.default_rng(43)
        exceed = 0
        for start in range(0, B, 2000):               # same draws as court_confirms, never stopped early
            m = min(2000, B - start)
            flips = gen.integers(0, 2, size=(m, x.size), dtype=np.int8).astype(float) * 2 - 1
            exceed += int(np.count_nonzero((flips * x).mean(axis=1) >= x.mean() - 1e-12))
        full = (1 + exceed) / (B + 1) < thr
        assert nl.court_confirms(x, 1, thr, B, 43) == full == expect


def test_t07_omega_sim_reproduces_the_stated_appendix_numbers() -> None:
    out = om.run(300, worlds=("NULL",), protocols_subset=("P0", "P4", "P4c", "P5"))
    res = {p: v["NULL"] for p, v in out["results"].items()}
    assert res["P0"]["rate"] >= 0.99                     # search-until-pass always "finds" something
    assert 0.011 < res["P5"]["rate"] < 0.047             # FR-22: diverse single-shot Court NULL FPR
    p5 = next(c for c in out["comparison"] if c["protocol"] == "P5" and c["world"] == "NULL")
    assert p5["overlap"]
    p4c = next(c for c in out["comparison"] if c["protocol"] == "P4c" and c["world"] == "NULL")
    assert p4c["overlap"]
    assert res["P4"]["rate"] <= 0.05                     # e-value peeking stays valid
    assert res["P4c"]["rate"] > 0.05                     # p-value peeking does not
    assert res["P4c"]["rate"] > res["P5"]["rate"]
    assert all("overlap" in c and "wilson" in c for c in out["comparison"])
    assert out["config"]["label"].startswith("[SIM]")


def test_t07_omega_sim_uncapped_v5_inflates_and_is_deterministic() -> None:
    p6 = om.run(60, worlds=("NULL",), protocols_subset=("P6",))["results"]["P6"]["NULL"]
    assert p6["rate"] > 0.05
    a = om.run(8, worlds=("NULL", "STRONG"), protocols_subset=("P5",))
    b = om.run(8, worlds=("NULL", "STRONG"), protocols_subset=("P5",))
    assert a == b


def test_t07_omega_sim_cli_and_env_config(tmp_path: Path) -> None:
    assert om.main([]) == 3
    assert om.main(["many", "out.json"]) == 3
    env = om.env_config({"NS_WORLDS": "null, strong", "NS_D": "300", "NS_BATCH": "50"})
    assert env["worlds"] == ("NULL", "STRONG")
    assert env["n_days"] == 300
    with pytest.raises(ValueError, match="unknown worlds"):
        om.env_config({"NS_WORLDS": "huge"})
    with pytest.raises(ValueError, match="unknown protocols"):
        om.run(1, worlds=("NULL",), protocols_subset=("P9",))


# ============================================================================================
# T-09  Metamorphic tests, twin parity, shrinkage, orthogonal basis
# ============================================================================================
def test_t09_four_metamorphic_relations_pass(frame, run_ctx) -> None:
    out = nl.metamorphic_tests(frame, run_ctx)
    assert [m["test"] for m in out] == ["price_x_k", "timestamp_shift_7d", "mirror_c_price", "day_shuffle"]
    assert all(m["pass"] for m in out), out


def test_t09_metamorphic_mirror_relation_detects_a_sign_bug(frame, run_ctx, monkeypatch) -> None:
    real = nl.event_study

    def sign_blind(fr: cs.BarFrame, mask: np.ndarray, **kw: Any) -> cs.EventResult:
        res = real(fr, mask, **kw)
        return dataclasses.replace(res, lift_atr=abs(res.lift_atr))   # a bug that loses the move's direction

    monkeypatch.setattr(nl, "event_study", sign_blind)
    out = {m["test"]: m["pass"] for m in nl.metamorphic_tests(frame, run_ctx)}
    assert out["mirror_c_price"] is False


def test_t09_twin_parity_on_200_random_events(frame, run_ctx) -> None:
    res = nl.twin_parity(frame, run_ctx, n_events=200, seed=42)
    assert res["events_sampled"] == 200
    assert res["events_compared"] > 100
    assert res["mismatches"] == []
    assert res["pass"]


def test_t09_twin_flags_an_off_by_one_engine(frame, run_ctx) -> None:
    rng = np.random.default_rng(5)
    mask = (rng.random(frame.n) < 0.05) & (frame.slot >= 21) & (frame.slot < frame.bars_per_day - 10)
    direction = np.ones(frame.n)
    idx = np.flatnonzero(mask)
    twin = tw.twin_event_study(nl.frame_to_twin_bars(frame), [tw.EventSpec(int(i), 1.0) for i in idx], target="T1", h=3,
                               side="continuation", event_mask_all=idx.tolist())
    shifted = np.roll(mask, 1)
    eng = cs.event_study(frame, shifted, direction=direction, side="continuation", target="T1", horizon=3, bootstrap=False)
    rows = [{"index": int(i), "signed": float(s), "baseline": float(b), "residual": float(s - b), "atr20": float(frame.atr20[i])}
            for i, s, b in zip(eng.event_index, eng.signed_outcome, eng.baseline_values, strict=True)]
    cmp = tw.compare_events(twin["rows"], rows)
    assert not cmp["pass"]
    assert cmp["mismatches"]


def test_t09_deflated_lift_tracks_planted_lift_on_2000_cards() -> None:
    rng = np.random.default_rng(42)
    n, top_k = 2000, 50
    mu = rng.normal(0, 0.15, n)                             # planted true lifts (ATR20)
    se = rng.uniform(0.06, 0.14, n)
    lift = mu + se * rng.normal(0, 1, n)
    fits = sb.fit_families([{"family": "T1", "lift": float(a), "se": float(b)} for a, b in zip(lift, se, strict=True)])
    fit = fits["T1"]
    assert fit.method == "EB"
    assert fit.n_cards == n
    assert abs(math.sqrt(fit.tau2) - 0.15) < 0.02
    deflated = np.array([sb.deflated_lift(float(a), float(b), fit) for a, b in zip(lift, se, strict=True)])
    top = np.argsort(-lift)[:top_k]                         # winner's-curse selection on raw lift
    truth = mu[top].mean()
    assert lift[top].mean() > truth                         # raw selected lift is inflated (F20)
    assert abs(deflated[top].mean() - truth) < 0.25 * abs(lift[top].mean() - truth)
    assert np.all(np.abs(deflated) <= np.abs(lift) + 1e-12)
    slope = np.polyfit(deflated, mu, 1)[0]                  # FM-20: recovered vs planted is calibrated
    assert 0.8 < slope < 1.25
    ratio = deflated[top].mean() / truth
    assert 0.8 < ratio < 1.25


def test_t09_orthogonal_basis_collapses_clones_and_reports_n_indep(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    days = [f"2027-01-{d:02d}" for d in range(1, 31)] + [f"2027-02-{d:02d}" for d in range(1, 29)]
    base_days = days[:20]
    base_x = {d: float(rng.normal()) for d in base_days}
    findings = [{"finding_id": f"F-CLONE-{k}", "event_days": set(base_days), "x_by_day": dict(base_x)} for k in range(3)]
    for k, chunk in enumerate((days[20:32], days[32:44], days[44:56])):
        findings.append({"finding_id": f"F-SOLO-{k}", "event_days": set(chunk), "x_by_day": {d: float(rng.normal()) for d in chunk}})
    rows, n_indep = sb.orthogonal_basis(findings)
    assert n_indep == 4
    eff = {r["finding_id"]: r["independent_effect_id"] for r in rows}
    assert len({eff[f"F-CLONE-{k}"] for k in range(3)}) == 1
    assert len({eff[f"F-SOLO-{k}"] for k in range(3)}) == 3
    a, _ = sb.write_basis_tables(tmp_path, rows, [])
    assert a.read_text(encoding="utf-8").splitlines()[0] == ",".join(sb.ORTHO_FIELDS)


# ============================================================================================
# T-10  Isolation (F1.2, F5, NFR-10), Skeptic re-tests and grades, Step F verify
# ============================================================================================
def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
    return names


@pytest.mark.parametrize("module", ["skeptic_checks.py", "twin_impl.py", "__init__.py"])
def test_t10_isolated_modules_import_no_core_stats_or_loaders(module: str) -> None:
    imported = _imports(SRC / module)
    forbidden = ("bn_forensics", "core_stats", "quartermaster", "split_seal", "bricks_compiler", "pandas", ".")
    bad = sorted(n for n in imported if n == "." or any(n == f or n.startswith(f + ".") or n.startswith(".") for f in forbidden))
    assert bad == [], f"{module} imports {bad}"


def test_t10_package_core_is_stdlib_only() -> None:
    allowed_third_party = {"yaml"}             # pyyaml, imported lazily inside load_yaml/dump_yaml
    stdlib = set(sys.stdlib_module_names)
    for name in _imports(SRC / "__init__.py"):
        top = name.split(".")[0]
        assert top in stdlib or top == "__future__" or top in allowed_third_party, name


@pytest.mark.parametrize("module", ["bn_forensics.skeptic_checks", "bn_forensics.twin_impl"])
def test_t10_isolated_modules_load_alone_at_runtime(module: str) -> None:
    code = (f"import sys, {module}\n"
            "print(sorted(m for m in sys.modules if m.startswith('bn_forensics')))\n"
            "print('pandas' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[0] == str(sorted(["bn_forensics", module]))
    assert out[1] == "False"


def _ledger_rows(card: str, week_effects: list[float], per_week: int, rng: np.random.Generator, noise: float = 0.05,
                 start: dt.date = dt.date(2027, 1, 4)) -> list[dict[str, Any]]:
    rows = []
    for w, eff in enumerate(week_effects):
        monday = start + dt.timedelta(weeks=w)
        for k in range(per_week):
            day = monday + dt.timedelta(days=k % 5)
            base = float(rng.normal(0, 0.02))
            clean = base + eff + float(rng.normal(0, noise))
            rows.append({"test_id": card, "date": day.isoformat(), "bar_ts": f"{day.isoformat()}T10:{k:02d}:00+05:30",
                         "raw_ret": clean, "clean_ret": clean, "baseline_ret": base,
                         "next_open_ret": base + eff + float(rng.normal(0, noise)), "placebo_ret": float(rng.normal(0, noise)),
                         "vol_tercile": 1 + (k % 3), "is_expiry": int(k == 3), "is_outlier_day": 0, "overlap_flag": int(k == 0)})
    return rows


def _as_text(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: cs.fmt_cell(v) for k, v in r.items()} for r in rows]


def test_t10_week_block_retest_fails_slow_regime_artifact() -> None:
    rng = np.random.default_rng(42)
    slow = _as_text(_ledger_rows("SLOW", [0.8] * 15 + [-0.6] * 15, 10, rng))
    resid = np.array([float(r["clean_ret"]) - float(r["baseline_ret"]) for r in slow])
    iid_half = 1.96 * resid.std(ddof=1) / math.sqrt(resid.size)
    assert resid.mean() - iid_half > 0            # an event-level i.i.d. CI would call it significant
    out = sk.run_retests(slow, B=2000, seed=42)
    assert out["results"][7][0] == "FAIL"
    steady = _as_text(_ledger_rows("STEADY", [0.3] * 30, 10, rng))
    assert sk.run_retests(steady, B=2000, seed=42)["results"][7][0] == "PASS"


def test_t10_retests_are_non_vacuous_and_counted() -> None:
    rng = np.random.default_rng(1)
    rows = _as_text(_ledger_rows("C", [0.3] * 30, 10, rng))
    out = sk.run_retests(rows, B=1000, seed=42)
    status = {k: s for k, (s, _) in out["results"].items()}
    assert sorted(status) == [1, 2, 3, 4, 5, 6, 7]
    assert status[5] == "VACUOUS"                  # raw == clean: never counted as a pass
    assert out["vacuous"] == 1
    assert out["ran"] == sum(1 for s in status.values() if s in ("PASS", "FAIL"))
    assert out["fails"] == 0
    assert out["ran"] >= 4


def test_t10_skeptic_tiers_and_omega_grade_mapping() -> None:
    assert sk.skeptic_tier(0, 5, unverified_causal=False, mechanism_refuted=False) == "A"
    assert sk.skeptic_tier(0, 5, unverified_causal=True, mechanism_refuted=False) == "B"
    assert sk.skeptic_tier(1, 4, unverified_causal=False, mechanism_refuted=False) == "B"
    assert sk.skeptic_tier(0, 3, unverified_causal=False, mechanism_refuted=False) == "B"
    assert sk.skeptic_tier(2, 6, unverified_causal=False, mechanism_refuted=False) == "C"
    assert sk.skeptic_tier(0, 2, unverified_causal=False, mechanism_refuted=False) == "C"
    base = {"court_verdict": "COURT-CONFIRMED", "exposure": "VIRGIN-COURT", "robust": "ROBUST", "tier": "A", "ran": 5,
            "fails": 0, "era_tag": "ERA-CONSISTENT", "ledger_state": "", "unverified_causal": False, "mechanism_refuted": False}
    assert sk.omega_grade(**base)[0] == "Ω-A"
    assert sk.omega_grade(**{**base, "exposure": "EXPOSED-COURT"})[0] == "Ω-B"           # R28 cap
    assert sk.omega_grade(**{**base, "exposure": "EXPOSED-COURT", "ledger_state": "LEDGER-CONFIRMED"})[0] == "Ω-A"
    assert sk.omega_grade(**{**base, "exposure": "EXPOSED-COURT"}, policy="ignore")[0] == "Ω-A"
    assert sk.omega_grade(**{**base, "fails": 1, "robust": "FRAGILE"})[:2] == ("NONE", "KILLED")
    assert sk.omega_grade(**{**base, "tier": "C"})[:2] == ("NONE", "KILLED")
    assert sk.omega_grade(**{**base, "court_verdict": "COURT-UNDERPOWERED"})[0] == "NONE"
    assert sk.omega_grade(**{**base, "court_verdict": "", "ledger_state": "LEDGER-CONFIRMED"})[0] == "Ω-C"
    assert sk.omega_grade(**{**base, "court_verdict": "COURT-REJECTED"})[0] == "NONE"


def test_t10_skeptic_subprocess_end_to_end(tmp_path: Path) -> None:
    run = tmp_path / "run"
    tables = run / "tables"
    tables.mkdir(parents=True)
    rng = np.random.default_rng(3)
    assert len(EVENT_LEDGER_FIELDS) == 12
    cs.write_csv(tables / "event_ledger.csv", EVENT_LEDGER_FIELDS,
                 _ledger_rows("CARD-1", [0.3] * 30, 10, rng) + _ledger_rows("CARD-2", [0.8] * 15 + [-0.6] * 15, 10, rng))
    cs.write_csv(tables / "court_verdicts.csv", ("card_id", "verdict", "exposure", "robust"), [
        {"card_id": "CARD-1", "verdict": "COURT-CONFIRMED", "exposure": "VIRGIN-COURT", "robust": "ROBUST"},
        {"card_id": "CARD-2", "verdict": "COURT-CONFIRMED", "exposure": "VIRGIN-COURT", "robust": "FRAGILE"}])
    cs.write_csv(tables / "skeptic_inputs.csv", ("card_id", "era_tag", "unverified_causal", "mechanism_status", "ledger_state", "lane"), [
        {"card_id": c, "era_tag": "ERA-CONSISTENT", "unverified_causal": "false", "mechanism_status": "", "ledger_state": "", "lane": "B"}
        for c in ("CARD-1", "CARD-2")])
    reg_rows = [{**dict.fromkeys(vc.REGISTRY_FIELDS, ""), "ID": f"{c}@T1h3", "Evidence grade": "B"} for c in ("CARD-1", "CARD-2")]
    cs.write_csv(run / "findings_registry.csv", vc.REGISTRY_FIELDS, reg_rows)
    (run / "test_plan.md").write_text("# Test plan\n- bootstrap_b: 2000\n- seed: 42\n- exposure_policy: cap\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "bn_forensics.skeptic_checks", "--run-dir", str(run)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "SKEPTIC (SELF-REVIEW)" in proc.stdout
    res = {r["card_id"]: r for r in vc.read_rows(tables / "skeptic_results.csv")}
    assert tuple(next(iter(res.values())).keys()) == sk.RESULT_FIELDS
    assert res["CARD-1"]["omega_grade"] == "Ω-A"
    assert res["CARD-2"]["omega_grade"] == "NONE"
    reg = {r["ID"]: r for r in vc.read_rows(run / "findings_registry.csv")}
    assert reg["CARD-1@T1h3"]["Evidence grade"] == "Ω-A"
    assert "SELF-REVIEW" in reg["CARD-1@T1h3"]["Skeptic verdict"]
    assert list(reg["CARD-1@T1h3"].keys()) == list(vc.REGISTRY_FIELDS)
    bad = subprocess.run([sys.executable, "-m", "bn_forensics.skeptic_checks"], capture_output=True, text=True)
    assert bad.returncode == 3


# --------------------------------------------------------------------------------------------
# Step F verify (FR-16)
# --------------------------------------------------------------------------------------------
GRADES = ("Ω-A", "Ω-B", "B", "C")
APPROACH_OF = ("AP34", "AP35", "AP36", "AP12")


def _row_masks(fr: cs.BarFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(100 + k)
    mask = (rng.random(fr.n) < 0.04) & (fr.slot >= 21)
    return mask, np.where(rng.random(fr.n) < 0.5, 1.0, -1.0)


def _registry_row(fr: cs.BarFrame, k: int, mask: np.ndarray, direction: np.ndarray) -> dict[str, str]:
    res = cs.event_study(fr, mask, direction=direction, side="continuation", target="T1", horizon=3, bootstrap=False)
    row = dict.fromkeys(vc.REGISTRY_FIELDS, "")
    row.update({"ID": f"C{k}@T1h3", "Phase": "7", "Label": "VIRGIN-COURT", "Event N": res.n_events, "Effective N": res.effective_n,
                "Event value": res.event_value, "Baseline": res.baseline, "Lift (points)": res.lift_points,
                "Lift (ATR20)": res.lift_atr, "Nominal MDE": 0.2, "Validation lift": res.lift_atr, "Evidence grade": GRADES[k],
                "Tradability tag": "T1", "Burn count": 1, "Approach ID": APPROACH_OF[k], "Lane": "B",
                "Skeptic verdict": "SURVIVED (SELF-REVIEW; ran 5, fails 0)"})
    return {key: cs.fmt_cell(v) for key, v in row.items()}


@pytest.fixture
def verify_run(tmp_path: Path, spot_table, frame) -> dict[str, Any]:
    """A complete, internally consistent run directory + canonical store for Step F."""
    store = tmp_path / "store"
    write_partitions(spot_table, store, "spot_1m")
    run = tmp_path / "run"
    tables = run / "tables"
    tables.mkdir(parents=True)
    registry, events, tests = [], [], []
    for k in range(4):
        mask, direction = _row_masks(frame, k)
        registry.append(_registry_row(frame, k, mask, direction))
        for i in np.flatnonzero(mask):
            events.append({"ID": f"C{k}@T1h3", "kind": "event", "date": str(frame.dates[i]), "minute": int(frame.minute[i]),
                           "direction": direction[i], "target": "T1", "horizon": 3, "side": "continuation", "baseline": "slot_matched"})
        tests.append({"approach_id": APPROACH_OF[k], "test_id": f"C{k}@T1h3", "status": "EXECUTED", "side": "continuation",
                      "grade": GRADES[k], "t_tag": "T1", "q_disc": 0.01, "se": 0.05, "gate_mde": 0.2, "effective_n": 40,
                      "skeptic_ran": 5, "cost_series": "underlying", "missing": "", "reason": ""})
    cs.write_csv(run / "findings_registry.csv", vc.REGISTRY_FIELDS, registry)
    cs.write_csv(tables / "registry_events.csv", vc.REGISTRY_EVENT_FIELDS, events)
    cs.write_csv(tables / "approach_tests.csv", vc.APPROACH_TEST_FIELDS, tests)
    gy = {**dict.fromkeys(vc.GRAVEYARD_FIELDS, ""), "ID": "G0@T1h3", "Evidence grade": "D", "Nominal MDE": "0.31",
          "graveyard_reason": "NOISE"}
    cs.write_csv(tables / "graveyard.csv", vc.GRAVEYARD_FIELDS, [gy])
    cs.write_csv(tables / "test_registry.csv", cs.REGISTRY_FIELDS, [
        {"test_id": f"C{k}@T1h3", "status": "EXECUTED"} for k in range(4)] + [{"test_id": "G0@T1h3", "status": "SKIPPED"}])
    plan = "# Test plan\n- seed: 42\n- day_limited: false\n- atr20_median_pts: 50\n- cost_pts_underlying: 5\n"
    (run / "test_plan.md").write_text(plan, encoding="utf-8")
    r11a, r11b = vc.recompute_tables(vc.read_rows(tables / "approach_tests.csv"), vc.read_plan(run / "test_plan.md"), [])
    vc.write_rows(tables / "approach_verdicts_11a.csv", vc.VERDICT_11A_FIELDS, r11a)
    vc.write_rows(tables / "approach_verdicts_11b.csv", vc.VERDICT_11B_FIELDS, r11b)
    return {"run": run, "store": store, "tables": tables, "frame": frame}


def _inject_off_by_one(run: Path, frame: cs.BarFrame, k: int = 1) -> None:
    mask, direction = _row_masks(frame, k)
    shifted = _registry_row(frame, k, np.roll(mask, 1), np.roll(direction, 1))
    rows = vc.read_rows(run / "findings_registry.csv")
    for r in rows:
        if r["ID"] == shifted["ID"]:
            for f in ("Event value", "Baseline", "Lift (ATR20)", "Lift (points)"):
                r[f] = shifted[f]
    vc.write_rows(run / "findings_registry.csv", vc.REGISTRY_FIELDS, rows)


def test_t10_registry_has_37_columns_in_prd_order() -> None:
    assert len(vc.REGISTRY_FIELDS) == 37
    assert len(set(vc.REGISTRY_FIELDS)) == 37
    assert vc.REGISTRY_FIELDS[:3] == ("ID", "Phase", "Family_m")
    assert vc.REGISTRY_FIELDS[31] == "Approach ID"
    assert vc.REGISTRY_FIELDS[32:] == ("Lane", "Deflated lift", "Mechanism status", "Burn count", "Independent effect ID")
    assert vc.GRAVEYARD_FIELDS[:37] == vc.REGISTRY_FIELDS


def test_t10_verify_passes_a_consistent_run(verify_run, cfg) -> None:
    log = bnf.EventLog(verify_run["run"] / "events.jsonl", run_id="t10")
    res = vc.run_verify(verify_run["run"], store=verify_run["store"], cfg=cfg, log=log)
    assert res.mismatches == []
    assert res.table_mismatches == []
    assert all(r["status"] == "PASS" for r in res.live), res.live
    assert len(res.live) == 12
    assert res.exit_code == 0
    assert res.verify_pass
    assert sorted(res.checked) == sorted(f"C{k}@T1h3" for k in range(4))
    assert "grade-A" in res.reasons["C0@T1h3"]
    doc = bnf.load_json(verify_run["tables"] / "verification_log.json")
    assert doc["schema_version"] == "1.0"
    assert doc["verify_pass"] is True
    assert log.records[-1]["event"] == "verification_completed"
    assert set(log.records[-1]["payload"]) == {"sample_rows", "mismatches", "demoted_count", "verify_pass"}


def test_t10_verify_catches_off_by_one_demotes_and_rebuilds_11a(verify_run, cfg) -> None:
    run, tables = verify_run["run"], verify_run["tables"]
    before = {r["approach_id"]: r["verdict"] for r in vc.read_rows(tables / "approach_verdicts_11a.csv")}
    assert before["AP35"] == "SUPPORTED FOR HYPOTHESIS"
    _inject_off_by_one(run, verify_run["frame"])
    res = vc.run_verify(run, store=verify_run["store"], cfg=cfg)
    assert res.exit_code == 2                                     # PARTIAL after 2 fix rounds
    assert res.demoted == ["C1@T1h3"]
    assert [r["round"] for r in res.rounds] == [0, 1, 2]
    assert all(r["mismatched_ids"] == ["C1@T1h3"] for r in res.rounds)
    assert {m["field"] for m in res.mismatches} >= {"Lift (ATR20)"}
    assert "C1@T1h3" not in {r["ID"] for r in vc.read_rows(run / "findings_registry.csv")}
    gy = {r["ID"]: r for r in vc.read_rows(tables / "graveyard.csv")}
    assert gy["C1@T1h3"]["Evidence grade"] == "D"
    assert gy["C1@T1h3"]["graveyard_reason"].startswith("D UNVERIFIED")
    assert "UNVERIFIED" in gy["C1@T1h3"]["Label"]
    after = {r["approach_id"]: r["verdict"] for r in vc.read_rows(tables / "approach_verdicts_11a.csv")}
    assert res.tables_rebuilt
    assert after["AP35"] != "SUPPORTED FOR HYPOTHESIS"
    assert after["AP34"] == before["AP34"]
    assert "VERIFY PARTIAL" in res.status_line()


def test_t10_verify_fix_rounds_call_the_fixer_and_stop_at_two(verify_run, cfg) -> None:
    run = verify_run["run"]
    _inject_off_by_one(run, verify_run["frame"])
    calls: list[int] = []
    res = vc.run_verify(run, store=verify_run["store"], cfg=cfg, fixer=lambda mm: calls.append(len(mm)))
    assert len(calls) == cfg["verify"]["max_fix_rounds"] == 2
    assert res.exit_code == 2


def test_t10_verify_fixer_that_repairs_the_row_clears_the_mismatch(verify_run, cfg) -> None:
    run, fr = verify_run["run"], verify_run["frame"]
    _inject_off_by_one(run, fr)
    mask, direction = _row_masks(fr, 1)
    good = _registry_row(fr, 1, mask, direction)

    def fixer(_mm: list[dict[str, Any]]) -> None:
        rows = [good if r["ID"] == good["ID"] else r for r in vc.read_rows(run / "findings_registry.csv")]
        vc.write_rows(run / "findings_registry.csv", vc.REGISTRY_FIELDS, rows)

    res = vc.run_verify(run, store=verify_run["store"], cfg=cfg, fixer=fixer)
    assert res.demoted == []
    assert [r["round"] for r in res.rounds] == [0, 1]
    assert res.exit_code == 0


def test_t10_verify_unremoved_mismatch_is_invalid(verify_run, cfg, monkeypatch) -> None:
    _inject_off_by_one(verify_run["run"], verify_run["frame"])
    monkeypatch.setattr(vc, "_demote", lambda *a, **k: None)
    with pytest.raises(bnf.InvalidPipeline) as exc:
        vc.run_verify(verify_run["run"], store=verify_run["store"], cfg=cfg)
    assert exc.value.exit_code == 4


def test_t10_verify_detects_published_11a_that_disagrees_with_2a3(verify_run, cfg) -> None:
    tables = verify_run["tables"]
    rows = vc.read_rows(tables / "approach_verdicts_11a.csv")
    for r in rows:
        if r["approach_id"] == "AP36":
            r["verdict"] = "CONTRADICTED"
    vc.write_rows(tables / "approach_verdicts_11a.csv", vc.VERDICT_11A_FIELDS, rows)
    res = vc.run_verify(verify_run["run"], store=verify_run["store"], cfg=cfg)
    assert any(m["ID"] == "AP36" and m["field"] == "verdict" for m in res.table_mismatches)
    assert res.exit_code == 2
    fixed = {r["approach_id"]: r["verdict"] for r in vc.read_rows(tables / "approach_verdicts_11a.csv")}
    assert fixed["AP36"] == "SUPPORTED FOR HYPOTHESIS"


def test_t10_live_assertion_catches_exposed_omega_a(verify_run, cfg) -> None:
    run = verify_run["run"]
    rows = vc.read_rows(run / "findings_registry.csv")
    rows[0]["Label"] = "EXPOSED-COURT"
    vc.write_rows(run / "findings_registry.csv", vc.REGISTRY_FIELDS, rows)
    live = {r["assertion_id"]: r["status"] for r in vc.live_assertions(run)}
    assert live["LIVE-03"] == "FAIL"
    assert sum(1 for s in live.values() if s == "FAIL") == 1


def test_t10_verify_2a3_matches_the_xray_rule_on_random_tests(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    doc = bnf.load_yaml(bnf.asset_dir("config") / "ap_cards.yaml")
    grades = ["Ω-A", "Ω-B", "Ω-C", "A", "B", "C", "D", None]
    tags = ["T1", "T2", "T3", "T?", "STRUCTURE-ONLY", None]
    tests = []
    for a in doc["approaches"]:
        if a.get("rollup"):
            continue
        for j in range(int(rng.integers(0, 5))):
            executed = rng.random() < 0.85
            tests.append({"approach_id": a["approach_id"], "test_id": f"{a['approach_id']}-{j}",
                          "status": "EXECUTED" if executed else "SKIPPED", "side": str(rng.choice(["continuation", "reversal", "both"])),
                          "grade": grades[int(rng.integers(0, len(grades)))], "t_tag": tags[int(rng.integers(0, len(tags)))],
                          "q_disc": float(rng.random()), "se": float(rng.uniform(0.01, 0.2)), "gate_mde": float(rng.uniform(0.05, 0.6)),
                          "effective_n": int(rng.integers(5, 120)), "skeptic_ran": int(rng.integers(0, 8)),
                          "cost_series": str(rng.choice(["underlying", "option_buyer", "structure_only"])),
                          "missing": [] if executed else ["chain_1m"], "reason": "" if executed else "needs chain_1m"})
    costs = {"underlying": 5.0, "option_buyer": None}
    vr = [{"h": 5, "vr": 1.2, "excludes_1": True}, {"h": 10, "vr": 0.95, "excludes_1": False}]
    for day_limited in (False, True):
        ref = [xp.verdict_2a3(a, tests, costs=costs, atr_points=50.0, day_limited=day_limited)
               for a in doc["approaches"] if not a.get("rollup")]
        for a in doc["approaches"]:
            if a.get("rollup"):
                ref.append(xp.rollup_verdict(a, ref, vr))
        ref.sort(key=lambda r: r["approach_id"])
        ref_b = xp.group_verdicts(ref, doc["groups"])
        text_tests = [{**{k: cs.fmt_cell(v) for k, v in t.items() if k != "missing"}, "missing": ";".join(t["missing"])} for t in tests]
        plan = {"atr20_median_pts": "50", "cost_pts_underlying": "5", "day_limited": str(day_limited).lower()}
        vr_text = [{k: cs.fmt_cell(v) for k, v in r.items()} for r in vr]
        got, got_b = vc.recompute_tables(text_tests, plan, vr_text)
        assert [(r["approach_id"], r["verdict"], r.get("label", "")) for r in got] == \
            [(r["approach_id"], r["verdict"], r.get("label", "")) for r in ref]
        assert [(r["group"], r["verdict"], r["best_evidence_grade"]) for r in got_b] == \
            [(r["group"], r["verdict"], r["best_evidence_grade"]) for r in ref_b]


def test_t10_verify_cli_contract(verify_run, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        vc.main([])
    assert exc.value.code == 3
    assert vc.main(["--run-dir", str(tmp_path / "missing"), "--store", str(verify_run["store"])]) == 3
    proc = subprocess.run([sys.executable, "-m", "bn_forensics.verify_checks", "--run-dir", str(verify_run["run"]),
                           "--store", str(verify_run["store"])], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("[Ω-S6] VERIFY PASS")


def test_t10_store_served_twice_for_one_date_is_rejected(tmp_path: Path, spot_table) -> None:
    write_partitions(spot_table, tmp_path / "store", "spot_1m", week="week_01")
    write_partitions(spot_table, tmp_path / "store", "spot_1m", week="week_02")
    with pytest.raises(bnf.InputError, match="ambiguous store"):
        vc.find_partitions(tmp_path / "store", "spot_1m", "BANKNIFTY")
