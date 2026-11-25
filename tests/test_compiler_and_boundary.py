"""FILE-044 · Idea Card v2 compiler, Leak Gate, Boundary ban, NS, MODEL/Mechanism cards, proposals,
silent-horizon log, X-ray and 2A.3 verdicts (T-04, T-05).

Covers V4-1, V4-2, V5-1..V5-4, V5-11, F2.3, F3.2, F3.3, F3.5, F4.7, F5.1..F5.5, FR-06..FR-08,
FR-10 (tags, cost model, EB shrinkage known answers), FR-21 proposals, Ω-7 era tags.
"""

from __future__ import annotations

import copy
import datetime as dt
import os
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bn_forensics as bnf
from bn_forensics import bricks_compiler as bc
from bn_forensics import core_stats as cs
from bn_forensics import quartermaster as qm
from bn_forensics import shrinkage_basis as sb
from bn_forensics import xray_phases as xp


@pytest.fixture(scope="module")
def cfg() -> dict:
    return bnf.load_config()


@pytest.fixture(scope="module")
def cal() -> dict:
    return qm.load_era_calendar()


@pytest.fixture(scope="module")
def lib() -> bc.BrickLibrary:
    return bc.BrickLibrary.load()


def _weekdays(start: dt.date, n: int) -> list[dt.date]:
    out, day = [], start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def _spot_frame(days: list[dt.date], seed: int = 42, bars: int = 375, minutes: int = 3) -> cs.BarFrame:
    """1-minute random walk at the S7 index grid, rebuilt into ``minutes`` bars (test data only)."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 52000.0
    open_min = 9 * 60 + 15
    for day in days:
        price *= float(np.exp(rng.normal(0, 0.004)))
        rets = rng.normal(0, 0.0005, bars)
        for i in range(bars):
            o = price
            c = o * float(np.exp(rets[i]))
            m = open_min + i
            ts = pd.Timestamp(dt.datetime(day.year, day.month, day.day, m // 60, m % 60), tz="Asia/Kolkata")
            rows.append((ts, o, max(o, c) + abs(rng.normal(0, 8)), min(o, c) - abs(rng.normal(0, 8)), c))
            price = c
    df = pd.DataFrame(rows, columns=["ts_open", "o", "h", "l", "c"])
    base = cs.frame_from_table(df)
    return cs.resample(base, minutes) if minutes > 1 else base


@pytest.fixture(scope="module")
def frame120() -> cs.BarFrame:
    return _spot_frame(_weekdays(dt.date(2027, 1, 4), 120))


@pytest.fixture(scope="module")
def ctx120(frame120: cs.BarFrame, lib: bc.BrickLibrary, cal: dict) -> bc.FeatureContext:
    return bc.FeatureContext(frame=frame120, lib=lib, cal=cal, available={"spot_1m"})


def _card(**over: object) -> dict:
    card = {"schema_version": "2.0", "card_id": "C-0001", "type": "RULE", "lane": "B",
            "provenance": "PERSONA:statistician-skeptic", "round": 1,
            "hypothesis": "Top-decile RSI(14) changes the fixed-horizon forward return",
            "event": "q90(rsi_14)", "side": "both", "target": "T1", "horizons": [3, 5],
            "baseline": "slot_matched", "cutoffs": "discovery_frozen", "data_needed": ["spot_1m"],
            "cost_series": "underlying", "prior_weight": 1.0, "expected_sign": "none"}
    card.update(over)
    return card


# ============================================================================================
# T-04 Leak Gate (V4-1, V4-2, F3.2, FM-40)
# ============================================================================================
@pytest.mark.parametrize("event", ["gt(close_{t+1}, 0)", "gt(lead(ret_1), 0)", "gt(day_high, 0)", "gt(future_ret, 0)",
                                   "gt(next_close, 0)", "lt(day_close, 0)"])
def test_t04_leaky_card_rejected_before_execution(event: str, lib: bc.BrickLibrary, cal: dict) -> None:
    cc = bc.compile_card(_card(event=event), lib, available={"spot_1m"}, cal=cal)
    assert cc.status == bc.STATUS_LEAK
    assert not cc.ok
    with pytest.raises(bc.CompileError):
        cc.evaluate(bc.FeatureContext(frame=None, lib=lib, cal=cal, available=set()))  # type: ignore[arg-type]


def test_t04_negative_lag_is_rejected(lib: bc.BrickLibrary, cal: dict) -> None:
    cc = bc.compile_card(_card(event="gt(lag(ret_1, -1), 0)"), lib, available={"spot_1m"}, cal=cal)
    assert cc.status == bc.STATUS_INVALID
    assert "look-ahead" in cc.reason


def test_t04_dynamic_leak_gate_catches_canary_and_passes_library(ctx120: bc.FeatureContext, lib: bc.BrickLibrary) -> None:
    spot_only = sorted(n for n, b in lib.bricks.items() if set(b.data_needed) <= {"spot_1m"})
    assert len(spot_only) >= 40
    failures = [n for n in spot_only if not bc.dynamic_leak_gate(n, ctx120)[0]]
    assert failures == []
    canary_lib = copy.deepcopy(lib)
    canary_lib.bricks["close_t_plus_1"] = bc.BrickDef("close_t_plus_1", "canary", "canary_lead_close", {}, "close_{t+1}",
                                                     "bar close", ("spot_1m",), "numeric")
    ctx = bc.FeatureContext(frame=ctx120.frame, lib=canary_lib, cal=ctx120.cal, available={"spot_1m"})
    ok, why = bc.dynamic_leak_gate("close_t_plus_1", ctx)
    assert not ok
    assert "look-ahead" in why


@pytest.mark.parametrize(("tokens", "available_at", "leaky"), [
    (["daily_metrics_DERIVED:EOD_INCLUSIVE"], "bar close", True),
    (["iv_daily_DERIVED"], "09:15 bar close", True),
    (["xcheck_csv"], "bar close", True),
    (["daily_metrics_DERIVED:INTRADAY_PREV_CLOSE"], "bar close", False),
    (["iv_daily_DERIVED"], "after 10:00 check_time", False),
    (["xcheck_csv"], "next session open", False),
])
def test_t04_vendor_availability_gate(tokens: list[str], available_at: str, leaky: bool, cal: dict) -> None:
    reasons = bc.availability_check(tokens, available_at, cal)
    assert bool(reasons) is leaky


def test_t04_eod_brick_proposal_leak_rejected(ctx120: bc.FeatureContext, cal: dict) -> None:
    eod = {"schema_version": "1.0", "proposal_type": "BRICK", "brick": "eod_vol_today",
           "formula": "(high - low) / ATR20", "available_at": "bar close",
           "data_needed": ["daily_metrics_DERIVED:EOD_INCLUSIVE"], "why": "test", "leak_gate": "pending"}
    verdict = bc.gate_brick_proposal(eod, ctx120, cal)
    assert not verdict.accepted
    assert "LEAK-REJECTED" in verdict.reason
    good = {**eod, "brick": "dist_prev_week_high_atr", "formula": "(close - prev_week_high) / ATR20",
            "data_needed": ["spot_1m"]}
    assert bc.gate_brick_proposal(good, ctx120, cal).accepted
    leaky = {**good, "brick": "leaky_future", "formula": "lag(close, 0) / ATR20"}
    assert not bc.gate_brick_proposal(leaky, ctx120, cal).accepted


# ============================================================================================
# T-04 Boundary Check (NG-1, NG-2, F4.7, FM-30)
# ============================================================================================
@pytest.mark.parametrize("hypothesis", [
    "Price reaches +1 ATR before -1 ATR after the event",
    "Buy when RSI crosses 30 and exit at target",
    "Place a stop-loss below the opening range",
    "Win-rate of the setup exceeds 60%",
    "First-touch of the upper barrier happens sooner",
    "Position sizing doubles on expiry days",
])
def test_t04_boundary_violations_rejected(hypothesis: str, lib: bc.BrickLibrary, cal: dict) -> None:
    cc = bc.compile_card(_card(hypothesis=hypothesis), lib, available={"spot_1m"}, cal=cal)
    assert cc.status == bc.STATUS_BOUNDARY


def test_t04_ap_register_is_boundary_clean_and_t7_off(lib: bc.BrickLibrary, cal: dict) -> None:
    statuses = {c["card_id"]: bc.compile_card(c, lib, available={"spot_1m"}, cal=cal).status for _, c in bc.load_ap_cards()}
    assert bc.STATUS_BOUNDARY not in statuses.values()
    assert bc.STATUS_LEAK not in statuses.values()
    t7 = bc.compile_card(_card(target="T7"), lib, available={"spot_1m"}, cal=cal)
    assert t7.status == bc.STATUS_BOUNDARY
    assert bc.compile_card(_card(target="T7"), lib, available={"spot_1m"}, cal=cal, t7_enabled=True).ok


def test_t04_schema_and_depth_gates(lib: bc.BrickLibrary, cal: dict) -> None:
    lane_m = _card(lane="M")
    assert bc.compile_card(lane_m, lib, available={"spot_1m"}, cal=cal).status == bc.STATUS_INVALID
    deep = _card(event="AND(gt(rsi_14, 50), gt(er_20, 0.3), gt(adx_14, 20), gt(ret_5, 0))")
    res = bc.compile_card(deep, lib, available={"spot_1m"}, cal=cal)
    assert res.status == bc.STATUS_INVALID
    assert "3 bricks" in res.reason
    unknown = bc.compile_card(_card(event="gt(not_a_brick, 1)"), lib, available={"spot_1m"}, cal=cal)
    assert unknown.status == bc.STATUS_INVALID
    missing = bc.compile_card(_card(event="q90(pcr)"), lib, available={"spot_1m"}, cal=cal)
    assert missing.status == bc.STATUS_NA
    assert "chain_1m" in missing.missing


def test_t04_cutoffs_frozen_on_discovery_only(ctx120: bc.FeatureContext, lib: bc.BrickLibrary, cal: dict) -> None:
    cc = bc.compile_card(_card(event="q90(rsi_14)"), lib, available={"spot_1m"}, cal=cal)
    fr = ctx120.frame
    disc = np.arange(fr.n_days) < 70
    with pytest.raises(bc.CompileError):
        cc.evaluate(ctx120)
    mask_fit, _ = cc.evaluate(ctx120, disc)
    cut = cc.cutoffs["$.q90[0]" if "$.q90[0]" in cc.cutoffs else next(iter(cc.cutoffs))]
    rsi = ctx120.brick("rsi_14")
    disc_bars = disc[fr.day_idx] & np.isfinite(rsi)
    assert cut == pytest.approx(float(np.quantile(rsi[disc_bars], 0.9)))
    assert cut != pytest.approx(float(np.quantile(rsi[np.isfinite(rsi)], 0.9)))
    mask_frozen, _ = cc.evaluate(ctx120)
    assert np.array_equal(mask_fit, mask_frozen)
    h1 = cc.card_hash()
    assert len(h1) == 64
    assert h1 == cc.card_hash()


# ============================================================================================
# T-04 NS (V5-1), MODEL (V5-3), Lane M (V5-4), Persona (FR-21)
# ============================================================================================
def test_t04_rsi71_clone_excluded_from_novelty(ctx120: bc.FeatureContext, lib: bc.BrickLibrary, cal: dict) -> None:
    disc = np.ones(ctx120.frame.n_days, bool)
    ap = bc.compile_card(_card(card_id="AP27-7030-C", lane="AP", event="gt(rsi_14, 70)"), lib, available={"spot_1m"}, cal=cal)
    clone = bc.compile_card(_card(card_id="C-0071", event="gt(rsi_14, 71)"), lib, available={"spot_1m"}, cal=cal)
    novel = bc.compile_card(_card(card_id="C-0099", event="AND(eq(month_start, 1), at_checkpoint(cp_1100))"), lib,
                            available={"spot_1m"}, cal=cal)
    fr = ctx120.frame

    def days(cc: bc.CompiledCard) -> set[str]:
        m, _ = cc.evaluate(ctx120, disc)
        return {str(d) for d in fr.dates[m]}

    ap_days, clone_days, novel_days = days(ap), days(clone), days(novel)
    assert cs.jaccard(clone_days, ap_days) >= 0.5
    ns = sb.novelty_score({"C-0071": clone_days, "C-0099": novel_days}, {"AP27-7030-C": ap_days})
    assert ns["ns"] == pytest.approx(0.5)
    assert [c["card_id"] for c in ns["clones"]] == ["C-0071"]


def _u_shape_frame(seed: int = 7) -> tuple[cs.BarFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    days = _weekdays(dt.date(2027, 1, 4), 160)
    per = 60
    feat = rng.uniform(-1, 1, len(days) * per)
    drift = np.where(np.abs(feat) > 0.6, 0.6, -0.3)
    rows, price, k = [], 50000.0, 0
    for day in days:
        for i in range(per):
            o = price
            c = o + drift[k - 1] * 10 + rng.normal(0, 10) if k > 0 else o + rng.normal(0, 10)
            m = 9 * 60 + 15 + i
            ts = pd.Timestamp(dt.datetime(day.year, day.month, day.day, m // 60, m % 60), tz="Asia/Kolkata")
            rows.append((ts, o, max(o, c) + 3, min(o, c) - 3, c))
            price, k = c, k + 1
    return cs.frame_from_table(pd.DataFrame(rows, columns=["ts_open", "o", "h", "l", "c"])), feat


@pytest.mark.skipif(not bc.sklearn_available(), reason="NOT-RUN(ENV): scikit-learn optional extra absent")
def test_t04_model_card_detects_u_shape_with_fold_local_cutoffs(lib: bc.BrickLibrary, cal: dict) -> None:
    fr, feat = _u_shape_frame()
    ctx = bc.FeatureContext(frame=fr, lib=lib, cal=cal, available={"spot_1m"}, extra_bricks={"u_feat": feat})
    card = _card(card_id="C-0300", type="MODEL", event="MODEL(gbdt, u_feat)", cutoffs="fold_local", horizons=[1])
    cc = bc.compile_card(card, lib, available={"spot_1m"}, cal=cal, extra_bricks={"u_feat"})
    assert cc.ok, cc.reason
    label = cs.outcome(fr, "T1", 1)["atr"]
    disc = np.ones(fr.n_days, bool)
    mask, _ = cc.evaluate(ctx, disc, label)
    auc = cc.cutoffs["oof_auc"]
    assert auc["lo"] > 0.5
    assert auc["informative"]
    fold_cuts = [cc.cutoffs[k] for k in ("fold0", "fold1", "fold2")]
    assert len(set(np.round(fold_cuts, 12))) == 3
    assert cc.cutoffs["frozen"] not in fold_cuts
    assert mask.sum() > 0
    frozen_mask, _ = cc.evaluate(ctx)
    assert frozen_mask.sum() > 0


def test_t04_model_card_not_run_env_without_sklearn(lib: bc.BrickLibrary, cal: dict) -> None:
    card = _card(card_id="C-0301", type="MODEL", event="MODEL(logistic, rsi_14, er_20)", cutoffs="fold_local")
    cc = bc.compile_card(card, lib, available={"spot_1m"}, cal=cal, sklearn_ok=False)
    assert cc.status == bc.STATUS_ENV
    assert "NOT-RUN(ENV)" in cc.reason


def test_t04_lane_m_gradient_consistent_vs_flat_refuted() -> None:
    rng = np.random.default_rng(42)
    n = 600
    dose = rng.uniform(0, 3, n)
    days = np.repeat(np.arange(200), 3)
    sign = sb.gradient_sign("smaller distance -> higher stall rate")
    assert sign == -1
    good_res = 0.6 - 0.25 * dose + rng.normal(0, 0.3, n)
    control = rng.normal(0, 0.3, 300)
    ok = sb.mechanism_check(good_res, dose, days, sign, control)
    assert ok["status"] == "MECHANISM-CONSISTENT"
    flat = 0.2 + rng.normal(0, 0.3, n)
    bad = sb.mechanism_check(flat, dose, days, sign, control)
    assert bad["status"] == "MECHANISM-REFUTED"
    assert "gradient" in bad["reason"]
    leaky_control = sb.mechanism_check(good_res, dose, days, sign, control + 0.3)
    assert leaky_control["status"] == "MECHANISM-REFUTED"
    assert sb.demote_grade("A") == "B"
    assert sb.demote_grade("Ω-A") == "Ω-B"


def test_t04_duplicate_persona_rejected_distinct_accepted() -> None:
    base = [set(range(0, 100)), set(range(10, 110))]
    near = [set(range(31, 131)), set(range(38, 138))]
    far = [set(range(500, 600))]
    proposal = {"schema_version": "1.0", "proposal_type": "PERSONA", "persona": "Flow-Watcher",
                "distance_from_existing": "median event-day Jaccard < 0.30 vs all existing personas"}
    dup = bc.gate_persona_proposal(proposal, near, {"Options-Greeks": base})
    assert not dup.accepted
    assert "cosmetic rename" in dup.reason
    assert bc.gate_persona_proposal(proposal, far, {"Options-Greeks": base}).accepted


def test_t04_proposals_written_safely(tmp_path: Path, ctx120: bc.FeatureContext, lib: bc.BrickLibrary, cal: dict) -> None:
    prop_dir = tmp_path / "proposals"
    brick = {"schema_version": "1.0", "proposal_type": "BRICK", "brick": "dist_prev_week_high_atr",
             "formula": "(close - prev_week_high) / ATR20", "available_at": "bar close", "data_needed": ["spot_1m"],
             "why": "Weekly level reaction", "leak_gate": "pending"}
    verdict = bc.gate_brick_proposal(brick, ctx120, cal)
    path = bc.write_proposal(brick, verdict, prop_dir, 1)
    assert path.parent == prop_dir
    assert path.is_file()
    pending = bc.write_bricks_pending(tmp_path / "bricks_pending.yaml", [brick])
    lib2 = bc.BrickLibrary.load(pending_path=pending)
    assert lib2.bricks["dist_prev_week_high_atr"].pending
    target = {"schema_version": "1.0", "proposal_type": "TARGET", "target": "T9_next_bar_range_ratio",
              "metric": "log(range_{t+1..t+h} / range_{t-h+1..t})", "boundary_check": "fixed horizon; zero hold/exit/first-touch",
              "tag": "STRUCTURE-ONLY"}
    assert bc.gate_target_proposal(target).accepted
    bad_target = {**target, "metric": "share of +1 ATR before -1 ATR"}
    assert not bc.gate_target_proposal(bad_target).accepted
    mech = {"schema_version": "1.0", "proposal_type": "MECHANISM", "mechanism": "Round-number stop clusters",
            "dose_variable": "dist_round_100", "negative_control": "bars > 0.5 ATR20 from 100-multiples"}
    assert bc.gate_mechanism_proposal(mech, lib).accepted
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.symlink(outside, tmp_path / "linked_proposals")
    with pytest.raises(bnf.PathViolation):
        bc.write_proposal(target, bc.gate_target_proposal(target), tmp_path / "linked_proposals", 2)


# ============================================================================================
# FR-10 tags, EB shrinkage and cost model known answers
# ============================================================================================
def test_t04_cost_model_known_answers(cfg: dict) -> None:
    model = sb.CostModel.from_config(cfg)
    day = dt.date(2026, 11, 20)
    assert model.leg_charges(Decimal("426"), 30, "buy", day) == Decimal("41.14")
    expected = {("200", 1): "5.23", ("400", 1): "5.71", ("800", 1): "6.65", ("400", 3): "4.13"}
    for (prem, lots), want in expected.items():
        got = model.option_cost_pts(prem, lots, 30, day).quantize(Decimal("0.01"))
        assert got == Decimal(want), (prem, lots, got)
    und = model.underlying_cost_pts(model.option_cost_pts("400", 1, 30, day))
    assert Decimal("11.3") < und < Decimal("11.5")
    assert model.stt_pct(dt.date(2026, 3, 31)) == Decimal("0.10")
    assert model.stt_pct(dt.date(2026, 4, 1)) == Decimal("0.15")
    with pytest.raises(bnf.InputError):
        model.option_cost_pts("400", 1, 0, day)


def test_t04_tradability_tags_from_deflated_lift_only(cfg: dict) -> None:
    costs = {"underlying": 2.0, "option": None, "option_buyer": None}
    atr_pts = 10.0
    assert sb.tradability_tag("T1", 0.45, atr_pts, "underlying", costs, cfg)["tag"] == "T1"
    assert sb.tradability_tag("T1", 0.25, atr_pts, "underlying", costs, cfg)["tag"] == "T2"
    t3 = sb.tradability_tag("T1", -0.1, atr_pts, "underlying", costs, cfg)
    assert t3["tag"] == "T3"
    assert t3["label"] == "REAL BUT UNTRADABLE"
    unknown = sb.tradability_tag("T1", 0.9, atr_pts, "option_buyer", costs, cfg)
    assert unknown["tag"] == "T?"
    assert "USER TO CONFIRM" in unknown["note"]
    assert sb.tradability_tag("T2", 3.0, atr_pts, "structure_only", costs, cfg)["tag"] == "STRUCTURE-ONLY"


def test_t04_empirical_bayes_known_answer() -> None:
    lifts = [0.30, -0.10, 0.05, 0.20]
    ses = [0.10, 0.10, 0.10, 0.10]
    tau2 = sb.eb_tau2(lifts, ses)
    assert tau2 == pytest.approx((0.09 + 0.01 + 0.0025 + 0.04) / 4 - 0.01)
    assert sb.deflate(0.30, 0.10, tau2) == pytest.approx(0.30 * tau2 / (tau2 + 0.01))
    records = [{"family": "T1", "lift": float(x), "se": 0.1, "status": "EVALUATED"} for x in np.linspace(-0.5, 0.5, 60)]
    records += [{"family": "T2", "lift": 0.1, "se": 0.1, "status": "DISCARDED"}]
    fits = sb.fit_families(records, haircut=0.6)
    assert fits["T1"].method == "EB"
    assert fits["T1"].n_cards == 60
    assert fits["T2"].method == "HAIRCUT"
    assert sb.deflated_lift(0.1, 0.1, fits["T2"]) == pytest.approx(0.06)
    assert sb.haircut_ratio([0.5, 0.5], [0.3, 0.4]) == pytest.approx(0.7)


# ============================================================================================
# T-05 horizon log, X-ray, verdicts (V5-11, F5.1..F5.5, FR-06, FR-07)
# ============================================================================================
def test_t05_h50_on_15m_bars_logged_not_dropped(tmp_path: Path, cfg: dict) -> None:
    fr15 = _spot_frame(_weekdays(dt.date(2027, 1, 4), 40), minutes=15)
    assert fr15.bars_per_day == 25
    ctx = cs.Ctx.from_config(cfg)
    runner = cs.TestRunner(ctx, fr15)
    mask = np.zeros(fr15.n, bool)
    mask[::7] = True
    out = runner.run_batch([{"test_id": "C-0500", "mask": mask, "target": "T1", "horizons": [1, 3, 6, 50]}])
    assert [r.horizon for r in out] == [1, 3, 6]
    skipped = [r for r in ctx.registry if r["status"] == "SKIPPED"]
    assert [r["horizon"] for r in skipped] == [50]
    assert "exceeds bars_per_day/4" in skipped[0]["reason"]
    counts = ctx.registry_counts()
    assert counts["planned"] == counts["executed"] + counts["logged_skipped"] == 4
    path = ctx.write_test_registry(tmp_path / "tables" / "test_registry.csv")
    text = path.read_text()
    assert "exceeds bars_per_day/4" in text
    ctx.registry.append({"test_id": "x", "status": "LOST"})
    with pytest.raises(bnf.InvalidPipeline):
        ctx.write_test_registry(tmp_path / "tables" / "bad.csv")


def test_t05_slot_event_uses_other_slots_vol_matched_baseline(frame120: cs.BarFrame, ctx120: bc.FeatureContext) -> None:
    lunch = ctx120.window_mask("lunch_lull")
    res = cs.event_study(frame120, lunch, target="T2", horizon=3, baseline="other_slots_vol_matched", bootstrap=False)
    raw = cs.outcome(frame120, "T2", 3)["atr"]
    other = ~lunch & np.isfinite(raw)
    assert res.n_events > 0
    assert np.nanmean(raw[other]) == pytest.approx(np.nanmean(raw[other]))
    assert np.isfinite(res.baseline)
    slot_res = cs.event_study(frame120, lunch, target="T2", horizon=3, baseline="slot_matched", bootstrap=False)
    assert slot_res.baseline != pytest.approx(res.baseline)


@pytest.fixture(scope="module")
def xray250(cfg: dict, cal: dict, lib: bc.BrickLibrary) -> tuple[xp.XrayResult, cs.Ctx]:
    days = _weekdays(dt.date(2027, 1, 4), 250)
    fr = _spot_frame(days)
    ctx = bc.FeatureContext(frame=fr, lib=lib, cal=cal, available={"spot_1m"})
    run = cs.Ctx.from_config(cfg)
    inp = xp.XrayInputs(frame=fr, ctx=ctx, cfg=cfg, run_ctx=run, fit_day_mask=np.ones(fr.n_days, bool),
                        eras={str(d): qm.era_of(d, cal) for d in days},
                        costs={"underlying": None, "option": None, "option_buyer": None})
    return xp.run_xray(inp, lib), run


def test_t05_spot_only_250_days_na_approaches_and_counts(xray250: tuple[xp.XrayResult, cs.Ctx]) -> None:
    res, run = xray250
    na = sorted(r["approach_id"] for r in res.verdicts_11a if r["verdict"] == bnf.NA_DATA_MISSING)
    want = ["AP01", "AP02", "AP03", "AP04", "AP05", "AP06", "AP07", "AP11", "AP19", "AP23", "AP24", "AP25", "AP26"]
    assert na == want
    for row in res.verdicts_11a:
        if row["verdict"] == bnf.NA_DATA_MISSING:
            assert row["missing"], row["approach_id"]
    assert len(res.verdicts_11a) == 36
    rollups = {r["approach_id"]: r for r in res.verdicts_11a if r["approach_id"] in ("AP32", "AP33")}
    assert all(r["runnable_tests"] == 0 for r in rollups.values())
    assert not any(t.get("approach_id") in ("AP32", "AP33") for t in res.tests)
    assert len(res.verdicts_11b) == 10
    for g in res.verdicts_11b:
        members = [r for r in res.verdicts_11a if r["group"] == g["group"]]
        assert g["members_S"] + g["members_C"] + g["members_I"] + g["members_NA"] == len(members)
    assert not any("PROXY" in str(r.get("best_test", "")) for r in res.verdicts_11a)
    counts = run.registry_counts()
    assert counts["planned"] == counts["executed"] + counts["logged_skipped"]
    assert not any(r["verdict"] == "SUPPORTED FOR HYPOTHESIS" for r in res.verdicts_11a)


def test_t05_every_phase_reported_and_regime_flip_rule(xray250: tuple[xp.XrayResult, cs.Ctx]) -> None:
    res, _ = xray250
    phases = {p.phase: p for p in res.phases}
    for ph in ["1.1", "1.3", "1.8", "2.1", "2.2", "2.7", "3.1", "3.8", "4.1", "4.9", "5.1", "5.3", "6.1", "6.5",
               "7.3", "7.5", "7.6", "8.1", "8.2", "8.6"]:
        assert ph in phases, ph
    assert phases["3.1"].status == bnf.NA_DATA_MISSING
    assert "fut_volume" in phases["3.1"].note
    assert phases["4.8"].status == bnf.NA_DATA_MISSING
    flip = next(r for r in phases["6.5"].rows if r["cell"] == "verdict")
    assert flip["label"] == "NO RELIABLE FLIP"
    assert len(phases["7.3"].rows) <= 45
    vr = [r for r in phases["2.2"].rows if str(r.get("lag", "")).startswith("VR(")]
    assert [r["h"] for r in vr] == [2, 4, 8, 16]
    with_events = {t.get("era_tag") for t in res.tests if t.get("status") == "EXECUTED" and t["n_events"] > 0}
    without_events = {t.get("era_tag") for t in res.tests if t.get("status") == "EXECUTED" and t["n_events"] == 0}
    assert with_events == {"ERA-B-ONLY"}
    assert without_events <= {"ERA-UNKNOWN"}


def test_t05_2a3_rule_ordering(cfg: dict) -> None:
    approach = {"approach_id": "AP99", "group": "G6", "name": "synthetic"}
    base = {"approach_id": "AP99", "status": "EXECUTED", "side": "continuation", "test_id": "AP99-C", "horizon": 5,
            "se": 0.05, "effective_n": 80, "gate_mde": 0.18, "q_disc": 0.01, "cost_series": "underlying"}
    costs = {"underlying": 3.0}
    supported = xp.verdict_2a3(approach, [{**base, "grade": "Ω-A", "t_tag": "T1", "skeptic_ran": 4}], costs=costs,
                               atr_points=10.0, day_limited=False)
    assert supported["verdict"] == "SUPPORTED FOR HYPOTHESIS"
    limited = xp.verdict_2a3(approach, [{**base, "grade": "Ω-A", "t_tag": "T1", "skeptic_ran": 4}], costs=costs,
                             atr_points=10.0, day_limited=True)
    assert limited["verdict"] != "SUPPORTED FOR HYPOTHESIS"
    untradable = xp.verdict_2a3(approach, [{**base, "grade": "Ω-B", "t_tag": "T3", "skeptic_ran": 4}], costs=costs,
                                atr_points=10.0, day_limited=False)
    assert untradable["verdict"] == "CONTRADICTED"
    assert untradable["label"] == "REAL BUT UNTRADABLE"
    contra = xp.verdict_2a3(approach, [{**base, "grade": "D", "t_tag": "T?"}], costs=costs, atr_points=10.0, day_limited=False)
    assert contra["verdict"] == "CONTRADICTED"
    assert "gate-adjusted MDE" in contra["reason"]
    stat_only = xp.verdict_2a3(approach, [{**base, "grade": "D"}], costs={"underlying": None}, atr_points=10.0, day_limited=False)
    assert stat_only["label"] == "STAT-ONLY"
    weak = xp.verdict_2a3(approach, [{**base, "grade": "D", "effective_n": 12}], costs=costs, atr_points=10.0, day_limited=False)
    assert weak["verdict"] == "INCONCLUSIVE"
    na = xp.verdict_2a3(approach, [{"approach_id": "AP99", "status": bnf.NA_DATA_MISSING, "missing": ["chain_1m"],
                                    "reason": "x"}], costs=costs, atr_points=10.0, day_limited=False)
    assert na["verdict"] == bnf.NA_DATA_MISSING
    assert na["missing"] == "chain_1m"


def test_t03_era_a_option_structure_card_is_era_specific(cal: dict) -> None:
    days = _weekdays(dt.date(2024, 9, 2), 80)
    fr = _spot_frame(days, bars=60, minutes=1)
    mask = np.zeros(fr.n, bool)
    mask[::50] = True
    res = cs.event_study(fr, mask, target="T1", horizon=5, bootstrap=False)
    eras = {str(d): qm.era_of(d, cal) for d in days}
    assert set(eras.values()) == {"A", "B1"}
    option_card = {"data_needed": ["spot_1m", "opt_1m", "expiry_calendar"]}
    assert xp.era_consistency(option_card, res, eras, cal) == "ERA-SPECIFIC"
    price_card = {"data_needed": ["spot_1m"]}
    assert xp.era_consistency(price_card, res, eras, cal) in ("ERA-CONSISTENT", "ERA-SPECIFIC")
