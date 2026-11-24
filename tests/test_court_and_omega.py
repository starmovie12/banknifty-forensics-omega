"""FILE-045 · Court, file-level seal, diversity, alpha spending, graduation, Playground (T-06).

Covers Ω-4 (seal), Ω-5 (clone collapse), Ω-8 (alpha_k), Ω-11 (graduation), FR-09 (Playground
modes, Idea Ledger, budget checkpoint, resume), FR-11 (docket freeze, power pre-check, B_court,
underpowered/rejected/confirmed, exposure), FM-17/31/32/34/36/37/39, plus the Court-level null FPR.
Ω-10 (``omega_sim``) is tested in FILE-047 (DEF-20).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bn_forensics as bnf
from bn_forensics import bricks_compiler as bc
from bn_forensics import core_stats as cs
from bn_forensics import court as ct
from bn_forensics import playground as pg
from bn_forensics import quartermaster as qm
from bn_forensics import split_seal as ss
from bn_forensics.shrinkage_basis import Candidate


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


def _planted_frame(days: list[dt.date], *, event_rate: float, effect_pts: float, seed: int = 11,
                   per_day: int = 60, h: int = 3) -> tuple[cs.BarFrame, np.ndarray]:
    """1-minute bars with an event flag; after each event bar the next ``h`` bars drift by ``effect_pts``."""
    rng = np.random.default_rng(seed)
    rows, flags = [], []
    price = 50000.0
    for day in days:
        has_event = rng.random() < event_rate
        ev_bar = int(rng.integers(25, per_day - h - 2)) if has_event else -1
        drift = 0.0
        remaining = 0
        for i in range(per_day):
            o = price
            step = rng.normal(0, 10)
            if remaining > 0:
                step += drift
                remaining -= 1
            c = o + step
            m = 9 * 60 + 15 + i
            ts = pd.Timestamp(dt.datetime(day.year, day.month, day.day, m // 60, m % 60), tz="Asia/Kolkata")
            rows.append((ts, o, max(o, c) + abs(rng.normal(0, 3)), min(o, c) - abs(rng.normal(0, 3)), c))
            flags.append(1.0 if i == ev_bar else 0.0)
            if i == ev_bar:
                drift, remaining = effect_pts / h, h
            price = c
    frame = cs.frame_from_table(pd.DataFrame(rows, columns=["ts_open", "o", "h", "l", "c"]))
    return frame, np.asarray(flags)


def _cand(cid: str, days: set[str], *, lane: str = "B", dl: float = 0.3, n_days: int = 200) -> Candidate:
    return Candidate(card_id=cid, card_hash=bnf.sha256_text(cid), lane=lane, sign=1, target="T1", horizon=3,
                     prior_weight=1.0, deflated_lift=dl, event_days=frozenset(days), playground_days=n_days)


# ============================================================================================
# Ω-5 diversity, Ω-8 alpha spending, B_court resolution
# ============================================================================================
def test_t06_fifteen_clones_collapse_to_one_slot() -> None:
    base = {f"2027-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(0, 180, 3)}
    clones = [_cand(f"CLONE-{i:02d}", base | {f"2028-01-{i + 1:02d}"}, dl=0.50 - i * 0.001) for i in range(15)]
    rng = np.random.default_rng(3)
    pool_days = [f"2027-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(300)]
    distinct = [_cand(f"DIST-{i:02d}", set(rng.choice(pool_days, 40, replace=False)), dl=0.30 - i * 0.001) for i in range(10)]
    chosen = ct.greedy_diverse(clones + distinct, 20, jaccard_max=0.5)
    ids = [c.card_id for c, _ in chosen]
    assert sum(cid.startswith("CLONE") for cid in ids) == 1
    assert ids[0] == "CLONE-00"
    assert all(j <= 0.5 for _, j in chosen)
    for i, (a, _) in enumerate(chosen):
        for b, _ in chosen[i + 1:]:
            assert cs.jaccard(a.event_days, b.event_days) <= 0.5
    with pytest.raises(bnf.InputError):
        ct.greedy_diverse(distinct, 51)


def test_t06_lane_m_slot_reservation() -> None:
    rng = np.random.default_rng(5)
    days = [f"D{i:04d}" for i in range(500)]
    strong = [_cand(f"B-{i:02d}", set(rng.choice(days, 50, replace=False)), dl=0.9 - i * 0.01) for i in range(20)]
    lane_m = [_cand(f"M-{i:02d}", set(rng.choice(days, 50, replace=False)), lane="M", dl=0.05) for i in range(6)]
    chosen = ct.greedy_diverse(strong + lane_m, 20, lane_m_reserve=5)
    assert sum(c.lane == "M" for c, _ in chosen) == 5
    assert len(chosen) == 20


def test_t06_alpha_spending_and_b_court() -> None:
    expected = {1: 0.030396, 2: 0.007599, 3: 0.003377}
    for k, a in expected.items():
        assert cs.alpha_k(k) == pytest.approx(a, abs=5e-7)
    assert [cs.b_court(cs.alpha_k(k), 20) for k in (1, 2, 3, 4)] == [10000, 26319, 59218, 105276]
    assert cs.alpha_k(3) / 20 == pytest.approx(1.69e-4, rel=0.01)
    assert sum(cs.alpha_k(k) for k in range(1, 10_000)) < 0.05
    with pytest.raises(bnf.InputError):
        cs.alpha_k(0)


def test_t06_exact_p_1e5_confirmable_at_session_3(cfg: dict) -> None:
    rng = np.random.default_rng(42)
    n = 60
    base = rng.normal(0, 1, n)
    z_target = float(cs.z_upper(1e-5))
    x = base - base.mean() + z_target * base.std() / math.sqrt(n)
    p, B = ct.court_pvalue(x, 1, 3, 20, cfg)
    assert B == 59218
    assert p < cs.alpha_k(3) / 20
    p_low_b = cs.sign_flip_pvalue(x, 1, 5000, 45)
    assert p_low_b >= 1 / 5001
    assert cs.alpha_k(3) / 20 < 1 / 5001


def test_t06_court_null_fpr_at_most_5_percent(cfg: dict) -> None:
    worlds, k_cards, n_days = 200, 20, 60
    confirmed_worlds = 0
    threshold = cs.alpha_k(1) / k_cards
    for w in range(worlds):
        rng = np.random.default_rng(10_000 + w)
        any_hit = False
        for _ in range(k_cards):
            x = rng.standard_t(5, n_days)
            p, _ = ct.court_pvalue(x, 1 if rng.random() < 0.5 else -1, 1, k_cards, cfg)
            if p < threshold:
                any_hit = True
                break
        confirmed_worlds += any_hit
    fpr = confirmed_worlds / worlds
    lo, _hi = cs.wilson_ci(confirmed_worlds, worlds)
    assert fpr <= 0.05
    assert lo <= 0.0304


# ============================================================================================
# Docket freeze, power pre-check, tamper (F32), seal (F31), graduation (Ω-11, F37)
# ============================================================================================
@pytest.fixture
def court_setup(tmp_path: Path, cfg: dict, cal: dict, lib: bc.BrickLibrary) -> dict:
    days = _weekdays(dt.date(2027, 1, 4), 300)
    frame, flags = _planted_frame(days, event_rate=0.55, effect_pts=40.0)
    manifest = ss.make_split(days, cfg, cal, data_version="f" * 64)
    pg_days = np.array(manifest.playground_days, dtype="datetime64[D]")
    court_days = np.array(manifest.court_days, dtype="datetime64[D]")
    pg_frame = frame.subset_days(pg_days)
    court_frame = frame.subset_days(court_days)
    flag_by_key = dict(zip(zip(frame.dates.astype(str), frame.minute, strict=True), flags, strict=True))

    def ctx_for(fr: cs.BarFrame) -> bc.FeatureContext:
        feat = np.array([flag_by_key[(str(d), m)] for d, m in zip(fr.dates, fr.minute, strict=True)])
        return bc.FeatureContext(frame=fr, lib=lib, cal=cal, available={"spot_1m"}, extra_bricks={"planted_feat": feat})

    card = {"schema_version": "2.0", "card_id": "C-0800", "type": "RULE", "lane": "B", "provenance": "PERSONA:selftest",
            "round": 1, "hypothesis": "Planted synthetic feature precedes a fixed-horizon forward-return lift",
            "event": "gt(planted_feat, 0.5)", "side": "both", "target": "T1", "horizons": [3], "baseline": "slot_matched",
            "cutoffs": "discovery_frozen", "data_needed": ["spot_1m"], "cost_series": "underlying", "prior_weight": 1.0,
            "expected_sign": "positive"}
    pctx = ctx_for(pg_frame)
    cc = bc.compile_card(card, lib, available={"spot_1m"}, cal=cal, extra_bricks={"planted_feat"})
    assert cc.ok, cc.reason
    run = cs.Ctx.from_config(cfg)
    mask, direction = cc.evaluate(pctx, np.ones(pg_frame.n_days, bool))
    res = cs.event_study(pg_frame, mask, direction=direction, target="T1", horizon=3, ctx=run)
    cand = Candidate(card_id="C-0800", card_hash=cc.card_hash(), lane="B", sign=res.sign, target="T1", horizon=3,
                     prior_weight=1.0, deflated_lift=res.lift_atr * 0.8, event_days=frozenset(res.event_day_list()),
                     playground_days=pg_frame.n_days, compiled=cc, card=card)
    return {"days": days, "manifest": manifest, "pg_frame": pg_frame, "court_frame": court_frame, "ctx_for": ctx_for,
            "cc": cc, "cand": cand, "run": run, "tmp": tmp_path, "playground_result": res}


def test_t06_docket_freeze_confirms_planted_card_and_graduates(court_setup: dict, cfg: dict) -> None:
    s = court_setup
    docket = ct.freeze_docket([s["cand"]], s["manifest"], cfg, K=1, run_id="run_t06", path=s["tmp"] / "court_docket.json")
    assert docket["status"] == "FROZEN"
    assert docket["session"] == 1
    assert docket["B_court"] == 10000
    assert docket["threshold"] == pytest.approx(cs.alpha_k(1) / 1)
    on_disk = json.loads((s["tmp"] / "court_docket.json").read_text())
    assert on_disk["docket_hash"] == ct.docket_hash(on_disk)
    ct.verify_docket(s["tmp"] / "court_docket.json", {"C-0800": s["cc"]}, s["manifest"])
    verdicts = ct.run_session(docket, {"C-0800": s["cc"]}, s["ctx_for"](s["court_frame"]), s["manifest"], cfg, s["run"],
                              tables_dir=s["tmp"] / "tables")
    v = verdicts[0]
    assert v.verdict == "COURT-CONFIRMED", v.row
    assert v.exposure == "VIRGIN-COURT"
    assert v.p < docket["threshold"]
    ledger = pd.read_csv(s["tmp"] / "tables" / "event_ledger.csv")
    assert list(ledger.columns) == list(ct.EVENT_LEDGER_FIELDS)
    assert len(ledger) == v.n_events
    assert set(ledger["date"]) <= set(s["manifest"].court_days)
    first_court = s["manifest"].court_days[0]
    ss.graduate(s["manifest"])
    assert s["manifest"].session_k == 1
    assert first_court in s["manifest"].playground_days
    with pytest.raises((bnf.InvalidPipeline, bnf.InputError)):
        ct.run_session(docket, {"C-0800": s["cc"]}, s["ctx_for"](s["court_frame"]), s["manifest"], cfg, s["run"])


def test_t06_docket_tamper_is_mode_invalid(court_setup: dict, cfg: dict) -> None:
    s = court_setup
    path = s["tmp"] / "court_docket.json"
    ct.freeze_docket([s["cand"]], s["manifest"], cfg, K=1, run_id="run_t06", path=path)
    doc = json.loads(path.read_text())
    doc["cards"][0]["sign"] = -doc["cards"][0]["sign"]
    path.write_text(json.dumps(doc))
    with pytest.raises(bnf.DocketTamper) as exc:
        ct.verify_docket(path, {"C-0800": s["cc"]}, s["manifest"])
    assert exc.value.exit_code == 4
    assert exc.value.tripwire_id == "F32"
    ct.freeze_docket([s["cand"]], s["manifest"], cfg, K=1, run_id="run_t06", path=path)
    mutated_cc = bc.CompiledCard(dict(s["cc"].card, hypothesis="changed after freeze"), s["cc"].status, ast=s["cc"].ast,
                                 cutoffs=dict(s["cc"].cutoffs))
    with pytest.raises(bnf.DocketTamper):
        ct.verify_docket(path, {"C-0800": mutated_cc}, s["manifest"])


def test_t06_power_deferred_opens_no_court_file(court_setup: dict, cfg: dict, cal: dict, tmp_path: Path, store_writer, make_bars) -> None:
    s = court_setup
    weak = Candidate(card_id="C-0801", card_hash="a" * 64, lane="B", sign=1, target="T1", horizon=3, prior_weight=1.0,
                     deflated_lift=0.2, event_days=frozenset(s["manifest"].playground_days[:20]),
                     playground_days=len(s["manifest"].playground_days))
    docket = ct.freeze_docket([weak], s["manifest"], cfg, K=1, run_id="run_t06", path=s["tmp"] / "deferred.json")
    assert docket["status"] == "POWER-DEFERRED"
    assert docket["power_precheck"]["passed"] is False
    assert docket["power_precheck"]["median_event_days"] < 30
    k_before = s["manifest"].session_k
    with pytest.raises(bnf.InputError):
        ct.run_session(docket, {}, s["ctx_for"](s["court_frame"]), s["manifest"], cfg, s["run"])
    assert s["manifest"].session_k == k_before
    table = make_bars(s["days"][:40], rng=np.random.default_rng(2), bars_per_day=3)
    store_writer(table, tmp_path / "store", "spot_1m", week=None)
    idx = qm.StoreIndex.scan([tmp_path / "store"])
    small_cfg = {**cfg, "split": {**cfg["split"], "warmup_days": 2}}
    m = ss.make_split(s["days"][:40], small_cfg, cal, data_version=idx.data_version())
    seal = ss.CourtSeal(m)
    seal.playground_reader(idx).read("spot_1m", "BANKNIFTY", dates=m.dates("playground_days"))
    assert seal.court_reads_during_playground() == 0
    assert not any(e["court"] for e in seal.audit)


def test_t06_underpowered_rejected_and_exposed_verdicts(cfg: dict) -> None:
    rng = np.random.default_rng(9)
    x25 = rng.normal(0.8, 1.0, 25)
    p25, _ = ct.court_pvalue(x25, 1, 1, 20, cfg)
    assert cfg["court"]["event_days_min"] > 25
    strong_wrong = rng.normal(-1.2, 1.0, 80)
    p_two = cs.sign_flip_pvalue(strong_wrong, 1, 10000, 1, two_sided=True)
    p_one, _ = ct.court_pvalue(strong_wrong, 1, 1, 20, cfg)
    assert p_two < 0.001
    assert p_one > 0.99
    days = [f"2026-0{1 + i // 28}-{1 + i % 28:02d}" for i in range(50)]
    exposure = {d: ("EXPOSED" if i < 35 else "VIRGIN") for i, d in enumerate(days)}
    tag, share = ct.exposure_tag(days, exposure)
    assert tag == "EXPOSED-COURT"
    assert share == pytest.approx(0.7)
    tag2, _ = ct.exposure_tag(days[20:], exposure)
    assert tag2 == "VIRGIN-COURT"
    assert 0.0 < p25 <= 1.0


def test_t06_underpowered_card_in_real_session(court_setup: dict, cfg: dict) -> None:
    s = court_setup
    sparse_card = dict(s["cc"].card, card_id="C-0802", event="AND(gt(planted_feat, 0.5), eq(weekday, 0))")
    cc = bc.compile_card(sparse_card, bc.BrickLibrary.load(), available={"spot_1m"}, cal=qm.load_era_calendar(),
                         extra_bricks={"planted_feat"})
    cc.evaluate(s["ctx_for"](s["pg_frame"]), np.ones(s["pg_frame"].n_days, bool))
    cand = Candidate(card_id="C-0802", card_hash=cc.card_hash(), lane="B", sign=1, target="T1", horizon=3, prior_weight=1.0,
                     deflated_lift=0.5, event_days=frozenset(s["manifest"].playground_days[:120]),
                     playground_days=len(s["manifest"].playground_days), compiled=cc, card=sparse_card)
    docket = ct.freeze_docket([cand], s["manifest"], cfg, K=1, run_id="run_t06", path=s["tmp"] / "d2.json")
    assert docket["status"] == "FROZEN"
    verdicts = ct.run_session(docket, {"C-0802": cc}, s["ctx_for"](s["court_frame"]), s["manifest"], cfg, s["run"])
    assert verdicts[0].event_days < 30
    assert verdicts[0].verdict == "COURT-UNDERPOWERED"


def test_t06_three_sessions_use_alpha_k_with_zero_overlap(cfg: dict, cal: dict) -> None:
    days = _weekdays(dt.date(2027, 1, 4), 250)
    m = ss.make_split(days, cfg, cal, data_version="a" * 64)
    used: list[set[str]] = []
    all_days = list(days)
    for k in (1, 2, 3):
        assert m.session_k + 1 == k
        assert cs.alpha_k(k) == pytest.approx(0.05 * 6 / (math.pi ** 2 * k * k))
        used.append(set(m.court_days))
        ss.graduate(m)
        all_days = _weekdays(dt.date(2027, 1, 4), len(all_days) + 70)
        block = ss.next_court_block(m, all_days, cfg, cal)
        assert block["ready"]
        ss.open_session_block(m, block)
    used.append(set(m.court_days))
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            assert used[i].isdisjoint(used[j])


# ============================================================================================
# Playground (FR-09, Ω-4, FM-36, NFR-11 via injected callback)
# ============================================================================================
@pytest.fixture
def playground_setup(tmp_path: Path, cfg: dict, cal: dict, lib: bc.BrickLibrary) -> dict:
    days = _weekdays(dt.date(2027, 1, 4), 160)
    frame, flags = _planted_frame(days, event_rate=0.5, effect_pts=30.0, per_day=60)
    m = ss.make_split(days, cfg, cal, data_version="b" * 64)
    pg_frame = frame.subset_days(np.array(m.playground_days, dtype="datetime64[D]"))
    key = dict(zip(zip(frame.dates.astype(str), frame.minute, strict=True), flags, strict=True))
    feat = np.array([key[(str(d), mm)] for d, mm in zip(pg_frame.dates, pg_frame.minute, strict=True)])
    ctx = bc.FeatureContext(frame=pg_frame, lib=lib, cal=cal, available={"spot_1m"}, extra_bricks={"planted_feat": feat})
    fallback = {"H1": [{"schema_version": "2.0", "card_id": "C-0900", "type": "RULE", "lane": "B", "provenance": "HUNT:H1",
                        "round": 1, "hypothesis": "Persona card on the planted feature", "event": "gt(planted_feat, 0.5)",
                        "side": "both", "target": "T1", "horizons": [3], "baseline": "slot_matched",
                        "cutoffs": "discovery_frozen", "data_needed": ["spot_1m"], "cost_series": "underlying",
                        "prior_weight": 1.0, "expected_sign": "positive"},
                       {"schema_version": "2.0", "card_id": "C-0901", "type": "RULE", "lane": "B", "provenance": "HUNT:H1",
                        "round": 1, "hypothesis": "Leaky card that peeks at the next close", "event": "gt(close_{t+1}, 0)",
                        "side": "both", "target": "T1", "horizons": [3], "baseline": "slot_matched",
                        "cutoffs": "discovery_frozen", "data_needed": ["spot_1m"], "cost_series": "underlying",
                        "prior_weight": 1.0, "expected_sign": "none"}]}
    return {"ctx": ctx, "manifest": m, "frame": frame, "fallback": fallback, "run_dir": tmp_path / "run"}


def test_t06_playground_with_court_days_is_seal_violation(playground_setup: dict, cfg: dict, lib: bc.BrickLibrary, cal: dict) -> None:
    s = playground_setup
    full_ctx = bc.FeatureContext(frame=s["frame"], lib=lib, cal=cal, available={"spot_1m"})
    with pytest.raises(bnf.SealViolation) as exc:
        pg.PlaygroundEngine(full_ctx, lib, cfg, cs.Ctx.from_config(cfg), s["manifest"], run_dir=s["run_dir"])
    assert exc.value.tripwire_id == "F31"


def test_t06_playground_runs_all_modes_logs_every_idea_and_ranks(playground_setup: dict, cfg: dict, lib: bc.BrickLibrary) -> None:
    s = playground_setup
    eng = pg.PlaygroundEngine(s["ctx"], lib, cfg, cs.Ctx.from_config(cfg), s["manifest"], run_dir=s["run_dir"],
                              ai_fallback=s["fallback"], haircut=0.5)
    res = eng.run()
    assert set(res.modes) == {f"H{i}" for i in range(1, 14)}
    assert res.modes["H1"].ideas == 2
    lines = (s["run_dir"] / "idea_ledger.jsonl").read_text().splitlines()
    assert len(lines) == res.m_playground
    recs = [json.loads(x) for x in lines]
    assert [r["m_lifetime"] for r in recs] == list(range(1, len(recs) + 1))
    assert {r["status"] for r in recs} <= {"EVALUATED", "DISCARDED", "REJECTED"}
    leaky = next(r for r in recs if r["card_id"] == "C-0901")
    assert leaky["status"] == "REJECTED"
    assert "LEAK-REJECTED" in leaky["reason"]
    assert all(len(r["card_hash"]) == 64 and len(r["event_days_hash"]) == 64 for r in recs)
    assert res.pool
    assert all(not hasattr(c, "verdict") for c in res.pool)
    ranked = ct.rank_pool(res.pool)
    assert ranked[0].card_id in ("C-0900", *(c.card_id for c in res.pool if "planted_feat" in c.card.get("event", "")))
    assert "H1" not in res.modes_open()
    assert res.modes["H6"].status == "OPEN"


def test_t06_saturation_counts_clusters_not_raw_ideas(playground_setup: dict, cfg: dict, lib: bc.BrickLibrary) -> None:
    s = playground_setup
    eng = pg.PlaygroundEngine(s["ctx"], lib, cfg, cs.Ctx.from_config(cfg), s["manifest"], run_dir=s["run_dir"])
    base = dict(s["fallback"]["H1"][0])
    dupes = []
    for i in range(6):
        card = dict(base, card_id=f"C-{950 + i}", hypothesis=f"Duplicate wording variant {i} of the planted card")
        dupes.append(card)
    eng.run_batch("H1", dupes)
    assert eng.modes["H1"].ideas == 6
    assert eng.modes["H1"].novel_clusters == 1
    evaluated = [e for e in eng.evaluated.values() if e.status == "EVALUATED"]
    assert len(evaluated) == 1


def test_t06_budget_checkpoint_and_resume(playground_setup: dict, cfg: dict, lib: bc.BrickLibrary) -> None:
    s = playground_setup
    cap = 40

    def checkpoint(state: dict) -> None:
        if state["used_ideas"] >= cap:
            raise bnf.PartialCheckpoint(f"idea budget {cap} reached", stage="Ω-S3", reason="max_ideas")

    eng = pg.PlaygroundEngine(s["ctx"], lib, cfg, cs.Ctx.from_config(cfg), s["manifest"], run_dir=s["run_dir"],
                              ai_fallback=s["fallback"], checkpoint_callback=checkpoint, haircut=0.5)
    with pytest.raises(bnf.PartialCheckpoint) as exc:
        eng.run()
    assert exc.value.exit_code == 2
    state = json.loads((s["run_dir"] / "playground_state.json").read_text())
    first = state["m_lifetime"]
    assert first >= cap
    with (s["run_dir"] / "idea_ledger.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "1.0", "m_lifetime": 9999, "card_')
    resumed = pg.PlaygroundEngine(s["ctx"], lib, cfg, cs.Ctx.from_config(cfg), s["manifest"], run_dir=s["run_dir"],
                                  ai_fallback=s["fallback"], haircut=0.5)
    assert resumed.ledger.m_lifetime == first
    res = resumed.run()
    ids = [json.loads(x)["card_id"] for x in (s["run_dir"] / "idea_ledger.jsonl").read_text().splitlines()]
    assert len(ids) == len(set(ids))
    assert res.m_playground == len(ids)


def test_t06_playground_module_never_reads_court_or_imports_chair() -> None:
    import ast

    src = Path(pg.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert "bn_forensics.chair" not in imported
    assert "bn_forensics.court" not in imported
    assert not any(m.startswith("pyarrow") for m in imported)
    for token in ("read_table", "StoreReader", "court_reader", "read_parquet"):
        assert token not in src
