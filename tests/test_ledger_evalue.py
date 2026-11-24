"""FILE-046 · Ledger e-values, e-BH, peeking safety, static peek-ban, post-freeze isolation (T-08).

Covers Ω-3 (peeking trap vs e-value peeking), V4-9 (25 post-freeze days), FR-13, FM-24, FM-33,
FM-44, NFR-04 (atomic state, backup, corrupt ledger exit 3, lock contention exit 3).
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

import bn_forensics as bnf
from bn_forensics import ledger as lg

SRC = Path(bnf.__file__).resolve().parent
# Declared simulation / trap modules that MUST implement p-value peeking to measure it (DEF-05).
TRAP_ALLOWLIST = {"null_lab.py": ("peeking_trap_",), "omega_sim.py": ("sim_p0_", "sim_p4c_")}


@pytest.fixture
def cfg() -> dict:
    return bnf.load_config()


def _days(n: int, start: int = 0) -> list[str]:
    origin = np.datetime64("2030-01-01")
    return [str(origin + np.timedelta64(i, "D")) for i in range(start, start + n)]


def _stream(seed: int, n_pre: int, n_fwd: int, effect: float, rate: float = 0.8) -> tuple[lg.CardSeries, lg.CardSeries]:
    rng = np.random.default_rng(seed)
    n = n_pre + n_fwd
    days = _days(n)
    b = rng.normal(0, 1, n)
    ev = rng.random(n) < rate
    x = np.where(ev, rng.normal(0, 1, n), np.nan)
    x[n_pre:] = np.where(ev[n_pre:], x[n_pre:] + effect, np.nan)
    return (lg.CardSeries(days[:n_pre], list(x[:n_pre]), list(b[:n_pre])),
            lg.CardSeries(days[n_pre:], list(x[n_pre:]), list(b[n_pre:])))


def _confirm_rate(tmp_path: Path, cfg: dict, effect: float, seeds: int, n_fwd: int, *, peek: bool = True) -> float:
    hits = 0
    for s in range(seeds):
        led = lg.Ledger(tmp_path / f"led_{effect}_{s}", cfg)
        hist, fwd = _stream(5000 + s, 200, n_fwd, effect)
        led.register("C-1", card_hash=f"{s:064x}", sign=1, history=hist, provenance="t", frozen_at="t", data_version="v")
        confirmed = False
        step = int(cfg["ledger"]["tick_days"]) if peek else n_fwd
        for start in range(0, n_fwd, step):
            part = lg.CardSeries(fwd.dates[start:start + step], fwd.x[start:start + step], fwd.b[start:start + step])
            led.feed("C-1", part)
            led.tick()
            if led.state["cards"]["C-1"]["state"] == "LEDGER-CONFIRMED":
                confirmed = True
                break
        hits += confirmed
    return hits / seeds


# ============================================================================================
# Math known answers
# ============================================================================================
def test_t08_mixture_log_e_formula_known_answer() -> None:
    s, v, grid = 3.0, 25, (0.15, 0.30, 0.60)
    manual = math.log(np.mean([(1 + v * r * r) ** -0.5 * math.exp(r * r * s * s / (2 * (1 + v * r * r))) for r in grid]))
    assert lg.mixture_log_e(s, v, grid) == pytest.approx(manual, rel=1e-12)
    assert lg.mixture_log_e(-3.0, v, grid) == pytest.approx(manual, rel=1e-12)
    assert lg.mixture_log_e(5.0, 0, grid) == 0.0
    assert math.log(20000 / 0.05) == pytest.approx(12.899, abs=1e-3)


def test_t08_step_up_ebh_hand_computed_five_cards() -> None:
    e_values = {"a": 250.0, "b": 60.0, "c": 40.0, "d": 10.0, "e": 1.0}
    log_e = {k: math.log(v) for k, v in e_values.items()}
    res = lg.e_bh(log_e, {"a": 1.0, "b": -2.0, "c": 0.5, "d": 1.0, "e": 1.0}, M=5, alpha=0.05)
    assert res["k_star"] == 3
    assert res["rejected"] == ["a", "b", "c"]
    assert res["confirmed"] == ["a", "c"]
    none = lg.e_bh({"a": math.log(99.0)}, {"a": 1.0}, M=5, alpha=0.05)
    assert none["k_star"] == 0


def test_t08_residual_is_predictable_and_baseline_centred() -> None:
    rng = np.random.default_rng(1)
    x = np.where(rng.random(40) < 0.6, rng.normal(0.5, 1, 40), np.nan)
    b = rng.normal(0, 1, 40)
    r = lg.predictable_residuals(x, b, prior_sum=3.0, prior_n=10)
    d = 20
    x2, b2 = x.copy(), b.copy()
    x2[d + 1:] = 99.0
    b2[d + 1:] = -99.0
    r2 = lg.predictable_residuals(x2, b2, prior_sum=3.0, prior_n=10)
    assert np.array_equal(r[: d + 1], r2[: d + 1], equal_nan=True)
    if np.isfinite(x[d]):
        expected = x[d] - (3.0 + np.nansum(b[:d])) / (10 + d)
        assert r[d] == pytest.approx(expected)
    own_mean_version = x - np.nanmean(x)
    assert not np.allclose(r[np.isfinite(r)], own_mean_version[np.isfinite(r)])


# ============================================================================================
# Ω-3: e-value peeking safe; p-value peeking trap fails
# ============================================================================================
def test_t08_evalue_peeking_null_fpr_at_most_5_percent(tmp_path: Path, cfg: dict) -> None:
    fpr = _confirm_rate(tmp_path, cfg, 0.0, seeds=60, n_fwd=250, peek=True)
    assert fpr <= 0.05


def _pvalue_peek_and_stop_trap(seed: int, n_fwd: int = 250, tick: int = 10, alpha: float = 0.05) -> bool:
    """Test-local p-value peek-and-stop (c = 1.00): re-test every tick, stop at the first p < alpha."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n_fwd)
    for end in range(tick, n_fwd + 1, tick):
        seen = x[:end]
        t = seen.mean() / (seen.std(ddof=1) / math.sqrt(seen.size))
        if sps.t.sf(abs(t), seen.size - 1) * 2 < alpha:
            return True
    return False


def test_t08_pvalue_peeking_trap_fails_while_evalue_does_not(tmp_path: Path, cfg: dict) -> None:
    trap = np.mean([_pvalue_peek_and_stop_trap(9000 + s) for s in range(300)])
    assert trap > 0.05
    e_fpr = _confirm_rate(tmp_path, cfg, 0.0, seeds=40, n_fwd=250, peek=True)
    assert e_fpr < trap
    assert e_fpr <= 0.05


def _peek_and_stop_functions(path: Path) -> list[str]:
    """AST detector: a loop that computes a p-value and breaks/returns on a comparison."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for loop in [n for n in ast.walk(func) if isinstance(n, (ast.For, ast.While))]:
            names = {(n.attr if isinstance(n, ast.Attribute) else getattr(n, "id", "")) for n in ast.walk(loop)
                     if isinstance(n, (ast.Name, ast.Attribute))}
            computes_p = any(k in name.lower() for name in names for k in ("pvalue", "p_value", "p_val", "ttest", "sign_flip", "sf"))
            stops = any(isinstance(n, (ast.Break, ast.Return)) for n in ast.walk(loop))
            compares = any(isinstance(n, ast.Compare) for n in ast.walk(loop))
            if computes_p and stops and compares:
                hits.append(func.name)
                break
    return hits


def test_t08_static_ban_no_pvalue_peek_and_stop_in_production() -> None:
    ledger_src = (SRC / "ledger.py").read_text(encoding="utf-8")
    for token in ("pvalue", "p_value", "p_val", "ttest", "sign_flip", "norm.sf", ".sf("):
        assert token not in ledger_src, token
    offenders = {}
    for path in sorted(SRC.glob("*.py")):
        hits = _peek_and_stop_functions(path)
        allowed = TRAP_ALLOWLIST.get(path.name, ())
        bad = [h for h in hits if not h.startswith(allowed)] if allowed else hits
        if bad:
            offenders[path.name] = bad
    assert offenders == {}
    trap_file = Path(__file__)
    assert _peek_and_stop_functions(trap_file) == ["_pvalue_peek_and_stop_trap"]


# ============================================================================================
# V4-9 post-freeze isolation, sign gate, planted power, degenerate series
# ============================================================================================
def test_t08_25_post_freeze_days_only_and_pre_freeze_ignored(tmp_path: Path, cfg: dict) -> None:
    led = lg.Ledger(tmp_path / "led", cfg)
    hist, fwd = _stream(7, 120, 25, 0.3)
    led.register("C-1", card_hash="1" * 64, sign=1, history=hist, provenance="t", frozen_at="t", data_version="v")
    replay = lg.CardSeries(hist.dates[-10:], hist.x[-10:], hist.b[-10:])
    assert led.feed("C-1", replay) == 0
    assert led.feed("C-1", fwd) == 25
    rec = led.tick()
    card = led.state["cards"]["C-1"]
    assert rec["windows_processed"] == 2
    assert len(card["pending"]) == 5
    assert card["processed_through"] == fwd.dates[19]
    assert card["v"] == int(np.isfinite(np.asarray(fwd.x[:20], float)).sum())
    assert led.feed("C-1", fwd) == 0


def test_t08_planted_effect_confirms_with_power_and_opposite_never(tmp_path: Path, cfg: dict) -> None:
    power = _confirm_rate(tmp_path / "p", cfg, 0.5, seeds=30, n_fwd=250, peek=False)
    assert power >= 0.5
    wrong = _confirm_rate(tmp_path / "w", cfg, -1.0, seeds=30, n_fwd=250, peek=True)
    assert wrong == 0.0


def test_t08_degenerate_series_zero_log_e(tmp_path: Path, cfg: dict) -> None:
    led = lg.Ledger(tmp_path / "led", cfg)
    days = _days(60)
    flat = lg.CardSeries(days[:40], [1.0] * 40, [1.0] * 40)
    led.register("C-1", card_hash="2" * 64, sign=1, history=flat, provenance="t", frozen_at="t", data_version="v")
    card = led.state["cards"]["C-1"]
    assert card["sd_freeze"] == 0.0
    assert "DEGENERATE-SERIES" in card["note"]
    led.feed("C-1", lg.CardSeries(days[40:], [5.0] * 20, [1.0] * 20))
    led.tick()
    assert card["log_e"] == 0.0
    assert card["state"] != "LEDGER-CONFIRMED"


def test_t08_registration_beyond_m_refused(tmp_path: Path, cfg: dict) -> None:
    small = {**cfg, "ledger": {**cfg["ledger"], "M": 3}}
    led = lg.Ledger(tmp_path / "led", small)
    hist, _ = _stream(3, 50, 0, 0.0)
    states = [led.register(f"C-{i}", card_hash=f"{i:064x}", sign=1, history=hist, provenance="t", frozen_at="t",
                           data_version="v") for i in range(5)]
    assert states == ["FWD-WATCH", "FWD-WATCH", "FWD-WATCH", "REFUSED-M", "REFUSED-M"]
    assert led.state["registered_lifetime"] == 3
    assert len(led.state["refused"]) == 2
    assert led.register("C-0-again", card_hash=f"{0:064x}", sign=1, history=hist, provenance="t", frozen_at="t",
                        data_version="v").startswith("ALREADY-REGISTERED")


def test_t08_decay_rule_two_windows_against_sign(tmp_path: Path, cfg: dict) -> None:
    led = lg.Ledger(tmp_path / "led", cfg)
    hist, fwd = _stream(21, 200, 40, -0.25, rate=1.0)
    led.register("C-1", card_hash="3" * 64, sign=1, history=hist, provenance="t", frozen_at="t", data_version="v")
    led.feed("C-1", fwd)
    led.tick()
    card = led.state["cards"]["C-1"]
    wins = card["windows"]
    against = [w for w in wins if w["mean"] < 0 and abs(w["mean"]) < 1.96 * w["se"]]
    if len(wins) >= 2 and all(w in against for w in wins[-2:]):
        assert card["state"] == "FWD-DECAYED"
    else:
        assert card["state"] in ("FWD-WATCH", "FWD-DECAYED")
    forced = {"windows": [{"n": 10, "mean": -0.1, "se": 0.3}, {"n": 10, "mean": -0.2, "se": 0.3}], "sign": 1}
    assert led._decayed(forced)
    assert not led._decayed({"windows": [{"n": 10, "mean": 0.1, "se": 0.3}, {"n": 10, "mean": -0.2, "se": 0.3}], "sign": 1})


# ============================================================================================
# NFR-04 state handling, lock contention, union FPR rule, cache
# ============================================================================================
def test_t08_atomic_state_backup_cards_csv_and_corrupt_exit_3(tmp_path: Path, cfg: dict) -> None:
    led = lg.Ledger(tmp_path / "led", cfg)
    hist, fwd = _stream(5, 100, 20, 0.2)
    led.register("C-1", card_hash="4" * 64, sign=1, history=hist, provenance="PERSONA:x", frozen_at="2026-11-22T16:00:00+05:30",
                 data_version="v")
    led.save()
    assert led.backup() is not None
    led.feed("C-1", fwd)
    led.tick()
    led.save()
    assert (tmp_path / "led" / "ledger.json.bak").exists()
    header = (tmp_path / "led" / "cards.csv").read_text().splitlines()[0].split(",")
    assert header == list(lg.CARDS_CSV_FIELDS)
    assert len(header) == 11
    reloaded = lg.Ledger(tmp_path / "led", cfg)
    assert reloaded.state == json.loads((tmp_path / "led" / "ledger.json").read_text())
    (tmp_path / "led" / "ledger.json").write_text("{not json")
    with pytest.raises(bnf.CorruptStateError) as exc:
        lg.Ledger(tmp_path / "led", cfg)
    assert exc.value.exit_code == 3
    (tmp_path / "led" / "ledger.json").write_text(json.dumps({"schema_version": "1.0", "cards": {"x": {"sign": 1}}}))
    with pytest.raises(bnf.CorruptStateError):
        lg.Ledger(tmp_path / "led", cfg)


def test_t08_concurrent_ledger_tick_fails_with_exit_3(tmp_path: Path, cfg: dict) -> None:
    first = lg.Ledger(tmp_path / "led", cfg)
    second = lg.Ledger(tmp_path / "led", cfg)
    with first.lock():
        with pytest.raises(bnf.LockContention) as exc:
            second.lock().acquire()
        assert exc.value.exit_code == 3
    with second.lock() as lock:
        assert lock.mode in ("flock", "o_excl")


def test_t08_union_fpr_rule_halves_alpha(tmp_path: Path, cfg: dict) -> None:
    led = lg.Ledger(tmp_path / "led", cfg)
    assert not led.apply_union_fpr(0.04)
    assert led.state["alpha"] == 0.05
    assert led.apply_union_fpr(0.061)
    assert led.state["alpha"] == 0.025
    assert "halved" in led.state["alpha_note"]


def test_t08_feature_cache_roundtrip(tmp_path: Path) -> None:
    series = lg.series_from_days(["d2", "d1", "d3"], {"d1": 0.5, "d3": -1.0}, {"d1": 0.1, "d2": 0.2, "d3": 0.3})
    assert series.dates == ["d1", "d2", "d3"]
    assert math.isnan(series.x[1])
    path = lg.store_cached_series(tmp_path / "cache", "dv1", "a" * 64, series)
    assert path.parent == tmp_path / "cache" / "dv1"
    back = lg.load_cached_series(tmp_path / "cache", "dv1", "a" * 64)
    assert back is not None
    assert back.dates == series.dates
    assert lg.load_cached_series(tmp_path / "cache", "dv2", "a" * 64) is None
