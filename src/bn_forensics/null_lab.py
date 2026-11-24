"""Pre-Playground Omega Null Lab (FILE-035, FR-05; Ω-1, Ω-2, Ω-3, Ω-9; §10.4 #12; Ω-S2).

Runs the Omega protocol on synthetic worlds shaped like the real data (``n`` Playground and Court
days) BEFORE any exploration:

* **Null-A** (Court FPR): ``K_search = 200`` searcher cards (parents + mutant clones) ranked by an
  Empirical-Bayes shrunk lift on Playground days, a greedy-diverse ``K = 20`` docket
  (``court.greedy_diverse``) and the single-shot sign-flip Court rule ``p < alpha_1/K`` with
  ``event_days >= 30`` and ``B_court`` draws (decision-exact early termination, same data).
* **Null-B**: the same protocol with day outcomes resampled from the real day-level distribution.
* **Ledger FPR** with peeking at every 10-day tick (baseline-centred mixture e-values + e-BH) and the
  **union** (Court U Ledger) false-claim rate (M-2); union > 5% halves the Ledger alpha (§10.4 #12).
* **Planted-edge recall** at ``delta in {0.1, 0.2, 0.3, 0.5, 0.8}`` vs the analytic power curve
  (``CALIBRATION WARNING`` beyond 25%, M-6) and the Null Lab haircut ratio for shrinkage.
* **Leak Canary** (static + dynamic +1-bar gate), **Peeking Trap** (p-value peek-and-stop must FAIL,
  Ω-3), **Uncapped-V5 regression** (must exceed 5%, Ω-9), **4 metamorphic tests** (price x k,
  timestamp shift, mirror, day shuffle) and **twin parity** on 200 random events (F5).

Any failure makes the pipeline ``MODE: INVALID PIPELINE`` (exit 4). The trap protocols live only in
functions prefixed ``peeking_trap_`` (DEF-05 allow-list); their output never feeds a verdict.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from bn_forensics import EventLog, InvalidPipeline
from bn_forensics.bricks_compiler import (
    BrickDef,
    BrickLibrary,
    FeatureContext,
    compile_card,
    dynamic_leak_gate,
)
from bn_forensics.core_stats import (
    BarFrame,
    Ctx,
    alpha_k,
    b_court,
    court_mde_sd,
    event_study,
    horizon_cap,
    n_new,
    write_csv,
    z_upper,
)
from bn_forensics.court import greedy_diverse
from bn_forensics.ledger import e_bh, mixture_log_e, predictable_residuals
from bn_forensics.shrinkage_basis import Candidate
from bn_forensics.twin_impl import Bar, EventSpec, compare_events, twin_event_study

SUMMARY_FIELDS = ("check", "value", "threshold", "status", "detail")
CANARY_EVENTS = ("gt(close_{t+1}, 0)", "gt(lead(ret_1), 0)", "gt(day_high, 0)", "lt(day_close, 0)",
                 "gt(future_ret, 0)", "gt(next_close, 0)", "gt(eod_range, 0)")


@dataclass
class NullLabResult:
    verdict: str
    fpr_court: float
    fpr_ledger: float
    fpr_union: float
    fpr_null_b: float | None
    recall_at_delta: dict[str, float]
    analytic_power: dict[str, float]
    calibration_warning: bool
    haircut: float
    canary_caught: float
    canary_effect_if_uncaught: float | None
    meta_pass: int
    twin_pass: bool
    trap_fpr: float
    uncapped_fpr: float
    ledger_alpha: float
    ledger_alpha_halved: bool
    digital_twin: dict[str, float]
    rows: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def status_line(self) -> str:
        return (f"[Ω-S2] NULL_LAB {self.verdict}: court_fpr={self.fpr_court:.3f}, ledger_fpr={self.fpr_ledger:.3f}, "
                f"canary={self.canary_caught:.0%}, meta={self.meta_pass}/4, twin={'PASS' if self.twin_pass else 'FAIL'}")


# =============================================================================================
# Synthetic worlds at real n
# =============================================================================================
@dataclass
class NullWorld:
    X: np.ndarray
    E: np.ndarray
    B: np.ndarray
    pg: np.ndarray
    court: np.ndarray


def make_world(rng: np.random.Generator, n_pg: int, n_court: int, *, k_search: int, parents: int, overlap: float,
               delta: float = 0.0, real_days: np.ndarray | None = None) -> NullWorld:
    """Searcher cards over ``n_pg + n_court`` days; card 0 carries ``delta`` sd on its event days."""
    n = n_pg + n_court
    per = max(1, k_search // parents)
    E = np.zeros((parents * per, n), dtype=bool)
    for p in range(parents):
        rate = rng.uniform(0.2, 0.45)
        base = rng.random(n) < rate
        E[p * per] = base
        for m in range(1, per):
            E[p * per + m] = (base & (rng.random(n) < overlap)) | (~base & (rng.random(n) < rate * (1 - overlap)))
    E = E[:k_search]
    if real_days is not None and real_days.size:
        pool = real_days[np.isfinite(real_days)]
        pool = (pool - pool.mean()) / (pool.std() or 1.0)
        X = rng.choice(pool, size=(k_search, n), replace=True)
    else:
        u = rng.normal(0, 1, n)
        a = np.where(rng.random(k_search) < 0.5, 1.0, -1.0)
        X = a[:, None] * 0.5 * u[None, :] + math.sqrt(0.75) * rng.normal(0, 1, (k_search, n))
    X[0] = X[0] + delta * E[0]
    base = rng.normal(0, 0.1, (k_search, n))
    pg = np.zeros(n, dtype=bool)
    pg[:n_pg] = True
    return NullWorld(X=X, E=E, B=base, pg=pg, court=~pg)


def _lift_se(X: np.ndarray, E: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = E & days[None, :]
    n = m.sum(axis=1)
    s = (X * m).sum(axis=1)
    sq = ((X ** 2) * m).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / n
        var = (sq - n * mean * mean) / np.maximum(n - 1, 1)
        se = np.sqrt(var / n)
    return np.nan_to_num(mean), np.nan_to_num(se, nan=np.inf), n


def court_confirms(x: np.ndarray, sign: int, threshold: float, B: int, seed: int, chunk: int = 2000) -> bool:
    """Decision-exact sign-flip rule ``(1 + #{T* >= T}) / (B + 1) < threshold`` on fixed data.

    Draws stop as soon as the exceedance count makes the decision certain; the decision equals the
    full-``B`` decision because the count can only grow.
    """
    x = np.asarray(x, float)
    t_obs = sign * x.mean()
    limit = math.ceil(threshold * (B + 1)) - 1
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    while done < B:
        m = min(chunk, B - done)
        flips = rng.integers(0, 2, size=(m, x.size), dtype=np.int8).astype(float) * 2 - 1
        exceed += int(np.count_nonzero(sign * (flips * x).mean(axis=1) >= t_obs - 1e-12))
        done += m
        if exceed > limit:
            return False
    return (1 + exceed) / (B + 1) < threshold


def _docket(world: NullWorld, K: int, jmax: float) -> list[tuple[int, int]]:
    lift, se, n = _lift_se(world.X, world.E, world.pg)
    tau2 = max(0.0, float(np.mean(lift ** 2 - np.where(np.isfinite(se), se, 0) ** 2)))
    deflated = lift * tau2 / (tau2 + np.where(np.isfinite(se), se, np.inf) ** 2)
    pg_idx = np.flatnonzero(world.pg)
    cands = []
    for i in range(world.X.shape[0]):
        if n[i] < 30:
            continue
        days = frozenset(int(d) for d in pg_idx[world.E[i, pg_idx]])
        cands.append(Candidate(card_id=f"N{i:04d}", card_hash=f"{i:064d}", lane="B", sign=1 if lift[i] >= 0 else -1,
                               target="T1", horizon=1, prior_weight=1.0, deflated_lift=float(deflated[i]),
                               event_days=days, playground_days=int(world.pg.sum())))
    chosen = greedy_diverse(cands, K, jaccard_max=jmax)
    return [(int(c.card_id[1:]), c.sign) for c, _ in chosen]


def _court_any(world: NullWorld, docket: Sequence[tuple[int, int]], session_k: int, K: int, B: int, min_days: int,
               seed: int) -> tuple[bool, set[int]]:
    thr = alpha_k(session_k) / K
    hits: set[int] = set()
    for i, sign in docket:
        x = world.X[i][world.E[i] & world.court]
        if x.size < min_days:
            continue
        if court_confirms(x, sign, thr, B, seed + session_k):
            hits.add(i)
    return bool(hits), hits


def _ledger_any(world: NullWorld, cards: Sequence[tuple[int, int]], n_fwd: int, rng: np.random.Generator, *, M: int,
                alpha: float, c: float, rho: Sequence[float], tick: int) -> bool:
    """Peek at every tick with baseline-centred mixture e-values + e-BH over the registered cards."""
    if not cards:
        return False
    hist = {}
    for i, sign in cards:
        x = np.where(world.E[i], world.X[i], np.nan)
        r = predictable_residuals(x, world.B[i])
        fin = r[~np.isnan(r)]
        sd = float(np.std(fin, ddof=1)) if fin.size > 2 else 0.0
        hist[i] = (sign, sd, float(np.nansum(world.B[i])), int((~np.isnan(world.B[i])).sum()))
    fwd_x = rng.normal(0, 1, (len(cards), n_fwd))
    fwd_e = rng.random((len(cards), n_fwd)) < 0.4
    fwd_b = rng.normal(0, 0.1, (len(cards), n_fwd))
    # residuals are predictable (r_d uses b_{<d} only), so each tick's prefix equals the full-series prefix
    fwd_r = np.vstack([predictable_residuals(np.where(fwd_e[row], fwd_x[row], np.nan), fwd_b[row], prior_sum=hist[i][2],
                                             prior_n=hist[i][3]) for row, (i, _sign) in enumerate(cards)])
    for end in range(tick, n_fwd + 1, tick):
        log_e, s_sign = {}, {}
        for row, (i, _sign) in enumerate(cards):
            sign, sd, _bsum, _bn = hist[i]
            seg = fwd_r[row, :end]
            fin = seg[~np.isnan(seg)]
            if sd <= 1e-12 or fin.size == 0:
                log_e[str(i)], s_sign[str(i)] = 0.0, 0.0
                continue
            s = sign * fin.sum() / (c * sd)
            log_e[str(i)], s_sign[str(i)] = mixture_log_e(s, int(fin.size), rho), s
        if e_bh(log_e, s_sign, M, alpha)["confirmed"]:
            return True
    return False


def peeking_trap_pvalue(rng: np.random.Generator, n_fwd: int, tick: int, alpha: float) -> bool:
    """Ω-3 trap: one null card, p-value peek-and-stop at every tick (c = 1.00). MUST inflate FPR."""
    x = rng.normal(0, 1, n_fwd)
    for end in range(tick, n_fwd + 1, tick):
        seen = x[:end]
        t = seen.mean() / (seen.std(ddof=1) / math.sqrt(seen.size))
        if 2 * sps.t.sf(abs(t), seen.size - 1) < alpha:
            return True
    return False


def peeking_trap_uncapped_v5(rng: np.random.Generator, n_pg: int, n_court: int, rounds: int, cfg: dict[str, Any]) -> bool:
    """Ω-9 regression: 24 rounds of fresh searches re-testing the SAME validation days (V5 uncapped)."""
    nl = cfg["null_lab"]
    court_x = rng.normal(0, 1, (int(nl["K_search"]), n_court))
    for _ in range(rounds):
        world = make_world(rng, n_pg, n_court, k_search=int(nl["K_search"]), parents=int(nl["parents"]),
                           overlap=float(nl["mutant_overlap"]))
        world.X[:, ~world.pg] = court_x
        lift, se, _ = _lift_se(world.X, world.E, world.pg)
        top = np.argsort(-np.abs(lift / se), kind="mergesort")[:5]
        for i in top:
            x = world.X[i][world.E[i] & world.court]
            if x.size < 10:
                continue
            t = np.sign(lift[i]) * x.mean() / (x.std(ddof=1) / math.sqrt(x.size))
            if sps.t.sf(t, x.size - 1) < 0.05 / top.size:
                return True
    return False


# =============================================================================================
# Metamorphic tests and twin parity on the real frame
# =============================================================================================
def _clone_frame(fr: BarFrame, **over: Any) -> BarFrame:
    args = {k: getattr(fr, k) for k in ("dates", "day_idx", "slot", "minute", "o", "h", "l", "c", "v", "oi", "masked",
                                         "days", "day_start", "day_end", "interval_min", "bars_per_day")}
    args.update(over)
    return BarFrame(**args, meta=dict(fr.meta))


def metamorphic_tests(fr: BarFrame, run_ctx: Ctx, seed: int = 42) -> list[dict[str, Any]]:
    """4 metamorphic relations (V5-12): price x k, timestamp shift, mirror C-price, day shuffle."""
    rng = np.random.default_rng(seed)
    h = max(1, min(5, horizon_cap(fr.bars_per_day)))
    mask = (rng.random(fr.n) < 0.05) & (fr.slot >= 20)
    direction = np.where(rng.random(fr.n) < 0.5, 1.0, -1.0)

    def lift(frame: BarFrame, m: np.ndarray = mask) -> tuple[float, float, int]:
        res = event_study(frame, m, direction=direction, side="continuation", target="T1", horizon=h, bootstrap=False, ctx=run_ctx)
        return res.lift_atr, res.lift_points, res.n_events

    base_atr, base_pts, base_n = lift(fr)
    out = []
    k = 3.0
    scaled = _clone_frame(fr, o=fr.o * k, h=fr.h * k, l=fr.l * k, c=fr.c * k)
    s_atr, s_pts, s_n = lift(scaled)
    out.append({"test": "price_x_k", "pass": abs(s_atr - base_atr) <= 1e-9 and abs(s_pts - k * base_pts) <= 1e-6 * max(1, abs(k * base_pts)) and s_n == base_n,
                "detail": f"lift_atr {base_atr:.6g} -> {s_atr:.6g}; points x{k}"})
    shifted = _clone_frame(fr, dates=fr.dates + np.timedelta64(7, "D"), days=fr.days + np.timedelta64(7, "D"))
    t_atr, _, t_n = lift(shifted)
    out.append({"test": "timestamp_shift_7d", "pass": abs(t_atr - base_atr) <= 1e-12 and t_n == base_n,
                "detail": f"lift_atr {base_atr:.6g} -> {t_atr:.6g}"})
    centre = float(np.nanmean(fr.c))
    mirrored = _clone_frame(fr, o=2 * centre - fr.o, h=2 * centre - fr.l, l=2 * centre - fr.h, c=2 * centre - fr.c)
    m_atr, _, m_n = lift(mirrored)
    out.append({"test": "mirror_c_price", "pass": abs(m_atr + base_atr) <= 1e-9 and m_n == base_n,
                "detail": f"lift_atr {base_atr:.6g} -> {m_atr:.6g} (expected negation)"})
    perm = rng.permutation(fr.n_days)
    idx = np.concatenate([np.arange(fr.day_start[d], fr.day_end[d]) for d in perm])
    new_day_idx = np.repeat(np.arange(fr.n_days), (fr.day_end - fr.day_start)[perm])
    starts = np.r_[0, np.cumsum((fr.day_end - fr.day_start)[perm])[:-1]]
    shuffled = BarFrame(dates=np.repeat(fr.days[np.arange(fr.n_days)], (fr.day_end - fr.day_start)[perm]),
                        day_idx=new_day_idx, slot=fr.slot[idx], minute=fr.minute[idx], o=fr.o[idx], h=fr.h[idx], l=fr.l[idx],
                        c=fr.c[idx], v=fr.v[idx], oi=fr.oi[idx], masked=fr.masked[idx], days=fr.days,
                        day_start=starts, day_end=starts + (fr.day_end - fr.day_start)[perm], interval_min=fr.interval_min,
                        bars_per_day=fr.bars_per_day, meta=dict(fr.meta))
    res_sh = event_study(shuffled, mask[idx], direction=direction[idx], side="continuation", target="T1", horizon=h,
                         bootstrap=False, ctx=run_ctx)
    out.append({"test": "day_shuffle", "pass": abs(res_sh.lift_atr - base_atr) <= 1e-9 and res_sh.n_events == base_n,
                "detail": f"lift_atr {base_atr:.6g} -> {res_sh.lift_atr:.6g}"})
    return out


def frame_to_twin_bars(fr: BarFrame) -> list[Bar]:
    return [Bar(str(fr.dates[i]), int(fr.minute[i]), float(fr.o[i]), float(fr.h[i]), float(fr.l[i]), float(fr.c[i]),
                bool(fr.masked[i])) for i in range(fr.n)]


def twin_parity(fr: BarFrame, run_ctx: Ctx, n_events: int = 200, seed: int = 42, tol_abs: float = 1e-6,
                tol_rel: float = 1e-3) -> dict[str, Any]:
    """F5: 200 random events across targets/horizons; counts exact, floats within tolerance."""
    rng = np.random.default_rng(seed)
    bars = frame_to_twin_bars(fr)
    cap = max(1, horizon_cap(fr.bars_per_day))
    candidates = np.flatnonzero(~fr.masked)
    chosen = np.sort(rng.choice(candidates, size=min(n_events, candidates.size), replace=False))
    # 10 random specs (target, horizon, side) x 20 random events each
    order = rng.permutation(chosen)
    groups: list[tuple[str, int, str, list[int]]] = []
    for chunk in np.array_split(order, max(1, min(10, order.size))):
        target = str(rng.choice(["T1", "T2", "T3", "T4", "T8"]))
        h = 1 if target == "T8" else int(rng.integers(1, cap + 1))
        side = str(rng.choice(["continuation", "reversal"]))
        groups.append((target, h, side, sorted(int(i) for i in chunk)))
    direction = np.where(rng.random(fr.n) < 0.5, 1.0, -1.0)
    mismatches: list[dict[str, Any]] = []
    compared = 0
    for target, h, side, idx in groups:
        mask = np.zeros(fr.n, dtype=bool)
        mask[idx] = True
        res = event_study(fr, mask, direction=direction, side=side, target=target, horizon=h, bootstrap=False, ctx=run_ctx)
        engine_rows = [{"index": int(i), "signed": float(s), "baseline": float(b), "residual": float(s - b), "atr20": float(fr.atr20[i])}
                       for i, s, b in zip(res.event_index, res.signed_outcome, res.baseline_values, strict=True)]
        tw = twin_event_study(bars, [EventSpec(i, float(direction[i])) for i in idx], target=target, h=h, side=side,
                              event_mask_all=idx)
        cmp = compare_events(tw["rows"], engine_rows, tol_abs=tol_abs, tol_rel=tol_rel)
        compared += cmp["compared"]
        mismatches += [{**m, "target": target, "h": h} for m in cmp["mismatches"]]
    return {"events_sampled": int(chosen.size), "events_compared": compared, "mismatches": mismatches, "pass": not mismatches}


def leak_canary(ctx: FeatureContext | None, lib: BrickLibrary, cal: dict[str, Any]) -> dict[str, Any]:
    """Static canary cards + the dynamic ``close_{t+1}`` brick must all be rejected (100%)."""
    caught, total = 0, 0
    template = {"schema_version": "2.0", "card_id": "CANARY", "type": "RULE", "lane": "B", "provenance": "NULL_LAB:canary",
                "round": 1, "hypothesis": "Leak canary card", "side": "both", "target": "T1", "horizons": [1],
                "baseline": "slot_matched", "cutoffs": "discovery_frozen", "data_needed": ["spot_1m"],
                "cost_series": "underlying", "prior_weight": 1.0, "expected_sign": "none"}
    for k, event in enumerate(CANARY_EVENTS):
        card = dict(template, card_id=f"CANARY-{k}", event=event)
        total += 1
        caught += compile_card(card, lib, available={"spot_1m"}, cal=cal).status == "LEAK-REJECTED"
    eod = dict(template, card_id="CANARY-EOD", event="gt(ret_1, 0)", data_needed=["spot_1m", "daily_metrics_DERIVED:EOD_INCLUSIVE"])
    total += 1
    caught += compile_card(eod, lib, available={"spot_1m", "daily_metrics_DERIVED:EOD_INCLUSIVE"}, cal=cal).status == "LEAK-REJECTED"
    effect = None
    if ctx is not None:
        lib2 = copy.copy(lib)
        lib2.bricks = {**lib.bricks, "canary_close_t1": BrickDef("canary_close_t1", "canary", "canary_lead_close", {},
                                                                  "close_{t+1}", "bar close", ("spot_1m",), "numeric")}
        cctx = FeatureContext(frame=ctx.frame, lib=lib2, cal=cal, available=ctx.available)
        total += 1
        ok, _ = dynamic_leak_gate("canary_close_t1", cctx)
        caught += not ok
        leaked = cctx.brick("canary_close_t1") > ctx.frame.c
        res = event_study(ctx.frame, leaked, direction=np.ones(ctx.frame.n), side="continuation", target="T1", horizon=1,
                          bootstrap=False)
        effect = float(res.lift_atr)
    return {"caught": caught, "total": total, "rate": caught / total if total else 0.0, "effect_if_uncaught": effect}


# =============================================================================================
# Orchestration
# =============================================================================================
def run_null_lab(*, n_playground_days: int, n_court_days: int, cfg: dict[str, Any], run_ctx: Ctx, lib: BrickLibrary,
                 cal: dict[str, Any], ctx: FeatureContext | None = None, real_day_values: np.ndarray | None = None,
                 worlds: int | None = None, recall_worlds: int | None = None, tables_dir: str | Path | None = None,
                 log: EventLog | None = None, raise_on_fail: bool = True) -> NullLabResult:
    """Ω-S2 audit at real ``n``; raises :class:`InvalidPipeline` (exit 4) on any failed gate."""
    nl, court_cfg, led = cfg["null_lab"], cfg["court"], cfg["ledger"]
    W = int(worlds if worlds is not None else nl["worlds"])
    RW = int(recall_worlds if recall_worlds is not None else nl["recall_worlds"])
    K = int(court_cfg["K_default"])
    jmax = float(court_cfg["jaccard_max"])
    min_days = int(court_cfg["event_days_min"])
    B = b_court(alpha_k(1), K, int(court_cfg["B_court_min"]), int(court_cfg["B_court_factor"]))
    seed0 = run_ctx.seed
    mk = {"k_search": int(nl["K_search"]), "parents": int(nl["parents"]), "overlap": float(nl["mutant_overlap"])}
    ledger_alpha = float(led["alpha"])
    per_world_court: list[bool] = []
    per_world_ledger_cards: list[tuple[int, NullWorld, list[tuple[int, int]]]] = []
    for w in range(W):
        rng = np.random.default_rng([seed0, 1, w])
        world = make_world(rng, n_playground_days, n_court_days, **mk)
        docket = _docket(world, K, jmax)
        c_any, _ = _court_any(world, docket, 1, K, B, min_days, seed0)
        per_world_court.append(c_any)
        per_world_ledger_cards.append((w, world, docket))

    def ledger_pass(alpha_used: float) -> list[bool]:
        # identical forward data for both alpha levels (common random numbers per world)
        out = []
        for w_id, world, docket in per_world_ledger_cards:
            out.append(_ledger_any(world, docket, int(nl["ledger_forward_days"]), np.random.default_rng([seed0, 2, w_id]),
                                   M=int(led["M"]), alpha=alpha_used, c=float(led["c"]), rho=led["rho_grid"], tick=int(led["tick_days"])))
        return out
    ledger_flags = ledger_pass(ledger_alpha)
    court_hits = sum(per_world_court)
    ledger_hits = sum(ledger_flags)
    union_hits = sum(a or b for a, b in zip(per_world_court, ledger_flags, strict=True))
    halved = False
    if union_hits / W > float(nl["fpr_max"]):
        halved = True
        ledger_alpha = float(led["alpha_halved"])
        ledger_flags = ledger_pass(ledger_alpha)
        ledger_hits = sum(ledger_flags)
        union_hits = sum(a or b for a, b in zip(per_world_court, ledger_flags, strict=True))
    fpr_court, fpr_ledger, fpr_union = court_hits / W, ledger_hits / W, union_hits / W
    fpr_b = None
    if real_day_values is not None and np.isfinite(real_day_values).sum() >= 20:
        b_hits = 0
        for w in range(W):
            rng = np.random.default_rng([seed0, 3, w])
            world = make_world(rng, n_playground_days, n_court_days, real_days=real_day_values, **mk)
            b_hits += _court_any(world, _docket(world, K, jmax), 1, K, B, min_days, seed0)[0]
        fpr_b = b_hits / W
    recall: dict[str, float] = {}
    analytic: dict[str, float] = {}
    ratios: list[float] = []
    z_thr = z_upper(alpha_k(1) / K)
    for delta in nl["deltas"]:
        hits = 0
        for w in range(RW):
            rng = np.random.default_rng([seed0, 4, w])
            world = make_world(rng, n_playground_days, n_court_days, delta=float(delta), **mk)
            docket = _docket(world, K, jmax)
            _, confirmed = _court_any(world, docket, 1, K, B, min_days, seed0)
            if 0 in confirmed and dict(docket).get(0, 0) > 0:
                hits += 1
            pg_lift = world.X[0][world.E[0] & world.pg].mean()
            court_lift = world.X[0][world.E[0] & world.court].mean()
            if any(i == 0 for i, _ in docket) and pg_lift > 0:
                ratios.append(court_lift / pg_lift)
        recall[f"{delta}"] = hits / RW
        n_ev = 0.325 * n_court_days
        analytic[f"{delta}"] = float(sps.norm.cdf(float(delta) * math.sqrt(n_ev) - z_thr))
    keys = [f"{d}" for d in nl["deltas"]]
    divergence = max(abs(recall[k] - analytic[k]) for k in keys)
    calib_warn = divergence > float(nl["calibration_divergence"])
    haircut = float(np.clip(np.median(ratios), 0.0, 1.0)) if ratios else 0.5
    trap_rng = np.random.default_rng([seed0, 5])
    trap = float(np.mean([peeking_trap_pvalue(trap_rng, int(nl["ledger_forward_days"]), int(led["tick_days"]), float(nl["peeking_alpha"]))
                          for _ in range(W)]))
    unc_rng = np.random.default_rng([seed0, 6])
    n_unc = max(60, W // 3)
    uncapped = float(np.mean([peeking_trap_uncapped_v5(unc_rng, n_playground_days, n_court_days, int(nl["uncapped_rounds"]), cfg)
                              for _ in range(n_unc)]))
    canary = leak_canary(ctx, lib, cal)
    meta = metamorphic_tests(ctx.frame, run_ctx) if ctx is not None else []
    meta_pass = sum(1 for m in meta if m["pass"])
    twin = twin_parity(ctx.frame, run_ctx, int(nl["twin_events"]), seed0, float(nl["twin_tol_abs"]), float(nl["twin_tol_rel"])) if ctx is not None else {"pass": False, "mismatches": [{"field": "no frame"}]}
    n_events_court = 0.325 * n_court_days
    mde_now = court_mde_sd(alpha_k(1), K, n_events_court)
    target_sd = float(cfg["targets"]["mde_target_atr20"])
    digital = {"n_court_days": float(n_court_days), "court_mde_sd": mde_now,
               "n_new_court_days_for_target": float(n_new(n_court_days, mde_now, target_sd)) if target_sd > 0 else float("nan")}
    fmax = float(nl["fpr_max"])
    rows = [
        {"check": "Null-A Court FPR (K=20 diverse, alpha_1/K)", "value": fpr_court, "threshold": f"<= {fmax}", "status": "PASS" if fpr_court <= fmax else "FAIL", "detail": f"{court_hits}/{W} worlds"},
        {"check": "Ledger e-value peeking FPR", "value": fpr_ledger, "threshold": f"<= {fmax}", "status": "PASS" if fpr_ledger <= fmax else "FAIL", "detail": f"{ledger_hits}/{W} worlds, alpha {ledger_alpha}"},
        {"check": "Union (Court U Ledger) FPR (M-2)", "value": fpr_union, "threshold": f"<= {fmax}", "status": "PASS" if fpr_union <= fmax else "FAIL",
         "detail": "Ledger alpha halved to 0.025 (§10.4 #12)" if halved else "Ledger alpha 0.05"},
        {"check": "Null-B day-shuffle Court FPR", "value": fpr_b if fpr_b is not None else "N/A — DATA MISSING", "threshold": f"<= {fmax}",
         "status": ("PASS" if fpr_b <= fmax else "FAIL") if fpr_b is not None else "N/A", "detail": "real day-level outcomes resampled"},
        {"check": "Planted-edge recall vs analytic power (M-6)", "value": divergence, "threshold": "<= 0.25",
         "status": "CALIBRATION WARNING" if calib_warn else "PASS", "detail": "; ".join(f"d={k}: {recall[k]:.3f} vs {analytic[k]:.3f}" for k in keys)},
        {"check": "Null Lab haircut ratio", "value": haircut, "threshold": "[0,1]", "status": "PASS", "detail": f"{len(ratios)} planted dockets"},
        {"check": "Leak Canary rejection", "value": canary["rate"], "threshold": "== 1.0", "status": "PASS" if canary["rate"] == 1.0 else "FAIL",
         "detail": f"{canary['caught']}/{canary['total']}; lift if uncaught {canary['effect_if_uncaught']}"},
        {"check": "Peeking Trap (p-value peek-and-stop) FPR", "value": trap, "threshold": f"> {fmax} (trap must FAIL)", "status": "PASS" if trap > fmax else "FAIL", "detail": "Ω-3"},
        {"check": "Uncapped-V5 regression FPR", "value": uncapped, "threshold": f"> {fmax}", "status": "PASS" if uncapped > fmax else "FAIL", "detail": f"Ω-9, {n_unc} worlds"},
        {"check": "Metamorphic tests", "value": meta_pass, "threshold": "4/4", "status": "PASS" if meta_pass == 4 else "FAIL",
         "detail": "; ".join(f"{m['test']}={'PASS' if m['pass'] else 'FAIL'}" for m in meta)},
        {"check": "twin_impl parity (200 events)", "value": len(twin.get("mismatches", [])), "threshold": "0 mismatches",
         "status": "PASS" if twin["pass"] else "FAIL", "detail": f"{twin.get('events_compared', 0)} events compared"},
        {"check": "Digital Twin Court days for MDE_target", "value": digital["n_new_court_days_for_target"], "threshold": "planning only",
         "status": "INFO", "detail": f"current Court MDE {mde_now:.3f} sd"},
    ]
    failures = [r["check"] for r in rows if r["status"] == "FAIL"]
    verdict = "PASS" if not failures else "INVALID"
    result = NullLabResult(verdict=verdict, fpr_court=fpr_court, fpr_ledger=fpr_ledger, fpr_union=fpr_union, fpr_null_b=fpr_b,
                           recall_at_delta=recall, analytic_power=analytic, calibration_warning=calib_warn, haircut=haircut,
                           canary_caught=canary["rate"], canary_effect_if_uncaught=canary["effect_if_uncaught"],
                           meta_pass=meta_pass, twin_pass=bool(twin["pass"]), trap_fpr=trap, uncapped_fpr=uncapped,
                           ledger_alpha=ledger_alpha, ledger_alpha_halved=halved, digital_twin=digital, rows=rows, failures=failures)
    if tables_dir is not None:
        write_csv(Path(tables_dir) / "null_lab_summary.csv", SUMMARY_FIELDS, rows)
    if log is not None:
        log.emit("null_lab_completed", {"court_fpr": fpr_court, "ledger_fpr": fpr_ledger, "canary_caught": canary["rate"],
                                        "meta_pass": meta_pass, "twin_pass": bool(twin["pass"]), "recall_map": recall},
                 stage="Ω-S2", role="AG-03")
    if raise_on_fail and failures:
        tw = "F8" if "Leak Canary rejection" in failures else ("F6-NULL" if any("FPR" in f for f in failures) else "F5")
        raise InvalidPipeline(f"MODE: INVALID PIPELINE - Null Lab failed: {failures}", tripwire_id=tw)
    return result


__all__ = [
    "CANARY_EVENTS", "SUMMARY_FIELDS", "NullLabResult", "NullWorld", "court_confirms", "frame_to_twin_bars", "leak_canary",
    "make_world", "metamorphic_tests", "peeking_trap_pvalue", "peeking_trap_uncapped_v5", "run_null_lab", "twin_parity",
]
