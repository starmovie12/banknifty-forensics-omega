"""Omega simulation harness (FILE-041, FR-22, Ω-10; S5 App A protocols, §10.4 #6).

Self-contained: numpy/scipy only (no package imports). Reproducible with seeds ``0..n-1``.

World model (day level): ``D`` trading days (``NS_D``, default 450), discovery = first 60%,
validation/Court = last 40%. A search batch of ``NS_BATCH`` (default 300) ideas = parents plus
3 mutant clones each (~85% event-day overlap). Day outcomes share a market factor
(``x_{i,d} = a_i * 0.5 * u_d + sqrt(0.75) * e_{i,d}``); in WEAK / MOD / STRONG worlds one parent
idea carries a planted shift of 0.15 / 0.35 / 0.70 sd on its event-days (DEF-10: S5's exact effect
sizes and code were not attached; these are documented reconstructions).

Protocols:
* ``P0`` search until the first idea with two-sided p < 0.05 (peek-and-stop on p);
* ``P1`` V5 two-round cap (BH discovery -> Bonferroni validation; round 2 adaptive q <= 0.01);
* ``P2`` 2-week cumulative re-selection and re-testing at p < 0.05;
* ``P4`` e-value peek-and-stop on 10-day ticks (Gaussian mixture, ``c = NS_C``);
* ``P4c`` p-value peek-and-stop on 10-day ticks at ``c = 1.00``;
* ``P5`` greedy-diverse docket (``K = 20``, Jaccard <= 0.5), single-shot one-sided ``p < alpha_1/K``;
* ``P6`` uncapped V5: 24 rounds re-using the same validation days;
* Court variants ``K = 10 / 20 / 50`` diverse.

``python -m bn_forensics.omega_sim 300 sim_out.json`` writes rates, 95% Wilson CIs and the
comparison with the S5 benchmark numbers. The p-value peeking protocols live only in the
``sim_p0_*`` / ``sim_p4c_*`` functions (DEF-05 allow-list); nothing here feeds a verdict.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

WORLD_EFFECTS = {"NULL": 0.0, "WEAK": 0.15, "MOD": 0.35, "STRONG": 0.70}
BENCHMARKS_450 = {
    ("P0", "NULL"): (1.000, 0.990, 1.000), ("P2", "NULL"): (0.695, 0.648, 0.738), ("P6", "NULL"): (0.325, 0.281, 0.372),
    ("P1", "NULL"): (0.000, None, None), ("P1", "MOD"): (0.025, None, None), ("P1", "STRONG"): (0.435, None, None),
    ("P5", "NULL"): (0.023, 0.011, 0.047), ("P5", "WEAK"): (0.030, None, None), ("P5", "MOD"): (0.250, None, None),
    ("P5", "STRONG"): (0.930, None, None), ("P4", "NULL"): (0.000, None, None), ("P4", "STRONG"): (0.238, None, None),
    ("P4c", "NULL"): (0.223, 0.184, 0.266),
}
BENCHMARKS_1100 = {("P5", "NULL"): (0.013, None, None), ("P5", "WEAK"): (0.220, None, None),
                   ("P5", "MOD"): (0.900, None, None), ("P5", "STRONG"): (1.000, None, None), ("P4", "MOD"): (0.333, None, None)}
RHO = (0.15, 0.30, 0.60)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def alpha_session(k: int, alpha: float = 0.05) -> float:
    return alpha * 6.0 / (math.pi ** 2 * k * k)


@dataclass
class World:
    X: np.ndarray        # ideas x days outcomes
    E: np.ndarray        # ideas x days event mask
    disc: np.ndarray     # bool days
    val: np.ndarray      # bool days
    planted: int         # index of the planted idea (-1 in NULL)


def make_world(rng: np.random.Generator, n_days: int, batch: int, effect: float) -> World:
    parents = max(1, batch // 4)
    E = np.zeros((parents * 4, n_days), dtype=bool)
    for p in range(parents):
        rate = rng.uniform(0.20, 0.45)
        base = rng.random(n_days) < rate
        E[p * 4] = base
        for m in range(1, 4):
            keep = base & (rng.random(n_days) < 0.92)
            add = ~base & (rng.random(n_days) < rate * 0.08)
            E[p * 4 + m] = keep | add
    E = E[:batch]
    u = rng.normal(0.0, 1.0, n_days)
    a = np.where(rng.random(batch) < 0.5, 1.0, -1.0)
    X = a[:, None] * 0.5 * u[None, :] + math.sqrt(0.75) * rng.normal(0.0, 1.0, (batch, n_days))
    planted = -1
    if effect != 0.0:
        planted = 0
        X[0] = X[0] + effect * E[0]
    n_disc = round(0.6 * n_days)
    disc = np.zeros(n_days, dtype=bool)
    disc[:n_disc] = True
    return World(X=X, E=E, disc=disc, val=~disc, planted=planted)


def z_scores(world: World, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = world.E & days[None, :]
    n = m.sum(axis=1)
    sums = (world.X * m).sum(axis=1)
    sq = ((world.X ** 2) * m).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / n
        var = (sq - n * mean * mean) / np.maximum(n - 1, 1)
        z = mean / np.sqrt(var / n)
    z[n < 3] = 0.0
    return np.nan_to_num(z), n


def t_sf(t: np.ndarray | float, n: np.ndarray | int) -> np.ndarray | float:
    """Upper tail of Student-t with ``n - 1`` df (the correct test for a sample-sd z statistic)."""
    return sps.t.sf(t, np.maximum(np.asarray(n) - 1, 1))


def greedy_diverse(order: Sequence[int], E: np.ndarray, days: np.ndarray, K: int, jmax: float = 0.5) -> list[int]:
    chosen: list[int] = []
    sets = E & days[None, :]
    for i in order:
        ok = True
        for j in chosen:
            inter = np.count_nonzero(sets[i] & sets[j])
            union = np.count_nonzero(sets[i] | sets[j])
            if union and inter / union > jmax:
                ok = False
                break
        if ok:
            chosen.append(int(i))
        if len(chosen) == K:
            break
    return chosen


def bh_select(p: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(p)
    m = p.size
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.flatnonzero(passed)) + 1) if passed.any() else 0
    return order[:k]


def _claim(world: World, idx: int, sign: float) -> tuple[bool, bool]:
    """(any claim, correct planted claim)."""
    return True, bool(world.planted >= 0 and idx == world.planted and sign > 0)


def sim_p0_search_until_pass(world: World) -> tuple[bool, bool]:
    """P0: walk the batch, stop at the first idea with two-sided p < 0.05 on all days (peeking)."""
    z, n = z_scores(world, np.ones(world.X.shape[1], dtype=bool))
    for i in range(z.size):
        if n[i] >= 3 and 2 * t_sf(abs(z[i]), n[i]) < 0.05:
            return _claim(world, i, np.sign(z[i]))
    return False, False


def sim_p1_two_round_cap(world: World, rng: np.random.Generator, batch: int, effect: float) -> tuple[bool, bool]:
    """P1: two rounds max; BH discovery (q 0.05 then adaptive 0.01) -> Bonferroni validation."""
    for q in (0.05, 0.01):
        z, n_disc = z_scores(world, world.disc)
        p_disc = 2 * t_sf(np.abs(z), n_disc)
        sel = bh_select(p_disc, q)
        if sel.size:
            zv, nv = z_scores(world, world.val)
            for i in sel:
                pv = t_sf(np.sign(z[i]) * zv[i], nv[i])
                if nv[i] >= 3 and pv < 0.05 / sel.size:
                    return _claim(world, int(i), np.sign(z[i]) * np.sign(zv[i]))
        fresh = make_world(rng, world.X.shape[1], batch, effect)
        world = World(fresh.X, fresh.E, world.disc, world.val, fresh.planted)
    return False, False


def sim_p2_cumulative(world: World) -> tuple[bool, bool]:
    """P2: every 10 validation days re-select the best idea on all data so far, test at p < 0.05."""
    val_idx = np.flatnonzero(world.val)
    for end in range(10, val_idx.size + 1, 10):
        upto = world.disc.copy()
        upto[val_idx[:end]] = True
        z, _ = z_scores(world, upto)
        best = int(np.argmax(np.abs(z)))
        seen_val = np.zeros_like(world.val)
        seen_val[val_idx[:end]] = True
        zv, nv = z_scores(world, seen_val)
        if nv[best] >= 3 and t_sf(np.sign(z[best]) * zv[best], nv[best]) < 0.05:
            return _claim(world, best, np.sign(z[best]))
    return False, False


def _forward_stream(world: World, idx: int) -> tuple[np.ndarray, float]:
    vals = world.X[idx][world.E[idx] & world.val]
    disc_vals = world.X[idx][world.E[idx] & world.disc]
    sd = float(np.std(disc_vals, ddof=1)) if disc_vals.size > 2 else 1.0
    return vals, sd


def sim_p4_evalue_peek(world: World, c: float, alpha: float = 0.05) -> tuple[bool, bool]:
    """P4: frozen top idea; Gaussian-mixture e-value on 10-day ticks; stop when e >= 1/alpha."""
    z, _ = z_scores(world, world.disc)
    best = int(np.argmax(np.abs(z)))
    sign = np.sign(z[best]) or 1.0
    vals, sd = _forward_stream(world, best)
    val_days = np.flatnonzero(world.val)
    ev_days = np.flatnonzero(world.E[best] & world.val)
    for end in range(10, val_days.size + 1, 10):
        seen = vals[ev_days <= val_days[end - 1]]
        v = seen.size
        if v == 0:
            continue
        s = sign * seen.sum() / (c * sd)
        terms = [-0.5 * math.log(1 + v * r * r) + r * r * s * s / (2 * (1 + v * r * r)) for r in RHO]
        top = max(terms)
        log_e = top + math.log(sum(math.exp(t - top) for t in terms) / len(RHO))
        if s > 0 and log_e >= math.log(1 / alpha):
            return _claim(world, best, sign)
    return False, False


def sim_p4c_pvalue_peek(world: World) -> tuple[bool, bool]:
    """P4c: same frozen idea, p-value peek-and-stop every 10 days at c = 1.00 (the banned trap)."""
    z, _ = z_scores(world, world.disc)
    best = int(np.argmax(np.abs(z)))
    sign = np.sign(z[best]) or 1.0
    vals, sd = _forward_stream(world, best)
    val_days = np.flatnonzero(world.val)
    ev_days = np.flatnonzero(world.E[best] & world.val)
    for end in range(10, val_days.size + 1, 10):
        seen = vals[ev_days <= val_days[end - 1]]
        if seen.size < 3:
            continue
        zz = sign * seen.mean() / (sd / math.sqrt(seen.size))
        if t_sf(zz, seen.size) < 0.05:
            return _claim(world, best, sign)
    return False, False


def sim_p5_diverse_court(world: World, K: int = 20, min_days: int = 10) -> tuple[bool, bool]:
    """P5: rank on discovery, greedy-diverse K, single-shot one-sided p < alpha_1/K on Court days."""
    z, _ = z_scores(world, world.disc)
    order = np.argsort(-np.abs(z), kind="mergesort")
    docket = greedy_diverse(order, world.E, world.disc, K)
    zv, nv = z_scores(world, world.val)
    thr = alpha_session(1) / K
    claim = correct = False
    for i in docket:
        if nv[i] >= min_days and t_sf(np.sign(z[i]) * zv[i], nv[i]) < thr:
            claim = True
            correct = correct or (world.planted >= 0 and i == world.planted and np.sign(z[i]) > 0)
    return claim, correct


def sim_p6_uncapped(world: World, rng: np.random.Generator, batch: int, effect: float, rounds: int = 24) -> tuple[bool, bool]:
    """P6: V5 without the round cap: 24 rounds of fresh searches re-using the same validation days."""
    for _round in range(rounds):
        z, n_disc = z_scores(world, world.disc)
        p_disc = 2 * t_sf(np.abs(z), n_disc)
        sel = bh_select(p_disc, 0.05)
        if not sel.size:
            sel = np.argsort(p_disc)[:5]
        zv, nv = z_scores(world, world.val)
        for i in sel:
            if nv[i] >= 3 and t_sf(np.sign(z[i]) * zv[i], nv[i]) < 0.05 / sel.size:
                return _claim(world, int(i), np.sign(z[i]))
        fresh = make_world(rng, world.X.shape[1], batch, effect)
        keep_val = world.X[:, world.val]
        X = fresh.X.copy()
        X[:, world.val] = keep_val
        world = World(X, fresh.E, world.disc, world.val, fresh.planted)
    return False, False


def run(n_seeds: int, *, n_days: int = 450, batch: int = 300, c: float = 1.10, worlds: Sequence[str] = tuple(WORLD_EFFECTS),
        only_p4: bool = False, only_p56: bool = False, protocols_subset: Sequence[str] | None = None) -> dict[str, Any]:
    """Run every protocol over ``n_seeds`` seeds per world; returns rates with Wilson CIs."""
    protocols: dict[str, Callable[[World, np.random.Generator, float], tuple[bool, bool]]] = {
        "P0": lambda w, r, e: sim_p0_search_until_pass(w),
        "P1": lambda w, r, e: sim_p1_two_round_cap(w, r, batch, e),
        "P2": lambda w, r, e: sim_p2_cumulative(w),
        "P4": lambda w, r, e: sim_p4_evalue_peek(w, c),
        "P4c": lambda w, r, e: sim_p4c_pvalue_peek(w),
        "P5": lambda w, r, e: sim_p5_diverse_court(w, 20),
        "P6": lambda w, r, e: sim_p6_uncapped(w, r, batch, e),
        "COURT_K10": lambda w, r, e: sim_p5_diverse_court(w, 10),
        "COURT_K50": lambda w, r, e: sim_p5_diverse_court(w, 50),
    }
    if only_p4:
        protocols = {k: v for k, v in protocols.items() if k in ("P4", "P4c")}
    elif only_p56:
        protocols = {k: v for k, v in protocols.items() if k in ("P5", "P6", "COURT_K10", "COURT_K50")}
    if protocols_subset is not None:
        unknown = sorted(set(protocols_subset) - set(protocols))
        if unknown:
            raise ValueError(f"unknown protocols {unknown}")
        protocols = {k: v for k, v in protocols.items() if k in protocols_subset}
    results: dict[str, dict[str, Any]] = {}
    for world_name in worlds:
        effect = WORLD_EFFECTS[world_name]
        counts = dict.fromkeys(protocols, 0)
        for seed in range(n_seeds):
            for p, fn in protocols.items():
                rng = np.random.default_rng([seed, n_days, hash_name(world_name), hash_name(p)])
                world = make_world(rng, n_days, batch, effect)
                claim, correct = fn(world, rng, effect)
                counts[p] += int(claim if world_name == "NULL" else correct)
        for p, k in counts.items():
            lo, hi = wilson(k, n_seeds)
            results.setdefault(p, {})[world_name] = {"k": k, "n": n_seeds, "rate": k / n_seeds, "wilson_lo": lo, "wilson_hi": hi}
    bench = BENCHMARKS_450 if n_days <= 700 else BENCHMARKS_1100
    comparison = []
    for (p, w), (rate, lo_b, hi_b) in sorted(bench.items()):
        got = results.get(p, {}).get(w)
        if got is None:
            continue
        b_lo = lo_b if lo_b is not None else rate
        b_hi = hi_b if hi_b is not None else rate
        overlap = not (got["wilson_hi"] < b_lo or got["wilson_lo"] > b_hi)
        comparison.append({"protocol": p, "world": w, "benchmark": rate, "benchmark_ci": [lo_b, hi_b], "rate": got["rate"],
                           "wilson": [got["wilson_lo"], got["wilson_hi"]], "overlap": overlap})
    return {"config": {"seeds": n_seeds, "NS_D": n_days, "NS_BATCH": batch, "NS_C": c, "worlds": list(worlds),
                       "effects_sd": {w: WORLD_EFFECTS[w] for w in worlds}, "label": "[SIM] synthetic until real-n Null Lab (O13)"},
            "results": results, "comparison": comparison}


def hash_name(name: str) -> int:
    """Stable small integer from a protocol/world name (no reliance on Python's salted hash)."""
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 100_003


def env_config(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    worlds_raw = env.get("NS_WORLDS", "all").strip()
    worlds = tuple(WORLD_EFFECTS) if worlds_raw.lower() == "all" else tuple(w.strip().upper() for w in worlds_raw.split(",") if w.strip())
    bad = [w for w in worlds if w not in WORLD_EFFECTS]
    if bad:
        raise ValueError(f"NS_WORLDS has unknown worlds {bad}")
    return {"n_days": int(env.get("NS_D", "450")), "batch": int(env.get("NS_BATCH", "300")), "c": float(env.get("NS_C", "1.10")),
            "worlds": worlds, "only_p4": env.get("NS_ONLY_P4", "0") == "1", "only_p56": env.get("NS_ONLY_P56", "0") == "1"}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or not args[0].isdigit():
        sys.stderr.write("usage: python -m bn_forensics.omega_sim <n_seeds> <out.json>\n")
        return 3
    cfg = env_config()
    out = run(int(args[0]), **cfg)
    Path(args[1]).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    p5 = out["results"].get("P5", {}).get("NULL")
    if p5:
        sys.stdout.write(f"P5 NULL FPR {p5['rate']:.3f} (Wilson {p5['wilson_lo']:.3f}-{p5['wilson_hi']:.3f})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
