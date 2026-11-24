"""Single-shot diverse Court (FILE-032, FR-11; §10.4 #2, #7, #9; R28).

1. **Docket freeze (Ω-S4):** rank the Playground pool by ``prior_weight * |deflated_lift|``, pick
   up to ``K`` cards greedily with event-day Jaccard ``<= 0.50`` against every chosen card
   (optional Lane M slot reservation, PR-08), hash every card and the Court date list, validate
   against ``court_docket.schema.json`` and write ``court_docket.json`` BEFORE any Court read.
2. **Power pre-check (§10.4 #9):** projected Court event-days = Playground event-day rate x fresh
   Court days; run only if the median >= 30 and ``Court_MDE_sd <= 0.60``. Otherwise
   ``POWER-DEFERRED``: no Court file opened, ``k`` unchanged, no alpha spent, ``WAIT`` action.
3. **Court session (Ω-S5):** open the sealed block once (docket hashes verified, F32), test every
   card one-sided in its frozen sign: ``x_d`` = day mean of (outcome - slot baseline) in ATR20,
   ``T = sign * mean(x_d)``, sign-flip null with ``B_court = max(10000, ceil(10/(alpha_k/K)))``
   draws and ``seed = 42 + k``; ``p = (1 + #{T* >= T}) / (B + 1)``; confirmed iff
   ``p < alpha_k / K`` and ``event_days >= 30`` (else ``COURT-UNDERPOWERED``). Every verdict carries
   ``VIRGIN-COURT`` / ``EXPOSED-COURT`` (> 50% exposed event-days, R28).
4. The 12-column ``tables/event_ledger.csv`` feeds the isolated Skeptic; Court days graduate.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bn_forensics import (
    DocketTamper,
    EventLog,
    InputError,
    InvalidPipeline,
    atomic_write_json,
    iso_ist,
    load_json,
    sha256_json,
)
from bn_forensics.core_stats import (
    BarFrame,
    Ctx,
    EventResult,
    alpha_k,
    b_court,
    court_mde_sd,
    event_study,
    jaccard,
    outcome,
    sign_flip_pvalue,
    slot_means,
    vol_terciles,
    write_csv,
)
from bn_forensics.schemas_validator import validate_or_raise
from bn_forensics.shrinkage_basis import Candidate
from bn_forensics.split_seal import SplitManifest, court_days_hash

EVENT_LEDGER_FIELDS = ("test_id", "date", "bar_ts", "raw_ret", "clean_ret", "baseline_ret", "next_open_ret",
                       "placebo_ret", "vol_tercile", "is_expiry", "is_outlier_day", "overlap_flag")
VERDICT_FIELDS = ("card_id", "session_k", "lane", "target", "horizon", "sign", "alpha_k", "K", "threshold",
                  "B_court", "n_events", "event_days", "T", "lift_atr", "lift_points", "ci_lo", "ci_hi", "p",
                  "verdict", "exposure", "exposed_share", "robust", "blocks_same_sign", "card_hash", "outcome_sd")


def rank_pool(pool: Sequence[Candidate]) -> list[Candidate]:
    """Deterministic ranking: ``prior_weight * |deflated_lift|`` desc, ``card_hash`` asc."""
    return sorted(pool, key=lambda c: (-c.rank_score, c.card_hash))


def greedy_diverse(pool: Sequence[Candidate], K: int, *, jaccard_max: float = 0.50,
                   lane_m_reserve: int = 0) -> list[tuple[Candidate, float]]:
    """Greedy-diverse docket: Jaccard <= ``jaccard_max`` vs every chosen card (Ω-5, F34).

    Lane M reservation (PR-08): up to ``lane_m_reserve`` diverse Lane M cards are seated first,
    then the remaining slots fill in rank order. Returns ``(card, max Jaccard vs chosen)``.
    """
    if not 1 <= K <= 50:
        raise InputError(f"K must be within 1..50, got {K}")
    ranked = rank_pool(pool)
    chosen: list[tuple[Candidate, float]] = []

    def try_add(cand: Candidate) -> bool:
        worst = max((jaccard(cand.event_days, c.event_days) for c, _ in chosen), default=0.0)
        if worst <= jaccard_max and all(c.card_hash != cand.card_hash for c, _ in chosen):
            chosen.append((cand, worst))
            return True
        return False

    reserve = min(lane_m_reserve, K)
    for cand in ranked:
        if reserve <= 0:
            break
        if cand.lane == "M" and try_add(cand):
            reserve -= 1
    for cand in ranked:
        if len(chosen) >= K:
            break
        if any(cand.card_hash == c.card_hash for c, _ in chosen):
            continue
        try_add(cand)
    order = {c.card_hash: i for i, c in enumerate(ranked)}
    return sorted(chosen, key=lambda t: order[t[0].card_hash])


def power_precheck(chosen: Sequence[Candidate], fresh_court_days: int, session_k: int, K: int,
                   cfg: dict[str, Any]) -> dict[str, Any]:
    """§10.4 #9: median projected Court event-days >= 30 and projected MDE <= ceiling."""
    pc = cfg["court"]["power_precheck"]
    proj = [c.event_rate * fresh_court_days for c in chosen]
    med = float(np.median(proj)) if proj else 0.0
    a_k = alpha_k(session_k, float(cfg["court"]["alpha_total"]))
    mde = court_mde_sd(a_k, K, med, float(cfg["stats"]["power"])) if med > 0 else None
    passed = bool(med >= float(pc["median_event_days_min"]) and mde is not None and mde <= float(pc["court_mde_ceiling_sd"]))
    return {"median_event_days": med, "projected_mde_sd": mde, "ceiling_sd": float(pc["court_mde_ceiling_sd"]),
            "passed": passed, "projections": proj}


def freeze_docket(pool: Sequence[Candidate], manifest: SplitManifest, cfg: dict[str, Any], *, K: int | None = None,
                  run_id: str, path: str | Path, log: EventLog | None = None) -> dict[str, Any]:
    """Ω-S4 freeze (+ power pre-check). Writes ``court_docket.json`` atomically; returns it."""
    court = cfg["court"]
    K = int(court["K_default"] if K is None else K)
    k = manifest.session_k + 1
    a_k = alpha_k(k, float(court["alpha_total"]))
    chosen = greedy_diverse(pool, K, jaccard_max=float(court["jaccard_max"]), lane_m_reserve=int(court["lane_m_reserve"]))
    check = power_precheck([c for c, _ in chosen], len(manifest.court_days), k, K, cfg)
    if court_days_hash(manifest.dates("court_days")) != manifest.court_days_hash:
        raise DocketTamper("split manifest court_days_hash mismatch before docket freeze")
    docket = {
        "schema_version": "1.0", "session": k, "alpha_k": round(a_k, 9), "K": K,
        "jaccard_max": float(court["jaccard_max"]), "frozen_at": iso_ist(), "data_version": manifest.data_version,
        "court_days_hash": manifest.court_days_hash, "B_court": b_court(a_k, K, int(court["B_court_min"]), int(court["B_court_factor"])),
        "power_precheck": {k2: check[k2] for k2 in ("median_event_days", "projected_mde_sd", "ceiling_sd", "passed")},
        "exposure": {"virgin_days": manifest.virgin_court_days, "exposed_days": manifest.exposed_court_days,
                     "policy": manifest.exposure_policy},
        "cards": [{"card_id": c.card_id, "card_hash": c.card_hash, "sign": int(c.sign), "jaccard_max_vs_chosen": round(j, 6),
                   "lane": c.lane, "target": c.target, "horizon": int(c.horizon), "prior_weight": float(c.prior_weight),
                   "rank_score": c.rank_score, "deflated_lift": c.deflated_lift,
                   "playground_event_days": len(c.event_days),
                   "projected_court_event_days": round(c.event_rate * len(manifest.court_days), 4)} for c, j in chosen],
        "threshold": a_k / K, "court_days_count": len(manifest.court_days), "seed": int(court["seed_offset_base"]) + k,
        "run_id": run_id, "status": "FROZEN" if check["passed"] else "POWER-DEFERRED",
    }
    docket["docket_hash"] = docket_hash(docket)
    validate_or_raise(docket, "court_docket", what="court_docket.json")
    atomic_write_json(path, docket)
    if log is not None:
        log.emit("docket_frozen", {"session_k": k, "K": K, "cards": len(chosen), "docket_hash": docket["docket_hash"]},
                 stage="Ω-S4", role="AG-02")
        if not check["passed"]:
            log.emit("power_deferred", {"session_k": k, "median_event_days": check["median_event_days"],
                                        "projected_mde_sd": check["projected_mde_sd"]}, stage="Ω-S5", role="AG-02")
    return docket


def docket_hash(docket: dict[str, Any]) -> str:
    """SHA-256 over the canonical docket without its own ``docket_hash``."""
    return sha256_json({k: v for k, v in docket.items() if k != "docket_hash"})


def verify_docket(path: str | Path, compiled: dict[str, Any], manifest: SplitManifest) -> dict[str, Any]:
    """F32 guard: docket file hash, every ``card_hash`` and ``court_days_hash`` must match."""
    docket = load_json(path)
    validate_or_raise(docket, "court_docket", what=str(path))
    if docket.get("docket_hash") != docket_hash(docket):
        raise DocketTamper("court_docket.json changed after freeze (docket_hash mismatch, F32)")
    if docket["court_days_hash"] != manifest.court_days_hash or court_days_hash(manifest.dates("court_days")) != docket["court_days_hash"]:
        raise DocketTamper("court_days_hash mismatch between docket and sealed split (F32)")
    for entry in docket["cards"]:
        cc = compiled.get(entry["card_id"])
        if cc is None:
            raise DocketTamper(f"docket card {entry['card_id']} missing from the frozen pool (F32)")
        if cc.card_hash() != entry["card_hash"]:
            raise DocketTamper(f"card {entry['card_id']} mutated after freeze (card_hash mismatch, F32)")
    return docket


# =============================================================================================
# Court session (Ω-S5)
# =============================================================================================
@dataclass
class CourtVerdict:
    card_id: str
    verdict: str
    exposure: str
    exposed_share: float
    p: float
    T: float
    event_days: int
    n_events: int
    robust: str
    result: EventResult | None
    row: dict[str, Any]


def exposure_tag(event_days: Sequence[str], exposure: dict[str, str], majority: float = 0.50) -> tuple[str, float]:
    """``EXPOSED-COURT`` when more than ``majority`` of the card's Court event-days are EXPOSED (R28)."""
    if not event_days:
        return "VIRGIN-COURT", 0.0
    share = float(np.mean([exposure.get(str(d)) == "EXPOSED" for d in event_days]))
    return ("EXPOSED-COURT" if share > majority else "VIRGIN-COURT"), share


def court_pvalue(x_d: np.ndarray, sign: int, session_k: int, K: int, cfg: dict[str, Any]) -> tuple[float, int]:
    """Day-level one-sided sign-flip p in the frozen sign with ``B_court`` draws, ``seed = 42 + k``."""
    court = cfg["court"]
    a_k = alpha_k(session_k, float(court["alpha_total"]))
    B = b_court(a_k, K, int(court["B_court_min"]), int(court["B_court_factor"]))
    seed = int(court["seed_offset_base"]) + int(session_k)
    p = sign_flip_pvalue(np.asarray(x_d, float), int(sign), B, seed, two_sided=False, chunk=int(court["chunk_rows"]))
    return p, B


def run_session(docket: dict[str, Any], compiled: dict[str, Any], court_ctx: Any, manifest: SplitManifest,
                cfg: dict[str, Any], run_ctx: Ctx, *, playground_vol_cutoffs: tuple[float, float] | None = None,
                expiries: Sequence[dt.date] = (), raw_frame: BarFrame | None = None,
                tables_dir: str | Path | None = None, log: EventLog | None = None) -> list[CourtVerdict]:
    """Single-shot Court on the sealed block (caller has opened it via ``CourtSeal.court_reader``)."""
    if docket["status"] != "FROZEN":
        raise InputError("Court session refused: docket is POWER-DEFERRED (no alpha spent)")
    graduated = {d for block in manifest.graduated for d in block}
    if set(manifest.court_days) & graduated or docket["session"] != manifest.session_k + 1:
        raise InvalidPipeline("Court block already used / session index out of order (F37)", tripwire_id="F37")
    frame: BarFrame = court_ctx.frame
    court_set = set(manifest.court_days)
    day_mask = np.array([str(d) in court_set for d in frame.days], dtype=bool)
    if not day_mask.any():
        raise InputError("Court frame holds no sealed Court days")
    k, K = int(docket["session"]), int(docket["K"])
    threshold = float(docket["threshold"])
    majority = float(cfg["court"]["exposure_majority"])
    min_days = int(cfg["court"]["event_days_min"])
    verdicts: list[CourtVerdict] = []
    ledger_rows: list[dict[str, Any]] = []
    outlier_days = _outlier_days(frame, day_mask, float(cfg["stats"]["outlier_day_frac"]))
    terc, _ = vol_terciles(frame, playground_vol_cutoffs)
    exp_set = {str(e) for e in expiries}
    for entry in docket["cards"]:
        cc = compiled[entry["card_id"]]
        sign = int(entry["sign"])
        mask, direction = cc.evaluate(court_ctx)
        res = event_study(frame, mask, direction=direction, side=cc.card.get("side", "both"), target=entry["target"],
                          horizon=int(entry["horizon"]), baseline=cc.card.get("baseline", "slot_matched"),
                          day_mask=day_mask, test_id=entry["card_id"], ctx=run_ctx, vol_cutoffs=playground_vol_cutoffs)
        n_days = res.effective_n
        t_stat = float(sign * np.mean(res.x_d)) if n_days else float("nan")
        p, B = court_pvalue(res.x_d, sign, k, K, cfg) if n_days else (1.0, int(docket["B_court"]))
        ev_days = [str(d) for d in res.event_days]
        exposure, share = exposure_tag(ev_days, manifest.exposure, majority)
        if n_days < min_days:
            verdict = "COURT-UNDERPOWERED"
        elif p < threshold:
            verdict = "COURT-CONFIRMED"
        else:
            verdict = "COURT-REJECTED"
        row = {"card_id": entry["card_id"], "session_k": k, "lane": entry["lane"], "target": entry["target"],
               "horizon": int(entry["horizon"]), "sign": sign, "alpha_k": docket["alpha_k"], "K": K, "threshold": threshold,
               "B_court": B, "n_events": res.n_events, "event_days": n_days, "T": t_stat, "lift_atr": res.lift_atr,
               "lift_points": res.lift_points, "ci_lo": res.ci_lo, "ci_hi": res.ci_hi, "p": p, "verdict": verdict,
               "exposure": exposure, "exposed_share": share, "robust": res.stability,
               "blocks_same_sign": res.blocks_same_sign, "card_hash": entry["card_hash"], "outcome_sd": res.outcome_sd}
        verdicts.append(CourtVerdict(entry["card_id"], verdict, exposure, share, p, t_stat, n_days, res.n_events,
                                     res.stability, res, row))
        ledger_rows += event_ledger_rows(entry["card_id"], frame, res, cc, court_ctx, day_mask, terc, exp_set,
                                         outlier_days, cfg, raw_frame=raw_frame)
    if tables_dir is not None:
        root = Path(tables_dir)
        write_csv(root / "court_verdicts.csv", VERDICT_FIELDS, [v.row for v in verdicts])
        write_csv(root / "event_ledger.csv", EVENT_LEDGER_FIELDS, ledger_rows)
    if log is not None:
        log.emit("court_verdict_issued", {"session_k": k, "alpha_k": docket["alpha_k"], "K": K,
                                          "confirmed": sum(v.verdict == "COURT-CONFIRMED" for v in verdicts),
                                          "rejected": sum(v.verdict == "COURT-REJECTED" for v in verdicts),
                                          "underpowered": sum(v.verdict == "COURT-UNDERPOWERED" for v in verdicts),
                                          "court_days_hash": docket["court_days_hash"]}, stage="Ω-S5", role="AG-02")
        log.emit("exposure_tagged", {"virgin_days": manifest.virgin_court_days, "exposed_days": manifest.exposed_court_days,
                                     "exposed_share_per_card": {v.card_id: v.exposed_share for v in verdicts}},
                 stage="Ω-S5", role="AG-02")
    return verdicts


def _outlier_days(frame: BarFrame, day_mask: np.ndarray, frac: float) -> set[str]:
    """Top/bottom ``frac`` of Court days by |day return| and by range (R20)."""
    d = frame.day_ohlc()
    ret = np.abs(d["close"] - d["open"])
    rng = d["high"] - d["low"]
    idx = np.flatnonzero(day_mask)
    n = max(1, int(np.ceil(frac * idx.size)))
    out: set[str] = set()
    for metric in (ret, rng):
        vals = metric[idx]
        order = np.argsort(vals)
        out |= {str(frame.days[idx[i]]) for i in order[-n:]}
        out |= {str(frame.days[idx[i]]) for i in order[:n]}
    return out


def event_ledger_rows(test_id: str, frame: BarFrame, res: EventResult, cc: Any, ctx: Any, day_mask: np.ndarray,
                      terc: np.ndarray, expiries: set[str], outlier_days: set[str], cfg: dict[str, Any], *,
                      raw_frame: BarFrame | None) -> list[dict[str, Any]]:
    """12-column ``tables/event_ledger.csv`` rows (S4 §8) for the isolated Skeptic."""
    target, h = res.target, res.horizon
    raw_out = outcome(raw_frame if raw_frame is not None else frame, target, h)["atr"]
    out = outcome(frame, target, h)["atr"]
    use = frame.day_mask_to_bar_mask(day_mask)
    sm = slot_means(out, frame.slot, use)
    shift = max(1, int(cfg["stats"]["placebo_shift_min"]) // frame.interval_min)
    _, direction = cc.evaluate(ctx)
    side = cc.card.get("side", "both")
    eff_dir = direction * (-1.0 if side == "reversal" else 1.0)
    if target not in ("T1", "T3", "T6", "T8"):
        eff_dir = np.ones(frame.n)
    nxt_open = np.full(frame.n, np.nan)
    idx_all = np.arange(frame.n - h)
    same = frame.day_idx[np.minimum(idx_all + h, frame.n - 1)] == frame.day_idx[idx_all]
    valid = idx_all[same & (idx_all + 1 < frame.n)]
    nxt_open[valid] = (frame.c[valid + h] - frame.o[valid + 1]) / frame.atr20[valid]
    rows = []
    last_event_in_day: dict[int, int] = {}
    has_cal = bool(expiries)
    for pos, i in enumerate(res.event_index):
        d = int(frame.day_idx[i])
        j = i + shift if (i + shift < frame.n and frame.day_idx[i + shift] == d) else i - shift
        placebo = float(out[j] * eff_dir[i] - sm[frame.slot[j]] * eff_dir[i]) if 0 <= j < frame.n and frame.day_idx[j] == d else float("nan")
        overlap = 1 if (d in last_event_in_day and i - last_event_in_day[d] < h) else 0
        last_event_in_day[d] = i
        date = str(frame.days[d])
        rows.append({
            "test_id": test_id, "date": date, "bar_ts": f"{date}T{int(frame.minute[i]) // 60:02d}:{int(frame.minute[i]) % 60:02d}",
            "raw_ret": float(raw_out[i] * eff_dir[i]) if raw_frame is not None else float(res.signed_outcome[pos]),
            "clean_ret": float(res.signed_outcome[pos]), "baseline_ret": float(res.baseline_values[pos]),
            "next_open_ret": float(nxt_open[i] * eff_dir[i]), "placebo_ret": placebo, "vol_tercile": int(terc[i]),
            "is_expiry": (1 if date in expiries else 0) if has_cal else "", "is_outlier_day": 1 if date in outlier_days else 0,
            "overlap_flag": overlap,
        })
    return rows


def playground_event_rows(test_id: str, frame: BarFrame, res: EventResult) -> list[dict[str, Any]]:
    """Discovery-side per-event rows (verification sample & leaderboard audit)."""
    return [{"test_id": test_id, "date": str(frame.days[frame.day_idx[i]]),
             "bar_ts": f"{frame.days[frame.day_idx[i]]}T{int(frame.minute[i]) // 60:02d}:{int(frame.minute[i]) % 60:02d}",
             "clean_ret": float(res.signed_outcome[p]), "baseline_ret": float(res.baseline_values[p])}
            for p, i in enumerate(res.event_index)]


__all__ = [
    "EVENT_LEDGER_FIELDS", "VERDICT_FIELDS", "Candidate", "CourtVerdict", "court_pvalue", "docket_hash",
    "event_ledger_rows", "exposure_tag", "freeze_docket", "greedy_diverse", "playground_event_rows",
    "power_precheck", "rank_pool", "run_session", "verify_docket",
]
