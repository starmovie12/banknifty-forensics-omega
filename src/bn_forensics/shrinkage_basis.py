"""Shrinkage, cost model, tradability tags, Lane M mechanism checks, NS and Orthogonal Basis
(FILE-031, FR-10; §10.4 #5, #8, #11).

* Empirical-Bayes winner's-curse shrinkage (§10.4 #8): ``lift_i ~ N(theta_i, se_i^2)``,
  ``theta ~ N(0, tau^2)``, ``tau^2 = max(0, mean(lift_i^2 - se_i^2))`` pooled over EVERY card of the
  same target family in the Idea Ledger (discards included);
  ``deflated_lift_i = lift_i * tau^2 / (tau^2 + se_i^2)``; fewer than 50 cards or ``tau^2 = 0``
  falls back to the Null Lab haircut ratio.
* Cost model (§10.4 #11) in :class:`decimal.Decimal` with every leg total rounded to paise:
  ``option_cost_pts = charges_rt / qty + 2 * (half_spread_eff + slippage)``,
  ``underlying_cost_pts = option_cost_pts / |delta|`` (default ``cost_series = option_buyer``).
* Tradability tags strictly from ``abs(deflated_lift)`` against the matching series cost (R16, R25):
  ``T1`` >= 2x cost, ``T2`` >= 1x, ``T3`` below cost (``REAL BUT UNTRADABLE``), ``T?`` unknown cost.
* Lane M dose-gradient and negative-control checks -> ``MECHANISM-CONSISTENT`` / ``MECHANISM-REFUTED``.
* AP-clone-aware Novelty Score (``J < 0.50`` vs every AP card) and the Orthogonal Findings Basis.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np

from bn_forensics import InputError
from bn_forensics.core_stats import STRUCTURE_TARGETS, jaccard, write_csv

PAISE = Decimal("0.01")
ORTHO_FIELDS = ("finding_id", "independent_effect_id", "representative", "members", "n_members",
                "max_jaccard_in_group", "max_corr_in_group")
OVERLAP_FIELDS = ("card_a", "card_b", "jaccard", "common_event_days", "note")


# =============================================================================================
# Empirical-Bayes shrinkage (§10.4 #8)
# =============================================================================================
def eb_tau2(lifts: Sequence[float], ses: Sequence[float]) -> float:
    """``tau^2 = max(0, mean(lift_i^2 - se_i^2))`` over finite pairs."""
    lift = np.asarray(lifts, float)
    se = np.asarray(ses, float)
    ok = np.isfinite(lift) & np.isfinite(se)
    if not ok.any():
        return 0.0
    return float(max(0.0, np.mean(lift[ok] ** 2 - se[ok] ** 2)))


def deflate(lift: float, se: float, tau2: float) -> float:
    """``deflated_lift = lift * tau^2 / (tau^2 + se^2)``."""
    if not (math.isfinite(lift) and math.isfinite(se)):
        return float("nan")
    denom = tau2 + se * se
    return lift * tau2 / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class ShrinkageFit:
    family: str
    n_cards: int
    tau2: float
    method: str          # "EB" | "HAIRCUT"
    haircut: float | None


def fit_families(records: Iterable[Mapping[str, Any]], *, min_cards: int = 50,
                 haircut: float | None = None) -> dict[str, ShrinkageFit]:
    """Per target family: EB fit over every evaluated card (discards included) or haircut fallback."""
    pools: dict[str, tuple[list[float], list[float]]] = {}
    for rec in records:
        if rec.get("status") == "REJECTED":
            continue
        lift, se = rec.get("lift"), rec.get("se")
        if lift is None or se is None:
            continue
        fam = str(rec.get("family") or rec.get("target") or "T1")
        pools.setdefault(fam, ([], []))
        pools[fam][0].append(float(lift))
        pools[fam][1].append(float(se))
    fits = {}
    for fam, (lifts, ses) in sorted(pools.items()):
        tau2 = eb_tau2(lifts, ses)
        if len(lifts) < min_cards or tau2 == 0.0:
            fits[fam] = ShrinkageFit(fam, len(lifts), tau2, "HAIRCUT", haircut)
        else:
            fits[fam] = ShrinkageFit(fam, len(lifts), tau2, "EB", None)
    return fits


def deflated_lift(lift: float, se: float, fit: ShrinkageFit | None) -> float | None:
    """Apply a family fit; returns None when the haircut fallback has no Null Lab ratio yet."""
    if fit is None:
        return None
    if fit.method == "EB":
        return deflate(lift, se, fit.tau2)
    if fit.haircut is None or not math.isfinite(lift):
        return None
    return float(lift * fit.haircut)


def haircut_ratio(planted: Sequence[float], recovered: Sequence[float]) -> float:
    """Null Lab haircut: median ``recovered / planted`` over planted-edge worlds, clipped to [0, 1]."""
    p = np.asarray(planted, float)
    r = np.asarray(recovered, float)
    ok = np.isfinite(p) & np.isfinite(r) & (p != 0)
    if not ok.any():
        raise InputError("haircut_ratio needs at least one finite planted/recovered pair")
    return float(np.clip(np.median(r[ok] / p[ok]), 0.0, 1.0))


# =============================================================================================
# Cost model (§10.4 #11, Decimal, per-leg paise rounding)
# =============================================================================================
@dataclass(frozen=True)
class CostModel:
    brokerage: Decimal
    exchange_pct: Decimal
    sebi_pct: Decimal
    ipft_pct: Decimal
    gst_pct: Decimal
    stamp_buy_pct: Decimal
    stt_schedule: tuple[tuple[dt.date, Decimal], ...]
    half_spread: Decimal
    slippage: Decimal
    delta_default: Decimal

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> CostModel:
        cm = cfg["cost_model"]
        sched = sorted(((dt.date.fromisoformat(r["from"]), Decimal(str(r["pct"]))) for r in cm["stt_sell_pct_schedule"]),
                       reverse=True)
        return cls(brokerage=Decimal(str(cm["brokerage_per_order"])), exchange_pct=Decimal(str(cm["exchange_pct"])),
                   sebi_pct=Decimal(str(cm["sebi_pct"])), ipft_pct=Decimal(str(cm["ipft_pct"])),
                   gst_pct=Decimal(str(cm["gst_pct"])), stamp_buy_pct=Decimal(str(cm["stamp_buy_pct"])),
                   stt_schedule=tuple(sched), half_spread=Decimal(str(cm["half_spread_estimate"])),
                   slippage=Decimal(str(cm["slippage"])), delta_default=Decimal(str(cm["delta_default"])))

    def stt_pct(self, on: dt.date) -> Decimal:
        for start, pct in self.stt_schedule:
            if on >= start:
                return pct
        return self.stt_schedule[-1][1]

    def leg_charges(self, premium: Decimal, qty: int, side: str, on: dt.date) -> Decimal:
        """One order's charges; components unrounded, the leg TOTAL rounded to paise (ROUND_HALF_UP)."""
        if side not in ("buy", "sell"):
            raise InputError(f"side must be buy or sell, got {side!r}")
        turnover = Decimal(premium) * Decimal(qty)
        hundred = Decimal(100)
        exchange = turnover * self.exchange_pct / hundred
        sebi = turnover * self.sebi_pct / hundred
        ipft = turnover * self.ipft_pct / hundred
        gst = (self.brokerage + exchange + sebi) * self.gst_pct / hundred
        total = self.brokerage + exchange + sebi + gst + ipft
        if side == "buy":
            total += turnover * self.stamp_buy_pct / hundred
        else:
            total += turnover * self.stt_pct(on) / hundred
        return total.quantize(PAISE, rounding=ROUND_HALF_UP)

    def charges_round_trip(self, premium: Decimal, qty: int, on: dt.date) -> Decimal:
        return self.leg_charges(premium, qty, "buy", on) + self.leg_charges(premium, qty, "sell", on)

    def option_cost_pts(self, premium: Decimal | float | str, lots: int, lot_size: int, on: dt.date,
                        half_spread: Decimal | None = None) -> Decimal:
        """``charges_rt(P, qty) / qty + 2 * (half_spread_eff + slippage)`` in option points."""
        if lot_size <= 0 or lots <= 0:
            raise InputError("lots and lot_size must be positive (lot size comes from contracts.csv)")
        qty = int(lots) * int(lot_size)
        prem = Decimal(str(premium))
        hs = self.half_spread if half_spread is None else Decimal(half_spread)
        return self.charges_round_trip(prem, qty, on) / Decimal(qty) + 2 * (hs + self.slippage)

    def underlying_cost_pts(self, option_cost: Decimal, delta: Decimal | float | None = None) -> Decimal:
        """Index-point equivalent for an option buyer: ``option_cost_pts / |delta|``."""
        d = abs(Decimal(str(delta))) if delta is not None else self.delta_default
        if d == 0:
            raise InputError("delta must be non-zero")
        return option_cost / d

    def worked_table(self, lot_size: int | None, on: dt.date, premiums: Sequence[str],
                     lots_list: Sequence[int]) -> list[dict[str, Any]]:
        """Break-even table for ``strategy_input_pack.cost_model`` (USER TO CONFIRM without lot size)."""
        rows = []
        for prem in premiums:
            for lots in lots_list:
                if lot_size:
                    opt = self.option_cost_pts(prem, lots, lot_size, on)
                    und = self.underlying_cost_pts(opt)
                    rows.append({"premium": float(prem), "lots": int(lots), "lot_size": int(lot_size),
                                 "option_cost_pts": float(opt.quantize(PAISE, rounding=ROUND_HALF_UP)),
                                 "underlying_cost_pts": float(und.quantize(PAISE, rounding=ROUND_HALF_UP))})
                else:
                    rows.append({"premium": float(prem), "lots": int(lots), "lot_size": None,
                                 "option_cost_pts": None, "underlying_cost_pts": None})
        return rows


def series_costs(cfg: Mapping[str, Any], *, lot_size: int | None, on: dt.date,
                 reference_premium: str | None = None) -> dict[str, float | None]:
    """Matching-series costs in points: underlying, option, option_buyer (index-equivalent)."""
    cm = cfg["cost_model"]
    model = CostModel.from_config(cfg)
    out: dict[str, float | None] = {"underlying": None, "option": None, "option_buyer": None}
    if cm.get("underlying_cost_pts") is not None:
        out["underlying"] = float(cm["underlying_cost_pts"])
    if lot_size:
        prem = reference_premium or cm["reference_premiums"][1]
        opt = model.option_cost_pts(prem, int(cm["reference_lots"][0]), lot_size, on)
        out["option"] = float(opt)
        out["option_buyer"] = float(model.underlying_cost_pts(opt))
    return out


def tradability_tag(target: str, deflated_lift_atr: float | None, atr_points: float, cost_series: str,
                    costs: Mapping[str, float | None], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """T1/T2/T3/T?/STRUCTURE-ONLY strictly from ``abs(deflated_lift)`` (R16, R25, F2.4)."""
    sh = cfg["shrinkage"]
    if target in STRUCTURE_TARGETS or cost_series == "structure_only":
        return {"tag": "STRUCTURE-ONLY", "ratio": None, "needed_effect_atr20": None, "cost_pts": None}
    cost = costs.get(cost_series)
    if cost is None or deflated_lift_atr is None or not math.isfinite(deflated_lift_atr) or not math.isfinite(atr_points) or atr_points <= 0:
        return {"tag": "T?", "ratio": None, "needed_effect_atr20": None, "cost_pts": cost,
                "note": "USER TO CONFIRM costs (D-1)" if cost is None else "lift or ATR unavailable"}
    lift_pts = abs(deflated_lift_atr) * atr_points
    ratio = lift_pts / cost if cost > 0 else float("inf")
    if ratio >= float(sh["t1_cost_multiple"]):
        tag = "T1"
    elif ratio >= float(sh["t2_cost_multiple"]):
        tag = "T2"
    else:
        tag = "T3"
    return {"tag": tag, "ratio": ratio, "needed_effect_atr20": cost / atr_points, "cost_pts": cost,
            "label": "REAL BUT UNTRADABLE" if tag == "T3" else ""}


# =============================================================================================
# Lane M mechanism checks (FR-10, V5-4)
# =============================================================================================
def gradient_sign(expected_gradient: str) -> int:
    """+1 if 'more dose -> more effect', -1 if 'less dose -> more effect' (text in the card)."""
    text = expected_gradient.lower()
    if "->" not in text:
        raise InputError(f"expected_gradient must read 'X -> Y', got {expected_gradient!r}")
    left, right = text.split("->", 1)
    low_words = r"\b(smaller|lower|less|fewer|closer|shorter|decreas\w*)\b"
    high_words = r"\b(larger|higher|more|greater|bigger|further|longer|increas\w*)\b"
    left_sign = -1 if re.search(low_words, left) else (1 if re.search(high_words, left) else 0)
    right_sign = -1 if re.search(low_words, right) else (1 if re.search(high_words, right) else 0)
    if left_sign == 0 or right_sign == 0:
        raise InputError(f"cannot read the gradient direction from {expected_gradient!r}")
    return left_sign * right_sign


def mechanism_check(residuals: np.ndarray, dose: np.ndarray, event_day: np.ndarray, expected_sign: int,
                    control_residuals: np.ndarray, *, n_bins: int = 3, control_max_ratio: float = 0.50) -> dict[str, Any]:
    """Dose gradient (bin lifts monotone in the expected direction, day-clustered slope t >= 2)
    and negative control (|control lift| < 50% of |treated lift|).
    """
    res = np.asarray(residuals, float)
    dose = np.asarray(dose, float)
    days = np.asarray(event_day)
    ok = np.isfinite(res) & np.isfinite(dose)
    res, dose, days = res[ok], dose[ok], days[ok]
    if res.size < 3 * n_bins:
        return {"status": "MECHANISM-REFUTED", "reason": "too few events for a dose gradient", "bin_lifts": [],
                "slope_t": float("nan"), "control_lift": float("nan")}
    edges = np.quantile(dose, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.searchsorted(edges, dose, side="right")
    bin_lifts = [float(np.mean(res[bins == b])) if (bins == b).any() else float("nan") for b in range(n_bins)]
    x = bins.astype(float) - bins.mean()
    slope = float(np.sum(x * (res - res.mean())) / np.sum(x * x)) if np.sum(x * x) > 0 else 0.0
    resid = res - res.mean() - slope * x
    uniq = np.unique(days)
    score = np.array([np.sum(x[days == d] * resid[days == d]) for d in uniq])
    se = math.sqrt(float(np.sum(score ** 2))) / float(np.sum(x * x)) if np.sum(x * x) > 0 else float("inf")
    slope_t = slope / se if se > 0 else 0.0
    diffs = np.diff(np.asarray(bin_lifts))
    monotone = bool(np.all(np.sign(diffs) == expected_sign)) if np.isfinite(diffs).all() else False
    treated = float(np.mean(res))
    ctrl = np.asarray(control_residuals, float)
    ctrl = ctrl[np.isfinite(ctrl)]
    control_lift = float(np.mean(ctrl)) if ctrl.size else float("nan")
    control_ok = bool(ctrl.size and abs(control_lift) < control_max_ratio * abs(treated))
    gradient_ok = monotone and expected_sign * slope_t >= 2.0
    status = "MECHANISM-CONSISTENT" if gradient_ok and control_ok else "MECHANISM-REFUTED"
    reason = []
    if not gradient_ok:
        reason.append(f"flat or wrong-way dose gradient (bins {[f'{b:.3f}' for b in bin_lifts]}, t={slope_t:.2f})")
    if not control_ok:
        reason.append(f"negative control lift {control_lift:.3f} not < {control_max_ratio:.0%} of treated {treated:.3f}")
    return {"status": status, "reason": "; ".join(reason) or "gradient and control consistent",
            "bin_lifts": bin_lifts, "slope_t": slope_t, "control_lift": control_lift, "treated_lift": treated}


def demote_grade(grade: str, tiers: int = 1) -> str:
    """Demote an evidence grade (A > B > C > D; Ω-A > Ω-B > dropped)."""
    ladders = (["A", "B", "C", "D"], ["Ω-A", "Ω-B", "NONE"])
    for ladder in ladders:
        if grade in ladder:
            return ladder[min(len(ladder) - 1, ladder.index(grade) + tiers)]
    return grade


# =============================================================================================
# Novelty Score and Orthogonal Basis (§10.4 #5, R25, FM-02, FM-21)
# =============================================================================================
def novelty_score(card_days: Mapping[str, Iterable[Any]], ap_days: Mapping[str, Iterable[Any]],
                  threshold: float = 0.50) -> dict[str, Any]:
    """Share of executed non-AP cards whose max Jaccard vs every AP card is below ``threshold``."""
    ap_sets = {k: set(v) for k, v in ap_days.items()}
    rows = []
    for cid, days in sorted(card_days.items()):
        s = set(days)
        best_ap, best = "", 0.0
        for ap_id, ap_set in sorted(ap_sets.items()):
            j = jaccard(s, ap_set)
            if j > best:
                best_ap, best = ap_id, j
        rows.append({"card_id": cid, "max_jaccard_vs_ap": best, "closest_ap": best_ap, "novel": best < threshold})
    n = len(rows)
    novel = sum(1 for r in rows if r["novel"])
    return {"ns": (novel / n) if n else None, "n_cards": n, "novel": novel,
            "clones": [r for r in rows if not r["novel"]], "rows": rows}


def _corr_on_common_days(a: Mapping[Any, float], b: Mapping[Any, float]) -> float:
    common = sorted(set(a) & set(b))
    if len(common) < 10:
        return 0.0
    xa = np.array([a[d] for d in common], float)
    xb = np.array([b[d] for d in common], float)
    if np.std(xa) == 0 or np.std(xb) == 0:
        return 0.0
    return float(np.corrcoef(xa, xb)[0, 1])


def orthogonal_basis(findings: Sequence[Mapping[str, Any]], *, jaccard_thr: float = 0.50,
                     corr_thr: float = 0.50) -> tuple[list[dict[str, Any]], int]:
    """Group findings whose event-day Jaccard >= thr or day-residual correlation >= thr.

    Each finding needs ``finding_id``, ``event_days`` (iterable) and ``x_by_day`` (day -> value).
    Returns basis rows and ``N_indep`` (number of independent effect groups).
    """
    ids = [str(f["finding_id"]) for f in findings]
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    jac = np.zeros((len(ids), len(ids)))
    cor = np.zeros((len(ids), len(ids)))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            jac[i, j] = jac[j, i] = jaccard(findings[i]["event_days"], findings[j]["event_days"])
            cor[i, j] = cor[j, i] = abs(_corr_on_common_days(findings[i].get("x_by_day", {}), findings[j].get("x_by_day", {})))
            if jac[i, j] >= jaccard_thr or cor[i, j] >= corr_thr:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(ids)):
        groups.setdefault(find(i), []).append(i)
    ordered = sorted(groups.values(), key=lambda g: ids[min(g)])
    rows = []
    for k, members in enumerate(ordered, start=1):
        eff = f"E-{k:02d}"
        rep = ids[min(members)]
        sub_j = max((jac[a, b] for a in members for b in members if a != b), default=0.0)
        sub_c = max((cor[a, b] for a in members for b in members if a != b), default=0.0)
        for m in sorted(members):
            rows.append({"finding_id": ids[m], "independent_effect_id": eff, "representative": rep,
                         "members": ";".join(ids[x] for x in sorted(members)), "n_members": len(members),
                         "max_jaccard_in_group": sub_j, "max_corr_in_group": sub_c})
    return rows, len(ordered)


def overlap_rows(cards: Mapping[str, Iterable[Any]], *, min_jaccard: float = 0.0,
                 notes: Mapping[tuple[str, str], str] | None = None) -> list[dict[str, Any]]:
    """Pairwise event-day Jaccard overlaps (``tables/overlap.csv``)."""
    sets = {k: set(v) for k, v in cards.items()}
    keys = sorted(sets)
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            j = jaccard(sets[a], sets[b])
            if j > min_jaccard:
                rows.append({"card_a": a, "card_b": b, "jaccard": j, "common_event_days": len(sets[a] & sets[b]),
                             "note": (notes or {}).get((a, b), "")})
    return rows


def write_basis_tables(tables_dir: str | Path, basis: Sequence[Mapping[str, Any]],
                       overlaps: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    """Write ``tables/orthogonal_basis.csv`` and ``tables/overlap.csv``."""
    root = Path(tables_dir)
    a = write_csv(root / "orthogonal_basis.csv", ORTHO_FIELDS, basis)
    b = write_csv(root / "overlap.csv", OVERLAP_FIELDS, overlaps)
    return a, b


__all__ = [
    "ORTHO_FIELDS", "OVERLAP_FIELDS", "CostModel", "ShrinkageFit", "deflate", "deflated_lift",
    "demote_grade", "eb_tau2", "fit_families", "gradient_sign", "haircut_ratio", "mechanism_check",
    "novelty_score", "orthogonal_basis", "overlap_rows", "series_costs", "tradability_tag",
    "write_basis_tables",
]
