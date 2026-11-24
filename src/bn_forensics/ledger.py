"""Lifelong anytime-valid e-value Ledger (FILE-033, FR-13; §10.4 #4, #12; X04, X05, FM-44).

Per registered card (frozen sign, frozen at the end of its pre-freeze history):

* predictable, baseline-centred residual on post-freeze event-days only:
  ``r_d = x_d - bbar_{d-1}`` where ``bbar_{d-1}`` is the expanding mean of the card's slot-matched
  baseline series through day ``d-1`` (pre-freeze history included) - never the card's own mean;
* ``S = sum(r_d)``, ``v = n_events``, ``sigma = c * sd_freeze`` (``c = 1.10``; ``sd_freeze`` = sd of the
  same residuals on pre-freeze event-days), ``s = sign_frozen * S / sigma``;
* ``log e = log(mean_rho[(1 + v rho^2)^(-1/2) exp(rho^2 s^2 / (2 (1 + v rho^2)))])`` over
  ``rho in {0.15, 0.30, 0.60}``; ``sd_freeze <= 1e-12`` or ``v == 0`` -> ``log e = 0``
  (``DEGENERATE-SERIES``);
* step-up e-BH over the registered cards: ``k* = max{k : e_(k) >= M / (alpha * k)}`` with the
  lifetime budget ``M = 20,000``; ranks ``1..k*`` are ``LEDGER-CONFIRMED`` only if ``s > 0``
  (sign gate).

Updates run on 10-trading-day ticks. The procedure is anytime-valid: it never computes or peeks
at a p-value (Ω-3, F33, NG-4). ``ledger.json`` is the single source of truth (backed up to
``ledger.json.bak`` before every tick) and regenerates ``ledger/cards.csv``.
"""

from __future__ import annotations

import datetime as dt
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bn_forensics import (
    AppLock,
    CorruptStateError,
    EventLog,
    InputError,
    atomic_write_json,
    iso_ist,
    load_json,
    sha256_json,
)
from bn_forensics.core_stats import write_csv

CARDS_CSV_FIELDS = ("card_id", "provenance", "round", "frozen_at", "data_version", "n_days_at_freeze", "state",
                    "last_fwd_days", "last_fwd_lift", "e_value", "note")
STATES = ("PROPOSED", "REGISTERED", "TESTED", "VALIDATED", "SEALED-OK", "FWD-WATCH", "LEDGER-CONFIRMED",
          "FWD-DECAYED", "RETIRED")


# =============================================================================================
# Pure math (§10.4 #4)
# =============================================================================================
def mixture_log_e(s: float, v: int, rho_grid: Sequence[float]) -> float:
    """Log of the Gaussian-mixture e-value averaged over the ``rho`` grid (log-sum-exp stable)."""
    if v <= 0:
        return 0.0
    terms = []
    for rho in rho_grid:
        a = 1.0 + v * rho * rho
        terms.append(-0.5 * math.log(a) + (rho * rho * s * s) / (2.0 * a))
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms) / len(terms))


def predictable_residuals(x: Sequence[float], b: Sequence[float], *, prior_sum: float = 0.0,
                          prior_n: int = 0) -> np.ndarray:
    """``r_d = x_d - bbar_{d-1}``: ``bbar`` is the expanding mean of ``b`` over days strictly before ``d``.

    ``x`` is NaN on non-event days (no residual); ``b`` is the slot-matched baseline series on
    every day (NaN when the day has no baseline). ``prior_sum``/``prior_n`` carry the pre-freeze
    history. Day ``d``'s residual depends only on ``x_d`` and ``b_{<d}`` (predictable).
    """
    xs = np.asarray(x, float)
    bs = np.asarray(b, float)
    out = np.full(xs.size, np.nan)
    total, count = float(prior_sum), int(prior_n)
    for i in range(xs.size):
        if count > 0 and np.isfinite(xs[i]):
            out[i] = xs[i] - total / count
        if np.isfinite(bs[i]):
            total += bs[i]
            count += 1
    return out


def e_bh(log_e: Mapping[str, float], s_sign: Mapping[str, float], M: int, alpha: float) -> dict[str, Any]:
    """Step-up e-BH with lifetime budget ``M``; confirmations pass the sign gate ``s > 0``."""
    ranked = sorted(log_e.items(), key=lambda kv: (-kv[1], kv[0]))
    k_star = 0
    for k, (_cid, le) in enumerate(ranked, start=1):
        if le >= math.log(M / (alpha * k)):
            k_star = k
    rejected = [cid for cid, _ in ranked[:k_star]]
    confirmed = [cid for cid in rejected if s_sign.get(cid, 0.0) > 0]
    return {"k_star": k_star, "rejected": rejected, "confirmed": confirmed,
            "top_threshold_log_e": math.log(M / alpha), "ranked": [cid for cid, _ in ranked]}


# =============================================================================================
# Ledger state
# =============================================================================================
@dataclass
class CardSeries:
    """Day-level series for one card: ``dates``, ``x`` (event-day outcome, NaN otherwise), ``b`` (baseline)."""

    dates: list[str]
    x: list[float]
    b: list[float]

    def validate(self) -> None:
        if not (len(self.dates) == len(self.x) == len(self.b)):
            raise InputError("CardSeries arrays must have equal length")
        if self.dates != sorted(self.dates) or len(set(self.dates)) != len(self.dates):
            raise InputError("CardSeries dates must be strictly increasing")


class Ledger:
    """``ledger/ledger.json`` manager (registration, ticks, e-BH, decay, cards.csv)."""

    def __init__(self, ledger_dir: str | Path, cfg: Mapping[str, Any], *, log: EventLog | None = None) -> None:
        self.dir = Path(ledger_dir)
        self.cfg = cfg["ledger"]
        self.path = self.dir / "ledger.json"
        self.log = log
        self.state = self._load()

    # ---- persistence ----------------------------------------------------------------------
    def _empty(self) -> dict[str, Any]:
        return {"schema_version": "1.0", "M": int(self.cfg["M"]), "alpha": float(self.cfg["alpha"]),
                "alpha_note": "", "c": float(self.cfg["c"]), "rho_grid": list(self.cfg["rho_grid"]),
                "registered_lifetime": 0, "cards": {}, "ticks": [], "refused": []}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        state = load_json(self.path)
        if not isinstance(state, dict) or state.get("schema_version") != "1.0" or "cards" not in state:
            raise CorruptStateError(f"corrupt ledger: {self.path}")
        for cid, card in state["cards"].items():
            for key in ("sign", "sd_freeze", "S", "v", "state", "base_sum", "base_n"):
                if key not in card:
                    raise CorruptStateError(f"corrupt ledger card {cid}: missing {key}")
        return state

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self.state)
        self.write_cards_csv()

    def backup(self) -> Path | None:
        if not self.path.exists():
            return None
        bak = self.path.with_suffix(".json.bak")
        shutil.copyfile(self.path, bak)
        return bak

    def lock(self) -> AppLock:
        """Exclusive app-private lock on the ledger directory (§8.3; contention exits 3)."""
        return AppLock(self.dir)

    # ---- alpha policy (§10.4 #12) ----------------------------------------------------------
    def apply_union_fpr(self, union_fpr: float, fpr_max: float = 0.05) -> bool:
        """Halve the Ledger alpha to 0.025 when the measured union (Court U Ledger) FPR > 5%."""
        if union_fpr > fpr_max:
            self.state["alpha"] = float(self.cfg["alpha_halved"])
            self.state["alpha_note"] = f"union null FPR {union_fpr:.3f} > {fpr_max:.2f}: Ledger alpha halved (§10.4 #12)"
            return True
        return False

    # ---- registration (FR-13, PR-17) --------------------------------------------------------
    def register(self, card_id: str, *, card_hash: str, sign: int, history: CardSeries, provenance: str,
                 frozen_at: str, data_version: str, round_: int = 1, target: str = "T1", horizon: int = 1,
                 note: str = "") -> str:
        """Register a frozen card with its pre-freeze history; returns the resulting state."""
        history.validate()
        cards = self.state["cards"]
        for existing_id, existing in cards.items():
            if existing["card_hash"] == card_hash:
                return f"ALREADY-REGISTERED:{existing_id}"
        if self.state["registered_lifetime"] >= int(self.state["M"]):
            self.state["refused"].append({"card_id": card_id, "reason": "lifetime budget M reached (WAIT)"})
            return "REFUSED-M"
        pre_r = predictable_residuals(history.x, history.b)
        finite = pre_r[np.isfinite(pre_r)]
        sd_freeze = float(np.std(finite, ddof=1)) if finite.size >= 2 else 0.0
        b = np.asarray(history.b, float)
        cards[card_id] = {
            "card_id": card_id, "card_hash": card_hash, "sign": int(1 if sign >= 0 else -1), "provenance": provenance,
            "round": int(round_), "frozen_at": frozen_at, "data_version": data_version, "target": target,
            "horizon": int(horizon), "freeze_date": history.dates[-1] if history.dates else "",
            "n_days_at_freeze": len(history.dates), "pre_event_days": int(finite.size), "sd_freeze": sd_freeze,
            "base_sum": float(np.nansum(b)), "base_n": int(np.isfinite(b).sum()), "S": 0.0, "v": 0,
            "log_e": 0.0, "s": 0.0, "state": "FWD-WATCH", "processed_through": history.dates[-1] if history.dates else "",
            "pending": [], "windows": [], "last_fwd_days": 0, "last_fwd_lift": None,
            "note": note + ("DEGENERATE-SERIES" if sd_freeze <= float(self.cfg["sd_degenerate"]) else ""),
        }
        self.state["registered_lifetime"] += 1
        return "FWD-WATCH"

    # ---- ticks -------------------------------------------------------------------------------
    def feed(self, card_id: str, forward: CardSeries) -> int:
        """Queue strictly post-freeze days for a card; pre-freeze or already-seen days are ignored."""
        forward.validate()
        card = self.state["cards"].get(card_id)
        if card is None:
            raise InputError(f"card {card_id} is not registered")
        cutoff = max(card["processed_through"], max((p[0] for p in card["pending"]), default=""))
        added = 0
        for d, x, b in zip(forward.dates, forward.x, forward.b, strict=True):
            if d > cutoff:
                card["pending"].append([d, None if not np.isfinite(x) else float(x), None if not np.isfinite(b) else float(b)])
                added += 1
        return added

    def _process_window(self, card: dict[str, Any], window: list[list[Any]]) -> None:
        xs = [np.nan if w[1] is None else w[1] for w in window]
        bs = [np.nan if w[2] is None else w[2] for w in window]
        r = predictable_residuals(xs, bs, prior_sum=card["base_sum"], prior_n=card["base_n"])
        fin = r[np.isfinite(r)]
        card["S"] += float(fin.sum())
        card["v"] += int(fin.size)
        b_arr = np.asarray(bs, float)
        card["base_sum"] += float(np.nansum(b_arr))
        card["base_n"] += int(np.isfinite(b_arr).sum())
        card["processed_through"] = window[-1][0]
        mean = float(fin.mean()) if fin.size else float("nan")
        se = float(fin.std(ddof=1) / math.sqrt(fin.size)) if fin.size >= 2 else float("nan")
        card["windows"].append({"through": window[-1][0], "n": int(fin.size), "mean": mean, "se": se})
        card["last_fwd_days"] = len(window)
        card["last_fwd_lift"] = card["sign"] * mean if fin.size else None
        sigma = float(self.state["c"]) * card["sd_freeze"]
        if card["sd_freeze"] <= float(self.cfg["sd_degenerate"]) or card["v"] == 0:
            card["log_e"], card["s"] = 0.0, 0.0
            if card["sd_freeze"] <= float(self.cfg["sd_degenerate"]) and "DEGENERATE-SERIES" not in card["note"]:
                card["note"] = (card["note"] + " DEGENERATE-SERIES").strip()
        else:
            card["s"] = card["sign"] * card["S"] / sigma
            card["log_e"] = mixture_log_e(card["s"], card["v"], self.state["rho_grid"])

    def _decayed(self, card: dict[str, Any]) -> bool:
        """FM-24: two consecutive windows against the frozen sign with CIs containing 0."""
        n = int(self.cfg["decay_windows"])
        wins = [w for w in card["windows"] if w["n"] >= 2 and np.isfinite(w["mean"]) and np.isfinite(w["se"])]
        if len(wins) < n:
            return False
        last = wins[-n:]
        return all(card["sign"] * w["mean"] < 0 and abs(w["mean"]) < 1.96 * w["se"] for w in last)

    def tick(self, *, today: str | None = None) -> dict[str, Any]:
        """Process every complete 10-day window, then run step-up e-BH and update lifecycle states."""
        tick_days = int(self.cfg["tick_days"])
        processed = 0
        for card in self.state["cards"].values():
            if card["state"] == "RETIRED":
                continue
            while len(card["pending"]) >= tick_days:
                window, card["pending"] = card["pending"][:tick_days], card["pending"][tick_days:]
                self._process_window(card, window)
                processed += 1
        log_e = {cid: c["log_e"] for cid, c in self.state["cards"].items() if c["state"] != "RETIRED"}
        s_sign = {cid: c["s"] for cid, c in self.state["cards"].items()}
        ebh = e_bh(log_e, s_sign, int(self.state["M"]), float(self.state["alpha"]))
        for cid, card in self.state["cards"].items():
            if card["state"] == "RETIRED":
                continue
            if cid in ebh["confirmed"]:
                card["state"] = "LEDGER-CONFIRMED"
            elif card["state"] != "LEDGER-CONFIRMED" and self._decayed(card):
                card["state"] = "FWD-DECAYED"
            elif card["state"] not in ("LEDGER-CONFIRMED", "FWD-DECAYED"):
                card["state"] = "FWD-WATCH"
        record = {"at": today or iso_ist(), "windows_processed": processed, "k_star": ebh["k_star"],
                  "confirmed": sorted(ebh["confirmed"]), "alpha": self.state["alpha"], "M": self.state["M"],
                  "digest": sha256_json({cid: [c["S"], c["v"], c["log_e"]] for cid, c in sorted(self.state["cards"].items())})}
        self.state["ticks"].append(record)
        if self.log is not None:
            self.log.emit("ledger_tick", record, stage="Ω-S7", role="AG-05")
        return record

    def retire(self, card_id: str, reason: str) -> None:
        card = self.state["cards"][card_id]
        card["state"] = "RETIRED"
        card["note"] = (card["note"] + f" RETIRED: {reason}").strip()

    # ---- reporting -----------------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        states = [c["state"] for c in self.state["cards"].values()]
        return {"fwd_watch": states.count("FWD-WATCH"), "confirmed": states.count("LEDGER-CONFIRMED"),
                "decayed": states.count("FWD-DECAYED"), "retired": states.count("RETIRED"),
                "registered_lifetime": int(self.state["registered_lifetime"]), "M": int(self.state["M"])}

    def rows(self) -> list[dict[str, Any]]:
        out = []
        for cid in sorted(self.state["cards"]):
            c = self.state["cards"][cid]
            out.append({"card_id": cid, "provenance": c["provenance"], "round": c["round"], "frozen_at": c["frozen_at"],
                        "data_version": c["data_version"], "n_days_at_freeze": c["n_days_at_freeze"], "state": c["state"],
                        "last_fwd_days": c["last_fwd_days"], "last_fwd_lift": c["last_fwd_lift"],
                        "e_value": math.exp(min(c["log_e"], 700.0)), "note": c["note"]})
        return out

    def write_cards_csv(self) -> Path:
        return write_csv(self.dir / "cards.csv", CARDS_CSV_FIELDS, self.rows())

    def next_tick_date(self, trading_days: Sequence[dt.date]) -> str:
        """Next tick = the date on which 10 more trading days will have accumulated (data calendar).

        Future trading days are unknown until data arrives, so the date is projected only from
        observed days; when it lies beyond the data it is ``PENDING-DATA``.
        """
        pend = max((len(c["pending"]) for c in self.state["cards"].values()), default=0)
        need = int(self.cfg["tick_days"]) - pend
        last = max((c["processed_through"] for c in self.state["cards"].values()), default="")
        future = [d for d in sorted(trading_days) if d.isoformat() > last]
        return future[need - 1].isoformat() if len(future) >= need > 0 else "PENDING-DATA"


# =============================================================================================
# Feature cache (cache/<data_version>/)
# =============================================================================================
def cache_path(cache_dir: str | Path, data_version: str, card_hash: str) -> Path:
    return Path(cache_dir) / data_version / f"ledger_series_{card_hash[:16]}.json"


def load_cached_series(cache_dir: str | Path, data_version: str, card_hash: str) -> CardSeries | None:
    path = cache_path(cache_dir, data_version, card_hash)
    if not path.exists():
        return None
    raw = load_json(path)
    series = CardSeries(raw["dates"], [np.nan if v is None else v for v in raw["x"]], [np.nan if v is None else v for v in raw["b"]])
    series.validate()
    return series


def store_cached_series(cache_dir: str | Path, data_version: str, card_hash: str, series: CardSeries) -> Path:
    payload = {"dates": series.dates, "x": [None if not np.isfinite(v) else float(v) for v in series.x],
               "b": [None if not np.isfinite(v) else float(v) for v in series.b]}
    return atomic_write_json(cache_path(cache_dir, data_version, card_hash), payload)


def series_from_days(all_days: Iterable[str], x_by_day: Mapping[str, float], b_by_day: Mapping[str, float]) -> CardSeries:
    """Assemble a :class:`CardSeries` over ``all_days`` (missing x -> NaN = non-event day)."""
    days = sorted(set(all_days))
    return CardSeries(days, [float(x_by_day.get(d, np.nan)) for d in days], [float(b_by_day.get(d, np.nan)) for d in days])


__all__ = [
    "CARDS_CSV_FIELDS", "STATES", "CardSeries", "Ledger", "cache_path", "e_bh", "load_cached_series",
    "mixture_log_e", "predictable_residuals", "series_from_days", "store_cached_series",
]
