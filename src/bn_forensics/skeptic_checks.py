"""Isolated Skeptic pass — 7 non-vacuous post-Court veto re-tests and Ω grades (FILE-034, FR-12).

Strictly isolated (F1.2, NFR-10): this script imports nothing from ``core_stats`` or any data
loader. It reads ONLY ``tables/*.csv`` (the 12-column ``event_ledger.csv``, ``court_verdicts.csv``,
``skeptic_inputs.csv``) and ``test_plan.md``, and writes ``tables/skeptic_results.csv``,
``tables/skeptic_distrust.csv`` and the Skeptic/grade columns of ``findings_registry.csv``.
Verdict label: ``SELF-REVIEW`` (scripted re-tests passed, never proof).

Re-tests (S4 §8): 1 placebo +/-30m & next-bar-open, 2 causal vol terciles, 3 ex-expiry,
4 ex-outlier days, 5 raw vs clean, 6 non-overlapping events, 7 week-block bootstrap.
Tiers: A (0 fails, >= 4 ran, no UNVERIFIED-CAUSAL / MECHANISM-REFUTED), B, C.
Ω mapping (PR-10, R28): Ω-A needs COURT-CONFIRMED + ROBUST + tier A + ERA-CONSISTENT/Era-B-only +
(VIRGIN-COURT or LEDGER-CONFIRMED); EXPOSED-COURT caps at Ω-B; Ω-C = LEDGER-CONFIRMED without
Court; COURT-UNDERPOWERED never receives any Ω grade.

Run: ``python3 -m bn_forensics.skeptic_checks --run-dir runs/<run_id>``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

RESULT_FIELDS = ("card_id", "registry_lift", "n_events", "event_days", "ran", "fails", "not_run", "vacuous", "tests",
                 "tier", "skeptic_verdict", "omega_grade", "court_verdict", "exposure", "robust", "era_tag",
                 "ledger_state", "mechanism_status", "unverified_causal", "reason", "label")
DISTRUST_FIELDS = ("rank", "card_id", "omega_grade", "registry_lift", "why_distrusted")
MIN_N = 30
KEEP_FRAC = 0.50


# =============================================================================================
# Minimal CSV / text IO (no package imports)
# =============================================================================================
def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_atomic(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _fmt(row.get(k)) for k in fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        txt = f"{v:.10g}"
        return "0" if txt == "-0" else txt
    return str(v)


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_test_plan(path: Path) -> dict[str, str]:
    """``key: value`` lines of ``test_plan.md`` (bootstrap B, seed, frozen parameters)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lstrip("-* ").strip()
        if ":" in stripped and not stripped.startswith("#"):
            key, val = stripped.split(":", 1)
            out[key.strip().strip("`").lower()] = val.strip().strip("`")
    return out


# =============================================================================================
# The 7 re-tests
# =============================================================================================
def _keeps(sub: float, registry: float) -> bool:
    """Same sign AND >= 50% of the Registry lift kept."""
    return math.isfinite(sub) and registry != 0 and np.sign(sub) == np.sign(registry) and abs(sub) >= KEEP_FRAC * abs(registry)


def _mean(vals: list[float]) -> float:
    v = [x for x in vals if math.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def week_id(date: str) -> int:
    d = dt.date.fromisoformat(date)
    iso = d.isocalendar()
    return iso[0] * 100 + iso[1]


def week_block_ci(rows: list[dict[str, Any]], B: int, seed: int) -> tuple[float, float, int]:
    """95% CI of the event-weighted mean residual resampling whole calendar weeks (seed 42)."""
    weeks: dict[int, list[float]] = {}
    for r in rows:
        weeks.setdefault(week_id(r["date"]), []).append(r["resid"])
    keys = sorted(weeks)
    sums = np.array([sum(weeks[k]) for k in keys])
    cnts = np.array([len(weeks[k]) for k in keys], dtype=float)
    if len(keys) < 2:
        return float("nan"), float("nan"), len(keys)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    boot = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi), len(keys)


def run_retests(rows: list[dict[str, str]], *, B: int = 5000, seed: int = 42) -> dict[str, Any]:
    """Run the 7 re-tests on one card's event-ledger rows."""
    ev = []
    for r in rows:
        clean, base = _f(r["clean_ret"]), _f(r["baseline_ret"])
        if not (math.isfinite(clean) and math.isfinite(base)):
            continue
        ev.append({"date": r["date"], "clean": clean, "raw": _f(r["raw_ret"]), "base": base, "resid": clean - base,
                   "next_open": _f(r["next_open_ret"]), "placebo": _f(r["placebo_ret"]),
                   "tercile": int(_f(r["vol_tercile"])) if math.isfinite(_f(r["vol_tercile"])) else 0,
                   "expiry": r.get("is_expiry", ""), "outlier": r.get("is_outlier_day", "0"), "overlap": r.get("overlap_flag", "0")})
    n = len(ev)
    registry = _mean([e["resid"] for e in ev])
    results: dict[int, tuple[str, str]] = {}
    # 1 placebo +/-30m and next-bar-open
    placebo = [e["placebo"] for e in ev if math.isfinite(e["placebo"])]
    nxt = [e["next_open"] - e["base"] for e in ev if math.isfinite(e["next_open"])]
    if len(placebo) < MIN_N or len(nxt) < MIN_N:
        results[1] = ("NOT RUN", f"N < {MIN_N} in cell")
    else:
        pl = _mean(placebo)
        ok_placebo = abs(pl) < 0.5 * abs(registry) or (np.sign(pl) != np.sign(registry))
        ok_next = _keeps(_mean(nxt), registry)
        results[1] = ("PASS" if ok_placebo and ok_next else "FAIL",
                      f"placebo lift {pl:.4f}, next-open lift {_mean(nxt):.4f} vs registry {registry:.4f}")
    # 2 vol terciles
    cells = {t: [e["resid"] for e in ev if e["tercile"] == t] for t in (1, 2, 3)}
    runnable = {t: v for t, v in cells.items() if len(v) >= MIN_N}
    if not runnable:
        results[2] = ("NOT RUN", f"all terciles N < {MIN_N}")
    else:
        ok = all(_keeps(_mean(v), registry) for v in runnable.values())
        results[2] = ("PASS" if ok else "FAIL", "; ".join(f"T{t}: {_mean(v):.4f} (N={len(v)})" for t, v in sorted(runnable.items())))
    # 3 ex-expiry
    has_cal = any(str(e["expiry"]).strip() != "" for e in ev)
    sub = [e["resid"] for e in ev if str(e["expiry"]).strip() in ("0", "0.0")]
    if not has_cal or len(sub) < MIN_N:
        results[3] = ("NOT RUN", "no expiry column/calendar" if not has_cal else f"N < {MIN_N}")
    else:
        results[3] = ("PASS" if _keeps(_mean(sub), registry) else "FAIL", f"ex-expiry lift {_mean(sub):.4f} (N={len(sub)})")
    # 4 ex-outlier days
    sub = [e["resid"] for e in ev if str(e["outlier"]).strip() in ("0", "0.0")]
    if len(sub) < MIN_N:
        results[4] = ("NOT RUN", f"N < {MIN_N}")
    else:
        results[4] = ("PASS" if _keeps(_mean(sub), registry) else "FAIL", f"ex-outlier lift {_mean(sub):.4f} (N={len(sub)})")
    # 5 raw vs clean
    changed = [e for e in ev if math.isfinite(e["raw"]) and abs(e["raw"] - e["clean"]) > 1e-12]
    if not changed:
        results[5] = ("VACUOUS", "0 Tier-1 rows changed inside the event set (raw == clean)")
    else:
        raw_l = _mean([e["raw"] - e["base"] for e in ev if math.isfinite(e["raw"])])
        results[5] = ("PASS" if _keeps(raw_l, registry) else "FAIL", f"raw lift {raw_l:.4f}, {len(changed)} events changed")
    # 6 non-overlapping
    removed = sum(1 for e in ev if str(e["overlap"]).strip() in ("1", "1.0"))
    sub = [e["resid"] for e in ev if str(e["overlap"]).strip() not in ("1", "1.0")]
    if removed == 0:
        results[6] = ("VACUOUS", "0 overlapping events removed")
    elif len(sub) < MIN_N:
        results[6] = ("NOT RUN", f"N < {MIN_N} after removing overlaps")
    else:
        results[6] = ("PASS" if _keeps(_mean(sub), registry) else "FAIL", f"non-overlap lift {_mean(sub):.4f}")
    # 7 week-block bootstrap
    lo, hi, n_weeks = week_block_ci(ev, B, seed) if ev else (float("nan"), float("nan"), 0)
    if n_weeks < 8:
        results[7] = ("NOT RUN", f"{n_weeks} distinct event weeks < 8")
    else:
        excludes = (lo > 0 or hi < 0) and np.sign((lo + hi) / 2) == np.sign(registry)
        results[7] = ("PASS" if excludes else "FAIL", f"week-block 95% CI [{lo:.4f}, {hi:.4f}] over {n_weeks} weeks")
    ran = sum(1 for s, _ in results.values() if s in ("PASS", "FAIL"))
    fails = sum(1 for s, _ in results.values() if s == "FAIL")
    return {"registry_lift": registry, "n_events": n, "event_days": len({e["date"] for e in ev}), "ran": ran,
            "fails": fails, "not_run": sum(1 for s, _ in results.values() if s == "NOT RUN"),
            "vacuous": sum(1 for s, _ in results.values() if s == "VACUOUS"), "results": results}


def skeptic_tier(fails: int, ran: int, *, unverified_causal: bool, mechanism_refuted: bool) -> str:
    """V5 internal Skeptic tier A / B / C (S4 §8)."""
    if fails >= 2 or ran < 3:
        return "C"
    if fails == 0 and ran >= 4:
        return "B" if (unverified_causal or mechanism_refuted) else "A"
    return "B"


def omega_grade(*, court_verdict: str, exposure: str, robust: str, tier: str, ran: int, fails: int, era_tag: str,
                ledger_state: str, unverified_causal: bool, mechanism_refuted: bool, policy: str = "cap") -> tuple[str, str, str]:
    """Ω grade, Skeptic verdict and reason (PR-10, R28, X06)."""
    ledger_ok = ledger_state == "LEDGER-CONFIRMED"
    if court_verdict == "COURT-UNDERPOWERED":
        return "NONE", "NOT RUN", "COURT-UNDERPOWERED: hypothesis_backlog + FWD-WATCH only, never an Ω grade (X06)"
    if court_verdict != "COURT-CONFIRMED":
        if not court_verdict and ledger_ok:
            return "Ω-C", "LEDGER", "LEDGER-CONFIRMED (forward, sign-gated) without Court"
        return "NONE", "NOT RUN", f"{court_verdict or 'no Court verdict'}: no Ω grade"
    flags = fails + (0 if robust == "ROBUST" else 1) + (1 if era_tag == "ERA-SPECIFIC" else 0)
    if tier == "C" or flags >= 2:
        return "NONE", "KILLED", f"tier {tier}, {flags} veto/era flags (fails {fails}, ran {ran}, robust {robust}, era {era_tag})"
    clean_a = flags == 0 and tier == "A" and not unverified_causal and not mechanism_refuted
    if clean_a:
        if exposure == "VIRGIN-COURT" or ledger_ok or policy == "ignore":
            return "Ω-A", "SURVIVED", "COURT-CONFIRMED + ROBUST + 0 Skeptic fails (>= 4 ran) + era-consistent + " + (
                "VIRGIN-COURT" if exposure == "VIRGIN-COURT" else ("LEDGER-CONFIRMED" if ledger_ok else "exposure policy ignore"))
        return "Ω-B", "DOWNGRADED", "otherwise Ω-A but EXPOSED-COURT: capped at Ω-B until virgin/Ledger confirmation (R28, D-12 cap)"
    why = []
    if flags == 1:
        why.append("1 veto or era flag")
    if tier == "B":
        why.append("Skeptic tier B")
    if unverified_causal:
        why.append("UNVERIFIED-CAUSAL")
    if mechanism_refuted:
        why.append("MECHANISM-REFUTED")
    return "Ω-B", "DOWNGRADED", "; ".join(why) or "tier B"


def _bool(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def run(run_dir: str | Path) -> dict[str, Any]:
    """Run the Skeptic pass for every Court verdict and Ledger-only card in ``run_dir``."""
    root = Path(run_dir)
    tables = root / "tables"
    plan = read_test_plan(root / "test_plan.md")
    B = int(plan.get("bootstrap_b", "5000"))
    seed = int(plan.get("seed", "42"))
    policy = plan.get("exposure_policy", "cap")
    ledger_rows = read_csv(tables / "event_ledger.csv")
    verdicts = {r["card_id"]: r for r in read_csv(tables / "court_verdicts.csv")}
    inputs = {r["card_id"]: r for r in read_csv(tables / "skeptic_inputs.csv")}
    by_card: dict[str, list[dict[str, str]]] = {}
    for r in ledger_rows:
        by_card.setdefault(r["test_id"], []).append(r)
    out = []
    for cid in sorted(set(verdicts) | set(inputs)):
        v = verdicts.get(cid, {})
        inp = inputs.get(cid, {})
        rt = run_retests(by_card.get(cid, []), B=B, seed=seed)
        unverified = _bool(inp.get("unverified_causal", "false"))
        mech_ref = inp.get("mechanism_status", "") == "MECHANISM-REFUTED"
        tier = skeptic_tier(rt["fails"], rt["ran"], unverified_causal=unverified, mechanism_refuted=mech_ref)
        grade, sk_verdict, reason = omega_grade(
            court_verdict=v.get("verdict", ""), exposure=v.get("exposure", ""), robust=v.get("robust", ""), tier=tier,
            ran=rt["ran"], fails=rt["fails"], era_tag=inp.get("era_tag", "ERA-B-ONLY"),
            ledger_state=inp.get("ledger_state", ""), unverified_causal=unverified, mechanism_refuted=mech_ref, policy=policy)
        if rt["ran"] < 3 and v.get("verdict") == "COURT-CONFIRMED":
            reason += "; UNTESTED-SKEPTIC (< 3 non-vacuous re-tests ran)"
        out.append({"card_id": cid, "registry_lift": rt["registry_lift"], "n_events": rt["n_events"], "event_days": rt["event_days"],
                    "ran": rt["ran"], "fails": rt["fails"], "not_run": rt["not_run"], "vacuous": rt["vacuous"],
                    "tests": "; ".join(f"{k}:{s} ({why})" for k, (s, why) in sorted(rt["results"].items())),
                    "tier": tier, "skeptic_verdict": sk_verdict, "omega_grade": grade,
                    "court_verdict": v.get("verdict", ""), "exposure": v.get("exposure", ""), "robust": v.get("robust", ""),
                    "era_tag": inp.get("era_tag", "ERA-B-ONLY"), "ledger_state": inp.get("ledger_state", ""),
                    "mechanism_status": inp.get("mechanism_status", ""), "unverified_causal": unverified,
                    "reason": reason, "label": "SELF-REVIEW"})
    write_csv_atomic(tables / "skeptic_results.csv", RESULT_FIELDS, out)
    distrust = distrusted(out)
    write_csv_atomic(tables / "skeptic_distrust.csv", DISTRUST_FIELDS, distrust)
    registry = root / "findings_registry.csv"
    if registry.exists():
        update_registry(registry, {r["card_id"]: r for r in out})
    return {"cards": len(out), "graded": {g: sum(1 for r in out if r["omega_grade"] == g) for g in ("Ω-A", "Ω-B", "Ω-C", "NONE")},
            "distrust": [d["card_id"] for d in distrust]}


def distrusted(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The 3 most believable surviving findings still distrusted, and why (S4 §8)."""
    survivors = [r for r in rows if r["omega_grade"] in ("Ω-A", "Ω-B", "Ω-C")]
    survivors.sort(key=lambda r: (-abs(_f(r["registry_lift"])) if math.isfinite(_f(r["registry_lift"])) else 0.0, r["card_id"]))
    out = []
    for i, r in enumerate(survivors[:3], start=1):
        reasons = ["SELF-REVIEW: scripted re-tests passed, never proof"]
        if r["exposure"] == "EXPOSED-COURT":
            reasons.append("Court days were seen by earlier research (R28)")
        if int(r["ran"]) < 7:
            reasons.append(f"only {r['ran']} of 7 re-tests were non-vacuous")
        if r["era_tag"] != "ERA-CONSISTENT":
            reasons.append(f"era coverage {r['era_tag']}")
        if r["ledger_state"] != "LEDGER-CONFIRMED":
            reasons.append("not yet confirmed on post-freeze forward days")
        out.append({"rank": i, "card_id": r["card_id"], "omega_grade": r["omega_grade"], "registry_lift": r["registry_lift"],
                    "why_distrusted": "; ".join(reasons)})
    return out


def update_registry(path: Path, results: dict[str, dict[str, Any]]) -> None:
    """Write Skeptic verdicts and final grades into ``findings_registry.csv`` (37 columns kept).

    Registry ``ID`` values are ``<card_id>@<target>h<horizon>``; rows are matched on the card id.
    """
    rows = read_csv(path)
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows:
        res = results.get(str(row.get("ID", "")).split("@")[0])
        if res is None:
            continue
        row["Skeptic verdict"] = f"{res['skeptic_verdict']} (SELF-REVIEW; ran {res['ran']}, fails {res['fails']})"
        if res["omega_grade"] != "NONE":
            row["Evidence grade"] = res["omega_grade"]
    write_csv_atomic(path, fields, rows)


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 3 (§10.3 input error), never argparse's default 2."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        raise SystemExit(3)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(prog="python -m bn_forensics.skeptic_checks", description="Isolated Skeptic pass (FILE-034)")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if not Path(args.run_dir).is_dir():
        sys.stderr.write(f"run dir not found: {args.run_dir}\n")
        return 3
    summary = run(args.run_dir)
    sys.stdout.write(f"[Ω-S6] SKEPTIC (SELF-REVIEW) cards={summary['cards']} grades={summary['graded']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
