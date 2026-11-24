# PRD_DEFECTS.md — PRD V3.0.0 defects found while building

Protocol: PRD §17.2 (pause only the affected file, log, continue; never deviate silently).
Each entry: ID · FILE-ID · FR/NFR · PRD section · exact discrepancy · provisional behaviour.

## Step 0 reconciliation (directive vs PRD, PRD-internal)

**DEF-01 · FILE-004 · NFR-02 · §7.7 vs §12**
NFR-02 measures RSS with `psutil`, but the §7.7 dependency table omits `psutil`.
Provisional: `psutil>=5.9,<8.0` is declared in the `[dev]` extra (used only by tests).

**DEF-02 · FILE-022 · FR-17 · §7.4 / §17.3 vs §10.2**
§7.4 and §17.3 describe `strategy_input_pack.schema.json` as `schema_version=1.0`. §10.2 says the V3
instance carries `schema_version` `1.1` (additive `exposure_summary`, `cost_model`, `backtest_bridge`).
Provisional: the schema file accepts instance versions `1.0` and `1.1`. The reporter always emits `1.1`
with the three V3 keys present.

**DEF-03 · FILE-013 · FR-08 · §7.4 / §9.3**
§7.4 and §9.3 call the starter library "~40 bricks", but the §9.3 family table names 59 brick IDs
(69 once the listed parameters `k`, `N` and call/put variants are expanded).
Provisional: every brick named in §9.3 is implemented and declared. Tests assert `>= 40`.

**DEF-04 · FILE-043..FILE-048 · T-01, T-13, T-15 · §7.5 vs §16**
§7.5 assigns T-groups to test files, but T-01, T-13 and T-15 have no owning test module.
T-13 and T-15 also require a live LLM session, which an offline `pytest` run cannot provide.
Provisional:
- T-01 checks are split between `tests/test_quartermaster.py` (schemas, config, path traversal,
  symlinks, locks, eval/exec scan, pins) and `tests/test_chair_and_reporting.py` (CLI exit codes,
  secret scan).
- T-13 is a deterministic answerability check: every one of the 10 Appendix F questions must be
  answerable from the report and pack, and the evaluator prompt must be well-formed. The live AI score
  is reported as `PENDING OFFLINE HANDOFF`, as FR-23 allows.
- T-15 runs 25 code-side benchmark scenarios through the validator and compiler gates (missing data,
  multi-era, boundary traps, prompt injection). The LLM pass rate itself stays an offline item.

**DEF-05 · FILE-033 / FILE-035 / FILE-041 · FR-13, FR-05, FR-22 · §9 FR-13 vs Ω-3, §10.4 #6**
FR-13 requires that static grep finds zero p-value peek-and-stop code anywhere in `src/bn_forensics/`.
But the Null Lab Peeking Trap (FR-05, Ω-3, T-08) and the `omega_sim.py` P0 and P4c protocols (FR-22)
must implement p-value peeking in `src/` to measure its false-positive rate. T-08 itself scopes the
grep to `ledger.py`.
Provisional: the AST/grep ban covers every production module. The only allowed peeking constructs are
the functions named `peeking_trap_*` in `null_lab.py` and, in `omega_sim.py`, the reproductions of the
S5 historical protocols that peek by construction: `sim_p0_*`, `sim_p1_*`, `sim_p2_*`, `sim_p4c_*` and
`sim_p6_*`. Their measured false-positive rates are the Ω-10 benchmark. The production-equivalent
`sim_p4_evalue_peek` and `sim_p5_diverse_court` must stay unflagged, and T-08 asserts this.
A test asserts that they exist only there, that no production module (court, ledger, playground,
chair, reporter, cli decision paths) calls them, and that their output never feeds a verdict.

**DEF-06 · FILE-024 · FR-19, T-16 · §10.3 exit 5 vs §16 T-16**
In T-16 the selftest world deliberately lacks an option-OI column. By the §10.3 rule
(`COMPLETE_WITH_GAPS` when a data family is missing) that alone would give exit `5`, yet T-16 requires
`bnf selftest` to exit `0`.
Provisional: the embedded pipeline run records its terminal state `COMPLETE_WITH_GAPS` in
`run_state.json`. The `selftest` command's own exit code is the verdict of its acceptance assertions:
`0` if all pass, `4` if any invariant fails.

**DEF-07 · FILE-023 (+ FILE-025..041) · NFR-04, NFR-05 · §7.2 / §7.5**
The 49-file blueprint has no shared-utility module. Atomic writes (NFR-04), symlink and traversal
guards (NFR-05), app-private locks (§8.3), SHA-256 helpers and the IST clock are needed by modules whose
§7.5 import edges do not connect (for example `ledger.py` imports only FILE-012 and FILE-025).
Provisional: these stdlib-only primitives live in FILE-023 `__init__.py` ("core constants"). It has
no third-party imports and no `core_stats` re-export, so the F1.2 and F5 isolation of FILE-034 and
FILE-036 still holds. The AST isolation test also inspects `__init__.py`.

**DEF-08 · FILE-017 / FILE-021 / FILE-040 · FR-18 · §10.2 / §10.5**
FR-18 requires schema validation of proposals, Idea Ledger records and `next_actions.yaml`, but none of
them has its own schema file among the six.
Provisional:
- proposals and the Idea Ledger JSONL record are `$defs` inside FILE-017, which is AG-01's output schema
  (§6.4);
- `next_actions` is a `$defs` entry inside FILE-021, as §7.4 states;
- digests (`<= 8` lines, ID-linked) are validated by a code-side rule in FILE-040.

**DEF-09 · FILE-020 · NFR-09, §8.3 · §10.2 run_state example**
NFR-09 and the resume protocol (§8.3) need `run_id`, terminal `state`, deliverable SHA-256 digests,
split and resume information stored in `run_state.json`. The §10.2 example lists none of these keys.
Provisional: they are optional properties of the `1.0` schema. Every key in the PRD example stays
required.

**DEF-10 · FILE-041 · FR-22, Ω-10 · §10.4 #6, §2 "Missing Referenced Code Files"**
S5's `never_stop_sim.py` and `court_variants.py` were never attached. The WEAK, MOD and STRONG effect
sizes and the exact P1, P2 and P6 protocols are therefore undefined.
Provisional: the protocols are reconstructed from §10.4 #6 and §22 Appendix D. Hard assertions cover
only what FR-22 states numerically (P5 NULL inside `(1.1%, 4.7%)`, P0 NULL `>= 99%`), plus the
qualitative ordering. Every other benchmark is reported with its Wilson CI and an `overlap` flag, and
is never tuned to match.

**DEF-11 · FILE-032 · FR-11 · §15 FM-37 vs §1 X09 / §8.3**
FM-37 states `N_min = max(60, days for MDE_court <= MDE_target)`. X09 and §8.3 replaced this with
`N_min = 60` plus the power pre-check.
Provisional: `N_min = 60` plus the §10.4 #9 power pre-check (the V3 rule).

**DEF-12 · FILE-016 · FR-02, R29 · §20 D-4**
Era-A's weekly-expiry weekday change ("Era A Sep 2023 shift") appears only as an open item in D-4.
Provisional: `era_calendar.yaml` records Era-A's expiry rule with `verified: false` and
`USER TO CONFIRM`. Expiry dates always come from the data (`contracts.csv`, `expiry_date`). The
calendar rule is used only for cross-checking, and a disagreement is logged.

**DEF-13 · FILE-043 · §16.2 X26**
Six V3.4 Appendix Y.1 IDs remain unmapped because S4 is unavailable, as the PRD itself states
("66 mapped + 6 unmapped"). Provisional: reported as is; no IDs are invented.

**DEF-14 · FILE-001 / FILE-004 · L1 hygiene (pre-existing repository state)**
The repository arrived with `Readme (4).md` (not in the §7.4 inventory) and a v2.2.0
`pyproject.toml` whose pins (`python <3.13`, `pandas <3.0`, `pyarrow <27`, `pytest <9`) contradict
§7.7.
Provisional: the content of `Readme (4).md` is folded into FILE-001 `README.md`, and the file is
removed from tracking (`git rm`), so `git ls-files` equals the 49-file inventory. FILE-004 is rewritten
to the §7.7 pins.

**DEF-15 · ST-1 exit gate · §17.1**
The ST-1 gate (`python3 -c 'import bn_forensics ...'`) only works once the src-layout package is
installed, which the PRD does not mention. Provisional: an editable install (`pip install -e .`) runs
before the gate.

**DEF-16 · FILE-017 · FR-08 · §10.2 Idea Card v2**
The Idea Card v2 example shows a `RULE` card only and defines no field for a `MODEL` card's model
family or features.
Provisional: no new key is added. A MODEL card writes its model in the existing `event` string with
the DSL form `MODEL(<logistic|ridge|gbdt>, <brick>, ...)`. The compiler validates that grammar.

**DEF-17 · FILE-037 · FR-16, T-10 · §9 FR-16 vs §16 T-10**
FR-16 says a skipped or unremoved mismatch becomes `MODE: INVALID`. T-10 says a persistent mismatch
after 2 fix rounds halts with exit `2`.
Provisional:
- a mismatch that is demoted to `D UNVERIFIED` and still unresolved after 2 rounds gives `PARTIAL`
  (exit `2`), as T-10 says;
- a mismatch that is neither demoted nor removed gives `MODE: INVALID` (exit `4`), as FR-16 says.

## ST-2

**DEF-18 · FILE-026 · FR-01, Q2 · §10.1 / §10.1b**
Q2 requires "the per-table list for derived tables, §10.1", but §10.1 gives no per-table column lists
for `bars_3m`, `bars_15m`, `chain_1m`, `chain_metrics_1m`, `iv_greeks_1m`, `iv_daily_DERIVED` or
`daily_metrics_DERIVED`. It only mentions `o_DERIVED…`, `available_from_ts`, `iv_status`, `version`
and `check_time`.
Provisional: `quartermaster.DERIVED_REQUIRED` holds the minimal identity and availability columns
observed in S7. A missing derived column makes that family `N/A — DATA MISSING` plus a DRT. It is
never a reject, because only `spot_1m` is mandatory.

**DEF-19 · FILE-015 / FILE-013 · FR-07 · §9.2 vs §9.3**
AP28 (MACD), AP29 (EMA 9/21 cross), AP30 (%b) and AP31 (Supertrend) need indicator bricks that the
§9.3 brick table does not list. AP18 (sweep-and-reject) also needs the bar extreme against the
previous-day level.
Provisional:
- the AP-support bricks `macd_hist`, `macd_line`, `ema_9_21`, `bb_pctb` and `supertrend_dir` are
  added, marked `ap_support`;
- `dist_pdh_atr` and `dist_pdl_atr` gain a `_x` (bar-extreme) variant.

AP03 (signed GEX) and AP04 (trade-side flow) reference data that no S7 table provides. They resolve
to `N/A — DATA MISSING` plus a DRT through `data_needed`, and no brick is invented.

## ST-4

**DEF-20 · FILE-030 / FILE-045 · FR-09, FR-18, FR-22 · §7.5 / §7.4**
1. §7.5 lists FILE-030's imports as FILE-012/025/027/029/031. FR-18 still requires every Idea Ledger
   record to be schema-validated, so `playground.py` also imports FILE-040 (`schemas_validator`).
   FILE-029 already depends on FILE-040 transitively, so this adds no new layering.
2. §7.4 describes FILE-045 as also testing "sim (Ω-10)". However, FILE-041 (`omega_sim.py`) is built
   in ST-5 and is not among FILE-045's imports in §7.5.
   Provisional: Ω-10 is tested in FILE-047, whose import list includes FILE-041. FILE-045 covers Ω-4,
   Ω-5, Ω-8, Ω-11 and the Court null FPR at Court level.
3. The Court `Candidate` type is shared by FILE-030 and FILE-032, but neither file may import the
   other. Provisional: it lives in FILE-031, which both are allowed to import.
