# Look-ahead: guards, canaries, audit

Python cannot sandbox a strategy, so the guards are layered and every planted leak must be caught by at least one of them. The canaries are strategies (and one feature, one run, one validation set-up) that leak on purpose; the audit runs every guard on every canary and on every built-in strategy and feature.

## Guards

| Guard | What it does | How it is tested |
|---|---|---|
| G1 context truncation | `DecisionContext(t)` holds read-only copies of the rows known at `t` (no `.base` to a longer array, no store or panel reference); the universe lists the instruments listed and not yet delisted at `t`; splits are applied from their ex-date. The store's as-of queries behind it (delisting and corporate-action rows known only from their own knowledge times) are check 2 | property test over random stores with delayed bars and random calls, extreme arguments (check 4) |
| G2 replay audit | replays the strategy from its first decision to each sampled instant `t` in the real store and in a copy poisoned after `t`; decisions must be bit-identical; a change to this package's module or class state during a world is also a catch (and is undone); a subject that fails on the real store is reported as `error`, not as passed | canaries C1, C1b, C1g, C1s, C2 to C5; self-tests show the purge and the registry override are needed; canary C1s tests the state snapshot |
| G3 lint | AST allow-list of imports (`__future__`, `math`, `typing`, `dataclasses`, `functools`, `collections`, `itertools`, NumPy, SciPy without `scipy.io`, the public strategy API through `from ... import` only: a plain `import quant_research_engine...` binds the package root, from which internal modules are reachable); `open`, `eval`, `exec`, `__import__`, `importlib`, NumPy file readers, reflective builtins (`getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`), and dunder attributes other than `__init__` rejected | unit tests; every built-in strategy and feature module; strategy files given to `backtest` |
| G4 availability audit | derives a feature value's availability as the latest `ts_avail` of the input rows in its lookback window and fails a feature that declares an earlier one | canary C6 |
| G5 fill timing | fills at the earliest one bar after the decision; same-bar fills only behind `unsafe_same_bar_fill`, labelled `UNSAFE` in every output; the report generator refuses such a run unless `--allow-unsafe` | canary C7, fill-timing property test (check 10) |
| G6 validation splitters | the cross-validation set-up refuses a splitter whose training labels overlap test labels for the label horizon in use, unless `allow_unpurged`; outputs are then labelled `UNPURGED`; the audit also records every such split a subject creates while it is replayed, declared or not | canary C8, an undeclared copy of it (test), purge/embargo property test (check 12), E5 |
| G7 universe | `instruments_at(t)` equals the definition (listed, not yet delisted, including later delistings); the audit flags a decision that names an instrument outside the universe | property test (check 2) |

### The replay audit in detail

For each sampled decision instant `t` (stream `audit.instants`) the audit builds a poisoned copy of the store (stream `audit.poison`): bars after `t` get prices multiplied by a random factor between 0.1 and 10 per bar and redrawn volumes; future delistings are cancelled for some instruments and added for others (so the set of final survivors differs); instruments listing after `t` may be removed and new ones are added; series rows released after `t` get redrawn values, and for every period with a vintage known at `t` later vintages are changed, removed, or appended; corporate actions announced after `t` are changed, removed, or added; liquidity rows are recomputed from the poisoned bars; the manifest names another seed and other generator parameters. Every row known at `t` keeps its value (the builder checks this).

Isolation is in-process. Before each world the store registry points every `open_store` call (with any path) at that world's store, and every module that was imported after the audit started or lives in the subjects' directories is removed from `sys.modules` (standard library, third-party packages, and this package excepted), together with the subject's own module; the strategy is then imported again and instantiated from its spec. The module globals and class attributes of this package are snapshotted before each world and compared afterwards: a change (for example a store handle parked on the `Strategy` class, canary C1s) is a catch and is undone, so it cannot carry a handle from one world to the next. The worlds' tables and panels are read-only. Decisions made before an exception are kept; two worlds differ at instant `k` if a decision at or before `k` differs or if they fail at different bars at or before `k`. A handle opened at import, in a constructor, through `open_store` with a path, memoized in a helper module (by `functools.lru_cache` or in a module global), or parked in this package therefore comes from the world being replayed or is reported. The self-tests show that with a naive audit (re-importing only the canary module) C1 and C1g pass, that without the registry override C1b passes. C1s was found by review 1, before the state snapshot existed; the test checks that the audit now catches it, and no test switches the snapshot off. A process per world would also isolate, but a spawned Windows process costs about a second and the audit replays hundreds of worlds; the in-process audit of 22 subjects at 50 instants on one `small` store takes minutes on one process.

## Canaries

| Id | Name | Leak | Guard named |
|---|---|---|---|
| C1 | `peek_next_close` | reads the next bar's close through a store handle memoized with `lru_cache` in a helper module | G2 |
| C1b | `peek_by_path` | calls `open_store` with the path in `QRE_STORE` inside `decide` and reads the next close; the registry hands every world its own store whatever the path. The E2 audit runs it without the variable; a self-test sets it to the real store's path | G2 |
| C1g | `peek_global_handle` | like C1, with the handle kept in a module global (self-test variant) | G2 |
| C1s | `package_state_handle` | reaches the store by an attribute chain from the package root and parks the handle on a class of this package (found by review 1) | G2 (and G3) |
| C2 | `full_sample_mean` | computes in its constructor each instrument's mean return over the whole sample and goes long the highest | G2 |
| C3 | `latest_vintage` | reads the last vintage of the revised `fund_x` instead of the as-of vintage | G2 |
| C4 | `survivors_only` | trades only instruments that never delist | G2 |
| C5 | `vendor_adjusted_close` | ranks by a price level adjusted by all splits, future ones included | G2 |
| C6 | `early_release_feature` | a panel feature that places `fund_x` at its period end instead of its release, and declares the period end | G4 |
| C7 | `same_bar_fill` | the built-in `signal_x_ls` run with `unsafe_same_bar_fill` | G5 |
| C8 | `overlapping_label_cv` | selects a lookback by shuffled K-fold on overlapping 5-bar labels | G6 |

The canaries live in `tests/canaries/` (not in the package; they import internals on purpose and are exempt from the lint's enforcement, though the table reports what the lint finds). `python -m quant_research_engine audit --canaries tests/canaries` runs every guard on them and on every built-in strategy and feature.

## Audit table

Full evaluation, commit `a1a0011`: 22 subjects on 10 `small` stores (seeds 4100 to 4109), 10 sampled decision instants per store and subject, 100 in all (`experiments/results/e2/audit_table.csv`). The data are synthetic and the results hold in this simulation, under these assumptions. "Guard named" is the guard each canary was written for, fixed before the evaluation. The last column counts the instants at which the replay audit (G2) saw a decision change.

| Subject | Kind | Guard named | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G2 instants caught |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 peek_next_close | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 100 / 100 |
| C1b peek_by_path | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 100 / 100 |
| C1g peek_global_handle | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 100 / 100 |
| C1s package_state_handle | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 100 / 100 |
| C2 full_sample_mean | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 99 / 100 |
| C3 latest_vintage | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 99 / 100 |
| C4 survivors_only | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 100 / 100 |
| C5 vendor_adjusted_close | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 99 / 100 |
| C6 early_release_feature | feature | G4 | passed | passed | passed | **caught** | passed | passed | passed | 0 / 100 |
| C7 same_bar_fill | run | G5 | passed | passed | passed | passed | **caught** | passed | passed | 0 / 100 |
| C8 overlapping_label_cv | cv | G6 | passed | passed | **caught** | passed | passed | **caught** | passed | 0 / 100 |
| F_low_vol feature_low_vol | feature | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| F_momentum feature_momentum | feature | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| F_reversal feature_reversal | feature | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| F_signal_x_ewma feature_signal_x_ewma | feature | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| F_volume_trend feature_volume_trend | feature | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_low_vol_ls low_vol_ls | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_momentum_ls momentum_ls | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_reversal_ls reversal_ls | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_signal_x_ls signal_x_ls | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_signal_x_voltarget signal_x_voltarget | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_volume_trend_topk volume_trend_topk | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |

- Every canary was caught by the guard named for it. No built-in strategy or feature raised an alarm in any guard.
- The lint also flags C1 to C5 and C8. Each imports a module that the allow-list forbids: `os`, `warnings`, the canaries' helper modules, `quant_research_engine.store.registry` (C1b, C2, C3, C5), or `quant_research_engine.inference.splitters` (C8). C1s also imports the package module by its plain name and reads a dunder attribute. The canaries are exempt from the lint's enforcement, but the table reports what it finds.
- C2, C3, and C5 were each missed at one of the 100 instants.

Detection probability of the replay audit with m sampled instants. Each value comes from 100 random draws of m instants out of the 100 audited (`detection.csv`). C6, C7, and C8 are left out because the replay does not catch them at any instant.

| Canary | m = 1 | m = 2 | m = 5 | m = 10 | m = 20 | m = 50 |
|---|---|---|---|---|---|---|
| C1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C1b | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C1g | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C1s | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C2 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C3 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

The committed table of the test run, `tests/data/audit_table.csv`, comes from `python -m quant_research_engine audit --canaries tests/canaries --seed 1 --instants 50`. It holds one store with 50 instants, and the test suite checks it against a fresh run. The Sharpe ratios that the leaking canaries earn on null markets, and the C7 ratio, are in [../experiments/README.md](../experiments/README.md#e2-leakage-audit).

## How to write a safe strategy

- Subclass `Strategy` (or compose a feature and a rule with `FeatureRuleStrategy`); read data only through the context's accessors; declare `history_bars` and the per-instrument series you read (`series_suffixes`).
- Keep state private and deterministic; draw randomness only from `ctx.rng(name)` (stream `strategy.<name>.<stream>`), keep the generator in your state if you want a sequence.
- Import only what the lint allows; never open files or stores.
- Declare a feature's inputs and lookback; compute panel values with windows that end at the bar; let the availability audit check your declaration.
- Run `python -m quant_research_engine audit` (and the replay audit on your strategy) before trusting a backtest.

## What the audit cannot prove

A pass is evidence, not proof. The replay audit only checks the sampled instants and only the perturbations the poison makes (prices, volumes, vintages, actions, delistings, listings, the manifest); a strategy that leaks through something the poison leaves unchanged, or only at instants it did not sample, passes. A module imported before the audit started that lives outside the subjects' directories survives the purge; the package-state snapshot compares module globals and class attributes shallowly (identities, and the items of lists, tuples, and dicts), so state hidden deeper in an object of this package, or in a third-party module, is not seen. Only `open_store` is redirected: a strategy that read a store file through `read_store(path)` would see the real store in both worlds, and only the lint, which forbids importing the store modules, stands in its way. The lint is a lint: Python offers more ways around it than any allow-list can close. The detection probabilities in the table above say how many instants the audit needed for each canary here; a different leak may need more.
