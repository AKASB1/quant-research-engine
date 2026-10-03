# Implementation

## Research rules, as implemented

The engine makes it hard to use future information by accident. The rules are enforced by guards rather than by convention ([docs/lookahead.md](docs/lookahead.md)):

- Every time-stamped row carries `ts_event` and `ts_avail`; a row is known at `t` iff `ts_avail <= t`. Every store query requires a knowledge instant; there is no default.
- A strategy receives only a `DecisionContext`: read-only copies of the rows known at the decision instant, adjusted as of that instant, without a reference to the store (G1).
- Every built-in strategy and feature, and every canary, goes through the replay audit (G2): decisions must be bit-identical when everything after the decision instant is perturbed.
- Strategy and feature modules pass an import allow-list lint (G3); features declare their inputs and lookback and their declared availability is audited against the inputs they read (G4).
- Fills happen at the earliest one bar after the decision; same-bar fills exist only behind an explicit unsafe flag and are labelled `UNSAFE` everywhere (G5).
- Cross-validation refuses splitters whose training labels overlap test labels (G6); the universe includes instruments that later delist (G7).
- Results are reproducible from one command with fixed seeds and committed configurations: byte for byte on one platform with the same package versions, numerically across platforms; every experiment's results have a manifest (commit, configuration hash, seeds, versions, hardware, workers, threads, machine load); the French illustration, a script, has a short one.

## Data layer

Parquet as storage (zstd, instants as `timestamp[us, tz=UTC]`), DuckDB views for SQL and the `ASOF JOIN`, PyArrow for I/O, NumPy for computation. Normalized schemas are those of the shared contract (version 1). Adapters (the synthetic generator, the local-CSV adapter) are separate from the schemas and write through the same validators.

## Backtest engines

`EventEngine` applies, per bar and instrument: splits, dividends, eligible fills, delisting exits, accruals, marks; decisions happen at bar closes and create orders as differences between targets and committed positions. `vector.run_target_shares` is an independent array implementation over target-share schedules. Both share only the cost function and the quantity rounding; they agree on fills exactly and on equity to 1e-12 where neither the participation cap nor a leverage limit binds.

## Costs

Cost model v1 of the contract: half-spread, square-root (or linear) impact from decision-time volatility and average volume, commission with a minimum and a per-share part, borrow on shorts, financing on negative cash, interest on positive cash, a participation cap on fills.

## Evaluation

Metrics and inference of the contract (section 6) plus five experiments on synthetic markets where the truth is known: E1 engine and accounting audit, E2 leakage audit, E3 calibration of the inference tools, E4 signal recovery, costs, and lag, E5 validation schemes ([experiments/README.md](experiments/README.md)). The experiments evaluate the engine's guarantees, not strategies; gross panels (costs zero) are the calibration, net panels are reported separately and labelled.

## Delivery order

- [x] OHLCV data adapter (local CSV, offline) and synthetic generator
- [x] Normalized Parquet store with availability timestamps, as-of queries, point-in-time adjustment
- [x] Event-driven backtester
- [x] Cost model
- [x] Portfolio accounting with the identity asserted on every bar and an independent validator
- [x] Walk-forward and purged splitters, look-ahead guards, canaries, replay audit
- [x] Simple signal API and a minimal feature layer (five features, three rules, six strategies)
- [x] Export of returns, signals, forecasts, and liquidity inputs for the decision layer (export v1)
- [x] Attribution report (by instrument and by long and short book)
- [x] Tier 2, item 1: CPCV comparison of validation schemes (E5b, results from commit `6892cb1`)
- [x] Tier 2, item 2: real-data adapters: Kenneth French (illustration from commit `0ea0e9b`), Binance (offline fixture only; download needs the owner's acceptance of its terms); SEC skipped (`QRE_SEC_USER_AGENT` not set)
- [x] Tier 2, item 3: capacity and impact (results from commit `df91f7c`)
- [x] Tier 2, item 4: effective number of trials and multiple testing (results from commit `d26a863`)
- [ ] Tier 2, items 5 to 8: factor exposure attribution, intraday bars and latency, nested validation, exports at three sizes (not started; README TODO)

## Checkpoint

Tier 1 is complete. The checks pass, the fixes for review 1's nine findings went in before the freeze, and the full evaluation (E1 to E5) ran once on commit `a1a0011`, with its results committed in `fbd3232`. Tier 2 items 1 to 4 are done, each with the commit that produced its result; items 5 to 8 are not started. What is not done is in the README's TODO section.

## Cross-project contracts

This repository follows quant-contracts version 1 ([docs/contracts.md](docs/contracts.md)). The hand-off to `dynamic-trading-engine` is the file export of section 8 (returns, signals, forecasts, liquidity inputs, with the instruments, bars, and corporate actions); the decision layer needs no code from this repository and estimates its own risk models. The definitions shared with it (time and randomness conventions, schemas, knowledge time, cost model, accounting identity, metrics) are those of the shared contract; each project implements them independently.
