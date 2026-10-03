# Architecture

Research and simulation software: everything here runs on synthetic data (or a user's local CSV file), and nothing in it trades or connects to a broker.

![framework](figures/framework.png)

## Modules

| Package | Responsibility |
|---|---|
| `data` | The version-1 CSV schemas, loaders and validators (line-numbered errors), deterministic writers, dataset manifests, configuration hashes |
| `store` | Calendars, the Parquet store with `store.json`, DuckDB views and the `ASOF JOIN`, `asof_vintage` (the one definition of as-of semantics), knowledge-bounded queries, point-in-time split and total-return adjustment, the liquidity function, the local-CSV adapter, the store registry (`open_store`) |
| `synth` | The synthetic market generator and its configuration models |
| `context` | `DecisionContext` (the only data path to a strategy), the harness-side `ContextBuilder`, decision types, the `Strategy` base class |
| `features`, `signals`, `portfolio`, `strategies` | Declared past-only features (context and panel implementations), cross-sectional transforms, portfolio rules, built-in strategies (one feature + one rule) |
| `costs` | Cost model v1: one function for both engines, accruals |
| `engine` | `EventEngine`, quantity rounding, log writers |
| `vector` | `run_target_shares`, the independent vectorized reference engine |
| `backtest` | Decision schedule and the loop that connects a strategy, its context, and the event engine |
| `validation` | The independent accounting validator (reads the logs), random engine scenarios, the engine comparison |
| `attribution` | Per-instrument and long/short-book attribution from the engine's per-bar records, checked against the totals |
| `guards` | Poison builder, replay audit (G2), lint (G3), availability audit (G4), canary loader, the audit table |
| `metrics`, `inference` | Metrics of the contract; PSR, SR0, DSR, PBO (CSCV), stationary bootstrap, splitters, the trial registry |
| `experiments` | The runner (serial or a `spawn` process pool), E1 to E5, manifests, statistics |
| `reports`, `export`, `cli` | Report generator, export v1 and its validator, the command line |

## The knowledge-time rule

A row is known at instant `t` if and only if its `ts_avail <= t`. Every store query takes a required `knowledge` argument; the context hands a strategy only copies of rows known at the decision instant. Instruments are known from `ts_list`, their delisting fields from `ts_delist`; corporate actions from their announcement. The universe at `t` includes instruments that delist later. Adjusted prices are computed as of `t` (a split announced but not yet effective changes nothing; a dividend enters a total-return series when its ex-bar is known).

## The flow of a backtest

1. The store is opened (or generated); the harness builds its calendar-aligned panel.
2. For each session of the calendar the event engine processes the bar: splits, dividends, eligible fills, delisting exits, accruals, marks; the accounting identity is asserted.
3. At a scheduled decision bar the `ContextBuilder` copies the rows known at the bar's close (and the portfolio, and open orders) into a `DecisionContext`; the strategy returns target weights, target shares, or orders.
4. The engine sizes the targets (equity or fixed capital, rounded to lots and the quantity grid) and creates the difference between the target and the committed position as an order. Orders fill at the earliest one bar later, at the next open (or the next close), with costs from the decision-time liquidity.
5. The run writes the logs of the contract (orders, fills, positions, equity), the attribution, and `run.json`; the independent validator re-derives the accounting from the logs, bars, and corporate actions.

## Two engines

`EventEngine` walks the sessions bar by bar and instrument by instrument, applying the order of operations of the contract. `vector.run_target_shares` takes target-share schedules and computes fills, positions, cash, and PnL with array operations (orders are differences of consecutive targets, rescaled by split ratios between decision and fill; positions are the latest filled target). The two share only the cost function and the quantity rounding. In the regime where neither the participation cap nor a leverage limit binds they agree on every fill exactly (quantities on a binary grid) and on equity to 1e-12 relative (E1, check 7). Experiments that run thousands of backtests (E3, E4) use the vectorized engine; context-driven strategies, the audit, and the examples use the event engine.

## The audit

`invariance_audit` replays a strategy from its first decision to each sampled instant in the real store and in a store poisoned after that instant, in-process: before each world the registry points `open_store` (any path) at that world's store, and every module outside the standard library, third-party packages, and this package (plus the subject's own module) is removed from `sys.modules`. Decisions must be bit-identical. See [lookahead.md](lookahead.md).
