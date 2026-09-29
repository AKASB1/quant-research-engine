# Implementation plan

## Research rules

The engine should make it difficult to accidentally use future information. Every dataset and feature needs a timestamp and an availability timestamp when they differ.

## Data layer

Use Parquet as the portable storage format and DuckDB for local analytical queries. Keep adapters separate from normalized schemas.

Initial tables:

- bars
- corporate actions
- instrument metadata
- factor values
- signals
- orders/fills

## Backtest engine

Implement an event loop around:

1. market event
2. strategy update
3. order generation
4. execution simulation
5. portfolio accounting
6. metric update

## Costs

Start with fixed commission + proportional spread/slippage, then add volume-aware impact.

## Evaluation

At minimum report:

- annualized return / volatility
- Sharpe and Sortino
- max drawdown
- turnover
- hit rate
- exposure and concentration
- cost contribution

## Delivery order

1. OHLCV data adapter
2. normalized Parquet store
3. factor pipeline
4. simple signal API
5. event-driven backtester
6. cost model
7. portfolio accounting
8. walk-forward runner
9. attribution report
