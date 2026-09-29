# Quant Research & Backtesting Engine

A research environment for testing market signals and portfolio rules with reproducible data, transaction costs, walk-forward evaluation, and event-driven backtesting.

**Status:** implementation scaffold.

## Scope

- market-data ingestion and normalization
- point-in-time feature / factor computation
- signal generation
- walk-forward train/test splits
- event-driven backtesting
- transaction cost and slippage models
- portfolio accounting
- performance and risk attribution
- experiment metadata and reproducible configs

## Proposed stack

Python 3.12 · Polars · PyArrow · DuckDB · NumPy · Pydantic · Matplotlib

## Research flow

```text
Raw Data
   │
   ▼
Normalized Store
   │
   ▼
Features / Factors
   │
   ▼
Signals
   │
   ▼
Portfolio Rules
   │
   ▼
Backtest + Costs
   │
   ▼
Attribution / Reports
```

## Repository layout

```text
src/quant_research_engine/
  data/
  features/
  signals/
  portfolio/
  backtest/
  costs/
  metrics/
  reports/
tests/
configs/
notebooks/
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Reference projects

- [microsoft/qlib](https://github.com/microsoft/qlib) — quantitative research workflow and data abstractions
- [polakowo/vectorbt](https://github.com/polakowo/vectorbt) — vectorized backtesting and research ergonomics
- [kernc/backtesting.py](https://github.com/kernc/backtesting.py) — compact backtesting API and strategy interface

## License

MIT

## Available now

Importable local primitives include point-in-time bars, a past-only trend signal, cash/position accounting, transaction costs, and return/drawdown helpers. Run `PYTHONPATH=src python -m unittest discover -s tests`. Parquet/DuckDB integration and a full backtest loop are planned.
