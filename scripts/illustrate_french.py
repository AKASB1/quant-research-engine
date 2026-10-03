"""Illustration, no claims: the Kenneth R. French Data Library adapter on one public file.

python scripts/illustrate_french.py [--cache data/cache] [--out experiments/results/t2_french]
                                    [--offline]

- Data: ``49_Industry_Portfolios_daily_CSV.zip`` (value-weighted daily returns of 49 industry
  portfolios; copyright Eugene F. Fama and Kenneth R. French). Downloaded once into the
  git-ignored cache; ``--offline`` refuses to download. Nothing downloaded is committed,
  packaged, or exported. The library revises history, so the series is the latest vintage and
  not point-in-time; each return is made known one day after its close (``ts_avail``).
- Run: the built-in ``momentum_ls`` (momentum(21, 5), long_short_quantile(0.2, equal, gross 1)),
  rebalanced every 21 bars from 2000 on, event engine, equity sizing. The source has no open and
  no volume: fills at ``next_close``, impact off, a placeholder volume that never binds the
  participation cap; spread and commission of the base cost model.
- Output: a small JSON of derived metrics only (no data) and a manifest. One fixed strategy, one
  trial, chosen before the data was read: the numbers illustrate that the adapter and the engine
  run on such a file, and say nothing about the strategy or about markets.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date

import numpy as np

from quant_research_engine.backtest import RunConfig, run_strategy
from quant_research_engine.costs import CostConfig, ImpactCfg
from quant_research_engine.data.manifest import write_json
from quant_research_engine.engine import EngineConfig
from quant_research_engine.experiments.common import git_state, hardware, software
from quant_research_engine.inference.sharpe import psr
from quant_research_engine.metrics import kurt, max_drawdown, sharpe, skew
from quant_research_engine.store.adapters import (
    fetch_french,
    parse_french_daily,
    store_from_returns,
)
from quant_research_engine.strategies.momentum_ls import MomentumLS

FILE = "49_Industry_Portfolios_daily"
START = date(2000, 1, 3)
WARMUP, REBALANCE = 30, 21


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join("data", "cache"))
    ap.add_argument("--out", default=os.path.join("experiments", "results", "t2_french"))
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    path = os.path.join(a.cache, f"{FILE}_CSV.zip")
    if a.offline and not os.path.exists(path):
        print(f"{path} is not in the cache and --offline was given", file=sys.stderr)
        return 2
    text = fetch_french(FILE, a.cache)
    dates, names, rets = parse_french_daily(text)
    st = store_from_returns(dates, names, rets, delay_days=1, start=START, source=FILE)
    costs = CostConfig(impact=ImpactCfg(model="none"))
    eng = EngineConfig(fill_model="next_close", initial_cash=1e7, sizing="equity", costs=costs)
    out = run_strategy(
        st, MomentumLS(), RunConfig(engine=eng, warmup_bars=WARMUP, rebalance_every=REBALANCE)
    )
    r = out.result
    e = r.equity
    w = WARMUP
    net = e[w + 1 :] / e[w:-1] - 1.0
    rows = r.equity_rows[w + 1 :]
    gross = np.asarray([x[4] + x[5] + x[11] for x in rows]) / e[w:-1]
    sr_bar = float(net.mean() / net.std(ddof=1))
    used = [d for d in dates if d >= START]
    summary = {
        "label": "illustration, no claims",
        "data": {
            "source": "Kenneth R. French Data Library, " + FILE + " (value-weighted daily returns)",
            "terms": "copyright notice of Eugene F. Fama and Kenneth R. French; no terms of use, "
            "licence, or click-through found on the data library page; the file says it was "
            "created from the CRSP database; downloaded to the git-ignored cache only",
            "vintage": "latest at download time; not point-in-time (the library revises history)",
            "first_date": used[0].isoformat(),
            "last_date": used[-1].isoformat(),
            "portfolios": len(names),
            "cache_file_sha256": _sha256(path),
        },
        "run": {
            "strategy": "momentum_ls (momentum(21, 5), long_short_quantile(0.2, equal, gross 1))",
            "rebalance_every_bars": REBALANCE,
            "warmup_bars": WARMUP,
            "fill_model": "next_close (the source has no open)",
            "costs": "base spread and commission; impact off (no volume in the source)",
            "ts_avail": "close + 1 day",
            "trials": 1,
        },
        "metrics": {
            "bars": int(net.size),
            "net_sharpe_annualized": sharpe(net, 252),
            "gross_sharpe_annualized": sharpe(gross, 252),
            "psr0_net": psr(sr_bar, 0.0, int(net.size), skew(net), kurt(net)),
            "max_drawdown": max_drawdown(e[w:]),
            "fills": len(r.fills),
            "max_identity_residual": r.max_residual,
        },
    }
    os.makedirs(a.out, exist_ok=True)
    write_json(os.path.join(a.out, "summary.json"), summary)
    write_json(
        os.path.join(a.out, "manifest.json"),
        {"experiment": "t2_french", "label": "illustration, no claims", "git": git_state(),
         "software": software(), "hardware": hardware()},
    )  # fmt: skip
    print(summary["metrics"])
    return 0 if math.isfinite(summary["metrics"]["net_sharpe_annualized"]) else 1


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
