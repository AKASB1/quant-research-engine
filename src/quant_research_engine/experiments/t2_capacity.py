"""Tier 2, item 3: capacity and impact (synthetic markets, assumed parameters).

``signal_x_ls`` (signal_x_ewma(3), long_short_quantile 0.2 equal, gross 1), rebalanced every 5
bars, run by the event engine (so the participation cap can bind) with equity sizing and the
base cost model (square-root impact, y = 0.5) on ``planted`` markets, at a grid of initial
capitals. Reported per capital: net and gross (before every cost) Sharpe ratios, the impact
cost and the total cost per year as fractions of equity, and the share of fill events (an
instrument and a bar with at least one fill) in which the fills reached the participation cap
(the cap is shared by the orders of an instrument and bar); the capacity is the capital
at which the mean net Sharpe ratio falls to half of its value at the smallest capital, and at
which it turns zero (log-linear interpolation; "not reached" when it does not).
"""

from __future__ import annotations

import math
import os

import numpy as np

from quant_research_engine.backtest import RunConfig, run_strategy
from quant_research_engine.costs import CostConfig
from quant_research_engine.data.manifest import write_json
from quant_research_engine.engine import EngineConfig
from quant_research_engine.experiments.common import mean_ci, write_rows
from quant_research_engine.experiments.markets import market
from quant_research_engine.metrics import sharpe


def jobs(cfg, mode):
    a, b = cfg["seeds"]
    return [("seed", s) for s in range(a, b + 1)]


def sort_key(r):
    return (float(r["capital"]), int(r["seed"]))


def run_job(spec, cfg, mode, outputs):
    from quant_research_engine.strategies.signal_x_ls import SignalXLS

    _, seed = spec
    st, _ = market(
        "t2_capacity", "planted", seed, n_instruments=cfg["n_instruments"], n_bars=cfg["n_bars"]
    )
    rows = []
    w = cfg["warmup"]
    panel = st.panel()
    cap_rate = CostConfig().participation_cap
    for cap in cfg["capitals"]:
        eng = EngineConfig(initial_cash=float(cap), costs=CostConfig(), sizing="equity")
        out = run_strategy(
            st, SignalXLS(), RunConfig(engine=eng, warmup_bars=w, rebalance_every=cfg["rebalance"])
        )
        r = out.result
        e = r.equity
        net = e[w + 1 :] / e[w:-1] - 1.0
        rows_eq = r.equity_rows[w + 1 :]
        # gross = before every cost: hold PnL + trade PnL + income (dividends; the cash rate is 0)
        gross = np.asarray([(x[4] + x[5] + x[11]) for x in rows_eq]) / e[w:-1]
        impact = np.asarray([x[7] for x in rows_eq]) / e[w:-1]
        bound, events = _cap_bound(r.fills, panel, eng.resolved(st.calendar_name), cap_rate)
        rows.append(
            {
                "seed": seed,
                "capital": float(cap),
                "net_sharpe": sharpe(net, 252),
                "gross_sharpe": sharpe(gross, 252),
                "impact_drag_annual": float(impact.mean() * 252),
                "cost_drag_annual": float((gross - net).mean() * 252),
                "cap_bound_share": bound / events if events else 0.0,
                "fill_events": events,
                "max_identity_residual": r.max_residual,
            }
        )
    return rows


def _cap_bound(fills, panel, eng, cap_rate) -> tuple[int, int]:
    """(fill events at which the fills of an instrument and bar reached the participation cap,
    fill events): a fill event is an instrument and a bar with at least one (non-delisting)
    fill."""
    stamps = panel.ts_open if eng.fill_model == "next_open" else panel.ts_event
    k_of = {int(t): k for k, t in enumerate(stamps)}
    col = {iid: j for j, iid in enumerate(panel.ids)}
    filled: dict[tuple[int, int], float] = {}
    for f in fills:
        if f[1] == "DELIST":
            continue
        key = (k_of[int(f[2])], col[f[3]])
        filled[key] = filled.get(key, 0.0) + abs(float(f[4]))
    bound = sum(
        1
        for (k, i), q in filled.items()
        if q >= cap_rate * float(panel.volume[k, i]) * (1.0 - 1e-9) - 1e-6
    )
    return bound, len(filled)


def _crossing(caps, vals, level):
    for (c0, v0), (c1, v1) in zip(
        zip(caps[:-1], vals[:-1], strict=True), zip(caps[1:], vals[1:], strict=True), strict=True
    ):
        if v0 > level >= v1:
            x0, x1 = math.log10(c0), math.log10(c1)
            return 10 ** (x0 + (x1 - x0) * (v0 - level) / (v0 - v1))
    return None


def finalize(cfg, rows, results_dir, outputs_dir, mode):
    files = {}
    p = os.path.join(results_dir, "rows.csv")
    write_rows(p, rows, ["capital", "seed"])
    files["rows.csv"] = p
    agg = []
    for cap in cfg["capitals"]:
        sel = [r for r in rows if r["capital"] == float(cap)]
        d = {"capital": float(cap), "seeds": len(sel)}
        for k in (
            "net_sharpe",
            "gross_sharpe",
            "impact_drag_annual",
            "cost_drag_annual",
            "cap_bound_share",
        ):
            d[f"{k}_mean"], d[f"{k}_ci95"], _ = mean_ci([r[k] for r in sel])
        agg.append(d)
    ap = os.path.join(results_dir, "aggregates.csv")
    write_rows(ap, agg, ["capital"])
    files["aggregates.csv"] = ap
    caps = [d["capital"] for d in agg]
    net = [d["net_sharpe_mean"] for d in agg]
    half = _crossing(caps, net, net[0] / 2) if net and net[0] > 0 else None
    zero = _crossing(caps, net, 0.0) if net and net[0] > 0 else None
    jp = os.path.join(results_dir, "aggregates.json")
    write_json(
        jp,
        {
            "net_sharpe_at_smallest_capital": net[0] if net else None,
            "capacity_half_sharpe": half if half is not None else "not reached in the grid",
            "capacity_zero_sharpe": zero if zero is not None else "not reached in the grid",
        },
    )
    files["aggregates.json"] = jp
    return files, {}
