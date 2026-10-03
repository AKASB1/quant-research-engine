"""E4: signal recovery, costs, and lag (synthetic markets with assumed parameters).

(a) ``planted_clean``: realized information coefficients of ``signal_x`` and of the latent
    signal (an oracle) against the standardized idiosyncratic return; the per-bar Sharpe ratio of
    the fundamental-law weights s / (sigma sqrt N) held close to close (a bound, from the
    generator's panels, not achievable under next-bar fills); the same weights at gross exposure
    1 through the event engine (next_open, equity sizing, a decision every bar, zero costs, no
    participation cap); the close-to-close Sharpe ratio of the ``signal_x`` weights over that of
    the latent weights. ``planted``: the two information coefficients against the standardized
    idiosyncratic return and, labelled, against the raw next-bar return.
(b) ``planted``: signal_x_ewma(3) with long_short_quantile(0.2, equal, gross 1), rebalanced every
    1, 5, 21 bars, at cost multiples 0, 0.5, 1, 2, 4 of the base cost model (vectorized engine,
    fixed capital; returns and turnover are measured against the fixed capital).
(c) ``planted`` with persistence 0.5, 0.9, 0.97: the same strategy rebalanced every 5 bars with
    base costs at fill delays 1, 2, 3 (next_open) and with next_close.
Pooled Sharpe ratios are computed from per-seed sums (sum r, sum r^2, n) over all seeds.
"""

from __future__ import annotations

import math
import os

import numpy as np

from quant_research_engine.backtest import RunConfig, run_strategy
from quant_research_engine.context import Strategy, TargetWeights
from quant_research_engine.costs import CostConfig, zero_costs
from quant_research_engine.data.manifest import write_json
from quant_research_engine.engine import EngineConfig
from quant_research_engine.engine.panelutil import liquidity_asof_panel
from quant_research_engine.engine.quantity import round_lot
from quant_research_engine.experiments.common import mean_ci, write_rows
from quant_research_engine.experiments.markets import market
from quant_research_engine.features import PanelSource
from quant_research_engine.strategies.base import FeatureRuleStrategy
from quant_research_engine.synth import close_to_close_returns
from quant_research_engine.vector import run_target_shares


class OracleWeights(Strategy):
    """Weights s_t / sum|s_t| from the latent signal: an oracle, labelled as such."""

    name = "oracle_fl"
    history_bars = 1

    def __init__(self, s, ts_event, ids):
        super().__init__()
        self._w = s / np.sum(np.abs(s), axis=1, keepdims=True)
        self._k = {int(t): k for k, t in enumerate(ts_event.tolist())}
        self._ids = ids

    def decide(self, ctx):
        k = self._k[ctx.t]
        return TargetWeights({i: float(self._w[k, j]) for j, i in enumerate(self._ids)})


def jobs(cfg, mode):
    a, b = cfg["seeds"]
    return [("seed", s) for s in range(a, b + 1)]


def sort_key(r):
    return (r["part"], str(r.get("config", "")), int(r["seed"]))


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def _ics(tr):
    alive = ~np.isnan(tr.r)
    alive[0] = False
    prev_s = np.vstack([np.zeros((1, tr.s.shape[1])), tr.s[:-1]])
    prev_x = np.vstack([np.zeros((1, tr.x.shape[1])), tr.x[:-1]])
    return {
        "ic_latent": _pearson(prev_s[alive], tr.idio_std[alive]),
        "ic_signal_x": _pearson(prev_x[alive], tr.idio_std[alive]),
        "ic_latent_raw": _pearson(prev_s[alive], tr.r[alive]),
        "ic_signal_x_raw": _pearson(prev_x[alive], tr.r[alive]),
    }


def _sums(r):
    r = np.asarray(r, dtype=np.float64)
    return {
        "sum_r": float(math.fsum(r.tolist())),
        "sum_r2": float(math.fsum((r * r).tolist())),
        "n": int(r.size),
        "sr_bar": float(r.mean() / r.std(ddof=1)),
    }


def _strategy_returns(st, cfg, rebalance, costs, fill_model="next_open", delay=1, liq=None):
    p = st.panel()
    T = p.shape[0]
    s = cfg["strategy"]
    strat = FeatureRuleStrategy(
        feature=s["feature"],
        feature_params=s["params"],
        rule="long_short_quantile",
        rule_params={"q": s["q"], "weighting": s["weighting"], "gross": 1.0},
    )
    src = PanelSource(st)
    first = int(cfg["first_decision_bar"])
    rows = np.arange(first, T)
    W = strat.weights_panel(src, rows)[::rebalance]
    dec = rows[::rebalance]
    c = p.close[dec]
    ok = np.isfinite(c) & (W != 0)
    cap = float(cfg["capital"])
    tg = np.where(
        ok, round_lot(np.where(ok, W * cap / np.where(ok, c, 1.0), 0.0), p.lot_size[None, :]), 0.0
    )
    vr = run_target_shares(
        p,
        dec,
        tg,
        fill_model=fill_model,
        fill_delay_bars=delay,
        initial_cash=cap,
        costs=costs,
        liquidity=liq,
    )
    e = vr.equity
    # fixed-capital sizing: returns are PnL over the fixed sizing capital (equity can shrink a lot
    # at high cost multiples, and a return over a shrinking or negative equity is meaningless)
    net = (e[first + 1 :] - e[first:-1]) / cap
    gross = (vr.hold_pnl[first + 1 :] + vr.trade_pnl[first + 1 :]) / cap
    return net, gross, vr


def _validate_sample(st, cfg, vr, out):
    """Quick mode: write the logs of one vectorized run and confirm them with the validator."""
    from quant_research_engine.validation.equivalence import write_vector_logs
    from quant_research_engine.validation.validator import validate_run

    p = st.panel()
    dec = np.arange(int(cfg["first_decision_bar"]), p.shape[0])[::5]
    write_vector_logs(vr, p, dec, EngineConfig(initial_cash=float(cfg["capital"])), out)
    rep = validate_run(out, st)
    if not rep.ok:
        raise AssertionError(f"validator failed on the E4 sample log: {rep.errors[:3]}")


def run_job(spec, cfg, mode, outputs):
    _, seed = spec
    N, T = cfg["n_instruments"], cfg["n_bars"]
    rows = []
    # (a) planted_clean
    st, tr = market("e4", "planted_clean", seed, n_instruments=N, n_bars=T)
    c = st.meta["generator"]["parameters"]
    sig = tr.sigma
    w_lat = tr.s / (sig[None, :] * math.sqrt(N))
    w_x = tr.x / (sig[None, :] * math.sqrt(N))
    lat = close_to_close_returns(w_lat, tr.r)
    xr = close_to_close_returns(w_x, tr.r)
    p = st.panel()
    costs = zero_costs().model_copy(update={"participation_cap": 1e9})
    rc = RunConfig(
        engine=EngineConfig(initial_cash=1e6, costs=costs, sizing="equity"), warmup_bars=0
    )
    out = run_strategy(st, OracleWeights(tr.s, p.ts_event, p.ids), rc)
    e = out.result.equity
    eng = e[2:] / e[1:-1] - 1.0
    base = {
        "part": "a",
        "seed": seed,
        "ic_config": c["ic"],
        "observable_noise": c["observable_noise"],
        "gap_share": c["gap_share"],
        "persistence": c["persistence"],
    }
    rows.append({**base, "config": "clean", **_ics(tr)})
    for name, r in (
        ("fl_bound_close_to_close", lat),
        ("signal_x_close_to_close", xr),
        ("engine_next_open_oracle", eng),
    ):
        rows.append(
            {
                **base,
                "config": name,
                **_sums(r),
                "cap_binds": out.result.cap_binds if name.startswith("engine") else 0,
            }
        )
    # (a) planted: the two information coefficients, also against the raw next-bar return
    st, tr = market("e4", "planted", seed, n_instruments=N, n_bars=T)
    rows.append({"part": "a", "seed": seed, "config": "planted", **_ics(tr)})
    liq = liquidity_asof_panel(st.panel())
    # (b) cost multiples
    for reb in cfg["b"]["rebalance"]:
        for mult in cfg["b"]["cost_mult"]:
            net, gross, vr = _strategy_returns(st, cfg, reb, CostConfig().scaled(mult), liq=liq)
            first = int(cfg["first_decision_bar"])
            if mode == "quick" and seed == cfg["seeds"][0] and reb == 5 and mult == 1.0:
                _validate_sample(st, cfg, vr, os.path.join(outputs, "logs", f"b_{seed}_reb5_x1"))
            rows.append(
                {
                    "part": "b",
                    "seed": seed,
                    "config": f"reb{reb}_x{mult}",
                    "rebalance": reb,
                    "cost_mult": mult,
                    "net_sharpe": float(net.mean() / net.std(ddof=1) * math.sqrt(252)),
                    "gross_sharpe": float(gross.mean() / gross.std(ddof=1) * math.sqrt(252)),
                    "turnover_per_bar": float(
                        np.mean(vr.turnover[first + 1 :] * vr.equity[first:-1])
                        / float(cfg["capital"])
                    ),
                    "min_equity_over_capital": float(vr.equity.min() / float(cfg["capital"])),
                    "cost_drag_annual": float((gross.mean() - net.mean()) * 252),
                    "max_identity_residual": vr.max_residual,
                }
            )
    # (c) persistence x fill delay / next_close
    for phi in cfg["c"]["phis"]:
        stp, _ = market("e4", "planted", seed, n_instruments=N, n_bars=T, persistence=phi)
        liqp = liquidity_asof_panel(stp.panel())
        variants = [("next_open", d) for d in cfg["c"]["delays"]]
        if cfg["c"].get("next_close"):
            variants.append(("next_close", 1))
        for fm, d in variants:
            net, gross, vr = _strategy_returns(
                stp, cfg, cfg["c"]["rebalance"], CostConfig(), fill_model=fm, delay=d, liq=liqp
            )
            rows.append(
                {
                    "part": "c",
                    "seed": seed,
                    "config": f"phi{phi}_{fm}_d{d}",
                    "persistence": phi,
                    "fill_model": fm,
                    "fill_delay": d,
                    "net_sharpe": float(net.mean() / net.std(ddof=1) * math.sqrt(252)),
                    "gross_sharpe": float(gross.mean() / gross.std(ddof=1) * math.sqrt(252)),
                    "max_identity_residual": vr.max_residual,
                }
            )
    return rows


def _pooled(rows):
    s = sum(r["sum_r"] for r in rows)
    s2 = sum(r["sum_r2"] for r in rows)
    n = sum(r["n"] for r in rows)
    m = s / n
    var = (s2 - n * m * m) / (n - 1)
    sr = m / math.sqrt(var)
    return sr, math.sqrt((1 + 0.5 * sr * sr) / n), n


def finalize(cfg, rows, results_dir, outputs_dir, mode):
    files = {}
    p = os.path.join(results_dir, "rows.csv")
    write_rows(p, rows, ["part", "config", "seed"])
    files["rows.csv"] = p
    agg = {}
    a = [r for r in rows if r["part"] == "a"]
    clean = [r for r in a if r["config"] == "clean"]
    if clean:
        ic = clean[0]["ic_config"]
        noise = clean[0]["observable_noise"]
        g = clean[0]["gap_share"]
        phi = clean[0]["persistence"]
        N = cfg["n_instruments"]
        for key in ("ic_latent", "ic_signal_x"):
            m, ci, n = mean_ci([r[key] for r in clean])
            agg[f"clean_{key}"] = {"mean": m, "ci95": ci, "seeds": n}
        agg["clean_ic_latent"]["configured"] = ic
        agg["clean_ic_signal_x"]["configured"] = ic / math.sqrt(1 + noise * noise)
        bound = ic * math.sqrt(N)
        for name in (
            "fl_bound_close_to_close",
            "signal_x_close_to_close",
            "engine_next_open_oracle",
        ):
            sr, se, n = _pooled([r for r in a if r["config"] == name])
            agg[name] = {
                "pooled_sr_bar": sr,
                "se": se,
                "bars": n,
                "annualized": sr * math.sqrt(252),
            }
        agg["fl_bound_close_to_close"]["target_sr_bar"] = bound
        agg["fl_bound_close_to_close"]["target_annualized"] = bound * math.sqrt(252)
        agg["engine_next_open_oracle"]["target_sr_bar"] = bound * (1 - g * (1 - phi))
        agg["engine_next_open_oracle"]["ratio_to_target"] = agg["engine_next_open_oracle"][
            "pooled_sr_bar"
        ] / (bound * (1 - g * (1 - phi)))
        agg["signal_x_over_latent"] = {
            "ratio": agg["signal_x_close_to_close"]["pooled_sr_bar"]
            / agg["fl_bound_close_to_close"]["pooled_sr_bar"],
            "expected": 1 / math.sqrt(1 + noise * noise),
        }
    planted = [r for r in a if r["config"] == "planted"]
    for key in ("ic_latent", "ic_signal_x", "ic_latent_raw", "ic_signal_x_raw"):
        if planted:
            m, ci, n = mean_ci([r[key] for r in planted])
            agg[f"planted_{key}"] = {"mean": m, "ci95": ci, "seeds": n}
    b_rows = []
    for reb in cfg["b"]["rebalance"]:
        curve = []
        for mult in cfg["b"]["cost_mult"]:
            sel = [
                r
                for r in rows
                if r["part"] == "b" and r["rebalance"] == reb and r["cost_mult"] == mult
            ]
            d = {"rebalance": reb, "cost_mult": mult}
            for key in ("net_sharpe", "gross_sharpe", "turnover_per_bar", "cost_drag_annual"):
                d[f"{key}_mean"], d[f"{key}_ci95"], d["seeds"] = mean_ci([r[key] for r in sel])
            b_rows.append(d)
            curve.append((mult, d["net_sharpe_mean"]))
        be = None
        for (m0, s0), (m1, s1) in zip(curve[:-1], curve[1:], strict=True):
            if s0 > 0 >= s1:
                be = m0 + (m1 - m0) * s0 / (s0 - s1)
                break
        if be is None:
            be = "above " + str(curve[-1][0]) if curve[-1][1] > 0 else "below " + str(curve[0][0])
        for d in b_rows:
            if d["rebalance"] == reb:
                d["break_even_multiple"] = be
    bp = os.path.join(results_dir, "costs.csv")
    write_rows(bp, b_rows, ["rebalance", "cost_mult"])
    files["costs.csv"] = bp
    c_rows = []
    for phi in cfg["c"]["phis"]:
        for r0 in sorted(
            {r["config"] for r in rows if r["part"] == "c" and r["persistence"] == phi}
        ):
            sel = [r for r in rows if r["part"] == "c" and r["config"] == r0]
            d = {
                "persistence": phi,
                "fill_model": sel[0]["fill_model"],
                "fill_delay": sel[0]["fill_delay"],
            }
            for key in ("net_sharpe", "gross_sharpe"):
                d[f"{key}_mean"], d[f"{key}_ci95"], d["seeds"] = mean_ci([r[key] for r in sel])
            c_rows.append(d)
    cp = os.path.join(results_dir, "lag.csv")
    write_rows(cp, c_rows, ["persistence", "fill_model", "fill_delay"])
    files["lag.csv"] = cp
    ap = os.path.join(results_dir, "aggregates.json")
    write_json(ap, agg)
    files["aggregates.json"] = ap
    return files, {}
