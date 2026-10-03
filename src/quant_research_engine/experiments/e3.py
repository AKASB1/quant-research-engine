"""E3: calibration of the inference tools on markets where the truth is known.

108 trials (18 signals x 2 portfolio rules x 3 rebalance intervals) on every market, run by the
vectorized engine with fixed-capital sizing (per-bar return = change in equity over the fixed
sizing capital, so a run whose equity shrinks keeps a meaningful return); every trial
starts at the same decision bar, so the
matrix of per-bar returns is aligned (1000 rows for 1260 bars). Two panels per market: gross
(all costs zero; the calibration: on a null market every trial's true Sharpe ratio is zero) and
net (base costs; reported separately, for what gets selected and what the PBO says). Per market
and panel: the in-sample best trial, its per-bar Sharpe, SR0 for the registered trials, the
naive PSR(0), the DSR, the PBO by CSCV, whether it uses signal_x; on null gross panels the
stationary-bootstrap interval of the Sharpe ratio of the selected trial and of a trial fixed in
advance. Part (f): the Lo adjustment on AR(1) returns.
"""

from __future__ import annotations

import math
import os

import numpy as np

from quant_research_engine.costs import CostConfig, zero_costs
from quant_research_engine.data.manifest import config_hash, sha256_file, write_json
from quant_research_engine.engine.panelutil import liquidity_asof_panel
from quant_research_engine.engine.quantity import round_lot
from quant_research_engine.experiments.common import mean_ci, wilson, write_rows
from quant_research_engine.experiments.markets import market, market_seed
from quant_research_engine.features import PanelSource, make_feature
from quant_research_engine.inference.bootstrap import (
    bootstrap_stat,
    percentile_interval,
    sharpe_rows,
)
from quant_research_engine.inference.pbo import pbo_cscv
from quant_research_engine.inference.registry import TrialRegistry
from quant_research_engine.inference.sharpe import psr
from quant_research_engine.metrics import lo_eta, sharpe
from quant_research_engine.portfolio import long_short_quantile
from quant_research_engine.rng import stream
from quant_research_engine.signals import zscore
from quant_research_engine.vector import run_target_shares


def trial_list(cfg) -> list[dict]:
    out = []
    for sig in cfg["grid"]["signals"]:
        for params in sig["params"]:
            for ri, rule in enumerate(cfg["grid"]["rules"]):
                for reb in cfg["grid"]["rebalance"]:
                    out.append(
                        {
                            "feature": sig["feature"],
                            "params": params,
                            "rule": ri,
                            "q": rule["q"],
                            "weighting": rule["weighting"],
                            "rebalance": reb,
                        }
                    )
    return out


def trial_id(i: int) -> str:
    return f"t{i:03d}"


def jobs(cfg, mode):
    out = []
    for kind in ("null", "planted"):
        a, b = cfg[kind]["seeds"]
        out += [("market", kind, s) for s in range(a, b + 1)]
    out += [("lo", i, 0) for i in range(len(cfg["lo"]["phis"]))]
    return out


def sort_key(r):
    return (
        r["part"],
        str(r.get("kind", "")),
        int(r["seed"]),
        str(r.get("panel", "")),
        str(r.get("trial", "")),
    )


def returns_matrix(st, cfg, trials, costs, liq=None, log_dir=None):
    """(rows, n_trials) per-bar returns after the first decision; also the max identity residual."""
    p = st.panel()
    T, N = p.shape
    src = PanelSource(st)
    first = int(cfg["first_decision_bar"])
    rows_all = np.arange(first, T)
    uni = src.universe_mask()
    liq = liq or liquidity_asof_panel(p)
    close = p.close
    cap = float(cfg["capital"])
    feats: dict = {}
    weights: dict = {}
    out = np.full((T - first - 1, len(trials)), np.nan)
    resid = 0.0
    for j, tr in enumerate(trials):
        fk = (tr["feature"], tuple(sorted(tr["params"].items())))
        if fk not in feats:
            f = make_feature(tr["feature"], **tr["params"]).panel(src)
            feats[fk] = zscore(np.where(uni[rows_all], f[rows_all], np.nan))
        wk = (fk, tr["rule"])
        if wk not in weights:
            weights[wk] = long_short_quantile(feats[fk], tr["q"], tr["weighting"], 1.0)
        dec = rows_all[:: tr["rebalance"]]
        W = weights[wk][:: tr["rebalance"]]
        c = close[dec]
        ok = np.isfinite(c) & (W != 0)
        tg = np.where(
            ok,
            round_lot(np.where(ok, W * cap / np.where(ok, c, 1.0), 0.0), p.lot_size[None, :]),
            0.0,
        )
        vr = run_target_shares(p, dec, tg, initial_cash=cap, costs=costs, liquidity=liq)
        e = vr.equity
        # fixed-capital sizing: per-bar return = change in equity over the sizing capital
        out[:, j] = (e[first + 1 :] - e[first:-1]) / cap
        resid = max(resid, vr.max_residual)
        if log_dir is not None and j == 0:
            from quant_research_engine.engine import EngineConfig
            from quant_research_engine.validation.equivalence import write_vector_logs
            from quant_research_engine.validation.validator import validate_run

            ec = EngineConfig(initial_cash=cap, costs=costs)
            write_vector_logs(vr, p, dec, ec, log_dir)
            rep = validate_run(log_dir, st)
            if not rep.ok:
                raise AssertionError(f"validator failed on the E3 sample log: {rep.errors[:3]}")
    return out, resid


def _panel_stats(R, trials, cfg, study_id, trials_path, seed_boot, fixed_idx, do_boot):
    reg = TrialRegistry(study_id, trials_path)
    for j, tr in enumerate(trials):
        reg.add(
            trial_id(j),
            {k: tr[k] for k in ("feature", "params", "q", "weighting", "rebalance")},
            R[:, j],
        )
    sel = reg.select()
    _, _, _, sr, t, sk, ku = reg.rows[sel]
    out = {
        "selected": trial_id(sel),
        "selected_feature": trials[sel]["feature"],
        "selected_rebalance": trials[sel]["rebalance"],
        "selected_rule": trials[sel]["rule"],
        "sr_bar": sr,
        "sharpe_ann": sr * math.sqrt(252),
        "skew": sk,
        "kurt": ku,
        "t_bars": t,
        "n_trials": reg.n,
        "degenerate_trials": reg.degenerate,
        "var_sr": reg.var_sr(),
        "sr0": reg.sr0(),
        "psr0": psr(sr, 0.0, t, sk, ku),
        "dsr": reg.dsr(sel),
        "pbo": pbo_cscv(R, cfg["S"]).pbo,
        "uses_signal_x": trials[sel]["feature"] == "signal_x_ewma",
    }
    if do_boot:
        b = cfg["bootstrap"]
        rng = stream(seed_boot, "bootstrap")
        for name, j in (("selected", sel), ("fixed", fixed_idx)):
            vals = bootstrap_stat(R[:, j], sharpe_rows, b["draws"], b["p"], rng)
            lo, hi = percentile_interval(vals)
            out[f"boot_{name}_lo"] = lo
            out[f"boot_{name}_hi"] = hi
            out[f"boot_{name}_covers_zero"] = lo <= 0.0 <= hi
    return out, reg.rows


def run_job(spec, cfg, mode, outputs):
    part = spec[0]
    if part == "lo":
        return _lo_job(spec[1], cfg)
    _, kind, seed = spec
    trials = trial_list(cfg)
    fixed = cfg["fixed_trial"]
    fixed_idx = next(
        j
        for j, t in enumerate(trials)
        if t["feature"] == fixed["feature"]
        and t["params"] == fixed["params"]
        and t["rule"] == fixed["rule"]
        and t["rebalance"] == fixed["rebalance"]
    )
    st, _ = market(
        "e3", cfg[kind]["synth"], seed, n_instruments=cfg["n_instruments"], n_bars=cfg["n_bars"]
    )
    liq = liquidity_asof_panel(st.panel())
    first_seed = cfg[kind]["seeds"][0] == seed
    rows = []
    for panel, costs in (("gross", zero_costs()), ("net", CostConfig())):
        log_dir = None
        if cfg.get("log_one_trial_per_panel") and first_seed:
            log_dir = os.path.join(outputs, "logs", f"{kind}_{seed}_{panel}")
        R, resid = returns_matrix(st, cfg, trials, costs, liq, log_dir)
        tdir = os.path.join(outputs, "trials")
        os.makedirs(tdir, exist_ok=True)
        tpath = os.path.join(tdir, f"{kind}_{seed}_{panel}.csv")
        stats, trows = _panel_stats(
            R,
            trials,
            cfg,
            f"e3_{kind}_{seed}_{panel}",
            tpath,
            market_seed("e3", seed),
            fixed_idx,
            kind == "null" and panel == "gross",
        )
        rows.append(
            {
                "part": "selected",
                "kind": kind,
                "seed": seed,
                "panel": panel,
                "max_identity_residual": resid,
                "trials_sha256": sha256_file(tpath),
                "sample_log_validated": log_dir is not None,
                **stats,
            }
        )
        for tr in trows:
            rows.append(
                {
                    "part": "trial",
                    "kind": kind,
                    "seed": seed,
                    "panel": panel,
                    "trial": tr[1],
                    "sr_bar": tr[3],
                    "t_bars": tr[4],
                    "skew": tr[5],
                    "kurt": tr[6],
                }
            )
    return rows


def _lo_job(i, cfg):
    lo = cfg["lo"]
    phi = float(lo["phis"][i])
    q = 252
    rho = [phi**k for k in range(1, q)]
    true_ann = lo["sr_bar"] * lo_eta(q, rho=rho, L=10**6)
    rows = []
    for s in range(lo["series_per_phi"]):
        g = stream(s, f"experiment.e3.lo.{phi}")
        e = g.standard_normal(lo["n_bars"] + 100) * math.sqrt(1 - phi * phi)
        x = np.empty_like(e)
        x[0] = g.standard_normal()
        for t in range(1, len(e)):
            x[t] = phi * x[t - 1] + e[t]
        r = lo["sr_bar"] + x[100:]
        sr = sharpe(r)
        rows.append(
            {
                "part": "lo",
                "seed": s,
                "phi": phi,
                "naive_ann": sr * math.sqrt(q),
                "adjusted_ann": sr * lo_eta(q, r=r, L=lo["L"]),
                "true_ann": true_ann,
            }
        )
    return rows


def finalize(cfg, rows, results_dir, outputs_dir, mode):
    files = {}
    sel = [r for r in rows if r["part"] == "selected"]
    trial_rows = [r for r in rows if r["part"] == "trial"]
    lo = [r for r in rows if r["part"] == "lo"]
    p = os.path.join(results_dir, "selected.csv")
    write_rows(p, sel, ["part", "kind", "seed", "panel"])
    files["selected.csv"] = p
    tp = os.path.join(outputs_dir, "trial_rows.csv")
    write_rows(tp, trial_rows, ["part", "kind", "seed", "panel", "trial"])
    lp = os.path.join(results_dir, "lo.csv")
    write_rows(lp, lo, ["part", "phi", "seed"])
    files["lo.csv"] = lp
    # the trials.csv of the first market of each panel, as an example
    import shutil

    for kind in ("null", "planted"):
        s0 = cfg[kind]["seeds"][0]
        for panel in ("gross", "net"):
            src = os.path.join(outputs_dir, "trials", f"{kind}_{s0}_{panel}.csv")
            dst = os.path.join(results_dir, f"trials_{kind}_{s0}_{panel}.csv")
            if os.path.exists(src):
                shutil.copyfile(src, dst)
                files[os.path.basename(dst)] = dst
    agg = []
    for kind in ("null", "planted"):
        for panel in ("gross", "net"):
            s = [r for r in sel if r["kind"] == kind and r["panel"] == panel]
            if not s:
                continue
            n = len(s)
            a = {"kind": kind, "panel": panel, "markets": n}
            for key in ("sr_bar", "sr0", "sharpe_ann", "pbo", "dsr", "psr0"):
                m, ci, _ = mean_ci([r[key] for r in s])
                a[f"{key}_mean"], a[f"{key}_ci95"] = m, ci
            ratio = [r["sr_bar"] / r["sr0"] for r in s if r["sr0"] > 0]
            a["sr_over_sr0_mean"], a["sr_over_sr0_ci95"], _ = mean_ci(ratio)
            for key, name in (("psr0", "naive_psr"), ("dsr", "dsr")):
                k = sum(1 for r in s if r[key] > 0.95)
                pr, lo_, hi = wilson(k, n)
                a[f"{name}_gt_095_share"], a[f"{name}_gt_095_lo"], a[f"{name}_gt_095_hi"] = (
                    pr,
                    lo_,
                    hi,
                )
            k = sum(1 for r in s if r["uses_signal_x"])
            a["selected_signal_x_share"], a["selected_signal_x_lo"], a["selected_signal_x_hi"] = (
                wilson(k, n)
            )
            k = sum(1 for r in s if r["pbo"] > 0.5)
            a["pbo_gt_05_share"] = k / n
            if kind == "null" and panel == "gross":
                for name in ("selected", "fixed"):
                    k = sum(1 for r in s if r[f"boot_{name}_covers_zero"])
                    a[f"boot_{name}_coverage"], a[f"boot_{name}_lo"], a[f"boot_{name}_hi"] = wilson(
                        k, n
                    )
            agg.append(a)
    for phi in sorted({r["phi"] for r in lo}):
        s = [r for r in lo if r["phi"] == phi]
        a = {"kind": "lo", "panel": f"phi={phi}", "markets": len(s), "true_ann": s[0]["true_ann"]}
        for key in ("naive_ann", "adjusted_ann"):
            m, ci, _ = mean_ci([r[key] for r in s])
            a[f"{key}_mean"], a[f"{key}_ci95"] = m, ci
        agg.append(a)
    ap = os.path.join(results_dir, "aggregates.csv")
    write_rows(ap, agg, ["kind", "panel"])
    files["aggregates.csv"] = ap
    hp = os.path.join(results_dir, "pbo_hist.csv")
    hist = []
    edges = np.linspace(0, 1, 11)
    for kind in ("null", "planted"):
        for panel in ("gross", "net"):
            v = np.asarray([r["pbo"] for r in sel if r["kind"] == kind and r["panel"] == panel])
            if v.size == 0:
                continue
            h, _ = np.histogram(v, bins=edges)
            for b in range(10):
                hist.append(
                    {
                        "kind": kind,
                        "panel": panel,
                        "bin_lo": float(edges[b]),
                        "bin_hi": float(edges[b + 1]),
                        "count": int(h[b]),
                    }
                )
    write_rows(hp, hist, ["kind", "panel", "bin_lo"])
    files["pbo_hist.csv"] = hp
    trials = trial_list(cfg)
    tl = os.path.join(results_dir, "trial_grid.csv")
    write_rows(
        tl,
        [
            {
                "trial": trial_id(j),
                "feature": t["feature"],
                "params": config_hash(t["params"])[:12],
                "params_text": ";".join(f"{k}={v}" for k, v in sorted(t["params"].items())),
                "rule": t["rule"],
                "q": t["q"],
                "weighting": t["weighting"],
                "rebalance": t["rebalance"],
            }
            for j, t in enumerate(trials)
        ],
        ["trial"],
    )
    files["trial_grid.csv"] = tl
    write_json(
        os.path.join(results_dir, "outputs_index.json"),
        {"trial_rows.csv": sha256_file(tp), "rows": len(trial_rows)},
    )
    files["outputs_index.json"] = os.path.join(results_dir, "outputs_index.json")
    return files, {"trial_rows_sha256": sha256_file(tp), "trial_rows": len(trial_rows)}
