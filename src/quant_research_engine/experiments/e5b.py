"""E5b (Tier 2): combinatorial purged cross-validation in the comparison of validation schemes.

Same markets, model, features, and labels as E5 (k-nearest-neighbour regression per instrument
on trailing returns and volatility; forward compounded return over h bars), on markets of its
own (seeds of E5b). Schemes: shuffled K-fold and contiguous K-fold without purge (UNPURGED),
purged K-fold, walk-forward (K = 5), and CPCV with 6 groups and 2 test groups (15 splits, 5
backtest paths, purge and embargo h). Two out-of-sample measures per scheme: the Spearman
correlation of predictions and labels (as in E5), and the per-bar Sharpe ratio of a simple
prediction-ranked long-short portfolio (weights: demeaned cross-sectional rank of the
predictions, gross 1, held one bar, no costs), each against the same measure on the held-out
segment. For CPCV every one of the 5 paths gives its own estimate; a market's CPCV estimate is
the mean over its paths, and the report shows the spread of the paths within a market next to
the spread across markets. "Closest to the held-out truth" is the smallest mean absolute bias
over markets; "noisiest" is the largest standard deviation of the per-market bias (estimate
minus held-out value). Both are point estimates: the tables give the intervals.
"""

from __future__ import annotations

import math
import os

import numpy as np

from quant_research_engine.data.manifest import write_json
from quant_research_engine.experiments.common import mean_ci, write_rows
from quant_research_engine.experiments.e5 import _features, _labels, knn_predict, spearman
from quant_research_engine.experiments.markets import market, market_seed
from quant_research_engine.inference.splitters import cpcv, make_splits
from quant_research_engine.rng import stream


def jobs(cfg, mode):
    out = []
    for kind in ("null", "planted"):
        a, b = cfg[kind]["seeds"]
        out += [("market", kind, s) for s in range(a, b + 1)]
    return out


def sort_key(r):
    return (r["kind"], int(r["seed"]), int(r["h"]), r["scheme"], int(r.get("path", -1)))


def _rank_portfolio_sharpe(pred: np.ndarray, ret_next: np.ndarray) -> float:
    """pred, ret_next: (bars, N) with NaN where absent; per-bar Sharpe of the rank portfolio."""
    out = []
    for i in range(pred.shape[0]):
        ok = np.isfinite(pred[i]) & np.isfinite(ret_next[i])
        if ok.sum() < 4:
            continue
        rk = np.argsort(np.argsort(pred[i, ok], kind="stable"), kind="stable").astype(np.float64)
        w = rk - rk.mean()
        g = np.abs(w).sum()
        if g == 0:
            continue
        out.append(float((w / g * ret_next[i, ok]).sum()))
    r = np.asarray(out)
    if r.size < 3 or r.std(ddof=1) == 0:
        return math.nan
    return float(r.mean() / r.std(ddof=1))


def cpcv_paths(n: int, n_groups: int, splits, n_paths: int) -> list[list[tuple[int, np.ndarray]]]:
    """Backtest paths of combinatorial purged CV: path p takes, for every group, the p-th split
    (in the order of ``splits``) in which that group is tested. Each path is a list of
    (split index, boolean mask over that split's test observations); together the masks of a
    path cover every observation exactly once."""
    edges = np.linspace(0, n, n_groups + 1).round().astype(int)
    occurrences = {
        g: [si for si, (_, _, combo) in enumerate(splits) if g in combo] for g in range(n_groups)
    }
    paths = []
    for p in range(n_paths):
        parts = []
        for g in range(n_groups):
            si = occurrences[g][p]
            test = splits[si][1]
            parts.append((si, (test >= edges[g]) & (test < edges[g + 1])))
        paths.append(parts)
    return paths


def _predict(F, L, train_obs, test_obs, N, k):
    """(len(test_obs), N) predictions; a model per instrument."""
    P = np.full((len(test_obs), N), np.nan)
    for i in range(N):
        tr = train_obs[np.isfinite(F[train_obs, i]).all(axis=1) & np.isfinite(L[train_obs, i])]
        ok = np.isfinite(F[test_obs, i]).all(axis=1)
        te = test_obs[ok]
        if len(te):
            P[np.flatnonzero(ok), i] = knn_predict(F[tr, i], L[tr, i], F[te, i], k)
    return P


def _scores(P, obs, L, rnext):
    lab = L[obs]
    return spearman(P.ravel(), lab.ravel()), _rank_portfolio_sharpe(P, rnext[obs])


def run_job(spec, cfg, mode, outputs):
    _, kind, seed = spec
    st, _ = market(
        "e5b", cfg[kind]["synth"], seed, n_instruments=cfg["n_instruments"], n_bars=cfg["n_bars"]
    )
    p = st.panel()
    T, N = p.shape
    F, c, bad = _features(p.ret)
    rnext = np.full((T, N), np.nan)
    rnext[:-1] = p.ret[1:]
    S, G, k = cfg["sample"], cfg["gap"], cfg["knn_k"]
    rows = []
    for h in cfg["horizons"]:
        L = _labels(c, bad, h, T)
        obs = np.arange(63, S - h)
        ho = np.arange(S + G, T - h)
        held_s, held_sr = _scores(_predict(F, L, obs, ho, N, k), ho, L, rnext)
        base = {
            "kind": kind,
            "seed": seed,
            "h": h,
            "heldout_spearman": held_s,
            "heldout_sharpe": held_sr,
        }
        for scheme in cfg["schemes"]:
            rng = stream(market_seed("e5b", seed) + h, "cv")
            sp = make_splits(scheme, obs.size, cfg["folds"], h, allow_unpurged=True, rng=rng)
            P = np.full((obs.size, N), np.nan)
            for train, test in sp.folds:
                P[test] = _predict(F, L, obs[train], obs[test], N, k)
            tested = np.isfinite(P).any(axis=1)
            s_, sr = _scores(P[tested], obs[tested], L, rnext)
            rows.append(
                {
                    **base,
                    "scheme": scheme,
                    "label": sp.label,
                    "path": -1,
                    "cv_spearman": s_,
                    "cv_sharpe": sr,
                    "bias_spearman": s_ - held_s,
                    "bias_sharpe": sr - held_sr,
                }
            )
        splits, n_paths = cpcv(obs.size, cfg["cpcv_groups"], cfg["cpcv_test_groups"], h)
        preds = []
        for train, test, _ in splits:
            preds.append(_predict(F, L, obs[train], obs[test], N, k))
        for path, parts in enumerate(cpcv_paths(obs.size, cfg["cpcv_groups"], splits, n_paths)):
            P = np.full((obs.size, N), np.nan)
            for si, sel in parts:
                P[splits[si][1][sel]] = preds[si][sel]
            s_, sr = _scores(P, obs, L, rnext)
            rows.append(
                {
                    **base,
                    "scheme": "cpcv",
                    "label": "purged",
                    "path": path,
                    "cv_spearman": s_,
                    "cv_sharpe": sr,
                    "bias_spearman": s_ - held_s,
                    "bias_sharpe": sr - held_sr,
                }
            )
    return rows


def finalize(cfg, rows, results_dir, outputs_dir, mode):
    files = {}
    p = os.path.join(results_dir, "rows.csv")
    write_rows(p, rows, ["kind", "seed", "h", "scheme", "path"])
    files["rows.csv"] = p
    agg = []
    schemes = list(cfg["schemes"]) + ["cpcv"]
    for kind in ("null", "planted"):
        for h in cfg["horizons"]:
            for scheme in schemes:
                sel = [
                    r for r in rows if r["kind"] == kind and r["h"] == h and r["scheme"] == scheme
                ]
                if not sel:
                    continue
                # one estimate per market: the mean over CPCV paths, or the single CV estimate
                by_seed: dict = {}
                for r in sel:
                    by_seed.setdefault(int(r["seed"]), []).append(r)
                d = {
                    "kind": kind,
                    "h": h,
                    "scheme": scheme,
                    "label": sel[0]["label"],
                    "markets": len(by_seed),
                }
                for m in ("spearman", "sharpe"):
                    est = [float(np.mean([x[f"cv_{m}"] for x in v])) for v in by_seed.values()]
                    bias = [float(np.mean([x[f"bias_{m}"] for x in v])) for v in by_seed.values()]
                    d[f"bias_{m}_mean"], d[f"bias_{m}_ci95"], _ = mean_ci(bias)
                    d[f"bias_{m}_sd"] = float(np.std(bias, ddof=1)) if len(bias) > 1 else math.nan
                    d[f"abs_bias_{m}_mean"], _, _ = mean_ci([abs(x) for x in bias])
                    d[f"cv_{m}_sd_across_markets"] = (
                        float(np.std(est, ddof=1)) if len(est) > 1 else math.nan
                    )
                    if scheme == "cpcv":
                        within = [
                            float(np.std([x[f"cv_{m}"] for x in v], ddof=1))
                            for v in by_seed.values()
                            if len(v) > 1
                        ]
                        d[f"cv_{m}_sd_within_market_paths"] = (
                            float(np.mean(within)) if within else math.nan
                        )
                agg.append(d)
    ap = os.path.join(results_dir, "aggregates.csv")
    write_rows(ap, agg, ["kind", "h", "scheme"])
    files["aggregates.csv"] = ap
    summary = {}
    for kind in ("null", "planted"):
        for h in cfg["horizons"]:
            sel = [d for d in agg if d["kind"] == kind and d["h"] == h]
            if sel:
                for m in ("spearman", "sharpe"):
                    closest = min(sel, key=lambda d: d[f"abs_bias_{m}_mean"])
                    noisiest = max(sel, key=lambda d: d[f"bias_{m}_sd"])
                    summary[f"{kind}_h{h}_{m}"] = {
                        "closest_to_heldout": closest["scheme"],
                        "noisiest": noisiest["scheme"],
                    }
    jp = os.path.join(results_dir, "aggregates.json")
    write_json(jp, summary)
    files["aggregates.json"] = jp
    return files, {}
