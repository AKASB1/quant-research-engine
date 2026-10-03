"""E5: validation schemes against a held-out segment.

Markets of 1533 bars: a 1260-bar sample, a 21-bar gap, a 252-bar held-out segment. Model:
k-nearest-neighbour regression (k = 25) fitted per instrument (distances only within one
instrument's observations) on standardized features (trailing compounded returns over 5, 21, 63
bars and the trailing volatility of 21 bars; standardized with the training observations of the
instrument) predicting the forward compounded return over h bars. Score: Spearman correlation
of predictions and realized labels, pooled over instruments. Schemes (K = 5): shuffled K-fold
and contiguous K-fold without purge (both through allow_unpurged, labelled UNPURGED), purged
K-fold with embargo h, walk-forward. Held-out score: the model fitted on every sample observation
whose label is realized inside the sample, scored on the held-out segment. Bias = cross-
validation score minus held-out score. The shuffled row on null markets measures canary C8.
"""

from __future__ import annotations

import math
import os

import numpy as np
from scipy.stats import rankdata

from quant_research_engine.data.manifest import write_json
from quant_research_engine.experiments.common import mean_ci, write_rows
from quant_research_engine.experiments.markets import market, market_seed
from quant_research_engine.inference.splitters import make_splits
from quant_research_engine.rng import stream


def jobs(cfg, mode):
    out = []
    for kind in ("null", "planted"):
        a, b = cfg[kind]["seeds"]
        out += [("market", kind, s) for s in range(a, b + 1)]
    return out


def sort_key(r):
    return (r["kind"], int(r["seed"]), int(r["h"]), r["scheme"])


def _features(r: np.ndarray):
    """(T, N, 4) features and the cumulative log-return panel."""
    lr = np.log1p(r)
    c = np.full((r.shape[0] + 1, r.shape[1]), np.nan)
    c[0] = 0.0
    c[1:] = np.cumsum(np.nan_to_num(lr), axis=0)
    bad = np.cumsum(np.isnan(lr), axis=0)
    bad = np.vstack([np.zeros((1, r.shape[1])), bad])
    T = r.shape[0]
    F = np.full((T, r.shape[1], 4), np.nan)
    for j, L in enumerate((5, 21, 63)):
        k = np.arange(L, T)
        ok = bad[k + 1] - bad[k + 1 - L] == 0
        F[L:, :, j] = np.where(ok, np.expm1(c[k + 1] - c[k + 1 - L]), np.nan)
    for k in range(21, T):
        w = r[k - 20 : k + 1]
        F[k, :, 3] = np.where(np.isnan(w).any(axis=0), np.nan, np.std(w, axis=0, ddof=1))
    return F, c, bad


def _labels(c, bad, h, T):
    """Label of observation i: compounded return of bars i+1 .. i+h."""
    L = np.full((T, c.shape[1]), np.nan)
    k = np.arange(0, T - h)
    ok = bad[k + 1 + h] - bad[k + 1] == 0
    L[k] = np.where(ok, np.expm1(c[k + 1 + h] - c[k + 1]), np.nan)
    return L


def knn_predict(Xtr, ytr, Xte, k):
    if len(ytr) == 0 or len(Xte) == 0:
        return np.full(len(Xte), np.nan)
    m = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    a = (Xtr - m) / sd
    b = (Xte - m) / sd
    d = ((b[:, None, :] - a[None, :, :]) ** 2).sum(axis=2)
    kk = min(k, len(ytr))
    idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    return ytr[idx].mean(axis=1)


def spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return math.nan
    a = rankdata(x[ok])
    b = rankdata(y[ok])
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def run_job(spec, cfg, mode, outputs):
    _, kind, seed = spec
    st, _ = market(
        "e5", cfg[kind]["synth"], seed, n_instruments=cfg["n_instruments"], n_bars=cfg["n_bars"]
    )
    p = st.panel()
    T, N = p.shape
    F, c, bad = _features(p.ret)
    S, G = cfg["sample"], cfg["gap"]
    k = cfg["knn_k"]
    rows = []
    for h in cfg["horizons"]:
        L = _labels(c, bad, h, T)
        obs = np.arange(63, S - h)  # label realized inside the sample: i + h <= S - 1
        ho = np.arange(S + G, T - h)
        # held-out score
        preds, labs = [], []
        for i in range(N):
            tr = obs[np.isfinite(F[obs, i]).all(axis=1) & np.isfinite(L[obs, i])]
            te = ho[np.isfinite(F[ho, i]).all(axis=1) & np.isfinite(L[ho, i])]
            preds.append(knn_predict(F[tr, i], L[tr, i], F[te, i], k))
            labs.append(L[te, i])
        held = spearman(np.concatenate(preds), np.concatenate(labs))
        for scheme in cfg["schemes"]:
            rng = stream(market_seed("e5", seed) + h, "cv")
            sp = make_splits(scheme, obs.size, cfg["folds"], h, allow_unpurged=True, rng=rng)
            preds, labs = [], []
            for train, test in sp.folds:
                for i in range(N):
                    trn = obs[train]
                    tst = obs[test]
                    trn = trn[np.isfinite(F[trn, i]).all(axis=1) & np.isfinite(L[trn, i])]
                    tst = tst[np.isfinite(F[tst, i]).all(axis=1) & np.isfinite(L[tst, i])]
                    preds.append(knn_predict(F[trn, i], L[trn, i], F[tst, i], k))
                    labs.append(L[tst, i])
            cv = spearman(np.concatenate(preds), np.concatenate(labs))
            rows.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "h": h,
                    "scheme": scheme,
                    "label": sp.label,
                    "cv_score": cv,
                    "heldout_score": held,
                    "bias": cv - held,
                }
            )
    return rows


def finalize(cfg, rows, results_dir, outputs_dir, mode):
    files = {}
    p = os.path.join(results_dir, "rows.csv")
    write_rows(p, rows, ["kind", "seed", "h", "scheme"])
    files["rows.csv"] = p
    agg = []
    for kind in ("null", "planted"):
        for h in cfg["horizons"]:
            for scheme in cfg["schemes"]:
                sel = [
                    r for r in rows if r["kind"] == kind and r["h"] == h and r["scheme"] == scheme
                ]
                if not sel:
                    continue
                d = {"kind": kind, "h": h, "scheme": scheme, "label": sel[0]["label"]}
                for key in ("bias", "cv_score", "heldout_score"):
                    d[f"{key}_mean"], d[f"{key}_ci95"], d["markets"] = mean_ci(
                        [r[key] for r in sel]
                    )
                d["bias_sd"] = (
                    float(np.std([r["bias"] for r in sel], ddof=1)) if len(sel) > 1 else math.nan
                )
                d["cv_score_sd"] = (
                    float(np.std([r["cv_score"] for r in sel], ddof=1))
                    if len(sel) > 1
                    else math.nan
                )
                agg.append(d)
    ap = os.path.join(results_dir, "aggregates.csv")
    write_rows(ap, agg, ["kind", "h", "scheme"])
    files["aggregates.csv"] = ap
    jp = os.path.join(results_dir, "aggregates.json")
    write_json(
        jp,
        {
            "c8_shuffled_bias_null": [
                {"h": d["h"], "bias_mean": d["bias_mean"], "bias_ci95": d["bias_ci95"]}
                for d in agg
                if d["kind"] == "null" and d["scheme"] == "shuffled_kfold"
            ]
        },
    )
    files["aggregates.json"] = jp
    return files, {}
