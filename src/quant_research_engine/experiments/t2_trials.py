"""Tier 2, item 4: the effective number of trials and multiple-testing corrections.

On the markets of E3 (the same seeds and the same derived market seeds, gross panel, the 108
trials), compare five decision rules for "the selected strategy is significant": the naive PSR(0)
of the selected trial at 0.95; the DSR with N = 108; the DSR with N = N_eff, the eigenvalue
participation ratio of the trials' return correlation matrix, (sum lambda)^2 / sum lambda^2;
Holm's step-down and Benjamini-Hochberg's step-up procedures at level 0.05 on the per-trial
one-sided p-values 1 - PSR(0) (a discovery counts when any trial is rejected). Under the null
every discovery is a false positive; on planted markets a discovery counts as power when a
signal_x trial is among the rejected (for Holm and BH) or is the selected trial (for the
DSR-type rules). The p-values ignore the dependence between trials, which Holm tolerates and BH
assumes away; the report says so.
"""

from __future__ import annotations

import os

import numpy as np

from quant_research_engine.costs import zero_costs
from quant_research_engine.data.manifest import write_json
from quant_research_engine.experiments import e3
from quant_research_engine.experiments.common import wilson, write_rows
from quant_research_engine.experiments.markets import market
from quant_research_engine.inference.sharpe import dsr, psr
from quant_research_engine.metrics import kurt, sharpe, skew

ALPHA = 0.05


def jobs(cfg, mode):
    out = []
    for kind in ("null", "planted"):
        a, b = cfg[kind]["seeds"]
        out += [("market", kind, s) for s in range(a, b + 1)]
    return out


def sort_key(r):
    return (r["kind"], int(r["seed"]))


def n_eff(R: np.ndarray) -> float:
    C = np.corrcoef(R.T)
    C = np.nan_to_num(C)
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0.0, None)
    return float(lam.sum() ** 2 / (lam * lam).sum())


def holm(p: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    rej = np.zeros(p.size, dtype=bool)
    m = p.size
    for rank, j in enumerate(order):
        if p[j] <= alpha / (m - rank):
            rej[j] = True
        else:
            break
    return rej


def bh(p: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    m = p.size
    thr = alpha * (np.arange(1, m + 1) / m)
    ok = p[order] <= thr
    rej = np.zeros(m, dtype=bool)
    if ok.any():
        kmax = int(np.flatnonzero(ok).max())
        rej[order[: kmax + 1]] = True
    return rej


def run_job(spec, cfg, mode, outputs):
    _, kind, seed = spec
    ecfg = cfg["e3"]
    trials = e3.trial_list(ecfg)
    st, _ = market(
        "e3", ecfg[kind]["synth"], seed, n_instruments=ecfg["n_instruments"], n_bars=ecfg["n_bars"]
    )
    R, _ = e3.returns_matrix(st, ecfg, trials, zero_costs())
    T = R.shape[0]
    srs = np.asarray([sharpe(R[:, j]) for j in range(R.shape[1])])
    sk = np.asarray([skew(R[:, j]) for j in range(R.shape[1])])
    ku = np.asarray([kurt(R[:, j]) for j in range(R.shape[1])])
    finite = np.isfinite(srs)
    sel = int(np.argmax(np.where(finite, srs, -np.inf)))
    V = float(np.var(srs[finite], ddof=1))
    ne = n_eff(R[:, finite])
    p = np.asarray(
        [1.0 - psr(srs[j], 0.0, T, sk[j], ku[j]) if finite[j] else 1.0 for j in range(R.shape[1])]
    )
    rh, rb = holm(p, ALPHA), bh(p, ALPHA)
    is_x = np.asarray([t["feature"] == "signal_x_ewma" for t in trials])
    return [
        {
            "kind": kind,
            "seed": seed,
            "n_trials": len(trials),
            "n_eff": ne,
            "naive_psr_sig": bool(1.0 - p[sel] > 0.95),
            "dsr_n108_sig": bool(dsr(srs[sel], T, sk[sel], ku[sel], len(trials), V) > 0.95),
            "dsr_neff_sig": bool(
                dsr(srs[sel], T, sk[sel], ku[sel], max(2, int(round(ne))), V) > 0.95
            ),
            "holm_any": bool(rh.any()),
            "bh_any": bool(rb.any()),
            "holm_signal_x": bool((rh & is_x).any()),
            "bh_signal_x": bool((rb & is_x).any()),
            "selected_signal_x": bool(is_x[sel]),
            "holm_rejections": int(rh.sum()),
            "bh_rejections": int(rb.sum()),
        }
    ]


def finalize(cfg, rows, results_dir, outputs_dir, mode):
    files = {}
    p = os.path.join(results_dir, "rows.csv")
    write_rows(p, rows, ["kind", "seed"])
    files["rows.csv"] = p
    agg = []
    for kind in ("null", "planted"):
        s = [r for r in rows if r["kind"] == kind]
        if not s:
            continue
        n = len(s)
        d = {
            "kind": kind,
            "markets": n,
            "n_eff_mean": float(np.mean([r["n_eff"] for r in s])),
            "n_eff_min": float(np.min([r["n_eff"] for r in s])),
            "n_eff_max": float(np.max([r["n_eff"] for r in s])),
        }
        rules = {
            "naive_psr": "naive_psr_sig",
            "dsr_n108": "dsr_n108_sig",
            "dsr_neff": "dsr_neff_sig",
            "holm": "holm_any",
            "bh": "bh_any",
        }
        for name, key in rules.items():
            if kind == "null":
                k = sum(1 for r in s if r[key])
            elif name in ("holm", "bh"):
                k = sum(1 for r in s if r[f"{name}_signal_x"])
            else:
                k = sum(1 for r in s if r[key] and r["selected_signal_x"])
            d[f"{name}_share"], d[f"{name}_lo"], d[f"{name}_hi"] = wilson(k, n)
        agg.append(d)
    ap = os.path.join(results_dir, "aggregates.csv")
    write_rows(ap, agg, ["kind"])
    files["aggregates.csv"] = ap
    write_json(
        os.path.join(results_dir, "aggregates.json"),
        {
            "alpha": ALPHA,
            "note": "null: share of markets with a discovery (false positive); planted: "
            "share with a discovery that involves a signal_x trial (power)",
        },
    )
    files["aggregates.json"] = os.path.join(results_dir, "aggregates.json")
    return files, {}
