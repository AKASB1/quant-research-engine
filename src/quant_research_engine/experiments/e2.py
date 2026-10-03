"""E2: leakage audit.

- Audit table: every guard on every canary and built-in subject on ``small`` stores, with
  ``instants_per_store`` sampled decision instants per store and subject (stream
  audit.instants); per-instant G2 flags give the detection probability for m sampled instants.
- The three leaks that pay on a null market (C1, C1b, C2) against an honest past-only momentum
  rule with the same weighting rule and schedule (one-feature momentum(21, 5): z-scores scaled to
  gross exposure 1, a decision every bar), on ``null_e3`` markets, event engine, equity sizing,
  zero costs (main) and the base cost model (labelled net). The true Sharpe ratio of the honest
  rule is zero there.
- C7: the built-in signal_x_ls with ``unsafe_same_bar_fill`` (UNSAFE) against the same strategy
  with next_open fills, on ``planted`` markets; the ratio of the Sharpe ratios.
"""

from __future__ import annotations

import importlib
import math
import os
import sys

import numpy as np

from quant_research_engine.backtest import RunConfig, run_strategy
from quant_research_engine.costs import CostConfig, zero_costs
from quant_research_engine.data.manifest import write_json
from quant_research_engine.engine import EngineConfig
from quant_research_engine.experiments.common import mean_ci, paired, write_rows
from quant_research_engine.experiments.markets import market
from quant_research_engine.guards.audit import (
    GUARDS,
    HEADER,
    all_subjects,
    detection_curve,
    run_audit,
)
from quant_research_engine.guards.replay import purge_modules
from quant_research_engine.metrics import sharpe
from quant_research_engine.store import registry


def canary_dir() -> str:
    from quant_research_engine.experiments.runner import repo_root

    return os.path.join(repo_root(), "tests", "canaries")


def jobs(cfg, mode):
    out = []
    a, b = cfg["audit"]["seeds"]
    out += [("audit", s) for s in range(a, b + 1)]
    a, b = cfg["sharpe"]["seeds"]
    out += [("sharpe", s) for s in range(a, b + 1)]
    a, b = cfg["c7"]["seeds"]
    out += [("c7", s) for s in range(a, b + 1)]
    return out


def sort_key(r):
    return (
        r["part"],
        str(r.get("subject_id", "")),
        str(r.get("panel", "")),
        int(r["seed"]),
        int(r.get("instant", -1)),
    )


def _subjects_by_id():
    return {s.id: s for s in all_subjects(canary_dir())}


def fresh_strategy(subject, store):
    """Instantiate a subject inside the store's world with helper modules purged (as the audit
    does), so a canary's memoized handle never points at another market."""
    for r in subject.roots:
        if r not in sys.path:
            sys.path.insert(0, r)
    with registry.world(store):
        purge_modules(subject.spec[0], subject.roots, set(sys.modules))
        mod = importlib.import_module(subject.spec[0])
        return getattr(mod, subject.spec[1])(**dict(subject.spec[2] or {}))


def _run(store, strat, warmup, cash, costs, fill_model="next_open"):
    eng = EngineConfig(
        initial_cash=cash,
        costs=costs,
        fill_model=fill_model,
        unsafe_same_bar_fill=fill_model == "same_close",
    )
    out = run_strategy(store, strat, RunConfig(engine=eng, warmup_bars=warmup, rebalance_every=1))
    e = out.result.equity
    r = e[warmup + 1 :] / e[warmup:-1] - 1.0
    return r, out.result


def _validate_sample(st, res, c, costs, out):
    """Quick mode: write the logs of one run and confirm them with the independent validator."""
    from quant_research_engine.engine.logs import write_run_logs
    from quant_research_engine.validation.validator import validate_run

    eng = EngineConfig(initial_cash=c["initial_cash"], costs=costs)
    write_run_logs(res, eng, out, {"seed": int(st.meta.get("seed") or 0)})
    rep = validate_run(out, st)
    if not rep.ok:
        raise AssertionError(f"validator failed on the E2 sample log: {rep.errors[:3]}")


def run_job(spec, cfg, mode, outputs):
    part, seed = spec
    rows = []
    if part == "audit":
        a = cfg["audit"]
        st, _ = market("e2", a["synth"], seed)
        res = run_audit(
            all_subjects(canary_dir()), [(seed, st)], a["instants_per_store"], a["warmup"]
        )
        for r in res:
            static = {g: r.cells[g] for g in GUARDS if g != "G2" and g != "G7"}
            for _s, k, c in r.g2_flags:
                rows.append(
                    {
                        "part": "audit",
                        "seed": seed,
                        "subject_id": r.subject.id,
                        "name": r.subject.name,
                        "kind": r.subject.kind,
                        "named_guard": r.subject.named_guard or "-",
                        "instant": k,
                        "g2_caught": c,
                        "g7": r.cells["G7"],
                        **static,
                    }
                )
        return rows
    if part == "sharpe":
        c = cfg["sharpe"]
        st, _ = market("e2", c["synth"], seed, n_instruments=c["n_instruments"], n_bars=c["n_bars"])
        subs = _subjects_by_id()
        for sid in c["subjects"]:
            for panel, costs in (("gross", zero_costs()), ("net", CostConfig())):
                if sid == "honest":
                    from quant_research_engine.strategies.base import FeatureOnly

                    strat = FeatureOnly(
                        feature="momentum", feature_params={"lookback": 21, "skip": 5}
                    )
                else:
                    strat = fresh_strategy(subs[sid], st)
                r, res = _run(st, strat, c["warmup"], c["initial_cash"], costs)
                if mode == "quick" and seed == c["seeds"][0] and sid == "honest" and panel == "net":
                    _validate_sample(
                        st, res, c, costs, os.path.join(outputs, "logs", f"sharpe_{seed}_honest")
                    )
                rows.append(
                    {
                        "part": "sharpe",
                        "seed": seed,
                        "subject_id": sid,
                        "panel": panel,
                        "sharpe_ann": sharpe(r, 252),
                        "sr_bar": sharpe(r),
                        "mean_return": float(np.mean(r)),
                        "bars": len(r),
                        "max_identity_residual": res.max_residual,
                    }
                )
        return rows
    c = cfg["c7"]
    st, _ = market("e2", c["synth"], seed, n_instruments=c["n_instruments"], n_bars=c["n_bars"])
    from quant_research_engine.strategies.signal_x_ls import SignalXLS

    for panel, costs in (("gross", zero_costs()), ("net", CostConfig())):
        for fm in ("same_close", "next_open"):
            r, res = _run(st, SignalXLS(), c["warmup"], c["initial_cash"], costs, fill_model=fm)
            rows.append(
                {
                    "part": "c7",
                    "seed": seed,
                    "subject_id": "C7" if fm == "same_close" else "signal_x_ls",
                    "panel": panel,
                    "fill_model": fm,
                    "label": "UNSAFE" if fm == "same_close" else "safe",
                    "sharpe_ann": sharpe(r, 252),
                    "sr_bar": sharpe(r),
                    "bars": len(r),
                    "max_identity_residual": res.max_residual,
                }
            )
    return rows


def finalize(cfg, rows, results_dir, outputs_dir, mode):
    files = {}
    audit = [r for r in rows if r["part"] == "audit"]
    p = os.path.join(results_dir, "audit_instants.csv")
    write_rows(p, audit, ["part", "seed", "subject_id"])
    files["audit_instants.csv"] = p
    # audit table: one row per subject
    subj = {}
    for r in audit:
        s = subj.setdefault(r["subject_id"], {"rows": [], "meta": r})
        s["rows"].append(r)
    table = []
    curves = []
    for sid in sorted(subj, key=lambda x: (x[0] != "C", x)):
        s = subj[sid]
        m = s["meta"]
        flags = [
            bool(x["g2_caught"])
            for x in sorted(s["rows"], key=lambda x: (int(x["seed"]), int(x["instant"])))
        ]
        cells = {g: m[g] for g in GUARDS if g not in ("G2", "G7")}
        cells["G2"] = "caught" if any(flags) else "passed"
        cells["G7"] = "caught" if any(x["g7"] == "caught" for x in s["rows"]) else "passed"
        table.append(
            [
                sid,
                m["name"],
                m["kind"],
                m["named_guard"],
                *[cells[g] for g in GUARDS],
                str(len(flags)),
                str(sum(flags)),
            ]
        )
        curve = detection_curve(flags, tuple(cfg["curve_m"]), cfg["curve_draws"], seed=0)
        for mm, prob in curve.items():
            curves.append(
                {
                    "subject_id": sid,
                    "m": mm,
                    "detection_probability": prob,
                    "instants_pooled": len(flags),
                }
            )
    tp = os.path.join(results_dir, "audit_table.csv")
    with open(tp, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(HEADER) + "\n" + "\n".join(",".join(x) for x in table) + "\n")
    files["audit_table.csv"] = tp
    cp = os.path.join(results_dir, "detection.csv")
    write_rows(cp, curves, ["subject_id", "m"])
    files["detection.csv"] = cp
    sh = [r for r in rows if r["part"] in ("sharpe", "c7")]
    sp = os.path.join(results_dir, "sharpe_rows.csv")
    write_rows(sp, sh, ["part", "seed", "subject_id", "panel"])
    files["sharpe_rows.csv"] = sp
    agg = []
    for part in ("sharpe", "c7"):
        for panel in ("gross", "net"):
            sel = [r for r in sh if r["part"] == part and r["panel"] == panel]
            by = {}
            for r in sel:
                by.setdefault(r["subject_id"], {})[int(r["seed"])] = r["sharpe_ann"]
            ref = by.get("honest" if part == "sharpe" else "signal_x_ls", {})
            for sid in sorted(by):
                m_, ci, n = mean_ci(list(by[sid].values()))
                d_, dci, _ = paired(ref, by[sid])
                row = {
                    "part": part,
                    "panel": panel,
                    "subject_id": sid,
                    "sharpe_ann_mean": m_,
                    "sharpe_ann_ci95": ci,
                    "n_markets": n,
                    "diff_vs_reference_mean": d_,
                    "diff_vs_reference_ci95": dci,
                }
                if part == "c7" and sid == "C7":
                    ratios = [
                        by[sid][k] / ref[k] for k in sorted(set(by[sid]) & set(ref)) if ref[k] != 0
                    ]
                    rm, rci, _ = mean_ci(ratios)
                    row["sharpe_ratio_vs_next_open_mean"] = rm
                    row["sharpe_ratio_vs_next_open_ci95"] = rci
                    row["ratio_of_mean_sharpe"] = (
                        m_ / mean_ci(list(ref.values()))[0] if ref else math.nan
                    )
                agg.append(row)
    ap = os.path.join(results_dir, "sharpe_aggregates.csv")
    write_rows(ap, agg, ["part", "panel", "subject_id"])
    files["sharpe_aggregates.csv"] = ap
    jp = os.path.join(results_dir, "aggregates.json")
    write_json(
        jp,
        {
            "subjects": len(table),
            "caught_by_named_guard": {
                x[0]: x[4 + GUARDS.index(x[3])] == "caught" for x in table if x[3] != "-"
            },
            "honest_alarms": [x[0] for x in table if x[3] == "-" and "caught" in x[4:11]],
        },
    )
    files["aggregates.json"] = jp
    return files, {}
