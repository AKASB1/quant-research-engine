"""E1: engine and accounting audit on random scenarios (stream e1.scenario).

Each seed is one scenario (2 to 12 instruments, 50 to 300 bars, splits, dividends, delistings,
late listings, zero-volume bars, shorts on or off, lot sizes, fill delays 1 to 3). Both engines
run on it; their logs are written and checked by the independent validator; the row records the
largest relative differences of fills and equity, the identity residuals, the validator result,
and wall-clock times (``wall_`` columns, excluded from every deterministic comparison).
Scenarios in which the participation cap binds or an order is rejected are outside the
equivalence regime: they are excluded from the comparison and listed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time

from quant_research_engine.engine.logs import write_run_logs
from quant_research_engine.experiments.common import mean_ci, write_rows
from quant_research_engine.validation.equivalence import (
    compare,
    run_event_targets,
    run_vector_targets,
    scenario_config,
    write_vector_logs,
)
from quant_research_engine.validation.scenarios import make_scenario
from quant_research_engine.validation.validator import validate_run

FIELDS = ("quantity", "ref_price", "price", "spread", "impact", "commission")


def jobs(cfg, mode):
    a, b = cfg["seeds"]
    return [("scenario", s) for s in range(a, b + 1)]


def sort_key(r):
    return (int(r["seed"]),)


def run_job(spec, cfg, mode, outputs):
    _, seed = spec
    sc = make_scenario(seed, 0)
    ecfg = scenario_config(sc)
    p = sc.store.panel()
    t0 = time.perf_counter()
    ev = run_event_targets(sc.store, sc.dec_idx, sc.targets, ecfg)
    t1 = time.perf_counter()
    vr = run_vector_targets(sc.store, sc.dec_idx, sc.targets, ecfg)
    t2 = time.perf_counter()
    excluded = bool(ev.cap_binds or ev.rejected)
    row = {
        "seed": seed,
        "n_instruments": p.shape[1],
        "n_bars": p.shape[0],
        "fill_delay": sc.fill_delay,
        "rebalance_every": sc.rebalance_every,
        "allow_short": sc.allow_short,
        "excluded": excluded,
        "cap_binds": ev.cap_binds,
        "rejected": len(ev.rejected),
        "n_fills_event": len(ev.fills),
        "residual_event": ev.max_residual,
        "residual_vector": vr.max_residual,
    }
    row.update({f"n_{k}": v for k, v in sc.features.items()})
    if not excluded:
        c = compare(ev, vr, p, ecfg.fill_model)
        row["fills_match"] = c["fills_match"]
        for f in FIELDS:
            row[f"max_diff_{f}"] = c.get(f"max_diff_{f}", 0.0)
        row["max_diff_equity"] = c["max_diff_equity"]
    keep = seed < int(cfg["seeds"][0]) + int(cfg.get("keep_logs", 0))
    base = os.path.join(outputs, "logs", str(seed)) if keep else tempfile.mkdtemp(prefix="qre-e1-")
    try:
        write_run_logs(ev, ecfg, os.path.join(base, "event"), {"seed": seed})
        rep_e = validate_run(os.path.join(base, "event"), sc.store)
        row["validator_event_ok"] = rep_e.ok
        row["validator_event_max_rel"] = rep_e.max_rel_error
        if not excluded:
            write_vector_logs(vr, p, sc.dec_idx, ecfg, os.path.join(base, "vector"))
            rep_v = validate_run(os.path.join(base, "vector"), sc.store)
            row["validator_vector_ok"] = rep_v.ok
            row["validator_vector_max_rel"] = rep_v.max_rel_error
    finally:
        if not keep:
            shutil.rmtree(base, ignore_errors=True)
    row["wall_event_s"] = t1 - t0
    row["wall_vector_s"] = t2 - t1
    row["wall_instrument_bars"] = p.shape[0] * p.shape[1]
    return [row]


def finalize(cfg, rows, results_dir, outputs_dir, mode):
    path = os.path.join(results_dir, "rows.csv")
    write_rows(path, rows, ["seed"])
    inc = [r for r in rows if not r["excluded"]]
    agg = {
        "scenarios": len(rows),
        "compared": len(inc),
        "excluded": [int(r["seed"]) for r in rows if r["excluded"]],
        "fills_mismatch": sum(1 for r in inc if not r["fills_match"]),
        "validator_failures_event": sum(1 for r in rows if not r["validator_event_ok"]),
        "validator_failures_vector": sum(1 for r in inc if not r.get("validator_vector_ok", True)),
        "max_identity_residual_event": max(r["residual_event"] for r in rows),
        "max_identity_residual_vector": max(r["residual_vector"] for r in rows),
        "max_validator_rel_error": max(
            max(r["validator_event_max_rel"], r.get("validator_vector_max_rel", 0.0)) for r in rows
        ),
    }
    for f in (*FIELDS, "equity"):
        agg[f"max_diff_{f}"] = max((r.get(f"max_diff_{f}", 0.0) for r in inc), default=0.0)
    for k in ("splits", "dividends", "delistings", "late_listings", "zero_volume_bars", "lots"):
        agg[f"total_{k}"] = sum(int(r[f"n_{k}"]) for r in rows)
        agg[f"scenarios_with_{k}"] = sum(1 for r in rows if int(r[f"n_{k}"]) > 0)
    for d in (1, 2, 3):
        agg[f"scenarios_fill_delay_{d}"] = sum(1 for r in rows if int(r["fill_delay"]) == d)
    agg["scenarios_shorts_on"] = sum(1 for r in rows if r["allow_short"])
    agg["violations"] = (
        agg["fills_mismatch"] + agg["validator_failures_event"] + agg["validator_failures_vector"]
    )
    ib = sum(r["wall_instrument_bars"] for r in inc)
    agg["wall_throughput_event_instrument_bars_per_s"] = (
        ib / sum(r["wall_event_s"] for r in inc) if inc else None
    )
    agg["wall_throughput_vector_instrument_bars_per_s"] = (
        ib / sum(r["wall_vector_s"] for r in inc) if inc else None
    )
    m, ci, n = mean_ci([r["wall_event_s"] for r in inc])
    agg["wall_event_s_mean"] = m
    apath = os.path.join(results_dir, "aggregates.json")
    from quant_research_engine.data.manifest import write_json

    write_json(apath, agg)
    return {"rows.csv": path, "aggregates.json": apath}, {"aggregates": agg}
