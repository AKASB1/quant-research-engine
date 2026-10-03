"""Microbenchmarks: throughput of both engines and as-of queries per second.

python scripts/microbench.py [--reps 5] [--out experiments/outputs/microbench.json]

Wall-clock numbers depend on the machine and its load: each case runs ``reps`` times and the
minimum is reported, with CPU time next to wall time and the sampled CPU load before and after.
They are never used in a deterministic comparison and are not committed results.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from quant_research_engine.engine import EngineConfig
from quant_research_engine.engine.quantity import round_lot
from quant_research_engine.experiments.common import cpu_load_percent, hardware, software
from quant_research_engine.experiments.markets import market
from quant_research_engine.store.asof import asof_vintage
from quant_research_engine.store.duck import asof_series_duckdb, connect
from quant_research_engine.validation.equivalence import run_event_targets, run_vector_targets


def timed(fn, reps):
    best_w, best_c = float("inf"), float("inf")
    for _ in range(reps):
        w0, c0 = time.perf_counter(), time.process_time()
        fn()
        best_w = min(best_w, time.perf_counter() - w0)
        best_c = min(best_c, time.process_time() - c0)
    return best_w, best_c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join("experiments", "outputs", "microbench.json"))
    a = ap.parse_args(argv)
    load0 = cpu_load_percent()
    st, tr = market("bench", "planted", 1, n_instruments=60, n_bars=1260)
    p = st.panel()
    T, N = p.shape
    out = {"hardware": hardware(), "software": software(), "reps": a.reps, "cases": {}}
    for every in (1, 5):
        dec = np.arange(1, T, every)
        te = p.ts_event[dec][:, None]
        uni = (
            (p.ts_list[None, :] <= te)
            & ((p.ts_delist[None, :] == -1) | (p.ts_delist[None, :] > te))
            & np.isfinite(p.close[dec])
        )
        w = np.nan_to_num(tr.s[dec]) / N
        tg = np.where(uni, round_lot(w * 1e7 / np.where(uni, p.close[dec], 1.0), 0.0), 0.0)
        ec = EngineConfig(initial_cash=1e7)
        for name, fn in (("event", run_event_targets), ("vector", run_vector_targets)):
            wall, cpu = timed(lambda fn=fn, dec=dec, tg=tg, ec=ec: fn(st, dec, tg, ec), a.reps)
            out["cases"][f"{name}_every{every}"] = {
                "wall_s": wall,
                "cpu_s": cpu,
                "bars": T,
                "instruments": N,
                "bars_per_s": T / wall,
                "instrument_bars_per_s": T * N / wall,
            }
    ser = st.tables["series"]
    cols = ser.columns
    ts = p.ts_event[::25]
    wall, cpu = timed(lambda: [asof_vintage(cols, int(t)) for t in ts], a.reps)
    out["cases"]["asof_vintage_full_table"] = {
        "wall_s": wall,
        "cpu_s": cpu,
        "queries": len(ts),
        "rows": len(ser),
        "queries_per_s": len(ts) / wall,
    }
    con = connect(st)
    try:
        wall, cpu = timed(lambda: [asof_series_duckdb(con, int(t)) for t in ts[:10]], a.reps)
    finally:
        con.close()
    out["cases"]["asof_duckdb_full_table"] = {
        "wall_s": wall,
        "cpu_s": cpu,
        "queries": 10,
        "rows": len(ser),
        "queries_per_s": 10 / wall,
    }
    sid = f"{p.ids[0]}.fund_x"
    wall, cpu = timed(lambda: [st.series(sid, knowledge=int(t)) for t in p.ts_event], a.reps)
    out["cases"]["store_series_one_id"] = {
        "wall_s": wall,
        "cpu_s": cpu,
        "queries": T,
        "queries_per_s": T / wall,
    }
    out["cpu_load_percent_before"] = load0
    out["cpu_load_percent_after"] = cpu_load_percent()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out["cases"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
