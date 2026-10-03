"""Command line: ``python -m quant_research_engine <command>``.

synth       generate a synthetic store (Parquet) and optionally its CSV tables
validate    check a CSV dataset, a store directory, or an export against the schemas
backtest    run one strategy on one store with the event engine; write and validate the logs
experiment  run the experiments (E1 to E5 and the Tier 2 experiments), full or --quick
audit       run every guard on the canaries and the built-in strategies and features
report      Markdown + PNG report of one backtest (refuses an UNSAFE run)
export      write export v1 (the files the decision layer reads)
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def cmd_synth(a) -> int:
    from quant_research_engine.store import write_store
    from quant_research_engine.store.csvdump import write_store_csv
    from quant_research_engine.synth import generate, load_synth_config

    cfg = load_synth_config(a.config)
    over = {k: v for k, v in (("n_instruments", a.n_instruments), ("n_bars", a.n_bars)) if v}
    if over:
        cfg = cfg.with_overrides(**over)
    st, _ = generate(cfg, a.seed)
    m = write_store(a.out, st)
    if a.csv:
        write_store_csv(st, a.csv)
    _print(
        {
            "store": a.out,
            "rows": {k: v["row_count"] for k, v in m["files"].items()},
            "seed": a.seed,
            "config": cfg.name,
        }
    )
    return 0


def cmd_validate(a) -> int:
    from quant_research_engine.export import validate_path

    rep = validate_path(a.path)
    _print(rep)
    return 0 if rep.get("ok") else 1


def _load_strategy(spec: str, params: dict, lint: bool):
    """``module:Class`` or ``path/to/file.py:Class``; a file is linted first (guard G3)."""
    import importlib
    import importlib.util

    target, cls = spec.rsplit(":", 1)
    if target.endswith(".py"):
        if lint:
            from quant_research_engine.guards.lint import lint_file

            issues = lint_file(target)
            if issues:
                for i in issues:
                    print(f"G3 {target}:{i.line}: {i.message}", file=sys.stderr)
                raise SystemExit(2)
        name = os.path.splitext(os.path.basename(target))[0]
        sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module(target)
    return getattr(mod, cls)(**params)


def cmd_backtest(a) -> int:
    from quant_research_engine.backtest import RunConfig, run_strategy
    from quant_research_engine.costs import CostConfig
    from quant_research_engine.data.manifest import read_json
    from quant_research_engine.engine import EngineConfig
    from quant_research_engine.engine.logs import write_run_logs
    from quant_research_engine.metrics import summary
    from quant_research_engine.store import read_store
    from quant_research_engine.validation.validator import validate_run

    st = read_store(a.store)
    strat = _load_strategy(a.strategy, json.loads(a.params), lint=not a.no_lint)
    costs = CostConfig.model_validate(read_json(a.costs)) if a.costs else CostConfig()
    eng = EngineConfig(
        fill_model=a.fill_model,
        unsafe_same_bar_fill=a.unsafe_same_bar_fill,
        fill_delay_bars=a.fill_delay,
        initial_cash=a.initial_cash,
        sizing=a.sizing,
        costs=costs,
        allow_short=not a.long_only,
    )
    rc = RunConfig(
        engine=eng,
        rebalance_every=a.rebalance_every,
        warmup_bars=a.warmup,
        strategy_seed=a.strategy_seed,
    )
    out = run_strategy(st, strat, rc)
    r = out.result
    meta = {
        "store": os.path.basename(os.path.normpath(a.store)),
        "store_seed": st.meta.get("seed"),
        "strategy": a.strategy,
        "params": json.loads(a.params),
        "rebalance_every": a.rebalance_every,
        "warmup_bars": a.warmup,
        "ppy": st.ppy,
        "data": st.meta.get("data", ""),
    }
    run = write_run_logs(r, eng, a.out, meta, strategy_id=getattr(strat, "name", "strategy"))
    rep = validate_run(a.out, st)
    e = r.equity
    ret = e[a.warmup + 1 :] / e[a.warmup : -1] - 1.0
    s = summary(ret, e[a.warmup :], st.ppy)
    _print(
        {
            "label": run["label"],
            "out": a.out,
            "fills": len(r.fills),
            "rejected": len(r.rejected),
            "max_identity_residual": r.max_residual,
            "validator_ok": rep.ok,
            "metrics": s,
            "note": "backtested on synthetic data with assumed parameters",
        }
    )
    return 0 if rep.ok else 1


def cmd_experiment(a) -> int:
    from quant_research_engine.experiments.runner import ALL, run

    ids = [x for x in (a.only.split(",") if a.only else ALL) if x]
    for i in ids:
        if i not in ALL:
            print(f"unknown experiment {i!r}", file=sys.stderr)
            return 2
    res = run(ids, "quick" if a.quick else "full", a.workers, out=a.out)
    _print(res)
    return 0


def cmd_audit(a) -> int:
    from quant_research_engine.guards.audit import all_subjects, run_audit, write_table
    from quant_research_engine.synth import generate, load_synth_config

    cfg = load_synth_config(a.config)
    subjects = all_subjects(a.canaries)
    stores = [(s, generate(cfg, s)[0]) for s in range(a.seed, a.seed + a.stores)]
    res = run_audit(subjects, stores, a.instants, a.warmup)
    write_table(a.out, res)
    rows = [r.row() for r in res]
    bad = [r[0] for r in rows if r[3] == "-" and ("caught" in r[4:11] or "error" in r[4:11])]
    errors = [r[0] for r in rows if "error" in r[4:11]]
    missed = [
        r[0] for r in res if r.subject.named_guard and r.cells[r.subject.named_guard] != "caught"
    ]
    _print(
        {
            "table": a.out,
            "subjects": len(rows),
            "honest_alarms": bad,
            "canaries_missed": missed,
            "errors": errors,
        }
    )
    return 0 if not bad and not missed and not errors else 1


def cmd_report(a) -> int:
    from quant_research_engine.reports import UnsafeRunError, make_report

    try:
        out = make_report(a.run, a.out, allow_unsafe=a.allow_unsafe, store_path=a.store)
    except UnsafeRunError as e:
        print(str(e), file=sys.stderr)
        return 3
    _print(out)
    return 0


def cmd_export(a) -> int:
    from quant_research_engine.export import export_store

    out = export_store(a, a.out)
    _print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m quant_research_engine",
        description="Point-in-time research engine (synthetic data; no live trading).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="generate a synthetic store")
    s.add_argument("--config", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--csv", help="also write the tables as CSV files into this directory")
    s.add_argument("--n-instruments", type=int)
    s.add_argument("--n-bars", type=int)
    s.set_defaults(fn=cmd_synth)
    v = sub.add_parser("validate", help="validate a CSV file, a store, or an export directory")
    v.add_argument("path")
    v.set_defaults(fn=cmd_validate)
    b = sub.add_parser("backtest", help="run one strategy on one store (event engine)")
    b.add_argument("--store", required=True)
    b.add_argument("--strategy", required=True, help="module:Class or path/to/file.py:Class")
    b.add_argument("--params", default="{}")
    b.add_argument("--out", required=True)
    b.add_argument("--rebalance-every", type=int, default=5)
    b.add_argument("--warmup", type=int, default=63)
    b.add_argument(
        "--fill-model",
        default="auto",
        choices=["auto", "next_open", "next_close", "same_close"],
        help="auto: next_open for equity_daily, next_close for crypto_daily",
    )
    b.add_argument("--unsafe-same-bar-fill", action="store_true")
    b.add_argument("--fill-delay", type=int, default=1)
    b.add_argument("--initial-cash", type=float, default=1e7)
    b.add_argument("--sizing", default="equity", choices=["equity", "fixed_capital"])
    b.add_argument("--costs", help="cost model JSON (default: the base model)")
    b.add_argument("--long-only", action="store_true")
    b.add_argument("--strategy-seed", type=int, default=0)
    b.add_argument("--no-lint", action="store_true", help="skip the G3 lint of a strategy file")
    b.set_defaults(fn=cmd_backtest)
    e = sub.add_parser("experiment", help="run the experiments")
    e.add_argument("action", choices=["run"])
    e.add_argument("--quick", action="store_true")
    e.add_argument(
        "--only", help="comma-separated experiment ids (e1..e5, e5b, t2_capacity, t2_trials)"
    )
    e.add_argument("--workers", type=int, default=2)
    e.add_argument(
        "--out", help="output root (default: experiments/results or experiments/outputs/quick)"
    )
    e.set_defaults(fn=cmd_experiment)
    u = sub.add_parser("audit", help="run every guard on canaries and built-in subjects")
    u.add_argument("--canaries", help="directory of canary modules (for example tests/canaries)")
    u.add_argument("--config", default=os.path.join("configs", "synth", "small.json"))
    u.add_argument("--seed", type=int, default=1)
    u.add_argument("--stores", type=int, default=1)
    u.add_argument("--instants", type=int, default=50)
    u.add_argument("--warmup", type=int, default=20)
    u.add_argument("--out", default=os.path.join("outputs", "audit", "audit_table.csv"))
    u.set_defaults(fn=cmd_audit)
    r = sub.add_parser("report", help="report of one backtest run")
    r.add_argument("--run", required=True, help="directory written by backtest")
    r.add_argument("--store", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--allow-unsafe", action="store_true")
    r.set_defaults(fn=cmd_report)
    x = sub.add_parser("export", help="write export v1")
    x.add_argument("--config", help="synthetic configuration (with --seed)")
    x.add_argument("--seed", type=int)
    x.add_argument("--store", help="an existing store directory instead of --config")
    x.add_argument("--n-instruments", type=int)
    x.add_argument("--n-bars", type=int)
    x.add_argument("--derived-only", action="store_true")
    x.add_argument("--terms", help="licence and terms note of a third-party source")
    x.add_argument("--out", required=True)
    x.set_defaults(fn=cmd_export)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)
