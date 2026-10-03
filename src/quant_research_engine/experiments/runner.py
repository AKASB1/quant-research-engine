"""One-command experiment runner.

``run(ids, mode, workers)`` reads ``experiments/configs/<id>.json`` (full) or
``experiments/configs/quick.json`` (quick: development seeds, small sizes), builds each
experiment's jobs, runs them serially or in a ``spawn`` process pool, sorts the rows by
(experiment, configuration, seed) before writing, and writes the rows, the aggregates, and a
manifest. Full results go to ``experiments/results/<id>/``; quick results and bulky per-run
outputs to the git-ignored ``experiments/outputs/``. Every row of a quick run carries
``mode = quick``.
"""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

from quant_research_engine.experiments.common import cpu_load_percent, manifest, write_manifest

EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5")
# Tier 2 experiments: each adds its own files and never changes what E1 to E5 compute.
TIER2 = ("e5b", "t2_capacity", "t2_trials")
ALL = EXPERIMENTS + TIER2


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def load_config(exp_id: str, mode: str, root: str | None = None) -> dict:
    root = root or repo_root()
    d = os.path.join(root, "experiments", "configs")
    if mode == "quick":
        with open(os.path.join(d, "quick.json"), encoding="utf-8") as f:
            return json.load(f)[exp_id]
    with open(os.path.join(d, f"{exp_id}.json"), encoding="utf-8") as f:
        return json.load(f)


def out_dirs(
    exp_id: str, mode: str, root: str | None = None, out: str | None = None
) -> tuple[str, str]:
    """(results directory, bulky-outputs directory)."""
    root = root or repo_root()
    if out:
        return os.path.join(out, exp_id), os.path.join(out, exp_id, "outputs")
    if mode == "quick":
        base = os.path.join(root, "experiments", "outputs", "quick", exp_id)
        return base, os.path.join(base, "outputs")
    return os.path.join(root, "experiments", "results", exp_id), os.path.join(
        root, "experiments", "outputs", exp_id
    )


def _execute(job):
    exp_id, spec, cfg, mode, outputs = job
    mod = importlib.import_module(f"quant_research_engine.experiments.{exp_id}")
    return mod.run_job(spec, cfg, mode, outputs)


def run_one(
    exp_id: str,
    mode: str = "full",
    workers: int = 1,
    root: str | None = None,
    out: str | None = None,
    only: list | None = None,
) -> dict:
    cfg = load_config(exp_id, mode, root)
    results_dir, outputs_dir = out_dirs(exp_id, mode, root, out)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    mod = importlib.import_module(f"quant_research_engine.experiments.{exp_id}")
    specs = mod.jobs(cfg, mode)
    if only:
        specs = [s for s in specs if s[0] in only]
    jobs = [(exp_id, s, cfg, mode, outputs_dir) for s in specs]
    load_before = cpu_load_percent()
    t0 = time.perf_counter()
    if workers <= 1:
        out_rows = [_execute(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            out_rows = list(ex.map(_execute, jobs, chunksize=1))
    wall = time.perf_counter() - t0
    load_after = cpu_load_percent()
    rows = [r for chunk in out_rows for r in chunk]
    for r in rows:
        r["mode"] = mode
    rows.sort(key=mod.sort_key)
    files, extra = mod.finalize(cfg, rows, results_dir, outputs_dir, mode)
    seeds = sorted({int(r["seed"]) for r in rows if "seed" in r})
    m = manifest(
        exp_id,
        mode,
        cfg,
        seeds,
        workers,
        files,
        extra,
        load={
            "cpu_load_percent_before": load_before,
            "cpu_load_percent_after": load_after,
            "wall_seconds": round(wall, 1),
        },
    )
    write_manifest(os.path.join(results_dir, "manifest.json"), m)
    return {"experiment": exp_id, "rows": len(rows), "wall_seconds": wall, "results": results_dir}


def run(
    ids, mode: str = "full", workers: int = 1, root: str | None = None, out: str | None = None
) -> list[dict]:
    return [run_one(i, mode, workers, root, out) for i in ids]
