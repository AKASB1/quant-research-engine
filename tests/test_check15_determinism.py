"""Check 15: byte-identical result files (outside the wall_ columns) for two runs in one process,
for two fresh processes, and for a serial and a parallel run; different seeds differ; removing
trials changes no other trial's returns; aggregates do not depend on the order of the rows."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from quant_research_engine.experiments import e1, e3
from quant_research_engine.experiments.common import deterministic_bytes
from quant_research_engine.experiments.markets import market
from quant_research_engine.experiments.runner import load_config, run_one
from quant_research_engine.rng import stream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny_root(tmp, e1_seeds=(1, 6)):
    full = load_config("e3", "quick", ROOT)
    cfg = {
        "e1": {**load_config("e1", "quick", ROOT), "seeds": list(e1_seeds), "keep_logs": 0},
        "e3": {
            **full,
            "null": {"synth": "null_e3", "seeds": [1, 2]},
            "planted": {"synth": "planted", "seeds": [3, 3]},
            "n_instruments": 20,
            "n_bars": 420,
            "bootstrap": {"p": 0.1, "draws": 50},
            "lo": {**full["lo"], "series_per_phi": 3, "n_bars": 300},
            "log_one_trial_per_panel": False,
        },
    }
    d = os.path.join(tmp, "experiments", "configs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "quick.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return tmp


def _det(path):
    if path.endswith(".json"):
        obj = json.load(open(path, encoding="utf-8"))
        if isinstance(obj, dict):
            obj = {k: v for k, v in obj.items() if not k.startswith("wall_")}
        return json.dumps(obj, sort_keys=True).encode()
    return deterministic_bytes(path)


def _files(out, exp):
    d = os.path.join(out, exp)
    return {
        f: _det(os.path.join(d, f))
        for f in sorted(os.listdir(d))
        if f.endswith((".csv", ".json")) and f != "manifest.json"
    }


def test_two_runs_in_one_process_are_identical(tmp_path):
    root = tiny_root(str(tmp_path / "root"))
    run_one("e1", "quick", 1, root, str(tmp_path / "a"))
    run_one("e1", "quick", 1, root, str(tmp_path / "b"))
    assert _files(str(tmp_path / "a"), "e1") == _files(str(tmp_path / "b"), "e1")


@pytest.mark.slow
def test_serial_and_parallel_runs_are_identical(tmp_path):
    """The concurrency test of the parallel runner (run 20 times in verification)."""
    root = tiny_root(str(tmp_path / "root"))
    run_one("e1", "quick", 1, root, str(tmp_path / "s"))
    run_one("e1", "quick", 2, root, str(tmp_path / "p"))
    assert _files(str(tmp_path / "s"), "e1") == _files(str(tmp_path / "p"), "e1")


@pytest.mark.slow
def test_fresh_processes_are_identical_and_e3_too(tmp_path):
    root = tiny_root(str(tmp_path / "root"))
    code = (
        "import sys; from quant_research_engine.experiments.runner import run_one;"
        "run_one(sys.argv[1], 'quick', 1, sys.argv[2], sys.argv[3])"
    )
    env = dict(os.environ, PYTHONUTF8="1")
    for exp in ("e1", "e3"):
        for k in ("x", "y"):
            subprocess.run(
                [sys.executable, "-c", code, exp, root, str(tmp_path / k)],
                check=True,
                timeout=900,
                env=env,
            )
        assert _files(str(tmp_path / "x"), exp) == _files(str(tmp_path / "y"), exp)


def test_different_seeds_differ(tmp_path):
    a = tiny_root(str(tmp_path / "ra"), (1, 4))
    b = tiny_root(str(tmp_path / "rb"), (5, 8))
    run_one("e1", "quick", 1, a, str(tmp_path / "a"))
    run_one("e1", "quick", 1, b, str(tmp_path / "b"))
    assert (
        _files(str(tmp_path / "a"), "e1")["rows.csv"]
        != _files(str(tmp_path / "b"), "e1")["rows.csv"]
    )


def test_removing_trials_changes_no_other_trial():
    cfg = load_config("e3", "quick", ROOT)
    cfg = {**cfg, "n_instruments": 20, "n_bars": 420}
    trials = e3.trial_list(cfg)
    st, _ = market("e3", "planted", 3, n_instruments=20, n_bars=420)
    from quant_research_engine.costs import CostConfig

    full, _ = e3.returns_matrix(st, cfg, trials, CostConfig())
    sub_idx = list(range(0, len(trials), 3))
    sub, _ = e3.returns_matrix(st, cfg, [trials[i] for i in sub_idx], CostConfig())
    assert np.array_equal(full[:, sub_idx], sub, equal_nan=True)


def test_aggregates_do_not_depend_on_row_order(tmp_path):
    cfg = load_config("e1", "quick", ROOT)
    rows = []
    for s in range(1, 5):
        rows += e1.run_job(("scenario", s), cfg, "quick", str(tmp_path / "o"))
    for r in rows:
        r["mode"] = "quick"
    perm = stream(0, "test.shuffle").permutation(len(rows))
    shuffled = [rows[i] for i in perm]
    for name, rr in (("a", rows), ("b", shuffled)):
        rr = sorted(rr, key=e1.sort_key)
        os.makedirs(tmp_path / name, exist_ok=True)
        e1.finalize(cfg, rr, str(tmp_path / name), str(tmp_path / name), "quick")
    assert _det(str(tmp_path / "a" / "rows.csv")) == _det(str(tmp_path / "b" / "rows.csv"))
    assert _det(str(tmp_path / "a" / "aggregates.json")) == _det(
        str(tmp_path / "b" / "aggregates.json")
    )
