"""The committed sample run in docs/examples/ is reproduced by the README command, stays below
100 KB, and passes the independent validator (check 9).

The regenerated files are compared with the committed ones by ``helpers.diff_files``: the CSV
logs cell by cell and the JSON files as parsed objects, floats at ``math.isclose(rel_tol=1e-9,
abs_tol=1e-9)`` (so round-off-sized values such as ``max_identity_residual`` cannot fail on
noise), everything else exactly, without ``content_sha256`` and without the hash table ``files``
of ``run.json``. The fills use ``sigma_bar`` in the impact term, and ``sigma_bar`` can differ in
the last digit between math libraries, so a byte comparison could fail on another CPU."""

import os

from helpers import diff_files

from quant_research_engine.cli import main
from quant_research_engine.store import write_store
from quant_research_engine.synth import generate, load_synth_config
from quant_research_engine.validation.validator import validate_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "docs", "examples", "signal_x_ls")
LOOSE = {"rel_tol": 1e-9, "abs_tol": 1e-9, "ignore": {"content_sha256", "files"}}


def run_example(tmp_path) -> tuple[str, object]:
    """The README's backtest command on the example store; returns (run directory, store)."""
    st, _ = generate(load_synth_config(os.path.join(ROOT, "configs", "synth", "example.json")), 1)
    store = str(tmp_path / "example")
    write_store(store, st)
    out = str(tmp_path / "run")
    args = [
        "backtest",
        "--store",
        store,
        "--strategy",
        "quant_research_engine.strategies.signal_x_ls:SignalXLS",
        "--warmup",
        "30",
        "--rebalance-every",
        "5",
        "--initial-cash",
        "1000000",
        "--out",
        out,
    ]
    assert main(args) == 0
    return out, st


def test_example_is_reproduced_and_validated(tmp_path):
    out, st = run_example(tmp_path)
    names = sorted(os.listdir(EXAMPLE))
    assert names == sorted(os.listdir(out))
    diffs = []
    total = 0
    for f in names:
        diffs += diff_files(os.path.join(EXAMPLE, f), os.path.join(out, f), **LOOSE)
        total += os.path.getsize(os.path.join(EXAMPLE, f))
    assert diffs == [], diffs[:5]
    assert total < 100_000
    rep = validate_run(EXAMPLE, st)
    assert rep.ok, rep.errors[:3]
