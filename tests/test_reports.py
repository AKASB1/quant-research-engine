"""The report generator writes Markdown and PNG figures, and refuses an UNSAFE run (G5)."""

import os

import pytest

from quant_research_engine.cli import main
from quant_research_engine.reports import UnsafeRunError, make_report
from quant_research_engine.store import write_store
from quant_research_engine.synth import generate, load_synth_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("rep")
    cfg = load_synth_config(os.path.join(ROOT, "configs", "synth", "small.json")).with_overrides(
        n_bars=200
    )
    st, _ = generate(cfg, 2)
    store = str(d / "store")
    write_store(store, st)
    base = [
        "backtest",
        "--store",
        store,
        "--strategy",
        "quant_research_engine.strategies.signal_x_ls:SignalXLS",
        "--warmup",
        "40",
        "--rebalance-every",
        "5",
    ]
    assert main(base + ["--out", str(d / "safe")]) == 0
    assert (
        main(
            base
            + ["--out", str(d / "unsafe"), "--fill-model", "same_close", "--unsafe-same-bar-fill"]
        )
        == 0
    )
    return d, store


def test_report_for_a_safe_run(runs):
    d, store = runs
    out = make_report(str(d / "safe"), str(d / "report"), store_path=store)
    assert os.path.exists(out["report"]) and len(out["figures"]) == 6
    text = open(out["report"], encoding="utf-8").read()
    assert "UNSAFE" not in text and "synthetic data" in text


def test_report_refuses_an_unsafe_run_unless_allowed(runs):
    d, store = runs
    with pytest.raises(UnsafeRunError):
        make_report(str(d / "unsafe"), str(d / "r2"), store_path=store)
    assert (
        main(["report", "--run", str(d / "unsafe"), "--store", store, "--out", str(d / "r3")]) == 3
    )
    out = make_report(str(d / "unsafe"), str(d / "r4"), allow_unsafe=True, store_path=store)
    text = open(out["report"], encoding="utf-8").read()
    assert text.startswith("> **UNSAFE") and text.rstrip().endswith("**")
