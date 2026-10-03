"""Regression tests for the findings of review 1."""

import os
import sys

import numpy as np
import pytest
from audit_helpers import CANARIES, ROOT, SUBJECTS, audit_world
from engine_helpers import tiny_store

from quant_research_engine.context import (
    ContextBuilder,
    Orders,
    OrderSpec,
    TargetWeights,
    empty_portfolio,
)
from quant_research_engine.costs import CostConfig
from quant_research_engine.engine import EngineConfig, EventEngine
from quant_research_engine.engine.logs import write_run_logs
from quant_research_engine.features import PanelSource, signal_x_ewma
from quant_research_engine.guards.canaries import load_subjects
from quant_research_engine.guards.lint import lint_source
from quant_research_engine.guards.replay import Subject, invariance_audit
from quant_research_engine.inference.pbo import pbo_cscv
from quant_research_engine.inference.registry import TrialRegistry
from quant_research_engine.rng import stream
from quant_research_engine.store.builder import build_store
from quant_research_engine.synth import generate, load_synth_config
from quant_research_engine.validation.validator import validate_run

if SUBJECTS not in sys.path:
    sys.path.insert(0, SUBJECTS)


def _subject(mod, cls):
    return Subject(mod, mod, (mod, cls, {}), roots=[SUBJECTS])


# 1a: a handle kept on a class of this package is caught by G2 (and undone)
@pytest.mark.slow
def test_package_state_leak_is_caught_and_undone():
    from quant_research_engine.context import Strategy

    st, inst, pois, rc = audit_world(seed=1, n_instants=10)
    s = {x.id: x for x in load_subjects(CANARIES)}["C1s"]
    r = invariance_audit(s, st, inst, rc, pois)
    assert all(r.caught)
    assert any("Strategy._qre_handle" in c for c in r.state_changes)
    assert "_qre_handle" not in Strategy.__dict__


# 1b: the lint rejects the attribute chain to the store and reflective access
def test_lint_rejects_package_root_imports_and_reflection():
    probe = (
        "import quant_research_engine.context\n"
        "from quant_research_engine.context import Strategy\n"
        "class S(Strategy):\n"
        "    def decide(self, ctx):\n"
        "        Strategy._h = quant_research_engine.store.registry.open_store()\n"
    )
    assert lint_source(probe)
    for src in (
        "x = ctx.__class__\n",
        "f = getattr(ctx, 'bars')\n",
        "d = locals()\n",
        "g = vars(ctx)\n",
        "h = (lambda: 0).__globals__\n",
    ):
        assert lint_source(src), src
    assert lint_source("class A:\n    def __init__(self):\n        super().__init__()\n") == []


# 2: exceptions keep earlier decisions, are compared by bar, and a failing subject is an error
@pytest.mark.slow
def test_subject_failing_everywhere_is_reported_as_error():
    from quant_research_engine.guards.audit import run_audit

    st, inst, pois, rc = audit_world(seed=1, n_instants=10)
    r = invariance_audit(_subject("raises_always", "RaisesAlways"), st, inst, rc, pois)
    assert r.real_error is not None
    res = run_audit([_subject("raises_always", "RaisesAlways")], [(1, st)], 3, 20)
    assert res[0].cells["G2"] == "error"


@pytest.mark.slow
def test_late_exception_gives_no_false_alarm_at_earlier_instants():
    st, inst, pois, rc = audit_world(seed=1, n_instants=10)
    r = invariance_audit(_subject("raises_late", "RaisesLate"), st, inst, rc, pois)
    for k, c in zip(r.instants, r.caught, strict=True):
        assert not c, (k, r.first_diff)


# 3: a zero-variance trial neither wins the selection nor turns PBO, V, SR0 into NaN
def test_zero_variance_trial():
    R = stream(3, "test.zero").standard_normal((480, 5)) * 0.01
    R0 = R.copy()
    R0[:, 2] = 0.0
    a = pbo_cscv(R0)
    assert np.isfinite(a.lambdas).all()
    reg = TrialRegistry("z")
    for j in range(5):
        reg.add(str(j), {"j": j}, R0[:, j])
    assert reg.select() != 2 and np.isfinite(reg.var_sr()) and np.isfinite(reg.sr0())
    assert reg.degenerate == 1


# 4: the validator checks reference prices, delays, volume, and the cap from the bars
def test_validator_catches_wrong_reference_price_and_early_fills(tmp_path, monkeypatch):
    st = tiny_store({"X": [(10.0 + k, 10.5 + k, 1e6) for k in range(6)]}, liquidity=None)
    cfg = EngineConfig(
        initial_cash=1e5, fill_delay_bars=2, costs=CostConfig(impact={"model": "none"})
    )

    def run(out):
        eng = EventEngine(st.panel(), cfg)
        for k in range(6):
            eng.process_bar(k)
            if k == 0:
                eng.submit(TargetWeights({"X": 0.5}), ["X"])
        write_run_logs(eng.result(), cfg, out)
        return validate_run(out, st)

    assert run(str(tmp_path / "ok")).ok
    orig = EventEngine._eligible_fills

    def prev_close(self, k, i, ref=None):  # fills at the previous close (look-ahead in execution)
        out = orig(self, k, i, ref)
        return [(o, q, float(self.p.close[k - 1, i])) for o, q, _ in out]

    monkeypatch.setattr(EventEngine, "_eligible_fills", prev_close)
    assert not run(str(tmp_path / "bad_ref")).ok
    monkeypatch.setattr(EventEngine, "_eligible_fills", orig)
    orig_new = EventEngine._new_order

    def no_delay(self, i, q, tif, target):
        orig_new(self, i, q, tif, target)
        if self.pending[i]:
            self.pending[i][-1].elig_bar -= 1  # one bar too early

    monkeypatch.setattr(EventEngine, "_new_order", no_delay)
    rep = run(str(tmp_path / "early"))
    assert not rep.ok and any("bars after its decision" in e for e in rep.errors)


# 5: unpurged splits made by an undeclared subject are observed (G6)
@pytest.mark.slow
def test_undeclared_unpurged_cv_is_observed():
    st, inst, pois, rc = audit_world(seed=1, n_instants=10)
    r = invariance_audit(_subject("undeclared_cv", "UndeclaredCV"), st, inst, rc, pois)
    assert r.unpurged_splits > 0


# 6: same-close fills share the participation cap with the bar's other fills
def test_same_close_fills_respect_the_cap_once():
    st = tiny_store({"X": [(10.0, 10.0, 1e6), (10.0, 10.0, 0.0), (10.0, 10.0, 1000.0)]})
    cfg = EngineConfig(
        fill_model="same_close",
        unsafe_same_bar_fill=True,
        initial_cash=1e6,
        costs=CostConfig(impact={"model": "none"}),
    )
    eng = EventEngine(st.panel(), cfg)
    eng.process_bar(0)
    eng.process_bar(1)
    eng.submit(Orders((OrderSpec("X", 300.0, "gtc"),)), ["X"])
    eng.fill_same_close()  # zero volume: the gtc order waits
    eng.process_bar(2)  # the carried order fills up to the cap (0.1 x 1000)
    eng.submit(Orders((OrderSpec("X", 300.0, "gtc"),)), ["X"])
    eng.fill_same_close()  # the cap of bar 2 is already used
    k2 = int(st.panel().ts_event[2])
    assert sum(abs(f[4]) for f in eng.fills if f[2] == k2) <= 100.0


# 7: the default fill model follows the calendar
def test_fill_model_auto_follows_the_calendar():
    cfg = EngineConfig()
    assert cfg.resolved("equity_daily").fill_model == "next_open"
    assert cfg.resolved("crypto_daily").fill_model == "next_close"
    c = load_synth_config(os.path.join(ROOT, "configs", "synth", "small.json")).with_overrides(
        calendar="crypto_daily", n_bars=60, n_instruments=4
    )
    st, _ = generate(c, 1)
    eng = EventEngine(st.panel(), EngineConfig(initial_cash=1e5), calendar=st.calendar_name)
    assert eng.cfg.fill_model == "next_close"


# 8: signal_x_ewma agrees between context and panel on a store with a mid-life gap
def test_signal_x_ewma_agrees_with_a_gap():
    st0, _ = generate(
        load_synth_config(os.path.join(ROOT, "configs", "synth", "small.json")).with_overrides(
            n_instruments=5, n_bars=120
        ),
        2,
    )
    b = st0.tables["bars"]
    p0 = st0.panel()
    gap_ts = int(p0.ts_event[60])
    keep = np.flatnonzero(~((b["instrument_id"] == p0.ids[0]) & (b["ts_event"] == gap_ts)))
    bars = b.take(keep)
    ser = st0.tables["series"]
    ks = np.flatnonzero(
        ~((ser["series_id"] == f"{p0.ids[0]}.signal_x") & (ser["ts_event"] == gap_ts))
    )
    st = build_store(
        st0.tables["instruments"], bars, st0.tables["corporate_actions"], ser.take(ks), st0.meta
    )
    p = st.panel()
    f = signal_x_ewma(3)
    panel = f.panel(PanelSource(st))
    bld = ContextBuilder(st, "probe", 0, None, f.series_suffixes)
    for k in range(55, 80):
        ctx = bld.build(int(p.ts_event[k]), empty_portfolio(1.0))
        v = f.compute(ctx)
        for j, iid in enumerate(ctx.universe()):
            a, c = v[j], panel[k, p.ids.index(iid)]
            assert (np.isnan(a) and np.isnan(c)) or abs(a - c) <= 1e-12 * max(1, abs(c)), (
                k,
                iid,
                a,
                c,
            )


# 9: audit worlds are read-only
@pytest.mark.slow
def test_audit_worlds_are_read_only():
    st, inst, pois, rc = audit_world(seed=1, n_instants=10)
    invariance_audit(_subject("honest_reversal", "HonestReversal"), st, inst[:2], rc, pois)
    with pytest.raises(ValueError):
        st.panel().close[0, 0] = 1.0
    with pytest.raises(ValueError):
        st.tables["bars"]["close"][0] = 1.0
