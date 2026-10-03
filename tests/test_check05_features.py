"""Check 5 (feature side): truncation invariance of every panel feature (bit for bit, 100 random
bars), agreement of the panel and the context implementations (1e-12), and of the strategies'
panel weights with their decisions."""

import os

import numpy as np
import pytest

from quant_research_engine.context import ContextBuilder, empty_portfolio
from quant_research_engine.features import PanelSource, make_feature
from quant_research_engine.rng import stream
from quant_research_engine.store.store import cut_store
from quant_research_engine.strategies import BUILTIN_FEATURES, BUILTIN_STRATEGIES
from quant_research_engine.synth import generate, load_synth_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ST = {}


def small(seed=3):
    if seed not in _ST:
        _ST[seed] = generate(
            load_synth_config(os.path.join(ROOT, "configs", "synth", "small.json")), seed
        )[0]
    return _ST[seed]


@pytest.mark.slow
@pytest.mark.parametrize("fid,name,params", BUILTIN_FEATURES, ids=[f[0] for f in BUILTIN_FEATURES])
def test_panel_feature_is_truncation_invariant(fid, name, params):
    st = small()
    p = st.panel()
    feat = make_feature(name, **params)
    full = feat.panel(PanelSource(st))
    ks = stream(5, "test.trunc").choice(np.arange(1, p.shape[0]), size=100, replace=False)
    checked = 0
    for k in ks.tolist():
        cs = cut_store(st, int(p.ts_event[k]))
        cut = feat.panel(PanelSource(cs))
        cp = cs.panel()
        assert cut.shape[0] == k + 1
        for j, iid in enumerate(cp.ids):
            a = full[k, p.ids.index(iid)]
            b = cut[k, j]
            assert (np.isnan(a) and np.isnan(b)) or a == b, (fid, k, iid, a, b)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("fid,name,params", BUILTIN_FEATURES, ids=[f[0] for f in BUILTIN_FEATURES])
def test_panel_and_context_implementations_agree(fid, name, params):
    st = small()
    p = st.panel()
    feat = make_feature(name, **params)
    panel = feat.panel(PanelSource(st))
    b = ContextBuilder(st, "probe", 0, None, feat.series_suffixes)
    n = 0
    for k in (
        stream(6, "test.ctxpanel").choice(np.arange(5, p.shape[0]), size=40, replace=False).tolist()
    ):
        ctx = b.build(int(p.ts_event[k]), empty_portfolio(1e6))
        v = feat.compute(ctx)
        for j, iid in enumerate(ctx.universe()):
            a, c = v[j], panel[k, p.ids.index(iid)]
            if np.isnan(a) or np.isnan(c):
                assert np.isnan(a) and np.isnan(c), (fid, k, iid, a, c)
            else:
                assert abs(a - c) <= 1e-12 * max(1.0, abs(a), abs(c)), (fid, k, iid, a, c)
                n += 1
    assert n > 0


@pytest.mark.parametrize(
    "sid,mod,cls,params", BUILTIN_STRATEGIES, ids=[s[0] for s in BUILTIN_STRATEGIES]
)
def test_strategy_panel_weights_equal_decisions(sid, mod, cls, params):
    import importlib

    st = small()
    p = st.panel()
    strat = getattr(importlib.import_module(mod), cls)(**params)
    rows = np.arange(80, p.shape[0], 37)
    W = strat.weights_panel(PanelSource(st), rows)
    b = ContextBuilder(st, strat.name, 0, strat.history_bars, strat.series_suffixes)
    for r, k in enumerate(rows.tolist()):
        d = strat.decide(b.build(int(p.ts_event[k]), empty_portfolio(1e6)))
        for j, iid in enumerate(p.ids):
            assert abs(d.weights.get(iid, 0.0) - W[r, j]) <= 1e-12, (sid, k, iid)
