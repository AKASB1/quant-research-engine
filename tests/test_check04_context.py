"""Check 4: the context (exact). Random stores (with delayed bars) and random calls: no accessor
returns a row with ts_avail after t; arrays are read-only; no reference to the store."""

import gc
import os

import numpy as np
import pytest

from quant_research_engine.context import ContextBuilder, empty_portfolio
from quant_research_engine.data.csvio import Table
from quant_research_engine.rng import stream
from quant_research_engine.store.builder import build_store
from quant_research_engine.store.store import Panel, Store
from quant_research_engine.synth import generate, load_synth_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOUR = 3_600_000_000


def delayed_store(seed: int) -> Store:
    """A small generated store whose bars become available 0 to 30 hours after ts_event."""
    cfg = load_synth_config(os.path.join(ROOT, "configs", "synth", "small.json")).with_overrides(
        n_instruments=6, n_bars=80
    )
    st, _ = generate(cfg, seed)
    b = st.tables["bars"]
    rng = stream(seed, "test.delay")
    cols = dict(b.columns)
    cols["ts_avail"] = cols["ts_event"] + rng.integers(0, 31, size=len(b)) * HOUR
    bars = Table(b.schema, cols)
    ser = st.tables["series"]
    return build_store(st.tables["instruments"], bars, st.tables["corporate_actions"], ser, st.meta)


def _reachable(obj, limit=5000):
    seen, out, todo = set(), [], [obj]
    while todo and len(seen) < limit:
        o = todo.pop()
        if id(o) in seen or isinstance(o, type | type(os)):
            continue
        seen.add(id(o))
        out.append(o)
        if isinstance(o, np.ndarray):
            continue
        todo.extend(gc.get_referents(o))
    return out


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_context_is_knowledge_bounded(seed):
    st = delayed_store(seed)
    p = st.panel()
    rng = stream(seed, "test.ctx")
    b = ContextBuilder(st, "probe", 0, None, (".signal_x", ".fund_x"))
    for _ in range(25):
        k = int(rng.integers(0, p.shape[0]))
        t = int(p.ts_event[k]) + int(rng.integers(-30, 30)) * HOUR
        ctx = b.build(t, empty_portfolio(1e6))
        ids = list(ctx.universe())
        assert ids == st.instruments_at(t)
        for n in (-3, 0, 1, 5, 10**6):
            for field in ("open", "high", "low", "close", "volume"):
                vals, te = ctx.bars(field, n)
                assert vals.shape == (len(te), len(ids))
                assert not vals.flags.writeable and not te.flags.writeable
                with pytest.raises(ValueError):
                    vals[...] = 0.0
                assert (te <= t).all()
                if n <= 0:
                    assert len(te) == 0
            # every non-NaN cell is a bar known at t, with the store's value
            vals, te = ctx.bars("close", n)
            for j, iid in enumerate(ids):
                kb = st.bars([iid], 10**6, knowledge=t)[iid]
                known = dict(zip(kb["ts_event"].tolist(), kb["close"].tolist(), strict=True))
                for r_, e in enumerate(te.tolist()):
                    v = vals[r_, j]
                    if np.isnan(v):
                        assert e not in known
                    else:
                        assert known[e] == v
            rv, rte = ctx.returns(n)
            assert rv.shape == (len(rte), len(ids))
            ac, ate = ctx.adjusted_close(n, total_return=False)
            ref = st.adjusted_close(ids, 10**6, knowledge=t)
            for j, iid in enumerate(ids):
                d = dict(zip(ref[iid][0].tolist(), ref[iid][1].tolist(), strict=True))
                for r_, e in enumerate(ate.tolist()):
                    if not np.isnan(ac[r_, j]):
                        assert ac[r_, j] == d[e]
        for iid in ids:
            for suf in (".signal_x", ".fund_x"):
                sid = iid + suf
                ref = st.series(sid, knowledge=t)
                if sid in b._ser:
                    te, val = ctx.series(sid)
                    assert te.tolist() == ref["ts_event"].tolist()
                    assert val.tolist() == ref["value"].tolist()
        with pytest.raises(KeyError):
            ctx.series("unknown.series")
        adv, sig = ctx.liquidity()
        liq = st.liquidity(knowledge=t)
        lmap = {
            str(i): (a, s)
            for i, a, s in zip(
                liq["instrument_id"], liq["adv_shares"], liq["sigma_bar"], strict=True
            )
        }
        for j, iid in enumerate(ids):
            if iid in lmap:
                assert (adv[j], sig[j]) == lmap[iid]
            else:
                assert np.isnan(adv[j])
        # no reference to the store, the panel, or any array that is not a cut copy
        for o in _reachable(ctx):
            assert not isinstance(o, Store | Panel | ContextBuilder | Table)
            if isinstance(o, np.ndarray):
                assert o.base is None
                if o.ndim == 2:
                    assert o.shape[0] <= int(np.searchsorted(p.ts_event, t, side="right"))


def test_rng_is_the_strategy_stream():
    st = delayed_store(5)
    ctx = ContextBuilder(st, "s", 42, None, ()).build(
        int(st.panel().ts_event[30]), empty_portfolio(1.0)
    )
    a = ctx.rng("noise").random(3)
    b = stream(42, "strategy.s.noise").random(3)
    assert a.tolist() == b.tolist()
