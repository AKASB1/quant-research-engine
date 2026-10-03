"""Check 14: the synthetic generator (engine part in test_check14_engine_part.py).

Standard errors are the standard deviation of per-seed estimates divided by sqrt(#seeds), over
20 development seeds; a per-seed correlation is the Pearson correlation pooled over all
instruments and bars (or periods) of the seed.
"""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from quant_research_engine.data.manifest import read_json
from quant_research_engine.store import write_store
from quant_research_engine.synth import close_to_close_returns, generate, load_synth_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = list(range(1, 21))
_CACHE: dict = {}


def cfg(name, **over):
    c = load_synth_config(os.path.join(ROOT, "configs", "synth", f"{name}.json"))
    return c.with_overrides(**over) if over else c


def gen(name, seed, **over):
    key = (name, seed, tuple(sorted(over.items())))
    if key not in _CACHE:
        _CACHE[key] = generate(cfg(name, **over), seed)
    return _CACHE[key]


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def within(estimates, target, k=4.0):
    est = np.asarray(estimates)
    m = est.mean()
    se = est.std(ddof=1) / math.sqrt(len(est))
    assert abs(m - target) <= k * se, f"mean {m:.5f} target {target:.5f} se {se:.5f}"
    return m, se


def planted_ic(store_truth):
    _, tr = store_truth
    alive = ~np.isnan(tr.r)
    alive[0] = False
    prev = np.zeros_like(tr.s)
    prev[1:] = tr.s[:-1]
    return pearson(prev[alive], tr.idio_std[alive])


def test_store_bytes_identical_across_fresh_processes(tmp_path):
    code = (
        "import sys; from quant_research_engine.synth import generate, load_synth_config;"
        "from quant_research_engine.store import write_store;"
        "c=load_synth_config(sys.argv[1]); s,_=generate(c, 7); write_store(sys.argv[2], s)"
    )
    paths = []
    for k in range(2):
        out = str(tmp_path / f"s{k}")
        env = dict(os.environ, PYTHONUTF8="1")
        subprocess.run(
            [sys.executable, "-c", code, os.path.join(ROOT, "configs", "synth", "small.json"), out],
            check=True,
            timeout=300,
            env=env,
        )
        paths.append(out)
    a, b = (read_json(os.path.join(p, "store.json")) for p in paths)
    assert a == b
    for name in a["files"]:
        with (
            open(os.path.join(paths[0], f"{name}.parquet"), "rb") as f0,
            open(os.path.join(paths[1], f"{name}.parquet"), "rb") as f1,
        ):
            assert f0.read() == f1.read()
    # in-process generation writes the same bytes too
    s, _ = generate(cfg("small"), 7)
    m = write_store(str(tmp_path / "s2"), s)
    assert m["files"] == a["files"]


def test_common_random_numbers_and_seed_dependence():
    a, _ = generate(cfg("small"), 3)
    b, _ = generate(cfg("small"), 3)
    c, _ = generate(cfg("small"), 4)
    assert np.array_equal(a.tables["bars"]["close"], b.tables["bars"]["close"])
    assert not np.array_equal(a.panel().close[:50], c.panel().close[:50], equal_nan=True)


@pytest.mark.slow
def test_planted_ic_and_null():
    within([planted_ic(gen("planted", s)) for s in SEEDS], 0.02)
    within([planted_ic(gen("null_e3", s)) for s in SEEDS], 0.0)


@pytest.mark.slow
def test_signal_x_correlation_with_latent():
    target = 1 / math.sqrt(1 + 1.0**2)
    est = []
    for s in SEEDS:
        _, tr = gen("planted", s)
        alive = ~np.isnan(tr.r)
        est.append(pearson(tr.x[alive], tr.s[alive]))
    m, _ = within(est, target)
    assert abs(m - 0.707) < 0.01


def test_ohlc_prices_returns_and_delistings():
    for s in SEEDS[:5]:
        st, tr = gen("small", s)
        b = st.tables["bars"]
        assert (b["low"] > 0).all()
        assert (b["high"] >= np.maximum(b["open"], b["close"])).all()
        assert (b["low"] <= np.minimum(b["open"], b["close"])).all()
        p = st.panel()
        diff = np.abs(p.ret - tr.r)
        assert np.nanmax(diff) <= 1e-12
        ins = st.tables["instruments"]
        for i, iid in enumerate(st.ids):
            if ins["ts_delist"][i] != -1:
                a, z = st._bar_slices[iid]
                assert ins["ts_delist"][i] == b["ts_event"][z - 1]
            assert ins["ts_list"][i] == b["ts_event"][st._bar_slices[iid][0]]


@pytest.mark.slow
def test_event_counts_consistent_with_rates():
    c = cfg("planted")
    ppy = c.ppy
    obs_split = exp_split = obs_del = exp_del = 0.0
    payers = []
    for s in SEEDS:
        st, tr = gen("planted", s)
        ca = st.tables["corporate_actions"]
        obs_split += float((ca["action"] == "split").sum())
        lead = c.events.announce_lead_bars
        elig = np.maximum(0, tr.last - (tr.listing + lead + 1) + 1)
        exp_split += float(elig.sum()) * c.events.split_rate_annual / ppy
        p = c.events.delist_hazard_annual / ppy
        n_el = np.maximum(0, c.n_bars - (tr.listing + c.events.min_life_bars))
        exp_del += float((1 - (1 - p) ** n_el).sum())
        obs_del += float((st.tables["instruments"]["ts_delist"] != -1).sum())
        div_ids = {
            str(x)
            for x, a in zip(ca["instrument_id"], ca["action"], strict=True)
            if a == "cash_dividend"
        }
        payers.append(len(div_ids) / c.n_instruments)
    assert abs(obs_split - exp_split) <= 4 * math.sqrt(exp_split)
    assert abs(obs_del - exp_del) <= 4 * math.sqrt(exp_del)
    within(payers, c.events.dividend_share)


@pytest.mark.slow
def test_regime_occupancy():
    c = cfg("planted")
    target = c.regime.p_calm_to_stress / (c.regime.p_calm_to_stress + c.regime.p_stress_to_calm)
    est = [float((gen("planted", s)[1].v > 1).mean()) for s in SEEDS]
    within(est, target)


@pytest.mark.slow
def test_shift_moves_vol_correlation_and_ic():
    c = cfg("shift")
    a = c.shift.at_bar
    vol_r, corr_r, ic_after = [], [], []
    for s in SEEDS:
        st, tr = gen("shift", s)
        r = tr.r
        full = ~np.isnan(r).any(axis=0)
        rr = r[1:, full]
        pre, post = rr[: a - 1], rr[a - 1 :]
        vol_r.append(float(np.median(post.std(axis=0) / pre.std(axis=0))))

        def avg_corr(x):
            cm = np.corrcoef(x.T)
            n = cm.shape[0]
            return (cm.sum() - n) / (n * (n - 1))

        corr_r.append(avg_corr(post) / avg_corr(pre))
        alive = ~np.isnan(tr.r)
        alive[:a] = False
        prev = np.zeros_like(tr.s)
        prev[1:] = tr.s[:-1]
        ic_after.append(pearson(prev[alive], tr.idio_std[alive]))
    # medians of per-instrument ratios: Student-t tails and regimes make a 10 % band reasonable
    assert abs(np.mean(vol_r) - c.shift.vol_mult) <= 0.1 * c.shift.vol_mult
    within(corr_r, c.shift.corr_mult)
    within(ic_after, c.ic * c.shift.ic_mult)


@pytest.mark.slow
def test_fundamental_law_bound_on_planted_clean():
    c = cfg("planted_clean")
    N = c.n_instruments
    rets = []
    for s in SEEDS:
        _, tr = gen("planted_clean", s)
        w = tr.s / (tr.sigma[None, :] * math.sqrt(N))
        rets.append(close_to_close_returns(w, tr.r))
    pooled = np.concatenate(rets)
    sr = pooled.mean() / pooled.std(ddof=1)
    target = c.ic * math.sqrt(N)
    se = math.sqrt((1 + 0.5 * sr * sr) / len(pooled))
    # band 10 % of the target = about 4.5 standard errors of the pooled estimate
    assert abs(sr - target) <= 0.10 * target, (sr, target, se)
    assert 0.10 * target / se > 3


@pytest.mark.slow
def test_fund_x_releases_vintages_and_correlation():
    c = cfg("planted_clean")
    fx = c.fund_x
    est, pred = [], []
    for s in SEEDS:
        st, tr = gen("planted_clean", s)
        p = st.panel()
        ser = st.tables["series"]
        v0s, z0s = [], []
        for i, iid in enumerate(st.ids):
            a, z = st._series_slices[f"{iid}.fund_x"]
            te, ta, vin, val = (ser[k][a:z] for k in ("ts_event", "ts_avail", "vintage", "value"))
            for j in range(len(te)):
                pe = int(np.searchsorted(p.ts_event, te[j]))
                assert (pe + 1) % fx.period_bars == 0
                lag = fx.lag0_bars if vin[j] == 0 else fx.lag1_bars
                assert ta[j] == p.ts_event[pe + lag]
                if vin[j] == 1:
                    rr = p.ret[pe + fx.lag0_bars + 1 : pe + fx.lag1_bars + 1, i]
                    n1 = fx.lag1_bars - fx.lag0_bars
                    z1 = math.fsum(rr.tolist()) / (tr.sigma[i] * math.sqrt(n1))
                    assert abs((val[j] - val[j - 1]) - fx.w * z1) <= 1e-12 * max(1.0, abs(val[j]))
                else:
                    rr = p.ret[pe + 1 : pe + fx.lag0_bars + 1, i]
                    z0s.append(math.fsum(rr.tolist()) / (tr.sigma[i] * math.sqrt(fx.lag0_bars)))
                    v0s.append(val[j])
        v0s, z0s = np.asarray(v0s), np.asarray(z0s)
        est.append(pearson(v0s, z0s))
        sd = z0s.std()
        pred.append(fx.w * sd / math.sqrt(1 + fx.w**2 * sd**2))
    m, se = within(np.asarray(est) - np.asarray(pred), 0.0)
    assert abs(np.mean(est) - 0.894) < 0.02
