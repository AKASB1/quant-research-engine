"""Synthetic market generator (shared contract, section 7; every number is an assumed value).

Per bar and instrument the simple total return is ``r = beta' f + e + c``: factors ``f`` with
annual volatilities and means, idiosyncratic ``e = sigma_i v_t (ic s_{t-1} + sqrt(1 - ic^2) z)``
with a latent AR(1) signal ``s`` (persistence ``phi``), a two-state volatility regime ``v_t``,
and a constant premium ``c``. Every unit shock is drawn as ``sqrt(g) a + sqrt(1 - g) b`` with
independent ``a`` (overnight) and ``b`` (intraday), and each mean is split ``g : 1 - g``, so the
overnight part carries the share ``g = gap_share`` of the variance and of each mean; the
intraday part is defined by ``(1 + r_on)(1 + r_id) = 1 + r``. Raw prices follow from the drawn
returns: ``close_t = ((1 + r_t) close_{t-1} - D_t) / ratio_t`` and the open likewise with
``r_on``, so the returns of contract 2.5 equal the drawn returns and no split or dividend
changes a mean. Streams: ``synth.<component>``; truth (the latent signal) is returned beside
the store and is never written into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from quant_research_engine import __version__
from quant_research_engine.data.csvio import NO_TS, Table
from quant_research_engine.data.manifest import config_hash
from quant_research_engine.data.schemas import SCHEMAS
from quant_research_engine.rng import stream
from quant_research_engine.store.builder import build_store
from quant_research_engine.store.calendars import make_calendar
from quant_research_engine.store.store import Store
from quant_research_engine.synth.config import SynthConfig

GENERATOR_NAME = "qre.synth"
RETURN_FLOOR = -0.9


@dataclass
class Truth:
    """What the generator knows and a strategy never sees (oracles and checks only)."""

    ids: list[str]
    s: np.ndarray  # (T, N) latent signal s_t after bar t
    x: np.ndarray  # (T, N) observable proxy signal_x
    idio_std: np.ndarray  # (T, N) e / (sigma_i v_t ...): ic_t s_{t-1} + sqrt(1 - ic_t^2) z_t
    r: np.ndarray  # (T, N) drawn simple return (NaN outside an instrument's life)
    sigma: np.ndarray  # (N,) calm idiosyncratic per-bar volatility
    v: np.ndarray  # (T,) regime multiplier
    vol_mult: np.ndarray  # (T,) shift volatility multiplier
    ic_t: np.ndarray  # (T,)
    listing: np.ndarray  # (N,) first bar index
    last: np.ndarray  # (N,) last bar index
    betas: np.ndarray  # (N, K)
    factor_returns: np.ndarray  # (T, K)


def _draw(rng: np.random.Generator, tails: str, shape) -> np.ndarray:
    if tails == "gaussian":
        return rng.standard_normal(shape)
    return rng.standard_t(5, size=shape) * math.sqrt(3.0 / 5.0)


def generate(cfg: SynthConfig, seed: int) -> tuple[Store, Truth]:
    T, N, ppy = cfg.n_bars, cfg.n_instruments, cfg.ppy
    g = cfg.gap_share
    ev = cfg.events

    def S(name: str) -> np.random.Generator:
        return stream(seed, f"synth.{name}")

    cal = make_calendar(cfg.calendar, date.fromisoformat(cfg.start), T)
    ids = [f"I{k:03d}" for k in range(N)]

    # instrument parameters
    K = len(cfg.factors)
    betas = np.zeros((N, K))
    if K:
        rb = S("betas")
        betas[:, 0] = rb.normal(cfg.beta_market_mean, cfg.beta_market_sd, N)
        if K > 1:
            betas[:, 1:] = rb.normal(0.0, cfg.beta_style_sd, (N, K - 1))
    if cfg.idio_vol_log_sd > 0:
        idio_annual = cfg.idio_vol_median * np.exp(S("idio_vol").normal(0, cfg.idio_vol_log_sd, N))
    else:
        idio_annual = np.full(N, cfg.idio_vol_median)
    sig = idio_annual / math.sqrt(ppy)
    price0 = cfg.price_median * np.exp(S("price0").normal(0, cfg.price_log_sd, N))
    adv = cfg.volume.adv_median * np.exp(S("adv").normal(0, cfg.volume.adv_log_sd, N))

    listing = np.zeros(N, dtype=np.int64)
    if ev.enabled and ev.late_listing_share > 0:
        rl = S("listing")
        late = rl.random(N) < ev.late_listing_share
        hi = max(22, T // 2)
        bar = rl.integers(21, hi, N)
        listing[late] = np.minimum(bar[late], T - 1)

    # regime
    v = np.ones(T)
    if cfg.regime.enabled:
        u = S("regime").random(T)
        stress = False
        for t in range(T):
            if t > 0:
                if stress:
                    stress = not (u[t] < cfg.regime.p_stress_to_calm)
                else:
                    stress = u[t] < cfg.regime.p_calm_to_stress
            v[t] = cfg.regime.stress_vol_mult if stress else 1.0

    # distribution shift
    vm = np.ones(T)
    ic_t = np.full(T, cfg.ic)
    mu_shift = np.zeros(T)
    lam_t = np.ones((T, N))
    kap_t = np.ones((T, N))
    fvol = np.array([f.annual_vol for f in cfg.factors]) / math.sqrt(ppy)
    fmean = np.array([f.mean_annual for f in cfg.factors]) / ppy
    if cfg.shift is not None and cfg.shift.at_bar < T:
        sh = cfg.shift
        a = sh.at_bar
        vm[a:] = sh.vol_mult
        ic_t[a:] = cfg.ic * sh.ic_mult
        mu_shift[a:] = sh.mu_shift_annual / ppy
        F = (betas**2) @ (fvol**2) if K else np.zeros(N)
        V = F + sig**2
        lam = np.sqrt(
            np.minimum(sh.corr_mult, np.where(F > 0, 0.95 * V / np.maximum(F, 1e-300), 1.0))
        )
        kap = np.sqrt((V - lam**2 * F) / sig**2)
        lam_t[a:] = lam
        kap_t[a:] = kap
    if np.any(np.abs(ic_t) >= 1):
        raise ValueError("|ic| must stay below 1")

    # latent signal (row 0 is s_{-1}, the stationary predecessor of the first bar)
    phi = cfg.persistence
    rs = S("signal")
    s_full = np.empty((T + 1, N))
    s_full[0] = rs.standard_normal(N)
    innov = rs.standard_normal((T, N))
    c_phi = math.sqrt(1.0 - phi * phi)
    for t in range(T):
        s_full[t + 1] = phi * s_full[t] + c_phi * innov[t]
    s_prev, s = s_full[:-1], s_full[1:]
    w_obs = S("observable").standard_normal((T, N))
    x = (s + cfg.observable_noise * w_obs) / math.sqrt(1.0 + cfg.observable_noise**2)

    # shocks and returns
    fsh = _draw(S("factor_shocks"), cfg.tails, (T, K, 2)) if K else np.zeros((T, 0, 2))
    zsh = _draw(S("idio_shocks"), cfg.tails, (T, N, 2))
    sg, s1g = math.sqrt(g), math.sqrt(1.0 - g)
    vv = (v * vm)[:, None]
    fmean_t = np.broadcast_to(fmean, (T, K)).copy()
    if K:
        fmean_t[:, 0] += mu_shift
    f_on = g * fmean_t + fvol * vv * sg * fsh[:, :, 0]
    f_id = (1 - g) * fmean_t + fvol * vv * s1g * fsh[:, :, 1]
    fac_on = (f_on @ betas.T) * lam_t if K else np.zeros((T, N))
    fac_id = (f_id @ betas.T) * lam_t if K else np.zeros((T, N))
    ict = ic_t[:, None]
    cic = np.sqrt(1.0 - ict * ict)
    sd_t = sig[None, :] * vv * kap_t
    e_on = sd_t * (g * ict * s_prev + cic * sg * zsh[:, :, 0])
    e_id = sd_t * ((1 - g) * ict * s_prev + cic * s1g * zsh[:, :, 1])
    idio_std = ict * s_prev + cic * (sg * zsh[:, :, 0] + s1g * zsh[:, :, 1])
    c = cfg.premium_annual / ppy
    r_on = fac_on + e_on + g * c
    r = r_on + fac_id + e_id + (1 - g) * c
    r = np.maximum(r, RETURN_FLOOR)
    r_on = np.maximum(r_on, RETURN_FLOOR)

    # events
    last = np.full(N, T - 1, dtype=np.int64)
    delisted = np.zeros(N, dtype=bool)
    dret = np.full(N, np.nan)
    split_at = np.zeros((T, N), dtype=bool)
    split_ratio = np.ones((T, N))
    div_ex = np.zeros((T, N), dtype=bool)
    lead = max(1, ev.announce_lead_bars)
    if ev.enabled:
        rd = S("delist")
        ud = rd.random((T, N))
        dr_draw = rd.normal(ev.delist_return_mean, ev.delist_return_sd, N)
        p = ev.delist_hazard_annual / ppy
        for i in range(N):
            start = listing[i] + ev.min_life_bars
            hits = np.flatnonzero(ud[start:, i] < p) if start < T else np.zeros(0, dtype=np.int64)
            if hits.size:
                last[i] = start + hits[0]
                delisted[i] = True
                dret[i] = float(np.clip(dr_draw[i], -1.0, 1.0))
        rsp = S("splits")
        us = rsp.random((T, N))
        ratios = np.asarray(ev.split_ratios, dtype=np.float64)
        pick = rsp.integers(0, len(ratios), (T, N)) if len(ratios) else np.zeros((T, N), np.int64)
        tt = np.arange(T)[:, None]
        ok = (tt >= listing[None, :] + lead + 1) & (tt <= last[None, :])
        if len(ratios):
            split_at = ok & (us < ev.split_rate_annual / ppy)
            split_ratio = np.where(split_at, ratios[pick], 1.0)
        rdv = S("dividends")
        payer = rdv.random(N) < ev.dividend_share
        phase = rdv.integers(0, ev.dividend_every_bars, N)
        if ev.dividend_yield_annual > 0:
            for i in np.flatnonzero(payer).tolist():
                js = np.arange(phase[i], T, ev.dividend_every_bars)
                js = js[(js >= listing[i] + lead + 1) & (js <= last[i])]
                div_ex[js, i] = True

    # prices and volumes (sequential in time: a dividend is sized at its announcement)
    vol_cfg = cfg.volume
    rv = S("volume")
    xv = np.empty((T, N))
    xv[0] = rv.normal(0, vol_cfg.ar_sd / math.sqrt(max(1e-12, 1 - vol_cfg.ar_phi**2)), N)
    vin = rv.normal(0, vol_cfg.ar_sd, (T, N))
    for t in range(1, T):
        xv[t] = vol_cfg.ar_phi * xv[t - 1] + vin[t]
    zero_vol = S("zero_volume").random((T, N)) < vol_cfg.zero_volume_prob
    rng_hl = np.abs(S("range").standard_normal((T, N, 2)))
    o = np.full((T, N), np.nan)
    h = np.full((T, N), np.nan)
    lo = np.full((T, N), np.nan)
    cl = np.full((T, N), np.nan)
    vol = np.full((T, N), np.nan)
    D = np.zeros((T, N))
    cp = price0.copy()
    cum_split = np.ones(N)
    q = cfg.events.dividend_yield_annual / 4.0
    alive = np.zeros((T, N), dtype=bool)
    for t in range(T):
        live = (t >= listing) & (t <= last)
        alive[t] = live
        # size the dividends announced at this bar's close later; amounts known at t - lead
        if t - lead >= 0:
            due = div_ex[t] & live
            if np.any(due):
                amt = np.round(q * cl[t - lead], 4)
                D[t] = np.where(due & (amt > 0), amt, 0.0)
        ratio = split_ratio[t]
        on = ((1.0 + r_on[t]) * cp - D[t]) / ratio
        cc = ((1.0 + r[t]) * cp - D[t]) / ratio
        ext = cfg.range_scale * sd_t[t] * s1g
        hh = np.maximum(on, cc) * (1.0 + ext * rng_hl[t, :, 0])
        ll = np.minimum(on, cc) * np.maximum(0.5, 1.0 - ext * rng_hl[t, :, 1])
        cum_split = np.where(live, cum_split * ratio, cum_split)
        vv_t = np.round(adv * np.exp(xv[t])) * cum_split
        vv_t = np.where(zero_vol[t] & (t > listing), 0.0, vv_t)
        o[t] = np.where(live, on, np.nan)
        h[t] = np.where(live, hh, np.nan)
        lo[t] = np.where(live, ll, np.nan)
        cl[t] = np.where(live, cc, np.nan)
        vol[t] = np.where(live, vv_t, np.nan)
        cp = np.where(live, cc, cp)
    if np.any(cl[alive] <= 0) or np.any(o[alive] <= 0):
        raise ValueError("generator produced a non-positive price")
    D = np.where(alive, D, 0.0)
    D[~div_ex] = 0.0

    r_out = np.where(alive, r, np.nan)
    store = _assemble(
        cfg,
        seed,
        cal,
        ids,
        listing,
        last,
        delisted,
        dret,
        o,
        h,
        lo,
        cl,
        vol,
        alive,
        split_at,
        split_ratio,
        D,
        lead,
        x,
        r_out,
        sig,
    )
    truth = Truth(
        ids,
        s,
        x,
        idio_std,
        r_out,
        sig,
        v,
        vm,
        ic_t,
        listing,
        last,
        betas,
        (f_on + f_id) if K else np.zeros((T, 0)),
    )
    return store, truth


def _assemble(
    cfg,
    seed,
    cal,
    ids,
    listing,
    last,
    delisted,
    dret,
    o,
    h,
    lo,
    cl,
    vol,
    alive,
    split_at,
    split_ratio,
    D,
    lead,
    x,
    r,
    sig,
) -> Store:
    T, N = cl.shape
    te, to = cal.ts_event, cal.ts_open
    # instruments
    ins = Table(
        SCHEMAS["instruments"],
        {
            "instrument_id": np.asarray(ids, dtype=object),
            "symbol": np.asarray(ids, dtype=object),
            "asset_class": np.asarray(
                ["equity" if cfg.calendar == "equity_daily" else "crypto"] * N, dtype=object
            ),
            "currency": np.asarray(["USD"] * N, dtype=object),
            "ts_list": te[listing].astype(np.int64),
            "ts_delist": np.where(delisted, te[last], NO_TS).astype(np.int64),
            "delist_return": np.where(delisted, dret, np.nan),
            "lot_size": np.full(N, float(cfg.lot_size)),
            "tick_size": np.full(N, float(cfg.tick_size)),
            "sector": np.asarray([""] * N, dtype=object),
        },
    )
    # bars
    cols = {k: [] for k in SCHEMAS["bars"].names}
    for i in range(N):
        rows = np.flatnonzero(alive[:, i])
        cols["instrument_id"].append(np.full(rows.size, ids[i], dtype=object))
        cols["ts_open"].append(to[rows])
        cols["ts_event"].append(te[rows])
        cols["ts_avail"].append(te[rows])
        for k, arr in (("open", o), ("high", h), ("low", lo), ("close", cl), ("volume", vol)):
            cols[k].append(arr[rows, i])
    bars = Table(SCHEMAS["bars"], {k: np.concatenate(v) for k, v in cols.items()})
    # corporate actions
    ca = {k: [] for k in SCHEMAS["corporate_actions"].names}
    for i in range(N):
        js = np.flatnonzero((split_at[:, i] | (D[:, i] > 0)) & alive[:, i])
        for j in js.tolist():
            if D[j, i] > 0:
                ca["instrument_id"].append(ids[i])
                ca["action"].append("cash_dividend")
                ca["ts_ex"].append(int(to[j]))
                ca["ts_avail"].append(int(te[j - lead]))
                ca["value"].append(float(D[j, i]))
            if split_at[j, i]:
                ca["instrument_id"].append(ids[i])
                ca["action"].append("split")
                ca["ts_ex"].append(int(to[j]))
                ca["ts_avail"].append(int(te[j - lead]))
                ca["value"].append(float(split_ratio[j, i]))
    actions = Table(
        SCHEMAS["corporate_actions"],
        {
            "instrument_id": np.asarray(ca["instrument_id"], dtype=object),
            "action": np.asarray(ca["action"], dtype=object),
            "ts_ex": np.asarray(ca["ts_ex"], dtype=np.int64),
            "ts_avail": np.asarray(ca["ts_avail"], dtype=np.int64),
            "value": np.asarray(ca["value"], dtype=np.float64),
        },
    )
    # series: fund_x (lagged, revised) and signal_x (vintage 0 at each bar)
    fx = cfg.fund_x
    nperiods = T // fx.period_bars + 1
    nraw = stream(seed, "synth.fund_x").standard_normal((nperiods, N))
    ser = {k: [] for k in SCHEMAS["series"].names}
    for i in range(N):
        sid_f = f"{ids[i]}.fund_x"
        for p, pe in enumerate(range(fx.period_bars - 1, T, fx.period_bars)):
            if pe < listing[i] or pe + fx.lag0_bars > last[i]:
                continue
            z0 = math.fsum(r[pe + 1 : pe + fx.lag0_bars + 1, i].tolist()) / (
                sig[i] * math.sqrt(fx.lag0_bars)
            )
            v0 = float(nraw[p, i] + fx.w * z0)
            ser["series_id"].append(np.asarray([sid_f], dtype=object))
            ser["ts_event"].append(np.asarray([te[pe]], dtype=np.int64))
            ser["ts_avail"].append(np.asarray([te[pe + fx.lag0_bars]], dtype=np.int64))
            ser["vintage"].append(np.asarray([0], dtype=np.int64))
            ser["value"].append(np.asarray([v0]))
            if pe + fx.lag1_bars <= last[i]:
                n1 = fx.lag1_bars - fx.lag0_bars
                seg = r[pe + fx.lag0_bars + 1 : pe + fx.lag1_bars + 1, i].tolist()
                z1 = math.fsum(seg) / (sig[i] * math.sqrt(n1))
                ser["series_id"].append(np.asarray([sid_f], dtype=object))
                ser["ts_event"].append(np.asarray([te[pe]], dtype=np.int64))
                ser["ts_avail"].append(np.asarray([te[pe + fx.lag1_bars]], dtype=np.int64))
                ser["vintage"].append(np.asarray([1], dtype=np.int64))
                ser["value"].append(np.asarray([v0 + fx.w * z1]))
        rows = np.flatnonzero(alive[:, i])
        ser["series_id"].append(np.full(rows.size, f"{ids[i]}.signal_x", dtype=object))
        ser["ts_event"].append(te[rows])
        ser["ts_avail"].append(te[rows])
        ser["vintage"].append(np.zeros(rows.size, dtype=np.int64))
        ser["value"].append(x[rows, i])
    series = Table(
        SCHEMAS["series"],
        {
            "series_id": np.concatenate(ser["series_id"])
            if ser["series_id"]
            else np.zeros(0, object),
            "ts_event": np.concatenate(ser["ts_event"])
            if ser["ts_event"]
            else np.zeros(0, np.int64),
            "ts_avail": np.concatenate(ser["ts_avail"])
            if ser["ts_avail"]
            else np.zeros(0, np.int64),
            "vintage": np.concatenate(ser["vintage"]) if ser["vintage"] else np.zeros(0, np.int64),
            "value": np.concatenate(ser["value"]) if ser["value"] else np.zeros(0),
        },
    )
    meta = {
        "calendar": cfg.calendar,
        "ppy": cfg.ppy,
        "generator": {
            "name": GENERATOR_NAME,
            "version": __version__,
            "parameters": cfg.model_dump(mode="json"),
        },
        "config_hash": config_hash(cfg),
        "seed": int(seed),
        "data": "synthetic, assumed parameters",
    }
    return build_store(ins, bars, actions, series, meta)


def close_to_close_returns(weights: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per-bar returns of weights decided at bar t and held from close t to close t + 1
    (no engine, no costs; the bound of contract 7, not achievable under contract 3.5)."""
    w = np.nan_to_num(weights[:-1])
    nxt = np.nan_to_num(r[1:])
    return np.sum(w * nxt, axis=1)
