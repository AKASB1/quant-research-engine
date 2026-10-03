"""Vectorized reference engine: ``run_target_shares`` (independent of the event engine).

Input: target shares per decision instant (what a strategy emits after sizing, in the shares of
the decision instant, already rounded to lots and the quantity grid). With ``fill_delay_bars = d``
the target of decision ``j`` (at the close of the instrument's bar ``b_j``) is the position after
the fills of instrument bar ``b_j + d``, deferred to the next bar with volume when that bar has
none (the orders that targets create are ``gtc``). Orders are differences of consecutive
targets, rescaled by the split ratios between decision and fill; positions are the latest
filled target rescaled to current shares. Everything is array arithmetic over the panel, with
two exceptions that carry no event logic: a loop over instruments for the mapping from
decisions to each instrument's own bars, and a scalar loop over bars for the cash recursion
(financing and interest depend on the previous cash). No participation cap and no leverage
limit: the equivalence regime with the event engine is the one in which neither binds.
Split ratios must be integers (2, 3, ...): then quantity arithmetic is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_research_engine.costs import CostConfig, fill_costs
from quant_research_engine.data.csvio import NO_TS
from quant_research_engine.engine.panelutil import liquidity_asof_panel

IDENTITY_TOL = 1e-9


@dataclass
class VectorResult:
    ids: list[str]
    ts_event: np.ndarray
    cash: np.ndarray
    position_value: np.ndarray
    equity: np.ndarray
    hold_pnl: np.ndarray
    trade_pnl: np.ndarray
    spread_cost: np.ndarray
    impact_cost: np.ndarray
    commission: np.ndarray
    borrow: np.ndarray
    financing: np.ndarray
    income: np.ndarray
    positions: np.ndarray  # (T, N) after each bar
    fills: dict  # arrays: k_fill, inst, dec, quantity, ref_price, price, costs, delist
    turnover: np.ndarray
    gross_exposure: np.ndarray
    net_exposure: np.ndarray
    max_residual: float
    impact_unavailable: int
    orders: dict | None = None
    fill_model: str = "next_open"

    @property
    def returns(self) -> np.ndarray:
        e = self.equity
        return e[1:] / e[:-1] - 1.0


def run_target_shares(
    panel,
    dec_idx,
    targets,
    *,
    fill_model: str = "next_open",
    fill_delay_bars: int = 1,
    initial_cash: float = 1_000_000.0,
    costs: CostConfig | None = None,
    liquidity: tuple[np.ndarray, np.ndarray] | None = None,
    force_general: bool = False,
) -> VectorResult:
    """Run target-share schedules; ``targets[j, i]`` is the target of instrument ``i`` at the
    close of calendar bar ``dec_idx[j]`` (0 for instruments outside the universe)."""
    if fill_model not in ("next_open", "next_close"):
        raise ValueError("the vectorized engine supports next_open and next_close only")
    if fill_delay_bars < 1:
        raise ValueError("fill_delay_bars must be >= 1")
    costs = costs or CostConfig()
    p = panel
    T, N = p.shape
    dec_idx = np.asarray(dec_idx, dtype=np.int64)
    targets = np.nan_to_num(np.asarray(targets, dtype=np.float64), nan=0.0)
    D = len(dec_idx)
    if D and np.any(np.diff(dec_idx) <= 0):
        raise ValueError("decision indices must increase")
    r = p.ratio
    if np.any(r != np.round(r)):
        raise ValueError("the vectorized engine needs integer split ratios")
    C = np.cumprod(r, axis=0)
    adv, sig = liquidity if liquidity is not None else liquidity_asof_panel(p)
    if D:
        sig_d, V_d = sig[dec_idx], adv[dec_idx]
    else:
        sig_d = V_d = np.zeros((0, N))
    d = int(fill_delay_bars)
    ref = p.open if fill_model == "next_open" else p.close
    te = p.ts_event

    if not force_general and _contiguous(p):
        out = _orders_contiguous(p, dec_idx, targets, C, d, ref, te)
    else:
        out = _orders_general(p, dec_idx, targets, C, d, ref, te)
    pos, f_k, f_i, f_j, f_q, f_m, o_j, o_i, o_q, dl_k, dl_i, dl_q, dl_m = out

    fk = np.concatenate(f_k) if f_k else np.zeros(0, np.int64)
    fi = np.concatenate(f_i) if f_i else np.zeros(0, np.int64)
    fj = np.concatenate(f_j) if f_j else np.zeros(0, np.int64)
    fq = np.concatenate(f_q) if f_q else np.zeros(0)
    fm = np.concatenate(f_m) if f_m else np.zeros(0)
    if fk.size:
        o_ = np.lexsort((fj, fi, fk))
        fk, fi, fj, fq, fm = fk[o_], fi[o_], fj[o_], fq[o_], fm[o_]
    price, spread, impact, comm, unavail = fill_costs(
        fq, fm, sig_d[fj, fi] if fk.size else fq, V_d[fj, fi] if fk.size else fq, costs
    )
    close = p.close
    # last close (raw) forward filled; marks
    src_row = np.maximum.accumulate(np.where(np.isnan(close), -1, np.arange(T)[:, None]), axis=0)
    lc = np.where(src_row >= 0, close[np.maximum(src_row, 0), np.arange(N)[None, :]], np.nan)
    lc_prev = np.vstack([np.full((1, N), np.nan), lc[:-1]])
    pos_prev = np.vstack([np.zeros((1, N)), pos[:-1]])
    has = p.has_bar
    rat = np.where(has, r, 1.0)
    pos_open = pos_prev * rat
    pc_adj = lc_prev / rat
    hold_cell = np.where(has & (pos_open != 0), pos_open * (close - pc_adj), 0.0)
    hold = hold_cell.sum(axis=1)
    div_cell = np.where(has & (pos_prev != 0), pos_prev * p.div, 0.0)
    divs = div_cell.sum(axis=1)
    trade_f = fq * (close[fk, fi] - fm)
    trade = np.bincount(fk, weights=trade_f, minlength=T) if fk.size else np.zeros(T)
    sp = np.bincount(fk, weights=spread, minlength=T) if fk.size else np.zeros(T)
    im = np.bincount(fk, weights=impact, minlength=T) if fk.size else np.zeros(T)
    cm = np.bincount(fk, weights=comm, minlength=T) if fk.size else np.zeros(T)
    flow = -(np.bincount(fk, weights=fq * price + comm, minlength=T) if fk.size else np.zeros(T))
    dk = np.asarray(dl_k, dtype=np.int64)
    di = np.asarray(dl_i, dtype=np.int64)
    dq = np.asarray(dl_q, dtype=np.float64)
    dm = np.asarray(dl_m, dtype=np.float64)
    if dk.size:
        trade = trade + np.bincount(dk, weights=dq * (close[dk, di] - dm), minlength=T)
        flow = flow + np.bincount(dk, weights=-dq * dm, minlength=T)
    short_val = np.where(pos_prev < 0, -pos_prev * lc_prev, 0.0).sum(axis=1)
    dt = np.zeros(T)
    if T > 1:
        dt[1:] = (te[1:] - te[:-1]) / 86_400_000_000
    borrow = short_val * costs.borrow_bps_annual / 1e4 * dt / 365.0
    pv = np.where(pos != 0, pos * lc, 0.0).sum(axis=1)
    cash = np.empty(T)
    fin = np.zeros(T)
    inc = np.zeros(T)
    c_prev = float(initial_cash)
    fr, cr = costs.financing_bps_annual / 1e4, costs.cash_rate_bps_annual / 1e4
    fl, dv, bo, dtl = flow.tolist(), divs.tolist(), borrow.tolist(), dt.tolist()
    cl_, fl_, il_ = [0.0] * T, [0.0] * T, [0.0] * T
    for k in range(T):
        f_ = max(0.0, -c_prev) * fr * dtl[k] / 365.0
        i_ = max(0.0, c_prev) * cr * dtl[k] / 365.0
        fl_[k], il_[k] = f_, i_
        c_prev = c_prev + fl[k] + dv[k] + i_ - bo[k] - f_
        cl_[k] = c_prev
    cash[:], fin[:], inc[:] = cl_, fl_, il_
    income = divs + inc
    equity = cash + pv
    e_prev = np.r_[initial_cash, equity[:-1]]
    resid = np.abs((equity - e_prev) - (hold + trade - sp - im - cm - borrow - fin + income))
    scale = np.maximum(1.0, np.abs(e_prev))
    if np.any(resid > IDENTITY_TOL * scale):
        k = int(np.argmax(resid / scale))
        raise ArithmeticError(f"vectorized identity violated at bar {k}: {resid[k]:.3g}")
    turnover = np.zeros(T)
    if fk.size:
        turnover = np.bincount(fk, weights=np.abs(fq) * fm, minlength=T) / e_prev
    gross = np.where(pos != 0, np.abs(pos * lc), 0.0).sum(axis=1) / equity
    net = pv / equity
    allk = np.r_[fk, dk]
    order = np.lexsort((np.r_[fi, di], allk)) if allk.size else np.zeros(0, np.int64)
    fills = {
        "k_fill": allk[order],
        "inst": np.r_[fi, di][order],
        "dec": np.r_[fj, np.full(dk.size, -1)][order],
        "quantity": np.r_[fq, dq][order],
        "ref_price": np.r_[fm, dm][order],
        "price": np.r_[price, dm][order],
        "spread": np.r_[spread, np.zeros(dk.size)][order],
        "impact": np.r_[impact, np.zeros(dk.size)][order],
        "commission": np.r_[comm, np.zeros(dk.size)][order],
        "delist": np.r_[np.zeros(fk.size, bool), np.ones(dk.size, bool)][order],
    }
    return VectorResult(
        ids=list(p.ids),
        ts_event=te,
        cash=cash,
        position_value=pv,
        equity=equity,
        hold_pnl=hold,
        trade_pnl=trade,
        spread_cost=sp,
        impact_cost=im,
        commission=cm,
        borrow=borrow,
        financing=fin,
        income=income,
        positions=pos,
        fills=fills,
        turnover=turnover,
        gross_exposure=gross,
        net_exposure=net,
        max_residual=float(np.max(resid / scale)) if T else 0.0,
        fill_model=fill_model,
        impact_unavailable=int(np.sum(unavail)),
        orders={
            "dec": np.concatenate(o_j) if o_j else np.zeros(0, np.int64),
            "inst": np.concatenate(o_i) if o_i else np.zeros(0, np.int64),
            "quantity": np.concatenate(o_q) if o_q else np.zeros(0),
        },
    )


def _orders_general(p, dec_idx, targets, C, d, ref, te):
    """Per-instrument mapping of decisions to each instrument's own bars (handles gaps)."""
    T, N = p.shape
    D = len(dec_idx)
    pos = np.zeros((T, N))
    f_k, f_i, f_j, f_q, f_m = [], [], [], [], []
    o_j, o_i, o_q = [], [], []
    dl_k, dl_i, dl_q, dl_m = [], [], [], []
    for i in range(N):
        bars = np.flatnonzero(p.has_bar[:, i])
        nb = bars.size
        if nb == 0:
            if D and np.any(targets[:, i] != 0):
                raise ValueError(f"target for {p.ids[i]} without bars")
            continue
        last = int(bars[-1])
        delists = bool(p.ts_delist[i] != NO_TS and te[last] == p.ts_delist[i])
        tg = targets[:, i] if D else np.zeros(0)
        b0 = np.searchsorted(bars, dec_idx, side="right") - 1
        listed = (p.ts_list[i] <= te[dec_idx]) & (
            (p.ts_delist[i] == NO_TS) | (p.ts_delist[i] > te[dec_idx])
        )
        bad = (tg != 0) & ((b0 < 0) | ~listed)
        if np.any(bad):
            raise ValueError(f"nonzero target for {p.ids[i]} outside its listed life")
        alive = (b0 >= 0) & ~(delists & (dec_idx >= last))
        E = np.where(alive, tg, 0.0)
        E_prev = np.zeros(D)
        if D > 1:
            E_prev[1:] = E[:-1] * (C[dec_idx[1:], i] / C[dec_idx[:-1], i])
        E_prev = np.where(alive, E_prev, 0.0)
        o = E - E_prev
        # fill bars (instrument bar index), deferred past zero-volume bars
        vol_i = p.volume[bars, i]
        nv = np.where(vol_i > 0, np.arange(nb), nb)
        nv = np.minimum.accumulate(nv[::-1])[::-1]
        nv = np.r_[nv, nb]
        elig = np.minimum(np.where(b0 >= 0, b0 + d, nb), nb)
        f = nv[elig]
        f = np.where(alive, f, nb)
        nz = np.flatnonzero(o != 0)
        o_j.append(nz)
        o_i.append(np.full(nz.size, i))
        o_q.append(o[nz])
        has = (o != 0) & (f < nb)
        js = np.flatnonzero(has)
        fc = bars[f[js]]
        q = o[js] * (C[fc, i] / C[dec_idx[js], i])
        f_k.append(fc)
        f_i.append(np.full(js.size, i))
        f_j.append(js)
        f_q.append(q)
        f_m.append(ref[fc, i])
        # positions after the fills of each instrument bar: latest decision whose fill bar <= b
        fb = np.where(alive, f, nb)
        order = np.flatnonzero(alive)
        pos_b = np.zeros(nb)
        if order.size:
            fb_a = fb[order]
            J = np.searchsorted(fb_a, np.arange(nb), side="right") - 1
            ok = J >= 0
            jj = order[np.maximum(J, 0)]
            pos_b = np.where(ok, E[jj] * (C[bars, i] / C[dec_idx[jj], i]), 0.0)
        if delists:
            q_exit = pos_b[-1]
            if q_exit != 0.0:
                dl_k.append(last)
                dl_i.append(i)
                dl_q.append(-q_exit)
                dl_m.append(p.close[last, i] * (1.0 + p.delist_return[i]))
            pos_b[-1] = 0.0
        # forward-fill onto the calendar
        col = np.zeros(T)
        idx = np.searchsorted(bars, np.arange(T), side="right") - 1
        okc = idx >= 0
        col[okc] = pos_b[idx[okc]]
        if delists:
            col[last + 1 :] = 0.0
        pos[:, i] = col

    return pos, f_k, f_i, f_j, f_q, f_m, o_j, o_i, o_q, dl_k, dl_i, dl_q, dl_m


def _contiguous(p) -> bool:
    cnt = p.has_bar.sum(axis=0)
    return bool(np.all((cnt == 0) | (cnt == p.last_bar - p.first_bar + 1)))


def _orders_contiguous(p, dec_idx, targets, C, d, ref, te):
    """The same computation on (decisions x instruments) arrays, for stores in which every
    instrument has a bar at every session between its first and last bar (same elementwise
    arithmetic as the general path, so the results are bit-identical)."""
    T, N = p.shape
    D = len(dec_idx)
    cols = np.arange(N)[None, :]
    first, last = p.first_bar, p.last_bar
    has_any = last >= 0
    lastc = np.maximum(last, 0)
    delists = has_any & (p.ts_delist != NO_TS) & (te[lastc] == p.ts_delist)
    dcol = dec_idx[:, None]
    b0 = np.where(dcol >= first[None, :], np.minimum(dcol, last[None, :]) - first[None, :], -1)
    b0 = np.where(has_any[None, :], b0, -1)
    te_d = te[dec_idx][:, None]
    listed = (p.ts_list[None, :] <= te_d) & (
        (p.ts_delist[None, :] == NO_TS) | (p.ts_delist[None, :] > te_d)
    )
    bad = (targets != 0) & ((b0 < 0) | ~listed)
    if np.any(bad):
        i = int(np.flatnonzero(bad.any(axis=0))[0])
        raise ValueError(f"nonzero target for {p.ids[i]} outside its listed life")
    alive = (b0 >= 0) & ~(delists[None, :] & (dcol >= last[None, :]))
    E = np.where(alive, targets, 0.0)
    E_prev = np.zeros((D, N))
    if D > 1:
        E_prev[1:] = E[:-1] * (C[dec_idx[1:]] / C[dec_idx[:-1]])
    E_prev = np.where(alive, E_prev, 0.0)
    o = E - E_prev
    volok = p.has_bar & (p.volume > 0)
    nv = np.where(volok, np.arange(T)[:, None], T)
    nv = np.minimum.accumulate(nv[::-1], axis=0)[::-1]
    nv = np.vstack([nv, np.full((1, N), T)])
    elig = np.where(b0 >= 0, first[None, :] + b0 + d, T)
    elig = np.where(elig > last[None, :], T, elig)
    f = nv[np.minimum(elig, T), cols]
    f = np.where(alive, f, T)
    oj, oi = np.nonzero(o != 0)
    srt = np.lexsort((oj, oi))  # list orders per instrument, then decision (general path order)
    oj, oi = oj[srt], oi[srt]
    has = (o != 0) & (f < T)
    js, is_ = np.nonzero(has)
    fc = f[js, is_]
    q = o[js, is_] * (C[fc, is_] / C[dec_idx[js], is_])
    marker = np.full((T + 1, N), -1, dtype=np.int64)
    aj, ai = np.nonzero(alive & (f < T))
    np.maximum.at(marker, (f[aj, ai], ai), aj)
    J = np.maximum.accumulate(marker[:T], axis=0)
    Jc = np.maximum(J, 0)
    pos = np.where(J >= 0, E[Jc, cols] * (C / C[dec_idx[Jc], cols]), 0.0) if D else np.zeros((T, N))
    dl = np.flatnonzero(delists)
    q_exit = pos[last[dl], dl] if dl.size else np.zeros(0)
    nzx = q_exit != 0.0
    dl_k = [int(x) for x in last[dl][nzx]]
    dl_i = [int(x) for x in dl[nzx]]
    dl_q = [float(-x) for x in q_exit[nzx]]
    dl_m = [
        float(p.close[k, i] * (1.0 + p.delist_return[i])) for k, i in zip(dl_k, dl_i, strict=True)
    ]
    for i in dl.tolist():
        pos[last[i] :, i] = 0.0
    return (
        pos,
        [fc],
        [is_],
        [js],
        [q],
        [ref[fc, is_]],
        [oj],
        [oi],
        [o[oj, oi]],
        dl_k,
        dl_i,
        dl_q,
        dl_m,
    )
