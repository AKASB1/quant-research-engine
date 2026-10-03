"""Event-driven backtest engine (shared contract sections 3 to 5).

The engine walks the merged timeline of the sessions of the store's calendar, instruments in
``instrument_id`` order, and applies in each bar of an instrument: (1) splits rescale the
opening quantity, the previous close used for PnL, the committed target, and pending orders;
(2) dividends on the pre-split opening quantity, credited at the close; (3) eligible orders fill
(FIFO; participation cap per instrument and bar; zero volume: gtc waits, day is cancelled);
(4) the delisting exit at ``close * (1 + delist_return)`` in the instrument's last bar, which
cancels its pending orders; then (5) accruals and (6) marks at the close. Every bar asserts the
accounting identity in memory and raises :class:`AccountingError` on a violation.

Orders created by a decision are the difference between the target and the committed position
(the position after all pending orders fill); a decision cancels nothing. An order submitted at
the close of an instrument's bar ``b`` fills at the earliest in bar ``b + fill_delay_bars``.
``same_close`` (fill at the decision bar's own close) exists only behind
``unsafe_same_bar_fill`` and labels every output ``UNSAFE``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from quant_research_engine.context.context import OpenOrder, PortfolioView
from quant_research_engine.context.decision import Orders, TargetShares, TargetWeights
from quant_research_engine.costs import CostConfig, accruals, fill_costs
from quant_research_engine.data.csvio import NO_TS, make_table
from quant_research_engine.engine.panelutil import bar_numbers, last_bar_at, liquidity_asof_panel
from quant_research_engine.engine.quantity import round_lot_scalar, snap_scalar

IDENTITY_TOL = 1e-9


class AccountingError(RuntimeError):
    pass


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fill_model: Literal["auto", "next_open", "next_close", "same_close"] = "auto"
    unsafe_same_bar_fill: bool = False
    fill_delay_bars: int = Field(default=1, ge=1)
    allow_short: bool = True
    max_leverage: float | None = Field(default=None, gt=0)
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    sizing: Literal["equity", "fixed_capital"] = "equity"
    costs: CostConfig = Field(default_factory=CostConfig)

    def resolved(self, calendar: str) -> EngineConfig:
        """``auto`` -> ``next_close`` for crypto_daily (a daily bar's open equals the previous
        close there), ``next_open`` otherwise (contract 3.5)."""
        if self.fill_model != "auto":
            return self
        fm = "next_close" if calendar == "crypto_daily" else "next_open"
        return self.model_copy(update={"fill_model": fm})

    def check(self) -> None:
        if self.fill_model == "same_close" and not self.unsafe_same_bar_fill:
            raise ValueError("fill_model 'same_close' requires unsafe_same_bar_fill (guard G5)")

    @property
    def unsafe(self) -> bool:
        return self.fill_model == "same_close"


@dataclass
class _Order:
    order_id: str
    inst: int
    qty: float
    tif: str
    k_submit: int
    ts_submit: int
    elig_bar: int
    sigma: float
    V: float
    original: float


@dataclass
class EventResult:
    ids: list[str]
    ts_event: np.ndarray
    equity_rows: list[tuple]
    positions_rows: list[tuple]
    orders: list[tuple]
    fills: list[tuple]
    rejected: list[tuple]
    cancelled: list[tuple]
    max_residual: float
    unsafe: bool
    fill_model: str
    impact_unavailable: int
    per_instrument: dict = field(default_factory=dict)
    books: dict = field(default_factory=dict)
    cap_binds: int = 0

    @property
    def equity(self) -> np.ndarray:
        return np.asarray([r[3] for r in self.equity_rows])

    def tables(self, strategy_id: str = "strategy"):
        eq = list(zip(*self.equity_rows, strict=True)) if self.equity_rows else [[]] * 12
        equity = make_table(
            "equity",
            ts_event=eq[0],
            cash=eq[1],
            position_value=eq[2],
            equity=eq[3],
            hold_pnl=eq[4],
            trade_pnl=eq[5],
            spread_cost=eq[6],
            impact_cost=eq[7],
            commission=eq[8],
            borrow=eq[9],
            financing=eq[10],
            income=eq[11],
        )
        ps = list(zip(*self.positions_rows, strict=True)) if self.positions_rows else [[]] * 5
        positions = make_table(
            "positions",
            ts_event=ps[0],
            instrument_id=ps[1],
            quantity=ps[2],
            mark_price=ps[3],
            value=ps[4],
        )
        od = sorted(self.orders, key=lambda r: (r[1], r[0]))
        od = list(zip(*od, strict=True)) if od else [[]] * 8
        orders = make_table(
            "orders",
            order_id=od[0],
            ts_submit=od[1],
            instrument_id=od[2],
            quantity=od[3],
            order_type=od[4],
            limit_price=od[5],
            tif=od[6],
            strategy_id=od[7],
        )
        fl = sorted(self.fills, key=lambda r: (r[2], r[0]))
        fl = list(zip(*fl, strict=True)) if fl else [[]] * 10
        fills = make_table(
            "fills",
            fill_id=fl[0],
            order_id=fl[1],
            ts_fill=fl[2],
            instrument_id=fl[3],
            quantity=fl[4],
            ref_price=fl[5],
            price=fl[6],
            spread_cost=fl[7],
            impact_cost=fl[8],
            commission=fl[9],
        )
        return {"orders": orders, "fills": fills, "positions": positions, "equity": equity}


class EventEngine:
    """Bar-by-bar engine over a store panel; decisions are submitted between bars."""

    def __init__(
        self,
        panel,
        cfg: EngineConfig,
        strategy_id: str = "strategy",
        calendar: str = "equity_daily",
    ):
        cfg = cfg.resolved(calendar)
        cfg.check()
        self.p = panel
        self.cfg = cfg
        self.sid = strategy_id
        T, N = panel.shape
        self.T, self.N = T, N
        self.bar_no = bar_numbers(panel.has_bar)
        self.last_at = last_bar_at(panel.has_bar)
        self.liq_adv, self.liq_sigma = liquidity_asof_panel(panel)
        self.bar_cols = [np.flatnonzero(panel.has_bar[k]).tolist() for k in range(T)]
        self.delists = [
            bool(
                panel.ts_delist[i] != NO_TS
                and panel.last_bar[i] >= 0
                and panel.ts_event[panel.last_bar[i]] == panel.ts_delist[i]
            )
            for i in range(N)
        ]
        self.lot = [float(x) for x in panel.lot_size]
        self.cash = float(cfg.initial_cash)
        self.pos = [0.0] * N
        self.E = [0.0] * N
        self.last_close = [math.nan] * N
        self.pending: list[deque[_Order]] = [deque() for _ in range(N)]
        self.k = -1
        self.equity_rows: list[tuple] = []
        self.positions_rows: list[tuple] = []
        self.orders: list[tuple] = []
        self.fills: list[tuple] = []
        self.rejected: list[tuple] = []
        self.cancelled: list[tuple] = []
        self.max_residual = 0.0
        self.n_orders = 0
        self.n_fills = 0
        self.impact_unavailable = 0
        self.equity_prev = float(cfg.initial_cash)
        self._row: list | None = None
        # attribution: per instrument and per book (long, short) of hold, trade, fill costs,
        # dividends, borrow; financing and interest are portfolio-level
        self.attr = np.zeros((N, 5))
        self.book = np.zeros((2, 5))
        self._bar_attr: dict[int, list] = {}
        self.cap_binds = 0
        self._cap_used: dict[tuple[int, int], float] = {}

    # ------------------------------------------------------------------ bars

    def _book_fills(self, k: int, items: list[tuple[_Order, float, float]]):
        """Cost and book fills (order, quantity, reference price) of one bar, in order."""
        if not items:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        q = np.asarray([x[1] for x in items])
        m = np.asarray([x[2] for x in items])
        sg = np.asarray([x[0].sigma for x in items])
        V = np.asarray([x[0].V for x in items])
        price, spread, impact, comm, unavail = fill_costs(q, m, sg, V, self.cfg.costs)
        trade = sp = im = cm = 0.0
        ts = (
            int(self.p.ts_event[k])
            if self.cfg.fill_model != "next_open"
            else int(self.p.ts_open[k])
        )
        for j, (o, qq, mm) in enumerate(items):
            i = o.inst
            c = float(self.p.close[k, i])
            pr, s_, i_, c_ = float(price[j]), float(spread[j]), float(impact[j]), float(comm[j])
            self.cash -= qq * pr + c_
            self.pos[i] = self.pos[i] + qq
            tp = qq * (c - mm)
            trade += tp
            sp += s_
            im += i_
            cm += c_
            self._acc(i, 1, tp, qq)
            self._acc(i, 2, s_ + i_ + c_, qq)
            self.n_fills += 1
            self.impact_unavailable += int(unavail[j])
            self.fills.append(
                (f"f{self.n_fills}", o.order_id, ts, self.p.ids[i], qq, mm, pr, s_, i_, c_)
            )
        return trade, sp, im, cm, 0.0

    def process_bar(self, k: int) -> None:
        assert k == self.k + 1
        self.k = k
        p, cfg = self.p, self.cfg
        if k == 0:
            dt = 0.0
        else:
            dt = (int(p.ts_event[k]) - int(p.ts_event[k - 1])) / 86_400_000_000
        short_val = 0.0
        self._bar_attr = {}
        self._q0 = {}
        rate = cfg.costs.borrow_bps_annual / 1e4 * dt / 365.0
        for i in range(self.N):
            if self.pos[i] < 0:
                short_val += -self.pos[i] * self.last_close[i]
                self._q0[i] = self.pos[i]
                self._acc(i, 4, -self.pos[i] * self.last_close[i] * rate, self.pos[i])
        borrow, financing, interest = accruals(self.cash, short_val, dt, cfg.costs)
        cash_prev = self.cash
        hold = trade = sp = im = cm = div_inc = 0.0
        fills_now: list[tuple[_Order, float, float]] = []
        delist_now: list[int] = []
        for i in self.bar_cols[k]:
            r = float(p.ratio[k, i])
            q0 = self.pos[i]
            self._q0[i] = q0
            pc = self.last_close[i]
            if r != 1.0:
                self.pos[i] = snap_scalar(q0 * r)
                self.E[i] = snap_scalar(self.E[i] * r)
                for o in self.pending[i]:
                    o.qty = snap_scalar(o.qty * r)
                pc = pc / r
            d = float(p.div[k, i])
            if d != 0.0 and q0 != 0.0:
                div_inc += q0 * d
                self._acc(i, 3, q0 * d, q0)
            c = float(p.close[k, i])
            if self.pos[i] != 0.0:
                hp = self.pos[i] * (c - pc)
                hold += hp
                self._acc(i, 0, hp, self.pos[i])
            if self.pending[i] and cfg.fill_model != "same_close":
                fills_now.extend(self._eligible_fills(k, i))
            elif self.pending[i]:
                fills_now.extend(self._eligible_fills(k, i, ref="close"))
            if k == p.last_bar[i] and self.delists[i]:
                delist_now.append(i)
        tr, sp, im, cm, _ = self._book_fills(k, fills_now)
        trade += tr
        for i in delist_now:
            c = float(p.close[k, i])
            if self.pos[i] != 0.0:
                m = c * (1.0 + float(p.delist_return[i]))
                q = -self.pos[i]
                self.cash -= q * m
                tp = q * (c - m)
                trade += tp
                self._acc(i, 1, tp, -q)
                self.n_fills += 1
                self.fills.append(
                    (
                        f"f{self.n_fills}",
                        "DELIST",
                        int(p.ts_event[k]),
                        p.ids[i],
                        q,
                        m,
                        m,
                        0.0,
                        0.0,
                        0.0,
                    )
                )
                self.pos[i] = 0.0
            for o in self.pending[i]:
                self.cancelled.append((k, o.order_id, p.ids[i], o.qty, "delisted"))
            self.pending[i].clear()
            self.E[i] = 0.0
        for i in self.bar_cols[k]:
            self.last_close[i] = float(p.close[k, i])
        self.cash += div_inc + interest - borrow - financing
        income = div_inc + interest
        self._close_attr()
        self._finish_bar(k, cash_prev, hold, trade, sp, im, cm, borrow, financing, income)

    def _acc(self, i: int, comp: int, value: float, sign_hint: float) -> None:
        rec = self._bar_attr.get(i)
        if rec is None:
            rec = self._bar_attr[i] = [0.0, 0.0, 0.0, 0.0, 0.0, sign_hint]
        rec[comp] += value
        if rec[5] == 0.0:
            rec[5] = sign_hint

    def _close_attr(self) -> None:
        """Assign this bar's per-instrument PnL to the long or short book by the sign of the
        opening position (or, when the instrument opened flat, of the first fill)."""
        for i, rec in self._bar_attr.items():
            q0 = self._q0.get(i, 0.0)
            s = q0 if q0 != 0.0 else rec[5]
            b = 0 if s >= 0 else 1
            for c in range(5):
                self.attr[i, c] += rec[c]
                self.book[b, c] += rec[c]
        self._bar_attr = {}

    def _finish_bar(self, k, cash_prev, hold, trade, sp, im, cm, borrow, financing, income):
        pv = 0.0
        for i in range(self.N):
            if self.pos[i] != 0.0:
                pv += self.pos[i] * self.last_close[i]
        eq = self.cash + pv
        if k == 0:
            resid = abs(
                eq - self.equity_prev - (hold + trade - sp - im - cm - borrow - financing + income)
            )
        else:
            resid = abs(
                (eq - self.equity_prev)
                - (hold + trade - sp - im - cm - borrow - financing + income)
            )
        tol = IDENTITY_TOL * max(1.0, abs(self.equity_prev))
        self.max_residual = max(self.max_residual, resid / max(1.0, abs(self.equity_prev)))
        if resid > tol:
            raise AccountingError(f"identity violated at bar {k}: residual {resid:.3g}")
        self._row = [
            int(self.p.ts_event[k]),
            self.cash,
            pv,
            eq,
            hold,
            trade,
            sp,
            im,
            cm,
            borrow,
            financing,
            income,
        ]
        self.equity_rows.append(tuple(self._row))
        self._eq_before_bar = self.equity_prev
        self.equity_prev = eq
        te = int(self.p.ts_event[k])
        for i in range(self.N):
            if self.pos[i] != 0.0:
                self.positions_rows.append(
                    (
                        te,
                        self.p.ids[i],
                        self.pos[i],
                        self.last_close[i],
                        self.pos[i] * self.last_close[i],
                    )
                )

    def _eligible_fills(self, k: int, i: int, ref: str | None = None):
        p = self.p
        b = int(self.bar_no[k, i])
        vol = float(p.volume[k, i])
        used = self._cap_used.get((k, i), 0.0)
        cap_left = self.cfg.costs.participation_cap * vol - used
        if ref is None:
            ref = "open" if self.cfg.fill_model == "next_open" else "close"
        m = float(p.open[k, i] if ref == "open" else p.close[k, i])
        out = []
        keep: deque[_Order] = deque()
        for o in self.pending[i]:
            if b < o.elig_bar:
                keep.append(o)
                continue
            if vol == 0.0:
                if o.tif == "day":
                    self.E[i] = self.E[i] - o.qty
                    self.cancelled.append((k, o.order_id, p.ids[i], o.qty, "zero_volume"))
                else:
                    keep.append(o)
                continue
            fq = o.qty
            if abs(fq) > cap_left:
                fq = math.copysign(round_lot_scalar(cap_left, self.lot[i]), o.qty)
                self.cap_binds += 1
            if fq != 0.0:
                out.append((o, fq, m))
                cap_left -= abs(fq)
                o.qty = o.qty - fq
            if o.qty != 0.0:
                if o.tif == "day":
                    self.E[i] = self.E[i] - o.qty
                    self.cancelled.append((k, o.order_id, p.ids[i], o.qty, "cap_day"))
                else:
                    keep.append(o)
        self.pending[i] = keep
        self._cap_used[(k, i)] = used + sum(abs(x[1]) for x in out)
        return out

    # ------------------------------------------------------------------ decisions

    def portfolio_view(self) -> PortfolioView:
        pos = {self.p.ids[i]: self.pos[i] for i in range(self.N) if self.pos[i] != 0.0}
        return PortfolioView(self.cash, self.equity_prev, MappingProxyType(pos))

    def open_orders_view(self) -> tuple[OpenOrder, ...]:
        out = []
        for i in range(self.N):
            for o in self.pending[i]:
                out.append(OpenOrder(o.order_id, self.p.ids[i], o.qty, o.tif, o.ts_submit))
        return tuple(out)

    def targets_from_decision(self, d, universe_idx: list[int]) -> dict[int, float]:
        """Target quantities per instrument index (rounded toward zero to lots and the grid)."""
        cfg = self.cfg
        if isinstance(d, TargetWeights):
            base = self.equity_prev if cfg.sizing == "equity" else cfg.initial_cash
            pos_ids = {self.p.ids[i]: i for i in universe_idx}
            out = {}
            for i in universe_idx:
                out[i] = 0.0
            for iid, w in d.weights.items():
                if iid not in pos_ids:
                    self.rejected.append((self.k, iid, w, "not_in_universe"))
                    continue
                i = pos_ids[iid]
                c = self.last_close[i]
                if math.isnan(c) or c <= 0:
                    continue
                out[i] = round_lot_scalar(w * base / c, self.lot[i])
            return out
        if isinstance(d, TargetShares):
            pos_ids = {self.p.ids[i]: i for i in universe_idx}
            out = {i: 0.0 for i in universe_idx}
            for iid, q in d.shares.items():
                if iid not in pos_ids:
                    self.rejected.append((self.k, iid, q, "not_in_universe"))
                    continue
                i = pos_ids[iid]
                out[i] = round_lot_scalar(q, self.lot[i])
            return out
        raise TypeError("targets_from_decision needs target weights or shares")

    def submit(self, d, universe_ids: list[str]) -> None:
        """Turn a decision taken at the close of the current bar into orders."""
        if d is None:
            return
        k = self.k
        idx = {iid: i for i, iid in enumerate(self.p.ids)}
        uni = [idx[u] for u in universe_ids]
        if isinstance(d, Orders):
            for o in d.orders:
                i = idx.get(o.instrument_id)
                if i is None or i not in uni:
                    self.rejected.append((k, o.instrument_id, o.quantity, "not_in_universe"))
                    continue
                q = round_lot_scalar(o.quantity, self.lot[i])
                if q != 0.0:
                    self._new_order(i, q, o.tif, self.E[i] + q)
            return
        targets = self.targets_from_decision(d, uni)
        for i in range(self.N):
            if i not in targets:
                if self.E[i] != 0.0 and i not in uni:
                    targets[i] = 0.0
                else:
                    continue
        for i in sorted(targets):
            tq = targets[i]
            q = tq - self.E[i]
            if q == 0.0:
                continue
            self._new_order(i, q, "gtc", tq)

    def _new_order(self, i: int, q: float, tif: str, target: float) -> None:
        cfg = self.cfg
        k = self.k
        if not cfg.allow_short and target < 0:
            self.rejected.append((k, self.p.ids[i], q, "short_not_allowed"))
            return
        if cfg.max_leverage is not None:
            gross = 0.0
            for j in range(self.N):
                e = target if j == i else self.E[j]
                if e != 0.0:
                    gross += abs(e) * self.last_close[j]
            if self.equity_prev <= 0 or gross / self.equity_prev > cfg.max_leverage:
                self.rejected.append((k, self.p.ids[i], q, "max_leverage"))
                return
        b0 = int(self.last_at[k, i])
        if b0 < 0:
            self.rejected.append((k, self.p.ids[i], q, "no_bar_yet"))
            return
        delay = 0 if cfg.fill_model == "same_close" else cfg.fill_delay_bars
        self.n_orders += 1
        oid = f"o{self.n_orders}"
        ts_sub = int(self.p.ts_event[k])
        o = _Order(
            oid,
            i,
            q,
            tif,
            k,
            ts_sub,
            b0 + delay,
            float(self.liq_sigma[k, i]),
            float(self.liq_adv[k, i]),
            q,
        )
        self.pending[i].append(o)
        self.E[i] = target
        self.orders.append((oid, ts_sub, self.p.ids[i], q, "market", None, tif, self.sid))

    def fill_same_close(self) -> None:
        """UNSAFE: fill the orders just submitted at the close of the decision bar itself."""
        k = self.k
        if not self.cfg.unsafe:
            return
        if k == 0:
            raise ValueError(
                "same-close fills on the opening bar would put flows on the opening row"
            )
        items = []
        for i in self.bar_cols[k]:
            if self.pending[i]:
                items.extend(self._eligible_fills(k, i, ref="close"))
        if not items:
            return
        row = self._row
        tr, sp, im, cm, _ = self._book_fills(k, items)
        self._close_attr()
        pv = 0.0
        for i in range(self.N):
            if self.pos[i] != 0.0:
                pv += self.pos[i] * self.last_close[i]
        eq = self.cash + pv
        row[1], row[2], row[3] = self.cash, pv, eq
        row[5] += tr
        row[6] += sp
        row[7] += im
        row[8] += cm
        prev = self._eq_before_bar
        resid = abs(
            (eq - prev) - (row[4] + row[5] - row[6] - row[7] - row[8] - row[9] - row[10] + row[11])
        )
        if resid > IDENTITY_TOL * max(1.0, abs(prev)):
            raise AccountingError(f"identity violated at bar {k} (same-close fills)")
        self.equity_rows[-1] = tuple(row)
        self.equity_prev = eq
        te = int(self.p.ts_event[k])
        self.positions_rows = [r for r in self.positions_rows if r[0] != te]
        for i in range(self.N):
            if self.pos[i] != 0.0:
                self.positions_rows.append(
                    (
                        te,
                        self.p.ids[i],
                        self.pos[i],
                        self.last_close[i],
                        self.pos[i] * self.last_close[i],
                    )
                )

    def result(self) -> EventResult:
        comps = ("hold_pnl", "trade_pnl", "fill_costs", "dividends", "borrow")
        per = {
            self.p.ids[i]: {c: float(self.attr[i, j]) for j, c in enumerate(comps)}
            for i in range(self.N)
            if np.any(self.attr[i] != 0)
        }
        books = {
            name: {c: float(self.book[b, j]) for j, c in enumerate(comps)}
            for b, name in enumerate(("long", "short"))
        }
        return EventResult(
            ids=list(self.p.ids),
            ts_event=self.p.ts_event[: self.k + 1],
            equity_rows=self.equity_rows,
            positions_rows=self.positions_rows,
            orders=self.orders,
            fills=self.fills,
            rejected=self.rejected,
            cancelled=self.cancelled,
            max_residual=self.max_residual,
            unsafe=self.cfg.unsafe,
            fill_model=self.cfg.fill_model,
            impact_unavailable=self.impact_unavailable,
            per_instrument=per,
            books=books,
            cap_binds=self.cap_binds,
        )
