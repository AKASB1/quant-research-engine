"""Independent accounting validator (does not import either engine).

Reads the CSV logs of one run (orders, fills, positions, equity, attribution, run.json) and the
store's bars, corporate actions, and liquidity rows, and recomputes bar by bar in plain Python:
split rescales, dividends, hold and trade PnL, fill costs from the cost model and the
decision-time liquidity, accruals, cash, positions, and equity. It confirms the accounting
identity, that cash and positions reconcile with the fills, that every fill comes after its
order's submission (at least one bar later unless the run is UNSAFE), that cash is never
negative while gross exposure stays at or below 1, and that the per-instrument attribution sums
to the totals.
"""

from __future__ import annotations

import bisect
import math
import os
from dataclasses import dataclass, field

from quant_research_engine.data.csvio import load_csv
from quant_research_engine.data.manifest import check_manifest, read_json

TOL = 1e-9


@dataclass
class ValidationReport:
    bars: int = 0
    fills: int = 0
    max_rel_error: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _load(log_dir: str, name: str):
    path = os.path.join(log_dir, f"{name}.csv")
    t = load_csv(path, name)
    with open(path, "rb") as f:
        check_manifest(path, f.read(), len(t))
    return t


def validate_run(log_dir: str, store) -> ValidationReport:
    rep = ValidationReport()
    run = read_json(os.path.join(log_dir, "run.json"))
    eng = run["engine"]
    costs = eng["costs"]
    unsafe = run.get("label") == "UNSAFE"
    eq = _load(log_dir, "equity")
    fills = _load(log_dir, "fills")
    pos_t = _load(log_dir, "positions")
    orders = _load(log_dir, "orders")

    bars = store.tables["bars"]
    by_inst: dict[str, dict[int, tuple]] = {}
    for i in range(len(bars)):
        by_inst.setdefault(str(bars["instrument_id"][i]), {})[int(bars["ts_event"][i])] = (
            int(bars["ts_open"][i]),
            float(bars["open"][i]),
            float(bars["close"][i]),
            float(bars["volume"][i]),
        )
    ev_sorted = {iid: sorted(d) for iid, d in by_inst.items()}
    ins = store.tables["instruments"]
    delist = {
        str(ins["instrument_id"][i]): (int(ins["ts_delist"][i]), float(ins["delist_return"][i]))
        for i in range(len(ins))
        if int(ins["ts_delist"][i]) != -1
    }
    fill_model = eng.get("fill_model", "next_open")
    delay = int(eng.get("fill_delay_bars", 1))
    cap_on = bool(run.get("participation_cap_enforced", True))
    opens_sorted = {iid: sorted((v[0], te) for te, v in d.items()) for iid, d in by_inst.items()}
    ratio: dict[tuple, float] = {}
    div: dict[tuple, float] = {}
    ca = store.tables["corporate_actions"]
    for i in range(len(ca)):
        iid = str(ca["instrument_id"][i])
        ex = int(ca["ts_ex"][i])
        bar = next((te for to, te in opens_sorted.get(iid, []) if to >= ex), None)
        if bar is None:
            continue
        if ca["action"][i] == "split":
            ratio[(iid, bar)] = ratio.get((iid, bar), 1.0) * float(ca["value"][i])
        else:
            div[(iid, bar)] = div.get((iid, bar), 0.0) + float(ca["value"][i])
    liq_rows: dict[str, list[tuple]] = {}
    liq = store.tables["liquidity"]
    for i in range(len(liq)):
        liq_rows.setdefault(str(liq["instrument_id"][i]), []).append(
            (
                int(liq["ts_avail"][i]),
                int(liq["ts_event"][i]),
                float(liq["adv_shares"][i]),
                float(liq["sigma_bar"][i]),
            )
        )
    order_info = {
        str(orders["order_id"][i]): (int(orders["ts_submit"][i]), str(orders["instrument_id"][i]))
        for i in range(len(orders))
    }

    for rows_ in liq_rows.values():
        rows_.sort(key=lambda x: x[1])
    liq_te = {iid: [x[1] for x in rows_] for iid, rows_ in liq_rows.items()}

    def liquidity_at(iid: str, t: int):
        """Latest row with ts_event <= t and ts_avail <= t (rows sorted by ts_event)."""
        rows_ = liq_rows.get(iid, [])
        j = bisect.bisect_right(liq_te.get(iid, []), t) - 1
        while j >= 0 and rows_[j][0] > t:
            j -= 1
        return (math.nan, math.nan) if j < 0 else (rows_[j][2], rows_[j][3])

    open_to_event = {iid: {to: te for to, te in lst} for iid, lst in opens_sorted.items()}

    def bar_of_fill(iid: str, ts_fill: int, at_open: bool) -> int | None:
        """The bar a fill belongs to: by its open for next_open fills, else by its close."""
        if at_open:
            return open_to_event.get(iid, {}).get(ts_fill)
        return ts_fill if ts_fill in by_inst.get(iid, {}) else None

    fills_by_bar: dict[int, list[int]] = {}
    for j in range(len(fills)):
        iid = str(fills["instrument_id"][j])
        at_open = fill_model == "next_open" and str(fills["order_id"][j]) != "DELIST"
        te = bar_of_fill(iid, int(fills["ts_fill"][j]), at_open)
        if te is None:
            rep.errors.append(f"fill {fills['fill_id'][j]} outside every bar of {iid}")
            continue
        fills_by_bar.setdefault(te, []).append(j)

    positions_at: dict[int, dict[str, float]] = {}
    for i in range(len(pos_t)):
        positions_at.setdefault(int(pos_t["ts_event"][i]), {})[str(pos_t["instrument_id"][i])] = (
            float(pos_t["quantity"][i])
        )

    ts = [int(x) for x in eq["ts_event"].tolist()]
    cash = float(eq["cash"][0])
    if abs(cash - float(eng["initial_cash"])) > TOL * max(1.0, cash):
        rep.errors.append("opening cash differs from the initial cash")
    pos: dict[str, float] = {}
    last_close: dict[str, float] = {}
    attr: dict[str, list[float]] = {}
    totals = [0.0] * 5
    hs = costs["half_spread_bps"]
    imp = costs["impact"]

    def add(iid, c, v):
        attr.setdefault(iid, [0.0] * 5)[c] += v
        totals[c] += v

    for r in range(len(ts)):
        e = ts[r]
        e_prev = float(eq["equity"][r - 1]) if r else float(eng["initial_cash"])
        tol = TOL * max(1.0, abs(e_prev))
        dt = (e - ts[r - 1]) / 86_400_000_000 if r else 0.0
        rate = costs["borrow_bps_annual"] / 1e4 * dt / 365.0
        borrow = 0.0
        for iid in sorted(pos):
            if pos[iid] < 0:
                b = -pos[iid] * last_close[iid] * rate
                borrow += b
                add(iid, 4, b)
        fin = max(0.0, -cash) * costs["financing_bps_annual"] / 1e4 * dt / 365.0
        interest = max(0.0, cash) * costs["cash_rate_bps_annual"] / 1e4 * dt / 365.0
        hold = dvd = 0.0
        active = sorted(iid for iid, d in by_inst.items() if e in d)
        pc_adj: dict[str, float] = {}
        for iid in active:
            q0 = pos.get(iid, 0.0)
            rt = ratio.get((iid, e), 1.0)
            pc = last_close.get(iid, math.nan)
            if rt != 1.0:
                pos[iid] = q0 * rt
                pc = pc / rt
            d = div.get((iid, e), 0.0)
            if d and q0:
                dvd += q0 * d
                add(iid, 3, q0 * d)
            pc_adj[iid] = pc
            c = by_inst[iid][e][2]
            if pos.get(iid, 0.0) != 0.0:
                h = pos[iid] * (c - pc)
                hold += h
                add(iid, 0, h)
        trade = sp = im = cm = 0.0
        filled: dict[str, float] = {}
        for j in sorted(
            fills_by_bar.get(e, []),
            key=lambda x: (x != x, str(fills["order_id"][x]) == "DELIST", x),
        ):
            iid = str(fills["instrument_id"][j])
            q = float(fills["quantity"][j])
            m = float(fills["ref_price"][j])
            price = float(fills["price"][j])
            s_, i_, c_ = (float(fills[k][j]) for k in ("spread_cost", "impact_cost", "commission"))
            c = by_inst[iid][e][2]
            oid = str(fills["order_id"][j])
            fid = fills["fill_id"][j]
            if oid == "DELIST":
                if s_ or i_ or c_ or price != m:
                    rep.errors.append(f"DELIST fill {fid} carries costs")
                d_ts, d_ret = delist.get(iid, (None, math.nan))
                if d_ts != e or abs(m - c * (1.0 + d_ret)) > 1e-12 * max(1.0, abs(m)):
                    rep.errors.append(
                        f"DELIST fill {fid} not at close * (1 + delist_return) of the last bar"
                    )
            else:
                # the reference price is the open (next_open) or the close of the fill bar
                o_, c_bar, v_ = by_inst[iid][e][1], by_inst[iid][e][2], by_inst[iid][e][3]
                exp_m = o_ if fill_model == "next_open" else c_bar
                if m != exp_m:
                    rep.errors.append(
                        f"fill {fid}: reference price {m!r} is not the bar's {fill_model} {exp_m!r}"
                    )
                if v_ <= 0:
                    rep.errors.append(f"fill {fid} in a bar without volume")
                filled[iid] = filled.get(iid, 0.0) + abs(q)
                if oid not in order_info:
                    rep.errors.append(f"fill {fills['fill_id'][j]} without order")
                    continue
                ts_sub, oi = order_info[oid]
                if oi != iid:
                    rep.errors.append(
                        f"fill {fills['fill_id'][j]} instrument differs from its order"
                    )
                tf = int(fills["ts_fill"][j])
                if (tf <= ts_sub) if not unsafe else (tf < ts_sub):
                    rep.errors.append(f"fill {fid} not after its order")
                # bars of the instrument between the decision and the fill (contract 3.5)
                evs = ev_sorted[iid]
                b_sub = bisect.bisect_right(evs, ts_sub) - 1
                b_fill = bisect.bisect_left(evs, e)
                need = 0 if unsafe else delay
                if b_sub < 0 or b_fill - b_sub < need:
                    rep.errors.append(
                        f"fill {fid} {b_fill - b_sub} bars after its decision (needs {need})"
                    )
                aq = abs(q)
                exp_sp = aq * m * hs / 1e4
                adv, sg = liquidity_at(iid, ts_sub)
                if math.isnan(adv) or math.isnan(sg) or imp["model"] == "none":
                    ibps = 0.0
                elif imp["model"] == "sqrt":
                    ibps = 1e4 * imp["y"] * sg * math.sqrt(aq / adv)
                else:
                    ibps = 1e4 * imp["y"] * sg * aq / adv
                exp_im = aq * m * ibps / 1e4
                exp_cm = max(
                    costs["min_commission"],
                    aq * m * costs["commission_bps"] / 1e4 + aq * costs["commission_per_share"],
                )
                exp_pr = m * (1.0 + math.copysign(1.0, q) * (hs + ibps) / 1e4)
                for name, a, b in (
                    ("spread", s_, exp_sp),
                    ("impact", i_, exp_im),
                    ("commission", c_, exp_cm),
                    ("price", price, exp_pr),
                ):
                    if abs(a - b) > 1e-9 * max(1.0, abs(b)):
                        rep.errors.append(f"fill {fills['fill_id'][j]}: {name} {a!r} != {b!r}")
            cash -= q * price + c_
            pos[iid] = pos.get(iid, 0.0) + q
            tp = q * (c - m)
            trade += tp
            sp += s_
            im += i_
            cm += c_
            add(iid, 1, tp)
            add(iid, 2, s_ + i_ + c_)
            rep.fills += 1
        if cap_on:
            cap = costs["participation_cap"]
            for iid, qsum in filled.items():
                if qsum > cap * by_inst[iid][e][3] * (1 + 1e-12) + 1e-9:
                    rep.errors.append(f"bar {r}: fills of {iid} exceed the participation cap")
        for iid in active:
            last_close[iid] = by_inst[iid][e][2]
        cash += dvd + interest - borrow - fin
        pv = sum(pos[i] * last_close[i] for i in sorted(pos) if pos[i] != 0.0)
        income = dvd + interest
        row = {k: float(eq[k][r]) for k in eq.schema.names if k != "ts_event"}
        mine = {
            "cash": cash,
            "position_value": pv,
            "equity": cash + pv,
            "hold_pnl": hold,
            "trade_pnl": trade,
            "spread_cost": sp,
            "impact_cost": im,
            "commission": cm,
            "borrow": borrow,
            "financing": fin,
            "income": income,
        }
        for k, v in mine.items():
            err = abs(v - row[k])
            rep.max_rel_error = max(rep.max_rel_error, err / max(1.0, abs(e_prev)))
            if err > tol:
                rep.errors.append(f"bar {r}: {k} {row[k]!r} != recomputed {v!r}")
        ident = (row["equity"] - e_prev) - (
            row["hold_pnl"]
            + row["trade_pnl"]
            - row["spread_cost"]
            - row["impact_cost"]
            - row["commission"]
            - row["borrow"]
            - row["financing"]
            + row["income"]
        )
        if r and abs(ident) > tol:
            rep.errors.append(f"bar {r}: identity residual {ident:.3g}")
        logged = positions_at.get(e, {})
        mine_pos = {i: q for i, q in pos.items() if abs(q) > 1e-12}
        if set(logged) != set(mine_pos):
            rep.errors.append(f"bar {r}: positions differ ({sorted(set(logged) ^ set(mine_pos))})")
        else:
            for i, q in mine_pos.items():
                if abs(q - logged[i]) > 1e-9 * max(1.0, abs(q)):
                    rep.errors.append(f"bar {r}: position {i} {logged[i]!r} != {q!r}")
        gross = sum(abs(pos[i] * last_close[i]) for i in pos if pos[i] != 0.0)
        eqv = cash + pv
        if eqv > 0 and gross / eqv <= 1.0 + 1e-12 and cash < -tol:
            rep.errors.append(f"bar {r}: negative cash without leverage")
        for i in list(pos):
            if pos[i] == 0.0:
                del pos[i]
        rep.bars += 1
    # attribution: the logged per-instrument file equals the recomputation, and sums to totals
    path = os.path.join(log_dir, "attribution.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")[1:]
        logged = {}
        for ln in lines:
            if ln:
                parts = ln.split(",")
                logged[parts[0]] = [float(x) for x in parts[1:]]
        scale = max(1.0, float(abs(eq["equity"]).max())) if len(eq) else 1.0
        for iid in sorted(set(logged) | set(attr)):
            a = logged.get(iid, [0.0] * 5)
            b = attr.get(iid, [0.0] * 5)
            for c in range(5):
                if abs(a[c] - b[c]) > TOL * scale:
                    rep.errors.append(f"attribution {iid} component {c}: {a[c]!r} != {b[c]!r}")
        sums = [math.fsum(v[c] for v in logged.values()) for c in range(5)]
        col_tot = [
            math.fsum(eq["hold_pnl"].tolist()),
            math.fsum(eq["trade_pnl"].tolist()),
            math.fsum((eq["spread_cost"] + eq["impact_cost"] + eq["commission"]).tolist()),
            None,
            math.fsum(eq["borrow"].tolist()),
        ]
        for c in (0, 1, 2, 4):
            if abs(sums[c] - col_tot[c]) > TOL * scale * max(1, len(eq)):
                rep.errors.append(f"attribution component {c} does not sum to the total")
    return rep
