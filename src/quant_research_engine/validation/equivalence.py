"""Running both engines on one target-share schedule and comparing them (check 7, E1)."""

from __future__ import annotations

import os

import numpy as np

from quant_research_engine.context.decision import TargetShares
from quant_research_engine.costs import CostConfig
from quant_research_engine.data.csvio import make_table, write_csv
from quant_research_engine.data.manifest import dataset_manifest, manifest_path, write_json
from quant_research_engine.engine.event import EngineConfig, EventEngine, EventResult
from quant_research_engine.vector import VectorResult, run_target_shares


def run_event_targets(
    store, dec_idx, targets, cfg: EngineConfig, strategy_id: str = "e1"
) -> EventResult:
    p = store.panel()
    eng = EventEngine(p, cfg, strategy_id, calendar=store.calendar_name)
    where = {int(k): j for j, k in enumerate(np.asarray(dec_idx).tolist())}
    for k in range(p.shape[0]):
        eng.process_bar(k)
        j = where.get(k)
        if j is not None:
            uni = store.instruments_at(int(p.ts_event[k]))
            sh = {p.ids[i]: float(targets[j, i]) for i in range(len(p.ids)) if targets[j, i] != 0}
            eng.submit(TargetShares(sh), uni)
    return eng.result()


def run_vector_targets(store, dec_idx, targets, cfg: EngineConfig) -> VectorResult:
    return run_target_shares(
        store.panel(),
        dec_idx,
        targets,
        fill_model=cfg.resolved(store.calendar_name).fill_model,
        fill_delay_bars=cfg.fill_delay_bars,
        initial_cash=cfg.initial_cash,
        costs=cfg.costs,
    )


def _rel(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))))


def compare(ev: EventResult, vr: VectorResult, panel, fill_model: str) -> dict:
    """Largest relative differences of fills (instant, quantity, prices, costs) and equity."""
    ts_of = panel.ts_open if fill_model == "next_open" else panel.ts_event
    vf = vr.fills
    vrows = []
    for n in range(len(vf["k_fill"])):
        k = int(vf["k_fill"][n])
        dl = bool(vf["delist"][n])
        ts = int(panel.ts_event[k]) if dl else int(ts_of[k])
        vrows.append(
            (
                ts,
                panel.ids[int(vf["inst"][n])],
                dl,
                float(vf["quantity"][n]),
                float(vf["ref_price"][n]),
                float(vf["price"][n]),
                float(vf["spread"][n]),
                float(vf["impact"][n]),
                float(vf["commission"][n]),
            )
        )
    erows = [(f[2], f[3], f[1] == "DELIST", f[4], f[5], f[6], f[7], f[8], f[9]) for f in ev.fills]
    key = lambda r: (r[0], r[1], r[2])  # noqa: E731
    vrows.sort(key=key)
    erows.sort(key=key)
    out = {
        "n_fills_event": len(erows),
        "n_fills_vector": len(vrows),
        "fills_match": len(erows) == len(vrows),
    }
    if out["fills_match"]:
        same_keys = all(key(a) == key(b) for a, b in zip(erows, vrows, strict=True))
        out["fills_match"] = same_keys
        if same_keys and erows:
            cols = list(zip(*erows, strict=True)), list(zip(*vrows, strict=True))
            for c, name in enumerate(
                ("quantity", "ref_price", "price", "spread", "impact", "commission"), start=3
            ):
                out[f"max_diff_{name}"] = _rel(cols[0][c], cols[1][c])
    eq_e = ev.equity
    eq_v = vr.equity[: len(eq_e)]
    out["max_diff_equity"] = _rel(eq_e, eq_v)
    return out


def write_vector_logs(
    vr: VectorResult, panel, dec_idx, cfg: EngineConfig, out_dir: str, strategy_id: str = "e1"
) -> None:
    """The four result files and run.json of a vectorized run (orders: target differences)."""
    os.makedirs(out_dir, exist_ok=True)
    fm = vr.fill_model
    ts_of = panel.ts_open if fm == "next_open" else panel.ts_event
    o = vr.orders
    od = sorted(
        (
            (
                f"v{int(j)}_{int(i)}",
                int(panel.ts_event[dec_idx[int(j)]]),
                panel.ids[int(i)],
                float(q),
            )
            for j, i, q in zip(o["dec"], o["inst"], o["quantity"], strict=True)
        ),
        key=lambda r: (r[1], r[0]),
    )
    orders = make_table(
        "orders",
        order_id=[r[0] for r in od],
        ts_submit=[r[1] for r in od],
        instrument_id=[r[2] for r in od],
        quantity=[r[3] for r in od],
        order_type=["market"] * len(od),
        limit_price=[None] * len(od),
        tif=["gtc"] * len(od),
        strategy_id=[strategy_id] * len(od),
    )
    f = vr.fills
    rows = []
    for n in range(len(f["k_fill"])):
        k, i = int(f["k_fill"][n]), int(f["inst"][n])
        dl = bool(f["delist"][n])
        rows.append(
            (
                int(panel.ts_event[k]) if dl else int(ts_of[k]),
                n,
                "DELIST" if dl else f"v{int(f['dec'][n])}_{i}",
                panel.ids[i],
                float(f["quantity"][n]),
                float(f["ref_price"][n]),
                float(f["price"][n]),
                float(f["spread"][n]),
                float(f["impact"][n]),
                float(f["commission"][n]),
            )
        )
    rows = [(f"f{r[1] + 1}",) + r for r in rows]
    rows.sort(key=lambda r: (r[1], r[0]))
    fills = make_table(
        "fills",
        fill_id=[r[0] for r in rows],
        order_id=[r[3] for r in rows],
        ts_fill=[r[1] for r in rows],
        instrument_id=[r[4] for r in rows],
        quantity=[r[5] for r in rows],
        ref_price=[r[6] for r in rows],
        price=[r[7] for r in rows],
        spread_cost=[r[8] for r in rows],
        impact_cost=[r[9] for r in rows],
        commission=[r[10] for r in rows],
    )
    T, N = vr.positions.shape
    lc = panel.close.copy()
    for k in range(1, T):
        m = np.isnan(lc[k])
        lc[k, m] = lc[k - 1, m]
    pr = [
        (
            int(panel.ts_event[k]),
            panel.ids[i],
            float(vr.positions[k, i]),
            float(lc[k, i]),
            float(vr.positions[k, i] * lc[k, i]),
        )
        for k in range(T)
        for i in range(N)
        if vr.positions[k, i] != 0
    ]
    positions = make_table(
        "positions",
        ts_event=[r[0] for r in pr],
        instrument_id=[r[1] for r in pr],
        quantity=[r[2] for r in pr],
        mark_price=[r[3] for r in pr],
        value=[r[4] for r in pr],
    )
    equity = make_table(
        "equity",
        ts_event=[int(x) for x in panel.ts_event],
        cash=vr.cash.tolist(),
        position_value=vr.position_value.tolist(),
        equity=vr.equity.tolist(),
        hold_pnl=vr.hold_pnl.tolist(),
        trade_pnl=vr.trade_pnl.tolist(),
        spread_cost=vr.spread_cost.tolist(),
        impact_cost=vr.impact_cost.tolist(),
        commission=vr.commission.tolist(),
        borrow=vr.borrow.tolist(),
        financing=vr.financing.tolist(),
        income=vr.income.tolist(),
    )
    gen = {"name": "qre.vector", "label": "safe", "parameters": {"strategy_id": strategy_id}}
    for name, t in (
        ("orders", orders),
        ("fills", fills),
        ("positions", positions),
        ("equity", equity),
    ):
        path = os.path.join(out_dir, f"{name}.csv")
        data = write_csv(path, t)
        write_json(manifest_path(path), dataset_manifest(data, len(t), gen, None))
    write_json(
        os.path.join(out_dir, "run.json"),
        {
            "label": "safe",
            "participation_cap_enforced": False,
            "engine": {
                **cfg.model_dump(mode="json"),
                "fill_model": fm,
            },
            "max_identity_residual": vr.max_residual,
        },
    )


def scenario_config(sc, costs: CostConfig | None = None) -> EngineConfig:
    return EngineConfig(
        fill_model="next_open",
        fill_delay_bars=sc.fill_delay,
        allow_short=sc.allow_short,
        initial_cash=1_000_000.0,
        costs=costs or CostConfig(),
    )
