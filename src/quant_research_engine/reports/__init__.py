"""Markdown and PNG report for one backtest run (the directory written by ``backtest``).

Figures: equity and drawdown, rolling Sharpe ratio (63 bars), gross against net cumulative
return, cumulative cost components, turnover, gross and net exposure. Tables: summary metrics,
cost components, attribution by instrument and by long and short book, data and configuration
hashes. A run labelled ``UNSAFE`` (same-close fills) is refused unless ``allow_unsafe`` is given,
and then every page and figure carries the banner.
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from quant_research_engine.data.csvio import load_csv  # noqa: E402
from quant_research_engine.data.manifest import config_hash, read_json  # noqa: E402
from quant_research_engine.metrics import summary  # noqa: E402

BANNER = "UNSAFE: same-bar fills (look-ahead in execution); not a result"
NOTE = "Backtested on synthetic data with assumed parameters. Not investment advice."


class UnsafeRunError(RuntimeError):
    pass


def _fig(path, unsafe, draw):
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=110)
    draw(ax)
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, NOTE, fontsize=6.5, color="#555555")
    if unsafe:
        fig.text(
            0.5,
            0.5,
            BANNER,
            fontsize=11,
            color="#c0392b",
            ha="center",
            va="center",
            alpha=0.85,
            rotation=12,
            weight="bold",
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def make_report(
    run_dir: str, out_dir: str, allow_unsafe: bool = False, store_path: str | None = None
) -> dict:
    run = read_json(os.path.join(run_dir, "run.json"))
    unsafe = run.get("label") == "UNSAFE"
    if unsafe and not allow_unsafe:
        raise UnsafeRunError(
            f"refusing to report an UNSAFE run ({run_dir}); pass --allow-unsafe to inspect it"
        )
    os.makedirs(out_dir, exist_ok=True)
    eq = load_csv(os.path.join(run_dir, "equity.csv"), "equity")
    warm = int(run.get("warmup_bars", 0))
    ppy = int(run.get("ppy", 252))
    E = eq["equity"]
    days = eq["ts_event"].astype("datetime64[us]")
    prev = E[warm:-1]
    r = E[warm + 1 :] / prev - 1.0
    gross = (eq["hold_pnl"][warm + 1 :] + eq["trade_pnl"][warm + 1 :]) / prev
    s = summary(r, E[warm:], ppy)
    sg = summary(gross, np.r_[1.0, np.cumprod(1 + gross)], ppy)
    d = days[warm + 1 :]
    figs = {}

    def save(name, draw):
        p = os.path.join(out_dir, f"{name}.png")
        _fig(p, unsafe, draw)
        figs[name] = p

    peak = np.maximum.accumulate(E[warm:])
    save(
        "equity_drawdown",
        lambda ax: (
            ax.plot(days[warm:], E[warm:], lw=1, label="equity"),
            ax.twinx().fill_between(
                days[warm:],
                -(peak - E[warm:]) / peak,
                color="#c0392b",
                alpha=0.25,
                label="drawdown",
            ),
            ax.set_title("Equity and drawdown (backtested, synthetic)"),
        ),
    )
    w = 63
    roll = np.full(r.size, np.nan)
    for i in range(w, r.size + 1):
        x = r[i - w : i]
        sd = x.std(ddof=1)
        roll[i - 1] = x.mean() / sd * math.sqrt(ppy) if sd > 0 else np.nan
    save(
        "rolling_sharpe",
        lambda ax: (
            ax.plot(d, roll, lw=1),
            ax.axhline(0, color="k", lw=0.6),
            ax.set_title(f"Rolling Sharpe ratio ({w} bars, annualized)"),
        ),
    )
    save(
        "gross_vs_net",
        lambda ax: (
            ax.plot(d, np.cumprod(1 + gross) - 1, lw=1, label="gross (before costs)"),
            ax.plot(d, np.cumprod(1 + r) - 1, lw=1, label="net"),
            ax.legend(fontsize=7),
            ax.set_title("Cumulative return: gross against net"),
        ),
    )
    comps = ("spread_cost", "impact_cost", "commission", "borrow", "financing")
    save(
        "cost_components",
        lambda ax: (
            [ax.plot(days, np.cumsum(eq[c]), lw=1, label=c) for c in comps],
            ax.legend(fontsize=7),
            ax.set_title("Cumulative costs (quote currency)"),
        ),
    )
    fills = load_csv(os.path.join(run_dir, "fills.csv"), "fills")
    turn = np.zeros(len(eq))
    pos_k = np.searchsorted(eq["ts_event"], fills["ts_fill"])
    np.add.at(turn, np.minimum(pos_k, len(eq) - 1), np.abs(fills["quantity"] * fills["ref_price"]))
    turn[1:] = turn[1:] / E[:-1]
    save(
        "turnover",
        lambda ax: (
            ax.plot(days, turn, lw=0.7),
            ax.set_title("Turnover per bar (traded / previous equity)"),
        ),
    )
    pos = load_csv(os.path.join(run_dir, "positions.csv"), "positions")
    gx = np.zeros(len(eq))
    nx = np.zeros(len(eq))
    pk = np.searchsorted(eq["ts_event"], pos["ts_event"])
    np.add.at(gx, pk, np.abs(pos["value"]))
    np.add.at(nx, pk, pos["value"])
    save(
        "exposure",
        lambda ax: (
            ax.plot(days, gx / E, lw=1, label="gross"),
            ax.plot(days, nx / E, lw=1, label="net"),
            ax.legend(fontsize=7),
            ax.set_title("Exposure (fraction of equity, at the close)"),
        ),
    )
    attr = []
    ap = os.path.join(run_dir, "attribution.csv")
    if os.path.exists(ap):
        with open(ap, encoding="utf-8") as f:
            lines = [x.split(",") for x in f.read().split("\n") if x]
        head = lines[0]
        attr = [dict(zip(head, x, strict=True)) for x in lines[1:]]
    store_note = ""
    if store_path and os.path.exists(os.path.join(store_path, "store.json")):
        sm = read_json(os.path.join(store_path, "store.json"))
        store_note = (
            f"store seed {sm.get('seed')}, generator {sm.get('generator', {}).get('name')}, "
            f"configuration hash {sm.get('config_hash', '')[:16]}, data: {sm.get('data', '')}"
        )
    md = []
    if unsafe:
        md.append(f"> **{BANNER}**\n")
    md.append(f"# Backtest report: {run.get('strategy', 'strategy')}\n")
    md.append(
        f"{NOTE} Simulated execution: {run['engine']['fill_model']} fills, fill delay "
        f"{run['engine']['fill_delay_bars']}, sizing {run['engine']['sizing']}, rebalance every "
        f"{run.get('rebalance_every')} bars after a warmup of {warm} bars.\n"
    )
    md.append("## Summary (net of costs)\n\n| metric | net | gross |\n|---|---|---|")
    for k in (
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "cvar_95",
        "skew",
        "kurt",
    ):
        md.append(f"| {k} | {s[k]:.4g} | {sg[k]:.4g} |")
    md.append("\n## Costs (sum over the run, quote currency)\n\n| component | total |\n|---|---|")
    for c in comps:
        md.append(f"| {c} | {float(np.sum(eq[c])):.2f} |")
    md.append(f"| income | {float(np.sum(eq['income'])):.2f} |")
    md.append(
        "\n## Attribution by book\n\n| book | hold_pnl | trade_pnl | fill_costs | dividends | borrow |\n|---|---|---|---|---|---|"  # noqa: E501
    )
    for book, v in sorted((run.get("books") or {}).items()):
        md.append(
            f"| {book} | "
            + " | ".join(
                f"{v[c]:.2f}"
                for c in ("hold_pnl", "trade_pnl", "fill_costs", "dividends", "borrow")
            )
            + " |"
        )
    if attr:
        md.append("\n## Attribution by instrument (largest absolute net contribution first)\n")
        md.append(
            "| instrument | hold_pnl | trade_pnl | fill_costs | dividends | borrow | net |\n|---|---|---|---|---|---|---|"  # noqa: E501
        )

        def net(a):
            return (
                float(a["hold_pnl"])
                + float(a["trade_pnl"])
                - float(a["fill_costs"])
                + float(a["dividends"])
                - float(a["borrow"])
            )

        for a in sorted(attr, key=lambda a: -abs(net(a)))[:15]:
            md.append(
                f"| {a['instrument_id']} | "
                + " | ".join(
                    f"{float(a[c]):.2f}"
                    for c in ("hold_pnl", "trade_pnl", "fill_costs", "dividends", "borrow")
                )
                + f" | {net(a):.2f} |"
            )
    md.append("\n## Data and configuration\n")
    md.append(f"- engine configuration hash: `{config_hash(run['engine'])}`")
    md.append(
        "- log hashes (SHA-256): "
        + ", ".join(f"{k} `{v[:16]}`" for k, v in sorted(run.get("files", {}).items()))
    )
    if store_note:
        md.append(f"- {store_note}")
    md.append(f"- largest accounting-identity residual: {run.get('max_identity_residual')}")
    md.append("\n## Figures\n")
    for name in figs:
        md.append(f"![{name}]({name}.png)")
    if unsafe:
        md.append(f"\n> **{BANNER}**")
    rp = os.path.join(out_dir, "report.md")
    with open(rp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    return {"report": rp, "figures": sorted(figs), "label": run.get("label")}
