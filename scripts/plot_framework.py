"""Framework figure: the guarded data path, the two engines, the guards, and the outputs.

python scripts/plot_framework.py [--out docs/figures]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c0392b", "#8a8a85"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=SURF, ec=color, lw=1.6
        )
    )
    ax.text(x + 0.08, y + h - 0.1, title, fontsize=8.2, weight="bold", color=INK, va="top")
    ax.text(x + 0.08, y + h - 0.36, body, fontsize=6.6, color=INK2, va="top", linespacing=1.3)


def arrow(ax, a, b, text="", color=INK2, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.0,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if text:
        ax.text(
            (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.07, text, fontsize=6.2, color=INK2, ha="center"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "figures"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURF)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    box(
        ax,
        0.2,
        3.5,
        2.2,
        1.6,
        "Point-in-time store",
        "Parquet + DuckDB views\nts_event, ts_avail on every row\nvintaged series (as-of)\n"
        "raw prices + corporate actions\nsynthetic generator or local CSV",
        BLUE,
    )
    box(
        ax,
        2.9,
        3.5,
        2.3,
        1.6,
        "Knowledge cut at t",
        "rows with ts_avail <= t only\nuniverse incl. later delistings\nsplit / total-return adjustment\n"  # noqa: E501
        "as of t; read-only copies,\nno store reference (G1)",
        BLUE,
    )
    box(
        ax,
        5.7,
        3.5,
        2.1,
        1.6,
        "Strategy.decide(ctx)",
        "feature -> transform -> rule\npast-only by construction\nlint allow-list (G3)\n"
        "availability audit (G4)\ntarget weights / shares / orders",
        ORANGE,
    )
    box(
        ax,
        8.3,
        3.5,
        2.5,
        1.6,
        "EventEngine",
        "fills >= 1 bar later (G5)\ncap, gtc / day, zero volume\nsplits, dividends, delistings\n"
        "cost model v1, accruals\nidentity asserted every bar",
        ORANGE,
    )
    box(
        ax,
        8.3,
        1.3,
        2.5,
        1.6,
        "Vectorized reference",
        "run_target_shares on panels\nindependent implementation\nidentical fills, equity 1e-12\n"
        "E1: 1000 random scenarios\nindependent log validator",
        AQUA,
    )
    box(
        ax,
        5.2,
        1.3,
        2.7,
        1.6,
        "Inference",
        "Sharpe, Sortino, CVaR, Lo\nPSR, SR0, DSR + trial registry\nPBO (CSCV, 12870 splits)\n"
        "stationary bootstrap\npurged / walk-forward CV (G6)",
        AQUA,
    )
    box(
        ax,
        2.6,
        1.3,
        2.2,
        1.6,
        "Replay audit (G2)",
        "poisoned copy after t:\nprices, vintages, actions,\ndelistings, listings, manifest\n"
        "fresh import + purge per world\ndecisions must be bit-identical",
        RED,
    )
    box(
        ax,
        0.2,
        1.3,
        2.0,
        1.6,
        "Canaries C1..C8",
        "leak on purpose:\nnext close, by path, cached\nsample mean, last vintage,\n"
        "survivors, vendor adjust,\nearly release, same bar, CV",
        RED,
    )
    box(
        ax,
        0.2,
        0.1,
        10.6,
        0.8,
        "Export v1 (files only) -> decision layer",
        "instruments, bars, corporate_actions, returns, signals, forecasts, liquidity + manifests;  "  # noqa: E501
        "every row recomputed bit for bit from the store cut at its ts_avail",
        GREY,
    )
    arrow(ax, (2.4, 4.3), (2.9, 4.3), "query(t)")
    arrow(ax, (5.2, 4.3), (5.7, 4.3), "ctx")
    arrow(ax, (7.8, 4.3), (8.3, 4.3), "decision")
    arrow(ax, (9.55, 3.5), (9.55, 2.9), "same targets", AQUA)
    arrow(ax, (8.3, 2.1), (7.9, 2.1), "returns")
    arrow(ax, (2.2, 2.1), (2.6, 2.1), "subjects", RED)
    arrow(ax, (3.7, 2.9), (4.0, 3.5), "real vs poisoned", RED, rad=-0.2)
    arrow(ax, (1.3, 3.5), (1.3, 0.9), "", GREY)
    ax.text(
        5.5,
        5.3,
        "quant-research-engine: one guarded path from data to decision "
        "(research and simulation software; synthetic data)",
        ha="center",
        fontsize=9.5,
        color=INK,
        weight="bold",
    )
    p = os.path.join(a.out, "framework.png")
    fig.savefig(p, dpi=120, facecolor=SURF)
    plt.close(fig)
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
