"""Result figures from the committed experiment results.

python scripts/plot_results.py [--results experiments/results] [--out docs/figures] [--allow-quick]

Reads experiments/results/<id>/ (E2 leakage, E3 calibration, E4 costs and lag, E5 validation,
E5b CPCV,
Tier 2 capacity).
Quick results (manifest mode "quick") are refused unless --allow-quick is given; such figures
carry a QUICK stamp and never go into the README.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1", "#c0392b"]
NOTE = "Synthetic markets, assumed parameters; simulated / backtested; not investment advice."


def rows(path):
    with open(path, encoding="utf-8") as f:
        lines = [x for x in f.read().split("\n") if x]
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(","), strict=True)) for ln in lines[1:]]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def check_mode(d, allow_quick):
    m = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    if m.get("mode") == "quick" and not allow_quick:
        sys.exit(f"{d}: quick results; pass --allow-quick to draw them (stamped QUICK)")
    return m.get("mode") == "quick", m


def finish(fig, path, quick, commit):
    fig.text(
        0.01,
        0.005,
        NOTE + (f"  Results commit {commit[:10]}." if commit else ""),
        fontsize=6.5,
        color=INK2,
    )
    if quick:
        fig.text(
            0.5,
            0.5,
            "QUICK (development seeds, small sizes)",
            fontsize=22,
            color="#c0392b",
            alpha=0.35,
            ha="center",
            va="center",
            rotation=20,
            weight="bold",
        )
    fig.savefig(path, dpi=110, facecolor=SURF)
    plt.close(fig)
    print(path)


def leakage(d, out, allow_quick):
    quick, m = check_mode(d, allow_quick)
    table = rows(os.path.join(d, "audit_table.csv"))
    guards = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    fig, axs = plt.subplots(
        1, 3, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.15, 1, 1]}, facecolor=SURF
    )
    ax = axs[0]
    M = np.array([[1.0 if r[g] == "caught" else 0.0 for g in guards] for r in table])
    ax.imshow(M, cmap="Reds", vmin=0, vmax=1.6, aspect="auto")
    ax.set_xticks(range(len(guards)), guards, fontsize=7)
    ax.set_yticks(
        range(len(table)), [f"{r['subject_id']} {r['name']}" for r in table], fontsize=6.5
    )
    for i, r in enumerate(table):
        if r["named_guard"] != "-":
            j = guards.index(r["named_guard"])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
    ax.set_title("Audit table: caught (red); framed = guard named for the canary", fontsize=8)
    ax = axs[1]
    det = rows(os.path.join(d, "detection.csv"))
    for k, sid in enumerate(
        sorted({r["subject_id"] for r in det if r["subject_id"].startswith("C")})
    ):
        xs = [r for r in det if r["subject_id"] == sid]
        if all(num(r["detection_probability"]) == 0 for r in xs):
            continue
        ax.plot(
            [num(r["m"]) for r in xs],
            [num(r["detection_probability"]) for r in xs],
            marker="o",
            ms=3,
            lw=1,
            color=C[k % len(C)],
            label=sid,
        )
    ax.set_xscale("log")
    ax.set_xlabel("sampled decision instants m", fontsize=7)
    ax.set_ylabel("P(caught by the replay audit)", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5)
    ax.set_title("Replay audit (G2): detection probability vs m", fontsize=8)
    ax = axs[2]
    agg = rows(os.path.join(d, "sharpe_aggregates.csv"))
    labels, means, cis, colors = [], [], [], []
    for part, panel in (("sharpe", "gross"), ("sharpe", "net"), ("c7", "gross"), ("c7", "net")):
        for r in agg:
            if r["part"] == part and r["panel"] == panel:
                tag = "UNSAFE " if r["subject_id"] == "C7" else ""
                labels.append(f"{tag}{r['subject_id']} [{panel}]")
                means.append(num(r["sharpe_ann_mean"]))
                cis.append(num(r["sharpe_ann_ci95"]))
                colors.append(C[5] if r["subject_id"] in ("C1", "C1b", "C2", "C7") else C[0])
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=cis, color=colors, alpha=0.8, error_kw={"lw": 0.8})
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("annualized Sharpe ratio, mean and 95 % t interval over markets", fontsize=7)
    ax.set_title("Sharpe: leaks vs honest (null); C7 vs next_open (planted)", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    finish(fig, os.path.join(out, "leakage.png"), quick, m.get("git", {}).get("commit", ""))


def calibration(d, out, allow_quick):
    quick, m = check_mode(d, allow_quick)
    sel = rows(os.path.join(d, "selected.csv"))
    agg = rows(os.path.join(d, "aggregates.csv"))
    lo = [r for r in agg if r["kind"] == "lo"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), facecolor=SURF)
    ax = axs[0]
    for k, kind in enumerate(("null", "planted")):
        s = [r for r in sel if r["kind"] == kind and r["panel"] == "gross"]
        ax.scatter(
            [num(r["sr0"]) * math.sqrt(252) for r in s],
            [num(r["sr_bar"]) * math.sqrt(252) for r in s],
            s=6,
            alpha=0.6,
            color=C[k],
            label=f"{kind} (gross)",
        )
    lim = ax.get_xlim()
    ax.plot(lim, lim, color=INK, lw=0.6)
    ax.set_xlabel("SR0 (expected max of 108 trials), annualized", fontsize=7)
    ax.set_ylabel("selected trial's Sharpe, annualized", fontsize=7)
    ax.legend(fontsize=6.5)
    ax.set_title("(a) best of 108 trials vs SR0", fontsize=8)
    ax = axs[1]
    labs, vals, err = [], [], []
    for r in agg:
        if r["kind"] in ("null", "planted"):
            for name in ("naive_psr", "dsr"):
                p_ = num(r[f"{name}_gt_095_share"])
                labs.append(f"{r['kind']}/{r['panel']}\n{name}")
                vals.append(p_)
                err.append(
                    [
                        [max(0.0, p_ - num(r[f"{name}_gt_095_lo"]))],
                        [max(0.0, num(r[f"{name}_gt_095_hi"]) - p_)],
                    ]
                )
    x = np.arange(len(labs))
    ax.bar(x, vals, color=[C[1] if "naive" in lb else C[2] for lb in labs], alpha=0.85)
    ax.errorbar(x, vals, yerr=np.hstack(err) if err else None, fmt="none", ecolor=INK, lw=0.8)
    ax.set_xticks(x, labs, fontsize=5.5, rotation=60)
    ax.axhline(0.05, color=INK, lw=0.6, ls="--")
    ax.set_title("(b, d) share 'significant' at 0.95 (Wilson 95 %)", fontsize=8)
    ax = axs[2]
    edges = np.linspace(0, 1, 11)
    for k, (kind, panel) in enumerate(
        (("null", "gross"), ("planted", "gross"), ("null", "net"), ("planted", "net"))
    ):
        v = [num(r["pbo"]) for r in sel if r["kind"] == kind and r["panel"] == panel]
        if v:
            ax.hist(v, bins=edges, histtype="step", lw=1.2, color=C[k], label=f"{kind}/{panel}")
    ax.set_xlabel("PBO (CSCV, S = 16)", fontsize=7)
    ax.legend(fontsize=6.5)
    ax.set_title("(c) PBO across markets", fontsize=8)
    ax = axs[3]
    phis = [num(r["panel"].split("=")[1]) for r in lo]
    ax.errorbar(
        phis,
        [num(r["naive_ann_mean"]) for r in lo],
        yerr=[num(r["naive_ann_ci95"]) for r in lo],
        marker="o",
        ms=3,
        color=C[1],
        label="naive sqrt(252) x SR",
    )
    ax.errorbar(
        phis,
        [num(r["adjusted_ann_mean"]) for r in lo],
        yerr=[num(r["adjusted_ann_ci95"]) for r in lo],
        marker="s",
        ms=3,
        color=C[2],
        label="Lo-adjusted (L = 10)",
    )
    ax.plot(phis, [num(r["true_ann"]) for r in lo], color=INK, lw=1, ls="--", label="true")
    ax.set_xlabel("AR(1) coefficient", fontsize=7)
    ax.legend(fontsize=6.5)
    ax.set_title("(f) annualizing autocorrelated returns", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish(fig, os.path.join(out, "calibration.png"), quick, m.get("git", {}).get("commit", ""))


def costs_lag(d, out, allow_quick):
    quick, m = check_mode(d, allow_quick)
    costs = rows(os.path.join(d, "costs.csv"))
    lag = rows(os.path.join(d, "lag.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=SURF)
    ax = axs[0]
    for k, reb in enumerate(sorted({int(r["rebalance"]) for r in costs})):
        s = [r for r in costs if int(r["rebalance"]) == reb]
        ax.errorbar(
            [num(r["cost_mult"]) for r in s],
            [num(r["net_sharpe_mean"]) for r in s],
            yerr=[num(r["net_sharpe_ci95"]) for r in s],
            marker="o",
            ms=3,
            color=C[k],
            label=f"rebalance every {reb}",
        )
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("cost multiple of the base model", fontsize=7)
    ax.set_ylabel("net Sharpe, annualized (mean, 95 % t)", fontsize=7)
    ax.legend(fontsize=6.5)
    ax.set_title("E4(b) signal_x_ewma(3), long-short q 0.2: costs", fontsize=8)
    ax = axs[1]
    phis = sorted({num(r["persistence"]) for r in lag})
    variants = sorted(
        {(r["fill_model"], int(r["fill_delay"])) for r in lag},
        key=lambda v: (v[0] != "next_close", v[1]),
    )
    w = 0.8 / max(1, len(variants))
    for k, v in enumerate(variants):
        s = [
            next(
                (
                    r
                    for r in lag
                    if num(r["persistence"]) == ph and (r["fill_model"], int(r["fill_delay"])) == v
                ),
                None,
            )
            for ph in phis
        ]
        ax.bar(
            np.arange(len(phis)) + k * w,
            [num(r["net_sharpe_mean"]) if r else math.nan for r in s],
            w,
            yerr=[num(r["net_sharpe_ci95"]) if r else 0 for r in s],
            color=C[k % len(C)],
            alpha=0.85,
            label=f"{v[0]} delay {v[1]}",
            error_kw={"lw": 0.8},
        )
    ax.set_xticks(
        np.arange(len(phis)) + w * (len(variants) - 1) / 2, [f"phi = {p}" for p in phis], fontsize=7
    )
    ax.axhline(0, color=INK, lw=0.6)
    ax.legend(fontsize=6.5)
    ax.set_ylabel("net Sharpe, annualized (mean, 95 % t)", fontsize=7)
    ax.set_title("E4(c) fill delay and persistence (base costs, rebalance 5)", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish(fig, os.path.join(out, "costs_lag.png"), quick, m.get("git", {}).get("commit", ""))


def validation(d, out, allow_quick):
    quick, m = check_mode(d, allow_quick)
    agg = rows(os.path.join(d, "aggregates.csv"))
    schemes = ["shuffled_kfold", "contiguous_kfold", "purged_kfold", "walk_forward"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=SURF, sharey=True)
    for ax, kind in zip(axs, ("null", "planted"), strict=True):
        hs = sorted({int(r["h"]) for r in agg if r["kind"] == kind})
        w = 0.8 / len(schemes)
        for k, sc in enumerate(schemes):
            s = [
                next(
                    (
                        r
                        for r in agg
                        if r["kind"] == kind and int(r["h"]) == h and r["scheme"] == sc
                    ),
                    None,
                )
                for h in hs
            ]
            unp = [str(r["h"]) for r in s if r and r["label"] == "UNPURGED"]
            lab = sc + (f" (UNPURGED at h = {', '.join(unp)})" if unp else "")
            ax.bar(
                np.arange(len(hs)) + k * w,
                [num(r["bias_mean"]) if r else math.nan for r in s],
                w,
                yerr=[num(r["bias_ci95"]) if r else 0 for r in s],
                color=C[k],
                alpha=0.85,
                label=lab,
                error_kw={"lw": 0.8},
            )
        ax.set_xticks(np.arange(len(hs)) + w * 1.5, [f"h = {h}" for h in hs], fontsize=7)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(f"E5 {kind} markets: CV score minus held-out score", fontsize=8)
    axs[0].set_ylabel("bias of the Spearman score (mean, 95 % t)", fontsize=7)
    axs[0].legend(fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish(fig, os.path.join(out, "validation.png"), quick, m.get("git", {}).get("commit", ""))


def e5b_fig(d, out, allow_quick):
    """E5b: per-market bias of each scheme's out-of-sample Sharpe ratio (estimate minus held-out
    value); for CPCV every backtest path is a small point and the mean over paths a large one."""
    quick, m = check_mode(d, allow_quick)
    rs = rows(os.path.join(d, "rows.csv"))
    schemes = ["shuffled_kfold", "contiguous_kfold", "purged_kfold", "walk_forward", "cpcv"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=SURF, sharey=True)
    rng = np.random.default_rng(0)  # jitter only
    for ax, kind in zip(axs, ("null", "planted"), strict=True):
        labels = []
        for k, sc in enumerate(schemes):
            sel = [r for r in rs if r["kind"] == kind and r["scheme"] == sc]
            seeds = sorted({int(r["seed"]) for r in sel})
            per = [[num(r["bias_sharpe"]) for r in sel if int(r["seed"]) == s] for s in seeds]
            if sc == "cpcv":
                pts = [v for p_ in per for v in p_]
                ax.scatter(k + rng.uniform(-0.25, 0.25, len(pts)), pts, s=4, color=C[k], alpha=0.35)
            means = [float(np.mean(p_)) for p_ in per]
            ax.scatter(
                k + rng.uniform(-0.12, 0.12, len(means)), means, s=12, color=C[k], alpha=0.85
            )
            labels.append(f"{sc}\n({sel[0]['label']})" if sel else sc)
        ax.set_xticks(range(len(schemes)), labels, fontsize=7)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(f"E5b {kind} markets: CV Sharpe minus held-out Sharpe (h = 5)", fontsize=8)
    axs[0].set_ylabel("per-bar Sharpe ratio, bias per market", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish(fig, os.path.join(out, "cpcv.png"), quick, m.get("git", {}).get("commit", ""))


def capacity_fig(d, out, allow_quick):
    """Tier 2 capacity: net and gross Sharpe ratio against the initial capital (log scale), and
    the share of fill events at which the participation cap bound."""
    quick, m = check_mode(d, allow_quick)
    agg = rows(os.path.join(d, "aggregates.csv"))
    caps = np.asarray([num(r["capital"]) for r in agg])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=SURF)
    ax = axs[0]
    for k, (key, lab) in enumerate((("net_sharpe", "net (base costs)"), ("gross_sharpe", "gross"))):
        mu = np.asarray([num(r[f"{key}_mean"]) for r in agg])
        ci = np.asarray([num(r[f"{key}_ci95"]) for r in agg])
        ax.errorbar(caps, mu, yerr=ci, color=C[k], marker="o", ms=3, lw=1, capsize=2, label=lab)
    ax.set_xscale("log")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("initial capital (log scale)", fontsize=7)
    ax.set_ylabel("annualized Sharpe ratio (mean, 95 % t)", fontsize=7)
    ax.set_title("Capacity: signal_x_ls every 5 bars, square-root impact", fontsize=8)
    ax.legend(fontsize=6.5)
    ax = axs[1]
    ax.plot(
        caps,
        [num(r["cap_bound_share_mean"]) for r in agg],
        color=C[2],
        marker="o",
        ms=3,
        label="cap reached",
    )
    ax.plot(
        caps,
        [num(r["impact_drag_annual_mean"]) for r in agg],
        color=C[3],
        marker="o",
        ms=3,
        label="impact per year / equity",
    )
    ax.set_xscale("log")
    ax.set_xlabel("initial capital (log scale)", fontsize=7)
    ax.set_title("Share of fill events at the participation cap; impact drag", fontsize=8)
    ax.legend(fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish(fig, os.path.join(out, "capacity.png"), quick, m.get("git", {}).get("commit", ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join("experiments", "results"))
    ap.add_argument("--out", default=os.path.join("docs", "figures"))
    ap.add_argument("--allow-quick", action="store_true")
    ap.add_argument("--only", default="e2,e3,e4,e5,e5b,t2_capacity")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    todo = {
        "e2": leakage,
        "e3": calibration,
        "e4": costs_lag,
        "e5": validation,
        "e5b": e5b_fig,
        "t2_capacity": capacity_fig,
    }
    for exp in a.only.split(","):
        d = os.path.join(a.results, exp)
        if os.path.exists(os.path.join(d, "manifest.json")):
            todo[exp](d, a.out, a.allow_quick)
        else:
            print(f"skip {exp}: no results in {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
