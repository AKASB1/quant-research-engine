"""Markdown result tables from the committed experiment results (no computation of new results).

python scripts/make_tables.py [--results experiments/results] [--only e1,e2,e3,e4,e5] [--compact]
"""

# ruff: noqa: E501  (one table row per line reads better here)

from __future__ import annotations

import argparse
import csv
import json
import math
import os


def rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, d=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "n/a"
    if abs(v) < 0.5 * 10 ** (-d):
        v = 0.0  # no "-0.000" for a value that rounds to zero
    return f"{v:.{d}f}"


def pm(m, ci, d=2):
    return f"{f(m, d)} ± {f(ci, d)}"


def pct(p, lo, hi):
    return f"{100 * float(p):.1f} % [{100 * float(lo):.1f}, {100 * float(hi):.1f}]"


def e1(d):
    a = json.load(open(os.path.join(d, "aggregates.json"), encoding="utf-8"))
    out = ["| Measure | Value |", "|---|---|"]
    out.append(
        f"| scenarios (compared / excluded: cap binding or rejected order) | {a['scenarios']} ({a['compared']} / {len(a['excluded'])}) |"
    )
    out.append(f"| fill mismatches between the engines | {a['fills_mismatch']} |")
    out.append(
        "| largest relative difference of fill quantity, reference price, price, spread, impact, commission | "
        + ", ".join(
            f"{a['max_diff_' + k]:.1g}"
            for k in ("quantity", "ref_price", "price", "spread", "impact", "commission")
        )
        + " |"
    )
    out.append(f"| largest relative difference of equity | {a['max_diff_equity']:.1e} |")
    out.append(
        f"| largest accounting-identity residual (event / vector), relative to equity | {a['max_identity_residual_event']:.1e} / {a['max_identity_residual_vector']:.1e} |"
    )
    out.append(
        f"| independent validator failures (event logs / vector logs) | {a['validator_failures_event']} / {a['validator_failures_vector']} |"
    )
    out.append(
        f"| largest relative error found by the validator | {a['max_validator_rel_error']:.1e} |"
    )
    out.append(
        f"| scenarios with splits / dividends / delistings / late listings / zero-volume bars / lot sizes | "
        f"{a['scenarios_with_splits']} / {a['scenarios_with_dividends']} / {a['scenarios_with_delistings']} / "
        f"{a['scenarios_with_late_listings']} / {a['scenarios_with_zero_volume_bars']} / {a['scenarios_with_lots']} |"
    )
    out.append(
        f"| fill delay 1 / 2 / 3 bars; shorts on | {a['scenarios_fill_delay_1']} / {a['scenarios_fill_delay_2']} / "
        f"{a['scenarios_fill_delay_3']}; {a['scenarios_shorts_on']} |"
    )
    out.append(f"| violations | {a['violations']} |")
    return "\n".join(out)


def e2(d):
    t = rows(os.path.join(d, "audit_table.csv"))
    g = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    out = [
        "| Subject | Kind | Guard named | " + " | ".join(g) + " | G2 instants caught |",
        "|---|---|---|" + "---|" * len(g) + "---|",
    ]
    for r in t:
        cells = ["**caught**" if r[x] == "caught" else r[x] for x in g]
        out.append(
            f"| {r['subject_id']} {r['name']} | {r['kind']} | {r['named_guard']} | "
            + " | ".join(cells)
            + f" | {r['g2_caught']} / {r['g2_instants']} |"
        )
    det = rows(os.path.join(d, "detection.csv"))
    ms = sorted({int(r["m"]) for r in det})
    out2 = ["| Canary | " + " | ".join(f"m = {m}" for m in ms) + " |", "|---|" + "---|" * len(ms)]
    for sid in sorted({r["subject_id"] for r in det if r["subject_id"].startswith("C")}):
        v = {int(r["m"]): r["detection_probability"] for r in det if r["subject_id"] == sid}
        if all(float(x) == 0 for x in v.values()):
            continue
        out2.append(f"| {sid} | " + " | ".join(f(v[m]) for m in ms) + " |")
    agg = rows(os.path.join(d, "sharpe_aggregates.csv"))
    out3 = [
        "| Part | Panel | Subject | Annualized Sharpe (mean ± 95 % t) | Paired difference vs reference | Markets |",
        "|---|---|---|---|---|---|",
    ]
    for r in agg:
        sub = ("UNSAFE " if r["subject_id"] == "C7" else "") + r["subject_id"]
        out3.append(
            f"| {r['part']} | {r['panel']} | {sub} | {pm(r['sharpe_ann_mean'], r['sharpe_ann_ci95'])} | "
            f"{pm(r['diff_vs_reference_mean'], r['diff_vs_reference_ci95'])} | {r['n_markets']} |"
        )
    c7 = [r for r in agg if r["subject_id"] == "C7"]
    out4 = [
        "| Panel | Per-market ratio of Sharpe ratios, C7 / next_open (mean ± 95 % t) | Ratio of mean Sharpe ratios |",
        "|---|---|---|",
    ]
    for r in c7:
        out4.append(
            f"| {r['panel']} | {pm(r['sharpe_ratio_vs_next_open_mean'], r['sharpe_ratio_vs_next_open_ci95'])} | {f(r['ratio_of_mean_sharpe'])} |"
        )
    return (
        "\n".join(out)
        + "\n\nDetection probability of the replay audit with m sampled instants (100 draws from the 100 audited):\n\n"
        + "\n".join(out2)
        + "\n\nSharpe ratios (event engine, equity sizing, decision every bar):\n\n"
        + "\n".join(out3)
        + "\n\n"
        + "\n".join(out4)
    )


def e3(d):
    agg = rows(os.path.join(d, "aggregates.csv"))
    out = [
        "| Market / panel | Selected trial's per-bar Sharpe | SR0 (N = 108) | Selected / SR0 | Naive PSR(0) > 0.95 | DSR > 0.95 | Mean PBO | Selected trial uses signal_x |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in agg:
        if r["kind"] == "lo":
            continue
        out.append(
            f"| {r['kind']} / {r['panel']} | {pm(r['sr_bar_mean'], r['sr_bar_ci95'], 4)} | {pm(r['sr0_mean'], r['sr0_ci95'], 4)} | "
            f"{pm(r['sr_over_sr0_mean'], r['sr_over_sr0_ci95'])} | {pct(r['naive_psr_gt_095_share'], r['naive_psr_gt_095_lo'], r['naive_psr_gt_095_hi'])} | "
            f"{pct(r['dsr_gt_095_share'], r['dsr_gt_095_lo'], r['dsr_gt_095_hi'])} | {pm(r['pbo_mean'], r['pbo_ci95'])} | "
            f"{pct(r['selected_signal_x_share'], r['selected_signal_x_lo'], r['selected_signal_x_hi'])} |"
        )
    boot = [r for r in agg if r["kind"] == "null" and r["panel"] == "gross"][0]
    out2 = [
        "| Trial (null markets, gross) | Coverage of the 95 % bootstrap interval of the Sharpe ratio |",
        "|---|---|",
        f"| selected (best of 108 on the same data) | {pct(boot['boot_selected_coverage'], boot['boot_selected_lo'], boot['boot_selected_hi'])} |",
        f"| fixed in advance (momentum 21/5, q 0.2 equal, every 5 bars) | {pct(boot['boot_fixed_coverage'], boot['boot_fixed_lo'], boot['boot_fixed_hi'])} |",
    ]
    out3 = [
        "| AR(1) coefficient | True annualized Sharpe | Naive sqrt(252) x SR | Lo-adjusted (L = 10) |",
        "|---|---|---|---|",
    ]
    for r in agg:
        if r["kind"] == "lo":
            out3.append(
                f"| {r['panel'].split('=')[1]} | {f(r['true_ann'])} | {pm(r['naive_ann_mean'], r['naive_ann_ci95'])} | {pm(r['adjusted_ann_mean'], r['adjusted_ann_ci95'])} |"
            )
    return "\n".join(out) + "\n\n" + "\n".join(out2) + "\n\n" + "\n".join(out3)


def e4(d):
    a = json.load(open(os.path.join(d, "aggregates.json"), encoding="utf-8"))
    out = [
        "| Quantity (100 instruments x 2520 bars, 20 seeds) | Measured | Expected |",
        "|---|---|---|",
    ]
    out.append(
        f"| IC of the latent signal (oracle), planted_clean | {pm(a['clean_ic_latent']['mean'], a['clean_ic_latent']['ci95'], 4)} | {f(a['clean_ic_latent']['configured'], 4)} |"
    )
    out.append(
        f"| IC of signal_x, planted_clean | {pm(a['clean_ic_signal_x']['mean'], a['clean_ic_signal_x']['ci95'], 4)} | {f(a['clean_ic_signal_x']['configured'], 4)} |"
    )
    b = a["fl_bound_close_to_close"]
    out.append(
        f"| per-bar Sharpe, fundamental-law weights close to close (a bound) | {f(b['pooled_sr_bar'], 4)} (se {f(b['se'], 4)}) | {f(b['target_sr_bar'], 4)} |"
    )
    e = a["engine_next_open_oracle"]
    out.append(
        f"| per-bar Sharpe, same weights through the event engine, next_open, zero costs (oracle) | {f(e['pooled_sr_bar'], 4)} (se {f(e['se'], 4)}) | {f(e['target_sr_bar'], 4)} (bound x (1 - g (1 - phi))) |"
    )
    s = a["signal_x_over_latent"]
    out.append(
        f"| Sharpe of signal_x weights / Sharpe of latent weights (close to close) | {f(s['ratio'], 3)} | {f(s['expected'], 3)} |"
    )
    for k, lab in (
        ("planted_ic_latent", "IC of the latent signal (oracle), planted"),
        ("planted_ic_signal_x", "IC of signal_x, planted"),
        (
            "planted_ic_latent_raw",
            "IC of the latent signal vs the raw next-bar return, planted (labelled: not the estimator above)",
        ),
        (
            "planted_ic_signal_x_raw",
            "IC of signal_x vs the raw next-bar return, planted (labelled)",
        ),
    ):
        out.append(f"| {lab} | {pm(a[k]['mean'], a[k]['ci95'], 4)} | – |")
    c = rows(os.path.join(d, "costs.csv"))
    out2 = [
        "| Rebalance | Cost multiple | Net Sharpe (ann.) | Gross Sharpe (ann.) | Turnover per bar | Cost drag per year | Break-even multiple |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in c:
        be = r["break_even_multiple"]
        out2.append(
            f"| {r['rebalance']} | {r['cost_mult']} | {pm(r['net_sharpe_mean'], r['net_sharpe_ci95'])} | {pm(r['gross_sharpe_mean'], r['gross_sharpe_ci95'])} | "
            f"{f(r['turnover_per_bar_mean'], 3)} | {f(r['cost_drag_annual_mean'], 3)} | {f(be) if be[:1].isdigit() else be} |"
        )
    lg = rows(os.path.join(d, "lag.csv"))
    out3 = [
        "| Persistence | Fill model | Fill delay | Net Sharpe (ann.) | Gross Sharpe (ann.) |",
        "|---|---|---|---|---|",
    ]
    for r in lg:
        out3.append(
            f"| {r['persistence']} | {r['fill_model']} | {r['fill_delay']} | {pm(r['net_sharpe_mean'], r['net_sharpe_ci95'])} | {pm(r['gross_sharpe_mean'], r['gross_sharpe_ci95'])} |"
        )
    return "\n".join(out) + "\n\n" + "\n".join(out2) + "\n\n" + "\n".join(out3)


def e5_label(r):
    """The splitter labels a split UNPURGED only when training and test labels overlap; at h = 1
    the unpurged schemes have no overlap, which is not the same as being purged."""
    if r["scheme"] in ("shuffled_kfold", "contiguous_kfold") and r["label"] == "purged":
        return "not flagged (no overlap)"
    return r["label"]


def e5(d):
    agg = rows(os.path.join(d, "aggregates.csv"))
    out = [
        "| Market | h | Scheme | Label | Bias: CV minus held-out Spearman (mean ± 95 % t) | Spread of the bias (sd) | CV score | Held-out score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in agg:
        out.append(
            f"| {r['kind']} | {r['h']} | {r['scheme']} | {e5_label(r)} | {pm(r['bias_mean'], r['bias_ci95'], 3)} | {f(r['bias_sd'], 3)} | "
            f"{pm(r['cv_score_mean'], r['cv_score_ci95'], 3)} | {pm(r['heldout_score_mean'], r['heldout_score_ci95'], 3)} |"
        )
    return "\n".join(out)


def e4_compact(d):
    """README: net Sharpe ratio by rebalance interval and cost multiple, with the break-even multiple."""
    c = rows(os.path.join(d, "costs.csv"))
    mults = sorted({r["cost_mult"] for r in c}, key=float)
    out = [
        "| Rebalance / cost multiple | " + " | ".join(mults) + " | Break-even multiple |",
        "|---" * (len(mults) + 2) + "|",
    ]
    for reb in sorted({r["rebalance"] for r in c}, key=int):
        sel = {r["cost_mult"]: r for r in c if r["rebalance"] == reb}
        cells = [pm(sel[m]["net_sharpe_mean"], sel[m]["net_sharpe_ci95"]) for m in mults]
        if reb == "5":
            cells[mults.index("1.0")] = f"**{cells[mults.index('1.0')]}**"
        be = sel[mults[0]]["break_even_multiple"]
        out.append(
            f"| every {reb} | " + " | ".join(cells) + f" | {f(be) if be[:1].isdigit() else be} |"
        )
    return "\n".join(out)


def e5_compact(d):
    """README: the bias at h = 5 by scheme, null and planted markets."""
    agg = [r for r in rows(os.path.join(d, "aggregates.csv")) if r["h"] == "5"]
    out = [
        "| Scheme (h = 5) | Label | Bias, null markets | Bias, planted markets |",
        "|---|---|---|---|",
    ]
    for scheme in dict.fromkeys(r["scheme"] for r in agg):
        by = {r["kind"]: r for r in agg if r["scheme"] == scheme}
        out.append(
            f"| {scheme} | {by['null']['label']} | {pm(by['null']['bias_mean'], by['null']['bias_ci95'], 3)} | "
            f"{pm(by['planted']['bias_mean'], by['planted']['bias_ci95'], 3)} |"
        )
    return "\n".join(out)


def e5b(d):
    agg = rows(os.path.join(d, "aggregates.csv"))
    out = [
        "| Market | Scheme | Label | Spearman: bias (mean ± 95 % t) | sd of the bias | mean abs. bias | Sharpe per bar: bias (mean ± 95 % t) | sd of the bias | mean abs. bias | sd over CPCV paths within a market (Spearman / Sharpe) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in agg:
        within = "–"
        if r["scheme"] == "cpcv":
            within = f"{f(r['cv_spearman_sd_within_market_paths'], 3)} / {f(r['cv_sharpe_sd_within_market_paths'], 3)}"
        out.append(
            f"| {r['kind']} | {r['scheme']} | {r['label']} | {pm(r['bias_spearman_mean'], r['bias_spearman_ci95'], 3)} | {f(r['bias_spearman_sd'], 3)} | "
            f"{f(r['abs_bias_spearman_mean'], 3)} | {pm(r['bias_sharpe_mean'], r['bias_sharpe_ci95'], 3)} | {f(r['bias_sharpe_sd'], 3)} | "
            f"{f(r['abs_bias_sharpe_mean'], 3)} | {within} |"
        )
    j = json.load(open(os.path.join(d, "aggregates.json"), encoding="utf-8"))
    out2 = [
        "| Market, measure | Smallest mean absolute bias | Largest sd of the bias |",
        "|---|---|---|",
    ]
    for k in sorted(j):
        out2.append(f"| {k} | {j[k]['closest_to_heldout']} | {j[k]['noisiest']} |")
    return "\n".join(out) + "\n\n" + "\n".join(out2)


def t2_capacity(d):
    agg = rows(os.path.join(d, "aggregates.csv"))
    out = [
        "| Initial capital | Net Sharpe (ann.) | Gross Sharpe (ann.) | Impact per year / equity | All costs per year / equity | Fill events at the participation cap |",
        "|---|---|---|---|---|---|",
    ]
    for r in agg:
        out.append(
            f"| {float(r['capital']):.0e} | {pm(r['net_sharpe_mean'], r['net_sharpe_ci95'])} | {pm(r['gross_sharpe_mean'], r['gross_sharpe_ci95'])} | "
            f"{pm(r['impact_drag_annual_mean'], r['impact_drag_annual_ci95'], 3)} | {pm(r['cost_drag_annual_mean'], r['cost_drag_annual_ci95'], 3)} | "
            f"{pm(r['cap_bound_share_mean'], r['cap_bound_share_ci95'], 3)} |"
        )
    j = json.load(open(os.path.join(d, "aggregates.json"), encoding="utf-8"))

    def cap(v):
        return f"{v:.2e}" if isinstance(v, float) else str(v)

    out2 = [
        "| Capacity (log-linear interpolation of the mean net Sharpe ratio; first crossing) | Capital |",
        "|---|---|",
        f"| mean net Sharpe ratio at the smallest capital | {f(j['net_sharpe_at_smallest_capital'])} |",
        f"| capital at which it falls to half | {cap(j['capacity_half_sharpe'])} |",
        f"| capital at which it falls to zero | {cap(j['capacity_zero_sharpe'])} |",
    ]
    return "\n".join(out) + "\n\n" + "\n".join(out2)


def t2_trials(d):
    agg = rows(os.path.join(d, "aggregates.csv"))
    rules = (
        ("naive_psr", "naive PSR(0) > 0.95"),
        ("dsr_n108", "DSR, N = 108, > 0.95"),
        ("dsr_neff", "DSR, N = round(N_eff), > 0.95"),
        ("holm", "Holm, 0.05"),
        ("bh", "Benjamini-Hochberg, 0.05"),
    )
    out = [
        "| Rule | Null markets: share with a discovery (false positive) | Planted markets: share with a discovery involving signal_x (power) |",
        "|---|---|---|",
    ]
    by = {r["kind"]: r for r in agg}
    for key, lab in rules:
        cells = [
            pct(by[k][f"{key}_share"], by[k][f"{key}_lo"], by[k][f"{key}_hi"]) if k in by else "–"
            for k in ("null", "planted")
        ]
        out.append(f"| {lab} | {cells[0]} | {cells[1]} |")
    out2 = [
        "| Market | Markets | Effective number of trials N_eff: mean (min to max) |",
        "|---|---|---|",
    ]
    for r in agg:
        out2.append(
            f"| {r['kind']} | {r['markets']} | {f(r['n_eff_mean'], 1)} ({f(r['n_eff_min'], 1)} to {f(r['n_eff_max'], 1)}) |"
        )
    return "\n".join(out) + "\n\n" + "\n".join(out2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join("experiments", "results"))
    ap.add_argument("--only", default="e1,e2,e3,e4,e5")
    ap.add_argument(
        "--compact", action="store_true", help="the README versions of the E4 and E5 tables"
    )
    a = ap.parse_args(argv)
    fns = {
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
        "e5": e5,
        "e5b": e5b,
        "t2_capacity": t2_capacity,
        "t2_trials": t2_trials,
    }
    if a.compact:
        fns = {"e4": e4_compact, "e5": e5_compact}
    for exp in [x for x in a.only.split(",") if x in fns]:
        d = os.path.join(a.results, exp)
        m = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        print(f"<!-- {exp}: results commit {m['git']['commit'][:10]}, mode {m['mode']} -->")
        print(fns[exp](d))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
