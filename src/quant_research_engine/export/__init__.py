"""Export v1: the files the decision layer reads (no code from this repository is needed).

A directory with ``manifest.json`` and ``instruments.csv``, ``bars.csv``,
``corporate_actions.csv``, ``returns.csv``, ``signals.csv``, ``forecasts.csv``,
``liquidity.csv`` (each with its ``.manifest.json``) and the optional ``series.csv``. Payload:

- ``returns``: the simple total returns of contract 2.5 (known from the bar's ts_avail);
- ``signals``: ``signal_x`` and every built-in feature at its default parameters (raw values,
  computed at the bar's close from data known then), for the instruments in the universe;
- ``forecasts``: horizons 1, 5, 21; ``mu = a + b x`` from an expanding-window, past-only ridge
  regression of the forward h-bar compounded return on ``x = signal_x_ewma(3)``, pooled over
  instruments, using only observations whose label is realized by the row's instant (slope
  penalty 10, intercept unpenalized, at least 500 observations); ``sigma = sigma_bar sqrt(h)``;
- ``liquidity``: the store's liquidity rows (one pure function, see the store).

Every row is recomputed bit for bit by the same functions from the store cut at the row's
``ts_avail`` (tested). No oracle value is exported. A store from a third-party source is refused
unless ``--derived-only`` is given with the source's terms; ``bars.csv`` and
``corporate_actions.csv`` are then header-only and ``derived_only`` is true.
"""

from __future__ import annotations

import os

import numpy as np

from quant_research_engine import QC_VERSION, __version__
from quant_research_engine.data.csvio import (
    Table,
    empty_table,
    load_csv,
    make_table,
    validate_table,
    write_csv,
)
from quant_research_engine.data.manifest import (
    check_manifest,
    dataset_manifest,
    manifest_path,
    read_json,
    sha256_file,
    write_json,
)
from quant_research_engine.data.schemas import SCHEMAS
from quant_research_engine.features import PanelSource, make_feature
from quant_research_engine.strategies import BUILTIN_FEATURES

EXPORT_VERSION = 1
FILES = ("instruments", "bars", "corporate_actions", "returns", "signals", "forecasts", "liquidity")
HORIZONS = (1, 5, 21)
RIDGE_LAMBDA = 10.0
MIN_OBS = 500
NEVER = np.iinfo(np.int64).max


def _feature_name(name: str, params: dict) -> str:
    return name + "".join(f"_{v}" for _, v in sorted(params.items()))


def returns_table(store) -> Table:
    p = store.panel()
    k, j = np.nonzero(~np.isnan(p.ret))
    order = np.lexsort((j, k))
    k, j = k[order], j[order]
    return Table(
        SCHEMAS["returns"],
        {
            "ts_event": p.ts_event[k].astype(np.int64),
            "ts_avail": p.avail[k, j].astype(np.int64),
            "instrument_id": np.asarray([p.ids[x] for x in j.tolist()], dtype=object),
            "ret": p.ret[k, j],
        },
    )


def signal_panels(store) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, values (T, N), ts_avail (T, N)) for signal_x and every built-in feature."""
    src = PanelSource(store)
    p = store.panel()
    out = []
    track = PanelSource(store, track=True)
    x = track.series_asof(".signal_x")
    out.append(("signal_x", x, track.used[-1]))
    for _, name, params in BUILTIN_FEATURES:
        f = make_feature(name, **params)
        out.append((_feature_name(name, params), f.panel(src), f.declared_availability(src)))
    del p
    return out


def signals_table(store) -> Table:
    p = store.panel()
    uni = PanelSource(store).universe_mask()
    rows = []
    for name, vals, av in signal_panels(store):
        ok = uni & np.isfinite(vals) & p.has_bar
        k, j = np.nonzero(ok)
        rows.append((k, j, name, vals[k, j], av[k, j]))
    ks = np.concatenate([r[0] for r in rows]) if rows else np.zeros(0, np.int64)
    js = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0, np.int64)
    names = (
        np.concatenate([np.full(r[0].size, r[2], dtype=object) for r in rows])
        if rows
        else np.zeros(0, object)
    )
    vals = np.concatenate([r[3] for r in rows]) if rows else np.zeros(0)
    avs = np.concatenate([r[4] for r in rows]) if rows else np.zeros(0, np.int64)
    ids = np.asarray(p.ids, dtype=object)[js]
    order = sorted(range(len(ks)), key=lambda i: (int(ks[i]), names[i], ids[i]))
    o = np.asarray(order, dtype=np.int64)
    return Table(
        SCHEMAS["signals"],
        {
            "ts_event": p.ts_event[ks[o]].astype(np.int64),
            "ts_avail": avs[o].astype(np.int64),
            "instrument_id": ids[o],
            "name": names[o],
            "value": vals[o],
        },
    )


def forecast_panels(store, h: int):
    """(mu (T, N), sigma (T, N)) at every bar from data known at the bar's close."""
    p = store.panel()
    T, N = p.shape
    src = PanelSource(store)
    x = make_feature("signal_x_ewma", halflife=3).panel(src)
    lr = np.log1p(p.ret)
    # label of observation j: compounded return of bars j+1 .. j+h (NaN if any is missing)
    y = np.full((T, N), np.nan)
    if T > h:
        acc = np.zeros((T - h, N))
        for lag in range(1, h + 1):
            acc = acc + lr[lag : T - h + lag]
        y[: T - h] = np.expm1(acc)
    ok = np.isfinite(x) & np.isfinite(y)
    s = [np.zeros(T) for _ in range(5)]  # n, sx, sy, sxx, sxy per observation bar, fixed order
    for j in range(N):
        m = ok[:, j]
        xj = np.where(m, x[:, j], 0.0)
        yj = np.where(m, y[:, j], 0.0)
        s[0] = s[0] + m
        s[1] = s[1] + xj
        s[2] = s[2] + yj
        s[3] = s[3] + xj * xj
        s[4] = s[4] + xj * yj
    cum = [np.cumsum(v) for v in s]
    mu = np.full((T, N), np.nan)
    for k in range(h, T):
        j = k - h  # observations j' <= k - h have labels realized by bar k
        n = cum[0][j]
        if n < MIN_OBS:
            continue
        mx, my = cum[1][j] / n, cum[2][j] / n
        sxx = cum[3][j] - n * mx * mx
        sxy = cum[4][j] - n * mx * my
        b = sxy / (sxx + RIDGE_LAMBDA)
        a = my - b * mx
        mu[k] = a + b * x[k]
    from quant_research_engine.engine.panelutil import liquidity_asof_panel

    _, sig = liquidity_asof_panel(p)
    return mu, sig * np.sqrt(h)


def forecasts_table(store) -> Table:
    p = store.panel()
    uni = PanelSource(store).universe_mask()
    rows = []
    for h in HORIZONS:
        mu, sg = forecast_panels(store, h)
        ok = uni & np.isfinite(mu) & p.has_bar
        k, j = np.nonzero(ok)
        rows.append((k, j, np.full(k.size, h), mu[k, j], sg[k, j]))
    ks = np.concatenate([r[0] for r in rows])
    js = np.concatenate([r[1] for r in rows])
    hs = np.concatenate([r[2] for r in rows]).astype(np.int64)
    mus = np.concatenate([r[3] for r in rows])
    sgs = np.concatenate([r[4] for r in rows])
    ids = np.asarray(p.ids, dtype=object)[js]
    o = np.asarray(
        sorted(range(len(ks)), key=lambda i: (int(ks[i]), ids[i], int(hs[i]))), dtype=np.int64
    )
    if o.size == 0:
        return empty_table(SCHEMAS["forecasts"])
    return Table(
        SCHEMAS["forecasts"],
        {
            "ts_event": p.ts_event[ks[o]].astype(np.int64),
            "ts_avail": p.avail[ks[o], js[o]].astype(np.int64),
            "instrument_id": ids[o],
            "horizon_bars": hs[o],
            "mu": mus[o],
            "sigma": sgs[o],
        },
    )


def write_tables(tables: dict[str, Table], out: str, generator: dict, seed) -> dict:
    os.makedirs(out, exist_ok=True)
    files = {}
    for name, t in tables.items():
        path = os.path.join(out, f"{name}.csv")
        data = write_csv(path, t)
        m = dataset_manifest(data, len(t), generator, seed)
        write_json(manifest_path(path), m)
        files[f"{name}.csv"] = {"content_sha256": m["content_sha256"], "row_count": len(t)}
    return files


def build_export_tables(store, derived_only: bool = False) -> dict[str, Table]:
    t = {
        "instruments": store.tables["instruments"],
        "bars": empty_table(SCHEMAS["bars"]) if derived_only else store.tables["bars"],
        "corporate_actions": empty_table(SCHEMAS["corporate_actions"])
        if derived_only
        else store.tables["corporate_actions"],
        "returns": returns_table(store),
        "signals": signals_table(store),
        "forecasts": forecasts_table(store),
        "liquidity": store.tables["liquidity"],
    }
    if not derived_only:
        t["series"] = store.tables["series"]
    return t


def export_from_store(
    store, out: str, derived_only: bool = False, terms: str | None = None, commit: str | None = None
) -> dict:
    source = store.meta.get("source")
    if source and not (derived_only and terms):
        raise ValueError(
            "a store from a third-party source exports only derived values: pass --derived-only and --terms"  # noqa: E501
        )
    gen = store.meta.get("generator") or {"name": "source", "parameters": source}
    tables = build_export_tables(store, derived_only)
    files = write_tables(tables, out, gen, store.meta.get("seed"))
    if commit is None:
        from quant_research_engine.experiments.common import git_state

        commit = git_state().get("commit")
    man = {
        "export_version": EXPORT_VERSION,
        "qc_version": QC_VERSION,
        "producer": "quant-research-engine",
        "producer_version": __version__,
        "producer_commit": commit,
        "calendar": store.calendar_name,
        "ppy": store.ppy,
        "seed": store.meta.get("seed"),
        "generator": store.meta.get("generator"),
        "source": source,
        "terms": terms,
        "derived_only": bool(derived_only),
        "files": files,
        "data": store.meta.get("data", "") or ("derived values only" if derived_only else ""),
        "payload": {
            "signals": ["signal_x"] + [_feature_name(n, p) for _, n, p in BUILTIN_FEATURES],
            "forecast_horizons": list(HORIZONS),
            "forecast_model": f"expanding ridge, lambda {RIDGE_LAMBDA}, "
            f"min {MIN_OBS} observations, x = signal_x_ewma(3)",
            "liquidity": "adv 20 bars, sigma 250 bars, half-life 20",
        },
    }
    write_json(os.path.join(out, "manifest.json"), man)
    return {
        "out": out,
        "files": {k: v["row_count"] for k, v in files.items()},
        "derived_only": bool(derived_only),
    }


def export_store(args, out: str) -> dict:
    from quant_research_engine.store import read_store
    from quant_research_engine.synth import generate, load_synth_config

    if args.store:
        st = read_store(args.store)
    else:
        cfg = load_synth_config(args.config)
        over = {
            k: v for k, v in (("n_instruments", args.n_instruments), ("n_bars", args.n_bars)) if v
        }
        st, _ = generate(cfg.with_overrides(**over) if over else cfg, args.seed)
    return export_from_store(st, out, args.derived_only, args.terms)


def validate_export(path: str) -> dict:
    man = read_json(os.path.join(path, "manifest.json"))
    errors, rows = [], {}
    if man.get("export_version") != EXPORT_VERSION or man.get("qc_version") != QC_VERSION:
        errors.append("unsupported export or contract version")
    tables = {}
    for name in FILES + (("series",) if "series.csv" in man.get("files", {}) else ()):
        fp = os.path.join(path, f"{name}.csv")
        try:
            t = load_csv(fp, name)
            with open(fp, "rb") as f:
                data = f.read()
            check_manifest(fp, data, len(t))
            if man["files"][f"{name}.csv"]["content_sha256"] != sha256_file(fp):
                errors.append(f"{name}.csv: hash differs from manifest.json")
            tables[name] = t
            rows[name] = len(t)
        except (ValueError, KeyError, OSError) as e:
            errors.append(f"{name}.csv: {e}")
    if man.get("derived_only"):
        for name in ("bars", "corporate_actions"):
            if rows.get(name):
                errors.append(f"{name}.csv must be header-only in a derived-only export")
    elif "bars" in tables and "instruments" in tables:
        b, ins = tables["bars"], tables["instruments"]
        last = {}
        for i in range(len(b)):
            last[str(b["instrument_id"][i])] = int(b["ts_event"][i])
        for i in range(len(ins)):
            d = int(ins["ts_delist"][i])
            if d != -1 and last.get(str(ins["instrument_id"][i])) != d:
                errors.append(
                    f"instrument {ins['instrument_id'][i]}: ts_delist is not its last bar"
                )
    return {"ok": not errors, "kind": "export", "rows": rows, "errors": errors}


def validate_path(path: str) -> dict:
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "manifest.json")):
        return validate_export(path)
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "store.json")):
        from quant_research_engine.store import read_store

        try:
            st = read_store(path, verify=True)
            for t in st.tables.values():
                validate_table(t)
            return {
                "ok": True,
                "kind": "store",
                "rows": {k: len(v) for k, v in st.tables.items()},
                "errors": [],
            }
        except ValueError as e:
            return {"ok": False, "kind": "store", "errors": [str(e)]}
    try:
        t = load_csv(path)
        out = {"ok": True, "kind": "csv", "rows": {os.path.basename(path): len(t)}, "errors": []}
        if os.path.exists(manifest_path(path)):
            with open(path, "rb") as f:
                check_manifest(path, f.read(), len(t))
        return out
    except ValueError as e:
        return {"ok": False, "kind": "csv", "errors": [str(e)]}


__all__ = [
    "build_export_tables",
    "export_from_store",
    "export_store",
    "make_table",
    "validate_export",
    "validate_path",
    "write_tables",
]
