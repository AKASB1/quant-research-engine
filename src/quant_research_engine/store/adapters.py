"""Third-party data adapters (Tier 2). Each works offline on a tiny fixture in its tests; downloads,
when the owner runs them, go to the git-ignored cache ``data/cache/`` and are never committed,
packaged, or exported (an export of such a store is derived-only).

- Kenneth R. French Data Library (daily portfolio and factor files; copyright Eugene F. Fama and
  Kenneth R. French; no click-through terms found): the files hold daily returns in percent, so
  the adapter builds a return-index series per portfolio. There is no open, no volume, and no
  liquidity: the open is the previous close, the volume is a placeholder that only lets fills
  happen, impact must be off, and fills use ``next_close``. The returns are the library's
  latest vintage (it revises history), so the series is not point-in-time; ``ts_avail`` is the
  bar's close plus a configured delay.
- Binance public data archive (daily klines): parsed into ``crypto_daily`` bars (open time in
  milliseconds before 2025 and microseconds from 2025 on). Not downloaded here: the archive's
  dataset terms make any use an acceptance of them and of the Binance Terms of Use, which only
  the owner can accept.
"""

from __future__ import annotations

import io
import math
import os
import re
import zipfile
from datetime import date

import numpy as np

from quant_research_engine.data.csvio import make_table, sort_table
from quant_research_engine.store.builder import build_store
from quant_research_engine.store.store import Store
from quant_research_engine.timeutil import US_PER_S, ts

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
PLACEHOLDER_VOLUME = 1e12
_ID = re.compile(r"[^A-Za-z0-9_-]")


def parse_french_daily(text: str) -> tuple[list[date], list[str], np.ndarray]:
    """First table of a French daily CSV: (dates, column names, returns as decimals, NaN for the
    library's missing codes -99.99 and -999)."""
    lines = text.replace("\r", "").split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith(",") and len(ln.split(",")) > 1)
    names = [_ID.sub("_", x.strip()) for x in lines[i].split(",")[1:]]
    dates, rows = [], []
    for ln in lines[i + 1 :]:
        parts = [x.strip() for x in ln.split(",")]
        if not parts or not re.fullmatch(r"\d{8}", parts[0]):
            break
        d = parts[0]
        dates.append(date(int(d[:4]), int(d[4:6]), int(d[6:])))
        vals = [float(x) for x in parts[1 : 1 + len(names)]]
        rows.append([math.nan if v <= -99.99 else v / 100.0 for v in vals])
    return dates, names, np.asarray(rows, dtype=np.float64)


def store_from_returns(
    dates,
    names,
    rets,
    *,
    delay_days: int = 1,
    start: date | None = None,
    source: str = "",
    calendar: str = "equity_daily",
) -> Store:
    """A store of return-index bars (level 100 before the first return); see the module notes."""
    keep = [k for k, d in enumerate(dates) if start is None or d >= start]
    dates = [dates[k] for k in keep]
    rets = rets[keep]
    rows = {
        k: []
        for k in (
            "instrument_id",
            "ts_open",
            "ts_event",
            "ts_avail",
            "open",
            "high",
            "low",
            "close",
            "volume",
        )
    }
    first = {}
    for j, name in enumerate(names):
        level = 100.0
        for k, d in enumerate(dates):
            r = rets[k, j]
            if math.isnan(r):
                continue
            prev = level
            level = level * (1.0 + r)
            te = (
                ts(d.year, d.month, d.day, 21, 0)
                if calendar == "equity_daily"
                else ts(d.year, d.month, d.day) + 86_400 * US_PER_S
            )
            to = (
                ts(d.year, d.month, d.day, 14, 30)
                if calendar == "equity_daily"
                else ts(d.year, d.month, d.day)
            )
            first.setdefault(name, te)
            for key, v in (
                ("instrument_id", name),
                ("ts_open", to),
                ("ts_event", te),
                ("ts_avail", te + delay_days * 86_400 * US_PER_S),
                ("open", prev),
                ("high", max(prev, level)),
                ("low", min(prev, level)),
                ("close", level),
                ("volume", PLACEHOLDER_VOLUME),
            ):
                rows[key].append(v)
    ids = sorted(first)
    ins = make_table(
        "instruments",
        instrument_id=ids,
        symbol=ids,
        asset_class=["index"] * len(ids),
        currency=["USD"] * len(ids),
        ts_list=[first[i] for i in ids],
        ts_delist=[None] * len(ids),
        delist_return=[None] * len(ids),
        lot_size=[0.0] * len(ids),
        tick_size=[0.0] * len(ids),
        sector=[""] * len(ids),
    )
    bars = sort_table(make_table("bars", **rows))
    empty_ca = make_table(
        "corporate_actions", instrument_id=[], action=[], ts_ex=[], ts_avail=[], value=[]
    )
    empty_s = make_table("series", series_id=[], ts_event=[], ts_avail=[], vintage=[], value=[])
    meta = {
        "calendar": calendar,
        "ppy": 252 if calendar == "equity_daily" else 365,
        "generator": None,
        "seed": None,
        "source": {
            "adapter": "french_daily",
            "note": source,
            "point_in_time": False,
            "delay_days": delay_days,
            "volume": "placeholder (no volume in the source)",
        },
        "data": "Kenneth R. French Data Library (latest vintage; not point-in-time); "
        "illustration only",
    }
    return build_store(ins, bars, empty_ca, empty_s, meta)


def fetch_french(name: str, cache_dir: str) -> str:
    """Download ``<name>_CSV.zip`` into the cache (once) and return the CSV text."""
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}_CSV.zip")
    if not os.path.exists(path):
        req = urllib.request.Request(
            FRENCH_BASE + f"{name}_CSV.zip", headers={"User-Agent": "quant-research-engine"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(path, "wb") as f:
            f.write(data)
    with zipfile.ZipFile(path) as z:
        member = z.namelist()[0]
        return z.read(member).decode("latin-1")


def parse_binance_klines(text: str) -> list[tuple]:
    """Rows (open_time_us, open, high, low, close, volume) of a daily kline CSV (with or without a
    header). Open times are milliseconds before 2025 and microseconds from 2025 on."""
    out = []
    for ln in io.StringIO(text):
        parts = ln.strip().split(",")
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        t = int(parts[0])
        t_us = t * 1000 if t < 10**14 else t
        out.append(
            (
                t_us,
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
        )
    return out


def store_from_binance(klines: dict[str, list[tuple]], delay_seconds: int = 0) -> Store:
    rows = {
        k: []
        for k in (
            "instrument_id",
            "ts_open",
            "ts_event",
            "ts_avail",
            "open",
            "high",
            "low",
            "close",
            "volume",
        )
    }
    first = {}
    day = 86_400 * US_PER_S
    for sym in sorted(klines):
        iid = _ID.sub("_", sym)
        for t_us, o, h, lo, c, v in sorted(klines[sym]):
            first.setdefault(iid, t_us + day)
            for key, val in (
                ("instrument_id", iid),
                ("ts_open", t_us),
                ("ts_event", t_us + day),
                ("ts_avail", t_us + day + delay_seconds * US_PER_S),
                ("open", o),
                ("high", h),
                ("low", lo),
                ("close", c),
                ("volume", v),
            ):
                rows[key].append(val)
    ids = sorted(first)
    ins = make_table(
        "instruments",
        instrument_id=ids,
        symbol=ids,
        asset_class=["crypto"] * len(ids),
        currency=["USDT"] * len(ids),
        ts_list=[first[i] for i in ids],
        ts_delist=[None] * len(ids),
        delist_return=[None] * len(ids),
        lot_size=[0.0] * len(ids),
        tick_size=[0.0] * len(ids),
        sector=[""] * len(ids),
    )
    bars = sort_table(make_table("bars", **rows))
    empty_ca = make_table(
        "corporate_actions", instrument_id=[], action=[], ts_ex=[], ts_avail=[], value=[]
    )
    empty_s = make_table("series", series_id=[], ts_event=[], ts_avail=[], vintage=[], value=[])
    meta = {
        "calendar": "crypto_daily",
        "ppy": 365,
        "generator": None,
        "seed": None,
        "source": {
            "adapter": "binance_daily_klines",
            "note": "Binance Vision archive (CC BY-NC-SA 4.0 terms)",
        },
        "data": "Binance public archive klines",
    }
    return build_store(ins, bars, empty_ca, empty_s, meta)
