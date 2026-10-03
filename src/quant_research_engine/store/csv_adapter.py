"""Local-CSV adapter: load a user's bar file into a store (works offline).

Configuration (JSON)::

    {"bars_file": "prices.csv",
     "columns": {"instrument_id": "ticker", "date": "day", "open": "o", "high": "h",
                 "low": "l", "close": "c", "volume": "v"},
     "calendar": "equity_daily", "ppy": 252,
     "availability_delay_seconds": 0,
     "corporate_actions_file": null,
     "asset_class": "equity", "currency": "USD", "lot_size": 0, "tick_size": 0.01,
     "source": "description of the data and its licence"}

``date`` is ``YYYY-MM-DD``: the bar's ``ts_open``/``ts_event`` are the session times of the
calendar (14:30:00Z/21:00:00Z for ``equity_daily``; 00:00:00Z and the next 00:00:00Z for
``crypto_daily``). ``ts_avail = ts_event + availability_delay_seconds``: the delay is
configured, never inferred. Instruments are derived from the bars (listing at the first bar,
no delistings). An optional corporate-action file is read in the contract's format. Malformed
input is rejected with the line number of the source file.
"""

from __future__ import annotations

import os
import re
from datetime import date

from quant_research_engine.data.csvio import DataError, load_csv, make_table, parse_csv_bytes
from quant_research_engine.data.manifest import read_json, sha256_file
from quant_research_engine.data.schemas import SCHEMAS
from quant_research_engine.store.builder import build_store
from quant_research_engine.store.store import Store
from quant_research_engine.timeutil import US_PER_S, format_ts, ts

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
REQUIRED = ("instrument_id", "date", "open", "high", "low", "close", "volume")


def _session(calendar: str, d: date) -> tuple[int, int]:
    if calendar == "equity_daily":
        return ts(d.year, d.month, d.day, 14, 30), ts(d.year, d.month, d.day, 21, 0)
    if calendar == "crypto_daily":
        o = ts(d.year, d.month, d.day)
        return o, o + 86_400 * US_PER_S
    raise ValueError(f"unsupported calendar {calendar!r}")


def load_local_csv(config_path: str) -> Store:
    cfg = read_json(config_path)
    base = os.path.dirname(os.path.abspath(config_path))
    bars_path = os.path.join(base, cfg["bars_file"])
    cal = cfg.get("calendar", "equity_daily")
    delay = int(cfg.get("availability_delay_seconds", 0)) * US_PER_S
    if delay < 0:
        raise ValueError("availability_delay_seconds must be >= 0")
    colmap = cfg["columns"]
    missing = [k for k in REQUIRED if k not in colmap]
    if missing:
        raise ValueError(f"column mapping lacks {missing}")
    with open(bars_path, "rb") as f:
        raw = f.read()
    if not raw:
        raise DataError("bars", 0, f"{cfg['bars_file']}: zero-byte file")
    lines = raw.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [x[:-1] if x.endswith("\r") else x for x in lines]
    header = lines[0].split(",")
    try:
        pos = {k: header.index(colmap[k]) for k in REQUIRED}
    except ValueError as e:
        raise DataError("bars", 1, f"{cfg['bars_file']}: mapped column missing: {e}") from None
    out_rows: list[tuple[str, int, str, int]] = []
    for i, ln in enumerate(lines[1:]):
        line = i + 2
        parts = ln.split(",")
        if len(parts) != len(header):
            raise DataError("bars", line, f"{cfg['bars_file']}: expected {len(header)} fields")
        m = _DATE_RE.match(parts[pos["date"]])
        if m is None:
            raise DataError("bars", line, f"{cfg['bars_file']}: bad date {parts[pos['date']]!r}")
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError as e:
            raise DataError("bars", line, f"{cfg['bars_file']}: {e}") from None
        to, te = _session(cal, d)
        iid = parts[pos["instrument_id"]]
        fields = [iid, format_ts(to), format_ts(te), format_ts(te + delay)]
        fields += [parts[pos[k]] for k in ("open", "high", "low", "close", "volume")]
        out_rows.append((iid, te, ",".join(fields), line))
    order = sorted(range(len(out_rows)), key=lambda j: (out_rows[j][0], out_rows[j][1], j))
    body = [SCHEMAS["bars"].header] + [out_rows[j][2] for j in order]
    try:
        bars = parse_csv_bytes(("\n".join(body) + "\n").encode("utf-8"), "bars")
    except DataError as e:
        src = out_rows[order[e.line - 2]][3] if e.line >= 2 else e.line
        msg = str(e).split(": ", 2)[-1]
        raise DataError("bars", src, f"{cfg['bars_file']}: {msg}") from None
    ids = sorted({str(x) for x in bars["instrument_id"].tolist()})
    first = {}
    for k in range(len(bars)):
        first.setdefault(str(bars["instrument_id"][k]), int(bars["ts_event"][k]))
    n = len(ids)
    ins = make_table(
        "instruments",
        instrument_id=ids,
        symbol=ids,
        asset_class=[cfg.get("asset_class", "equity")] * n,
        currency=[cfg.get("currency", "USD")] * n,
        ts_list=[first[i] for i in ids],
        ts_delist=[None] * n,
        delist_return=[None] * n,
        lot_size=[float(cfg.get("lot_size", 0))] * n,
        tick_size=[float(cfg.get("tick_size", 0.01))] * n,
        sector=[""] * n,
    )
    ca_file = cfg.get("corporate_actions_file")
    if ca_file:
        ca = load_csv(os.path.join(base, ca_file), "corporate_actions")
    else:
        ca = make_table(
            "corporate_actions", instrument_id=[], action=[], ts_ex=[], ts_avail=[], value=[]
        )
    series = make_table("series", series_id=[], ts_event=[], ts_avail=[], vintage=[], value=[])
    meta = {
        "calendar": cal,
        "ppy": int(cfg.get("ppy", 252)),
        "generator": None,
        "seed": None,
        "source": {
            "adapter": "local_csv",
            "bars_file_sha256": sha256_file(bars_path),
            "availability_delay_seconds": delay // US_PER_S,
            "note": str(cfg.get("source", "local CSV supplied by the user")),
        },
    }
    return build_store(ins, bars, ca, series, meta)
