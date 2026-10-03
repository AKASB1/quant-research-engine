"""Loaders, validators, and writers for the version-1 CSV files (shared contract 1.3, 1.5, 2).

A loader rejects a missing or different header, a wrong field count, an unparsable or
out-of-range value, an empty required field, an unsorted or duplicated key, and any violated
schema rule; the error names the 1-based line number (the header is line 1). A header with no
rows is a valid empty dataset; a zero-byte file is invalid. Writers produce UTF-8, LF line
endings, no quoting, floats in the shortest round-trip form, integral values of the columns in
``INTEGRAL_COLUMNS`` without a decimal point; NaN, inf, and -0.0 never appear.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from quant_research_engine.data.schemas import INTEGRAL_COLUMNS, SCHEMAS, Schema
from quant_research_engine.timeutil import format_ts_array, parse_ts

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_TOKEN_RE = re.compile(r'^[^,"\s]+$')
_TEXT_RE = re.compile(r'^[^,"\r\n]*$')
_UPPER_RE = re.compile(r"^[A-Z]+$")
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")
REL_TOL = 1e-9


class DataError(ValueError):
    """A malformed dataset; ``line`` is the 1-based line number (0 for the whole file)."""

    def __init__(self, schema: str, line: int, message: str):
        self.schema = schema
        self.line = line
        super().__init__(f"{schema}: line {line}: {message}")


@dataclass
class Table:
    """Columns of one dataset as NumPy arrays (instants as int64 microseconds, text as str)."""

    schema: Schema
    columns: dict[str, np.ndarray]

    def __len__(self) -> int:
        first = self.schema.columns[0].name
        return len(self.columns[first])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    def take(self, idx: np.ndarray) -> Table:
        return Table(self.schema, {k: v[idx] for k, v in self.columns.items()})


def empty_table(schema: Schema) -> Table:
    cols: dict[str, np.ndarray] = {}
    for c in schema.columns:
        cols[c.name] = _empty_array(c.kind)
    return Table(schema, cols)


def _empty_array(kind: str) -> np.ndarray:
    if kind in ("ts", "ts?", "int"):
        return np.zeros(0, dtype=np.int64)
    if kind in ("float", "float?"):
        return np.zeros(0, dtype=np.float64)
    if kind == "bool":
        return np.zeros(0, dtype=bool)
    return np.zeros(0, dtype=object)


def make_table(schema: Schema | str, **cols) -> Table:
    """Build a table from Python sequences; instants are int microseconds (or None)."""
    if isinstance(schema, str):
        schema = SCHEMAS[schema]
    out: dict[str, np.ndarray] = {}
    for c in schema.columns:
        v = cols[c.name]
        if c.kind in ("ts", "int"):
            out[c.name] = np.asarray(v, dtype=np.int64)
        elif c.kind == "ts?":
            out[c.name] = np.asarray([-1 if x is None else x for x in v], dtype=np.int64)
        elif c.kind in ("float", "float?"):
            out[c.name] = np.asarray([math.nan if x is None else x for x in v], dtype=np.float64)
        elif c.kind == "bool":
            out[c.name] = np.asarray(v, dtype=bool)
        else:
            out[c.name] = np.asarray(list(v), dtype=object)
    return Table(schema, out)


# Optional instants are stored as -1 when empty (no instant before 1970 is used).
NO_TS = -1


def _parse_field(col, raw: str, schema: str, line: int):
    k = col.kind
    if raw != raw.strip():
        raise DataError(schema, line, f"{col.name}: padding spaces are not allowed")
    if k in ("ts?", "float?") and raw == "":
        return NO_TS if k == "ts?" else math.nan
    if raw == "" and k != "text":
        raise DataError(schema, line, f"{col.name}: required field is empty")
    try:
        if k in ("ts", "ts?"):
            return parse_ts(raw)
        if k in ("float", "float?"):
            if not _FLOAT_RE.match(raw):
                raise ValueError(f"not a finite decimal number: {raw!r}")
            v = float(raw)
            if not math.isfinite(v):
                raise ValueError(f"not finite: {raw!r}")
            if v == 0.0 and math.copysign(1.0, v) < 0:
                raise ValueError("-0.0 is not allowed")
            return v
        if k == "int":
            if not _INT_RE.match(raw):
                raise ValueError(f"not an integer: {raw!r}")
            return int(raw)
        if k == "bool":
            if raw not in ("true", "false"):
                raise ValueError(f"not a boolean: {raw!r}")
            return raw == "true"
        if k == "id":
            if not _ID_RE.match(raw):
                raise ValueError(f"bad instrument id {raw!r}")
            return raw
        if k == "token":
            if not _TOKEN_RE.match(raw):
                raise ValueError(f"bad token {raw!r}")
            return raw
        if k == "text":
            if not _TEXT_RE.match(raw):
                raise ValueError(f"bad text {raw!r}")
            return raw
        if k == "upper":
            if not _UPPER_RE.match(raw):
                raise ValueError(f"expected uppercase letters, got {raw!r}")
            return raw
        if k == "enum":
            if raw not in col.choices:
                raise ValueError(f"expected one of {list(col.choices)}, got {raw!r}")
            return raw
    except ValueError as e:
        raise DataError(schema, line, f"{col.name}: {e}") from None
    raise AssertionError(k)


def parse_csv_bytes(data: bytes, schema: Schema | str) -> Table:
    """Parse and validate CSV bytes against a schema; raise DataError naming the line."""
    if isinstance(schema, str):
        schema = SCHEMAS[schema]
    name = schema.name
    if len(data) == 0:
        raise DataError(name, 0, "zero-byte file (a valid empty dataset still has a header)")
    if data.startswith(b"\xef\xbb\xbf"):
        raise DataError(name, 1, "byte order mark is not allowed")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DataError(name, 0, f"not UTF-8: {e}") from None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in lines]
    if not lines:
        raise DataError(name, 1, "missing header")
    if lines[0] != schema.header:
        raise DataError(name, 1, f"header mismatch: expected {schema.header!r}, got {lines[0]!r}")
    ncol = len(schema.columns)
    values: list[list] = [[] for _ in range(ncol)]
    for i, ln in enumerate(lines[1:]):
        line = i + 2
        parts = ln.split(",")
        if len(parts) != ncol:
            raise DataError(name, line, f"expected {ncol} fields, got {len(parts)}")
        for j, col in enumerate(schema.columns):
            values[j].append(_parse_field(col, parts[j], name, line))
    cols: dict[str, np.ndarray] = {}
    for j, col in enumerate(schema.columns):
        if col.kind in ("ts", "ts?", "int"):
            cols[col.name] = np.asarray(values[j], dtype=np.int64)
        elif col.kind in ("float", "float?"):
            cols[col.name] = np.asarray(values[j], dtype=np.float64)
        elif col.kind == "bool":
            cols[col.name] = np.asarray(values[j], dtype=bool)
        else:
            cols[col.name] = np.asarray(values[j], dtype=object)
    table = Table(schema, cols)
    validate_table(table)
    return table


def load_csv(path: str, schema: Schema | str | None = None) -> Table:
    from quant_research_engine.data.schemas import schema_for_file

    if schema is None:
        schema = schema_for_file(path)
    with open(path, "rb") as f:
        data = f.read()
    return parse_csv_bytes(data, schema)


# ----------------------------------------------------------------------------- validation


def _keys(table: Table, names: tuple[str, ...]) -> list[tuple]:
    cols = [table[n].tolist() for n in names]
    return list(zip(*cols, strict=True)) if cols else []


def _check_sorted_unique(table: Table) -> None:
    s = table.schema
    if len(table) < 2:
        return
    keys = _keys(table, s.sort_keys)
    for i in range(1, len(keys)):
        if keys[i] < keys[i - 1]:
            raise DataError(s.name, i + 2, f"rows not sorted by {list(s.sort_keys)}")
    if s.unique_keys:
        seen: dict[tuple, int] = {}
        for i, k in enumerate(_keys(table, s.unique_keys)):
            if k in seen:
                raise DataError(s.name, i + 2, f"duplicate key {list(s.unique_keys)} = {list(k)}")
            seen[k] = i


def _first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def _rule(table: Table, mask: np.ndarray, message: str) -> None:
    i = _first(mask)
    if i is not None:
        raise DataError(table.schema.name, i + 2, message)


def _close(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) <= REL_TOL * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))


def validate_table(table: Table) -> None:
    """Every row rule and cross-row rule of the schema; raise DataError naming the line."""
    s = table.schema
    n = len(table)
    if n == 0:
        return
    name = s.name
    if name == "instruments":
        dl, dr = table["ts_delist"], table["delist_return"]
        _rule(table, (dl == NO_TS) & ~np.isnan(dr), "delist_return given without ts_delist")
        _rule(table, (dl != NO_TS) & np.isnan(dr), "ts_delist given without delist_return")
        _rule(table, ~np.isnan(dr) & (dr < -1), "delist_return below -1")
        _rule(table, (dl != NO_TS) & (dl < table["ts_list"]), "delisting before the listing")
        _rule(table, table["lot_size"] < 0, "lot_size must be >= 0")
        _rule(table, table["tick_size"] < 0, "tick_size must be >= 0")
    elif name == "bars":
        o, h, lo, c, v = (table[k] for k in ("open", "high", "low", "close", "volume"))
        _rule(table, table["ts_open"] >= table["ts_event"], "ts_open must be before ts_event")
        _rule(table, table["ts_avail"] < table["ts_event"], "ts_avail before ts_event")
        _rule(table, (o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0), "prices must be > 0")
        _rule(table, h < np.maximum(np.maximum(o, c), lo), "high below open, close, or low")
        _rule(table, lo > np.minimum(np.minimum(o, c), h), "low above open, close, or high")
        _rule(table, v < 0, "volume must be >= 0")
    elif name == "corporate_actions":
        _rule(table, table["ts_avail"] > table["ts_ex"], "ts_avail after ts_ex")
        _rule(table, table["value"] <= 0, "value must be > 0")
    elif name == "series":
        _rule(table, table["ts_avail"] < table["ts_event"], "ts_avail before ts_event")
        _rule(table, table["vintage"] < 0, "vintage must be >= 0")
    elif name in ("returns", "signals", "forecasts", "liquidity"):
        _rule(table, table["ts_avail"] < table["ts_event"], "ts_avail before ts_event")
        if name == "returns":
            _rule(table, table["ret"] < -1, "ret below -1")
        if name == "forecasts":
            _rule(table, table["horizon_bars"] < 1, "horizon_bars must be >= 1")
            sg = table["sigma"]
            _rule(table, ~np.isnan(sg) & (sg < 0), "sigma must be >= 0")
        if name == "liquidity":
            _rule(table, table["adv_shares"] <= 0, "adv_shares must be > 0")
            _rule(table, table["sigma_bar"] < 0, "sigma_bar must be >= 0")
    elif name == "orders":
        _rule(table, table["quantity"] == 0, "quantity must not be zero")
        lp = table["limit_price"]
        is_mkt = table["order_type"] == "market"
        _rule(table, is_mkt & ~np.isnan(lp), "limit_price given for a market order")
        _rule(table, ~is_mkt & (np.isnan(lp) | (lp <= 0)), "limit order needs limit_price > 0")
    elif name == "fills":
        _rule(table, table["quantity"] == 0, "quantity must not be zero")
        _rule(table, (table["ref_price"] <= 0) | (table["price"] <= 0), "prices must be > 0")
        costs = ("spread_cost", "impact_cost", "commission")
        _rule(table, np.any([table[k] < 0 for k in costs], axis=0), "costs must be >= 0")
        dl = table["order_id"] == "DELIST"
        _rule(
            table,
            dl & np.any([table[k] != 0 for k in costs], axis=0),
            "a DELIST fill has no costs",
        )
    elif name == "positions":
        _rule(table, table["quantity"] == 0, "zero positions are not listed")
        _rule(table, table["mark_price"] <= 0, "mark_price must be > 0")
        _rule(
            table,
            ~_close(table["value"], table["quantity"] * table["mark_price"]),
            "value != quantity * mark_price",
        )
    elif name == "equity":
        _validate_equity(table)
    _check_sorted_unique(table)
    if name == "series":
        _validate_vintages(table)
    if name == "bars":
        _validate_bar_overlap(table)


def _validate_vintages(table: Table) -> None:
    sid, te, ta, vin = (table[k] for k in ("series_id", "ts_event", "ts_avail", "vintage"))
    for i in range(len(table)):
        first = i == 0 or sid[i] != sid[i - 1] or te[i] != te[i - 1]
        if first:
            if vin[i] != 0:
                raise DataError("series", i + 2, "the first vintage of a period must be 0")
        else:
            if vin[i] != vin[i - 1] + 1:
                raise DataError("series", i + 2, "vintages must increase by one")
            if ta[i] <= ta[i - 1]:
                raise DataError("series", i + 2, "ts_avail must increase with the vintage")


def _validate_bar_overlap(table: Table) -> None:
    iid, to, te = table["instrument_id"], table["ts_open"], table["ts_event"]
    same = iid[1:] == iid[:-1]
    bad = same & (to[1:] < te[:-1])
    i = _first(bad)
    if i is not None:
        raise DataError("bars", i + 3, "bar overlaps the previous bar of the instrument")


def _validate_equity(table: Table) -> None:
    c = table.columns
    _rule(
        table, ~_close(c["equity"], c["cash"] + c["position_value"]), "equity != cash + positions"
    )
    for k in ("spread_cost", "impact_cost", "commission", "borrow", "financing"):
        _rule(table, c[k] < 0, f"{k} must be >= 0")
    flows = ("hold_pnl", "trade_pnl", "spread_cost", "impact_cost", "commission")
    flows += ("borrow", "financing", "income")
    if any(c[k][0] != 0 for k in flows) or c["position_value"][0] != 0:
        raise DataError("equity", 2, "the opening row has no positions and no flows")
    e = c["equity"]
    d = e[1:] - e[:-1]
    pnl = c["hold_pnl"][1:] + c["trade_pnl"][1:] + c["income"][1:]
    cost = sum(c[k][1:] for k in ("spread_cost", "impact_cost", "commission", "borrow"))
    cost = cost + c["financing"][1:]
    resid = np.abs(d - (pnl - cost))
    bad = resid > REL_TOL * np.maximum(1.0, np.abs(e[:-1]))
    i = _first(bad)
    if i is not None:
        raise DataError("equity", i + 3, f"accounting identity violated (residual {resid[i]:.3g})")


# ----------------------------------------------------------------------------- writing


def fmt_float(v: float, integral: bool) -> str:
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"cannot write non-finite value {v!r}")
    if v == 0.0:
        v = 0.0  # never -0.0
    if integral and v.is_integer():
        return str(int(v))
    return repr(v)


def format_column(col, arr: np.ndarray) -> list[str]:
    k = col.kind
    if k == "ts":
        return format_ts_array(arr)
    if k == "ts?":
        arr = np.asarray(arr, dtype=np.int64)
        out = [""] * len(arr)
        ok = np.flatnonzero(arr != NO_TS)
        if ok.size:
            fmt = format_ts_array(arr[ok])
            for j, s in zip(ok.tolist(), fmt, strict=True):
                out[j] = s
        return out
    if k in ("float", "float?"):
        integral = col.name in INTEGRAL_COLUMNS
        out = []
        for v in np.asarray(arr, dtype=np.float64).tolist():
            if math.isnan(v):
                if k == "float":
                    raise ValueError(f"{col.name}: missing value in a required column")
                out.append("")
            else:
                out.append(fmt_float(v, integral))
        return out
    if k == "int":
        return [str(int(x)) for x in np.asarray(arr).tolist()]
    if k == "bool":
        return ["true" if x else "false" for x in np.asarray(arr).tolist()]
    return [str(x) for x in arr.tolist()]


def table_to_bytes(table: Table) -> bytes:
    s = table.schema
    cols = [format_column(c, table[c.name]) for c in s.columns]
    lines = [s.header]
    lines.extend(",".join(row) for row in zip(*cols, strict=True))
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_csv(path: str, table: Table) -> bytes:
    """Validate and write; returns the bytes written."""
    validate_table(table)
    data = table_to_bytes(table)
    with open(path, "wb") as f:
        f.write(data)
    return data


def sort_table(table: Table) -> Table:
    """Sort rows by the schema's sort keys (stable, byte order for text)."""
    s = table.schema
    n = len(table)
    if n < 2:
        return table
    keys = _keys(table, s.sort_keys)
    order = sorted(range(n), key=lambda i: keys[i])
    return table.take(np.asarray(order, dtype=np.int64))
