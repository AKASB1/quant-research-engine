"""Version-1 CSV schemas of the shared contract (section 2) as column specifications.

Kinds: ``id`` (instrument id: letters, digits, ``_``, ``-``), ``token`` (non-empty text without
comma, quote, or whitespace), ``text`` (may be empty, no comma or quote), ``upper`` (uppercase
letters), ``enum``, ``ts`` / ``ts?`` (instant, optional), ``float`` / ``float?``, ``int``,
``bool``. Columns in :data:`INTEGRAL_COLUMNS` are written without a decimal point when integral.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = 1
INTEGRAL_COLUMNS = frozenset(
    {"volume", "adv_shares", "quantity", "lot_size", "horizon_bars", "vintage"}
)


@dataclass(frozen=True)
class Column:
    name: str
    kind: str
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class Schema:
    name: str
    columns: tuple[Column, ...]
    sort_keys: tuple[str, ...]
    unique_keys: tuple[str, ...] = field(default=())

    @property
    def header(self) -> str:
        return ",".join(c.name for c in self.columns)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


def _c(name: str, kind: str, *choices: str) -> Column:
    return Column(name, kind, tuple(choices))


INSTRUMENTS = Schema(
    "instruments",
    (
        _c("instrument_id", "id"),
        _c("symbol", "token"),
        _c("asset_class", "enum", "equity", "crypto", "index", "other"),
        _c("currency", "upper"),
        _c("ts_list", "ts"),
        _c("ts_delist", "ts?"),
        _c("delist_return", "float?"),
        _c("lot_size", "float"),
        _c("tick_size", "float"),
        _c("sector", "text"),
    ),
    ("instrument_id",),
    ("instrument_id",),
)
BARS = Schema(
    "bars",
    (
        _c("instrument_id", "id"),
        _c("ts_open", "ts"),
        _c("ts_event", "ts"),
        _c("ts_avail", "ts"),
        _c("open", "float"),
        _c("high", "float"),
        _c("low", "float"),
        _c("close", "float"),
        _c("volume", "float"),
    ),
    ("instrument_id", "ts_event"),
    ("instrument_id", "ts_event"),
)
CORPORATE_ACTIONS = Schema(
    "corporate_actions",
    (
        _c("instrument_id", "id"),
        _c("action", "enum", "split", "cash_dividend"),
        _c("ts_ex", "ts"),
        _c("ts_avail", "ts"),
        _c("value", "float"),
    ),
    ("instrument_id", "ts_ex", "action"),
    ("instrument_id", "ts_ex", "action"),
)
SERIES = Schema(
    "series",
    (
        _c("series_id", "token"),
        _c("ts_event", "ts"),
        _c("ts_avail", "ts"),
        _c("vintage", "int"),
        _c("value", "float"),
    ),
    ("series_id", "ts_event", "vintage"),
    ("series_id", "ts_event", "vintage"),
)
RETURNS = Schema(
    "returns",
    (_c("ts_event", "ts"), _c("ts_avail", "ts"), _c("instrument_id", "id"), _c("ret", "float")),
    ("ts_event", "instrument_id"),
    ("ts_event", "instrument_id"),
)
SIGNALS = Schema(
    "signals",
    (
        _c("ts_event", "ts"),
        _c("ts_avail", "ts"),
        _c("instrument_id", "id"),
        _c("name", "token"),
        _c("value", "float"),
    ),
    ("ts_event", "name", "instrument_id"),
    ("ts_event", "name", "instrument_id"),
)
FORECASTS = Schema(
    "forecasts",
    (
        _c("ts_event", "ts"),
        _c("ts_avail", "ts"),
        _c("instrument_id", "id"),
        _c("horizon_bars", "int"),
        _c("mu", "float"),
        _c("sigma", "float?"),
    ),
    ("ts_event", "instrument_id", "horizon_bars"),
    ("ts_event", "instrument_id", "horizon_bars"),
)
LIQUIDITY = Schema(
    "liquidity",
    (
        _c("ts_event", "ts"),
        _c("ts_avail", "ts"),
        _c("instrument_id", "id"),
        _c("adv_shares", "float"),
        _c("sigma_bar", "float"),
    ),
    ("ts_event", "instrument_id"),
    ("ts_event", "instrument_id"),
)
ORDERS = Schema(
    "orders",
    (
        _c("order_id", "token"),
        _c("ts_submit", "ts"),
        _c("instrument_id", "id"),
        _c("quantity", "float"),
        _c("order_type", "enum", "market", "limit"),
        _c("limit_price", "float?"),
        _c("tif", "enum", "day", "gtc"),
        _c("strategy_id", "token"),
    ),
    ("ts_submit", "order_id"),
    ("order_id",),
)
FILLS = Schema(
    "fills",
    (
        _c("fill_id", "token"),
        _c("order_id", "token"),
        _c("ts_fill", "ts"),
        _c("instrument_id", "id"),
        _c("quantity", "float"),
        _c("ref_price", "float"),
        _c("price", "float"),
        _c("spread_cost", "float"),
        _c("impact_cost", "float"),
        _c("commission", "float"),
    ),
    ("ts_fill", "fill_id"),
    ("fill_id",),
)
POSITIONS = Schema(
    "positions",
    (
        _c("ts_event", "ts"),
        _c("instrument_id", "id"),
        _c("quantity", "float"),
        _c("mark_price", "float"),
        _c("value", "float"),
    ),
    ("ts_event", "instrument_id"),
    ("ts_event", "instrument_id"),
)
EQUITY = Schema(
    "equity",
    (
        _c("ts_event", "ts"),
        _c("cash", "float"),
        _c("position_value", "float"),
        _c("equity", "float"),
        _c("hold_pnl", "float"),
        _c("trade_pnl", "float"),
        _c("spread_cost", "float"),
        _c("impact_cost", "float"),
        _c("commission", "float"),
        _c("borrow", "float"),
        _c("financing", "float"),
        _c("income", "float"),
    ),
    ("ts_event",),
    ("ts_event",),
)

SCHEMAS: dict[str, Schema] = {
    s.name: s
    for s in (
        INSTRUMENTS,
        BARS,
        CORPORATE_ACTIONS,
        SERIES,
        RETURNS,
        SIGNALS,
        FORECASTS,
        LIQUIDITY,
        ORDERS,
        FILLS,
        POSITIONS,
        EQUITY,
    )
}


def schema_for_file(path: str) -> Schema:
    """Schema of a file named ``<schema>.csv`` or ``<schema>_v1.csv``."""
    import os

    base = os.path.basename(path)
    stem = base[:-4] if base.endswith(".csv") else base
    if stem.endswith("_v1"):
        stem = stem[:-3]
    if stem not in SCHEMAS:
        raise ValueError(f"no schema for file {base!r}")
    return SCHEMAS[stem]
