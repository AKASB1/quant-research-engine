"""Check 1: golden fixtures, loaders, validators, byte-identical round trips, malformed variants."""

import os

import pytest
from conftest import GOLDEN

from quant_research_engine.data import DataError, load_csv, parse_csv_bytes, table_to_bytes
from quant_research_engine.data.manifest import config_hash
from quant_research_engine.data.schemas import SCHEMAS

FILES = sorted(f for f in os.listdir(GOLDEN) if f.endswith("_v1.csv"))


def test_all_twelve_schemas_have_a_fixture():
    assert len(FILES) == 12
    assert {f[:-7] for f in FILES} == set(SCHEMAS)


@pytest.mark.parametrize("name", FILES)
def test_fixture_loads_and_round_trips_byte_identical(name):
    path = os.path.join(GOLDEN, name)
    with open(path, "rb") as f:
        data = f.read()
    table = load_csv(path)
    assert len(table) >= 1
    assert table_to_bytes(table) == data


@pytest.mark.parametrize("schema", sorted(SCHEMAS))
def test_header_only_is_valid_empty_dataset(schema):
    t = parse_csv_bytes((SCHEMAS[schema].header + "\n").encode(), schema)
    assert len(t) == 0
    assert table_to_bytes(t) == (SCHEMAS[schema].header + "\n").encode()


def test_crlf_is_accepted():
    with open(os.path.join(GOLDEN, "bars_v1.csv"), "rb") as f:
        data = f.read()
    t = parse_csv_bytes(data.replace(b"\n", b"\r\n"), "bars")
    assert table_to_bytes(t) == data


def _golden(name):
    with open(os.path.join(GOLDEN, name + "_v1.csv"), encoding="utf-8") as f:
        return f.read().split("\n")[:-1]


def _join(lines):
    return ("\n".join(lines) + "\n").encode()


def _replace(name, line_no, old, new):
    lines = _golden(name)
    assert old in lines[line_no - 1]
    lines[line_no - 1] = lines[line_no - 1].replace(old, new, 1)
    return _join(lines)


MALFORMED = [
    # (description, schema, bytes, expected line)
    ("wrong header", "bars", _replace("bars", 1, "volume", "vol"), 1),
    ("wrong field count", "bars", _replace("bars", 3, ",1200000", ""), 3),
    ("bad timestamp", "bars", _replace("bars", 2, "2024-03-04T21:00:00Z", "2024-03-04 21:00"), 2),
    (
        "ts_avail before ts_event",
        "returns",
        _replace("returns", 2, "21:00:00Z,A", "20:00:00Z,A"),
        2,
    ),
    ("high below low", "bars", _replace("bars", 4, "102.4,101.2", "101.0,101.2"), 4),
    ("non-positive price", "bars", _replace("bars", 5, "40.0,40.8", "0.0,40.8"), 5),
    ("duplicate key", "liquidity", _join(_golden("liquidity") + [_golden("liquidity")[2]]), 4),
    (
        "unsorted rows",
        "instruments",
        _join([_golden("instruments")[0], _golden("instruments")[2], _golden("instruments")[1]]),
        3,
    ),
    ("zero-byte file", "series", b"", 0),
    ("non-increasing vintage", "series", _replace("series", 3, ",1,1.05", ",0,1.05"), 3),
    (
        "unknown action",
        "corporate_actions",
        _replace("corporate_actions", 3, "split", "spinoff"),
        3,
    ),
    (
        "delisting before the listing",
        "instruments",
        _replace(
            "instruments",
            2,
            "2024-01-02T21:00:00Z,,",
            "2024-01-02T21:00:00Z,2023-12-01T21:00:00Z,-0.5",
        ),
        2,
    ),
    ("nan value", "signals", _replace("signals", 2, ",0.8", ",nan"), 2),
    ("negative zero", "returns", _replace("returns", 4, ",0.0", ",-0.0"), 4),
    ("byte order mark", "orders", b"\xef\xbb\xbf" + _join(_golden("orders")), 1),
    (
        "delist_return without ts_delist",
        "instruments",
        _replace("instruments", 3, ",,,0", ",,-0.5,0"),
        3,
    ),
    (
        "market order with limit price",
        "orders",
        _replace("orders", 2, "market,,", "market,10.0,"),
        2,
    ),
    ("equity identity violated", "equity", _replace("equity", 3, "50.0,2.03", "51.0,2.03"), 3),
    ("padding spaces", "liquidity", _replace("liquidity", 2, ",0.02", ", 0.02"), 2),
    ("empty required field", "fills", _replace("fills", 2, ",101.5,", ",,"), 2),
]


def test_at_least_twelve_malformed_variants():
    assert len(MALFORMED) >= 12


@pytest.mark.parametrize("desc,schema,data,line", MALFORMED, ids=[m[0] for m in MALFORMED])
def test_malformed_variant_rejected_with_line_number(desc, schema, data, line):
    with pytest.raises(DataError) as e:
        parse_csv_bytes(data, schema)
    assert e.value.line == line, str(e.value)
    assert f"line {line}" in str(e.value)


def test_config_hash_omits_defaults_and_is_canonical():
    a = config_hash({"b": 1, "a": 0.5})
    b = config_hash({"a": 0.5, "b": 1})
    assert a == b and len(a) == 64
