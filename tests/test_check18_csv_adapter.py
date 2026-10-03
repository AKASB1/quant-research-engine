"""Check 18 (local-CSV part): column mapping, availability delay, line-numbered rejections."""

import json
import os

import pytest

from quant_research_engine.data import DataError, table_to_bytes
from quant_research_engine.store import read_store, write_store
from quant_research_engine.store.csv_adapter import load_local_csv
from quant_research_engine.timeutil import parse_ts

ROWS = [
    "day,ticker,o,h,l,c,v,extra",
    "2024-03-04,AAA,10.0,10.5,9.9,10.2,1000,x",
    "2024-03-05,AAA,10.2,10.4,10.0,10.1,1100,x",
    "2024-03-04,BBB,5.0,5.1,4.9,5.05,2000,y",
    "2024-03-05,BBB,5.05,5.2,5.0,5.1,0,y",
]


def _write(tmp_path, rows, delay=3600):
    (tmp_path / "prices.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    cfg = {
        "bars_file": "prices.csv",
        "columns": {
            "instrument_id": "ticker",
            "date": "day",
            "open": "o",
            "high": "h",
            "low": "l",
            "close": "c",
            "volume": "v",
        },
        "calendar": "equity_daily",
        "ppy": 252,
        "availability_delay_seconds": delay,
        "source": "synthetic test file",
    }
    p = tmp_path / "adapter.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return str(p)


def test_maps_columns_and_applies_delay(tmp_path):
    store = load_local_csv(_write(tmp_path, ROWS))
    assert store.ids == ["AAA", "BBB"]
    bars = store.tables["bars"]
    assert bars["ts_event"][0] == parse_ts("2024-03-04T21:00:00Z")
    assert bars["ts_avail"][0] == parse_ts("2024-03-04T22:00:00Z")
    assert bars["close"].tolist() == [10.2, 10.1, 5.05, 5.1]
    k = parse_ts("2024-03-04T21:30:00Z")
    assert all(len(t) == 0 for t in store.bars(["AAA", "BBB"], 5, knowledge=k).values())
    out = str(tmp_path / "store")
    write_store(out, store)
    back = read_store(out)
    for name in ("instruments", "bars", "liquidity"):
        assert table_to_bytes(back.tables[name]) == table_to_bytes(store.tables[name])


@pytest.mark.parametrize(
    "line,bad",
    [
        (2, "2024-03-04,AAA,10.0,10.5,9.9,10.2,1000"),  # wrong field count
        (3, "2024/03/05,AAA,10.2,10.4,10.0,10.1,1100,x"),  # bad date
        (4, "2024-03-04,BBB,5.0,4.0,4.9,5.05,2000,y"),  # high below low
        (5, "2024-03-05,BBB,5.05,5.2,5.0,-5.1,0,y"),  # non-positive price
        (5, "2024-03-04,BBB,5.0,5.1,4.9,5.05,2000,y"),  # duplicate key
    ],
)
def test_rejects_malformed_input_with_source_line(tmp_path, line, bad):
    rows = list(ROWS)
    rows[line - 1] = bad
    with pytest.raises(DataError) as e:
        load_local_csv(_write(tmp_path, rows))
    assert e.value.line == line, str(e.value)


def test_negative_delay_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_local_csv(_write(tmp_path, ROWS, delay=-1))
    assert os.path.exists(tmp_path / "prices.csv")
