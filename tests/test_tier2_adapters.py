"""Check 18 (Tier 2 part): the French and Binance adapters on tiny offline fixtures, with the
network disabled."""

import socket

import numpy as np
import pytest

from quant_research_engine.export import export_from_store, validate_export
from quant_research_engine.store.adapters import (
    parse_binance_klines,
    parse_french_daily,
    store_from_binance,
    store_from_returns,
)

FRENCH = """ This file was created by a test using synthetic numbers.
 It mimics the layout of a daily portfolio file.

 Average Value Weighted Returns -- Daily
,NoDur,Durbl,Hlth ,Other
20240102,   0.50,  -1.00,   0.10, -99.99
20240103,  -0.20,   0.30,   0.00,   0.40
20240104,   1.00,   0.20,  -0.50,   0.10

 Average Equal Weighted Returns -- Daily
,NoDur,Durbl,Hlth ,Other
20240102,   9.00,   9.00,   9.00,   9.00
"""

BINANCE = """1704067200000,42283.58,44184.10,42180.77,44179.55,27174.29903,1704153599999,1,2,3,4,0
1704153600000,44179.55,45879.63,44148.34,44946.91,65146.40661,1704239999999,1,2,3,4,0
open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
1735689600000000,93576.00,95151.15,92888.00,94591.79,10373.32613,1735775999999999,1,2,3,4,0
"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_french_parser_and_store(tmp_path):
    dates, names, r = parse_french_daily(FRENCH)
    assert names == ["NoDur", "Durbl", "Hlth", "Other"] and len(dates) == 3
    assert r[0, 0] == 0.005 and np.isnan(r[0, 3])
    st = store_from_returns(dates, names, r, delay_days=1, source="fixture")
    b = st.tables["bars"]
    a = st.bars(["NoDur"], 10, knowledge=int(b["ts_avail"].max()))["NoDur"]
    assert np.allclose(a["close"], [100.5, 100.5 * 0.998, 100.5 * 0.998 * 1.01])
    assert (b["ts_avail"] - b["ts_event"] == 86_400_000_000).all()
    assert st.meta["source"]["point_in_time"] is False
    with pytest.raises(ValueError):
        export_from_store(st, str(tmp_path / "x"), commit="t")
    export_from_store(
        st, str(tmp_path / "y"), derived_only=True, terms="copyright notice only", commit="t"
    )
    assert validate_export(str(tmp_path / "y"))["ok"]


def test_binance_parser_handles_both_timestamp_units_and_header():
    rows = parse_binance_klines(BINANCE)
    assert len(rows) == 3
    assert rows[0][0] == 1704067200000 * 1000 and rows[2][0] == 1735689600000000
    st = store_from_binance({"BTCUSDT": rows})
    b = st.tables["bars"]
    assert (b["ts_event"] - b["ts_open"] == 86_400_000_000).all()
    assert st.calendar_name == "crypto_daily" and b["close"].tolist()[-1] == 94591.79


def test_french_illustration_refuses_to_download_when_offline(tmp_path):
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "illustrate_french.py")
    spec = importlib.util.spec_from_file_location("illustrate_french", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--offline", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "out")])
    assert rc == 2 and not (tmp_path / "out").exists()
