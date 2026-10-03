from datetime import date

from quant_research_engine.store.calendars import crypto_daily, equity_daily, monthly, weekly
from quant_research_engine.timeutil import format_ts


def test_equity_daily_sessions_and_holidays():
    c = equity_daily(date(2023, 12, 22), 5)
    ev = [format_ts(x) for x in c.ts_event]
    # 23/24 Dec weekend, 25 Dec holiday, 1 Jan holiday
    assert ev == [
        "2023-12-22T21:00:00Z",
        "2023-12-26T21:00:00Z",
        "2023-12-27T21:00:00Z",
        "2023-12-28T21:00:00Z",
        "2023-12-29T21:00:00Z",
    ]
    assert format_ts(c.ts_open[0]) == "2023-12-22T14:30:00Z"
    assert (
        equity_daily(date(2024, 1, 1), 1).ts_event[0]
        == equity_daily(date(2024, 1, 2), 1).ts_event[0]
    )


def test_crypto_daily_and_derived():
    c = crypto_daily(date(2024, 2, 27), 6)
    assert format_ts(c.ts_open[0]) == "2024-02-27T00:00:00Z"
    assert format_ts(c.ts_event[0]) == "2024-02-28T00:00:00Z"
    m = monthly(c)
    assert [format_ts(x) for x in m.ts_event] == ["2024-03-01T00:00:00Z", "2024-03-04T00:00:00Z"]
    e = equity_daily(date(2024, 3, 4), 10)
    w = weekly(e)
    assert [format_ts(x) for x in w.ts_event] == ["2024-03-08T21:00:00Z", "2024-03-15T21:00:00Z"]
    assert format_ts(w.ts_open[1]) == "2024-03-11T14:30:00Z"
