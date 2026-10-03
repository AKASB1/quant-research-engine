"""Trading calendars (shared contract 1.1).

``equity_daily``: Monday to Friday except 1 January and 25 December (no observance shift), every
session 14:30:00Z to 21:00:00Z (no daylight-saving handling). ``crypto_daily``: every day,
ts_open 00:00:00Z, ts_event 00:00:00Z of the next day. ``weekly`` and ``monthly`` take the last
session of each ISO week or calendar month of a daily calendar. Bars per year are declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from quant_research_engine.timeutil import US_PER_S, ts

PPY = {"equity_daily": 252, "crypto_daily": 365, "weekly": 52, "monthly": 12}
HOLIDAYS_MD = ((1, 1), (12, 25))


@dataclass(frozen=True)
class Calendar:
    name: str
    ts_open: np.ndarray  # int64 microseconds
    ts_event: np.ndarray

    def __len__(self) -> int:
        return len(self.ts_event)


def _is_equity_session(d: date) -> bool:
    return d.weekday() < 5 and (d.month, d.day) not in HOLIDAYS_MD


def equity_daily(start: date, n: int) -> Calendar:
    """The first ``n`` sessions on or after ``start``."""
    opens, events = [], []
    d = start
    while len(events) < n:
        if _is_equity_session(d):
            opens.append(ts(d.year, d.month, d.day, 14, 30))
            events.append(ts(d.year, d.month, d.day, 21, 0))
        d += timedelta(days=1)
    return Calendar("equity_daily", np.asarray(opens, np.int64), np.asarray(events, np.int64))


def crypto_daily(start: date, n: int) -> Calendar:
    o = ts(start.year, start.month, start.day)
    day = 86_400 * US_PER_S
    opens = o + day * np.arange(n, dtype=np.int64)
    return Calendar("crypto_daily", opens, opens + day)


def make_calendar(name: str, start: date, n: int) -> Calendar:
    if name == "equity_daily":
        return equity_daily(start, n)
    if name == "crypto_daily":
        return crypto_daily(start, n)
    raise ValueError(f"unknown daily calendar {name!r}")


def _period_last(cal: Calendar, key) -> Calendar:
    ev = cal.ts_event
    days = (ev // (86_400 * US_PER_S)).tolist()
    keys = [
        key(date(1970, 1, 1) + timedelta(days=int(d) - (1 if cal.name == "crypto_daily" else 0)))
        for d in days
    ]
    last = [i for i in range(len(keys)) if i == len(keys) - 1 or keys[i + 1] != keys[i]]
    idx = np.asarray(last, dtype=np.int64)
    first = [0] + [i + 1 for i in last[:-1]]
    return Calendar("", cal.ts_open[np.asarray(first, dtype=np.int64)], ev[idx])


def weekly(cal: Calendar) -> Calendar:
    """Periods = ISO weeks; each period's bar spans its first session's open to its last close."""
    c = _period_last(cal, lambda d: d.isocalendar()[:2])
    return Calendar("weekly", c.ts_open, c.ts_event)


def monthly(cal: Calendar) -> Calendar:
    c = _period_last(cal, lambda d: (d.year, d.month))
    return Calendar("monthly", c.ts_open, c.ts_event)
