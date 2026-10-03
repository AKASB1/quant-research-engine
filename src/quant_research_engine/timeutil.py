"""UTC instants: files use ``YYYY-MM-DDTHH:MM:SSZ``; memory uses int64 microseconds since the epoch.

No naive or local times, no time-zone names. A day count between two instants is the number of
seconds divided by 86400.
"""

from __future__ import annotations

import calendar
import re
from datetime import UTC, datetime

import numpy as np

US_PER_S = 1_000_000
US_PER_DAY = 86_400 * US_PER_S
_TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$")


def parse_ts(text: str) -> int:
    """Parse ``YYYY-MM-DDTHH:MM:SSZ`` into microseconds since the epoch; raise ValueError."""
    m = _TS_RE.match(text)
    if m is None:
        raise ValueError(f"bad timestamp {text!r} (expected YYYY-MM-DDTHH:MM:SSZ)")
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    datetime(y, mo, d, h, mi, s)  # range check (raises ValueError)
    return calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) * US_PER_S


def format_ts(us: int) -> str:
    us = int(us)
    if us % US_PER_S:
        raise ValueError("instants in files have seconds precision")
    return datetime.fromtimestamp(us // US_PER_S, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_ts_array(us: np.ndarray) -> list[str]:
    arr = np.asarray(us, dtype=np.int64)
    if arr.size and np.any(arr % US_PER_S):
        raise ValueError("instants in files have seconds precision")
    s = np.datetime_as_string(arr.astype("datetime64[us]").astype("datetime64[s]"), unit="s")
    return [x + "Z" for x in s.tolist()]


def ts(y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> int:
    return calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) * US_PER_S


def days_between(a_us: int, b_us: int) -> float:
    return (b_us - a_us) / US_PER_DAY


def to_datetime64(us: np.ndarray) -> np.ndarray:
    return np.asarray(us, dtype=np.int64).astype("datetime64[us]")
