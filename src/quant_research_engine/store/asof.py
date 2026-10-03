"""The single definition of as-of semantics for vintaged series (shared contract 2.4).

As of an instant ``t`` the value of (series_id, ts_event) is the row with the largest vintage
among the rows with ``ts_avail <= t``; with no such row the period is unknown at ``t``.
Every other as-of implementation (the DuckDB ``ASOF JOIN`` among them) is tested against this.
"""

from __future__ import annotations

import numpy as np


def asof_vintage(rows: dict[str, np.ndarray], t: int) -> np.ndarray:
    """Indices (ascending) of the rows that are the as-of values at ``t``.

    ``rows`` holds ``series_id``, ``ts_event``, ``ts_avail``, ``vintage`` sorted by
    (series_id, ts_event, vintage). The comparison ``ts_avail <= t`` is inclusive.
    """
    avail = np.asarray(rows["ts_avail"], dtype=np.int64)
    known = np.flatnonzero(avail <= t)
    if known.size == 0:
        return known
    sid = np.asarray(rows["series_id"], dtype=object)[known]
    te = np.asarray(rows["ts_event"], dtype=np.int64)[known]
    vin = np.asarray(rows["vintage"], dtype=np.int64)[known]
    # Within the sorted known rows, the last row of each (series_id, ts_event) run carries the
    # largest vintage, because rows are sorted by vintage inside each period.
    nxt_same = np.zeros(known.size, dtype=bool)
    if known.size > 1:
        nxt_same[:-1] = (sid[1:] == sid[:-1]) & (te[1:] == te[:-1])
        if np.any(nxt_same[:-1] & (vin[1:] <= vin[:-1])):
            raise ValueError("rows are not sorted by vintage within a period")
    return known[~nxt_same]


def asof_vintage_bruteforce(rows: dict[str, np.ndarray], t: int) -> np.ndarray:
    """Reference scan used by the tests: no sorting assumptions."""
    best: dict[tuple, int] = {}
    n = len(rows["ts_avail"])
    for i in range(n):
        if int(rows["ts_avail"][i]) > t:
            continue
        key = (rows["series_id"][i], int(rows["ts_event"][i]))
        j = best.get(key)
        if j is None or int(rows["vintage"][i]) > int(rows["vintage"][j]):
            best[key] = i
    return np.asarray(sorted(best.values()), dtype=np.int64)
