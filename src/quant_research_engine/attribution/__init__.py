"""Attribution of PnL by instrument and by long and short book.

The event engine records, per bar and instrument, hold PnL, trade PnL, fill costs (spread,
impact, commission), dividends, and borrow; each instrument-bar is assigned to the long or the
short book by the sign of its opening position (or of its first fill when it opened flat).
Financing and interest on cash are portfolio-level. :func:`attribution` returns both views and
checks that each sums to the totals of the equity rows (the identity of the contract).
"""

from __future__ import annotations

import math

COMPONENTS = ("hold_pnl", "trade_pnl", "fill_costs", "dividends", "borrow")


def attribution(result) -> dict:
    rows = result.equity_rows
    totals = {
        "hold_pnl": math.fsum(r[4] for r in rows),
        "trade_pnl": math.fsum(r[5] for r in rows),
        "fill_costs": math.fsum(r[6] + r[7] + r[8] for r in rows),
        "borrow": math.fsum(r[9] for r in rows),
        "financing": math.fsum(r[10] for r in rows),
        "income": math.fsum(r[11] for r in rows),
    }
    per = result.per_instrument
    books = result.books
    scale = max(1.0, max((abs(r[3]) for r in rows), default=1.0))
    check = {}
    for c in ("hold_pnl", "trade_pnl", "fill_costs", "borrow"):
        s_inst = math.fsum(v[c] for v in per.values())
        s_book = math.fsum(v[c] for v in books.values())
        check[c] = max(abs(s_inst - totals[c]), abs(s_book - totals[c])) <= 1e-9 * scale * max(
            1, len(rows)
        )
    return {
        "by_instrument": per,
        "by_book": books,
        "totals": totals,
        "sums_match": all(check.values()),
        "check": check,
    }
