"""C5 vendor_adjusted_close: ranks instruments by price level on a close adjusted by every split
in the table, future ones included (as a vendor's back-adjusted series is), so its decisions
depend on future splits through the levels. Guard: G2."""

import os

import numpy as np
from _common import demeaned_weights

from quant_research_engine.context import Strategy
from quant_research_engine.store.registry import open_store

CANARY = {
    "id": "C5",
    "name": "vendor_adjusted_close",
    "guard": "G2",
    "class": "VendorAdjustedClose",
}


class VendorAdjustedClose(Strategy):
    name = "vendor_adjusted_close"
    history_bars = 1

    def decide(self, ctx):
        ca = open_store(os.environ.get("QRE_STORE")).tables["corporate_actions"]
        close, _ = ctx.bars("close", 1)
        ids = ctx.universe()
        levels = []
        for j, iid in enumerate(ids):
            f = 1.0
            for a, i2, ex, v in zip(
                ca["action"], ca["instrument_id"], ca["ts_ex"], ca["value"], strict=True
            ):
                if a == "split" and i2 == iid and ex > ctx.t:
                    f *= float(v)
            c = close[0, j] if close.shape[0] else np.nan
            levels.append(np.log(c / f) if np.isfinite(c) else np.nan)
        lv = np.asarray(levels)
        ok = np.isfinite(lv)
        ranks = np.full(lv.shape, np.nan)
        if ok.any():
            order = np.argsort(lv[ok], kind="stable")
            r = np.empty(order.size)
            r[order] = np.arange(order.size, dtype=np.float64)
            ranks[ok] = -r
        return demeaned_weights(ids, ranks)
