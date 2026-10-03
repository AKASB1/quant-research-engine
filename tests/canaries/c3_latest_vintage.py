"""C3 latest_vintage: for each instrument, takes the latest period of the revised series fund_x
whose first release is known, but reads that period's last vintage (possibly not yet
published) instead of the as-of vintage. Guard: G2 (the poison changes later vintages)."""

import os

import numpy as np
from _common import demeaned_weights

from quant_research_engine.context import Strategy
from quant_research_engine.store.registry import open_store

CANARY = {"id": "C3", "name": "latest_vintage", "guard": "G2", "class": "LatestVintage"}


class LatestVintage(Strategy):
    name = "latest_vintage"
    history_bars = 1

    def decide(self, ctx):
        st = open_store(os.environ.get("QRE_STORE"))
        ser = st.tables["series"]
        ids = ctx.universe()
        vals = []
        for iid in ids:
            sl = st._series_slices.get(f"{iid}.fund_x")
            if sl is None:
                vals.append(np.nan)
                continue
            a, b = sl
            te, ta, vin, val = (ser[k][a:b] for k in ("ts_event", "ts_avail", "vintage", "value"))
            first_known = (vin == 0) & (ta <= ctx.t)
            if not first_known.any():
                vals.append(np.nan)
                continue
            period = te[first_known].max()
            rows = np.flatnonzero(te == period)
            vals.append(float(val[rows[np.argmax(vin[rows])]]))
        return demeaned_weights(ids, vals)
