"""DuckDB views over a store and the as-of query as an ``ASOF JOIN``.

On-disk stores are exposed as views over their Parquet files; in-memory stores through
registered Arrow tables. DuckDB runs with one thread here (results are compared bit for bit).
The ``ASOF JOIN`` implementation of contract 2.4 is a second implementation; the tests compare
it with :func:`quant_research_engine.store.asof.asof_vintage`.
"""

from __future__ import annotations

import os

import duckdb
import numpy as np

from quant_research_engine._threads import DUCKDB_THREADS
from quant_research_engine.store.store import STORE_TABLES, Store, table_to_arrow


def connect(store: Store) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads TO {DUCKDB_THREADS}")
    for name in STORE_TABLES:
        if store.path and os.path.exists(os.path.join(store.path, f"{name}.parquet")):
            fp = os.path.join(store.path, f"{name}.parquet").replace("\\", "/").replace("'", "''")
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{fp}')")
        else:
            con.register(f"_{name}_arrow", table_to_arrow(store.tables[name]))
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM _{name}_arrow")
    return con


ASOF_SQL = """
WITH rows AS (
    SELECT series_id, epoch_us(ts_event) AS te, epoch_us(ts_avail) AS ta, vintage, value
    FROM series
),
probe AS (
    SELECT DISTINCT series_id, te, CAST(? AS BIGINT) AS t FROM rows
)
SELECT p.series_id, p.te, r.vintage, r.value
FROM probe p ASOF JOIN rows r
  ON p.series_id = r.series_id AND p.te = r.te AND p.t >= r.ta
ORDER BY p.series_id, p.te
"""


def asof_series_duckdb(con: duckdb.DuckDBPyConnection, knowledge: int) -> dict[str, np.ndarray]:
    """As-of rows of every series at ``knowledge`` via ``ASOF JOIN`` (inclusive)."""
    res = con.execute(ASOF_SQL, [int(knowledge)]).fetchall()
    return {
        "series_id": np.asarray([r[0] for r in res], dtype=object),
        "ts_event": np.asarray([r[1] for r in res], dtype=np.int64),
        "vintage": np.asarray([r[2] for r in res], dtype=np.int64),
        "value": np.asarray([r[3] for r in res], dtype=np.float64),
    }
