"""Thread settings for numerical libraries (set before NumPy, Polars, or DuckDB load).

The machine this project runs on is shared, and results compared byte for byte must not depend
on multi-threaded reductions. Entry points call :func:`pin_threads` before importing NumPy;
the values are recorded in every manifest by :func:`thread_settings`.
"""

from __future__ import annotations

import os

DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
}
DUCKDB_THREADS = 1


def pin_threads() -> None:
    """Set the thread variables unless the caller already set them."""
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)


def thread_settings() -> dict[str, str]:
    out = {key: os.environ.get(key, "") for key in sorted(DEFAULTS)}
    out["DUCKDB_THREADS"] = str(DUCKDB_THREADS)
    return out
