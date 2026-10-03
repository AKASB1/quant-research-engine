"""Shared test helpers: the golden store, random point-in-time data, and the numeric comparison
of generated files with committed ones."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re

import numpy as np
from conftest import GOLDEN

from quant_research_engine.data import load_csv, make_table
from quant_research_engine.store.builder import build_store
from quant_research_engine.timeutil import parse_ts


def golden_store():
    t = {
        n: load_csv(os.path.join(GOLDEN, f"{n}_v1.csv"))
        for n in ("instruments", "bars", "corporate_actions", "series")
    }
    return build_store(
        t["instruments"],
        t["bars"],
        t["corporate_actions"],
        t["series"],
        {"calendar": "equity_daily", "ppy": 252, "generator": {"name": "golden"}, "seed": None},
    )


def T(s: str) -> int:
    return parse_ts(s)


def random_series_rows(rng: np.random.Generator, n_series: int = 3, n_periods: int = 4):
    """Random revised series sorted by (series_id, ts_event, vintage); instants on a coarse grid
    so that ties between periods and probe instants are frequent."""
    rows = {"series_id": [], "ts_event": [], "ts_avail": [], "vintage": [], "value": []}
    step = 3_600_000_000
    for s in range(n_series):
        sid = f"S{s}.x"
        events = sorted(set(rng.integers(0, 20, size=n_periods).tolist()))
        for e in events:
            nv = int(rng.integers(1, 4))
            av = e + int(rng.integers(0, 3))
            for v in range(nv):
                rows["series_id"].append(sid)
                rows["ts_event"].append(e * step)
                rows["ts_avail"].append(av * step)
                rows["vintage"].append(v)
                rows["value"].append(float(rng.normal()))
                av += int(rng.integers(1, 4))
    return make_table("series", **rows)


# --- comparing generated files with committed ones --------------------------------------------
#
# Floats written at full precision differ in the last digit between math libraries (for example,
# NumPy's AVX-512 path for ``power`` is one unit in the last place off the scalar ``pow`` for
# some arguments), so a committed fixture cannot be compared byte for byte on every machine.
# These helpers compare CSV files cell by cell and JSON files as parsed objects: text that is
# equal passes; an integer, an identifier, or a timestamp must be equal as text; other cells
# must parse as floats and agree within the given tolerance.

_INT = re.compile(r"[+-]?\d+")


def _float(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None


def _close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def diff_csv_text(
    a: str, b: str, rel_tol: float, abs_tol: float = 0.0, name: str = ""
) -> list[str]:
    """Differences between two CSV texts (empty list: equal under the rule above)."""
    ra = list(csv.reader(io.StringIO(a)))
    rb = list(csv.reader(io.StringIO(b)))
    if len(ra) != len(rb):
        return [f"{name}: {len(ra)} lines vs {len(rb)}"]
    out = []
    for i, (la, lb) in enumerate(zip(ra, rb, strict=True)):
        if len(la) != len(lb):
            out.append(f"{name}:{i + 1}: {len(la)} cells vs {len(lb)}")
            continue
        for j, (x, y) in enumerate(zip(la, lb, strict=True)):
            if x == y:
                continue
            fx, fy = _float(x), _float(y)
            exact = _INT.fullmatch(x) or _INT.fullmatch(y)
            if exact or fx is None or fy is None or not _close(fx, fy, rel_tol, abs_tol):
                out.append(f"{name}:{i + 1}:{j + 1}: {x!r} vs {y!r}")
    return out


def diff_json(
    a, b, rel_tol: float, abs_tol: float = 0.0, ignore=frozenset(), path: str = "$"
) -> list[str]:
    """Differences between two parsed JSON values; keys in ``ignore`` are skipped at any depth.
    Integers, booleans, strings, and null must be equal; a float must agree within the
    tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka = {k for k in a if k not in ignore}
        kb = {k for k in b if k not in ignore}
        if ka != kb:
            return [f"{path}: keys {sorted(ka ^ kb)} differ"]
        out = []
        for k in sorted(ka):
            out += diff_json(a[k], b[k], rel_tol, abs_tol, ignore, f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += diff_json(x, y, rel_tol, abs_tol, ignore, f"{path}[{i}]")
        return out
    numeric = (int, float)
    if (
        (isinstance(a, float) or isinstance(b, float))
        and isinstance(a, numeric)
        and isinstance(b, numeric)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        return [] if _close(float(a), float(b), rel_tol, abs_tol) else [f"{path}: {a!r} vs {b!r}"]
    if type(a) is not type(b) or a != b:
        return [f"{path}: {a!r} vs {b!r}"]
    return []


def diff_files(
    committed: str, generated: str, rel_tol: float, abs_tol: float = 0.0, ignore=frozenset()
) -> list[str]:
    """Compare one committed file with its regenerated counterpart: ``.csv`` cell by cell,
    ``.json`` as parsed objects, anything else byte for byte."""
    name = os.path.basename(committed)
    if name.endswith(".csv"):
        with (
            open(committed, encoding="utf-8", newline="") as f,
            open(generated, encoding="utf-8", newline="") as g,
        ):
            return diff_csv_text(f.read(), g.read(), rel_tol, abs_tol, name)
    if name.endswith(".json"):
        with open(committed, encoding="utf-8") as f, open(generated, encoding="utf-8") as g:
            return diff_json(json.load(f), json.load(g), rel_tol, abs_tol, ignore, name)
    with open(committed, "rb") as f, open(generated, "rb") as g:
        return [] if f.read() == g.read() else [f"{name}: bytes differ"]
