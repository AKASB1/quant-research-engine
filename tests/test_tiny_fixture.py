"""The committed tiny CSV fixture is what the generator writes, and it passes the loaders.

The generated tables are compared with the committed ones cell by cell (``helpers.diff_files``):
floats agree to a relative 1e-12, everything else exactly, and each manifest as parsed JSON
without its ``content_sha256``. A byte comparison would fail on a machine whose math library
differs in the last digit: on a CPU with AVX-512, NumPy 2.5.3 computes four of the 250
``sigma_bar`` weights one unit in the last place lower than here, and five ``sigma_bar`` cells
of ``liquidity.csv`` then differ in their last digits (an independent Linux check)."""

import math
import os

import numpy as np
from helpers import diff_files

from quant_research_engine.data import load_csv
from quant_research_engine.data.manifest import check_manifest
from quant_research_engine.store import liquidity
from quant_research_engine.store.csvdump import write_store_csv
from quant_research_engine.store.store import STORE_TABLES
from quant_research_engine.synth import generate, load_synth_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "tests", "data", "tiny")
REL = 1e-12
AVX512_LAGS = (6, 22, 42, 62)  # where NumPy's AVX-512 power differs from math.pow


def _write_tiny(out: str) -> None:
    st, _ = generate(load_synth_config(os.path.join(ROOT, "configs", "synth", "tiny.json")), 1)
    write_store_csv(st, out)


def _differences(out: str) -> list[str]:
    diffs = []
    for name in STORE_TABLES:
        diffs += diff_files(
            os.path.join(TINY, name + ".csv"), os.path.join(out, name + ".csv"), REL
        )
        diffs += diff_files(
            os.path.join(TINY, name + ".manifest.json"),
            os.path.join(out, name + ".manifest.json"),
            REL,
            ignore={"content_sha256"},
        )
    return diffs


def test_tiny_fixture_is_reproduced(tmp_path):
    _write_tiny(str(tmp_path))
    assert _differences(str(tmp_path)) == []
    total = 0
    for name in STORE_TABLES:
        path = os.path.join(TINY, name + ".csv")
        t = load_csv(path, name)
        with open(path, "rb") as f:
            data = f.read()
        check_manifest(path, data, len(t))
        with open(os.path.join(TINY, name + ".manifest.json"), "rb") as f:
            total += len(data) + len(f.read())
    assert total < 100_000


def test_last_place_weights_change_the_bytes_but_not_the_comparison(tmp_path, monkeypatch):
    """Reproduces the AVX-512 situation on any machine: the four weights one unit in the last
    place lower change ``liquidity.csv`` but not what the comparison accepts."""

    def lowered():
        # the scalar pow everywhere, one unit in the last place lower at the four lags: the
        # weights NumPy's AVX-512 path produces, whatever path this machine's NumPy takes
        w = np.array(
            [math.pow(0.5, k / liquidity.HALF_LIFE) for k in range(liquidity.SIGMA_WINDOW)]
        )
        for k in AVX512_LAGS:
            w[k] = np.nextafter(w[k], 0.0)
        return w

    monkeypatch.setattr(liquidity, "_weights", lowered)
    _write_tiny(str(tmp_path))
    with open(os.path.join(TINY, "liquidity.csv"), encoding="utf-8") as f:
        committed = f.read().split("\n")
    with open(os.path.join(tmp_path, "liquidity.csv"), encoding="utf-8") as f:
        generated = f.read().split("\n")
    changed = [i for i, (a, b) in enumerate(zip(committed, generated, strict=True)) if a != b]
    # 5 rows here: the same five sigma_bar cells, with the same values, as the Linux check
    assert len(changed) >= 1, "the perturbation must change liquidity.csv"
    assert _differences(str(tmp_path)) == []
