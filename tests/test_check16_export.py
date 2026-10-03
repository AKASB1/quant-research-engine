"""Check 16: export v1 (golden bytes, validation, round trip, knowledge rule, recomputation of
every derived row from the store cut at its ts_avail, the derived-only refusal)."""

import os
import shutil

import numpy as np
import pytest
from conftest import GOLDEN

from quant_research_engine.data import load_csv
from quant_research_engine.data.manifest import read_json
from quant_research_engine.export import (
    export_from_store,
    forecast_panels,
    signal_panels,
    validate_export,
    write_tables,
)
from quant_research_engine.rng import stream
from quant_research_engine.store.store import cut_store
from quant_research_engine.synth import generate, load_synth_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_FILES = (
    "instruments",
    "bars",
    "corporate_actions",
    "returns",
    "signals",
    "forecasts",
    "liquidity",
)


def test_writer_reproduces_the_golden_bytes(tmp_path):
    tables = {n: load_csv(os.path.join(GOLDEN, f"{n}_v1.csv")) for n in EXPORT_FILES}
    write_tables(tables, str(tmp_path), {"name": "golden"}, None)
    for n in EXPORT_FILES:
        assert (
            open(tmp_path / f"{n}.csv", "rb").read()
            == open(os.path.join(GOLDEN, f"{n}_v1.csv"), "rb").read()
        )


@pytest.fixture(scope="module")
def planted_export(tmp_path_factory):
    cfg = load_synth_config(os.path.join(ROOT, "configs", "synth", "planted.json")).with_overrides(
        n_instruments=12, n_bars=400
    )
    st, _ = generate(cfg, 4)
    out = str(tmp_path_factory.mktemp("exp") / "export")
    export_from_store(st, out, commit="test")
    return st, out


def test_export_validates_round_trips_and_respects_knowledge(planted_export, tmp_path):
    st, out = planted_export
    rep = validate_export(out)
    assert rep["ok"], rep["errors"]
    man = read_json(os.path.join(out, "manifest.json"))
    assert man["export_version"] == 1 and man["derived_only"] is False
    again = str(tmp_path / "again")
    tables = {n: load_csv(os.path.join(out, f"{n}.csv"), n) for n in (*EXPORT_FILES, "series")}
    write_tables(tables, again, {"name": "x"}, None)
    for n in (*EXPORT_FILES, "series"):
        assert (
            open(os.path.join(out, f"{n}.csv"), "rb").read()
            == open(os.path.join(again, f"{n}.csv"), "rb").read()
        )
        if n not in ("instruments", "corporate_actions"):
            t = tables[n]
            assert (t["ts_avail"] >= t["ts_event"]).all()
    sig = tables["signals"]
    assert "signal_x" in set(sig["name"].tolist()) and len(set(sig["name"].tolist())) == 6
    fc = tables["forecasts"]
    assert set(fc["horizon_bars"].tolist()) == {1, 5, 21}


@pytest.mark.slow
def test_every_derived_row_is_recomputed_bit_for_bit_from_the_cut_store(planted_export):
    st, out = planted_export
    p = st.panel()
    full_sig = {name: vals for name, vals, _ in signal_panels(st)}
    full_fc = {h: forecast_panels(st, h) for h in (1, 5, 21)}
    liq = load_csv(os.path.join(out, "liquidity.csv"), "liquidity")
    ks = stream(8, "test.export").choice(np.arange(30, p.shape[0]), size=25, replace=False).tolist()
    for k in ks:
        t = int(p.ts_event[k])
        cs = cut_store(st, t)
        cp = cs.panel()
        cut_sig = {name: vals for name, vals, _ in signal_panels(cs)}
        for name, vals in cut_sig.items():
            for j, iid in enumerate(cp.ids):
                a, b = full_sig[name][k, p.ids.index(iid)], vals[k, j]
                assert (np.isnan(a) and np.isnan(b)) or a == b, (name, k, iid)
        for h in (1, 5, 21):
            mu, sg = forecast_panels(cs, h)
            for j, iid in enumerate(cp.ids):
                for full, cut in ((full_fc[h][0], mu), (full_fc[h][1], sg)):
                    a, b = full[k, p.ids.index(iid)], cut[k, j]
                    assert (np.isnan(a) and np.isnan(b)) or a == b, (h, k, iid)
        rows = np.flatnonzero(liq["ts_avail"] == t)
        cl = cs.tables["liquidity"]
        for r in rows.tolist():
            m = np.flatnonzero(
                (cl["instrument_id"] == liq["instrument_id"][r])
                & (cl["ts_event"] == liq["ts_event"][r])
            )
            assert m.size == 1
            assert (
                cl["adv_shares"][m[0]] == liq["adv_shares"][r]
                and cl["sigma_bar"][m[0]] == liq["sigma_bar"][r]
            )


def test_third_party_store_needs_derived_only(tmp_path):
    from quant_research_engine.store.csv_adapter import load_local_csv

    src = os.path.join(ROOT, "tests", "data", "adapter")
    shutil.copytree(src, tmp_path / "a")
    st = load_local_csv(str(tmp_path / "a" / "adapter.json"))
    with pytest.raises(ValueError):
        export_from_store(st, str(tmp_path / "x"), commit="test")
    export_from_store(
        st, str(tmp_path / "y"), derived_only=True, terms="test fixture, no terms", commit="test"
    )
    rep = validate_export(str(tmp_path / "y"))
    assert rep["ok"], rep["errors"]
    assert rep["rows"]["bars"] == 0 and rep["rows"]["corporate_actions"] == 0
    assert read_json(str(tmp_path / "y" / "manifest.json"))["derived_only"] is True
