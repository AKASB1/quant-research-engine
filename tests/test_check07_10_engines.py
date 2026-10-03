"""Check 7 (engine equivalence on random scenarios), check 9 (identity and the independent
validator on the logs of every scenario), and check 10 (fill timing properties)."""

import os

import numpy as np
import pytest
from engine_helpers import tiny_store

from quant_research_engine.attribution import attribution
from quant_research_engine.context import Orders, OrderSpec, TargetShares, TargetWeights
from quant_research_engine.costs import CostConfig
from quant_research_engine.data.csvio import NO_TS
from quant_research_engine.engine import EngineConfig, EventEngine
from quant_research_engine.engine.logs import write_run_logs
from quant_research_engine.validation.equivalence import (
    compare,
    run_event_targets,
    run_vector_targets,
    scenario_config,
    write_vector_logs,
)
from quant_research_engine.validation.scenarios import make_scenario
from quant_research_engine.validation.validator import validate_run

SEEDS = range(1, 51)
PER_SEED = 4
_RUNS: dict = {}


def _scenario_runs():
    if not _RUNS:
        for s in SEEDS:
            for idx in range(PER_SEED):
                sc = make_scenario(s, idx)
                cfg = scenario_config(sc)
                ev = run_event_targets(sc.store, sc.dec_idx, sc.targets, cfg)
                vr = run_vector_targets(sc.store, sc.dec_idx, sc.targets, cfg)
                _RUNS[(s, idx)] = (sc, cfg, ev, vr)
    return _RUNS


@pytest.mark.slow
def test_engines_agree_on_200_random_scenarios(tmp_path):
    runs = _scenario_runs()
    excluded, compared, worst = [], 0, {}
    feats = {}
    for (s, idx), (sc, cfg, ev, vr) in runs.items():
        if ev.cap_binds or ev.rejected:
            excluded.append((s, idx, ev.cap_binds, len(ev.rejected)))
            continue
        c = compare(ev, vr, sc.store.panel(), cfg.fill_model)
        assert c["fills_match"], (s, idx, c)
        for k in ("quantity", "ref_price", "price", "spread", "impact", "commission"):
            v = c.get(f"max_diff_{k}", 0.0)
            assert v <= 1e-10, (s, idx, k, v)
            worst[k] = max(worst.get(k, 0.0), v)
        assert c["max_diff_equity"] <= 1e-12, (s, idx, c["max_diff_equity"])
        compared += 1
        for k, v in sc.features.items():
            feats[k] = feats.get(k, 0) + v
        feats["fill_delay_" + str(sc.fill_delay)] = (
            feats.get("fill_delay_" + str(sc.fill_delay), 0) + 1
        )
        feats["shorts_on" if sc.allow_short else "shorts_off"] = (
            feats.get("shorts_on" if sc.allow_short else "shorts_off", 0) + 1
        )
    assert compared >= 200 - len(excluded) and compared >= 190
    assert len(excluded) <= 10, excluded
    for k in (
        "splits",
        "dividends",
        "delistings",
        "late_listings",
        "zero_volume_bars",
        "lots",
        "fill_delay_1",
        "fill_delay_2",
        "fill_delay_3",
        "shorts_on",
        "shorts_off",
    ):
        assert feats.get(k, 0) > 0, k


@pytest.mark.slow
def test_validator_confirms_logs_of_every_scenario(tmp_path):
    for (s, idx), (sc, cfg, ev, vr) in list(_scenario_runs().items()):
        d = str(tmp_path / f"{s}_{idx}")
        write_run_logs(ev, cfg, os.path.join(d, "event"))
        r = validate_run(os.path.join(d, "event"), sc.store)
        assert r.ok, (s, idx, r.errors[:3])
        if not (ev.cap_binds or ev.rejected):
            write_vector_logs(vr, sc.store.panel(), sc.dec_idx, cfg, os.path.join(d, "vector"))
            r = validate_run(os.path.join(d, "vector"), sc.store)
            assert r.ok, (s, idx, r.errors[:3])
        assert ev.max_residual <= 1e-9 and vr.max_residual <= 1e-9
        assert attribution(ev)["sums_match"], (s, idx)


def test_validator_detects_a_corrupted_log(tmp_path):
    sc, cfg, ev, _ = _scenario_runs()[(1, 0)] if _RUNS else (None, None, None, None)
    if sc is None:
        sc = make_scenario(1, 0)
        cfg = scenario_config(sc)
        ev = run_event_targets(sc.store, sc.dec_idx, sc.targets, cfg)
    d = str(tmp_path / "log")
    write_run_logs(ev, cfg, d)
    path = os.path.join(d, "equity.csv")
    lines = open(path, encoding="utf-8").read().split("\n")
    parts = lines[5].split(",")
    parts[1] = repr(float(parts[1]) + 1.0)
    parts[3] = repr(float(parts[3]) + 1.0)
    lines[5] = ",".join(parts)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    with pytest.raises(ValueError):
        validate_run(d, sc.store)  # the manifest hash no longer matches
    from quant_research_engine.data.manifest import dataset_manifest, manifest_path, write_json

    with pytest.raises(ValueError):  # the loader's own identity check rejects the file
        data = open(path, "rb").read()
        write_json(manifest_path(path), dataset_manifest(data, len(lines) - 2, {"name": "x"}, None))
        validate_run(d, sc.store)
    # a fill price that disagrees with the cost model passes every loader; the validator finds it
    write_run_logs(ev, cfg, d)
    fp = os.path.join(d, "fills.csv")
    fl = open(fp, encoding="utf-8").read().split(chr(10))
    j = next(n for n in range(1, len(fl)) if fl[n] and ",DELIST," not in fl[n])
    parts = fl[j].split(",")
    parts[6] = repr(float(parts[6]) * 1.001)
    fl[j] = ",".join(parts)
    data = chr(10).join(fl).encode("utf-8")
    open(fp, "wb").write(data)
    write_json(
        manifest_path(fp),
        dataset_manifest(data, len([x for x in fl if x]) - 1, {"name": "x"}, None),
    )
    r = validate_run(d, sc.store)
    assert not r.ok and any("price" in e or "cash" in e for e in r.errors)


@pytest.mark.slow
def test_fill_timing_properties_on_random_scenarios():
    for (_s, _idx), (sc, cfg, ev, _) in list(_scenario_runs().items())[:120]:
        p = sc.store.panel()
        orders = {o[0]: o for o in ev.orders}
        pos = {iid: j for j, iid in enumerate(p.ids)}
        bar_no = np.cumsum(p.has_bar, axis=0) - 1
        filled_per_bar: dict = {}
        for f in ev.fills:
            i = pos[f[3]]
            k = int(np.searchsorted(p.ts_event, f[2]))  # bar whose open or close is ts_fill
            if f[1] == "DELIST":
                assert k == p.last_bar[i]
                continue
            o = orders[f[1]]
            kd = int(np.searchsorted(p.ts_event, o[1]))
            b_dec = int(np.cumsum(p.has_bar[: kd + 1, i])[-1]) - 1
            assert bar_no[k, i] >= b_dec + cfg.fill_delay_bars
            assert p.volume[k, i] > 0  # nothing fills in a zero-volume bar
            filled_per_bar[(k, i)] = filled_per_bar.get((k, i), 0.0) + abs(f[4])
        for (k, i), q in filled_per_bar.items():
            assert q <= cfg.costs.participation_cap * p.volume[k, i] + 1e-9
        # no position survives an instrument's delisting
        for row in ev.positions_rows:
            i = pos[row[1]]
            if p.ts_delist[i] != NO_TS:
                assert row[0] < p.ts_delist[i]


def test_trading_continues_with_delays_two_and_three():
    prices = [(10.0 + 0.1 * k, 10.0 + 0.1 * k, 1e6) for k in range(12)]
    st = tiny_store({"X": prices})
    for delay in (2, 3):
        eng = EventEngine(st.panel(), EngineConfig(fill_delay_bars=delay, initial_cash=1e6))
        for k in range(12):
            eng.process_bar(k)
            eng.submit(TargetShares({"X": 100.0 * (1 + k % 3)}), ["X"])
        r = eng.result()
        fill_bars = sorted({int(np.searchsorted(st.panel().ts_open, f[2])) for f in r.fills})
        assert len(fill_bars) >= 12 - delay - 1, (delay, fill_bars)
        assert min(fill_bars) == delay


def test_pending_order_cancelled_at_delisting():
    st = tiny_store(
        {"X": [(10.0, 10.0, 1e6), (10.0, 10.0, 0.0), (10.0, 10.0, 0.0)]}, delist={"X": -0.2}
    )
    eng = EventEngine(st.panel(), EngineConfig(initial_cash=1e5))
    eng.process_bar(0)
    eng.submit(TargetShares({"X": 100.0}), ["X"])  # waits: zero volume until the delisting
    eng.process_bar(1)
    eng.process_bar(2)
    r = eng.result()
    assert r.fills == [] and r.cancelled and r.cancelled[0][4] == "delisted"
    assert eng.E[0] == 0.0 and not eng.pending[0]


def test_day_orders_cancel_on_zero_volume_and_gtc_wait():
    st = tiny_store({"X": [(10.0, 10.0, 1e6), (10.0, 10.0, 0.0), (11.0, 11.0, 1e6)]})
    for tif, n in (("day", 0), ("gtc", 1)):
        eng = EventEngine(st.panel(), EngineConfig(initial_cash=1e5))
        eng.process_bar(0)
        eng.submit(Orders((OrderSpec("X", 10.0, tif),)), ["X"])
        eng.process_bar(1)
        eng.process_bar(2)
        r = eng.result()
        assert len(r.fills) == n, tif
        if n:
            assert r.fills[0][5] == 11.0


def test_unsafe_same_bar_fill_is_flagged_and_labelled(tmp_path):
    st = tiny_store({"X": [(10.0, 10.0, 1e6), (10.0, 12.0, 1e6), (12.0, 12.0, 1e6)]})
    with pytest.raises(ValueError):
        EventEngine(st.panel(), EngineConfig(fill_model="same_close"))
    cfg = EngineConfig(
        fill_model="same_close",
        unsafe_same_bar_fill=True,
        initial_cash=1e5,
        costs=CostConfig(impact={"model": "none"}),
    )
    eng = EventEngine(st.panel(), cfg)
    eng.process_bar(0)
    eng.process_bar(1)
    eng.submit(TargetWeights({"X": 0.5}), ["X"])
    eng.fill_same_close()
    eng.process_bar(2)
    r = eng.result()
    assert r.unsafe and r.fills[0][2] == int(st.panel().ts_event[1]) and r.fills[0][5] == 12.0
    run = write_run_logs(r, cfg, str(tmp_path / "u"))
    assert run["label"] == "UNSAFE"
    assert "UNSAFE" in open(tmp_path / "u" / "equity.manifest.json", encoding="utf-8").read()
    assert validate_run(str(tmp_path / "u"), st).ok
