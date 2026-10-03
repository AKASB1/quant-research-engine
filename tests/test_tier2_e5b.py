"""Tier 2 item 1: the backtest paths of combinatorial purged CV and a small E5b run."""

import numpy as np

from quant_research_engine.experiments import e5b
from quant_research_engine.inference.splitters import cpcv, violations


def test_cpcv_paths_cover_every_observation_once_and_use_every_split_group_once():
    for n, groups, k, h in ((600, 6, 2, 5), (97, 6, 2, 1), (1000, 8, 3, 21)):
        splits, n_paths = cpcv(n, groups, k, h)
        paths = e5b.cpcv_paths(n, groups, splits, n_paths)
        assert len(paths) == n_paths
        used = {}
        for parts in paths:
            covered = np.concatenate([splits[si][1][sel] for si, sel in parts])
            assert np.array_equal(np.sort(covered), np.arange(n))
            for si, sel in parts:
                key = (si, int(splits[si][1][sel][0]))
                used[key] = used.get(key, 0) + 1
        # every (split, tested group) pair appears in exactly one path
        assert len(used) == len(splits) * k and set(used.values()) == {1}
        # and every split is purged for the label horizon
        assert violations([(tr, te) for tr, te, _ in splits], h) == 0


def _small_cfg():
    return {
        "cpcv_groups": 6, "cpcv_test_groups": 2, "folds": 5, "gap": 21, "heldout": 60,
        "horizons": [5], "id": "e5b", "knn_k": 10, "n_bars": 381, "n_instruments": 8,
        "sample": 300, "null": {"seeds": [1, 1], "synth": "null"},
        "planted": {"seeds": [2, 2], "synth": "planted"},
        "schemes": ["shuffled_kfold", "contiguous_kfold", "purged_kfold", "walk_forward"],
    }  # fmt: skip


def test_e5b_job_rows_labels_and_determinism():
    cfg = _small_cfg()
    a = e5b.run_job(("market", "planted", 2), cfg, "quick", None)
    b = e5b.run_job(("market", "planted", 2), cfg, "quick", None)
    assert repr(a) == repr(b)
    schemes = [r["scheme"] for r in a]
    assert schemes == cfg["schemes"] + ["cpcv"] * 5
    labels = {r["scheme"]: r["label"] for r in a}
    assert labels["shuffled_kfold"] == "UNPURGED" and labels["contiguous_kfold"] == "UNPURGED"
    assert labels["purged_kfold"] == "purged" and labels["cpcv"] == "purged"
    for r in a:
        assert np.isfinite(r["cv_spearman"]) and np.isfinite(r["heldout_spearman"])
        assert r["bias_spearman"] == r["cv_spearman"] - r["heldout_spearman"]
