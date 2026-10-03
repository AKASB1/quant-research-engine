"""The numeric comparison of generated files with committed ones (``helpers.diff_*``)."""

import json

from helpers import diff_csv_text, diff_files, diff_json

HEAD = "ts_event,ts_avail,instrument_id,adv_shares,sigma_bar\n"
ROW = "2010-02-04T21:00:00Z,2010-02-04T21:00:00Z,I003,1194983.95,0.028550409169237203\n"
STRICT = 1e-12


def _csv(row: str = ROW, extra: str = "") -> str:
    return HEAD + row + extra


def test_last_digit_change_of_a_float_cell_passes():
    linux = ROW.replace("0.028550409169237203", "0.0285504091692372")
    assert diff_csv_text(_csv(), _csv(linux), STRICT) == []


def test_relative_change_of_1e_9_fails():
    v = 0.028550409169237203 * (1 + 1e-9)
    assert diff_csv_text(_csv(), _csv(ROW.replace("0.028550409169237203", repr(v))), STRICT)


def test_changed_identifier_timestamp_or_integer_fails():
    assert diff_csv_text(_csv(), _csv(ROW.replace("I003", "I004")), STRICT)
    assert diff_csv_text(
        _csv(),
        _csv(ROW.replace("2010-02-04T21:00:00Z,2010", "2010-02-05T21:00:00Z,2010", 1)),
        STRICT,
    )
    # integers are exact whatever the tolerance
    assert diff_csv_text("a,n\nx,134\n", "a,n\nx,135\n", 1.0)
    assert diff_csv_text("a,n\nx,0\n", "a,n\nx,0.0\n", 1.0)


def test_changed_number_of_rows_or_cells_fails():
    assert diff_csv_text(_csv(), _csv(extra=ROW), STRICT)
    assert diff_csv_text(_csv(), _csv(ROW.rstrip("\n") + ",1\n"), STRICT)


def test_manifest_fields_other_than_the_hash_must_agree(tmp_path):
    m = {
        "content_sha256": "a" * 64,
        "row_count": 134,
        "seed": 1,
        "generator": {"name": "qre.synth", "ic": 0.02},
    }
    a, b = tmp_path / "a.manifest.json", tmp_path / "b.manifest.json"
    a.write_text(json.dumps(m), encoding="utf-8")
    b.write_text(json.dumps({**m, "content_sha256": "b" * 64}), encoding="utf-8")
    assert diff_files(str(a), str(b), STRICT, ignore={"content_sha256"}) == []
    assert diff_files(str(a), str(b), STRICT)  # the hash counts unless it is left out
    for change in (
        {"row_count": 135},
        {"seed": 2},
        {"generator": {"name": "qre.synth", "ic": 0.03}},
    ):
        b.write_text(json.dumps({**m, **change}), encoding="utf-8")
        assert diff_files(str(a), str(b), STRICT, ignore={"content_sha256"}), change


def test_loose_json_rule_of_the_example_run():
    loose = {"rel_tol": 1e-9, "abs_tol": 1e-9, "ignore": {"content_sha256", "files"}}
    run = {
        "max_identity_residual": 3e-16,
        "files": {"fills": "a" * 64},
        "cap_binds": 1,
        "label": "x",
        "books": {"long": {"hold_pnl": -41349.346243355525}},
        "unsafe": False,
        "seed": None,
    }
    other = json.loads(json.dumps(run))
    other["max_identity_residual"] = 9e-16  # round-off-sized values cannot fail on noise
    other["files"] = {"fills": "b" * 64}
    other["books"]["long"]["hold_pnl"] = -41349.34624335553
    assert diff_json(run, other, **loose) == []
    for path, value in ((("cap_binds",), 2), (("label",), "y"), (("unsafe",), True), (("seed",), 0),
                        (("books", "long", "hold_pnl"), -41349.4)):  # fmt: skip
        changed = json.loads(json.dumps(run))
        node = changed
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        assert diff_json(run, changed, **loose), path
    assert diff_json(run, {**run, "extra": 1}, **loose)
