"""Writing the logs of one run: the four result files of the contract (orders, fills, positions,
equity) with manifests, the per-instrument attribution, and ``run.json``. A run with
``unsafe_same_bar_fill`` carries the label ``UNSAFE`` in ``run.json`` and in every manifest."""

from __future__ import annotations

import os

from quant_research_engine.data.csvio import write_csv
from quant_research_engine.data.manifest import dataset_manifest, manifest_path, write_json
from quant_research_engine.engine.event import EngineConfig, EventResult

COMPONENTS = ("hold_pnl", "trade_pnl", "fill_costs", "dividends", "borrow")


def write_run_logs(
    result: EventResult,
    cfg: EngineConfig,
    out_dir: str,
    meta: dict | None = None,
    strategy_id: str = "strategy",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    label = "UNSAFE" if result.unsafe else "safe"
    gen = {"name": "qre.engine.event", "label": label, "parameters": {"strategy_id": strategy_id}}
    hashes = {}
    for name, table in result.tables(strategy_id).items():
        path = os.path.join(out_dir, f"{name}.csv")
        data = write_csv(path, table)
        write_json(
            manifest_path(path), dataset_manifest(data, len(table), gen, (meta or {}).get("seed"))
        )
        hashes[name] = dataset_manifest(data, len(table), gen, None)["content_sha256"]
    ids = sorted(result.per_instrument)
    attr = {"instrument_id": ids}
    for c in COMPONENTS:
        attr[c] = [result.per_instrument[i][c] for i in ids]
    lines = ["instrument_id," + ",".join(COMPONENTS)]
    for j, iid in enumerate(ids):
        lines.append(iid + "," + ",".join(repr(float(attr[c][j]) + 0.0) for c in COMPONENTS))
    with open(os.path.join(out_dir, "attribution.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    run = {
        "label": label,
        "engine": {**cfg.model_dump(mode="json"), "fill_model": result.fill_model},
        "participation_cap_enforced": True,
        "max_identity_residual": result.max_residual,
        "rejected_orders": len(result.rejected),
        "cancelled_orders": len(result.cancelled),
        "impact_unavailable": result.impact_unavailable,
        "cap_binds": result.cap_binds,
        "books": result.books,
        "files": hashes,
    }
    run.update(meta or {})
    write_json(os.path.join(out_dir, "run.json"), run)
    return run
