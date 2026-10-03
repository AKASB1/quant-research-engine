"""Write the tables of a store as version-1 CSV files with their manifests (small stores only:
the committed test fixture and inspection; bulk data stays in Parquet and is never committed)."""

from __future__ import annotations

import os

from quant_research_engine.data.csvio import write_csv
from quant_research_engine.data.manifest import dataset_manifest, manifest_path, write_json
from quant_research_engine.store.store import STORE_TABLES, Store


def write_store_csv(store: Store, out_dir: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    gen = store.meta.get("generator") or {"name": "unknown"}
    hashes = {}
    for name in STORE_TABLES:
        path = os.path.join(out_dir, f"{name}.csv")
        data = write_csv(path, store.tables[name])
        m = dataset_manifest(data, len(store.tables[name]), gen, store.meta.get("seed"))
        write_json(manifest_path(path), m)
        hashes[name] = m["content_sha256"]
    return hashes
