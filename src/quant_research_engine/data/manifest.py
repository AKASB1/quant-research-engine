"""Dataset manifests (``<name>.manifest.json``), canonical JSON, and configuration hashes.

A configuration hash is the SHA-256 (lowercase hexadecimal) of the canonical JSON of a
configuration: UTF-8, sorted keys, no whitespace, integers without a decimal point, floats in
their shortest round-trip form, and every field equal to its default omitted (so a field added
later with a default value changes no existing hash).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

from quant_research_engine import QC_VERSION
from quant_research_engine.data.schemas import SCHEMA_VERSION


def _check_finite(obj: Any) -> None:
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("non-finite float in canonical JSON")
    if isinstance(obj, dict):
        for v in obj.values():
            _check_finite(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _check_finite(v)


def canonical_json(obj: Any) -> str:
    _check_finite(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(obj: Any) -> str:
    """SHA-256 of the canonical JSON; pydantic models are dumped without their defaults."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json", exclude_defaults=True)
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj: Any) -> None:
    """UTF-8, sorted keys, two-space indentation, LF, trailing newline."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dataset_manifest(
    data: bytes, row_count: int, generator: dict, seed: int | None
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "qc_version": QC_VERSION,
        "generator": generator,
        "seed": seed,
        "row_count": row_count,
        "content_sha256": sha256_bytes(data),
    }


def manifest_path(csv_path: str) -> str:
    assert csv_path.endswith(".csv")
    return csv_path[:-4] + ".manifest.json"


def check_manifest(csv_path: str, data: bytes, row_count: int) -> dict:
    """Read the manifest beside a CSV and check versions, row count, and hash."""
    mp = manifest_path(csv_path)
    if not os.path.exists(mp):
        raise ValueError(f"missing manifest {os.path.basename(mp)}")
    m = read_json(mp)
    if m.get("schema_version") != SCHEMA_VERSION or m.get("qc_version") != QC_VERSION:
        raise ValueError(f"{os.path.basename(mp)}: unsupported schema or contract version")
    if m.get("row_count") != row_count:
        raise ValueError(f"{os.path.basename(mp)}: row_count {m.get('row_count')} != {row_count}")
    if m.get("content_sha256") != sha256_bytes(data):
        raise ValueError(f"{os.path.basename(mp)}: content_sha256 does not match the file")
    return m
