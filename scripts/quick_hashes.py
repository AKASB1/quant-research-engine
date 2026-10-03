"""SHA-256 of every quick result file of E1 to E5, without the wall_ columns (CSV) or wall_ keys
(JSON) and without the manifests: the fingerprint that later commits must reproduce on the same
platform and package versions. On another platform floats can differ in the last digit and most
files hash differently (an independent Linux check matched 8 of 25), so compare numerically there.

python scripts/quick_hashes.py [--root experiments/outputs/quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os


def det_bytes(path: str) -> bytes:
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)

        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if not k.startswith("wall_")}
            if isinstance(o, list):
                return [strip(v) for v in o]
            return o

        return json.dumps(strip(obj), sort_keys=True).encode("utf-8")
    with open(path, encoding="utf-8") as f:
        rows = f.read().split("\n")
    head = rows[0].split(",")
    keep = [i for i, c in enumerate(head) if not c.startswith("wall_")]
    return "\n".join(",".join(r.split(",")[i] for i in keep) if r else r for r in rows).encode(
        "utf-8"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("experiments", "outputs", "quick"))
    a = ap.parse_args(argv)
    for exp in ("e1", "e2", "e3", "e4", "e5"):
        d = os.path.join(a.root, exp)
        if not os.path.isdir(d):
            print(f"{exp}: missing")
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p) and f.endswith((".csv", ".json")) and f != "manifest.json":
                print(f"{exp}/{f} {hashlib.sha256(det_bytes(p)).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
