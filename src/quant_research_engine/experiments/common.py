"""Shared helpers of the experiments: deterministic CSV output, manifests, and statistics
(Student-t intervals, Wilson intervals, paired differences). Adapted from the owner's
rollout-engine (MIT): ``bench/stats.py`` and ``bench/output.py``."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys

import numpy as np
from scipy.stats import t as student_t

from quant_research_engine import __version__
from quant_research_engine._threads import thread_settings
from quant_research_engine.data.manifest import config_hash, sha256_file, write_json

# ------------------------------------------------------------------ statistics


def mean_ci(xs, level: float = 0.95) -> tuple[float, float, int]:
    """Mean and half-width of the Student-t interval; NaN entries are dropped."""
    x = np.asarray(
        [v for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))],
        dtype=np.float64,
    )
    n = len(x)
    if n == 0:
        return math.nan, math.nan, 0
    m = float(math.fsum(x.tolist()) / n)
    if n < 2:
        return m, math.nan, n
    sd = math.sqrt(math.fsum(((x - m) ** 2).tolist()) / (n - 1))
    return m, float(student_t.ppf(0.5 + level / 2, n - 1)) * sd / math.sqrt(n), n


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float, float]:
    """Share and Wilson interval (lo, hi)."""
    if n == 0:
        return math.nan, math.nan, math.nan
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + level / 2))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def paired(a: dict, b: dict) -> tuple[float, float, int]:
    """Mean and t half-width of b - a over common keys."""
    keys = sorted(set(a) & set(b))
    return mean_ci([b[k] - a[k] for k in keys])


# ------------------------------------------------------------------ output


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool | np.bool_):
        return "true" if v else "false"
    if isinstance(v, int | np.integer):
        return str(int(v))
    if isinstance(v, float | np.floating):
        v = float(v)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v + 0.0)
    return str(v)


def write_rows(path: str, rows: list[dict], first: list[str]) -> bytes:
    cols = list(first)
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    lines = [",".join(cols)] + [",".join(fmt(r.get(c)) for c in cols) for r in rows]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        lines = [x for x in f.read().split("\n") if x]
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(","), strict=True)) for ln in lines[1:]]


def deterministic_bytes(path: str) -> bytes:
    """File bytes without the columns whose names start with ``wall_``."""
    rows = open(path, encoding="utf-8").read().split("\n")
    head = rows[0].split(",")
    keep = [i for i, c in enumerate(head) if not c.startswith("wall_")]
    return "\n".join(",".join(r.split(",")[i] for i in keep) if r else r for r in rows).encode(
        "utf-8"
    )


# ------------------------------------------------------------------ manifests


def git_state(cwd: str | None = None) -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20, cwd=cwd
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=20,
                cwd=cwd,
            ).stdout.strip()
        )
        return {"commit": sha or "unknown", "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            k = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return round(m.ullTotalPhys / 1e9, 1)
        return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) * 1024 / 1e9, 1)
    except OSError:
        pass
    return None


def cpu_load_percent() -> float | None:
    """Machine CPU load in percent (Windows: Win32_Processor LoadPercentage; else load average)."""
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average",  # noqa: E501
                ],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            return float(out) if out else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    try:
        return round(100.0 * os.getloadavg()[0] / (os.cpu_count() or 1), 1)
    except OSError:
        return None


def software() -> dict:
    import duckdb
    import pyarrow
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "polars": "not used",
        "pyarrow": pyarrow.__version__,
        "duckdb": duckdb.__version__,
        "quant_research_engine": __version__,
    }


def hardware() -> dict:
    return {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
    }


def manifest(
    exp_id: str,
    mode: str,
    cfg: dict,
    seeds: list[int],
    workers: int,
    files: dict[str, str],
    extra: dict | None = None,
    load: dict | None = None,
) -> dict:
    m = {
        "experiment": exp_id,
        "mode": mode,
        "label": "QUICK" if mode == "quick" else "full",
        "config_hash": config_hash(cfg),
        "config": cfg,
        "seeds": {
            "first": min(seeds) if seeds else None,
            "last": max(seeds) if seeds else None,
            "count": len(seeds),
        },
        "git": git_state(),
        "software": software(),
        "hardware": hardware(),
        "workers": workers,
        "threads": thread_settings(),
        "machine_load": load or {},
        "files": {name: sha256_file(path) for name, path in sorted(files.items())},
        "data": "synthetic, assumed parameters",
    }
    m.update(extra or {})
    return m


def write_manifest(path: str, m: dict) -> None:
    write_json(path, m)
