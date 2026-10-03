#!/usr/bin/env bash
# Create .venv in the repository and install the package with its dev tools.
# Usage (repository root): bash scripts/setup_env.sh
# Interpreter: $PYTHON if set, else python3 (or python) on PATH (Python 3.12 or newer).
# Then activate it: source .venv/bin/activate   (Git Bash on Windows: source .venv/Scripts/activate)
set -euo pipefail
export PYTHONUTF8=1
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if python3 -c "import sys" >/dev/null 2>&1; then PY=python3; else PY=python; fi
fi
"$PY" -m venv .venv
if [ -x .venv/bin/python ]; then VPY=.venv/bin/python; else VPY=.venv/Scripts/python.exe; fi
"$VPY" -m pip install --disable-pip-version-check -q -e ".[dev]"
"$VPY" -c "import quant_research_engine as q, numpy, scipy, pyarrow, duckdb; print('ready:', q.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'pyarrow', pyarrow.__version__, 'duckdb', duckdb.__version__)"
