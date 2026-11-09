# Contributing

This is a portfolio repository with one maintainer. Issues and pull requests are welcome; this page lists what a change needs before it is merged.

## Environment

- Python 3.12 (`pyproject.toml`). `scripts/setup_env.ps1` (PowerShell) or `bash scripts/setup_env.sh` creates `.venv` in the repository and installs the package with its development tools.
- The stack: NumPy, SciPy, PyArrow, DuckDB, Pydantic, and Matplotlib; the tests use pytest, pytest-repeat, and ruff.
- The tests and the experiments download no data: they generate their synthetic markets. The third-party adapters are tested on offline fixtures; only `scripts/illustrate_french.py` downloads a file ([docs/data.md](docs/data.md)).

## Checks

Run these before opening a pull request. CI runs the same steps (`.github/workflows/ci.yml`):

```bash
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m ruff format --check src tests scripts
.venv/bin/python -m pytest -q
.venv/bin/python -m quant_research_engine experiment run --quick --workers 2
.venv/bin/python -m quant_research_engine audit --canaries tests/canaries --seed 1 --instants 50 --out outputs/audit/audit_table.csv
.venv/bin/python -m quant_research_engine export --config configs/synth/planted.json --seed 1 --out outputs/export/planted
.venv/bin/python -m quant_research_engine validate outputs/export/planted
```

## Results

The results in the README and in [experiments/README.md](experiments/README.md) come from the committed files in `experiments/results/`, each experiment with its manifest. The cross-platform comparison in "Reproducibility across platforms" comes from an independent Linux check and has no committed file.

A change to result-affecting code (the code and configuration that the experiments execute, in `src/quant_research_engine/`, `configs/`, and `experiments/configs/`) must keep the quick results of the evaluation commit. Run `experiment run --quick`, then `python scripts/quick_hashes.py` on the same platform and package versions, and compare with `experiments/quick-hashes-a1a0011.txt`. A change that moves a result reruns the affected experiments once on the committed change and commits the result files with their manifests. `python scripts/make_tables.py` writes the result tables from `experiments/results/`; the README's highlights and tables are written by hand from them.

## Figures and diagrams

`python scripts/plot_results.py` draws the figures in `docs/figures/` from the committed results, including the two README versions. The architecture diagram is drawn from `docs/diagrams/architecture.d2` with [D2](https://d2lang.com) (`bash scripts/render_diagrams.sh` or `powershell -ExecutionPolicy Bypass -File scripts\render_diagrams.ps1`); the committed SVG was drawn with D2 0.9.0. `python scripts/make_assets.py` writes the icon. CI needs none of these.

## Style

- A strategy reads data only through its `DecisionContext`; a row is known at instant `t` only if its `ts_avail <= t` ([docs/architecture.md](docs/architecture.md)).
- Strategy and feature modules pass the import lint (G3). Canaries live in `tests/canaries/`, and `audit` runs every guard on them and on every built-in strategy and feature ([docs/lookahead.md](docs/lookahead.md)).
- Randomness comes from named streams of the configured seed, so the same configuration and seed give the same results.
- Documentation states what was measured, under which assumptions, and names the result file.
