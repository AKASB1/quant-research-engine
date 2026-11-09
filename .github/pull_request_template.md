## What this change does

## Checklist

- [ ] `ruff check src tests scripts` and `ruff format --check src tests scripts` are clean
- [ ] `python -m pytest -q` passes
- [ ] `python -m quant_research_engine experiment run --quick --workers 2` runs
- [ ] Tests added or updated for the changed behavior
- [ ] If the change touches result-affecting code: the quick fingerprint still matches, or the affected experiments were rerun and their result files and manifests committed
- [ ] Documentation updated; every new number names its result file
