# Create .venv in the repository and install the package with its dev tools.
# Usage (repository root): powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# Interpreter: $env:PYTHON if set, else python on PATH (Python 3.12 or newer).
# Then use .venv\Scripts\python.exe, or activate the environment: .venv\Scripts\Activate.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$py = "python"
if ($env:PYTHON) { $py = $env:PYTHON }
& $py -m venv .venv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -q -e ".[dev]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -c "import quant_research_engine as q, numpy, scipy, pyarrow, duckdb; print('ready:', q.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'pyarrow', pyarrow.__version__, 'duckdb', duckdb.__version__)"
exit $LASTEXITCODE
