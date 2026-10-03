import os

for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("POLARS_MAX_THREADS", "1"),
    ("PYTHONUTF8", "1"),
):
    os.environ.setdefault(_k, _v)

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "data", "golden")
CANARIES = os.path.join(HERE, "canaries")
