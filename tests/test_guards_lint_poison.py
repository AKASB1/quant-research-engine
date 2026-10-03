"""G3 lint and the poison builder."""

import numpy as np
from audit_helpers import audit_world

from quant_research_engine.guards.lint import lint_source


def test_lint_allows_the_public_strategy_api():
    src = (
        "from __future__ import annotations\nimport math\nimport numpy as np\n"
        "from scipy import stats\nfrom quant_research_engine.context import Strategy\n"
        "from quant_research_engine.features import momentum\n"
        "x = np.zeros(3)\n"
    )
    assert lint_source(src) == []


def test_lint_rejects_file_access_and_dynamic_code():
    bad = {
        "import os\n": "os",
        "from quant_research_engine.store import open_store\n": "store",
        "open('x')\n": "open",
        "eval('1')\n": "eval",
        "exec('1')\n": "exec",
        "__import__('os')\n": "__import__",
        "import importlib\n": "importlib",
        "import scipy.io\n": "scipy.io",
        "from scipy import io\n": "io",
        "import numpy as np\nnp.load('x')\n": "load",
        "import numpy\nnumpy.loadtxt('x')\n": "loadtxt",
        "from numpy import memmap\n": "memmap",
        "import numpy as np\nnp.fromfile('x')\n": "fromfile",
        "import numpy as np\nnp.genfromtxt('x')\n": "genfromtxt",
        "from .helper import x\n": "relative",
        "import pickle\n": "pickle",
    }
    for src, what in bad.items():
        issues = lint_source(src)
        assert issues, (src, what)


def test_poison_keeps_the_known_world_and_changes_the_future():
    st, inst, pois, _ = audit_world(seed=1, n_instants=20)
    p = st.panel()
    flips = 0
    for k in inst:
        t = int(p.ts_event[k])
        w = pois[k]
        assert w.instruments_at(t) == st.instruments_at(t)
        a = st.tables["instruments"]
        b = w.tables["instruments"]
        sa = {str(i) for i, d in zip(a["instrument_id"], a["ts_delist"], strict=True) if d == -1}
        sb = {str(i) for i, d in zip(b["instrument_id"], b["ts_delist"], strict=True) if d == -1}
        flips += sa != sb
        assert w.meta["seed"] != st.meta["seed"]
        # bars after t differ, bars up to t do not
        for iid in st.instruments_at(t)[:3]:
            ra = st.bars([iid], 10**6, knowledge=t)[iid]
            rb = w.bars([iid], 10**6, knowledge=t)[iid]
            assert np.array_equal(ra["close"], rb["close"])
        # the last vintage of some known fund_x period differs
        diff = 0
        for iid in st.instruments_at(t):
            sa_ = st.series(f"{iid}.fund_x", knowledge=10**18)
            sb_ = w.series(f"{iid}.fund_x", knowledge=10**18)
            known = st.series(f"{iid}.fund_x", knowledge=t)["ts_event"].tolist()
            da = dict(zip(sa_["ts_event"].tolist(), sa_["value"].tolist(), strict=True))
            db = dict(zip(sb_["ts_event"].tolist(), sb_["value"].tolist(), strict=True))
            diff += sum(da.get(e) != db.get(e) for e in known)
        assert diff > 0
    assert flips >= len(inst) // 2


def test_every_builtin_strategy_and_feature_module_passes_the_lint():
    import os

    import quant_research_engine.features as feats
    import quant_research_engine.strategies as strats
    from quant_research_engine.guards.lint import lint_file

    for pkg in (feats, strats):
        d = os.path.dirname(pkg.__file__)
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                assert lint_file(os.path.join(d, f)) == [], f
