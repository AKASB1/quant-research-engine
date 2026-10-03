"""Trial registry: every configuration evaluated in a study is a trial.

A study writes ``trials.csv`` with the header ``study_id,trial_id,config_hash,sr_bar,t_bars,
skew,kurt`` (per-bar Sharpe of each trial), row by row as trials are evaluated. N of the DSR is
the number of registered trials; V is the sample variance (ddof = 1) of ``sr_bar`` across them.
"""

from __future__ import annotations

import os

import numpy as np

from quant_research_engine.data.manifest import config_hash
from quant_research_engine.inference.sharpe import dsr, expected_max_sharpe
from quant_research_engine.metrics import kurt, sharpe, skew

HEADER = "study_id,trial_id,config_hash,sr_bar,t_bars,skew,kurt"


class TrialRegistry:
    def __init__(self, study_id: str, path: str | None = None):
        self.study_id = study_id
        self.path = path
        self.rows: list[tuple] = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(HEADER + "\n")

    def add(self, trial_id: str, config, returns) -> tuple:
        r = np.asarray(returns, dtype=np.float64)
        row = (
            self.study_id,
            str(trial_id),
            config_hash(config),
            sharpe(r),
            len(r),
            skew(r),
            kurt(r),
        )
        self.rows.append(row)
        if self.path:
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(
                    ",".join(
                        [
                            row[0],
                            row[1],
                            row[2],
                            repr(float(row[3])),
                            str(row[4]),
                            repr(float(row[5])),
                            repr(float(row[6])),
                        ]
                    )
                    + "\n"
                )
        return row

    @property
    def n(self) -> int:
        return len(self.rows)

    def var_sr(self) -> float:
        srs = np.asarray([r[3] for r in self.rows], dtype=np.float64)
        srs = srs[
            np.isfinite(srs)
        ]  # a zero-variance trial has no Sharpe ratio; it stays a trial in N
        return float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0

    def select(self) -> int:
        """Index of the trial with the highest per-bar Sharpe ratio (ties: lowest index; trials
        without a finite Sharpe ratio are never selected)."""
        srs = np.asarray([r[3] for r in self.rows], dtype=np.float64)
        return int(np.argmax(np.where(np.isfinite(srs), srs, -np.inf)))

    @property
    def degenerate(self) -> int:
        return int(sum(1 for r in self.rows if not np.isfinite(r[3])))

    def sr0(self) -> float:
        return expected_max_sharpe(self.n, self.var_sr())

    def dsr(self, trial_index: int) -> float:
        _, _, _, sr, t, sk, ku = self.rows[trial_index]
        return dsr(sr, t, sk, ku, self.n, self.var_sr())
