# Experiments

Every number produced here is **simulated or backtested on synthetic markets with assumed parameters** ([../docs/data.md](../docs/data.md)). The experiments evaluate this engine's own guarantees (are the engines right, do the guards catch what they should, how well do the inference tools work where the truth is known), not strategies; nothing here says anything about real markets.

## How to reproduce

From the repository root, with the environment of the README:

```text
python -m quant_research_engine experiment run --workers 2            # every experiment (E1-E5, e5b, t2_capacity, t2_trials) -> experiments/results/<id>/
python -m quant_research_engine experiment run --quick --workers 2    # development seeds, small sizes -> experiments/outputs/quick/ (ignored)
python -m quant_research_engine experiment run --only e3 --workers 2  # one experiment
python scripts/plot_results.py                                        # figures from experiments/results -> docs/figures
```

Configurations are committed and frozen: `experiments/configs/e1.json` to `e5.json` (the full evaluation), `e5b.json`, `t2_capacity.json`, and `t2_trials.json` (Tier 2, each fixed before its single run), and `quick.json`; synthetic markets in `configs/synth/`. A seed of experiment `<id>` fixes the whole market through the derived seed `stream(seed, "experiment.<id>")`, so every strategy sees the same data and quick runs of different experiments never share a market. Each results directory of an experiment holds the rows, the aggregates, and `manifest.json` (commit and whether the tree was clean, configuration and its hash, seeds, Python/NumPy/SciPy/PyArrow/DuckDB versions, hardware class, worker count, thread settings, CPU load before and after, wall time). Bulky per-run files (E3's 86,400 trial rows, the per-market `trials.csv`, logs) stay in the ignored `experiments/outputs/`; their SHA-256 is recorded in the manifest. The exception is `results/t2_french/`, the output of a script rather than of the runner: it holds `summary.json` and a short manifest (commit, clean flag, software, hardware).

## Protocol

- Seeds: development, tests, and `--quick` use 1 to 99. Full evaluation: E1 3000-3999; E2 4100-4109 (`small` audit stores), 4000-4009 (`null_e3`, canary Sharpe ratios), 4200-4209 (`planted`, C7); E3 1000-1199 (`null_e3`) and 2000-2199 (`planted`); E4 5000-5019; E5 6000-6049 (`null`) and 7000-7049 (`planted`).
- Defaults: cost model v1 of the contract with its default parameters; `next_open` fills; vectorized runs with fixed-capital sizing (10 million, gross exposure 1; returns over the fixed capital); event-engine runs with equity sizing; `ppy` 252.
- Statistics: means with 95 % Student-t intervals across markets or seeds; proportions with Wilson intervals; paired differences where two things share seeds. Exact ties are named.
- Gross panels (all costs zero) are the calibration of the inference tools: on a null market every trial's true Sharpe ratio is zero. Net panels (base costs) are reported separately and labelled; their Sharpe ratios are not draws from the null that SR0 describes.
- The protocol, the grids, the sizes, the tolerances, and the rule for what the README highlights were fixed before the full evaluation; the README highlights no "best" configuration (headline numbers are fixed in advance: E1 largest differences and violations, E2 the full table, E3 the gross-panel null rates of naive PSR and DSR and the mean PBO, E4 the base case rebalance 5 / cost multiple 1 next to the grid, E5 the bias at h = 5 next to the full table).

## Results of the full evaluation

E1 to E5 ran once, on commit `a1a0011` with a clean tree, with 2 workers and one thread per numerical library. The machine had an Intel Core i9-14900KF (32 logical CPUs, 102.9 GB RAM) and ran Windows 11, Python 3.12.3, and NumPy 2.5.3. It was shared with other jobs.

| Experiment | Wall time | CPU load before / after (%) |
|---|---|---|
| E1 | 89 s | 6 / 36 |
| E2 | 770 s | 15 / 18 |
| E3 | 773 s | 22 / 37 |
| E4 | 122 s | 31 / 44 |
| E5 | 914 s | 32 / 45 |

These figures come from the manifests. `scripts/make_tables.py` generates the tables below from `experiments/results/`. Every number is simulated on synthetic markets with assumed parameters and holds only in this simulation, under these assumptions.

## Reproducibility across platforms

Results are byte-identical for the same configuration and seed on one platform with the same package versions. The full evaluation and every later quick check ran on the build machine:
- Windows 11, Intel Core i9-14900KF (no AVX-512);
- Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, PyArrow 25.0.1, DuckDB 1.5.6.

There the E1 to E5 quick results of the evaluation commit were reproduced byte for byte (`experiments/quick-hashes-a1a0011.txt`).

Across platforms the results agree numerically, not byte for byte. The facts in this section come from an independent check that ran the quick experiments of the uploaded package of commit `894393f` on Linux (x86-64 with AVX-512, Python 3.12.3, the same package versions, one BLAS thread). This repository cannot reproduce that comparison without a Linux run.

- **Files that matched.** 8 of the 25 fingerprinted quick files matched byte for byte: `e1/aggregates.json`, `e1/rows.csv`, `e2/aggregates.json`, `e2/audit_instants.csv`, `e2/audit_table.csv`, `e2/detection.csv`, `e3/pbo_hist.csv`, and `e3/trial_grid.csv`. The other 17 differed.
- **What differed.** A cell-by-cell comparison with the build machine's quick files, `wall_` columns left out, found no differing value other than floats and two SHA-256 fields that hash differing float bytes: `trials_sha256` in `e3/selected.csv` and the `trial_rows.csv` entry in `e3/outputs_index.json`. No selected parameter, count, flag, or decision changed.
- **Size of the differences outside E5 and E5b.** Floats differed by less than 3e-11 relative; the largest was the E3 deflated Sharpe ratio, at 2.2e-11. The exception is the `max_identity_residual` columns, which are round-off: below 1e-11 on both platforms.
- **Size of the differences in E5 and E5b.** The rank-correlation scores differed by at most 5.6e-6 in absolute value (`cv_score` and `bias` in `e5/rows.csv`, `cv_spearman` and `bias_spearman` in `e5b/rows.csv`; the held-out scores by at most 4.6e-7). For scale, the scores are about 1e-2 and their standard deviations across markets about 0.01 to 0.03.
- **One cause, tested.** On a CPU with AVX-512, NumPy 2.5.3 takes an AVX-512 path for `power` that is one unit in the last place below the scalar `pow` for some arguments. Four of the 250 `sigma_bar` weights of `store/liquidity.py` are affected, and through them five `sigma_bar` cells of the tiny fixture. With NumPy's AVX-512 paths switched off, the fixture matched.
- **The other cause, not isolated.** With those paths switched off, 9 of the 25 files matched (one more), and the E5 differences stayed (up to 5.9e-6). They come presumably from the last digit of `log1p`, `expm1`, and `exp` differing between the Windows runtime and glibc, which would move some nearest-neighbour sets (k = 25) at their boundary.

Consequences:
- Compare results across platforms numerically. The tests that compare generated files with committed ones do so cell by cell, with a float tolerance (`tests/helpers.py`).
- The statements about E5 and E5b in these documents rest on differences of at least 1e-4. One E5 case closer to zero than that is called borderline in its section.

## E1 Engine and accounting audit

1000 scenarios (seeds 3000 to 3999, stream `e1.scenario`). Each scenario has:
- 2 to 12 instruments and 50 to 300 bars;
- random target-share schedules with sign changes, decisions every 1 to 5 bars, and `gtc` orders;
- splits with ratios 2 or 3 (some between a decision and its fill), dividends, delistings, late listings, zero-volume bars, lot sizes, shorts on or off, and fill delays of 1 to 3 bars.

Both engines run every scenario with logs written. The accounting identity is asserted in memory on every bar, and the independent validator re-derives fills, costs, and accounting from the logs and the store (`results/e1/aggregates.json`, `rows.csv`).

| Measure | Value |
|---|---|
| scenarios (compared / excluded: cap binding or rejected order) | 1000 (956 / 44) |
| fill mismatches between the engines | 0 |
| largest relative difference of fill quantity, reference price, price, spread, impact, commission | 0, 0, 0, 0, 0, 0 |
| largest relative difference of equity | 2.1e-14 |
| largest accounting-identity residual (event / vector), relative to equity | 2.1e-15 / 1.8e-15 |
| independent validator failures (event logs / vector logs) | 0 / 0 |
| largest relative error found by the validator | 2.0e-14 |
| scenarios with splits / dividends / delistings / late listings / zero-volume bars / lot sizes | 981 / 996 / 861 / 862 / 999 / 996 |
| fill delay 1 / 2 / 3 bars; shorts on | 307 / 323 / 370; 614 |
| violations | 0 |

- **What counts as a violation.** A violation is a fill mismatch or a validator failure, so zero is the expectation and any violation would be a bug.
- **Excluded scenarios.** The engines are meant to agree only where neither the participation cap nor a rejection applies, so the 44 scenarios where the cap bound or an order was rejected are excluded from the comparison. Their event-engine logs were still validated, and the identity residuals above cover all 1000 scenarios.
- **Throughput.** Measured inside E1 on these small scenarios, the engines processed 1.26e5 (event) and 1.64e6 (vectorized) instrument-bars per second. These are wall-clock figures: total instrument-bars over total run time of the compared scenarios, per-run setup included, 2 workers, load 6 to 36 %. `scripts/microbench.py` measures larger runs, with CPU time next to wall time.

## E2 Leakage audit

**Audit table.** 22 subjects (11 canaries, 6 built-in strategies, 5 one-feature strategies) ran against every guard on 10 `small` stores (seeds 4100 to 4109), 10 sampled decision instants per store and subject (`results/e2/audit_table.csv`; per-instant flags in `audit_instants.csv`).

| Subject | Kind | Guard named | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G2 instants caught |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 peek_next_close | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 100 / 100 |
| C1b peek_by_path | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 100 / 100 |
| C1g peek_global_handle | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 100 / 100 |
| C1s package_state_handle | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 100 / 100 |
| C2 full_sample_mean | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 99 / 100 |
| C3 latest_vintage | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 99 / 100 |
| C4 survivors_only | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 100 / 100 |
| C5 vendor_adjusted_close | strategy | G2 | passed | **caught** | **caught** | passed | passed | passed | passed | 99 / 100 |
| C6 early_release_feature | feature | G4 | passed | passed | passed | **caught** | passed | passed | passed | 0 / 100 |
| C7 same_bar_fill | run | G5 | passed | passed | passed | passed | **caught** | passed | passed | 0 / 100 |
| C8 overlapping_label_cv | cv | G6 | passed | passed | **caught** | passed | passed | **caught** | passed | 0 / 100 |
| F_low_vol feature_low_vol | feature | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| F_momentum feature_momentum | feature | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| F_reversal feature_reversal | feature | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| F_signal_x_ewma feature_signal_x_ewma | feature | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| F_volume_trend feature_volume_trend | feature | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_low_vol_ls low_vol_ls | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_momentum_ls momentum_ls | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_reversal_ls reversal_ls | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_signal_x_ls signal_x_ls | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_signal_x_voltarget signal_x_voltarget | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |
| S_volume_trend_topk volume_trend_topk | strategy | - | passed | passed | passed | passed | passed | passed | passed | 0 / 100 |

**Detection probability.** The probability that the replay audit catches a canary when m instants are sampled, from 100 draws without replacement of m out of the 100 audited instants (`detection.csv`). C6, C7, and C8 are not caught by the replay at any instant, so they are not in the curve. C5 was missed at 1 of its 100 instants, yet it was caught in all 100 single-instant draws: the instant that missed it was never drawn.

| Canary | m = 1 | m = 2 | m = 5 | m = 10 | m = 20 | m = 50 |
|---|---|---|---|---|---|---|
| C1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C1b | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C1g | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C1s | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C2 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C3 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Sharpe ratios.** C1, C1b, and C2 were compared with the honest one-feature momentum(21, 5) rule (a decision every bar; the same weighting as C1 and C1b, while C2 holds all of its weight in one instrument) on 10 `null_e3` markets, 60 × 1260 (seeds 4000 to 4009). On these markets the honest rule's true Sharpe ratio is zero.
- Runs use the event engine, equity sizing, 10 million initial cash, and a warmup of 25 bars.
- The gross panel (all costs zero) is the comparison. The net panel (base costs) is labelled.
- C7 (UNSAFE same-close fills) was compared with the same strategy, `signal_x_ls`, at `next_open` on 10 `planted` markets, 60 × 1260 (seeds 4200 to 4209, warmup 40).
- The paired difference is taken against the honest rule or against `signal_x_ls` at `next_open`, on the same markets (`sharpe_aggregates.csv`, `sharpe_rows.csv`).

| Part | Panel | Subject | Annualized Sharpe (mean ± 95 % t) | Paired difference vs reference | Markets |
|---|---|---|---|---|---|
| sharpe | gross | C1 | 7.89 ± 0.90 | 7.89 ± 0.86 | 10 |
| sharpe | gross | C1b | 7.89 ± 0.90 | 7.89 ± 0.86 | 10 |
| sharpe | gross | C2 | 0.86 ± 0.20 | 0.86 ± 0.23 | 10 |
| sharpe | gross | honest | 0.00 ± 0.23 | 0.00 ± 0.00 | 10 |
| sharpe | net | C1 | 5.23 ± 0.95 | 6.65 ± 0.89 | 10 |
| sharpe | net | C1b | 5.23 ± 0.95 | 6.65 ± 0.89 | 10 |
| sharpe | net | C2 | 0.86 ± 0.20 | 2.28 ± 0.20 | 10 |
| sharpe | net | honest | -1.42 ± 0.17 | 0.00 ± 0.00 | 10 |
| c7 | gross | UNSAFE C7 | 1.20 ± 0.27 | -0.04 ± 0.11 | 10 |
| c7 | gross | signal_x_ls | 1.24 ± 0.27 | 0.00 ± 0.00 | 10 |
| c7 | net | UNSAFE C7 | -0.74 ± 0.30 | -0.03 ± 0.10 | 10 |
| c7 | net | signal_x_ls | -0.71 ± 0.29 | 0.00 ± 0.00 | 10 |

| Panel | Per-market ratio of Sharpe ratios, C7 / next_open (mean ± 95 % t) | Ratio of mean Sharpe ratios |
|---|---|---|
| gross | 0.97 ± 0.09 | 0.97 |
| net | 1.02 ± 0.16 | 1.04 |

- **C1 and C1b tie exactly** in every market and both panels.
- **The leaks earned more than the honest rule.** At zero costs, the three leaks earned more than the honest rule, whose interval contains its true value of zero.
- **C7 changed nothing measurable.** The UNSAFE same-close fill did not change the Sharpe ratio of `signal_x_ls` measurably: both ratio intervals contain 1. With persistence 0.9 and an overnight share of 0.25, the expected gap is about 2.5 % of the planted per-bar mean, which these 10 markets cannot resolve. This is an expectation, not a measured cause.
- **C8** is measured by the shuffled K-fold rows of E5 on null markets: a bias of 0.260 ± 0.005 at h = 5 and 0.444 ± 0.010 at h = 21.

## E3 Calibration of the inference tools

The trial grid has 108 trials (`results/e3/trial_grid.csv`):
- **Signals:** `momentum` (lookbacks 10, 21, 42, 63, 126, 252, skip 5), `reversal` (1, 3, 5, 10, 21), `low_vol` (21, 63, 126), and `signal_x_ewma` (half-lives 1, 3, 10, 30).
- **Rules:** `long_short_quantile` with (q 0.2, equal weights) and with (q 0.3, signal weights).
- **Rebalancing:** every 1, 5, or 21 bars.

Markets and engine:
- 200 `null_e3` markets (seeds 1000 to 1199) and 200 `planted` markets (seeds 2000 to 2199), each 60 × 1260.
- The vectorized engine runs with fixed capital of 10 million and gross exposure 1. Every trial starts at bar 259, which gives 1000 aligned rows of returns.
- The CSCV uses S = 16. The bootstrap uses p = 0.1 and 1000 draws.

The gross panel (all costs zero) is the calibration. The net panel (base costs) is labelled. Its SR0 rests on the variance of the 108 net Sharpe ratios, which differ mainly by turnover, so the net SR0 and the ratios built on it are not a null comparison. They show only what gets selected and what the PBO says (`aggregates.csv`, `selected.csv`, `pbo_hist.csv`, `lo.csv`).

| Market / panel | Selected trial's per-bar Sharpe | SR0 (N = 108) | Selected / SR0 | Naive PSR(0) > 0.95 | DSR > 0.95 | Mean PBO | Selected trial uses signal_x |
|---|---|---|---|---|---|---|---|
| null / gross | 0.0692 ± 0.0024 | 0.0769 ± 0.0024 | 0.91 ± 0.03 | 86.0 % [80.5, 90.1] | 0.0 % [0.0, 1.9] | 0.48 ± 0.03 | 21.5 % [16.4, 27.7] |
| null / net | 0.0328 ± 0.0030 | 0.3291 ± 0.0072 | 0.10 ± 0.01 | 18.5 % [13.7, 24.5] | 0.0 % [0.0, 1.9] | 0.21 ± 0.02 | 19.5 % [14.6, 25.5] |
| planted / gross | 0.1118 ± 0.0036 | 0.1027 ± 0.0028 | 1.10 ± 0.03 | 100.0 % [98.1, 100.0] | 0.5 % [0.1, 2.8] | 0.20 ± 0.02 | 88.5 % [83.3, 92.2] |
| planted / net | 0.0561 ± 0.0034 | 0.3142 ± 0.0071 | 0.18 ± 0.01 | 52.0 % [45.1, 58.8] | 0.0 % [0.0, 1.9] | 0.18 ± 0.02 | 69.5 % [62.8, 75.5] |

| Trial (null markets, gross) | Coverage of the 95 % bootstrap interval of the Sharpe ratio |
|---|---|
| selected (best of 108 on the same data) | 37.5 % [31.1, 44.4] |
| fixed in advance (momentum 21/5, q 0.2 equal, every 5 bars) | 95.0 % [91.0, 97.3] |

| AR(1) coefficient | True annualized Sharpe | Naive sqrt(252) x SR | Lo-adjusted (L = 10) |
|---|---|---|---|
| 0.0 | 0.79 | 0.81 ± 0.04 | 0.82 ± 0.05 |
| 0.1 | 0.72 | 0.78 ± 0.05 | 0.72 ± 0.05 |
| 0.2 | 0.65 | 0.82 ± 0.05 | 0.68 ± 0.04 |
| 0.3 | 0.58 | 0.84 ± 0.06 | 0.63 ± 0.05 |

- **Null, gross.** The naive PSR(0) flagged the selected trial in 86.0 % [80.5, 90.1] of null markets. The DSR flagged none of them (0.0 % [0.0, 1.9]).
- **Correlated trials.** The trials are strongly correlated, so the effective number of trials is below 108. The selected Sharpe ratio averaged 0.91 ± 0.03 of SR0(N = 108). A larger SR0 lowers the DSR, so on null markets the DSR rejected less often, and on planted markets it had less power, than with an SR0 that matched the observed maximum.
- **Planted, gross.** The selected trial used `signal_x` in 88.5 % of the markets, but the DSR exceeded 0.95 in 1 market of 200 (0.5 % [0.1, 2.8]).
- **PBO.** The mean PBO was 0.48 ± 0.03 on null markets and 0.20 ± 0.02 on planted markets (gross).
- **Bootstrap coverage.** The interval of the selected trial covered zero in 37.5 % of null markets. For the trial fixed in advance (momentum 21/5, q 0.2 equal, every 5 bars) it covered zero in 95.0 %.
- **Lo adjustment.** At AR(1) coefficients 0.1, 0.2, and 0.3 the naive annualization overstated the true Sharpe ratio (0.78 vs 0.72, 0.82 vs 0.65, 0.84 vs 0.58), and its intervals exclude the true value. The Lo-adjusted intervals contain the true value at every coefficient. At coefficient 0 the naive and adjusted values (0.81, 0.82) both sit near the true 0.79.

## E4 Signal recovery, costs, and lag

20 seeds (5000 to 5019), 100 instruments by 2520 bars.

**(a) Recovery.** `planted_clean` has no factors, equal idiosyncratic volatility, no regime, and no events; `planted` has factors, regimes, and events.
- The information coefficients use the correlation with the standardized idiosyncratic return. The rows marked raw use the raw next-bar return instead, which is a different estimator and is labelled.
- The fundamental-law row is a bound from the generator's panels (close-to-close holding, which no engine allows).
- The event-engine row (next open, equity sizing, a decision every bar, zero costs, participation cap off) uses the latent signal and is an oracle.
- Expected value of the engine row: the bound × (1 − 0.25 × (1 − 0.9)) (`results/e4/aggregates.json`).

| Quantity (100 instruments x 2520 bars, 20 seeds) | Measured | Expected |
|---|---|---|
| IC of the latent signal (oracle), planted_clean | 0.0196 ± 0.0011 | 0.0200 |
| IC of signal_x, planted_clean | 0.0138 ± 0.0011 | 0.0141 |
| per-bar Sharpe, fundamental-law weights close to close (a bound) | 0.1962 (se 0.0045) | 0.2000 |
| per-bar Sharpe, same weights through the event engine, next_open, zero costs (oracle) | 0.1901 (se 0.0045) | 0.1950 (bound x (1 - g (1 - phi))) |
| Sharpe of signal_x weights / Sharpe of latent weights (close to close) | 0.704 | 0.707 |
| IC of the latent signal (oracle), planted | 0.0204 ± 0.0011 | – |
| IC of signal_x, planted | 0.0150 ± 0.0009 | – |
| IC of the latent signal vs the raw next-bar return, planted (labelled: not the estimator above) | 0.0165 ± 0.0010 | – |
| IC of signal_x vs the raw next-bar return, planted (labelled) | 0.0121 ± 0.0010 | – |

**(b) Costs.** The strategy is `signal_x_ewma(3)` with `long_short_quantile(0.2, equal, gross 1)` on `planted` markets, using the vectorized engine and fixed capital of 10 million. Returns and turnover are measured against the fixed capital (DECISIONS: QC-CHANGE).
- The cost multiple scales every cost rate; the cash rate and the participation cap stay unchanged.
- The break-even multiple comes from linear interpolation of the mean net Sharpe curve (`costs.csv`).
- The base case, fixed in advance, is rebalancing every 5 bars at cost multiple 1: net 0.09 ± 0.22, gross 1.42 ± 0.19.

| Rebalance | Cost multiple | Net Sharpe (ann.) | Gross Sharpe (ann.) | Turnover per bar | Cost drag per year | Break-even multiple |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 1.76 ± 0.17 | 1.76 ± 0.17 | 0.351 | 0.000 | 0.66 |
| 1 | 0.5 | 0.42 ± 0.21 | 1.76 ± 0.17 | 0.351 | 0.100 | 0.66 |
| 1 | 1.0 | -0.92 ± 0.30 | 1.76 ± 0.17 | 0.351 | 0.200 | 0.66 |
| 1 | 2.0 | -3.62 ± 0.51 | 1.76 ± 0.17 | 0.351 | 0.407 | 0.66 |
| 1 | 4.0 | -9.18 ± 0.80 | 1.76 ± 0.17 | 0.351 | 0.882 | 0.66 |
| 5 | 0.0 | 1.42 ± 0.19 | 1.42 ± 0.19 | 0.169 | 0.000 | 1.07 |
| 5 | 0.5 | 0.75 ± 0.20 | 1.42 ± 0.19 | 0.169 | 0.050 | 1.07 |
| 5 | 1.0 | 0.09 ± 0.22 | 1.42 ± 0.19 | 0.169 | 0.099 | 1.07 |
| 5 | 2.0 | -1.17 ± 0.28 | 1.42 ± 0.19 | 0.169 | 0.199 | 1.07 |
| 5 | 4.0 | -3.32 ± 0.35 | 1.42 ± 0.19 | 0.169 | 0.416 | 1.07 |
| 21 | 0.0 | 0.73 ± 0.12 | 0.73 ± 0.12 | 0.071 | 0.000 | 1.19 |
| 21 | 0.5 | 0.42 ± 0.13 | 0.73 ± 0.12 | 0.071 | 0.023 | 1.19 |
| 21 | 1.0 | 0.11 ± 0.14 | 0.73 ± 0.12 | 0.071 | 0.046 | 1.19 |
| 21 | 2.0 | -0.47 ± 0.16 | 0.73 ± 0.12 | 0.071 | 0.092 | 1.19 |
| 21 | 4.0 | -1.42 ± 0.18 | 0.73 ± 0.12 | 0.071 | 0.186 | 1.19 |

**(c) Lag.** The same strategy rebalanced every 5 bars at base costs on `planted` markets with persistence 0.5, 0.9, and 0.97, filled at `next_open` with delays of 1, 2, and 3 bars, and at `next_close` (`lag.csv`).

| Persistence | Fill model | Fill delay | Net Sharpe (ann.) | Gross Sharpe (ann.) |
|---|---|---|---|---|
| 0.5 | next_close | 1 | -1.89 ± 0.22 | 0.12 ± 0.15 |
| 0.5 | next_open | 1 | -1.68 ± 0.24 | 0.35 ± 0.16 |
| 0.5 | next_open | 2 | -1.91 ± 0.22 | 0.10 ± 0.15 |
| 0.5 | next_open | 3 | -1.97 ± 0.21 | 0.04 ± 0.14 |
| 0.9 | next_close | 1 | -0.04 ± 0.21 | 1.29 ± 0.18 |
| 0.9 | next_open | 1 | 0.09 ± 0.22 | 1.42 ± 0.19 |
| 0.9 | next_open | 2 | -0.08 ± 0.22 | 1.26 ± 0.19 |
| 0.9 | next_open | 3 | -0.18 ± 0.18 | 1.15 ± 0.15 |
| 0.97 | next_close | 1 | 0.76 ± 0.17 | 1.75 ± 0.14 |
| 0.97 | next_open | 1 | 0.82 ± 0.18 | 1.81 ± 0.16 |
| 0.97 | next_open | 2 | 0.75 ± 0.18 | 1.74 ± 0.15 |
| 0.97 | next_open | 3 | 0.68 ± 0.16 | 1.67 ± 0.13 |

- **Fill delay.** By paired differences over the 20 seeds, going from delay 1 to delay 2 lowered the net and gross Sharpe ratios at every persistence level, for example net 0.235 ± 0.069 at persistence 0.5. Going from delay 2 to delay 3 was resolved only at persistence 0.9 (net 0.108 ± 0.078); at 0.5 and 0.97 its interval contains zero. The drop from delay 1 to delay 3 was smaller at 0.97 than at 0.9 (difference 0.132 ± 0.089 net); 0.5 and 0.9 are a near-tie (0.022 ± 0.101).
- **`next_close` against `next_open`.** At delay 1, `next_close` fills gave lower gross and net Sharpe ratios than `next_open` fills at every persistence level (paired; for example net 0.214 ± 0.064 lower at 0.5 and 0.062 ± 0.038 lower at 0.97).
- **Net sign.** The net Sharpe ratio was negative at persistence 0.5 for all four fill settings and positive at persistence 0.97 (0.68 to 0.82). At persistence 0.9 only `next_open` with delay 3 had an interval below zero (−0.18 ± 0.18); the other three intervals contain zero.

## E5 Validation schemes

50 `null` markets (seeds 6000 to 6049) and 50 `planted` markets (seeds 7000 to 7049), each 60 × 1533: a sample of 1260 bars, a gap of 21, and a held-out segment of 252.
- **Model.** k-nearest-neighbour regression with k = 25, per instrument. Features: trailing returns over 5, 21, and 63 bars and the 21-bar trailing volatility, standardized with the instrument's training observations. The target is the forward compounded return over h bars.
- **Score.** The Spearman correlation pooled over instruments.
- **Schemes** (K = 5):
  - shuffled K-fold and contiguous K-fold without purge, both run through `allow_unpurged` and labelled UNPURGED;
  - purged K-fold with an embargo of h;
  - walk-forward.
- **Bias.** The cross-validation score minus the held-out score of the model fitted on the whole sample. The table gives its mean ± t interval and its standard deviation across markets (`results/e5/aggregates.csv`).

| Market | h | Scheme | Label | Bias: CV minus held-out Spearman (mean ± 95 % t) | Spread of the bias (sd) | CV score | Held-out score |
|---|---|---|---|---|---|---|---|
| null | 1 | shuffled_kfold | not flagged (no overlap) | 0.000 ± 0.003 | 0.010 | -0.001 ± 0.001 | -0.001 ± 0.002 |
| null | 1 | contiguous_kfold | not flagged (no overlap) | 0.001 ± 0.003 | 0.010 | 0.000 ± 0.001 | -0.001 ± 0.002 |
| null | 1 | purged_kfold | purged | 0.001 ± 0.003 | 0.010 | 0.000 ± 0.001 | -0.001 ± 0.002 |
| null | 1 | walk_forward | purged | 0.001 ± 0.003 | 0.009 | 0.001 ± 0.001 | -0.001 ± 0.002 |
| null | 5 | shuffled_kfold | UNPURGED | 0.260 ± 0.005 | 0.018 | 0.256 ± 0.002 | -0.004 ± 0.005 |
| null | 5 | contiguous_kfold | UNPURGED | 0.003 ± 0.006 | 0.022 | -0.001 ± 0.003 | -0.004 ± 0.005 |
| null | 5 | purged_kfold | purged | 0.001 ± 0.006 | 0.022 | -0.003 ± 0.003 | -0.004 ± 0.005 |
| null | 5 | walk_forward | purged | 0.002 ± 0.006 | 0.020 | -0.002 ± 0.003 | -0.004 ± 0.005 |
| null | 21 | shuffled_kfold | UNPURGED | 0.444 ± 0.010 | 0.035 | 0.439 ± 0.003 | -0.005 ± 0.009 |
| null | 21 | contiguous_kfold | UNPURGED | 0.004 ± 0.010 | 0.037 | -0.001 ± 0.005 | -0.005 ± 0.009 |
| null | 21 | purged_kfold | purged | -0.004 ± 0.011 | 0.037 | -0.010 ± 0.005 | -0.005 ± 0.009 |
| null | 21 | walk_forward | purged | 0.000 ± 0.010 | 0.037 | -0.005 ± 0.006 | -0.005 ± 0.009 |
| planted | 1 | shuffled_kfold | not flagged (no overlap) | -0.003 ± 0.003 | 0.012 | -0.001 ± 0.002 | 0.002 ± 0.003 |
| planted | 1 | contiguous_kfold | not flagged (no overlap) | -0.004 ± 0.004 | 0.013 | -0.002 ± 0.002 | 0.002 ± 0.003 |
| planted | 1 | purged_kfold | purged | -0.004 ± 0.004 | 0.013 | -0.002 ± 0.002 | 0.002 ± 0.003 |
| planted | 1 | walk_forward | purged | -0.004 ± 0.003 | 0.012 | -0.002 ± 0.002 | 0.002 ± 0.003 |
| planted | 5 | shuffled_kfold | UNPURGED | 0.258 ± 0.005 | 0.019 | 0.258 ± 0.002 | 0.000 ± 0.005 |
| planted | 5 | contiguous_kfold | UNPURGED | -0.002 ± 0.006 | 0.020 | -0.001 ± 0.003 | 0.000 ± 0.005 |
| planted | 5 | purged_kfold | purged | -0.004 ± 0.006 | 0.020 | -0.004 ± 0.003 | 0.000 ± 0.005 |
| planted | 5 | walk_forward | purged | -0.004 ± 0.005 | 0.019 | -0.004 ± 0.003 | 0.000 ± 0.005 |
| planted | 21 | shuffled_kfold | UNPURGED | 0.444 ± 0.010 | 0.033 | 0.441 ± 0.003 | -0.003 ± 0.009 |
| planted | 21 | contiguous_kfold | UNPURGED | -0.002 ± 0.011 | 0.038 | -0.005 ± 0.006 | -0.003 ± 0.009 |
| planted | 21 | purged_kfold | purged | -0.013 ± 0.011 | 0.038 | -0.016 ± 0.006 | -0.003 ± 0.009 |
| planted | 21 | walk_forward | purged | -0.006 ± 0.010 | 0.034 | -0.010 ± 0.006 | -0.003 ± 0.009 |

- **Shuffled K-fold.** It overstated the held-out score when labels overlap: by 0.258 to 0.260 at h = 5 and by 0.444 at h = 21, on null and planted markets alike. At h = 1 its bias was 0.000 (null) and −0.003 (planted).
- **The other three schemes.** Contiguous K-fold without purge, purged K-fold, and walk-forward had mean biases between −0.013 and +0.004 at every h.
  - Contiguous K-fold without purge showed no bias distinguishable from zero at h = 5 and 21. Only observations within h bars of a fold boundary share labels with the test fold.
  - Some small negative biases on planted markets have intervals that exclude zero: purged K-fold at h = 21 (−0.013 ± 0.011, an interval ending 0.002 below zero) and walk-forward at h = 1 (−0.004 ± 0.003, ending 0.0006 below zero). At h = 1 the intervals of contiguous K-fold and purged K-fold (−0.00375 ± 0.00364 and −0.00373 ± 0.00364) end within 0.0002 of zero, which is borderline and not a finding.

## E5b CPCV in the comparison of validation schemes (Tier 2 item 1, commit `6892cb1`)

`python -m quant_research_engine experiment run --only e5b --workers 2` (configuration `experiments/configs/e5b.json`; quick entry in `quick.json`). It ran once, on commit `6892cb1` with a clean tree: 2 workers, 373 s wall time, CPU load 29 % before and 28 % after (manifest).

Setup:
- **Markets:** 25 `null` markets (seeds 8000 to 8024) and 25 `planted` markets (seeds 9000 to 9024), each 60 × 1533, with E5's sample, gap, and held-out segment. The model, features, and labels are those of E5, at h = 5.
- **Schemes:** the four schemes of E5, plus CPCV with 6 groups and 2 test groups. CPCV has 15 splits, purged and embargoed by h, and 5 backtest paths: path p takes, for every group, the p-th split in which that group is tested (`cpcv_paths`, tested in `tests/test_tier2_e5b.py`).
- **Measures:** the Spearman score, and the per-bar Sharpe ratio of a long-short portfolio. The portfolio's weights are the demeaned cross-sectional ranks of the predictions, at gross exposure 1, held one bar, with no costs.
- **Bias:** a scheme's estimate minus the same measure on the held-out segment. A market's CPCV estimate is the mean over its 5 paths, and the last column gives the spread of the paths within a market (`results/e5b/aggregates.csv`, `rows.csv`).

| Market | Scheme | Label | Spearman: bias (mean ± 95 % t) | sd of the bias | mean abs. bias | Sharpe per bar: bias (mean ± 95 % t) | sd of the bias | mean abs. bias | sd over CPCV paths within a market (Spearman / Sharpe) |
|---|---|---|---|---|---|---|---|---|---|
| null | shuffled_kfold | UNPURGED | 0.257 ± 0.007 | 0.018 | 0.257 | 0.591 ± 0.025 | 0.061 | 0.591 | – |
| null | contiguous_kfold | UNPURGED | -0.001 ± 0.008 | 0.020 | 0.016 | 0.020 ± 0.026 | 0.062 | 0.052 | – |
| null | purged_kfold | purged | -0.003 ± 0.008 | 0.020 | 0.016 | 0.014 ± 0.026 | 0.063 | 0.049 | – |
| null | walk_forward | purged | 0.000 ± 0.008 | 0.020 | 0.016 | 0.016 ± 0.028 | 0.067 | 0.053 | – |
| null | cpcv | purged | -0.003 ± 0.008 | 0.020 | 0.017 | 0.016 ± 0.025 | 0.061 | 0.051 | 0.005 / 0.016 |
| planted | shuffled_kfold | UNPURGED | 0.253 ± 0.009 | 0.021 | 0.253 | 0.573 ± 0.032 | 0.077 | 0.573 | – |
| planted | contiguous_kfold | UNPURGED | -0.006 ± 0.009 | 0.022 | 0.017 | 0.005 ± 0.033 | 0.080 | 0.066 | – |
| planted | purged_kfold | purged | -0.008 ± 0.009 | 0.021 | 0.017 | 0.002 ± 0.032 | 0.079 | 0.064 | – |
| planted | walk_forward | purged | -0.005 ± 0.008 | 0.020 | 0.016 | 0.005 ± 0.029 | 0.070 | 0.058 | – |
| planted | cpcv | purged | -0.006 ± 0.008 | 0.021 | 0.017 | 0.005 ± 0.031 | 0.075 | 0.063 | 0.005 / 0.017 |

"Smallest mean absolute bias" and "largest sd of the bias" (`aggregates.json`) are point estimates. Among the four schemes other than shuffled K-fold, they differ by at most 0.002 in Spearman score and 0.010 in Sharpe ratio, so in this run the four are not distinguishable. Every one of their mean biases has an interval that contains zero.

| Market, measure | Smallest mean absolute bias | Largest sd of the bias |
|---|---|---|
| null_h5_sharpe | purged_kfold | walk_forward |
| null_h5_spearman | walk_forward | purged_kfold |
| planted_h5_sharpe | walk_forward | contiguous_kfold |
| planted_h5_spearman | walk_forward | contiguous_kfold |

## Kenneth French illustration (Tier 2 item 2, commit `0ea0e9b`; illustration, no claims)

Run with `python scripts/illustrate_french.py`. It is not part of `experiment run`, because it needs a one-time download into the git-ignored `data/cache/`. With the file cached, `--offline` reproduces it without the network. The result, `results/t2_french/summary.json`, holds derived metrics only, and its manifest records a clean tree on `0ea0e9b`. Data, terms, run, and numbers are in the README. This is the only result in the repository that does not come from synthetic data, and it is labelled "illustration, no claims".

## Capacity and impact (Tier 2 item 3, commit `df91f7c`)

`python -m quant_research_engine experiment run --only t2_capacity --workers 2` (configuration `experiments/configs/t2_capacity.json`). It ran once, on commit `df91f7c` with a clean tree: 2 workers, 54 s wall time, CPU load 38 % before and 11 % after. No other project's run was active, and the owner's training job was running.

Setup:
- **Strategy and markets:** `signal_x_ls` on 10 `planted` markets, 60 × 1260 (seeds 10000 to 10009), warmup 40, a decision every 5 bars, event engine, equity sizing, base cost model, at initial capitals of 1e6, 1e7, 3e7, 1e8, 3e8, 1e9, and 3e9.
- **Per run:** the net and gross annualized Sharpe ratios; impact and all costs per year as fractions of the previous bar's equity; and the share of fill events (an instrument and a bar with at least one non-delisting fill) whose filled quantity reached the participation cap times the bar's volume (`results/t2_capacity/aggregates.csv`, `rows.csv`).
- **Capacity:** the first crossing of the mean net Sharpe curve through half its value at the smallest capital, and through zero, by linear interpolation in log capital (`aggregates.json`).

| Initial capital | Net Sharpe (ann.) | Gross Sharpe (ann.) | Impact per year / equity | All costs per year / equity | Fill events at the participation cap |
|---|---|---|---|---|---|
| 1e+06 | 0.48 ± 0.20 | 0.95 ± 0.19 | 0.026 ± 0.004 | 0.042 ± 0.004 | 0.003 ± 0.004 |
| 1e+07 | 0.00 ± 0.20 | 0.98 ± 0.18 | 0.072 ± 0.008 | 0.087 ± 0.008 | 0.082 ± 0.037 |
| 3e+07 | -0.37 ± 0.15 | 0.97 ± 0.15 | 0.104 ± 0.009 | 0.120 ± 0.009 | 0.251 ± 0.059 |
| 1e+08 | -0.77 ± 0.21 | 0.95 ± 0.20 | 0.139 ± 0.009 | 0.154 ± 0.009 | 0.541 ± 0.053 |
| 3e+08 | -1.23 ± 0.30 | 0.68 ± 0.27 | 0.159 ± 0.008 | 0.174 ± 0.008 | 0.775 ± 0.025 |
| 1e+09 | -1.19 ± 0.30 | 0.41 ± 0.28 | 0.159 ± 0.008 | 0.173 ± 0.009 | 0.924 ± 0.013 |
| 3e+09 | -0.75 ± 0.22 | 0.24 ± 0.25 | 0.117 ± 0.012 | 0.129 ± 0.013 | 0.981 ± 0.005 |

| Capacity (log-linear interpolation of the mean net Sharpe ratio; first crossing) | Capital |
|---|---|
| mean net Sharpe ratio at the smallest capital | 0.48 |
| capital at which it falls to half | 3.18e+06 |
| capital at which it falls to zero | 1.01e+07 |

## Effective number of trials and multiple testing (Tier 2 item 4, commit `d26a863`)

`python -m quant_research_engine experiment run --only t2_trials --workers 2` (configuration `experiments/configs/t2_trials.json`). It ran once, on commit `d26a863` with a clean tree: 2 workers, 541 s wall time, CPU load 25 % before and 96 % after (the owner's training job; no other project's run was active).

Setup:
- **Markets and returns:** the 200 null and 200 planted markets of E3 (seeds 1000 to 1199 and 2000 to 2199, the same derived market seeds), with the gross per-bar returns of the 108 trials.
- **Rules:** see the README. In short:
  - N_eff = (Σλ)² / Σλ² over the eigenvalues of the trials' return correlation matrix;
  - p-value of trial j = 1 − PSR_j(0);
  - Holm and Benjamini-Hochberg at 0.05.
- **Consistency check against E3:** the selection, the naive PSR flag, and the N = 108 DSR flag equal E3's committed values on all 400 markets (checked from `results/e3/selected.csv`; also on the 40 quick markets).

| Rule | Null markets: share with a discovery (false positive) | Planted markets: share with a discovery involving signal_x (power) |
|---|---|---|
| naive PSR(0) > 0.95 | 86.0 % [80.5, 90.1] | 88.5 % [83.3, 92.2] |
| DSR, N = 108, > 0.95 | 0.0 % [0.0, 1.9] | 0.5 % [0.1, 2.8] |
| DSR, N = round(N_eff), > 0.95 | 2.0 % [0.8, 5.0] | 40.0 % [33.5, 46.9] |
| Holm, 0.05 | 3.0 % [1.4, 6.4] | 57.5 % [50.6, 64.1] |
| Benjamini-Hochberg, 0.05 | 3.5 % [1.7, 7.0] | 63.5 % [56.6, 69.9] |

| Market | Markets | Effective number of trials N_eff: mean (min to max) |
|---|---|---|
| null | 200 | 11.0 (10.1 to 11.7) |
| planted | 200 | 11.0 (10.1 to 11.9) |

## TODO

- **Variance of the E2 Sharpe comparison.** Run more markets for the C7 comparison: with 10 markets the ratio interval (±0.09 gross) cannot resolve the expected gap of a few percent.
- **Throughput.** Measure the engines' throughput inside the experiments on an idle machine. Today it is wall-clock on a shared machine, per-run setup included (E1).
