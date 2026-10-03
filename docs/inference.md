# Metrics and inference

All quantities are per bar and use the declared bars per year `ppy`; `T` is the number of bars, `r_t` the net per-bar returns, `rf` the per-bar risk-free rate (default 0). Every function is pure and lives in `quant_research_engine.metrics` or `quant_research_engine.inference`.

## Metrics (`metrics`)

| Function | Definition |
|---|---|
| `ann_return(equity, ppy)` | `(E_T / E_0) ** (ppy / T) - 1` |
| `ann_vol(r, ppy)` | `std(r, ddof=1) * sqrt(ppy)` |
| `sharpe(r, ppy=None, rf=0)` | `mean(r - rf) / std(r - rf, ddof=1)`, times `sqrt(ppy)` when `ppy` is given (per bar otherwise; PSR and DSR take the per-bar value) |
| `sortino(r, ppy=None, rf=0)` | `mean(r - rf) / sqrt(mean(min(r - rf, 0)^2))` (denominator over all `T` bars), times `sqrt(ppy)` |
| `max_drawdown(equity)` | largest `(peak_t - E_t) / peak_t`, running peak including `E_0` |
| `calmar(equity, ppy)` | annualized return over maximum drawdown |
| `cvar(r, level=0.95)` | minus the mean of the `ceil(0.05 T)` smallest returns (a positive loss) |
| `hit_rate(r, exposure_prev)` | share of bars with `r > 0` among bars with nonzero exposure at the previous close |
| `skew(r)`, `kurt(r)` | `mean((r-m)^3)/s^3` and `mean((r-m)^4)/s^4` with the `1/T` standard deviation `s` (raw kurtosis; a normal distribution has 3) |
| `autocorr(r, L)` | centered sample autocorrelations, denominator the sum of squares |
| `lo_eta(q, rho=None, r=None, L=10)` | `q / sqrt(q + 2 sum_{k=1}^{q-1} (q - k) rho_k)` with `rho_k` for `k <= min(q-1, L)` and 0 beyond |
| `lo_sharpe(r, ppy, L=10)` | `lo_eta(ppy) * sharpe(r)` (Lo, 2002) |

For the fixed-capital runs of the vectorized experiments (E3, E4 b and c) the per-bar return is the change in equity over the fixed sizing capital; elsewhere it is `(E_t - E_{t-1}) / E_{t-1}` (see `contracts.md`, deviations).

## Selection-aware inference (`inference`)

- `psr(sr, sr_star, t_bars, skew, kurt) = Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1) / 4 SR^2))` (Bailey and Lopez de Prado), raw kurtosis, per-bar Sharpe ratios.
- `expected_max_sharpe(n, var) = sqrt(V) ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))`, `g` = 0.5772156649015329: SR0, the expected maximum per-bar Sharpe of `N` independent trials whose Sharpe ratios have variance `V`.
- `dsr(sr, t_bars, skew, kurt, n, var) = PSR(SR0)`.
- `TrialRegistry(study_id, path)`: every configuration evaluated in a study is a trial; `trials.csv` (`study_id,trial_id,config_hash,sr_bar,t_bars,skew,kurt`) is written as trials are evaluated. `N` is the number of registered trials and `V` the sample variance (ddof 1) of their per-bar Sharpe ratios. No result is called significant without the DSR over its registered trials.
- `pbo_cscv(returns, S=16)`: probability of backtest overfitting by combinatorially symmetric cross-validation. The first `T mod S` rows are dropped; the rest is cut into `S` equal blocks; for each of the `C(S, S/2)` choices of in-sample blocks (12870 for `S` = 16) the in-sample best trial (ties: lowest index) is ranked out of sample: `omega = (1 + #{trials with strictly lower OOS Sharpe}) / (N + 1)`, `lambda = ln(omega / (1 - omega))`; PBO is the share of splits with `lambda <= 0`. Block sums are combined in a fixed block order.
- `stationary_indices(n, length, draws, p, rng)` and `bootstrap_stat(x, stat, draws, p, rng)`: stationary bootstrap (Politis and Romano) on the circle of `T` bars; the next index is the successor with probability `1 - p` and uniform with probability `p` (mean block length `1/p`). Block length rule: fixed (E3 uses `p` = 0.1). `percentile_interval(values, level)`.
- `make_splits(name, n, k, h, allow_unpurged=False, rng=None, embargo=None)`: `purged_kfold` (test block `[a, b)`, training observations with `a - h <= i < b + e` removed, `e` = `h` by default), `walk_forward` (K + 1 blocks; training ends at least `h` bars before the test block), `contiguous_kfold` and `shuffled_kfold` (the mistakes, kept to measure them), and `cpcv(n, groups, k_test, h)` with `C(N, k)` splits and `C(N, k) k / N` paths. Guard G6: a splitter whose training labels overlap test labels for horizon `h` is refused unless `allow_unpurged`, and the result is then labelled `UNPURGED`.

## Known answers (tests, to 1e-9)

- `SR = 1.25 / sqrt(252)`, `T` = 250, skewness -0.5, raw kurtosis 6: `PSR(0)` = 0.8876753070, `PSR(0.5 / sqrt(252))` = 0.7668629015.
- `SR0` with `V` = 1: 1.5745983013 (`N` = 10), 2.5306028932 (100), 3.2551215137 (1000); with `N` = 100 and `V` = 0.25 / 252: 0.0797064991 per bar (1.2653014466 annualized); the DSR of the `SR` above is 0.4940703731.
- Lo, AR(1) with coefficient 0.2: `eta(12)` = 2.8788486905 and `eta(252)` = 12.9722102136 without truncation; 2.8788486939 and 12.9722104255 with `L` = 10.
- `C(16, 8)` = 12870; CPCV with 6 groups and 2 test groups: 15 splits, 5 paths.
- PBO is 1 on a constructed matrix in which the in-sample best trial is always the out-of-sample worst (a trial and its mirror image, block means that never sum to zero), and 0 when one trial has a drift that dominates every set of blocks.

## Calibration on synthetic markets (E3)

Full evaluation, commit `a1a0011` (`experiments/results/e3/`). The results are simulated on synthetic markets with assumed parameters and hold in this simulation, under these assumptions.

Setup:
- 108 trials on 200 null markets (`null_e3`, seeds 1000 to 1199) and 200 planted markets (seeds 2000 to 2199), each 60 instruments by 1260 bars.
- The vectorized engine with fixed capital; 1000 aligned rows of per-bar returns per market.
- Gross panel: all costs zero. It is the calibration, because on a null market every trial's true Sharpe ratio is zero there.
- Net panel: base costs, labelled. Its SR0 rests on the spread of net Sharpe ratios across trials of very different turnover, so the SR0 comparison and the PSR and DSR rates of the net rows are not a null calibration.

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

What this shows, in this simulation:

- **The naive PSR does not account for selection.** On null markets the best of 108 trials had a naive PSR(0) above 0.95 in 86.0 % [80.5, 90.1] of the markets. The DSR over the registered trials was above 0.95 in none of them (0.0 % [0.0, 1.9]).
- **SR0 with N = 108 was above the observed maximum.** The selected per-bar Sharpe ratio averaged 0.91 ± 0.03 of SR0 on null markets. SR0 assumes independent trials, and these trials are correlated: the same 18 signals appear under 2 rules and 3 rebalance intervals. A larger SR0 lowers every DSR, so the DSR here is more conservative than it would be with an SR0 that matched the observed maximum.
- **This cost power.** On planted markets the selected trial used `signal_x` in 88.5 % [83.3, 92.2] of the markets, but its DSR exceeded 0.95 in 1 market of 200 (0.5 % [0.1, 2.8]). The selected Sharpe ratio averaged 1.10 ± 0.03 of SR0. (The test suite's check 17 is a smoke test with a much stronger planted signal, ic 0.15 on 20 instruments by 1000 bars, and asserts a DSR above 0.95; it is not comparable with E3.) Tier 2 item 4 recomputes the DSR with an effective number of trials: on the same 400 markets, the eigenvalue participation ratio gave N_eff ≈ 11. With it the DSR's power on planted markets was 40.0 % [33.5, 46.9], and its false-positive rate on null markets 2.0 % [0.8, 5.0] ([README](../README.md#effective-number-of-trials-and-multiple-testing-commit-d26a863)).
- **PBO.** The mean PBO was 0.48 ± 0.03 on null markets, close to the 0.5 of a selection that carries no information. On planted markets it was 0.20 ± 0.02 (gross); the histograms are in `pbo_hist.csv` and the calibration figure.
- **Bootstrap coverage.** The stationary-bootstrap interval of the Sharpe ratio of the selected trial covered the true value of zero in 37.5 % [31.1, 44.4] of null markets. The same interval for a trial fixed in advance covered it in 95.0 % [91.0, 97.3]. An interval computed after selection on the same data does not have its nominal coverage.
- **Lo adjustment.** For AR(1) returns with coefficients 0.1 to 0.3, the naive annualization `sqrt(252) x SR` overstated the true annualized Sharpe ratio, and its intervals exclude the true value. The Lo-adjusted value (L = 10) had intervals that contain the true value at every coefficient.
