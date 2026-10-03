# Data

Every result in this repository comes from **synthetic data with assumed parameters**. The generator plants a known, small predictability so that the machinery can be checked against the truth; a planted signal that is recovered shows that the machinery works, not that any strategy makes money, and nothing here says anything about real markets.

## The synthetic market (`synth.generate(config, seed)`)

Per bar `t` and instrument `i` the simple total return is `r = beta_i' f_t + e_{i,t} + c` (floored at -0.9, a documented floor that essentially never binds):

- **Factors** `f_t`: the first is the market; each has an annual volatility and an annual mean; loadings `beta_i` (market: normal with mean 1 and standard deviation 0.25; style factors: mean 0, standard deviation 0.5).
- **Idiosyncratic part** `e_{i,t} = sigma_i v_t (ic s_{i,t-1} + sqrt(1 - ic^2) z_{i,t})`, `sigma_i` the calm per-bar volatility (log-normal across instruments, or equal), `v_t` the regime multiplier, `z` unit-variance noise. The correlation of `s_{i,t-1}` with the standardized idiosyncratic return is exactly `ic`.
- **Latent signal** `s_{i,t} = phi s_{i,t-1} + sqrt(1 - phi^2) u_{i,t}` (unit variance, `phi` = `persistence`). It is never stored; it is returned beside the store for oracles and checks only.
- **Observable proxy** `signal_x = (s + n w) / sqrt(1 + n^2)` (`n` = `observable_noise`), published as the series `<id>.signal_x` at the bar's `ts_avail`; its correlation with `s` is `1/sqrt(1 + n^2)` (0.707 for `n` = 1).
- **Regime**: two states, calm (multiplier 1) and stress (`stress_vol_mult`), Markov transitions per bar.
- **Tails**: Gaussian or Student-t with 5 degrees of freedom scaled to unit variance.
- **Overnight and intraday**: every unit shock is drawn as `sqrt(g) a + sqrt(1 - g) b` with independent `a` and `b`, and every mean is split `g : 1 - g` (`g` = `gap_share`); the overnight return is the first part and the intraday return is defined by `(1 + r_on)(1 + r_id) = 1 + r`. The variance and mean shares hold to first order (the product definition adds a cross term of second order). A fill at the next open therefore misses the overnight share of the next bar's planted mean.
- **Prices**: initial price log-normal; `close_t = ((1 + r_t) close_{t-1} - D_t) / ratio_t` and `open_t = ((1 + r_on) close_{t-1} - D_t) / ratio_t` (`D_t` the dividend per pre-split share credited in the bar, `ratio_t` the split ratio): the returns of the contract (2.5) equal the drawn returns (to about 3e-16 in practice), and neither a split nor a dividend changes a mean. High and low extend the open-close range by half-normal amounts scaled by the intraday volatility.
- **Volume**: log-normal around an instrument's average volume with an AR(1) log deviation, in the shares of the bar (multiplied by the cumulative split factor); zero-volume bars (no trading) with a small probability.
- **Events**: delistings with an annual hazard (after a minimum life; `delist_return` ~ N(mean, sd) clipped to [-1, 1]; `ts_delist` = `ts_event` of the last bar), splits with an annual rate and ratios 2 or 3 (announced `announce_lead_bars` before the ex-bar), cash dividends paid by a share of instruments every `dividend_every_bars` bars (sized at the announcement as `yield / 4` times the close then, rounded to 4 decimals), and late listings (listing at a uniform bar from bar 21 to the middle of the sample).
- **Distribution shift** `shift = {at_bar, vol_mult, corr_mult, ic_mult, mu_shift_annual}`: from `at_bar` all volatilities are multiplied by `vol_mult`, loadings by `sqrt(corr_mult)` (capped at 95 % of an instrument's variance) with the idiosyncratic volatility rescaled to keep the total variance (so pairwise correlations scale by `corr_mult`), `ic` by `ic_mult`, and `mu_shift_annual / ppy` is added to the market mean per bar.
- **`fund_x`** (an extension of this repository; the decision layer ignores it): every 21 bars a period closes at a bar's close; vintage 0 is published 5 bars later with value `n + w z0`, vintage 1 15 bars after the period end with value `value_0 + w z1`, where `z0` (`z1`) is the sum of the drawn returns of the 5 bars after the period end (bars 6 to 15) divided by `sigma_i sqrt(5)` (`sqrt(10)`), `n` a unit-variance draw (stream `synth.fund_x`), `w` = 2. The series carries information about returns after its period end, so reading it before its release (canary C6) or its revision before it exists (canary C3) is a leak; read as of its `ts_avail` it is useless.

Every component draws from its own stream `synth.<component>`; the same configuration and seed give byte-identical Parquet files on one platform with the same package versions (tested across fresh processes). On another platform floats can differ in their last digit ([../experiments/README.md](../experiments/README.md#reproducibility-across-platforms)).

### Assumed parameter values

Defaults (every configuration starts from them): `equity_daily` calendar from 2010-01-04, `ppy` 252; factors market (annual vol 0.16), style1 (0.08), style2 (0.06), means 0; idiosyncratic annual vol log-normal with median 0.30 and log sd 0.25; regime stress multiplier 2, P(calm to stress) 0.01 and P(stress to calm) 0.05 per bar; Student-t(5) tails; `gap_share` 0.25; `ic` 0.02, `persistence` 0.9, `observable_noise` 1.0; average volume log-normal (median 1e6 shares, log sd 1.0), AR(1) log deviation phi 0.8 and innovation sd 0.3, zero-volume probability 0.001; initial price log-normal (median 40, log sd 0.6); high-low range scale 0.5; delisting hazard 0.04 per year (minimum life 30 bars), delist return N(-0.3, 0.25); splits 0.08 per year (ratios 2, 3; announced 10 bars ahead); dividend yield 2 % for half the instruments, every 63 bars; late listings 10 %; lot size 0; tick 0.01.

| Configuration (`configs/synth/`) | Size | Differences from the defaults |
|---|---|---|
| `small` | 20 x 500 | high event rates (delist 0.2/yr, splits 0.5/yr, dividends for all, late 15 %), zero-volume 0.005; the audit stores |
| `null` | 60 x 1533 | `ic` 0 (E5) |
| `null_e3` | 60 x 1260 | `ic` 0, no delistings (a forced exit has a nonzero mean, which would be a premium) (E2, E3) |
| `planted` | 60 x 1260 | market mean 0.05 per year (E2, E3, E4, E5 with size overrides) |
| `planted_clean` | 100 x 2520 | no factors, equal idiosyncratic vol 0.25, regime off, Gaussian tails, no events, no zero-volume bars (the fundamental-law check) |
| `shift` | 60 x 1260 | planted + shift at bar 630: vol x1.5, correlation x1.5, ic x0.5, market mean shifted by -0.05 per year (to 0) |
| `tiny` | 4 x 60 | very high event rates; the committed CSV fixture `tests/data/tiny/` (the tests regenerate it and compare it cell by cell, floats to a relative 1e-12) |
| `example` | 8 x 120 | high event rates; the committed sample run `docs/examples/` (the tests regenerate it and compare it numerically) |

Experiments override sizes as their configurations say (`experiments/configs/`).

## The liquidity inputs

One pure function (`store.liquidity`) computes, for every bar of an instrument and from its bars up to that bar only: `adv_shares`, the mean volume of the last 20 bars in the shares of the current bar (volumes before a split's ex-bar are multiplied by the split's ratio, so a split inside the window does not halve or double the average); and `sigma_bar`, the square root of the weighted mean of the squared returns of the last 250 bars with weight `0.5 ** (age / 20)` (age 0 for the newest; returns are split- and dividend-neutral). A row exists from the 21st bar on and only when `adv_shares` is positive. Sums run over lags in a fixed order, so recomputing a row from the store cut at its `ts_avail` gives the same bits (tested). The cost model reads these rows as of the decision instant.

## Local-CSV adapter (`store.csv_adapter`)

`load_local_csv(config.json)` maps the columns of a user's daily bar file (`instrument_id`, `date`, open, high, low, close, volume), places each bar on the calendar's session times, sets `ts_avail = ts_event + availability_delay_seconds` (configured, never inferred), derives instruments (listing at the first bar), optionally reads corporate actions in the contract's format, runs the validators (errors name the source line), and builds a store. It works offline. Exports of such a store are refused unless `--derived-only` is given with the source's terms; `bars.csv` and `corporate_actions.csv` are then header-only, `series.csv` is left out, and `derived_only` is true.

## Third-party sources (Tier 2 adapters)

Every adapter lives in `store/adapters.py` and is tested offline on a tiny synthetic fixture with the network disabled (`tests/test_tier2_adapters.py`). Downloads go only to the git-ignored cache `data/cache/` and are never committed, put into an upload package, or exported: the export of a store built from third-party data is refused unless `--derived-only` is given, and then `bars.csv` and `corporate_actions.csv` are header-only.

| Source | What the adapter does | Licence and terms found (what was checked) | Status |
|---|---|---|---|
| Kenneth R. French Data Library, daily portfolio files | Parses the first table of a daily file (returns in percent; -99.99 and -999 mean missing) and builds a return-index series per portfolio, starting at 100. The files have no open and no volume, so the open is set to the previous close, the volume is a placeholder that never binds the participation cap, impact must be off, and fills use `next_close`. `ts_avail` is the bar's close plus a configured delay (1 day in the illustration). | The data library page carries a copyright notice of Eugene F. Fama and Kenneth R. French. No terms of use, licence, permission statement, or click-through was found on the page (read when the adapter was written and again before the download). The daily file's header says it was created from the CRSP database. The library revises history, so a download is the latest vintage and not point-in-time. | One file downloaded to the cache (`49_Industry_Portfolios_daily_CSV.zip`, 4.2 MB, its SHA-256 recorded in the illustration's summary); one illustration, labelled "illustration, no claims"; only derived metrics are committed. |
| Binance public data archive (`data.binance.vision`), daily klines | Parses daily klines (open time in milliseconds before 2025 and microseconds from 2025 on, with or without a header) into `crypto_daily` bars. | The archive's dataset terms (version 1.0, last updated 2026-08-26) license the data under CC BY-NC-SA 4.0 and state that accessing or using the datasets means agreeing to them and to the Binance Terms of Use. Accepting terms is the owner's decision. | Not downloaded. Tested on an offline fixture only. |
| SEC EDGAR `companyfacts` | Not implemented. | Needs a User-Agent of the form the SEC requires, which only the owner can supply (`QRE_SEC_USER_AGENT`); the variable was not set. | Skipped. |

The French illustration is `python scripts/illustrate_french.py` (it downloads the file once into the cache; `--offline` refuses to download). Its result, `experiments/results/t2_french/summary.json`, holds only derived metrics and the SHA-256 of the cached file. No raw value from the file is committed.

## What is never committed

Parquet and DuckDB files, generated stores (`outputs/`), per-run outputs and logs (`experiments/outputs/`), quick results, and anything downloaded (`data/cache/`). The committed data are the golden fixtures of the contract (`tests/data/golden/`), the tiny generated CSV fixture (`tests/data/tiny/`, 74 KB), the adapter test file (`tests/data/adapter/`), the audit table of the test run, the sample run (`docs/examples/`), and the aggregated experiment results (`experiments/results/`).
