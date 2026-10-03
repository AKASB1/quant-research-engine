# Contracts

This repository follows **quant-contracts version 1**, a language-neutral contract shared with a decision-layer project that is built independently and shares no code with this one. This document restates everything a reader of this repository needs. Changes are deliberate, versioned, and logged; the deviations and extensions of this repository are listed at the end.

## 1. Conventions

**Time.** Every instant is UTC. Files use `YYYY-MM-DDTHH:MM:SSZ` (seconds precision, `Z` suffix). In memory, instants are int64 microseconds since the epoch. A bar covers `[ts_open, ts_event)`; its open, high, low, close, and volume are final at `ts_event`; `ts_avail >= ts_event` is the earliest instant a user may know the row. A day count is seconds / 86400; an annual rate accrues as `rate * days / 365`.

**Calendars.** `equity_daily`: Monday to Friday except 1 January and 25 December (no observance shift), every session `14:30:00Z` to `21:00:00Z` (no daylight-saving handling). `crypto_daily`: every day, `ts_open` 00:00:00Z, `ts_event` 00:00:00Z of the next day. `weekly` and `monthly` take the last session of each ISO week or calendar month of a daily calendar. Bars per year (`ppy`) are declared, never inferred: 252, 365, 52, 12.

**Randomness.** One stream per component name: `stream(seed, name)` is a NumPy `Generator` over `PCG64DXSM` with state `(s1 << 64) | s2`, `s1 = splitmix64(seed ^ fnv1a64(name))`, `s2 = splitmix64(s1 ^ 0x9E3779B97F4A7C15 ^ fnv1a64(name))`, increment `((6364136223846793005 << 64) | 1442695040888963407) | 1`. Stream names used here: `synth.<component>` (including `synth.fund_x`), `strategy.<strategy>.<name>`, `bootstrap`, `cv`, `audit.poison`, `audit.instants`, `audit.instants.curve`, `e1.scenario`, `experiment.<id>`, `experiment.e3.lo.<phi>`. Common random numbers: what a seed generates depends only on the generator configuration and the seed.

**Units.** Prices, money, and costs are float64 in the quote currency; returns are decimals; 1 bp = 1e-4. Quantities are float64 in units, rounded toward zero to the instrument's `lot_size` when it is positive, and held on a binary grid of 2^-20 units (see "Deviations").

**Determinism.** The same configuration, data, and seed give identical output bytes, except columns whose names start with `wall_` and the manifest fields that record time, host, and load. Ties break by `instrument_id` (byte order), then time, then input order. Reductions that reach a result run on sorted arrays with one BLAS thread, in an explicit order, or with `math.fsum`. Reproducibility is claimed for one machine and software stack, recorded in each manifest. Across platforms, floats can differ in their last digit: an independent Linux check found such differences, and the tests therefore compare generated files with committed ones numerically.

**Files.** CSV: UTF-8 without BOM, LF (CRLF accepted on read), comma separated, one header line equal to the schema, no quoting, no padding. Floats use the shortest round-trip representation; `volume`, `adv_shares`, `quantity`, `lot_size`, `horizon_bars`, `vintage` are written without a decimal point when integral; `NaN`, `inf`, and `-0.0` never appear; a missing value is an empty field where allowed. A header with no rows is a valid empty dataset; a zero-byte file is invalid. Loaders reject a wrong header, a wrong field count, an unparsable or out-of-range value, an empty required field, unsorted or duplicated keys, and every schema rule below, naming the 1-based line number. Beside `<name>.csv` lies `<name>.manifest.json` with `schema_version` (1), `qc_version` (1), `generator` (name, version, parameters), `seed`, `row_count`, and `content_sha256`. JSON is UTF-8 with sorted keys and two-space indentation.

**Configuration hash.** SHA-256 (lowercase hex) of the canonical JSON: sorted keys, no whitespace, integers without a decimal point, shortest round-trip floats, and every field equal to its default omitted.

## 2. Schemas (version 1)

`instrument_id`: non-empty ASCII letters, digits, `_`, `-`.

| File | Header | Sorted by | Rules |
|---|---|---|---|
| `instruments` | `instrument_id,symbol,asset_class,currency,ts_list,ts_delist,delist_return,lot_size,tick_size,sector` | `instrument_id` | `asset_class` in equity, crypto, index, other; `currency` uppercase letters; `ts_delist` empty or the `ts_event` of the last bar; `delist_return` empty iff `ts_delist` empty, else >= -1; `ts_delist >= ts_list`; `lot_size >= 0` (0 = fractional); `tick_size >= 0` |
| `bars` | `instrument_id,ts_open,ts_event,ts_avail,open,high,low,close,volume` | `instrument_id, ts_event` | `ts_open < ts_event <= ts_avail`; bars of an instrument do not overlap; prices > 0; high >= max(open, close, low); low <= min(open, close, high); volume >= 0 (0 = no trading); prices are raw (not adjusted) |
| `corporate_actions` | `instrument_id,action,ts_ex,ts_avail,value` | `instrument_id, ts_ex, action` | `action` in split, cash_dividend; `ts_avail <= ts_ex` (announcement); value > 0; split: new shares per old share; dividend: cash per pre-split share |
| `series` | `series_id,ts_event,ts_avail,vintage,value` | `series_id, ts_event, vintage` | per `(series_id, ts_event)` vintages 0, 1, 2, ... with strictly increasing `ts_avail >= ts_event` |
| `returns` | `ts_event,ts_avail,instrument_id,ret` | `ts_event, instrument_id` | `ret = (close_t * ratio_t + dividend_t) / close_{t-1} - 1`; no return for an instrument's first bar |
| `signals` | `ts_event,ts_avail,instrument_id,name,value` | `ts_event, name, instrument_id` | finite value |
| `forecasts` | `ts_event,ts_avail,instrument_id,horizon_bars,mu,sigma` | `ts_event, instrument_id, horizon_bars` | `mu`: expected simple return over the next `horizon_bars` bars; `sigma` empty or its standard deviation |
| `liquidity` | `ts_event,ts_avail,instrument_id,adv_shares,sigma_bar` | `ts_event, instrument_id` | `adv_shares > 0`, `sigma_bar >= 0`, from data known at `ts_avail` |
| `orders` | `order_id,ts_submit,instrument_id,quantity,order_type,limit_price,tif,strategy_id` | `ts_submit, order_id` | quantity signed, nonzero; `market` has an empty `limit_price`; `tif` day or gtc |
| `fills` | `fill_id,order_id,ts_fill,instrument_id,quantity,ref_price,price,spread_cost,impact_cost,commission` | `ts_fill, fill_id` | costs >= 0; a forced delisting exit has `order_id` `DELIST` and no costs |
| `positions` | `ts_event,instrument_id,quantity,mark_price,value` | `ts_event, instrument_id` | one row per nonzero position at each bar's end; `value = quantity * mark_price` |
| `equity` | `ts_event,cash,position_value,equity,hold_pnl,trade_pnl,spread_cost,impact_cost,commission,borrow,financing,income` | `ts_event` | one row per bar of the engine calendar, the first is the opening row (initial cash, no flows); the identity of section 5 holds row to row |

**As-of semantics of series.** As of `t`, the value of `(series_id, ts_event)` is the row with the largest vintage among those with `ts_avail <= t`; with none it is unknown.

## 3. Knowledge, decisions, fills

- A row is known at `t` iff `ts_avail <= t`. An instrument is known from `ts_list`; its `ts_delist` and `delist_return` read as empty before `ts_delist`. A corporate action is known from its announcement `ts_avail`.
- **Point-in-time adjustment.** The split-adjusted close as of `t` divides the raw close of every bar before a split's ex-bar by the split's ratio, for splits with `ts_ex <= t` only. The total-return series as of `t` is the cumulative product of the returns of the bars known at `t`, scaled so that its last value equals the last split-adjusted close; a dividend enters when its ex-bar is known.
- **Universe** at `t`: instruments with `ts_list <= t` and (`ts_delist` empty or `ts_delist > t`), including those that delist later.
- **Decisions.** A decision at `t` sees rows known at `t` and the engine's state after all fills up to `t`; its orders are submitted at `t`. Pending orders are not cancelled by the next decision: the new order is the difference between the target and the position that will exist when all pending orders have filled.
- **Fills.** An order submitted at the end of bar `k` of an instrument fills at the earliest in bar `k + fill_delay_bars` (at least 1) of that instrument. Reference price `next_open` (default for `equity_daily`) or `next_close` (default for `crypto_daily`). Filling at the close of the decision bar (`same_close`) exists only behind `unsafe_same_bar_fill`; such a run is labelled `UNSAFE` everywhere and the report generator refuses it. The participation cap limits a fill to `participation_cap * volume` of the fill bar: the rest of a `gtc` order waits for the next bar, the rest of a `day` order is cancelled. Nothing fills in a zero-volume bar: a `gtc` order waits, a `day` order is cancelled.
- **Shorts and leverage.** Shorts only with `allow_short`. `max_leverage` (gross position value over equity, optional) rejects an order that would exceed it; every rejected order is recorded.
- **Order of operations in each bar of an instrument:** (1) splits rescale the opening quantity, the previous close used for PnL, and pending order quantities; (2) dividends are computed on the pre-split opening quantity and credited at the close; (3) eligible orders fill; (4) in the bar with `ts_event = ts_delist`, any remaining position is closed at `close * (1 + delist_return)` as a `DELIST` fill without costs, and pending orders are cancelled; (5) accruals; (6) marks at the close (an instrument without a bar at an instant is marked at its last close).

## 4. Cost model v1

Configuration (all optional, defaults shown): `commission_bps` 1.0, `commission_per_share` 0.0, `min_commission` 0.0, `half_spread_bps` 2.0, `impact` `{"model": "sqrt", "y": 0.5}` (`sqrt`, `linear`, or `none`), `borrow_bps_annual` 50.0, `financing_bps_annual` 100.0, `cash_rate_bps_annual` 0.0, `participation_cap` 0.1.

For a fill of signed quantity `q` at reference price `m`, with the decision-time `sigma` (`sigma_bar`) and `V` (`adv_shares`):

- `spread_cost = |q| m half_spread_bps / 1e4`
- `impact_bps = 1e4 y sigma sqrt(|q| / V)` (`sqrt`), `1e4 y sigma |q| / V` (`linear`), 0 (`none`); 0 and counted as `impact_unavailable` when `sigma` or `V` is unknown
- `impact_cost = |q| m impact_bps / 1e4`
- `price = m (1 + sign(q) (half_spread_bps + impact_bps) / 1e4)`
- `commission = max(min_commission, |q| m commission_bps / 1e4 + |q| commission_per_share)`
- Impact never reads the fill bar's own volume or range; the participation cap does (it is a physical limit, not a cost).
- Accruals at each bar's close with `dt` days since the previous close: `borrow = sum_short |q_{t-1}| P_{t-1} borrow_bps_annual / 1e4 dt / 365`; `financing = max(0, -cash_{t-1}) financing_bps_annual / 1e4 dt / 365`; interest income `max(0, cash_{t-1}) cash_rate_bps_annual / 1e4 dt / 365`.

## 5. Accounting and attribution

`E_t = cash_t + sum_i q_{i,t} P_{i,t}` (marks at the close). For each bar:

`E_t - E_{t-1} = hold_pnl + trade_pnl - spread_cost - impact_cost - commission - borrow - financing + income`

with `hold_pnl = sum_i q_{i,t-1}^adj (P_{i,t} - P_{i,t-1}^adj)` (after the split of the bar), `trade_pnl = sum_f q_f (P_{i,t} - m_f)`, and `income` = dividends received minus dividends paid on shorts plus interest. The identity holds to `1e-9 * max(1, |E_{t-1}|)` per bar; engines assert it in memory on every bar. `gross_pnl = hold_pnl + trade_pnl`; per-bar net return `r_t = (E_t - E_{t-1}) / E_{t-1}`; turnover `sum |q_f| m_f / E_{t-1}`; gross exposure `sum |q_i P_i| / E`, net exposure `sum q_i P_i / E`. Attribution by instrument and by long and short book sums to the totals.

## 6. Metrics and inference

See [inference.md](inference.md) for every formula with its conventions and the known answers.

## 7. Synthetic market

See [data.md](data.md): the parameters mean what the shared contract says (factors with annual volatilities, a two-state volatility regime, Gaussian or Student-t(5) tails scaled to unit variance, an overnight share `gap_share` of variance and mean, a latent AR(1) signal with persistence `phi` and information coefficient `ic`, an observable proxy `signal_x` with `observable_noise`, events, and distribution shift).

## 8. Export v1

The research engine writes a directory holding `manifest.json`, `instruments.csv`, `bars.csv`, `corporate_actions.csv`, `returns.csv`, `signals.csv`, `forecasts.csv`, and `liquidity.csv`, each a version-1 schema with its own `.manifest.json`. `manifest.json` lists the files with their `content_sha256`, the calendar, `ppy`, the seed and generator (or the data source with its licence note), the producer's commit, and `export_version` 1. A consumer validates every file against its schema and hash and checks that no `ts_avail` precedes its `ts_event`. An export derived from third-party data has header-only `bars.csv` and `corporate_actions.csv`, no `series.csv`, and `derived_only: true`. The decision layer needs no code from this repository, only these files; it estimates its own risk models.

## Deviations and extensions (this repository)

Deviations (each is a conservative reading of the contract; the owner reconciles them):

- **Quantity grid.** After rounding toward zero to `lot_size`, quantities are held on a binary grid of 2^-20 units (about 1e-6; rounded to nearest after a split rescale). Quantity arithmetic is then exact in float64, the two engines agree bit for bit on quantities, and a position equals its netted target with no floating-point residue. Fractional quantities remain fractional.
- **Returns of fixed-capital runs.** In the vectorized experiments that size on a fixed capital (E3, E4 b and c), per-bar returns are the change in equity over the fixed sizing capital, not over the previous equity: with heavy costs the equity of a fixed-capital run can shrink toward zero or below, where a return over the previous equity is meaningless. Runs sized on equity keep `r_t = (E_t - E_{t-1}) / E_{t-1}`.
- **Participation cap per bar.** The cap `participation_cap * volume` limits the total filled in an instrument's bar; pending orders share it first in, first out.
- **Leverage and shorts.** `max_leverage` (optional) rejects an order whose committed gross exposure at the decision's closes would exceed it; with `allow_short` false an order whose target is negative is rejected; every rejection is logged with its reason.
- **Same-close fills on the opening bar** are refused (they would put flows on the opening row).
- **Vectorized engine and split ratios.** The vectorized engine accepts integer split ratios only (the generator draws 2 and 3); then every rescale is exact.

Extensions:

- **`fund_x`.** The generator also writes a lagged, revised series `<instrument_id>.fund_x` (see [data.md](data.md)). It gives the look-ahead canaries something to leak; the decision layer ignores it.
- **Export payload.** `signals.csv` holds `signal_x` and the built-in features at their default parameters (raw values): `momentum_21_5`, `reversal_5`, `low_vol_63`, `volume_trend_21`, `signal_x_ewma_3`. `forecasts.csv` holds horizons 1, 5, and 21 with `mu` from an expanding-window, past-only ridge regression of the forward compounded return on `signal_x_ewma(3)` (pooled over instruments; slope penalty 10, intercept not penalized; at least 500 observations whose labels are realized by the row's instant) and `sigma = sigma_bar * sqrt(h)`. `series.csv` (optional in the contract) holds `signal_x` and `fund_x`. Every derived row is recomputed bit for bit from the store cut at its `ts_avail` (tested).
- **Liquidity function.** `adv_shares` is the mean volume of the last 20 bars in the shares of the current bar (volumes before a split are multiplied by its ratio); `sigma_bar` the square root of the weighted mean of the squared returns of the last 250 bars with weight `0.5 ** (age / 20)`; a row exists from the 21st bar.
- **Attribution file.** Each run also writes `attribution.csv` (per instrument: hold PnL, trade PnL, fill costs, dividends, borrow) and `run.json` (engine configuration, label `safe` or `UNSAFE`, books, residuals, log hashes).
