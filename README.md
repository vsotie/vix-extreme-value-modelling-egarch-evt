# Tail Risk for the VIX: A Hybrid Extreme Value Theory Approach

Code accompanying the MSc dissertation *Tail Risk for the Volatility Index (VIX):
A Hybrid Extreme Value Theory Approach* (Vatsal Sotie, MSc Financial Mathematics
with Data Science, University of Bath, 2026).

The analysis follows the two-stage approach of McNeil and Frey (2000). Daily VIX log returns are filtered with an AR(1)-EGARCH(1,1)
model with Student-t innovations. A generalised Pareto distribution (GPD) is then
fitted to the standardised residuals above their 90th percentile, and the two
stages combine into one-day-ahead Value-at-Risk (VaR) and Expected Shortfall (ES)
for the upper tail, where a spike in the VIX is the loss. A rolling backtest
(1,000-day estimation window, 3,933 daily forecasts from 22 December 2010 to
30 June 2026) compares this model with the same two-stage model built on a
symmetric AR(1)-GARCH(1,1) filter, a static peaks-over-threshold model on the raw
returns, and the two filters with Student-t quantiles and no extreme value stage.
Two further benchmarks ask whether the generalised Pareto form itself improves the
forecasts: filtered historical simulation on the same EGARCH filter (EGARCH-FHS,
which changes only the tail stage) and an EGARCH model with skewed Student-t
innovations (EGARCH-skewed-t, a competing whole model that re-estimates the mean,
variance and density together). Sensitivity analyses cover the threshold, the lag
structure, the window length and the 97.5% level.

## Requirements

Python 3.13 and the package versions in `requirements.txt`, which are the
versions used for the dissertation.

```
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

## Data

The data are not included in this repository, and they are not submitted with the
dissertation. The series is the Cboe VIX daily close published by FRED, the Federal
Reserve Bank of St. Louis (series `VIXCLS`,
https://fred.stlouisfed.org/series/VIXCLS), requested for 1 January 2007 to
30 June 2026 and retrieved on 1 August 2026. Anyone can request or download it
there. Acknowledgement: Chicago Board Options Exchange, CBOE Volatility Index: VIX
[VIXCLS], retrieved from FRED, Federal Reserve Bank of St. Louis,
https://fred.stlouisfed.org/series/VIXCLS, 1 August 2026. FRED's notes for the series state
"Copyright, 2016, Chicago Board Options Exchange, Inc. Reprinted with
permission", so the series is not redistributed here. Days on which the index
was not calculated are missing in the source and are dropped together with
their dates. Every script reads the file `data/vix_raw.csv`.

**Forecast files.** The files `results/*forecasts*.csv` contain the VaR and ES
forecasts, the volatility forecasts and the fitted parameters of each window, but not
the realised log returns. A realised return is a function of two consecutive closes,
and a run of returns together with one close reproduces the closes, so including them
would redistribute the series. The scripts that evaluate the forecasts restore the
returns, by date, from your own copy of `data/vix_raw.csv` through
`src/forecast_io.py`, and the scripts that write forecast files drop the column
(`write_forecasts`). Evaluation therefore needs `data/vix_raw.csv` in place, as every
other script does. The restored returns agree with those stored earlier to within
1e-16 (the last digit of a double), and the evaluation scripts print identical logs
when run on the files without the column.

To reproduce the results, download the series from FRED as above and save it as
`data/vix_raw.csv` (columns `Date,Close`, 1 January 2007 to 30 June 2026, days with
missing values dropped), or let `main.py` create it with an API key (see below).
The file from which the results were computed is identified by these values, which
show whether a later download matches it:

```
SHA-256  a4b56444b899772ee02068887c9fc6632eddfb5e6f107ad60a69f8e8f5695c5d
rows     4,934    first 2007-01-03 (12.04)    last 2026-06-30 (16.45)
min      9.14     max 82.69                   sum 97,640.34
```

The checksum is of the file as written on Windows, with CRLF line endings. A copy
whose line endings have been converted has a different checksum and the same
values; the last three lines are unaffected.

### How the data were retrieved

`load_vix` in `src/utils.py` requests the series from the FRED API and writes it
to `data/vix_raw.csv`; `main.py` calls it for the window above. It downloads only
when that file is absent. If the file is absent, run `main.py` with a key set as
below to retrieve the series as FRED publishes it now. FRED can revise past
values, so a later retrieval need not match the file identified above, and
results computed from it may differ slightly. The key is read
from the environment variable `FRED_API_KEY` and is not stored in the
repository. A free key is available at
https://fred.stlouisfed.org/docs/api/api_key.html.

```
$env:FRED_API_KEY = "your_key"    # Windows PowerShell
export FRED_API_KEY=your_key      # macOS / Linux
python src/main.py
```

With `data/vix_raw.csv` in place, no key is needed. To see what FRED has revised
since your download, keep that file in `data/` and run:

```
python src/compare_fred.py
```

`compare_fred.py` needs the key. It downloads the series with the same function
into `data/vix_fred_latest.csv` and lists every date whose value differs from
`data/vix_raw.csv` or that appears in only one of the two files. It does not
modify `data/vix_raw.csv`.

### Source

Chicago Board Options Exchange, CBOE Volatility Index: VIX [VIXCLS], retrieved
from FRED, Federal Reserve Bank of St. Louis,
https://fred.stlouisfed.org/series/VIXCLS, 1 August 2026.

## Repository layout

```
README.md, requirements.txt, .gitignore
src/        analysis scripts, run from the repository root (tables below)
results/    logs (the printed output of each script) and the forecast files
Graphs/     figures saved by the scripts; Graphs/2006-2026/ holds saved copies
            of the figures that main.py displays on screen
data/       not in the repository; vix_raw.csv goes here (see Data)
```

## Running the analysis

Run every script from the repository root, for example `python src/main.py`.
Each script prints its results; the logs in `results/` are those printed outputs
saved to file (for example `python src/evaluate.py > results/log.txt`). Every
script needs `data/vix_raw.csv`, so download the series first (see Data), or run
`main.py` with a key to create it. The rolling scripts
(`run_all.py`, `sens_backtest.py`, `sens975.py`, `sens_full.py`, `sens_window.py`,
`sens_window90.py`, `bench_asym.py`) refit the models on every window and take several minutes
each. Run `level975.py` before `regime_analysis.py`, whose figures read
`results/forecasts_975.csv`.

### Forecast files

| Script | What it does |
|---|---|
| `forecast_io.py` | `write_forecasts` writes a forecast table without the realised-return column; `read_forecasts` reads one and restores the returns, by date, from `data/vix_raw.csv` |

### Data check

| Script | What it does | Output |
|---|---|---|
| `compare_fred.py` | Downloads the series afresh (needs `FRED_API_KEY`) and lists every date whose value differs from `data/vix_raw.csv` or that appears in only one of the two | printed only; the download is saved as `data/vix_fred_latest.csv` |

### In-sample analysis

| Script | What it does | Output in `results/` |
|---|---|---|
| `main.py` | Data download and summary statistics; autocorrelation, Ljung-Box and cross-correlation analysis; Engle-Ng sign-bias test; GARCH and EGARCH fits and their comparison on a common sample of 4,932 returns; residual diagnostics and sub-period stability; threshold diagnostics, adopted GPD fit and the unconditional benchmark | `log_20260927_main.txt` (full output); `log_20260814_evt.txt` (its extreme value sections) |
| `verify_report_numbers.py` | Recomputes figures quoted in the dissertation that are not printed by another script | `log_20260926_report_verification.txt` |
| `sens_lag_insample.py` | Sensitivity of the model choice and tail estimates to the EGARCH lag orders | `log_20260906_sens_lag_insample.txt` |

### Threshold selection and tail estimation

These scripts share `evt_common.py`, which loads the data and fits the adopted
filter exactly as `main.py` does.

| Script | What it does | Output in `results/` |
|---|---|---|
| `evt_mrl_regions.py` | Mean residual life of the residuals: data behind each part of the plot and its slope by region | `log_20260814_mrl_regions.txt` |
| `evt_constancy.py` | Parameter-stability intervals and the thresholds at which they intersect | `log_20260814_constancy_criterion.txt` |
| `evt_threshold_sensitivity.py` | Tail quantiles and ES across candidate thresholds | `log_20260814_threshold_sensitivity.txt` |
| `evt_shape_by_threshold.py` | Shape estimate across thresholds and its linear trend | `log_20260814_threshold_plateau_quant.txt` |
| `evt_shape_inference.py` | Likelihood-ratio and Wald tests of a zero shape parameter, profile-likelihood and Wald intervals, check of the quantile formula | `log_20260926_shape_inference.txt` |
| `evt_drift_test.py` | Bootstrap test of equal shape at the 80th and 98th percentile thresholds (fixed seed) | `log_20260926_drift_test_seeded.txt` |
| `evt_gpd_fit_check.py` | Checks behind the reading of the GPD QQ plot (largest exceedances against their fitted quantiles) and of the modified-scale stability plot (intersection of its intervals) | `log_20260927_gpd_fit_check.txt` |
| `evt_t_comparison.py` | Residual upper tail against the fitted Student-t: parametric bootstrap of the GPD shape, threshold, scale and tail quantiles under the t (fixed seed) | `log_20260927_t_comparison.txt` |
| `insample_coverage.py` | In-sample VaR violation counts and coverage tests | `log_20260825_insample_coverage.txt` |
| `insample_equivalence.py` | Shows that in-sample violations of a conditional model are the residuals above the tail quantile | `log_20260825_insample_equivalence.txt` |
| `var_illustration.py` | 95% VaR and ES in calm, median and turbulent volatility states | `log_20260825_var_illustration.txt` |

### Rolling backtest

The backtest estimates five models in every window: EGARCH-EVT, GARCH-EVT,
Static POT, EGARCH-t and GARCH-t. The dissertation reports the first four.
GARCH-t is still computed and its results remain in the forecast files and logs,
but it is not reported (decision of 30 September 2026): each of the other three
comparison models differs from EGARCH-EVT in one component, and GARCH-t differs
in two.

| Script | What it does | Output |
|---|---|---|
| `backtest.py` | Estimation and one-day forecasts of all five models for one window (imported by `run_all.py` and `check.py`) | none |
| `check.py` | Quick check on the first 50 windows | printed only |
| `run_all.py` | Runs every window | `results/forecasts.csv` |
| `evaluate.py` | Kupiec, Christoffersen, variance-forecast accuracy, Diebold-Mariano and ES tests | `results/log_20260901_backtest_eval.txt` |
| `var_accuracy.py` | Pinball loss of the VaR forecasts, overall and by the sign of the previous return | `results/log_20260901_var_accuracy.txt` |
| `level975.py` | The 97.5% level (the Basel expected-shortfall level and, with 99%, the desk-level VaR backtesting level). Derives the 97.5% VaR and ES of all five models exactly from the stored forecasts, which are closed-form functions of the fitted parameters within each window (checked by recovering the stored 99.5% values), then runs the coverage, pinball/Diebold-Mariano and ES tests with the published implementations | `results/forecasts_975.csv`, `results/log_20260928_level975.txt` |
| `regime_analysis.py` | Breach ratios, pinball loss and ES shortfall by market regime at 99% (printed); its two figures also show the 97.5% values from `results/forecasts_975.csv` | `results/log_20260912_regime_analysis.txt`, `Graphs/regime_*.png` |
| `regime975.py` | The same regime breakdown at 97.5% | `results/log_20260928_regime975.txt` |
| `make_outputs.py` | VaR path figures and the numbers quoted with them (VaR ranges, violation dates, consecutive-day violations, the Chapter 1 episodes) | `Graphs/backtest_var_path*.png`, `results/log_20260927_make_outputs.txt` |
| `coherence_checks.py` | Numbers behind the cross-chapter wording of Chapters 3 and 7: window kurtosis with and without the largest rises, return s.d. and upper quantiles by window, the largest rises of the evaluation period and the index level in the month before each | `results/log_20260928_coherence_checks.txt` |
| `sens_backtest.py` | Out-of-sample sensitivity to the threshold and to the EGARCH lag orders | `results/sens_forecasts.csv`, `results/log_20260906_sens_backtest.txt` |
| `sens975.py` | The same sensitivity checks at 97.5%, for VaR and ES. Re-estimates with the code of `sens_backtest.py` (its stored file has no 97.5% level and no GPD shape, so the level cannot be derived) and checks its 95% and 99% columns against the published run | `results/sens975_forecasts.csv`, `results/log_20260928_sens975.txt` |
| `sens_full.py` | The sensitivity checks reported in Section 7.6: threshold and lag order at all four levels (95%, 97.5%, 99%, 99.5%), VaR and ES, with the Kupiec, Christoffersen, pinball/Diebold-Mariano and ES bootstrap tests of the main backtest. Re-estimates with the code of `sens_backtest.py` and `sens975.py` and checks its columns against both published runs and against `forecasts.csv` | `results/sens_full_forecasts.csv`, `results/log_20261001_sens_full.txt` |
| `compare_forecasts.py` | Compares a rerun of a backtest with the published forecast file | `results/log_20260927_backtest_rerun.txt` |

### Checks added after the independent review (2 October 2026)

| Script | What it does | Output in `results/` |
|---|---|---|
| `shock_size_asymmetry.py` | Next-day absolute return after rises and after falls within ten equal-count bins of the size of the day's return (moving-block bootstrap interval), and regressions of later magnitude on the size, the prior 20-day mean and a rise indicator, with and without their interaction, with HAC errors (Section 3.4, Appendix C) | `log_20261002_shock_size_asymmetry.txt` |
| `early_selection.py` | Repeats the BIC comparison of the three specifications, the asymmetry estimate, the Engle-Ng test and the GPD fit on the first 1,000 returns and on the returns before 6 October 2014 (Section 5.1) | `log_20261002_early_selection.txt` |
| `exceedance_clustering.py` | Consecutive exceedances, the Christoffersen independence test and the extremal index (intervals and runs estimators) for the raw returns and the residuals at the 90th, 95th and 97.5th percentiles (Section 5.3) | `log_20261002_exceedance_clustering.txt` |
| `tail_stability.py` | Residuals around 6 October 2014; GPD fits and likelihood-ratio tests of a common tail before and after that date and across the seven windows; the filter refitted on each side; EGARCH-EVT violation rates before and after (Sections 5.4 and 7.2) | `log_20261002_tail_stability.txt` |
| `forecast_uncertainty.py` | Exact and block-bootstrap intervals for the violation rates, power of the Kupiec test, bootstrap intervals for the ES discrepancy, and the GPD shape in the rolling windows (Sections 7.2 and 7.4) | `log_20261002_forecast_uncertainty.txt` |
| `bench_asym.py` | Rolling EGARCH-skewed-t and EGARCH-FHS forecasts at the four levels, same 3,933 windows as the main backtest (about five minutes with `joblib`) | `bench_asym_forecasts.csv`, `log_20261002_bench_asym.txt` |
| `eval_bench.py` | Kupiec, Christoffersen, pinball/Diebold-Mariano and ES tests of the two benchmarks against EGARCH-EVT (Section 7.5) | `log_20261002_eval_bench.txt` |
| `bench_levels.py` | Mean VaR and ES of the benchmarks relative to EGARCH-EVT, the volatility ratio of the skewed-t model to the adopted filter, and the window refitted after a non-converged default fit | `log_20261002_bench_levels.txt` |

### Window-length study (internal check, not reported in the dissertation)

| Script | What it does | Output |
|---|---|---|
| `sens_window.py` | Rolling forecasts with 1,000-day and 1,500-day windows on the same dates | `results/window_forecasts.csv` |
| `sens_window90.py` | Rolling forecasts with a 90-day window on the same dates | `results/window90_forecasts.csv` |
| `window_comparison.py` | Coverage and accuracy, 1,000 against 1,500 days | `results/log_20260912_window_comparison.txt`, `Graphs/window_regime_pinball.png` |
| `window90_comparison.py` | Estimator breakdown at 90 days and the violation rate against window length | `results/log_20260912_window90_comparison.txt`, `Graphs/window_length_curve.png` |

### Figures

`main.py` saves the extreme value diagnostic figures (`mrl_*`, `gpd_stability_*`,
`gpd_qq_*`) to `Graphs/` and displays the figures of the data analysis and the
residual QQ plot on screen; saved copies of the latter are in `Graphs/2006-2026/`.

## Reproducibility

Every log in `results/` is the printed output of the script listed for it, and
log names keep the date of the analysis. All of them were regenerated on 26 and
27 September 2026 with the package versions in `requirements.txt` on Python
3.13.13 (Linux) and compared line by line with the published files.

- **In-sample and extreme value logs:** identical, except one value in
  `log_20260814_threshold_plateau_quant.txt` that differs in its last printed
  digit (0.6803 against 0.6802; the unrounded value is 0.680250), a
  floating-point difference between platforms. The `Date` and `Time` lines of
  the model summary in `log_20260927_main.txt` change with every run.
- **Rolling backtests:** `run_all.py` and `sens_backtest.py` were rerun in full
  and compared with the published forecast files by `compare_forecasts.py`;
  the comparison and every line in which the evaluation outputs differ are in
  `log_20260927_backtest_rerun.txt`. Of the 3,933 dates, 2,925 are identical in
  every column of `forecasts.csv`, and the static model is identical on all of
  them. Elsewhere the differences come from the numerical optimiser stopping at
  slightly different points on different machines; they exceed one part in a
  thousand on at most five dates in any column. No violation changes in any of
  the 25 VaR series of the two files, so every violation count, coverage test
  and ES test is reproduced exactly. Pinball-loss means move by at most 0.0004
  (in units of 1e-4) and Diebold-Mariano statistics by at most 0.001, except
  for the EGARCH(2,1,2) alternative in `sens_backtest.py`, whose pinball losses
  move by up to 0.005 and whose Diebold-Mariano p-values move from 0.640 to
  0.663 (95% VaR) and from 0.853 to 0.860 (99% VaR). The published logs are computed from the published forecast
  files and are reproduced exactly from them.
- **Figures (27 September 2026):** the two parameter-stability figures were
  regenerated by `main.py` after a change to their axis label only (the modified
  scale is written with the dissertation's notation); the other four extreme value
  figures regenerate pixel-identical.
- **97.5% level (28 September 2026):** `level975.py` recovers the stored 99.5%
  forecasts of all five models to within 1e-13 and repeats the published ES
  bootstrap p-values exactly before computing the 97.5% results. `sens975.py`,
  run with Python 3.13.13 (Linux) and the pinned versions, reproduces the
  published sensitivity forecasts as the 27 September rerun did: 2,954 of 3,933
  dates identical, differences at optimiser tolerance on 532 dates (545 for the
  EGARCH(2,1,2) alternative), no violation changed. Its 97.5% forecasts at the
  90th-percentile threshold agree with those of `level975.py` except on the same
  tolerance dates, with no violation changed. File comparisons read both CSV
  files with exact float parsing (`float_precision="round_trip"`), so that a
  difference is a difference in the forecasts and not in CSV parsing.
  `regime_analysis.py` prints the same log as before; only its figures changed.
- **Sensitivity at all four levels (1 October 2026):** `sens_full.py`, run with
  Python 3.13.13 (Linux) and the pinned versions, reproduces the 97.5% columns of
  `sens975_forecasts.csv` exactly and the 95% and 99% columns of
  `sens_forecasts.csv` as the earlier reruns did (532 tolerance dates, 545 for the
  EGARCH(2,1,2) alternative, no violation changed). At the 90th-percentile
  threshold its forecasts at 95%, 99% and 99.5% agree with `forecasts.csv` except
  on the same 532 tolerance dates, with no violation changed.
- **Additions of 2 October 2026:** the eight scripts above were first run in one
  session on Linux with Python 3.11.15, `arch` 8.0.0, `statsmodels` 0.14.6 and
  `scipy` 1.17.1 (the pinned `scipy` is 1.18.0, and `numpy` 2.4.4 and `pandas` 3.0.2
  were used), and then rerun in a second Linux environment with Python 3.13.13 and
  the pinned versions. Both runs reproduce the full-sample estimates of the
  dissertation (GPD shape 0.1006, EGARCH asymmetry 0.2226, degrees of freedom
  4.884). Their bootstrap p-values for the ES test differ from Table 7.4 by up to
  0.012, which is the Monte Carlo error of the 10,000-draw bootstrap; the
  dissertation prints the values of Table 7.4. The logs of the five analyses other
  than the benchmarks are identical between the two environments. For the
  benchmarks (`bench_asym.py`, `eval_bench.py`, `bench_levels.py`), the stored
  `bench_asym_forecasts.csv` is from the Python 3.11 run, in which the adopted
  filter's volatility forecast equals the stored forecast exactly. In the Python
  3.13.13 rerun every violation count, coverage test and ES test is identical, and
  the pinball losses and Diebold-Mariano p-values differ by at most 0.001; the
  adopted filter's volatility forecast differs from the stored one on 150 days (by up
  to 1.6%, optimiser tolerance). `bench_asym.py` refits the skewed-t model with a
  tighter tolerance (and then from the Student-t solution) when the default fit does
  not converge; this applies to one window (2011-12-23) in both environments, and
  no window remains unconverged. The column `k_retry` of `bench_asym_forecasts.csv`
  records which fit was kept. These added analyses have not been run on the
  author's own computer.
- **Window-length study:** `window_comparison.py` reproduces its log exactly
  from the published forecast files. The log of `window90_comparison.py` and
  the two window figures were regenerated by these scripts on 27 September
  2026.

## Citation

If you use this code, please cite the dissertation it accompanies:

Sotie, V. (2026). Tail Risk for the Volatility Index (VIX): A Hybrid Extreme
Value Theory Approach. Unpublished MSc dissertation, Financial Mathematics
with Data Science, University of Bath. Supervisor: Christian Rohrbeck.

Section and chapter numbers in this README refer to that dissertation.

## References

McNeil, A. J. and Frey, R. (2000). Estimation of tail-related risk measures for
heteroscedastic financial time series: an extreme value approach. *Journal of
Empirical Finance*, 7(3-4), 271-300.
