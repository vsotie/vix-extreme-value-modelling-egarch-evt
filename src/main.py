"""VIX data exploration and EGARCH estimation.
Window: Jan 2007 - June 2026 (proposal Phase II). Source: FRED VIXCLS."""

import os
import requests
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from scipy import stats
from utils import (load_vix, iid_band, plot_vix_series, plot_return_distribution,
                   plot_acf_panels, ccf, plot_ccf, engle_ng_sign_bias,
                   residual_battery, window_stability, plot_qq_student_t,
                   mean_residual_life, gpd_fit, gpd_stability, plot_mrl,
                   plot_gpd_stability, plot_gpd_diagnostics, gpd_tail_measures)

# =============================================================================
# 0. Parameters
# =============================================================================
LB_LAGS     = [5, 10, 20]   # Ljung-Box horizons ~ week / fortnight / month
NLAGS       = 30            # lags shown in ACF panels
CCF_MAX_LAG = 25            # CCF computed for k = -25..+25
# FRED API key, read from the environment and not stored in the repository
# (free key: https://fred.stlouisfed.org/docs/api/api_key.html). It can stay unset:
# a copy of the series saved at data/vix_raw.csv is used whenever present (the data
# are not part of the repository; see README). The key is needed only to download
# the series from FRED.
API_KEY     = os.environ.get("FRED_API_KEY")

WINDOWS = [
    ('2007-2009 GFC',        '2007-01-01', '2009-12-31'),
    ('2010-2012 Euro',       '2010-01-01', '2012-12-31'),
    ('2013-2016 Recovery',   '2013-01-01', '2016-12-31'),
    ('2017-2019 Low-vol',    '2017-01-01', '2019-12-31'),
    ('2020-2021 COVID',      '2020-01-01', '2021-12-31'),
    ('2022-2024 Tightening', '2022-01-01', '2024-12-31'),
    ('2025-2026 Recent',     '2025-01-01', '2026-06-30')
]

# =============================================================================
# 1. Data pull and processing
#    Holiday ('.') entries dropped as date-value PAIRS (date-alignment fix)
# =============================================================================
vix = load_vix(API_KEY, '2007-01-01', '2026-06-30')

vix_returns = np.log(vix['Close'] / vix['Close'].shift(1)).dropna()
G = len(vix_returns)
band = iid_band(G)

# =============================================================================
# 2. Summary statistics and the data itself (beats 1-2)
# =============================================================================
stats_summary = pd.Series({
    'Mean':                         vix_returns.mean(),
    'Std Dev':                      vix_returns.std(),
    'Skewness':                     vix_returns.skew(),
    'Excess kurtosis (Fisher)':     vix_returns.kurtosis(),
    'Kurtosis (Pearson, normal=3)': vix_returns.kurtosis() + 3,
    'Min':                          vix_returns.min(),
    'Max':                          vix_returns.max(),
    'N':                            G,
})
print("=== Summary statistics ===")
print(stats_summary, '\n')

plot_vix_series(vix['Close'], vix_returns)
plot_return_distribution(vix_returns)

# =============================================================================
# 3. Autocorrelation structure (beat 3)
#    a(s): raw returns | m(s): absolute returns | b(s): squared returns
# =============================================================================
a = acf(vix_returns,       nlags=NLAGS, fft=True)[1:]
m = acf(vix_returns.abs(), nlags=NLAGS, fft=True)[1:]
b = acf(vix_returns**2,    nlags=NLAGS, fft=True)[1:]

acf_table = pd.DataFrame({'a(s) raw': a, 'm(s) abs': m, 'b(s) sq': b},
                         index=pd.RangeIndex(1, NLAGS + 1, name='s'))
print(f"=== ACF values, lags 1-10 (95% band +/-{band:.4f}) ===")
print(acf_table.head(10).round(4), '\n')

plot_acf_panels(a, m, b, band, G)

for name, s in {'r': vix_returns,
                '|r|': vix_returns.abs(),
                'r^2': vix_returns**2}.items():
    print(f"=== Ljung-Box on {name} ===")
    print(acorr_ljungbox(s, lags=LB_LAGS, return_df=True), '\n')

# =============================================================================
# 4. Return-volatility cross-correlation (beat 4)
#    Positive values at k >= 1: inverse leverage effect -> EGARCH over GARCH.
# =============================================================================
ccf_abs = ccf(vix_returns, vix_returns.abs(), CCF_MAX_LAG)
print("=== CCF r_t vs |r_{t+k}|, k = -5..5 ===")
print(ccf_abs.loc[-5:5].round(4).to_frame('ccf'), '\n')

plot_ccf(ccf_abs, band, G)

# =============================================================================
# 5. Model estimation (beat 5)
#    Symmetric GARCH benchmark also supplies residuals for the sign-bias test,
#    which is cited in one line only.
# =============================================================================
garch_fit = arch_model(vix_returns * 100, mean='Constant',
                       vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
z_garch = (garch_fit.resid / garch_fit.conditional_volatility).dropna()
_, sb_lm, sb_p = engle_ng_sign_bias(z_garch.values)
print(f"Engle-Ng joint sign-bias: LM = {sb_lm:.3f}, p = {sb_p:.4f}\n")

egarch_ar1 = arch_model(vix_returns * 100, mean='AR', lags=1,
                        vol='EGARCH', p=1, o=1, q=1, dist='t').fit(disp='off')
print("=== EGARCH(1,1)-t, AR(1) mean (adopted model) ===")
print(egarch_ar1.summary(), '\n')

# Model comparison on a common sample (Table 5.1 of the report). The AR(1)
# model loses the first return to its lag and is fitted to 4,932 returns, so
# the constant-mean models are refitted to the same 4,932 returns; otherwise
# their log-likelihoods and BICs are not comparable. The sign-bias test above
# keeps the full-sample GARCH fit.
r_common = vix_returns.iloc[1:]
garch_cmp = arch_model(r_common * 100, mean='Constant',
                       vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
egarch_const = arch_model(r_common * 100, mean='Constant',
                          vol='EGARCH', p=1, o=1, q=1, dist='t').fit(disp='off')

comparison = pd.DataFrame({
    'LL':     [garch_cmp.loglikelihood, egarch_const.loglikelihood,
               egarch_ar1.loglikelihood],
    'AIC':    [garch_cmp.aic, egarch_const.aic, egarch_ar1.aic],
    'BIC':    [garch_cmp.bic, egarch_const.bic, egarch_ar1.bic],
    'params': [garch_cmp.num_params, egarch_const.num_params,
               egarch_ar1.num_params],
    'N':      [garch_cmp.nobs, egarch_const.nobs, egarch_ar1.nobs],
}, index=['GARCH(1,1)-t const', 'EGARCH(1,1)-t const', 'EGARCH(1,1)-t AR(1)'])
print("=== Model comparison (common sample; log-likelihoods on the x100 scale) ===")
print(comparison.round(1), '\n')

# =============================================================================
# 6. Residual diagnostics of the adopted model (beats 6, 8)
# =============================================================================
z_ar1 = residual_battery(egarch_ar1, LB_LAGS, 'EGARCH-t AR(1) mean')

print("=== Residual ACF, lags 1-10 ===")
print(pd.Series(acf(z_ar1, nlags=10, fft=True)[1:],
                index=range(1, 11)).round(4), '\n')

plot_qq_student_t(z_ar1, egarch_ar1.params['nu'])

# =============================================================================
# 7. Sub-period stability (beat 7)
#    Raw returns vs standardised residuals: the contrast is the result.
# =============================================================================
print("=== Sub-period stability ===")
print(pd.concat([
    window_stability(vix_returns, WINDOWS, 'raw returns'),
    window_stability(z_ar1,       WINDOWS, 'residuals'),
]).round(4), '\n')
# =============================================================================
# 8. EVT stage: threshold selection and GPD fit on the residual upper tail
#    (Objective II). Upper tail only: the risk measures are one-day VaR and
#    ES in the spike direction, and the residual QQ shows the lower tail is
#    already thinner than the fitted t. For one-day measures McNeil & Frey
#    (2000) likewise fit a single tail.
# =============================================================================
Q_GRID   = np.arange(0.80, 0.981, 0.005)  # candidate threshold quantiles
Q_CHOSEN = 0.90   # McNeil & Frey's convention: a 10% tail fraction (their
                  # k = 100 of n = 1000), giving n_u = 494 here. The diagnostics
                  # bound the admissible region but do not select a point in it:
                  # below about u = 0.75 (the 80th percentile) the MRL shows no
                  # rise where a positive-shape GPD would rise, and from there
                  # up it is close to linear, so thresholds from the 80th
                  # percentile up are admissible; the parameter-stability
                  # intervals rule out nothing above the 50th percentile
                  # (results/log_20260814_mrl_regions.txt,
                  # results/log_20260814_constancy_criterion.txt).
                  # The 90th percentile lies inside that region.

mrl_z = mean_residual_life(z_ar1)
plot_mrl(mrl_z, 'EGARCH standardised residuals',
         save_path='Graphs/mrl_residuals.png')

stab_z = gpd_stability(z_ar1, Q_GRID)
plot_gpd_stability(stab_z, 'EGARCH standardised residuals',
                   save_path='Graphs/gpd_stability_residuals.png')

print("=== GPD fits at candidate thresholds (residual upper tail) ===")
print(stab_z.loc[[0.85, 0.90, 0.925, 0.95]].round(4), '\n')

u_z   = z_ar1.quantile(Q_CHOSEN)
exc_z = z_ar1[z_ar1 > u_z] - u_z
fit_z = gpd_fit(exc_z.values)
print(f"=== Adopted GPD fit, residuals: u = {u_z:.4f} "
      f"(q = {Q_CHOSEN}), n_u = {fit_z['n']} ===")
print(f"xi   = {fit_z['xi']:.4f}  (se {fit_z['se_xi']:.4f})")
print(f"beta = {fit_z['beta']:.4f}  (se {fit_z['se_beta']:.4f})\n")

plot_gpd_diagnostics(exc_z.values, fit_z['xi'], fit_z['beta'], u_z,
                     'EGARCH standardised residuals',
                     save_path='Graphs/gpd_qq_residuals.png')

tails_z = gpd_tail_measures(u_z, fit_z['xi'], fit_z['beta'],
                            n=len(z_ar1), n_u=fit_z['n'])
nu_hat = egarch_ar1.params['nu']
t_q = stats.t.ppf(tails_z.index, nu_hat) * np.sqrt((nu_hat - 2) / nu_hat)
tails_z['t_q (fitted t)'] = t_q
tails_z['empirical'] = [z_ar1.quantile(q) for q in tails_z.index]
print("=== Residual tail quantiles: GPD vs fitted t vs empirical ===")
print("(z_q feeds VaR_{t+1} = mu_{t+1} + sigma_{t+1} * z_q; "
      "ES analogous)")
print(tails_z.round(4), '\n')

# =============================================================================
# 9. Benchmark: unconditional POT/GPD on raw returns (same apparatus).
#    Same threshold quantile, chosen symmetrically: xi is stable over
#    u ~ 0.05-0.12 and the MRL is approximately linear above u ~ 0.06.
# =============================================================================
mrl_r = mean_residual_life(vix_returns)
plot_mrl(mrl_r, 'raw VIX log returns', save_path='Graphs/mrl_raw.png')

stab_r = gpd_stability(vix_returns, Q_GRID)
plot_gpd_stability(stab_r, 'raw VIX log returns',
                   save_path='Graphs/gpd_stability_raw.png')

print("=== GPD fits at candidate thresholds (raw-return upper tail) ===")
print(stab_r.loc[[0.85, 0.90, 0.925, 0.95]].round(4), '\n')

u_r   = vix_returns.quantile(Q_CHOSEN)
exc_r = vix_returns[vix_returns > u_r] - u_r
fit_r = gpd_fit(exc_r.values)
print(f"=== Benchmark GPD fit, raw returns: u = {u_r:.4f} "
      f"(q = {Q_CHOSEN}), n_u = {fit_r['n']} ===")
print(f"xi   = {fit_r['xi']:.4f}  (se {fit_r['se_xi']:.4f})")
print(f"beta = {fit_r['beta']:.4f}  (se {fit_r['se_beta']:.4f})\n")

plot_gpd_diagnostics(exc_r.values, fit_r['xi'], fit_r['beta'], u_r,
                     'raw VIX log returns',
                     save_path='Graphs/gpd_qq_raw.png')

tails_r = gpd_tail_measures(u_r, fit_r['xi'], fit_r['beta'],
                            n=len(vix_returns), n_u=fit_r['n'])
tails_r['empirical'] = [vix_returns.quantile(q) for q in tails_r.index]
print("=== Static return-scale tail measures (unconditional benchmark) ===")
print(tails_r.round(4), '\n')
