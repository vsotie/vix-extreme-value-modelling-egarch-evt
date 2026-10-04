"""Plotting utilities for VIX data-exploration diagnostics."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import requests
import os
import json


def load_vix(api_key, start, end, cache_path='data/vix_raw.csv'):
    """Load VIX closes from a local cache, downloading once if absent.

    Pinning the data to a file makes the analysis reproducible: FRED may
    revise values, and every figure in the write-up is tied to one fixed
    dataset. That file is not part of the repository (the data are not
    redistributed; see README), and no download is made while it is present;
    src/compare_fred.py compares it with a fresh download.

    Holiday entries ('.') are dropped as date-value PAIRS; dropping them
    from values alone misaligns the date index from the first holiday on.
    """
    if os.path.exists(cache_path):
        vix = pd.read_csv(cache_path, parse_dates=['Date']).set_index('Date')
        print(f"VIX loaded from cache: {cache_path} "
              f"({len(vix)} rows, {vix.index.min().date()} to "
              f"{vix.index.max().date()})")
        return vix

    if not api_key:
        raise RuntimeError(
            f"{cache_path} is missing and FRED_API_KEY is not set. Save the "
            "series from FRED at data/vix_raw.csv, or set FRED_API_KEY to download "
            "it (see README).")

    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={'series_id': 'VIXCLS', 'observation_start': start,
                'observation_end': end, 'file_type': 'json',
                'api_key': api_key},
        timeout=30,
    )
    payload = resp.json()
    if 'observations' not in payload:
        raise RuntimeError(
            f"FRED returned no observations (HTTP {resp.status_code}): "
            f"{payload.get('error_message', json.dumps(payload)[:300])}"
        )

    records = [(pd.Timestamp(o['date']), float(o['value']))
               for o in payload['observations'] if o['value'] != '.']
    vix = pd.DataFrame(records, columns=['Date', 'Close']).set_index('Date')

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    vix.to_csv(cache_path)
    print(f"VIX downloaded and cached to {cache_path} ({len(vix)} rows)")
    return vix

def engle_ng_sign_bias(z):
    """Engle-Ng (1993) sign and size bias tests on standardised residuals.

    Auxiliary OLS: z_t^2 on [const, S-, S- * z_{t-1}, S+ * z_{t-1}],
    where S- = 1{z_{t-1} < 0}. Individual t-tests per coefficient;
    joint LM = n * R^2 ~ chi^2_3 under the null of no neglected asymmetry.
    Note: chi^2_3 reference is approximate on fitted residuals
    (parameter-estimation effect).

    Parameters
    ----------
    z : 1-D array-like of standardised residuals from a fitted
        symmetric GARCH model

    Returns
    -------
    table : pd.DataFrame with coef, t, p per regressor
    lm    : float, joint LM statistic
    lm_p  : float, chi^2(3) p-value of the joint test
    """
    z = np.asarray(z, float)
    z2    = z[1:] ** 2                        # dependent: today's squared residual
    z_lag = z[:-1]                            # yesterday's residual
    s_neg = (z_lag < 0).astype(float)         # negative-shock dummy
    X = sm.add_constant(np.column_stack([
        s_neg,                                # sign bias
        s_neg * z_lag,                        # negative size bias
        (1 - s_neg) * z_lag,                  # positive size bias
    ]))
    ols = sm.OLS(z2, X).fit()

    labels = ['const', 'sign bias (phi1)',
              'neg size bias (phi2)', 'pos size bias (phi3)']
    table = pd.DataFrame({'coef': ols.params, 't': ols.tvalues,
                          'p': ols.pvalues}, index=labels)

    lm   = len(z2) * ols.rsquared
    lm_p = stats.chi2.sf(lm, 3)
    return table, lm, lm_p


def plot_vix_series(vix_close, returns, save_path=None):
    """Two-panel view of the data: levels and log returns.

    Beat 1. The level panel shows spikes and reversion to the long-run
    median; the return panel shows the same episodes on the modelling
    scale, where bursts of large moves alternate with quiet stretches.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(vix_close.index, vix_close.values, color='steelblue', lw=0.7)
    axes[0].axhline(vix_close.median(), color='black', lw=0.9, ls='--',
                    label=f'Median ({vix_close.median():.1f})')
    axes[0].set_ylabel('VIX close', fontsize=10)
    axes[0].legend(loc='upper right', fontsize=9, frameon=False)

    axes[1].plot(returns.index, returns.values, color='steelblue', lw=0.5)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_ylabel(r'$r_t=\ln(\mathrm{VIX}_t/\mathrm{VIX}_{t-1})$',
                       fontsize=10)
    axes[1].set_xlabel('Date', fontsize=10)

    for ax in axes:
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle(f'Cboe VIX, Jan 2007 - June 2026 (N={len(returns)})',
                 fontsize=13, y=0.98)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_return_distribution(returns, save_path=None):
    """Histogram against a fitted normal, and a normal QQ plot.

    Beat 2. The histogram shows the peaked centre, the QQ plot shows the
    tails, which is where the departure from normality matters.
    """
    r = np.asarray(returns.dropna(), float)
    mu, sd = r.mean(), r.std()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(r, bins=120, density=True, color='steelblue',
                 alpha=0.75, edgecolor='none')
    grid = np.linspace(r.min(), r.max(), 500)
    axes[0].plot(grid, stats.norm.pdf(grid, mu, sd), color='firebrick',
                 lw=1.4, label='Fitted normal')
    axes[0].set_xlabel('$r_t$', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].set_title('Distribution of VIX log returns', fontsize=12)
    axes[0].legend(fontsize=9, frameon=False)

    x = np.sort(r)
    p = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    theo = stats.norm.ppf(p, mu, sd)
    axes[1].scatter(theo, x, s=8, color='steelblue', alpha=0.6)
    lim = [min(theo.min(), x.min()), max(theo.max(), x.max())]
    axes[1].plot(lim, lim, color='firebrick', lw=1.2)
    axes[1].set_xlabel('Normal quantiles', fontsize=10)
    axes[1].set_ylabel('Sample quantiles', fontsize=10)
    axes[1].set_title('Normal QQ plot', fontsize=12)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_qq_student_t(z, nu, save_path=None):
    """Standardised residuals against the fitted standardised t.

    Beats 6 and 8. The fitted distribution tracks the body of the
    residuals; departure in the extreme quantiles is the reason the tail
    is treated separately with extreme value theory.
    """
    zz = np.sort(np.asarray(z.dropna(), float))
    p = (np.arange(1, len(zz) + 1) - 0.5) / len(zz)
    theo = stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)   # unit-variance t

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(theo, zz, s=8, color='seagreen', alpha=0.6)
    lim = [min(theo.min(), zz.min()), max(theo.max(), zz.max())]
    ax.plot(lim, lim, color='firebrick', lw=1.2)
    ax.set_xlabel(f'Standardised $t(\\hat\\nu={nu:.2f})$ quantiles', fontsize=10)
    ax.set_ylabel('Residual quantiles', fontsize=10)
    ax.set_title('EGARCH standardised residuals vs fitted distribution',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def window_stability(series, windows, label, lb_lag=10, q=0.95):
    """One-table sub-period check (beat 7).

    Full-sample diagnostics can average over a filter that works in calm
    markets and fails in crises. Per sub-period: dispersion, tail
    thickness, Ljung-Box p-values for remaining clustering, and the share
    of observations above the pooled upper quantile, which stays near
    1-q if the series behaves like draws from a single distribution.
    """
    s_all = series.dropna()
    thr = s_all.quantile(q)
    rows = []
    for wlabel, start, end in windows:
        s = s_all.loc[start:end]
        rows.append({
            'series': label, 'window': wlabel, 'N': len(s),
            'sd': s.std(), 'kurtosis': s.kurtosis() + 3,
            'LB |x| p': acorr_ljungbox(s.abs(), lags=[lb_lag],
                                       return_df=True).loc[lb_lag, 'lb_pvalue'],
            'LB x^2 p': acorr_ljungbox(s ** 2, lags=[lb_lag],
                                       return_df=True).loc[lb_lag, 'lb_pvalue'],
            f'share > {int(q*100)}th pct': (s > thr).mean(),
        })
    return pd.DataFrame(rows).set_index(['series', 'window'])

def iid_band(n):
    """95% white-noise band +/-1.96/sqrt(n) for correlation diagnostics."""
    return 1.96 / np.sqrt(n)


def acf_panel(ax, ac, title, color, band):
    """Single ACF panel: lag-0 dropped, flat i.i.d. band, stem display.

    Parameters
    ----------
    ax    : matplotlib Axes
    ac    : 1-D array of autocorrelations for lags 1..len(ac)
    title : panel title
    color : stem/marker colour
    band  : half-width of the 95% i.i.d. band (from iid_band(n))
    """
    lags = np.arange(1, len(ac) + 1)
    ax.axhspan(-band, band, color='0.85', zorder=0)
    ax.plot([], [], color='0.85', lw=8,
            label=f'95% i.i.d. band ($\\pm${band:.3f})')
    ax.axhline(0, color='black', lw=0.8)
    ax.vlines(lags, 0, ac, color=color, lw=1.4)
    ax.scatter(lags, ac, color=color, s=18, zorder=3)
    ax.set_ylim(min(-0.06, 1.2 * ac.min()), max(0.12, 1.2 * ac.max()))
    ax.set_xlim(0.5, len(ac) + 0.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('ACF', fontsize=10)
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    ax.grid(True, axis='y', alpha=0.3)


def plot_acf_panels(a, m, b, band, n, save_path=None):
    """Three-panel ACF figure: raw, absolute, squared log returns.

    a, m, b : autocorrelation arrays (lags 1..NLAGS) for r, |r|, r^2
    band    : half-width of the 95% i.i.d. band
    n       : sample size (figure title)
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    acf_panel(axes[0], a, 'ACF of VIX Log Returns', 'steelblue', band)
    acf_panel(axes[1], m, 'ACF of Absolute VIX Log Returns', 'seagreen', band)
    acf_panel(axes[2], b, 'ACF of Squared VIX Log Returns', 'firebrick', band)
    axes[2].set_xlabel('Lag', fontsize=10)
    fig.suptitle(f'Autocorrelation Structure of VIX Log Returns '
                 f'(Jan 2007 - June 2026, N={n})', fontsize=13, y=0.995)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def ccf(x, y, max_lag):
    """Sample CCF of x_t against y_{t+k}, k = -max_lag..+max_lag.
    Full-sample standardisation; n - |k| pairs at lag k."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    vals = {}
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            vals[k] = np.sum(x[:n - k] * y[k:]) / (n - k)
        else:
            vals[k] = np.sum(x[-k:] * y[:n + k]) / (n + k)
    return pd.Series(vals).sort_index()


def plot_ccf(ccf_series, band, n, proxy_label=r'|r_{t+k}|', save_path=None):
    """Two-sided CCF stem plot with i.i.d. band.

    ccf_series  : pd.Series indexed by lead k (negative..positive)
    band        : half-width of the 95% i.i.d. band
    n           : sample size (figure title)
    proxy_label : volatility-proxy label for axis text (LaTeX, no $)
    """
    ks = ccf_series.index.values
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhspan(-band, band, color='0.85', zorder=0)
    ax.plot([], [], color='0.85', lw=8,
            label=f'95% i.i.d. band ($\\pm${band:.3f})')
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.6, ls=':')
    ax.vlines(ks, 0, ccf_series.values, color='darkorange', lw=1.4)
    ax.scatter(ks, ccf_series.values, color='darkorange', s=18, zorder=3)
    ax.set_xlabel(f'Lead $k$ (positive: $r_t$ leading ${proxy_label}$)',
                  fontsize=10)
    ax.set_ylabel(f'$\\mathrm{{Corr}}(r_t, {proxy_label})$', fontsize=10)
    ax.set_title(f'Return-Volatility Cross-Correlation, VIX Log Returns '
                 f'(Jan 2007 - June 2026, N={n})', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig



def residual_battery(fit, lb_lags=(5, 10, 20), label=''):
    """Post-fit diagnostics on standardised residuals of an arch fit.

    LB on z (mean handled?), |z| and z^2 (clustering removed? — the i.i.d.
    check the EVT stage requires). Chi^2 reference approximate on fitted
    residuals (Li-Mak); standard practice, note in caption.

    Returns the standardised residuals for downstream use (EVT stage).
    """
    z = (fit.resid / fit.conditional_volatility).dropna()
    print(f"--- Residual battery: {label} ---")
    for name, s in {'z': z, '|z|': z.abs(), 'z^2': z**2}.items():
        print(f"Ljung-Box on {name}:")
        print(acorr_ljungbox(s, lags=list(lb_lags), return_df=True), '\n')
    return z

# =============================================================================
# EVT stage: threshold selection and GPD fitting (upper tail)
# Notation follows Coles (2001) Ch. 4 and McNeil & Frey (2000) Sec. 2.2.
# =============================================================================

def mean_residual_life(x, u_grid=None, n_min=30):
    """Empirical mean excess function over a grid of candidate thresholds.

    e_n(u) = mean(x - u | x > u). For a GPD with shape xi < 1 the
    theoretical mean excess is linear in u, so the plot should be
    approximately linear above any threshold at which the GPD
    approximation holds. 95% pointwise intervals use the normal
    approximation +/- 1.96 * sd(excesses)/sqrt(n_u).

    Parameters
    ----------
    x      : 1-D array-like, the series (upper tail analysed as given;
             negate the series first for the lower tail)
    u_grid : thresholds to evaluate; default 50 points from the 50th to
             the 99th percentile
    n_min  : drop grid points with fewer exceedances than this

    Returns
    -------
    pd.DataFrame indexed by u with n_u, mean_excess, ci_lo, ci_hi
    """
    x = np.asarray(pd.Series(x).dropna(), float)
    if u_grid is None:
        u_grid = np.linspace(np.quantile(x, 0.50), np.quantile(x, 0.99), 50)
    rows = []
    for u in u_grid:
        exc = x[x > u] - u
        if len(exc) < n_min:
            continue
        half = 1.96 * exc.std(ddof=1) / np.sqrt(len(exc))
        rows.append({'u': u, 'n_u': len(exc), 'mean_excess': exc.mean(),
                     'ci_lo': exc.mean() - half, 'ci_hi': exc.mean() + half})
    return pd.DataFrame(rows).set_index('u')


def _gpd_nll(params, y):
    """Negative log-likelihood of the GPD at (xi, beta) for excesses y."""
    xi, beta = params
    if beta <= 0:
        return np.inf
    z = 1.0 + xi * y / beta
    if np.any(z <= 0):
        return np.inf
    n = len(y)
    if abs(xi) < 1e-9:                      # exponential limit
        return n * np.log(beta) + np.sum(y) / beta
    return n * np.log(beta) + (1.0 + 1.0 / xi) * np.sum(np.log(z))


def gpd_fit(y):
    """Maximum-likelihood GPD fit to excesses y = x - u > 0.

    Standard errors come from the observed information (numerical
    Hessian of the negative log-likelihood at the MLE), the standard
    likelihood approach used by McNeil & Frey (2000, Table 1).
    Asymptotic theory requires xi > -0.5 (Smith 1985); flagged if not.

    Returns
    -------
    dict with xi, beta, se_xi, se_beta, cov (2x2), nll, n
    """
    y = np.asarray(y, float)
    from scipy.optimize import minimize

    # Candidate starting values, best first. Nelder-Mead cannot escape an
    # infeasible start (the objective is +inf on the whole simplex), so the
    # start must be checked before use and a fallback kept. This matters in
    # rolling estimation, where a short crisis window can defeat the default.
    starts = []
    try:
        xi0, _, beta0 = stats.genpareto.fit(y, floc=0)
        starts.append([xi0, beta0])
    except Exception:
        pass
    starts.append([0.1, y.mean()])            # mild heavy tail, moment-ish scale
    starts.append([0.0001, y.mean()])         # near-exponential
    starts.append([0.5, y.mean() / 2])

    best = None
    for x0 in starts:
        if not np.isfinite(_gpd_nll(x0, y)):
            continue                          # infeasible start, skip
        res = minimize(_gpd_nll, x0=x0, args=(y,), method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000})
        if res.success and np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError(
            f"GPD fit failed to converge from any start (k={len(y)}, "
            f"max excess={y.max():.4g})"
        )
    res = best
    xi, beta = res.x

    h = np.array([1e-5 * max(1, abs(xi)), 1e-5 * max(1, abs(beta))])
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i = np.eye(2)[i] * h[i]
            e_j = np.eye(2)[j] * h[j]
            H[i, j] = (_gpd_nll(res.x + e_i + e_j, y)
                       - _gpd_nll(res.x + e_i - e_j, y)
                       - _gpd_nll(res.x - e_i + e_j, y)
                       + _gpd_nll(res.x - e_i - e_j, y)) / (4 * h[i] * h[j])
    cov = np.linalg.inv(H)
    if xi <= -0.5:
        print(f"WARNING: xi = {xi:.3f} <= -0.5, MLE asymptotics unreliable")
    return {'xi': xi, 'beta': beta,
            'se_xi': np.sqrt(cov[0, 0]), 'se_beta': np.sqrt(cov[1, 1]),
            'cov': cov, 'nll': res.fun, 'n': len(y)}


def gpd_stability(x, q_grid, n_min=30):
    """GPD parameter-stability analysis across quantile-based thresholds.

    For each threshold u = quantile(x, q): fit the GPD to the excesses
    and record the shape xi and the modified scale sigma* = beta - xi*u,
    both of which are constant in u above any valid threshold (threshold
    stability property of the GPD). The sigma* standard error uses the
    delta method: Var = Var(beta) + u^2 Var(xi) - 2u Cov(xi, beta).

    Returns
    -------
    pd.DataFrame indexed by q with u, n_u, xi, se_xi, sigma_star, se_ss
    """
    x = np.asarray(pd.Series(x).dropna(), float)
    rows = []
    for q in q_grid:
        u = np.quantile(x, q)
        exc = x[x > u] - u
        if len(exc) < n_min:
            continue
        f = gpd_fit(exc)
        var_ss = (f['cov'][1, 1] + u**2 * f['cov'][0, 0]
                  - 2 * u * f['cov'][0, 1])
        rows.append({'q': round(float(q), 4), 'u': u, 'n_u': f['n'],
                     'xi': f['xi'], 'se_xi': f['se_xi'],
                     'sigma_star': f['beta'] - f['xi'] * u,
                     'se_ss': np.sqrt(max(var_ss, 0))})
    return pd.DataFrame(rows).set_index('q')


def plot_mrl(mrl_df, series_label, save_path=None):
    """Mean residual life plot with 95% pointwise intervals.

    Used for threshold selection only: the threshold is chosen as the
    lowest u above which the plot is approximately linear (Coles 2001).
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(mrl_df.index, mrl_df['ci_lo'], mrl_df['ci_hi'],
                    color='0.85', label='95% pointwise interval')
    ax.plot(mrl_df.index, mrl_df['mean_excess'], color='steelblue', lw=1.4)
    ax.set_xlabel('Threshold $u$', fontsize=10)
    ax.set_ylabel('Mean excess $e_n(u)$', fontsize=10)
    ax.set_title(f'Mean residual life plot, {series_label}', fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_gpd_stability(stab_df, series_label, save_path=None):
    """Parameter-stability plots: xi_hat and modified scale vs threshold.

    Both are constant in u wherever the GPD approximation holds; the
    threshold is chosen from the lowest region where the estimates are
    stable relative to their 95% intervals.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, col, se, label, color in [
            (axes[0], 'xi', 'se_xi', r'Shape $\hat\xi$', 'steelblue'),
            (axes[1], 'sigma_star', 'se_ss',
             r'Modified scale $\hat\psi^{*} = \hat\psi_u - \hat\xi u$', 'seagreen')]:
        ax.errorbar(stab_df['u'], stab_df[col], yerr=1.96 * stab_df[se],
                    fmt='o-', color=color, ms=3.5, lw=1.0, elinewidth=0.8,
                    capsize=2)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel('Threshold $u$', fontsize=10)
    fig.suptitle(f'GPD parameter stability, {series_label}', fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_gpd_diagnostics(exc, xi, beta, u, series_label, save_path=None):
    """QQ plot of excesses against the fitted GPD.

    One-job figure: does the fitted GPD describe the exceedances? Points
    on the diagonal say yes; systematic curvature says the threshold or
    the model is wrong.
    """
    y = np.sort(np.asarray(exc, float))
    p = (np.arange(1, len(y) + 1) - 0.5) / len(y)
    theo = stats.genpareto.ppf(p, xi, loc=0, scale=beta)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(theo + u, y + u, s=10, color='steelblue', alpha=0.7)
    lim = [u, max(theo.max(), y.max()) + u]
    ax.plot(lim, lim, color='firebrick', lw=1.2)
    ax.set_xlabel(f'Fitted GPD quantiles', fontsize=10)
    ax.set_ylabel('Empirical quantiles', fontsize=10)
    ax.set_title(f'GPD QQ plot, {series_label} '
                 f'($u$={u:.3f}, $\\hat\\xi$={xi:.3f})', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def gpd_tail_measures(u, xi, beta, n, n_u, qs=(0.95, 0.99, 0.995)):
    """Tail quantiles and expected shortfall from the GPD tail estimator.

    Inverts the McNeil-Frey (2000, Eq. 9) tail estimator:
        z_q  = u + (beta/xi) * [ ((1-q) * n/n_u)^(-xi) - 1 ]
        ES_q = (z_q + beta - xi*u) / (1 - xi)          (xi < 1)
    Valid for q > 1 - n_u/n (the quantile must lie beyond the threshold).

    Returns pd.DataFrame indexed by q with z_q and es_q.
    """
    rows = {}
    for q in qs:
        if q <= 1 - n_u / n:
            continue
        zq = u + beta / xi * (((1 - q) * n / n_u) ** (-xi) - 1)
        es = (zq + beta - xi * u) / (1 - xi) if xi < 1 else np.nan
        rows[q] = {'z_q': zq, 'es_q': es}
    return pd.DataFrame(rows).T


# =============================================================================
# Backtesting: coverage tests for VaR violation sequences
# Kupiec (1995) unconditional coverage; Christoffersen (1998) independence
# and conditional coverage. Used in-sample as a sanity check and, later, on
# the rolling out-of-sample forecasts.
# =============================================================================

def violation_tests(returns, var_forecast, p=0.05, label=''):
    """Kupiec and Christoffersen tests on a VaR violation sequence.

    A violation is returns_t > var_forecast_t (upper-tail VaR: the risk is a
    spike). Three tests:

    Kupiec unconditional coverage. H0: the violation probability equals p.
        LR_uc = -2 log[ (1-p)^(n-x) p^x / ((1-pi)^(n-x) pi^x) ],  pi = x/n
        LR_uc ~ chi2_1 under H0.
    Christoffersen independence. H0: violations are serially independent,
        against a first-order Markov alternative with transition
        probabilities pi_01, pi_11. LR_ind ~ chi2_1.
    Conditional coverage. LR_cc = LR_uc + LR_ind ~ chi2_2, testing correct
        rate AND independence jointly. This is the test a clustered
        violation sequence fails even when its overall count is right.

    Returns a dict of counts, rates and the three (statistic, p-value) pairs.
    """
    r = np.asarray(returns, float)
    v = np.asarray(var_forecast, float)
    hit = (r > v).astype(int)
    n, x = len(hit), int(hit.sum())
    pi = x / n

    # Kupiec
    if 0 < x < n:
        ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
        ll_alt = (n - x) * np.log(1 - pi) + x * np.log(pi)
        lr_uc = -2 * (ll_null - ll_alt)
    else:
        lr_uc = np.nan

    # Christoffersen independence: transition counts
    prev, cur = hit[:-1], hit[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi_all = (n01 + n11) / (n00 + n01 + n10 + n11)
    # 0*log(0) = 0 by convention. Without it the statistic is undefined
    # whenever no two violations are consecutive (n11 = 0) -- which is the
    # BEST case for independence and common in short windows, so returning
    # NaN there would silently drop the well-behaved windows.
    def _xlogy(a, b):
        return 0.0 if a == 0 else a * np.log(b)

    if 0 < pi_all < 1:
        ll_null = (_xlogy(n00 + n10, 1 - pi_all) + _xlogy(n01 + n11, pi_all))
        ll_alt = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
                  + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
        lr_ind = -2 * (ll_null - ll_alt)
    else:
        lr_ind = np.nan
    lr_cc = lr_uc + lr_ind if np.isfinite(lr_uc) and np.isfinite(lr_ind) else np.nan

    return {
        'label': label, 'n': n, 'violations': x,
        'rate': pi, 'expected_rate': p, 'expected_count': n * p,
        'LR_uc': lr_uc, 'p_uc': stats.chi2.sf(lr_uc, 1) if np.isfinite(lr_uc) else np.nan,
        'LR_ind': lr_ind, 'p_ind': stats.chi2.sf(lr_ind, 1) if np.isfinite(lr_ind) else np.nan,
        'LR_cc': lr_cc, 'p_cc': stats.chi2.sf(lr_cc, 2) if np.isfinite(lr_cc) else np.nan,
        'pi01': pi01, 'pi11': pi11,
    }
