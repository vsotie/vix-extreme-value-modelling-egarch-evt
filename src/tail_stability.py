"""Is pooling the standardised residuals across regimes defensible in the tail?

Two questions raised on the pooling argument of Sections 3.5 and 5.4, which compared
residual standard deviations and exceedance rates at the 95th percentile but not the
shape of the tail itself:

  (a) The VIX methodology changed on 6 October 2014, when Cboe added SPX Weeklys to the
      option series used in the calculation (Cboe press release of 4 September 2014).
      Does the return series, the filter or the residual tail differ before and after?
  (b) Does the tail of the residuals differ across the seven windows of Table 3.3?

For each split the script reports, at the common threshold u (the 90th percentile of
the full-sample residuals, as in the dissertation): the exceedance count against its
expected 10% share (exact binomial test), the GPD fit in each part, a likelihood-ratio
test of a common (shape, scale) against part-specific values, a likelihood-ratio test of
a common shape with part-specific scales, and the empirical 97.5% and 99% quantiles.
The out-of-sample violation rates of EGARCH-EVT before and after 6 October 2014 are
also compared.

Run from the repository root:
    python src/tail_stability.py > results/log_20261002_tail_stability.txt
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize
sys.path.insert(0, "src")
from evt_common import load_returns, fit_adopted, residuals
from utils import gpd_fit, _gpd_nll
from evaluate import kupiec
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from forecast_io import read_forecasts, write_forecasts

warnings.filterwarnings("ignore")
SWITCH = pd.Timestamp("2014-10-06")
WINDOWS = [("2007-2009 GFC", "2007-01-01", "2009-12-31"), ("2010-2012 Euro", "2010-01-01", "2012-12-31"),
           ("2013-2016 Recovery", "2013-01-01", "2016-12-31"), ("2017-2019 Low-vol", "2017-01-01", "2019-12-31"),
           ("2020-2021 COVID", "2020-01-01", "2021-12-31"), ("2022-2024 Tightening", "2022-01-01", "2024-12-31"),
           ("2025-2026 Recent", "2025-01-01", "2026-06-30")]

r = load_returns(verbose=False)
fit = fit_adopted(r)
z = residuals(fit)                       # standardised residuals, indexed by date
u = np.quantile(z.values, 0.90)
exc_all = z.values[z.values > u] - u
gf = gpd_fit(exc_all)
print(f"Residuals: {len(z)} ({z.index[0].date()} to {z.index[-1].date()}); u = {u:.4f}; "
      f"exceedances {len(exc_all)}; GPD xi = {gf['xi']:.4f} (se {gf['se_xi']:.4f}), psi = {gf['beta']:.4f}")
print(f"Adopted filter: theta(gamma[1]) = {fit.params['gamma[1]']:.4f}, nu = {fit.params['nu']:.3f}")


def common_shape_nll(groups):
    """Minimise the total GPD negative log-likelihood with one shape and a scale per group."""
    G = len(groups)
    def f(p):
        xi = p[0]
        tot = 0.0
        for g, y in enumerate(groups):
            v = _gpd_nll([xi, np.exp(p[1 + g])], y)
            if not np.isfinite(v):
                return 1e12
            tot += v
        return tot
    best = None
    for xi0 in (gf["xi"], 0.05, 0.2):
        p0 = np.r_[xi0, [np.log(gf["beta"])] * G]
        res = minimize(f, p0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-9, "maxiter": 20000, "maxfev": 20000})
        if best is None or res.fun < best.fun:
            best = res
    return best.fun


def split_report(title, masks):
    """masks: list of (label, boolean array over z)."""
    print(f"\n=== {title} ===")
    print(f"  {'part':<22}{'n':>6}{'k>u':>6}{'exp':>7}{'binom p':>9}{'xi':>8}{'se':>7}{'psi':>8}"
          f"{'q97.5':>8}{'q99':>8}{'ES99(emp)':>10}{'k>q95':>7}{'k>q99':>7}")
    u95, u99 = np.quantile(z.values, 0.95), np.quantile(z.values, 0.99)
    exc_list, fits = [], []
    for lab, m in masks:
        zz = z.values[m]; n = len(zz); k = int((zz > u).sum())
        pb = stats.binomtest(k, n, 0.10).pvalue
        e = zz[zz > u] - u
        exc_list.append(e)
        if len(e) >= 25:
            g = gpd_fit(e); fits.append(g)
            xi_s, se_s, psi_s = f"{g['xi']:.3f}", f"{g['se_xi']:.3f}", f"{g['beta']:.3f}"
        else:
            g = None; fits.append(None); xi_s = se_s = psi_s = "n/a"
        q975, q99 = np.quantile(zz, 0.975), np.quantile(zz, 0.99)
        es99 = zz[zz >= q99].mean()
        print(f"  {lab:<22}{n:>6}{k:>6}{0.10*n:>7.1f}{pb:>9.3f}{xi_s:>8}{se_s:>7}{psi_s:>8}"
              f"{q975:>8.3f}{q99:>8.3f}{es99:>10.3f}{int((zz > u95).sum()):>7}{int((zz > u99).sum()):>7}")
    ok = [i for i, e in enumerate(exc_list) if len(e) >= 25]
    groups = [exc_list[i] for i in ok]
    if len(groups) >= 2:
        pooled = gpd_fit(np.concatenate(groups))
        sep = sum(fits[i]["nll"] for i in ok)
        lr2 = 2 * (pooled["nll"] - sep); df2 = 2 * (len(groups) - 1)
        shared = common_shape_nll(groups)
        lr_shape = 2 * (shared - sep); df_shape = len(groups) - 1
        lr_scale = 2 * (pooled["nll"] - shared); df_scale = len(groups) - 1
        print(f"  Parts with at least 25 exceedances: {len(groups)}")
        print(f"  LR, common (xi, psi) against separate:            {lr2:7.3f} on {df2:>2} df, p = {1-stats.chi2.cdf(lr2, df2):.3f}")
        print(f"  LR, common xi (separate psi) against separate:    {lr_shape:7.3f} on {df_shape:>2} df, p = {1-stats.chi2.cdf(lr_shape, df_shape):.3f}")
        print(f"  LR, common (xi, psi) against common xi:           {lr_scale:7.3f} on {df_scale:>2} df, p = {1-stats.chi2.cdf(lr_scale, df_scale):.3f}")
    counts = [int((z.values[m] > u).sum()) for _, m in masks]; ns = [int(m.sum()) for _, m in masks]
    chi = stats.chi2_contingency(np.array([counts, np.array(ns) - np.array(counts)]))
    print(f"  Chi-square test of equal exceedance rates across parts: chi2 = {chi[0]:.2f} on {chi[2]} df, p = {chi[1]:.3f}")


# ------------------------------------------------------------------ (a) October 2014
print("\n=== (a) October 2014: returns and residuals around the change ===")
vix = pd.read_csv("data/vix_raw.csv", parse_dates=["Date"]).set_index("Date")["Close"]
win = z.loc["2014-09-29":"2014-10-24"]
print(f"  {'date':<12}{'close':>8}{'log return':>12}{'std. residual':>15}")
for d in win.index:
    print(f"  {d.date()!s:<12}{vix.loc[d]:>8.2f}{r.loc[d]:>12.4f}{z.loc[d]:>15.3f}")
pre_n = z.loc[:"2014-10-03"]; post_n = z.loc["2014-10-06":]
d5 = z.loc["2014-09-29":"2014-10-03"]; a5 = z.loc["2014-10-06":"2014-10-10"]
print(f"  Residual on the first day under the new methodology (6 Oct 2014): {z.loc[SWITCH]:.3f}")
print(f"  Mean |residual|: five days before {np.abs(d5).mean():.3f}, five days from the change {np.abs(a5).mean():.3f}, whole sample {np.abs(z).mean():.3f}")
print(f"  Largest |residual| in the 20 sessions from 6 Oct 2014: {np.abs(z.loc[SWITCH:].iloc[:20]).max():.3f} on "
      f"{np.abs(z.loc[SWITCH:].iloc[:20]).idxmax().date()} (rank {int((np.abs(z) >= np.abs(z.loc[SWITCH:].iloc[:20]).max()).sum())} of {len(z)} by |residual|)")

m_pre = (z.index < SWITCH); m_post = ~m_pre
print(f"\n  Pre-change returns: {int((r.index < SWITCH).sum())}, post-change: {int((r.index >= SWITCH).sum())}")
for lab, m in (("pre", m_pre), ("post", m_post)):
    zz = z.values[m]; rr_ = r.loc[z.index].values[m]
    print(f"  {lab:<5} residuals: sd {zz.std(ddof=1):.3f}, skew {stats.skew(zz):.3f}, kurtosis (Pearson) {stats.kurtosis(zz, fisher=False):.3f}; "
          f"returns: mean {rr_.mean():+.4f}, sd {rr_.std(ddof=1):.4f}, kurtosis {stats.kurtosis(rr_, fisher=False):.3f}")
    lb = acorr_ljungbox(np.abs(zz), lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    lb2 = acorr_ljungbox(zz ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    print(f"        Ljung-Box (10 lags) p-value: |residual| {lb:.3f}, residual^2 {lb2:.3f}")

split_report("Tail of the pooled residuals before and after 6 October 2014",
             [("before 6 Oct 2014", m_pre), ("from 6 Oct 2014", m_post)])

# filter refitted separately in each part
print("\n  Filter refitted separately (AR(1)-EGARCH(1,1)-t on the returns of each part):")
for lab, sub in (("before 6 Oct 2014", r[r.index < SWITCH]), ("from 6 Oct 2014", r[r.index >= SWITCH])):
    f_ = arch_model(sub * 100, mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    zs = (f_.resid / f_.conditional_volatility).dropna().values
    us = np.quantile(zs, 0.90); es_ = zs[zs > us] - us; gs = gpd_fit(es_)
    print(f"   {lab:<18} n={len(sub):>5}  theta {f_.params['gamma[1]']:+.3f} (t {f_.tvalues['gamma[1]']:.1f})  "
          f"alpha {f_.params['alpha[1]']:.3f}  beta {f_.params['beta[1]']:.3f}  nu {f_.params['nu']:.2f}  "
          f"GPD at own 90th pct: k={len(es_)}, xi {gs['xi']:.3f} (se {gs['se_xi']:.3f})")

# out-of-sample violation rates before and after the change
print("\n  EGARCH-EVT out-of-sample violation rates before and after 6 October 2014:")
fc = read_forecasts("results/forecasts.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
f975 = pd.read_csv("results/forecasts_975.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
rv = fc["realized"].values
cols = {0.95: fc["eevt_VaR_0.95"].values, 0.975: f975["eevt_VaR_0.975"].values,
        0.99: fc["eevt_VaR_0.99"].values, 0.995: fc["eevt_VaR_0.995"].values}
pre_f = (fc["date"] < SWITCH).values
print(f"  forecasts before: {int(pre_f.sum())}, from the change: {int((~pre_f).sum())}")
print(f"  {'level':<7}{'part':<18}{'n':>6}{'viol':>6}{'exp':>7}{'rate %':>8}{'Kupiec p':>10}   Fisher exact p (equal rates)")
for q, v in cols.items():
    I = (rv > v).astype(int)
    tab = []
    for lab, m in (("before", pre_f), ("from change", ~pre_f)):
        n_ = int(m.sum()); x_ = int(I[m].sum())
        tab.append([x_, n_ - x_])
        _, pk = kupiec(x_, n_, 1 - q)
        print(f"  {q:<7}{lab:<18}{n_:>6}{x_:>6}{n_*(1-q):>7.1f}{100*x_/n_:>8.2f}{pk:>10.3f}", end="")
        print(f"   {stats.fisher_exact(tab)[1]:.3f}" if lab == "from change" else "")

# ------------------------------------------------------------------ (b) the seven windows
masks = [(lab, (z.index >= a) & (z.index <= b)) for lab, a, b in WINDOWS]
split_report("Tail of the pooled residuals across the seven windows of Table 3.3", masks)
