"""Uncertainty around the backtest results (reviewer request).

For the four models and four levels of the main backtest, this script reports
  1. the violation rate with an exact (Clopper-Pearson) 95% interval and, for
     EGARCH-EVT, a circular block-bootstrap interval (block length 20) that allows
     for dependence between violation days;
  2. the power of the Kupiec test at n = 3,933: the probability of rejecting, at the
     5% level, a model whose true violation rate is a multiple of the nominal rate,
     and the smallest multiple rejected with 80% power;
  3. for the expected-shortfall test of McNeil and Frey (2000), the mean standardised
     discrepancy on violation days with a bootstrap 95% interval, the one-sided
     bootstrap p-value of the dissertation (seed 20260901), and the comparison of
     the eight primary p-values with the Bonferroni level 0.05/8.

Run from the repository root:
    python src/forecast_uncertainty.py > results/log_20261002_forecast_uncertainty.txt
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "src")
from evaluate import kupiec
from sens975 import es_test
from forecast_io import read_forecasts, write_forecasts

np.random.seed(20261002)
fc = read_forecasts("results/forecasts.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
f975 = pd.read_csv("results/forecasts_975.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
assert (fc["date"].values == f975["date"].values).all()
rv = fc["realized"].values; n = len(rv)
LEV = [0.95, 0.975, 0.99, 0.995]
MODELS = {"EGARCH-EVT": "eevt", "GARCH-EVT": "gevt", "Static POT": "s", "EGARCH-t": "et"}


def var_series(pfx, q):
    return (f975 if q == 0.975 else fc)[f"{pfx}_VaR_{q}"].values


def cp_interval(x, n, alpha=0.05):
    lo = 0.0 if x == 0 else stats.beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    return lo, hi


def circ_block_ci(I, L=20, B=5000):
    N = len(I); nb = int(np.ceil(N / L)); out = np.empty(B)
    for b in range(B):
        starts = np.random.randint(0, N, nb)
        idx = (starts[:, None] + np.arange(L)[None, :]) % N
        out[b] = I[idx.ravel()[:N]].mean()
    return np.percentile(out, [2.5, 97.5])


print(f"n = {n} forecasts, {fc['date'].min().date()} to {fc['date'].max().date()}")
print("\n=== 1. Violation rates with exact 95% intervals (percent of days) ===")
print(f"{'model':<12}{'level':>7}{'viol':>6}{'exp':>7}{'rate':>7}{'exact 95% CI':>17}{'Kupiec p':>10}{'binom 1-sided P(X>=x)':>23}")
for name, pfx in MODELS.items():
    for q in LEV:
        I = (rv > var_series(pfx, q)).astype(int); x = int(I.sum()); p0 = 1 - q
        lo, hi = cp_interval(x, n)
        _, pk = kupiec(x, n, p0)
        ps = stats.binom.sf(x - 1, n, p0)
        print(f"{name:<12}{q:>7.3f}{x:>6}{n*p0:>7.1f}{100*x/n:>7.2f}{f'[{100*lo:.2f}, {100*hi:.2f}]':>17}{pk:>10.3f}{ps:>23.3f}")
print("\nEGARCH-EVT, circular block bootstrap (block length 20, 5,000 replications):")
for q in LEV:
    I = (rv > var_series("eevt", q)).astype(float)
    lo, hi = circ_block_ci(I)
    print(f"  level {q:.3f}: rate {100*I.mean():.2f}%, interval [{100*lo:.2f}, {100*hi:.2f}]  (nominal {100*(1-q):.1f}%)")

print("\n=== 2. Power of the Kupiec test (5%% level) at n = %d ===" % n)
xs = np.arange(0, n + 1)
def lr_vec(p0):
    pi = np.clip(xs / n, 1e-300, 1 - 1e-16)
    ll_alt = (n - xs) * np.log(1 - pi) + xs * np.log(pi)
    ll_alt = np.where(xs == 0, 0.0, ll_alt)
    ll_null = (n - xs) * np.log(1 - p0) + xs * np.log(p0)
    return -2 * (ll_null - ll_alt)
crit = stats.chi2.ppf(0.95, 1)
def power(p0, p1):
    rej = lr_vec(p0) > crit
    return float(stats.binom.pmf(xs, n, p1)[rej].sum())
ratios = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
print(f"{'level':>7}{'nominal %':>11}" + "".join(f"{f'x{k}':>8}" for k in ratios) + f"{'smallest ratio, 80% power':>28}")
for q in LEV:
    p0 = 1 - q
    pw = [power(p0, min(k * p0, 0.999)) for k in ratios]
    grid = np.arange(1.01, 6.0, 0.01)
    k80 = next((k for k in grid if power(p0, k * p0) >= 0.8), np.nan)
    print(f"{q:>7.3f}{100*p0:>11.1f}" + "".join(f"{p:>8.2f}" for p in pw) + f"{k80:>28.2f}")
print("(power for a true violation rate equal to the stated multiple of the nominal rate)")

print("\n=== 3. Expected-shortfall discrepancy with bootstrap 95% intervals ===")
print(f"{'model':<12}{'level':>7}{'viol':>6}{'mean D':>9}{'sd D':>8}{'95% CI for mean D':>22}{'one-sided p':>13}")
pvals = []
for name, pfx, sg in (("EGARCH-EVT", "eevt", "e_sig"), ("GARCH-EVT", "gevt", "g_sig")):
    for q in LEV:
        src = f975 if q == 0.975 else fc
        var = src[f"{pfx}_VaR_{q}"].values; es = src[f"{pfx}_ES_{q}"].values; sig = fc[sg].values
        nD, obs, p = es_test(rv, var, es, sig)
        D = ((rv - es) / sig)[rv > var]
        bs = np.array([np.random.choice(D, len(D), replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pvals.append((name, q, p))
        print(f"{name:<12}{q:>7.3f}{nD:>6}{obs:>9.3f}{D.std(ddof=1):>8.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{p:>13.4f}")
print("\nThe p-values here use one seed per test (20260901) and can differ from Table 7.4 of the dissertation, which comes "
      "from evaluate.py and level975.py, by Monte Carlo error of the 10,000-draw bootstrap (up to about 0.012 here).")
print(f"Bonferroni level for the eight primary tests: 0.05/8 = {0.05/8:.4f}; smallest p-value {min(p for _,_,p in pvals):.4f}")
print("Tests below 0.05: " + ", ".join(f"{a} {q}: {p:.3f}" for a, q, p in pvals if p < 0.05))
print("Tests below the Bonferroni level: " + (", ".join(f"{a} {q}: {p:.3f}" for a, q, p in pvals if p < 0.05/8) or "none"))

print("\n=== 4. GPD shape estimated in the rolling windows (EGARCH-EVT, 90th-percentile threshold) ===")
xi = fc["e_xi"].values
print(f"  mean {xi.mean():.3f}, median {np.median(xi):.3f}, 5th-95th percentile [{np.percentile(xi, 5):.3f}, {np.percentile(xi, 95):.3f}], "
      f"share of windows with a negative estimate {np.mean(xi < 0):.3f}; full-sample estimate 0.1006 (Section 6.4)")
print("  mean by calendar year: " + ", ".join(f"{y} {v:.3f}" for y, v in fc.groupby(fc['date'].dt.year)['e_xi'].mean().items()))
