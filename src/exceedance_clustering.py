"""Direct test of clustering in threshold exceedances, raw returns and residuals.

The Ljung-Box tests of Chapter 5 look at |z_t| and z_t^2 across all days. The tail
stage uses only the days above the threshold, so the relevant question is whether
those days arrive in clusters. For the raw returns and for the standardised residuals
of the adopted filter, and for thresholds at the 90th, 95th and 97.5th percentile of
each series, this script reports:

  * the number of exceedances that follow an exceedance on the previous day, against
    the number expected under independence, and the Christoffersen (1998) likelihood
    ratio test of first-order independence of the exceedance indicator;
  * the intervals estimator of the extremal index (Ferro and Segers, 2003), with a
    bootstrap confidence interval from resampling the inter-exceedance times. An extremal
    index of 1 means exceedances arrive as isolated events, and a value below 1 means they
    arrive in clusters of mean size 1/theta;
  * the runs estimator of the extremal index with run length 5, as a cross-check. Under
    independence this estimator is not 1 but about (1 - p)^5 (p the exceedance share),
    because exceedances less than five days apart are counted as one cluster by chance, so
    the comparison column "runs5 if indep." is the value to read it against.

Run from the repository root:
    python src/exceedance_clustering.py > results/log_20261002_exceedance_clustering.txt
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "src")
from evt_common import load_returns, fit_adopted, residuals
from evaluate import christoffersen_ind

np.random.seed(20261002)
r = load_returns(verbose=False)
z = residuals(fit_adopted(r))


def intervals_estimator(I):
    """Ferro-Segers (2003) intervals estimator of the extremal index from a 0/1 series."""
    idx = np.flatnonzero(I)
    N = len(idx)
    if N < 3:
        return np.nan
    T = np.diff(idx)
    if T.max() <= 2:
        th = 2 * T.sum() ** 2 / ((N - 1) * np.sum(T ** 2))
    else:
        th = 2 * np.sum(T - 1) ** 2 / ((N - 1) * np.sum((T - 1) * (T - 2)))
    return min(1.0, th)


def intervals_boot(I, B=2000):
    idx = np.flatnonzero(I); T = np.diff(idx); N = len(idx)
    out = []
    for _ in range(B):
        Tb = np.random.choice(T, size=len(T), replace=True)
        if Tb.max() <= 2:
            th = 2 * Tb.sum() ** 2 / ((N - 1) * np.sum(Tb ** 2))
        else:
            den = np.sum((Tb - 1) * (Tb - 2))
            th = 2 * np.sum(Tb - 1) ** 2 / ((N - 1) * den) if den > 0 else 1.0
        out.append(min(1.0, th))
    return np.percentile(out, [2.5, 97.5])


def runs_estimator(I, run=5):
    """Fraction of exceedances that start a cluster, a cluster being ended by `run` consecutive non-exceedances."""
    idx = np.flatnonzero(I)
    if len(idx) == 0:
        return np.nan
    clusters = 1 + int(np.sum(np.diff(idx) > run))
    return clusters / len(idx)


print(f"{'series':<18}{'level':>7}{'exceed.':>9}{'after exc.':>12}{'expected':>10}{'ind. LR p':>11}"
      f"{'theta (int.)':>14}{'95% CI':>18}{'theta (runs5)':>15}{'runs5 if indep.':>17}")
for name, series in (("raw returns", r.values), ("std. residuals", z.values)):
    N = len(series)
    for q in (0.90, 0.95, 0.975):
        u = np.quantile(series, q)
        I = (series > u).astype(int)
        k = int(I.sum()); pairs = int(((I[:-1] == 1) & (I[1:] == 1)).sum())
        p_hat = k / N
        expected = (N - 1) * p_hat ** 2
        lr, p, pi01, pi11 = christoffersen_ind(I)
        th = intervals_estimator(I); lo, hi = intervals_boot(I)
        thr = runs_estimator(I)
        print(f"{name:<18}{q:>7.3f}{k:>9}{pairs:>12}{expected:>10.1f}{p:>11.4f}{th:>14.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}{thr:>15.3f}{(1 - p_hat) ** 5:>17.3f}")
print("\nThe bootstrap interval is capped at 1, so [1.000, 1.000] means every resample estimated theta = 1.")
print("'after exc.': exceedances whose previous day is also an exceedance. 'expected': (N-1) times the squared exceedance share.")
print("Both series use thresholds at their own 90th, 95th and 97.5th percentile of the N values (raw returns N = %d, residuals N = %d)." % (len(r), len(z)))
