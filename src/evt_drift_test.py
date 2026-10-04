"""Is the rise in the GPD shape estimate between the 80th and 98th percentile
thresholds larger than sampling variation? Bootstrap of the joint distribution
of the two estimates by resampling the residual series, so that the nesting of
the two exceedance sets is reproduced in every replicate.

The seed is fixed so that the result can be reproduced. The p-value is the
share of recentred bootstrap differences at least as large in absolute value
as the observed difference.

Run from Code/:  python src/evt_drift_test.py
Output: results/log_20260926_drift_test_seeded.txt
"""
import numpy as np
from evt_common import fit_adopted, load_returns, residuals, mean_excess
from utils import gpd_fit

SEED, B = 20260814, 800


def fit_at(x, q):
    f = gpd_fit(mean_excess(x, np.quantile(x, q)))
    return f["xi"], f["se_xi"]


def main():
    z = residuals(fit_adopted(load_returns(verbose=False))).values
    n = len(z)
    (x80, s80), (x98, s98) = fit_at(z, 0.80), fit_at(z, 0.98)
    obs = x98 - x80
    print(f"Observed: xi(80th)={x80:.4f}  xi(98th)={x98:.4f}  difference={obs:+.4f}")
    print()
    rng = np.random.default_rng(SEED)
    a, b = np.empty(B), np.empty(B)
    for i in range(B):
        zb = rng.choice(z, n, replace=True)
        a[i], b[i] = fit_at(zb, 0.80)[0], fit_at(zb, 0.98)[0]
    d = b - a
    print("Bootstrap of the JOINT distribution (resampling the residual series,")
    print("so the nesting between the two estimates is reproduced in every replicate):")
    print(f"  replicates: {B}  (seed {SEED})")
    print(f"  sd(xi_80)  = {a.std(ddof=1):.4f}   (observed-information se was {s80:.4f})")
    print(f"  sd(xi_98)  = {b.std(ddof=1):.4f}   (observed-information se was {s98:.4f})")
    print(f"  correlation(xi_80, xi_98) = {np.corrcoef(a, b)[0, 1]:+.3f}")
    print(f"  sd(difference) = {d.std(ddof=1):.4f}")
    print(f"  if the two were INDEPENDENT: sqrt(sd1^2+sd2^2) = "
          f"{np.sqrt(a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2):.4f}")
    print()
    p = np.mean(np.abs(d - d.mean()) >= abs(obs))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  observed difference / sd(difference) = {obs / d.std(ddof=1):.2f}")
    print(f"  two-sided bootstrap p-value for H0: xi is the same at both thresholds = {p:.4f}")
    print(f"  95% bootstrap interval for the difference: [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
