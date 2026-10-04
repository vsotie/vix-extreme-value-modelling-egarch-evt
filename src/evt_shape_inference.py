"""Inference on the GPD shape parameter of the residual upper tail at the
adopted threshold (90th percentile): likelihood-ratio test of xi = 0, Wald
test, profile-likelihood and Wald confidence intervals, and a check that the
tail quantile formula equals Coles (2001) Eq. (4.13).

The endpoints of the profile-likelihood interval are found by root-finding on
the profile deviance.

Run from Code/:  python src/evt_shape_inference.py
Output: results/log_20260926_shape_inference.txt
"""
import numpy as np
from scipy import optimize, stats
from evt_common import fit_adopted, load_returns, residuals, mean_excess
from utils import _gpd_nll, gpd_fit, gpd_tail_measures


def main():
    z = residuals(fit_adopted(load_returns(verbose=False))).values
    n = len(z)
    u = np.quantile(z, 0.90)
    y = mean_excess(z, u)
    k = len(y)
    f = gpd_fit(y)
    xi, beta, se = f["xi"], f["beta"], f["se_xi"]
    ll_gpd = -f["nll"]
    ll_exp = -(k * np.log(y.mean()) + k)              # exponential MLE
    dev = 2 * (ll_gpd - ll_exp)

    print(f"GPD (Coles Thm 4.1): u={u:.4f}  k={k}  xi={xi:.4f} (se {se:.4f})  sigma={beta:.4f}")
    print()
    print("LR test H0: xi=0 (exponential excesses, Gumbel domain)")
    print(f"  ll(GPD)={ll_gpd:.4f}  ll(exp)={ll_exp:.4f}  D={dev:.4f}  p={stats.chi2.sf(dev, 1):.4f}")
    t = xi / se
    print(f"  Wald: t = {t:.3f}, p = {2 * stats.norm.sf(abs(t)):.4f}")

    def profile_dev(x):
        r = optimize.minimize_scalar(lambda b: _gpd_nll([x, b], y), bounds=(1e-6, 10),
                                     method="bounded", options={"xatol": 1e-12})
        return 2 * (ll_gpd + r.fun)

    crit = stats.chi2.ppf(0.95, 1)
    lo = optimize.brentq(lambda x: profile_dev(x) - crit, xi - 0.3, xi)
    hi = optimize.brentq(lambda x: profile_dev(x) - crit, xi, xi + 0.4)
    print(f"  95% profile-likelihood CI for xi: [{lo:.4f}, {hi:.4f}]")
    print(f"  95% Wald CI for xi:               [{xi - 1.96 * se:.4f}, {xi + 1.96 * se:.4f}]")
    print()
    print("=== Formula check: my z_q vs Coles Eq (4.13) x_m = u + (sigma/xi)[(m*zeta_u)^xi - 1] ===")
    tm = gpd_tail_measures(u, xi, beta, n, k)
    for q in (0.95, 0.99, 0.995):
        m, zeta = 1 / (1 - q), k / n
        coles = u + (beta / xi) * ((m * zeta) ** xi - 1)
        mine = tm.loc[q, "z_q"]
        print(f"  q={q}: mine={mine:.6f}  Coles(4.13) with m=1/(1-q)={coles:.6f}  diff={abs(mine - coles):.2e}")


if __name__ == "__main__":
    main()
