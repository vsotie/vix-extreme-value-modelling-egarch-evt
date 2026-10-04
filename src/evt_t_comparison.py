"""The residual upper tail against the fitted Student-t of the first stage.

Three questions, all about the standardised residuals of the adopted
AR(1)-EGARCH(1,1)-t fit and the GPD fitted above their 90th percentile:

1. Shape. The Student-t with nu degrees of freedom has GPD shape 1/nu. How
   does the residual estimate compare, allowing for the uncertainty in both?
2. What a GPD fitted at this threshold would report if the residuals really
   followed the fitted t. A parametric bootstrap draws samples of the same
   size from the standardised t(nu_hat), applies the same threshold rule and
   GPD fit, and records the shape estimate and the tail quantiles. This gives
   the sampling distribution of each statistic under the hypothesis that the
   fitted t is the true innovation distribution (nu held at nu_hat; the
   estimation of the filter is not re-simulated).
3. Sensitivity to nu. The bootstrap is repeated with nu at each end of its
   95% Wald interval.

Run from Code/:  python src/evt_t_comparison.py
Output: results/log_20260927_t_comparison.txt
"""
import numpy as np
from scipy import stats
from evt_common import fit_adopted, load_returns, residuals
from utils import gpd_fit, gpd_tail_measures

LEVELS = (0.95, 0.99, 0.995)
B = 2000
SEED = 20260927


def t_quantile(q, nu):
    """Quantile of the Student-t scaled to unit variance."""
    return stats.t.ppf(q, nu) * np.sqrt((nu - 2) / nu)


def gpd_quantile(q, u, xi, psi, n, k):
    """GPD tail estimator of the q-quantile (McNeil and Frey, 2000, Eq. 10)."""
    return u + psi / xi * (((1 - q) * n / k) ** (-xi) - 1)


def fit_tail(x):
    u = np.quantile(x, 0.90)
    y = x[x > u] - u
    f = gpd_fit(y)
    return u, f, len(y)


def bootstrap_t(nu, n, reps, seed):
    """Samples of size n from the unit-variance t(nu); for each, the threshold,
    GPD shape and scale, GPD tail quantiles and empirical quantiles."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt((nu - 2) / nu)
    us, xs, ps, zq, eq = [], [], [], [], []
    for _ in range(reps):
        x = rng.standard_t(nu, size=n) * scale
        ub, fb, kb = fit_tail(x)
        us.append(ub); xs.append(fb["xi"]); ps.append(fb["beta"])
        zq.append([gpd_quantile(q, ub, fb["xi"], fb["beta"], n, kb) for q in LEVELS])
        eq.append(np.quantile(x, LEVELS))
    return tuple(np.array(v) for v in (us, xs, ps, zq, eq))


def main():
    fit = fit_adopted(load_returns(verbose=False))
    z = residuals(fit).values
    n = len(z)
    nu, se_nu = fit.params["nu"], fit.std_err["nu"]
    u, f, k = fit_tail(z)
    xi, se_xi, psi = f["xi"], f["se_xi"], f["beta"]

    print("=== 1. Shape: residual GPD against the fitted Student-t ===")
    print(f"residual GPD (u = {u:.4f}, k = {k}): xi = {xi:.4f} (se {se_xi:.4f})")
    print(f"fitted t: nu = {nu:.4f} (se {se_nu:.3f}); implied shape 1/nu = {1 / nu:.4f} "
          f"(delta-method se {se_nu / nu ** 2:.4f})")
    lo, hi = 1 / (nu + 1.96 * se_nu), 1 / (nu - 1.96 * se_nu)
    print(f"95% interval for 1/nu from the Wald interval for nu: [{lo:.4f}, {hi:.4f}]")
    print("95% profile-likelihood interval for xi: see log_20260926_shape_inference.txt")
    print()

    tails = gpd_tail_measures(u, xi, psi, n=n, n_u=k, qs=LEVELS)
    us, xs, ps, zq, eq = bootstrap_t(nu, n, B, SEED)

    print(f"=== 2. Parametric bootstrap under the fitted t (B = {B}, n = {n}, seed {SEED}) ===")
    print("GPD shape fitted above the 90th percentile of t samples:")
    print(f"  mean {xs.mean():.4f}  sd {xs.std(ddof=1):.4f}  "
          f"2.5-97.5% [{np.quantile(xs, 0.025):.4f}, {np.quantile(xs, 0.975):.4f}]")
    print(f"  share of replicates with xi <= observed {xi:.4f}: {np.mean(xs <= xi):.4f}")
    print("Threshold and scale (same fits):")
    print(f"  threshold u: t-boot mean {us.mean():.4f} [{np.quantile(us, 0.025):.4f}, {np.quantile(us, 0.975):.4f}]"
          f"   residuals {u:.4f}")
    print(f"  GPD scale:   t-boot mean {ps.mean():.4f} [{np.quantile(ps, 0.025):.4f}, {np.quantile(ps, 0.975):.4f}]"
          f"   residuals {psi:.4f}")
    print()
    print("Upper-tail quantiles: observed residuals against the bootstrap under the t")
    print(f"{'q':>6} {'fitted t':>9} {'obs GPD':>8} {'t-boot GPD mean':>16} {'2.5-97.5%':>18} "
          f"{'P(>=obs)':>9} | {'obs emp':>8} {'t-boot emp 2.5-97.5%':>21} {'P(>=obs)':>9}")
    for j, q in enumerate(LEVELS):
        og, oe = tails.loc[q, "z_q"], np.quantile(z, q)
        print(f"{q:>6} {t_quantile(q, nu):>9.4f} {og:>8.4f} {zq[:, j].mean():>16.4f} "
              f"[{np.quantile(zq[:, j], 0.025):.4f}, {np.quantile(zq[:, j], 0.975):.4f}] "
              f"{np.mean(zq[:, j] >= og):>9.4f} | {oe:>8.4f} "
              f"   [{np.quantile(eq[:, j], 0.025):.4f}, {np.quantile(eq[:, j], 0.975):.4f}] "
              f"{np.mean(eq[:, j] >= oe):>9.4f}")
    print()

    print(f"=== 3. Sensitivity: nu at each end of its 95% interval (B = 500 each) ===")
    for nu_s in (nu - 1.96 * se_nu, nu + 1.96 * se_nu):
        _, xs_s, _, zq_s, _ = bootstrap_t(nu_s, n, 500, SEED)
        worst = max(np.mean(zq_s[:, j] >= tails.loc[q, "z_q"]) for j, q in enumerate(LEVELS))
        print(f"  nu = {nu_s:.3f}: xi mean {xs_s.mean():.4f}, share <= observed {np.mean(xs_s <= xi):.4f}; "
              f"largest P(GPD quantile >= observed) over the three levels {worst:.4f}")

if __name__ == "__main__":
    main()
