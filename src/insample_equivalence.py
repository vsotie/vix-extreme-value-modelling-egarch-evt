"""Why the in-sample coverage test of a conditional model adds no evidence.

For a conditional model VaR_t = mu_t + sigma_t * z_q, and the standardised
residual is z_hat_t = (r_t - mu_t) / sigma_t with sigma_t > 0, so
    r_t > VaR_t   <=>   z_hat_t > z_q.
Counting in-sample VaR violations is therefore counting residuals above z_q,
which is the residual quantile comparison already reported. This script
checks the identity day by day.

Run from Code/:  python src/insample_equivalence.py
Output: results/log_20260825_insample_equivalence.txt
"""
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from evt_common import fit_adopted, load_returns, residuals
from utils import gpd_fit, gpd_tail_measures

LEVELS = (0.95, 0.99, 0.995)


def main():
    print("Is the in-sample violation count anything other than a restatement of")
    print("the residual quantile comparison?  Claim: for a CONDITIONAL model,")
    print("   r_t > mu_t + sigma_t * z_q   <=>   z_hat_t > z_q")
    print("so counting VaR violations IS counting residuals above z_q.")
    print()
    r = load_returns()
    fit = fit_adopted(r)
    z = residuals(fit)
    idx = z.index
    sigma = (fit.conditional_volatility / 100).loc[idx]
    mu = ((r * 100 - fit.resid) / 100).loc[idx]
    rr = r.loc[idx]

    u = z.quantile(0.90)
    f = gpd_fit((z[z > u] - u).values)
    tails = gpd_tail_measures(u, f["xi"], f["beta"], n=len(z), n_u=f["n"], qs=LEVELS)

    for q in LEVELS:
        z_q = tails.loc[q, "z_q"]
        hit_var = (rr > mu + sigma * z_q).values
        hit_res = (z > z_q).values
        print(f"q={q}:  VaR violations = {hit_var.sum()}   residuals above z_q = "
              f"{hit_res.sum()}   identical: {np.array_equal(hit_var, hit_res)}")
        print(f"        z_q(GPD) = {z_q:.4f}  vs empirical residual quantile "
              f"{z.quantile(q):.4f}   -> rate {100 * hit_var.mean():.2f}% vs target "
              f"{100 * (1 - q):.1f}%")

    lb = acorr_ljungbox(np.abs(r), lags=[5, 10, 20])
    print()
    print("CONCLUSION: for the conditional models the in-sample coverage test is")
    print("algebraically the SAME statistic as the residual quantile comparison")
    print("already reported. It cannot disagree with it. No independent evidence.")
    print()
    print("The static benchmark is NOT equivalent (constant VaR vs actual returns),")
    print("but its clustered violations follow from volatility clustering, which")
    print(f"Ch. 3 already establishes (Ljung-Box on |r| at lags 5, 10, 20, "
          f"p <= {lb['lb_pvalue'].max():.1e}).")


if __name__ == "__main__":
    main()
