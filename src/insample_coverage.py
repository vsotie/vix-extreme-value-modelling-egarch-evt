"""In-sample coverage check of the VaR machinery (NOT the rolling backtest).

The AR(1)-EGARCH(1,1)-t filter and the GPD are fitted once on the whole
sample, and the in-sample one-day VaR at levels 0.95, 0.99 and 0.995 is
compared with the realised returns for three models:
  EGARCH-EVT      VaR_t = mu_t + sigma_t * z_q,  z_q from the residual GPD
  EGARCH-t        VaR_t = mu_t + sigma_t * t_q,  t_q the standardised-t quantile
  Static POT      constant VaR from a GPD fitted to the raw returns
Kupiec, Christoffersen independence and conditional-coverage tests come from
utils.violation_tests.

Run from Code/:  python src/insample_coverage.py
Output: results/log_20260825_insample_coverage.txt
"""
import numpy as np
import pandas as pd
from scipy import stats
from evt_common import fit_adopted, load_returns, residuals
from utils import gpd_fit, gpd_tail_measures, violation_tests

Q_CHOSEN = 0.90
LEVELS = (0.95, 0.99, 0.995)


def gpd_quantiles(x, n):
    """Threshold at the Q_CHOSEN quantile, GPD fit, and z_q at LEVELS."""
    u = x.quantile(Q_CHOSEN)
    f = gpd_fit((x[x > u] - u).values)
    return gpd_tail_measures(u, f["xi"], f["beta"], n=n, n_u=f["n"], qs=LEVELS)


def main():
    print("IN-SAMPLE COVERAGE CHECK  (NOT the rolling backtest.)")
    print("Parameters are fitted on the whole sample, so every model here is")
    print("flattered. Purpose: verify the machinery and see whether the")
    print("conditional-vs-unconditional contrast appears at all.")
    print()
    r = load_returns()
    fit = fit_adopted(r)
    z = residuals(fit)
    idx = z.index                                   # 4,932 dates (AR(1) loses one)
    sigma = (fit.conditional_volatility / 100).loc[idx]
    mu = ((r * 100 - fit.resid) / 100).loc[idx]     # fitted conditional mean
    rr = r.loc[idx].values
    nu = fit.params["nu"]

    tails_z = gpd_quantiles(z, n=len(z))
    tails_r = gpd_quantiles(r, n=len(r))            # static benchmark, raw returns

    for q in LEVELS:
        t_q = stats.t.ppf(q, nu) * np.sqrt((nu - 2) / nu)
        var = {"EGARCH-EVT (conditional)": mu + sigma * tails_z.loc[q, "z_q"],
               "EGARCH-t only (no EVT)": mu + sigma * t_q,
               "Static POT (unconditional)": pd.Series(tails_r.loc[q, "z_q"], index=idx)}
        rows = []
        for label, v in var.items():
            d = violation_tests(rr, v.values, p=1 - q)
            rows.append({"model": label, "viol": d["violations"],
                         "exp": round(d["expected_count"], 1),
                         "rate%": round(100 * d["rate"], 2),
                         "p_uc": round(d["p_uc"], 4), "p_ind": round(d["p_ind"], 4),
                         "p_cc": round(d["p_cc"], 4), "pi11%": round(100 * d["pi11"], 2)})
        print(f"=== VaR level {q:.3f}  (target {100 * (1 - q):.1f}% violations, n = {len(rr):,}) ===")
        print(pd.DataFrame(rows).to_string(index=False))
        print("  pi11% = P(violation tomorrow | violation today); target is the")
        print(f"  unconditional rate {100 * (1 - q):.1f}% if violations are independent.")
        print()


if __name__ == "__main__":
    main()
