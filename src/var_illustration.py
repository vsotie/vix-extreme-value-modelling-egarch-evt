"""Illustrative one-day 95% VaR and ES of the VIX in three volatility states.

Uses the in-sample AR(1)-EGARCH(1,1)-t fit and the residual GPD at the 90th
percentile. For a state with conditional volatility sigma (return units),
    VaR_0.95 = mu + sigma * z_0.95,   ES_0.95 = mu + sigma * ES_z,0.95,
with the mean term evaluated at r_t = 0, i.e. mu = constant / 100.
The states are the 10th percentile, median and 90th percentile of the fitted
conditional volatility; the three dated rows apply the same formula with the
fitted sigma_t of that day.

Run from Code/:  python src/var_illustration.py
Output: results/log_20260825_var_illustration.txt.
"""
from evt_common import fit_adopted, load_returns, residuals
from utils import gpd_fit, gpd_tail_measures

STATES = [("calm (10th pct)", 0.10), ("median", 0.50), ("turbulent (90th pct)", 0.90)]
DATES = ["2017-10-05", "2020-03-16", "2025-04-07"]


def main():
    r = load_returns()
    fit = fit_adopted(r)
    z = residuals(fit)
    u = z.quantile(0.90)
    f = gpd_fit((z[z > u] - u).values)
    tails = gpd_tail_measures(u, f["xi"], f["beta"], n=len(z), n_u=f["n"], qs=(0.95,))
    z95, es95 = tails.loc[0.95, "z_q"], tails.loc[0.95, "es_q"]
    print(f"GPD: u={u:.4f} k={f['n']} xi={f['xi']:.4f} beta={f['beta']:.4f}")
    print(f"z_0.95 = {z95:.4f}   ES_0.95 = {es95:.4f}")
    print()

    sig_all = (fit.conditional_volatility / 100).dropna()
    sig = {name: sig_all.quantile(p) for name, p in STATES}
    print("Conditional volatility, return units:")
    for name, _ in STATES:
        print(f"  {name:<23}sigma = {sig[name]:.4f}  ({100 * sig[name]:.2f}%)")
    print()

    mu = fit.params["Const"] / 100                  # AR(1) mean at r_t = 0
    print("=== VaR_0.95 and ES_0.95 by volatility state (mean term at r_t = 0) ===")
    print(f"(mu_next = {mu:.5f} = {100 * mu:.2f}%, contributes about {100 * mu:.2f} pp)")
    print(f"  {'state':<22}{'sigma':>9}{'VaR_0.95':>11}{'ES_0.95':>11}{'u in returns':>15}")
    for name, _ in STATES:
        s = sig[name]
        print(f"  {name:<22}{100 * s:>8.2f}%{100 * (mu + s * z95):>10.2f}%"
              f"{100 * (mu + s * es95):>10.2f}%{100 * (mu + s * u):>14.2f}%")
    print()

    print("=== Sanity check on real dates ===")
    for d in DATES:                                # same mean term, fitted sigma_t
        s = sig_all.loc[d]
        print(f"  {d}: sigma={100 * s:.2f}%  VaR_0.95={100 * (mu + s * z95):.2f}%")


if __name__ == "__main__":
    main()
