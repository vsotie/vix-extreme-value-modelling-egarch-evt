"""Checks behind the Chapter 6 reading of two EVT figures.

1. GPD QQ plot (Graphs/gpd_qq_residuals.png). Lists the largest exceedances
   of the adopted threshold against their fitted GPD quantiles, using the
   plotting positions of utils.plot_gpd_diagnostics, and counts the points
   above and below the diagonal in the upper part of the plot.
2. Parameter-stability plot (Graphs/gpd_stability_residuals.png). For the
   modified scale psi* = psi_u - xi*u, the intersection of the 95% intervals
   at all grid thresholds from each starting threshold upwards, as
   evt_constancy.py does for the shape.

Run from Code/:  python src/evt_gpd_fit_check.py
Output: results/log_20260927_gpd_fit_check.txt
"""
import numpy as np
from scipy import stats
from evt_common import fit_adopted, load_returns, residuals
from utils import gpd_fit, gpd_stability

Q_CHOSEN = 0.90
Q_GRID = np.arange(0.80, 0.981, 0.005)   # as in main.py


def main():
    z = residuals(fit_adopted(load_returns(verbose=False))).values
    n = len(z)
    u = np.quantile(z, Q_CHOSEN)
    y = np.sort(z[z > u] - u)
    k = len(y)
    f = gpd_fit(y)
    xi, psi = f["xi"], f["beta"]
    p = (np.arange(1, k + 1) - 0.5) / k
    fitted = stats.genpareto.ppf(p, xi, loc=0, scale=psi) + u
    obs = y + u

    print("Residual quantiles (ends of the plotted grids): " + "  ".join(
        f"{q:.2f}: {np.quantile(z, q):.4f}" for q in (0.50, 0.80, 0.90, 0.98, 0.99)))
    print()
    print(f"=== 1. GPD QQ plot: u = {u:.4f}, k = {k}, xi = {xi:.4f}, psi = {psi:.4f} ===")
    print(" rank  fitted  observed  obs-fitted  share of residuals at or above")
    for i in range(k - 15, k):
        print(f"{i + 1:5d}  {fitted[i]:6.3f}  {obs[i]:8.3f}  {obs[i] - fitted[i]:+10.3f}  {(k - i) / n:.5f}")
    for lo, hi in [(3.2, 4.8), (4.8, np.inf)]:
        m = (fitted > lo) & (fitted <= hi)
        print(f"fitted quantile in ({lo}, {hi}]: {m.sum()} points, {np.sum(obs[m] > fitted[m])} above "
              f"the diagonal, {np.sum(obs[m] < fitted[m])} below; largest |obs - fitted| "
              f"{np.max(np.abs(obs[m] - fitted[m])):.3f}")
    print()

    stab = gpd_stability(z, Q_GRID)
    lo_ci = stab["sigma_star"] - 1.96 * stab["se_ss"]
    hi_ci = stab["sigma_star"] + 1.96 * stab["se_ss"]
    print("=== 2. Modified scale psi* = psi_u - xi*u: intersection of 95% intervals above each u0 ===")
    for q0 in (0.80, 0.85, 0.90, 0.95):
        sel = stab.index >= q0 - 1e-9
        a, b = lo_ci[sel].max(), hi_ci[sel].min()
        verdict = "NON-EMPTY" if a <= b else "EMPTY"
        print(f"  from the {q0:.3f} quantile ({sel.sum()} grid points): [{a:+.4f}, {b:+.4f}] -> {verdict}")
    print(f"  psi* estimate: {stab['sigma_star'].iloc[0]:.4f} at q = {stab.index[0]:.3f}, "
          f"{stab.loc[0.9, 'sigma_star']:.4f} at q = 0.900, "
          f"{stab['sigma_star'].iloc[-1]:.4f} at q = {stab.index[-1]:.3f}")


if __name__ == "__main__":
    main()
