"""The GPD shape estimate and its 95% confidence interval at each threshold
from the 80th to the 97th percentile, compared with the adopted estimate at the
90th percentile; descriptive drift slopes; and the fit of the mean residual
life plot above several starting thresholds against the slope xi/(1-xi).

Run from Code/:  python src/evt_shape_by_threshold.py
Output: results/log_20260814_threshold_plateau_quant.txt (the file name is
historical; the dissertation does not claim a plateau).
"""
import numpy as np
import pandas as pd
from evt_common import fit_adopted, load_returns, residuals, mean_excess
from utils import gpd_fit, mean_residual_life

GRID = [round(0.80 + 0.01 * i, 2) for i in range(18)]      # 0.80 ... 0.97
DRIFT = [(0.8, 0.85), (0.8, 0.92), (0.855, 0.92), (0.8, 0.95), (0.92, 0.97)]
MRL_FROM = [0.0, 0.5, 0.75, 1.0, 1.283, 1.5]


def main():
    z = residuals(fit_adopted(load_returns(verbose=False))).values
    u_ref = np.quantile(z, 0.90)
    xi_ref = gpd_fit(mean_excess(z, u_ref))["xi"]

    rows = []
    for q in GRID:
        u = np.quantile(z, q)
        f = gpd_fit(mean_excess(z, u))
        lo, hi = f["xi"] - 1.96 * f["se_xi"], f["xi"] + 1.96 * f["se_xi"]
        rows.append({"q": q, "u": u, "n_u": f["n"], "xi": f["xi"], "se": f["se_xi"],
                     "lo": lo, "hi": hi, "ref_in_CI": bool(lo <= xi_ref <= hi),
                     "dist_in_SE": abs(f["xi"] - xi_ref) / f["se_xi"]})
    t = pd.DataFrame(rows).set_index("q")

    print(f"=== xi_hat with 95% CI; reference = q=0.90 (xi={xi_ref:.4f}) ===")
    print(t.round(4).to_string())
    print()
    inside = t[t.ref_in_CI]
    print(f"All thresholds whose 95% CI contains the chosen xi_hat={xi_ref:.4f}:")
    print(f"  q = {inside.index.min()} to {inside.index.max()}  "
          f"({len(inside)} of {len(t)} grid points)")
    print(f"  max |xi(u) - xi_ref| in SE units over q<=0.95: "
          f"{t.loc[t.index <= 0.95, 'dist_in_SE'].max():.2f} SE")
    print()

    print("=== Drift: OLS slope of xi_hat on u over sub-ranges ===")
    print("(descriptive only - the estimates are nested, so a conventional")
    print(" slope s.e. would be understated; magnitude is what matters)")
    for a, b in DRIFT:
        s = t[(t.index >= a) & (t.index <= b)]
        slope = np.polyfit(s.u, s.xi, 1)[0]
        print(f"  q in [{a},{b}]: slope = {slope:+.4f} per unit u; xi range "
              f"{s.xi.min():.4f}-{s.xi.max():.4f} (spread {s.xi.max() - s.xi.min():.4f} "
              f"vs typical se {s.se.mean():.4f})")
    print()

    print("=== MRL consistency check ===")
    print("Theory: above a valid threshold e(u) = (beta + xi*u)/(1-xi), a straight")
    print(f"line of slope xi/(1-xi). Fitted xi={xi_ref:.4f} implies slope {xi_ref / (1 - xi_ref):.4f}.")
    m = mean_residual_life(z, u_grid=np.linspace(np.quantile(z, 0.50),
                                                 np.quantile(z, 0.985), 60))
    u, e = m.index.values, m.mean_excess.values
    half = (m.ci_hi.values - m.ci_lo.values) / 2
    for u0 in MRL_FROM:
        s = u >= u0
        p = np.polyfit(u[s], e[s], 1)
        fitted = np.polyval(p, u[s])
        resid = e[s] - fitted
        r2 = 1 - np.sum(resid ** 2) / np.sum((e[s] - e[s].mean()) ** 2)
        print(f"  fit from u>={u0:.3f} (n={s.sum()} pts): slope={p[0]:+.4f}  R2={r2:.3f}  "
              f"max|resid|/CIhalfwidth={np.max(np.abs(resid) / half[s]):.2f}")


if __name__ == "__main__":
    main()
