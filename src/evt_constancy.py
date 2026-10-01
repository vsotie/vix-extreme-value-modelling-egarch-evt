"""Parameter-stability check on the residual upper tail over an extended
threshold grid (down to the 40th percentile), and the intersection of the
shape-parameter confidence intervals above each candidate threshold.

The intersection test asks whether one common value of xi is consistent with
every estimate at or above a candidate threshold u0. It rules thresholds out;
it cannot select one.

Run from Code/:  python src/evt_constancy.py
Output: results/log_20260814_constancy_criterion.txt
"""
import numpy as np
import pandas as pd
from evt_common import fit_adopted, load_returns, residuals, mean_excess
from utils import gpd_fit

GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
        0.85, 0.90, 0.925, 0.95, 0.97]
U0 = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]


def main():
    z = residuals(fit_adopted(load_returns(verbose=False))).values
    rows = []
    for q in GRID:
        u = np.quantile(z, q)
        f = gpd_fit(mean_excess(z, u))
        rows.append({"q": q, "u": u, "k": f["n"], "xi": f["xi"],
                     "CI_lo": f["xi"] - 1.96 * f["se_xi"],
                     "CI_hi": f["xi"] + 1.96 * f["se_xi"]})
    t = pd.DataFrame(rows).set_index("q")

    print("=== The horizontal-line criterion, applied to an EXTENDED grid ===")
    print("(extend below the 80th percentile, where the MRL already says the GPD fails,")
    print(" to see whether the stability plot can detect the violation at all)")
    print()
    print(t.round({"u": 3, "xi": 4, "CI_lo": 4, "CI_hi": 4}).to_string())
    print()
    print("=== Existence of a common value: intersection of all CIs above each candidate u0 ===")
    for q0 in U0:
        sub = t[t.index >= q0 - 1e-12]
        lo, hi = sub["CI_lo"].max(), sub["CI_hi"].min()
        verdict = ("NON-EMPTY: constant value admissible" if lo <= hi
                   else "EMPTY: constancy REJECTED")
        print(f"  u0 at {q0:.2f} pctile: intersection of CIs above it = "
              f"[{lo:+.4f},{hi:+.4f}]  ->  {verdict}")


if __name__ == "__main__":
    main()
