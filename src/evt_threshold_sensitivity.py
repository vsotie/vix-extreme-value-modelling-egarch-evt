"""Sensitivity of the final residual tail quantities (z_q and ES) to the
threshold, over a grid from the 80th to the 97.5th percentile.

Run from Code/:  python src/evt_threshold_sensitivity.py
Output: results/log_20260814_threshold_sensitivity.txt
"""
import numpy as np
import pandas as pd
from evt_common import fit_adopted, load_returns, residuals, mean_excess
from utils import gpd_fit, gpd_tail_measures

GRID = [0.800, 0.825, 0.850, 0.855, 0.860, 0.870, 0.880, 0.885, 0.890,
        0.900, 0.910, 0.920, 0.925, 0.930, 0.940, 0.950, 0.960, 0.975]
LEVELS = (0.95, 0.99, 0.995)


def spread(s):
    s = s.dropna()
    return s.min(), s.max(), 100 * (s.max() - s.min()) / s.mean()


def main():
    z = residuals(fit_adopted(load_returns(verbose=False))).values
    n = len(z)
    rows = []
    for q in GRID:
        u = np.quantile(z, q)
        f = gpd_fit(mean_excess(z, u))
        tm = gpd_tail_measures(u, f["xi"], f["beta"], n, f["n"], qs=LEVELS)
        get = lambda lv, col: tm.loc[lv, col] if lv in tm.index else np.nan
        rows.append({"q": q, "u": u, "n_u": f["n"], "xi": f["xi"],
                     "se_xi": f["se_xi"], "sigma*": f["beta"] - f["xi"] * u,
                     "z_0.95": get(0.95, "z_q"), "z_0.99": get(0.99, "z_q"),
                     "z_0.995": get(0.995, "z_q"), "ES_0.99": get(0.99, "es_q")})
    cols4 = ["xi", "se_xi", "sigma*", "z_0.95", "z_0.99", "z_0.995", "ES_0.99"]
    t = pd.DataFrame(rows).set_index("q").round({"u": 3, **{c: 4 for c in cols4}})
    print("=== Sensitivity of the FINAL risk numbers to the threshold choice ===")
    print(t.to_string())
    print()
    emp = np.quantile(z, LEVELS)
    print(f"Empirical: z_0.95={emp[0]:.4f}  z_0.99={emp[1]:.4f}  z_0.995={emp[2]:.4f}")
    for title, sub in [("--- across the flat region q=0.855-0.920 ---",
                        t.loc[(t.index >= 0.855) & (t.index <= 0.920)]),
                       ("--- across the whole grid q=0.80-0.975 ---", t)]:
        print()
        print(title)
        for c in ["z_0.95", "z_0.99", "z_0.995", "ES_0.99"]:
            lo, hi, sp = spread(sub[c])
            print(f"{c}: {lo:.4f} to {hi:.4f}  (spread {sp:.1f}% of mean)")


if __name__ == "__main__":
    main()
