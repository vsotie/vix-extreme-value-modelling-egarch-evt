"""How much data sits under each part of the residual mean residual life (MRL)
plot, where the mean excess is decreasing, and how its slope compares with the
slope xi/(1-xi) implied by the adopted GPD fit.

Run from Code/:  python src/evt_mrl_regions.py
Output: results/log_20260814_mrl_regions.txt
"""
import numpy as np
from evt_common import fit_adopted, load_returns, residuals, mean_excess
from utils import gpd_fit


def main():
    z = residuals(fit_adopted(load_returns(verbose=False))).values
    n = len(z)

    print("How much data sits under each part of the residual MRL plot")
    print("     u  quantile    n_u  % of sample  mean excess  CI halfwidth")
    for u in [0.0, 0.5, 0.75, 1.0, 1.283, 1.5, 2.0, 2.2, 2.5, 2.75, 3.0, 3.13]:
        e = mean_excess(z, u)
        half = 1.96 * e.std(ddof=1) / np.sqrt(len(e))
        print(f"  {u:4.2f}    {(z <= u).mean():.4f}  {len(e):5d}  "
              f"{100 * len(e) / n:10.1f}%  {e.mean():11.4f}  {half:12.4f}")
    print()

    def half(u):
        e = mean_excess(z, u)
        return 1.96 * e.std(ddof=1) / np.sqrt(len(e)), len(e)

    h1, _ = half(1.0)
    print("Width of the 95% band, relative to the band at u=1.0:")
    for u in [1.0, 1.5, 2.0, 2.5, 3.0]:
        h, k = half(u)
        print(f"  u= {u:.1f}: halfwidth {h:.4f}  =  {h / h1:.2f}x the width at u=1.0   (n_u={k})")
    print()

    k1 = len(mean_excess(z, 1.0))
    print("Overlapping-data effect: fraction of the u=1.0 exceedances still present at higher u")
    for u in [1.283, 1.5, 2.0, 2.5, 3.0]:
        k = len(mean_excess(z, u))
        print(f"  u={u:.3f}: {k:4d} obs, i.e. {100 * k / k1:5.1f}% of the u=1.0 set "
              "— the two estimates share every one of them")
    print()

    u90 = np.quantile(z, 0.90)
    xi = gpd_fit(mean_excess(z, u90))["xi"]
    slope_th = xi / (1 - xi)
    print("=== Is the low region actually DECREASING? (a positive-xi GPD forbids it) ===")
    print(f"Theory: e(u) = (beta_u0 + xi*u)/(1-xi), slope xi/(1-xi) = {slope_th:+.3f} for xi={xi:.4f}")
    prev = None
    for u in np.arange(-0.10, 1.40, 0.05):
        e = mean_excess(z, u)
        m = e.mean()
        tag = "" if prev is None else ("  up" if m > prev else "  down")
        print(f"  u={u:5.2f}  n_u={len(e):5d}  e_n(u)={m:.4f}{tag}")
        prev = m
    print()

    def slope(a, b):
        g = np.arange(a, b + 1e-9, 0.02)
        me = np.array([mean_excess(z, u).mean() for u in g])
        return np.polyfit(g, me, 1)[0], len(g)

    print(f"=== Slope of e_n(u) by region, against the theoretical {slope_th:+.3f} ===")
    for label, a, b in [("below the turning point", -0.10, 0.72),
                        ("0.75 to 1.30", 0.75, 1.29),
                        ("1.00 to 2.00", 1.00, 2.00),
                        ('2.00 to 3.10 (the "clean" part)', 2.00, 3.10)]:
        s, k = slope(a, b)
        print(f"  {label:<34}: slope = {s:+.4f}  ({k} grid points)")
    print()

    e2, e3 = mean_excess(z, 2.0), mean_excess(z, 3.0)
    h2 = 1.96 * e2.std(ddof=1) / np.sqrt(len(e2))
    h3 = 1.96 * e3.std(ddof=1) / np.sqrt(len(e3))
    rise = e3.mean() - e2.mean()
    print("=== Is the rise over 2.0-3.0 bigger than noise? ===")
    print(f"  u=2.0: e_n={e2.mean():.4f}  95% halfwidth={h2:.4f}  n_u={len(e2)}")
    print(f"  u=3.0: e_n={e3.mean():.4f}  95% halfwidth={h3:.4f}  n_u={len(e3)}")
    print(f"  observed rise over that span = {rise:+.4f}")
    print(f"  rise predicted by xi={xi:.4f}  = {slope_th * 1.0:+.4f}")
    print(f"  halfwidth at the right end   = {h3:.4f}  -> rise is "
          f"{'inside' if abs(rise - slope_th) < h3 else 'outside'} it")


if __name__ == "__main__":
    main()
