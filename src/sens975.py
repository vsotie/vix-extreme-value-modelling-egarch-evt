"""The 97.5% level for the out-of-sample sensitivity checks of Section 7.5.

The published sensitivity run (sens_backtest.py -> results/sens_forecasts.csv)
stored VaR at 95% and 99% only, and not the fitted GPD shape, so the 97.5% level
cannot be derived from it as it is for the main backtest (level975.py). This
script repeats the same run with the same code (filters, data and GPD fit are
imported from sens_backtest.py) and adds the 97.5% VaR and ES:
  V-a  EGARCH(1,1,1)-EVT at threshold quantiles 0.85, 0.875, 0.90, 0.925;
  V-b  EGARCH(2,1,2)-EVT at the 0.90 threshold.
The 95% and 99% columns are recomputed as a check against the published file.

Run from Code/ (Python 3.13, requirements.txt):
    python src/sens975.py > results/log_20260928_sens975.txt
Output: results/sens975_forecasts.csv
"""
import sys, time
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
sys.path.insert(0, "src")
import sens_backtest as sb
from utils import gpd_fit, gpd_tail_measures

LEV = [0.95, 0.975, 0.99]


def tails(z, mu, sig, thr, levels):
    """VaR and ES on the return scale; identical to sens_backtest.varset for VaR."""
    u = np.quantile(z, thr); exc = z[z > u] - u
    gf = gpd_fit(exc)
    tm = gpd_tail_measures(u, gf["xi"], gf["beta"], n=len(z), n_u=len(exc), qs=tuple(levels))
    return {q: ((mu + sig * tm.loc[q, "z_q"]) / 100.0, (mu + sig * tm.loc[q, "es_q"]) / 100.0) for q in levels}


def one(i):
    w100 = sb.rr[i - sb.N:i] * 100.0
    out = {"date": sb.dates[i], "realized": sb.rr[i]}
    mu, sig, z = sb.fit_egarch(w100, 1, 1, 1)
    out["sig"] = sig / 100.0
    for thr in sb.THRS:
        t = tails(z, mu, sig, thr, LEV)
        for q in LEV:
            out[f"base_{thr:.3f}_{q}"] = t[q][0]
            out[f"baseES_{thr:.3f}_{q}"] = t[q][1]
    mu2, sig2, z2 = sb.fit_egarch(w100, *sb.ALT)
    t2 = tails(z2, mu2, sig2, 0.900, LEV)
    for q in LEV:
        out[f"alt_{q}"] = t2[q][0]
    return out


def es_test(r, var, es, sig, B=10000, seed=20260901):
    """McNeil-Frey bootstrap ES test, as in evaluate.mf_es_test."""
    np.random.seed(seed)
    viol = r > var
    D = ((r - es) / sig)[viol]
    obs = D.mean(); Dc = D - obs
    bs = np.array([np.random.choice(Dc, len(D), replace=True).mean() for _ in range(B)])
    return len(D), obs, np.mean(bs >= obs)


if __name__ == "__main__":
    t0 = time.time()
    rows = Parallel(n_jobs=2)(delayed(one)(i) for i in range(sb.N, sb.G))
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv("results/sens975_forecasts.csv", index=False)
    # Compare files as files: both are read back with exact (round-trip) float parsing,
    # so that a difference is a difference in the forecasts, not in CSV parsing.
    rd = lambda f: pd.read_csv(f, parse_dates=["date"], float_precision="round_trip").sort_values("date").reset_index(drop=True)
    df = rd("results/sens975_forecasts.csv")
    rv = df["realized"].values; n = len(rv)
    print(f"97.5% SENSITIVITY RERUN: {n} forecasts in {(time.time()-t0)/60:.1f} min")
    import platform, arch, scipy
    print(f"Python {platform.python_version()} ({platform.system()}), arch {arch.__version__}, scipy {scipy.__version__}\n")

    print("=== 1. Check against the published sensitivity run (results/sens_forecasts.csv) ===")
    pub = rd("results/sens_forecasts.csv")
    assert (pub["date"].values == df["date"].values).all()
    same_all = np.ones(n, bool)
    for c in [c for c in pub.columns if c not in ("date", "realized")]:
        a, b = pub[c].values, df[c].values
        same = a == b; same_all &= same
        flips = int(((rv > a) != (rv > b)).sum())
        big = int((np.abs(a - b) / np.abs(a) > 1e-6).sum())
        print(f"  {c:<18} days differing {int((~same).sum()):>5}  rel diff > 1e-6 on {big:>3}  max rel diff {np.max(np.abs(a-b)/np.abs(a)):.1e}  violation flips {flips}")
    print(f"  days identical in every published column: {int(same_all.sum())} of {n}\n")

    print("=== 2. Check against the 97.5% forecasts derived from the main backtest (results/forecasts_975.csv) ===")
    d975 = rd("results/forecasts_975.csv")
    a, b = d975["eevt_VaR_0.975"].values, df["base_0.900_0.975"].values
    print(f"  VaR days differing {int((a != b).sum())}  rel diff > 1e-6 on {int((np.abs(a-b)/np.abs(a) > 1e-6).sum())}  max rel diff {np.max(np.abs(a-b)/np.abs(a)):.1e}  "
          f"violation flips {int(((rv > a) != (rv > b)).sum())}")
    a, b = d975["eevt_ES_0.975"].values, df["baseES_0.900_0.975"].values
    print(f"  ES  days differing {int((a != b).sum())}  rel diff > 1e-6 on {int((np.abs(a-b)/np.abs(a) > 1e-6).sum())}  max rel diff {np.max(np.abs(a-b)/np.abs(a)):.1e}")
    print("  (differences of the last binary digits come from deriving by algebra rather than computing directly;")
    print("   those above 1e-6 are the optimiser-tolerance days documented in log_20260927_backtest_rerun.txt)\n")

    print("=== 3. Threshold sensitivity at q=0.975, EGARCH(1,1,1)-EVT (exp %.1f violations) ===" % (n * 0.025))
    for thr in sb.THRS:
        v = df[f"base_{thr:.3f}_0.975"].values; x = int((rv > v).sum()); _, p = sb.kupiec(x, n, 0.025)
        nD, obs, pes = es_test(rv, v, df[f"baseES_{thr:.3f}_0.975"].values, df["sig"].values)
        print(f"  thr={thr:.3f}: viol={x} rate={100*x/n:.3f}% Kupiec_p={p:.4f} "
              f"pinball={sb.pinball(rv, v, 0.975).mean()*1e4:.4f}  ES test: mean_disc {obs:.4f} p {pes:.4f}")
    print()

    print("=== 4. Lag structure at q=0.975: EGARCH(1,1,1) vs EGARCH(2,1,2), threshold 0.90 ===")
    vb = df["base_0.900_0.975"].values; va = df["alt_0.975"].values
    xb, xa = int((rv > vb).sum()), int((rv > va).sum())
    st, p = sb.dm(sb.pinball(rv, va, 0.975) - sb.pinball(rv, vb, 0.975))
    print(f"  (1,1,1) viol={xb} pinball={sb.pinball(rv, vb, 0.975).mean()*1e4:.4f} | "
          f"(2,1,2) viol={xa} pinball={sb.pinball(rv, va, 0.975).mean()*1e4:.4f} | DM(alt-base) {st:.3f} p {p:.4f}")
