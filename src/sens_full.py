"""Out-of-sample sensitivity checks of Section 7.5 at all four levels, VaR and ES.

The earlier runs covered part of the levels only: sens_backtest.py stored VaR at
95% and 99%, and sens975.py added VaR and ES at 97.5%. This script repeats the
same run with the same code (filters, data and GPD fit come from
sens_backtest.py and sens975.py) and stores VaR and ES at 95%, 97.5%, 99% and
99.5% for
  V-a  EGARCH(1,1,1)-EVT at threshold quantiles 0.85, 0.875, 0.90, 0.925;
  V-b  EGARCH(2,1,2)-EVT at the 0.90 threshold (with its own volatility forecast,
       which standardises its ES discrepancies).
Every level is evaluated with the tests of the main backtest: Kupiec,
Christoffersen independence and conditional coverage (evaluate.py), pinball loss
with Diebold-Mariano (sens_backtest.py) and the McNeil-Frey ES bootstrap
(sens975.es_test, seed 20260901 for every test).

Run from Code/ (Python 3.13, requirements.txt):
    python src/sens_full.py > results/log_20261001_sens_full.txt
Output: results/sens_full_forecasts.csv
"""
import sys, time, platform
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
sys.path.insert(0, "src")
import sens_backtest as sb
from sens975 import tails, es_test
from evaluate import kupiec, christoffersen_ind

LEV = [0.95, 0.975, 0.99, 0.995]


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
    out["alt_sig"] = sig2 / 100.0
    t2 = tails(z2, mu2, sig2, 0.900, LEV)
    for q in LEV:
        out[f"alt_{q}"] = t2[q][0]
        out[f"altES_{q}"] = t2[q][1]
    return out


def coverage(rv, var, q):
    I = (rv > var).astype(int); x = int(I.sum()); n = len(I)
    lr_uc, p_uc = kupiec(x, n, 1 - q)
    lr_i, p_i, _, pi11 = christoffersen_ind(I)
    p_cc = 1 - stats.chi2.cdf(lr_uc + lr_i, 2)
    pairs = int(((I[:-1] == 1) & (I[1:] == 1)).sum())
    return x, p_uc, pairs, p_i, p_cc


def compare(a, b, rv):
    same = a == b
    rel = np.abs(a - b) / np.abs(a)
    return int((~same).sum()), int((rel > 1e-6).sum()), float(rel.max()), int(((rv > a) != (rv > b)).sum())


if __name__ == "__main__":
    t0 = time.time()
    rows = Parallel(n_jobs=2)(delayed(one)(i) for i in range(sb.N, sb.G))
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv("results/sens_full_forecasts.csv", index=False)
    rd = lambda f: pd.read_csv(f, parse_dates=["date"], float_precision="round_trip").sort_values("date").reset_index(drop=True)
    df = rd("results/sens_full_forecasts.csv")
    rv = df["realized"].values; n = len(rv)
    import arch, scipy
    print(f"FULL SENSITIVITY RERUN, ALL FOUR LEVELS: {n} forecasts in {(time.time()-t0)/60:.1f} min")
    print(f"Python {platform.python_version()} ({platform.system()}), arch {arch.__version__}, scipy {scipy.__version__}\n")

    print("=== 1. Checks against the published files (columns shared with this run) ===")
    print("  file / column                         days differing  rel diff > 1e-6  max rel diff  violation flips")
    pub = rd("results/sens_forecasts.csv")
    assert (pub["date"].values == df["date"].values).all()
    for c in [c for c in pub.columns if c not in ("date", "realized")]:
        d, big, mx, fl = compare(pub[c].values, df[c].values, rv)
        print(f"  sens_forecasts    {c:<20} {d:>14}  {big:>15}  {mx:>12.1e}  {fl:>15}")
    p975 = rd("results/sens975_forecasts.csv")
    for c in [c for c in p975.columns if c.endswith("0.975") or c == "sig"]:
        d, big, mx, fl = compare(p975[c].values, df[c].values, rv)
        print(f"  sens975_forecasts {c:<20} {d:>14}  {big:>15}  {mx:>12.1e}  {fl:>15}")
    main = rd("results/forecasts.csv")
    assert (main["date"].values == df["date"].values).all()
    for q in (0.95, 0.99, 0.995):
        for kind, col_main, col_here in (("VaR", f"eevt_VaR_{q}", f"base_0.900_{q}"), ("ES", f"eevt_ES_{q}", f"baseES_0.900_{q}")):
            d, big, mx, fl = compare(main[col_main].values, df[col_here].values, rv)
            print(f"  forecasts.csv     {col_main:<20} {d:>14}  {big:>15}  {mx:>12.1e}  {fl if kind == 'VaR' else '-':>15}")
    print("  (differences above 1e-6 are the optimiser-tolerance days documented in log_20260927_backtest_rerun.txt)\n")

    print("=== 2. Threshold sensitivity, EGARCH(1,1,1)-EVT, all four levels ===")
    print("  q      thr    viol   exp   Kupiec_p  pairs  ind_p   cc_p    pinball(x1e4)  ES: mean_disc  ES p")
    for q in LEV:
        for thr in sb.THRS:
            v = df[f"base_{thr:.3f}_{q}"].values
            x, p_uc, pairs, p_i, p_cc = coverage(rv, v, q)
            nD, obs, pes = es_test(rv, v, df[f"baseES_{thr:.3f}_{q}"].values, df["sig"].values)
            print(f"  {q:<6} {thr:.3f}  {x:>4}  {n*(1-q):>5.1f}  {p_uc:>8.4f}  {pairs:>5}  {p_i:.4f}  {p_cc:.4f}  "
                  f"{sb.pinball(rv, v, q).mean()*1e4:>12.4f}  {obs:>13.4f}  {pes:.4f}")
        print()

    print("=== 3. Lag structure, EGARCH(1,1,1) vs EGARCH(2,1,2), threshold 0.90, all four levels ===")
    print("  q      model     viol  Kupiec_p  pairs  ind_p   cc_p    pinball(x1e4)  ES: mean_disc  ES p   | DM(alt-base)  p")
    for q in LEV:
        vb = df[f"base_0.900_{q}"].values; va = df[f"alt_{q}"].values
        st, pdm = sb.dm(sb.pinball(rv, va, q) - sb.pinball(rv, vb, q))
        for lab, v, es, sg in (("(1,1,1)", vb, df[f"baseES_0.900_{q}"].values, df["sig"].values),
                               ("(2,1,2)", va, df[f"altES_{q}"].values, df["alt_sig"].values)):
            x, p_uc, pairs, p_i, p_cc = coverage(rv, v, q)
            nD, obs, pes = es_test(rv, v, es, sg)
            tail = f" | {st:>11.3f}  {pdm:.4f}" if lab == "(2,1,2)" else ""
            print(f"  {q:<6} {lab}  {x:>4}  {p_uc:>8.4f}  {pairs:>5}  {p_i:.4f}  {p_cc:.4f}  "
                  f"{sb.pinball(rv, v, q).mean()*1e4:>12.4f}  {obs:>13.4f}  {pes:.4f}{tail}")
    print("\n(ES p: one-sided bootstrap p-value against an ES that is too low; DM < 0 favours the alternative)")
