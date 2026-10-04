"""Evaluation of the asymmetric-tail benchmarks (FHS, EGARCH-skewed-t) against EGARCH-EVT.

Reads results/bench_asym_forecasts.csv (bench_asym.py) and the main forecast files.
For each model and level: violations, Kupiec, Christoffersen independence and
conditional coverage, mean pinball loss with the Diebold-Mariano test against
EGARCH-EVT (Newey-West, 5 lags; negative statistic favours the benchmark), and the
McNeil-Frey ES bootstrap test with the model's own volatility forecast (seed 20260901).

Run from the repository root:
    python src/eval_bench.py > results/log_20261002_eval_bench.txt
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "src")
from evaluate import kupiec, christoffersen_ind
from sens_backtest import pinball, dm
from sens975 import es_test
from forecast_io import read_forecasts, write_forecasts

rd = lambda f: read_forecasts(f, parse_dates=["date"], float_precision="round_trip").sort_values("date").reset_index(drop=True)
fc, f975, bm = rd("results/forecasts.csv"), rd("results/forecasts_975.csv"), rd("results/bench_asym_forecasts.csv")
assert (fc["date"].values == bm["date"].values).all() and (f975["date"].values == bm["date"].values).all()
rv = fc["realized"].values; n = len(rv)
LEV = [0.95, 0.975, 0.99, 0.995]

rel = np.abs(bm["f_sig"].values - fc["e_sig"].values) / fc["e_sig"].values
print(f"FHS uses the EGARCH-EVT filter: volatility forecast differs from the stored e_sig on {(rel > 1e-6).sum()} days "
      f"(max relative difference {rel.max():.1e}; optimiser tolerance, see log_20260927_backtest_rerun.txt)")
print(f"Skewed-t shape: median nu {bm['k_nu'].median():.2f}, median lambda {bm['k_lam'].median():.3f} "
      f"(lambda > 0: right-skewed); lambda range {bm['k_lam'].min():.3f} to {bm['k_lam'].max():.3f}\n")


def var_of(model, q):
    if model == "EGARCH-EVT": return (f975 if q == 0.975 else fc)[f"eevt_VaR_{q}"].values
    if model == "FHS": return bm[f"fhs_VaR_{q}"].values
    if model == "EGARCH-skewed-t": return bm[f"skt_VaR_{q}"].values
    if model == "EGARCH-t": return (f975 if q == 0.975 else fc)[f"et_VaR_{q}"].values

def es_of(model, q):
    if model == "EGARCH-EVT": return (f975 if q == 0.975 else fc)[f"eevt_ES_{q}"].values, fc["e_sig"].values
    if model == "FHS": return bm[f"fhs_ES_{q}"].values, bm["f_sig"].values
    if model == "EGARCH-skewed-t": return bm[f"skt_ES_{q}"].values, bm["k_sig"].values
    return None, None

print("model             level  viol   exp  rate%  Kupiec p  pairs  ind p   cc p   pinball(1e-4)  DM vs EGARCH-EVT (stat, p)   ES: viol  mean D     p")
for q in LEV:
    base = pinball(rv, var_of("EGARCH-EVT", q), q)
    for model in ("EGARCH-EVT", "FHS", "EGARCH-skewed-t", "EGARCH-t"):
        v = var_of(model, q); I = (rv > v).astype(int); x = int(I.sum())
        _, pk = kupiec(x, n, 1 - q)
        lr_i, p_i, _, _ = christoffersen_ind(I)
        lr_u = -2 * ((n - x) * np.log(1 - (1 - q)) + x * np.log(1 - q) - (n - x) * np.log(1 - x / n) - x * np.log(x / n)) if x > 0 else -2 * n * np.log(q)
        p_cc = 1 - stats.chi2.cdf(lr_u + lr_i, 2)
        pairs = int(((I[:-1] == 1) & (I[1:] == 1)).sum())
        pb = pinball(rv, v, q)
        if model == "EGARCH-EVT":
            dmtxt = "-"
        else:
            st, pdm = dm(pb - base); dmtxt = f"{st:+.3f}, {pdm:.4f}"
        es, sg = es_of(model, q)
        if es is not None:
            nD, obs, pe = es_test(rv, v, es, sg); estxt = f"{nD:>8}  {obs:>+7.3f}  {pe:.4f}"
        else:
            estxt = "-"
        print(f"{model:<17}{q:>6.3f}{x:>6}{n*(1-q):>6.1f}{100*x/n:>7.2f}{pk:>10.4f}{pairs:>6}{p_i:>8.4f}{p_cc:>7.4f}{pb.mean()*1e4:>14.3f}   {dmtxt:<28}{estxt}")
    print()
print("DM statistic: mean pinball loss of the benchmark minus that of EGARCH-EVT; negative favours the benchmark.")
