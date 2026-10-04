"""Level of the benchmark forecasts relative to EGARCH-EVT (supports Section 7.5).

For the 99% and 99.5% levels, the mean over the 3,933 forecast days of the ratio of
each benchmark's VaR (and ES) to the EGARCH-EVT VaR (and ES), and the volatility forecast of the
skewed-t model relative to the adopted filter, and the windows in which the skewed-t optimiser
needed a retry or did not converge.

Run from the repository root:
    python src/bench_levels.py > results/log_20261002_bench_levels.txt
"""
import pandas as pd
from forecast_io import read_forecasts, write_forecasts
b = read_forecasts("results/bench_asym_forecasts.csv")
f = read_forecasts("results/forecasts.csv")
assert (b.date == f.date).all()
print(f"n = {len(b)} forecasts, {b.date.iloc[0]} to {b.date.iloc[-1]}")
print(f"{'level':>6} {'benchmark':>10} {'VaR ratio':>10} {'ES ratio':>9}")
for q in ("0.99", "0.995"):
    for name, col in (("FHS", "fhs"), ("skewed-t", "skt")):
        v = (b[f"{col}_VaR_{q}"] / f[f"eevt_VaR_{q}"]).mean()
        e = (b[f"{col}_ES_{q}"] / f[f"eevt_ES_{q}"]).mean()
        print(f"{q:>6} {name:>10} {v:10.3f} {e:9.3f}")
rat = b.k_sig / b.f_sig
print(f"Skewed-t volatility forecast / adopted-filter forecast: mean {rat.mean():.3f}, "
      f"5th-95th percentile [{rat.quantile(.05):.3f}, {rat.quantile(.95):.3f}], "
      f"correlation {b.k_sig.corr(b.f_sig):.4f}")
if "k_retry" in b:
    print("Skewed-t windows refitted after a non-converged default fit:", ", ".join(b.loc[b.k_retry > 0, "date"]) or "none")
print("Skewed-t windows not converged:", ", ".join(b.loc[~b.k_ok, "date"]) or "none")
