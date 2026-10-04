"""Regime breakdown at the 97.5% level (companion to regime_analysis.py, which is at 99%).

Same regimes, models and statistics as regime_analysis.py, applied to the 97.5%
forecasts derived from the published backtest (results/forecasts_975.csv,
written by level975.py). At 97.5% each regime has 2.5 times as many expected
violations as at 99% (about 10 to 25 rather than 4 to 10).

Run from Code/:  python src/regime975.py > results/log_20260928_regime975.txt
"""
import numpy as np, pandas as pd
from forecast_io import read_forecasts, write_forecasts
Q = 0.975
df = read_forecasts("results/forecasts.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
d975 = pd.read_csv("results/forecasts_975.csv", parse_dates=["date"], float_precision="round_trip").sort_values("date").reset_index(drop=True)
assert (df.date.values == d975.date.values).all()
for c in d975.columns:
    if c != "date":
        df[c] = d975[c].values
r = df["realized"].values
def pinball(rv, v, a): d = rv - v; return np.where(d >= 0, a * d, (a - 1) * d)
REG = [("2010-12", "2010-01-01", "2012-12-31"), ("2013-16", "2013-01-01", "2016-12-31"),
       ("2017-19", "2017-01-01", "2019-12-31"), ("2020-21", "2020-01-01", "2021-12-31"),
       ("2022-24", "2022-01-01", "2024-12-31"), ("2025-26", "2025-01-01", "2026-12-31")]
MODELS = ("eevt", "gevt", "s")

print(f"=== Breach ratio (actual/expected) by regime, q={Q}: EGARCH-EVT / GARCH-EVT / Static POT ===")
for lab, a, b in REG:
    m = ((df.date >= a) & (df.date <= b)).values; nd = int(m.sum()); exp = nd * (1 - Q)
    cnt = [int((r[m] > df.loc[m, f"{mm}_VaR_{Q}"].values).sum()) for mm in MODELS]
    print(f"  {lab}  days {nd:>5}  expected {exp:5.1f}  violations {cnt[0]:>3} / {cnt[1]:>3} / {cnt[2]:>3}  "
          f"ratio {cnt[0]/exp:.2f} / {cnt[1]/exp:.2f} / {cnt[2]/exp:.2f}")
print()
print(f"=== Mean pinball loss x1e4 by regime, q={Q}; EG-GA < 0 means EGARCH-EVT more accurate ===")
for lab, a, b in REG:
    m = ((df.date >= a) & (df.date <= b)).values
    L = {mm: pinball(r[m], df.loc[m, f"{mm}_VaR_{Q}"].values, Q).mean() * 1e4 for mm in MODELS}
    print(f"  {lab}  EGARCH {L['eevt']:.3f}  GARCH {L['gevt']:.3f}  Static {L['s']:.3f} | EG-GA {L['eevt']-L['gevt']:+.3f}")
print()
print(f"=== ES shortfall by regime: mean (realized-ES)/sigma on EGARCH-EVT breach days, q={Q} ===")
for lab, a, b in REG:
    sub = df.loc[((df.date >= a) & (df.date <= b)).values]
    br = sub.realized > sub[f"eevt_VaR_{Q}"]
    disc = ((sub.realized - sub[f"eevt_ES_{Q}"]) / sub.e_sig)[br]
    print(f"  {lab}  breaches {int(br.sum()):>3}  mean_disc {disc.mean():+.3f}")
