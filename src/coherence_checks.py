"""Numbers behind the cross-chapter wording of Chapters 3 and 7.

1. Window kurtosis (Table 3.3 convention: Pearson scale, bias-corrected,
   pandas .kurt() + 3) with and without the largest one or two days, to show
   how far the high-kurtosis windows rest on single observations.
2. Return quantiles by window, for the tail of each window.
3. The largest daily rises in the evaluation period of the rolling backtest
   (first forecast 22 Dec 2010), with the backtest regime of each.
4. The VIX level in the month (21 sessions) before each of the largest rises.

Run from Code/:  python src/coherence_checks.py
Output: results/log_20260928_coherence_checks.txt
"""
import numpy as np
import pandas as pd

vix = pd.read_csv("data/vix_raw.csv", parse_dates=["Date"]).set_index("Date")["Close"]
r = np.log(vix / vix.shift(1)).dropna()

WINDOWS = [("Global financial crisis", "2007", "2009"), ("Euro sovereign crisis", "2010", "2012"),
           ("Recovery", "2013", "2016"), ("Low volatility", "2017", "2019"),
           ("COVID-19", "2020", "2021"), ("Tightening", "2022", "2024"), ("Recent", "2025", "2026")]


def kurt(x):
    return x.kurt() + 3


print("=== 1. Window kurtosis with and without the largest rises ===")
for name, a, b in WINDOWS:
    x = r[a:b]
    top = x.sort_values(ascending=False)
    print(f"{a}-{b} {name:<24} n {len(x):>5}  kurt {kurt(x):6.2f}  "
          f"without largest day {kurt(x.drop(top.index[:1])):6.2f} ({top.index[0].date()} {top.iloc[0]:+.4f})  "
          f"without two largest {kurt(x.drop(top.index[:2])):6.2f} ({top.index[1].date()} {top.iloc[1]:+.4f})")
print()

print("=== 2. Return s.d. and upper quantiles by window ===")
for name, a, b in WINDOWS:
    x = r[a:b]
    print(f"{a}-{b} {name:<24} sd {x.std():.4f}  q0.99 {x.quantile(0.99):.4f}  q0.995 {x.quantile(0.995):.4f}  "
          f"max {x.max():+.4f}")
print()

print("=== 3. Largest daily rises in the evaluation period (from 22 Dec 2010) ===")
REG = [("2010-12", "2010-12-22", "2012-12-31"), ("2013-16", "2013-01-01", "2016-12-31"),
       ("2017-19", "2017-01-01", "2019-12-31"), ("2020-21", "2020-01-01", "2021-12-31"),
       ("2022-24", "2022-01-01", "2024-12-31"), ("2025-26", "2025-01-01", "2026-12-31")]
ev = r["2010-12-22":]
for rank, (d, v) in enumerate(ev.sort_values(ascending=False).head(12).items(), 1):
    reg = next(lab for lab, a, b in REG if pd.Timestamp(a) <= d <= pd.Timestamp(b))
    prev = r[:d].iloc[-2]
    print(f"{rank:>2}. {d.date()}  r {v:+.4f}  previous day {prev:+.4f}  regime {reg}")
full_rank = r.sort_values(ascending=False)
print("Largest rises in the full sample:", ", ".join(f"{d.date()} {v:+.4f}" for d, v in full_rank.head(5).items()))
print()

print("=== 4. VIX level in the 21 sessions before each of the largest rises ===")
for d in ["2018-02-05", "2024-12-18", "2024-08-05", "2021-01-27", "2021-11-26", "2025-04-04"]:
    d = pd.Timestamp(d)
    pre = vix[:d].iloc[-22:-1]          # the 21 closes before day d
    print(f"{d.date()}  close {vix[d]:6.2f}  previous close {pre.iloc[-1]:6.2f}  "
          f"max of the other 20 {pre.iloc[:-1].max():6.2f}  median of the 21 {pre.median():6.2f}  "
          f"days of the 21 at or above 19: {(pre >= 19).sum()}")
print(f"Sample median close {vix.median():.2f}")
