"""
Verification log for numbers quoted in Chapters 1, 3 and 5 of the report that
were not previously written to a run log (review of 26 Sep 2026, List 1 items
4, 5, 6, 8 and 13).

Everything here is recomputed from data/vix_raw.csv with the same
transformations as main.py and backtest.py, so no number depends on memory.

Run from the Code/ directory:
    python src/verify_report_numbers.py | tee results/log_20260926_report_verification.txt
"""
import inspect
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import arch
from arch import arch_model
from arch.univariate import EGARCH

import sys
sys.path.insert(0, "src")
from utils import ccf, window_stability   # the exact helpers used for the report

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

WINDOWS = [
    ('2007-2009 GFC',        '2007-01-01', '2009-12-31'),
    ('2010-2012 Euro',       '2010-01-01', '2012-12-31'),
    ('2013-2016 Recovery',   '2013-01-01', '2016-12-31'),
    ('2017-2019 Low-vol',    '2017-01-01', '2019-12-31'),
    ('2020-2021 COVID',      '2020-01-01', '2021-12-31'),
    ('2022-2024 Tightening', '2022-01-01', '2024-12-31'),
    ('2025-2026 Recent',     '2025-01-01', '2026-06-30'),
]

vix = pd.read_csv("data/vix_raw.csv", parse_dates=["Date"]).set_index("Date")
close = vix["Close"]
r = np.log(close / close.shift(1)).dropna()
N = len(r)
band = 1.96 / np.sqrt(N)

print(f"arch {arch.__version__}")
print(f"closes: {len(close)}  {close.index[0].date()} -> {close.index[-1].date()}")
print(f"log returns N = {N}  {r.index[0].date()} -> {r.index[-1].date()}")
print(f"95% i.i.d. band 1.96/sqrt(N) = {band:.4f}\n")

# ---------------------------------------------------------------- A. Ch.1 closes
print("=== A. Closes and returns quoted in Chapter 1, Section 1.3 ===")
sd = r.std()
for d in ["2018-02-01", "2018-02-02", "2018-02-05",
          "2024-12-17", "2024-12-18",
          "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07", "2025-04-08", "2025-04-09"]:
    ts = pd.Timestamp(d)
    c = close.get(ts, np.nan)
    rr = r.get(ts, np.nan)
    print(f"  {d}  close {c:8.2f}   log return {rr:+.4f} ({rr*100:+.1f}%)   = {rr/sd:+.2f} sample sd")
print(f"  5 Feb 2018 vs 1 Feb 2018: log change {np.log(close['2018-02-05']/close['2018-02-01']):+.4f}")
print(f"  5 Feb 2018 level change vs 2 Feb: {(close['2018-02-05']/close['2018-02-02']-1)*100:+.1f}%")
print(f"  largest one-day decline in sample: {r.min():+.4f} on {r.idxmin().date()}")
print(f"  largest one-day rise in sample:    {r.max():+.4f} on {r.idxmax().date()}\n")

# ---------------------------------------------------------------- B. Level stats
print("=== B. Level statistics quoted in Chapter 3, Section 3.1 ===")
print(f"  full-sample median close: {close.median():.2f}")
print(f"  maximum close: {close.max():.2f} on {close.idxmax().date()}")
c08 = close.loc["2008"]
print(f"  maximum close in 2008: {c08.max():.2f} on {c08.idxmax().date()}")
top = close.sort_values(ascending=False)
print("  five highest closes:", ", ".join(f"{v:.2f} ({i.date()})" for i, v in top.head(5).items()))
print(f"  minimum close: {close.min():.2f} on {close.idxmin().date()}")
print("  median close by window:")
for lab, a, b in WINDOWS:
    print(f"    {lab:22s} {close.loc[a:b].median():6.2f}   (n closes {close.loc[a:b].size})")
meds = [close.loc[a:b].median() for _, a, b in WINDOWS]
print(f"  ratio max/min window median: {max(meds)/min(meds):.2f}\n")

# ---------------------------------------------------------------- C. Tail counts
print("=== C. Exceedance counts, Chapters 1 and 3 ===")
m = r.mean()
dev = (r - m) / sd
for k in (3, 4, 5):
    n_all = int((dev.abs() > k).sum())
    n_pos = int((dev > k).sum())
    n_neg = int((dev < -k).sum())
    pred = N * 2 * stats.norm.sf(k)
    print(f"  |r - mean| > {k} sd: {n_all:3d}  (positive {n_pos}, negative {n_neg});  normal predicts {pred:.3f}")
print()

# ---------------------------------------------------------------- D. Feb 2018
print("=== D. Returns 2-9 February 2018 ===")
print(r.loc["2018-02-02":"2018-02-09"].round(4).to_string(), "\n")

# ---------------------------------------------------------------- E. ACF
print("=== E. Sample ACF (statsmodels acf, fft=True, as in main.py), lags 1-30 ===")
a_r = acf(r, nlags=30, fft=True)[1:]
a_a = acf(r.abs(), nlags=30, fft=True)[1:]
a_s = acf(r ** 2, nlags=30, fft=True)[1:]
tab = pd.DataFrame({"r": a_r, "|r|": a_a, "r^2": a_s}, index=range(1, 31))
tab.index.name = "lag"
print(tab.round(4).to_string())
for col in tab:
    out = [int(s) for s in tab.index if abs(tab.loc[s, col]) > band]
    print(f"  {col:4s} lags outside +/-{band:.4f}: {out}")
# first lag at which |r| ACF falls inside the band
inside = [int(s) for s in tab.index if abs(tab.loc[s, "|r|"]) <= band]
print(f"  first lag at which the |r| ACF is inside the band: {inside[0] if inside else 'none'}\n")

# ---------------------------------------------------------------- F. Ljung-Box raw
print("=== F. Ljung-Box on raw series (Chapter 3, Table 3.2) ===")
for name, s in {"r": r, "|r|": r.abs(), "r^2": r ** 2}.items():
    lb = acorr_ljungbox(s, lags=[5, 10, 20], return_df=True)
    print(f"  {name}:")
    for lag in (5, 10, 20):
        print(f"    Q({lag:2d}) = {lb.loc[lag,'lb_stat']:9.2f}   p = {lb.loc[lag,'lb_pvalue']:.3e}")
print()

# ---------------------------------------------------------------- G. CCF
print("=== G. CCF of r_t with |r_{t+k}| (utils.ccf, as in main.py), k = -25..25 ===")
c = ccf(r.values, r.abs().values, 25)
print(c.round(4).to_string())
pos_out = [k for k in range(1, 26) if abs(c[k]) > band]
neg_vals = [k for k in range(1, 26) if c[k] < 0]
print(f"  k >= 1 outside the band: {pos_out}")
print(f"  k >= 1 with negative value: {neg_vals}")
run = 0
for k in range(1, 26):
    if c[k] > band:
        run = k
    else:
        break
print(f"  continuous run of k >= 1 above the band ends at k = {run}\n")

# ---------------------------------------------------------------- H. EGARCH fit
print("=== H. AR(1)-EGARCH(1,1)-t on returns x100 (as in main.py) ===")
fit = arch_model(r * 100, mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1,
                 dist="t").fit(disp="off")
print(fit.params.round(4).to_string())
nobs = int(fit.nobs)
ll100 = fit.loglikelihood
print(f"  nobs = {nobs}   log-likelihood (x100 scale) = {ll100:.1f}   BIC = {fit.bic:.1f}   AIC = {fit.aic:.1f}")
print(f"  log-likelihood on the original return scale = LL + nobs*ln(100) = {ll100 + nobs*np.log(100):.1f}"
      f"   (shift {nobs*np.log(100):.1f})")
al, th, be, nu = fit.params["alpha[1]"], fit.params["gamma[1]"], fit.params["beta[1]"], fit.params["nu"]
print(f"  news impact at z=+1: alpha+gamma = {al+th:+.4f};  at z=-1: alpha-gamma = {al-th:+.4f}")
print(f"  ratio of |impact| (+1 vs -1): {(al+th)/abs(al-th):.2f}")
print(f"  standardised t kurtosis 3 + 6/(nu-4) = {3 + 6/(nu-4):.3f}")
print(f"  t tail shape 1/nu = {1/nu:.4f}")
print(f"  E|z| under the fitted standardised t = {np.sqrt((nu-2)/nu)*stats.t(nu).expect(lambda x: abs(x)):.4f};"
      f"  sqrt(2/pi) = {np.sqrt(2/np.pi):.4f}")
mu, phi = fit.params["Const"], fit.params["VIX[1]"] if "VIX[1]" in fit.params else fit.params.iloc[1]
print(f"  mu (x100) = {mu:.4f} -> {mu/100:+.5f} per day;  phi = {phi:.4f};"
      f"  implied unconditional mean mu/(1-phi) = {mu/(1-phi):.4f} (x100)")
print(f"  mu t-stat = {fit.tvalues['Const']:.2f}, p = {fit.pvalues['Const']:.4f}\n")

print("=== I. Location of the return distribution (the mu-hat question) ===")
print(f"  sample mean = {r.mean():+.6f}   t = {r.mean()/(sd/np.sqrt(N)):.2f}")
print(f"  sample median = {r.median():+.6f}")
print(f"  share of positive returns = {(r > 0).mean():.4f};  zero returns = {(r == 0).sum()}")
df_t, loc_t, sc_t = stats.t.fit(r.values)
print(f"  i.i.d. Student-t MLE of the raw returns: df = {df_t:.3f}, location = {loc_t:+.6f}, scale = {sc_t:.5f}")
print(f"  mode of a Gaussian KDE: ", end="")
kde = stats.gaussian_kde(r.values)
grid = np.linspace(-0.1, 0.1, 4001)
print(f"{grid[np.argmax(kde(grid))]:+.5f}\n")

# ---------------------------------------------------------------- J. windows
z = (fit.resid / fit.conditional_volatility).dropna()
print(f"=== J. Sub-period table (utils.window_stability), residual N = {len(z)} ===")
print("  Ljung-Box lag used by main.py: the default lb_lag = 10")
w_raw10 = window_stability(r, WINDOWS, "raw returns", lb_lag=10)
w_res10 = window_stability(z, WINDOWS, "residuals", lb_lag=10)
w_res5 = window_stability(z, WINDOWS, "residuals", lb_lag=5)
print(pd.concat([w_raw10, w_res10]).round(4).to_string())
print("  residuals with lb_lag = 5 (for comparison with any table labelled Q(5)):")
print(w_res5.round(4).to_string())
kt = 3 + 6 / (nu - 4)
print(f"  residual kurtosis vs the fitted standardised t ({kt:.3f}):")
for (s, w), row in w_res10.iterrows():
    print(f"    {w:22s} {row['kurtosis']:.3f}  {'<=' if row['kurtosis'] <= kt else '>'} t")
print()

# ---------------------------------------------------------------- K. arch form
print("=== K. EGARCH recursion as implemented by arch ===")
doc = inspect.getdoc(EGARCH)
i0 = doc.find(r"\ln\sigma")
print(doc[i0:i0 + 260])
print()

# ---------------------------------------------------------------- L. residual ACF
print("=== L. Residual ACF of z, |z|, z^2, lags 1-10 ===")
rt = pd.DataFrame({"z": acf(z, nlags=10, fft=True)[1:],
                   "|z|": acf(z.abs(), nlags=10, fft=True)[1:],
                   "z^2": acf(z ** 2, nlags=10, fft=True)[1:]}, index=range(1, 11))
print(rt.round(4).to_string())

# ---------------------------------------------------------------- M. window LB, exact
print("\n=== M. Ljung-Box Q(10) p-values on |r| within each window (Chapter 3, Table 3.3) ===")
for lab, a, b in WINDOWS:
    s = r.loc[a:b]
    lb = acorr_ljungbox(s.abs(), lags=[10], return_df=True)
    print(f"  {lab:22s} N={s.size:5d}  Q(10)={lb.loc[10,'lb_stat']:8.2f}  p={lb.loc[10,'lb_pvalue']:.2e}")

# ---------------------------------------------------------------- N. largest rises (Ch.3 Section 3.5)
print("\n=== N. Largest daily log increases and the level in the preceding month (Chapter 3, Section 3.5) ===")
top = r.sort_values(ascending=False).head(5)
for d, x in top.items():
    prev = close.loc[d - pd.Timedelta(days=31): d - pd.Timedelta(days=1)]
    print(f"  {d.date()}  r={x:+.4f}  prior close={prev.iloc[-1]:.2f}  "
          f"max close in prior month={prev.max():.2f}  max excluding prior session={prev.iloc[:-1].max():.2f}")
