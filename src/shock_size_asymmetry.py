"""Model-free test of the inverse leverage effect that controls for shock size.

The cross-correlation of Section 3.4, Corr(r_t, |r_{t+h}|), mixes the effect of the
sign of a shock with the effect of its size: large days are mostly rises (positive
skewness), and large days are followed by large days (volatility clustering), so a
positive cross-correlation does not by itself show that rises matter more than
falls of the same size. This script compares what follows a rise and a fall of
similar absolute size, in two ways:

  1. Matched bins. The returns are cut into ten equal-count bins of |r_t|. Within
     each bin the mean of the later absolute return is compared between rises and
     falls, and the bin differences are averaged with weights proportional to the
     bin sizes. The confidence interval is a moving-block bootstrap (block length
     20, 5,000 replications) of the (r_t, later |r|) pairs with the bin edges held
     fixed.
  2. Regression. |r_{t+1}| is regressed on |r_t|, the mean absolute return of the
     previous 20 days (the volatility state before the shock) and an indicator of
     a rise, with Newey-West standard errors (10 lags). A second version adds the
     interaction of the rise indicator with |r_t|.

Both are run for the next-day magnitude and for the mean magnitude over the next
five days.

Run from the repository root:
    python src/shock_size_asymmetry.py > results/log_20261002_shock_size_asymmetry.txt
"""
import numpy as np, pandas as pd
import statsmodels.api as sm

np.random.seed(20261002)
vix = pd.read_csv("data/vix_raw.csv", parse_dates=["Date"]).set_index("Date")
r = np.log(vix["Close"] / vix["Close"].shift(1)).dropna()
x = r.values; G = len(x)
a = np.abs(x)

H = 5
# pair t with the later magnitudes; keep t where all H later returns exist and 20 earlier exist
t_idx = np.arange(20, G - H)
rt = x[t_idx]; at = a[t_idx]
y1 = a[t_idx + 1]
y5 = np.mean([a[t_idx + k] for k in range(1, H + 1)], axis=0)
state = np.array([a[t - 20:t].mean() for t in t_idx])      # mean |r| over the 20 days before day t
up = (rt > 0).astype(float)
n = len(t_idx)
print(f"Pairs: {n} (first return date {r.index[t_idx[0]].date()}, last {r.index[t_idx[-1]].date()})")
print(f"Rises {int(up.sum())}, falls {int(n - up.sum())}")

# ---------- 1. matched bins on |r_t| ----------
edges = np.quantile(at, np.linspace(0, 1, 11)); edges[-1] += 1e-12
bins = np.clip(np.digitize(at, edges[1:-1]), 0, 9)

def binned(idx, y):
    num = 0.0; den = 0.0; rows = []
    for b in range(10):
        m = bins[idx] == b
        u_ = m & (up[idx] == 1); d_ = m & (up[idx] == 0)
        if u_.sum() < 2 or d_.sum() < 2:
            continue
        diff = y[idx][u_].mean() - y[idx][d_].mean()
        w = m.sum()
        num += w * diff; den += w
        rows.append((b, m.sum(), u_.sum(), d_.sum(), at[idx][u_].mean(), at[idx][d_].mean(),
                     y[idx][u_].mean(), y[idx][d_].mean(), diff))
    return num / den, rows

def mbb(y, B=5000, L=20):
    out = np.empty(B); nb = int(np.ceil(n / L))
    for k in range(B):
        starts = np.random.randint(0, n - L + 1, nb)
        idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
        out[k] = binned(idx, y)[0]
    return out

print("\n=== 1. Matched bins of |r_t| (ten equal-count bins) ===")
for lab, y in (("next-day |r|", y1), ("mean |r| over next 5 days", y5)):
    est, rows = binned(np.arange(n), y)
    print(f"\n-- response: {lab}")
    print("  bin  n    rises falls  mean|r_t| rise  mean|r_t| fall  later|r| after rise  after fall  diff")
    for b, nb_, nu, nd, au, ad, yu, yd, df_ in rows:
        print(f"  {b+1:>3} {nb_:>4} {nu:>5} {nd:>5}  {au:>13.4f}  {ad:>14.4f}  {yu:>18.4f}  {yd:>10.4f}  {df_:>+.4f}")
    bs = mbb(y)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    base = y.mean()
    print(f"  Size-matched difference (rise minus fall), weighted over bins: {est:+.4f}  "
          f"95% block-bootstrap CI [{lo:+.4f}, {hi:+.4f}]  (mean response {base:.4f}; difference = {100*est/base:+.1f}% of it)")
    print(f"  Share of bootstrap replications with difference <= 0: {np.mean(bs <= 0):.4f}")
    est9, _ = binned(np.where(bins < 9)[0], y)
    print(f"  Excluding the top bin (where rises are larger than falls on average): {est9:+.4f}")
    print(f"  Bins with a positive difference: {sum(1 for r_ in rows if r_[-1] > 0)} of {len(rows)}")

# ---------- 2. regression with HAC errors ----------
print("\n=== 2. Regression of later magnitude on |r_t|, prior 20-day mean |r| and a rise indicator ===")
for lab, y in (("next-day |r|", y1), ("mean |r| over next 5 days", y5)):
    for spec in ("base", "interaction"):
        cols = [np.ones(n), at, state, up]
        names = ["const", "|r_t|", "mean|r| prior 20d", "rise"]
        if spec == "interaction":
            cols.append(up * at); names.append("rise x |r_t|")
        X = np.column_stack(cols)
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        print(f"\n-- response: {lab}; specification: {spec}   (n={n}, R2={res.rsquared:.3f})")
        for nm, b_, se, t_, p_ in zip(names, res.params, res.bse, res.tvalues, res.pvalues):
            print(f"  {nm:<20} {b_:>+9.4f}  se {se:.4f}  t {t_:>6.2f}  p {p_:.4f}")
