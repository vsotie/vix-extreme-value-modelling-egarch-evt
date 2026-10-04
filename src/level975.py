"""The 97.5% level, derived from the published rolling forecasts.

The 97.5% level is the Basel market-risk level for expected shortfall (MAR33.3)
and one of the two levels of desk-level VaR backtesting (MAR32.18). It was not
among the levels of the original backtest (95%, 99%, 99.5%). No re-estimation is
needed: within each window every forecast is a closed-form function of fitted
quantities, so the 97.5% forecast follows exactly from the stored ones.

  GPD models (EGARCH-EVT, GARCH-EVT, Static POT). With threshold u, scale psi,
  shape xi, exceedance share k/n, location mu and scale sigma (mu=0, sigma=1 for
  Static POT),
      VaR_q = mu + sigma*(u + psi/xi*(((1-q)*n/k)^(-xi) - 1)) = A + B*(1-q)^(-xi),
      (1-xi)*ES_q - VaR_q = sigma*psi - xi*(mu + sigma*u) = C,
  with A, B, C constant in q. The stored 95% and 99% forecasts and xi give A and
  B; the stored 99% ES gives C. The stored 99.5% values serve as a check.
  Student-t models: VaR_q = mu + sigma * t_nu^{-1}(q) * sqrt((nu-2)/nu); mu from
  the stored 95% forecast, checked against the stored 99.5% forecast.

Every test uses the published implementation: coverage and ES tests are imported
from evaluate.py, pinball loss and Diebold-Mariano from the same formulas as
var_accuracy.py. The ES bootstrap first repeats the published sequence of calls
(seed 20260901) to confirm the published p-values, then is reseeded with 20260901
for the 97.5% tests.

Run from Code/:  python src/level975.py > results/log_20260928_level975.txt
Output: results/forecasts_975.csv (97.5% columns only, merged on date).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "src")
import evaluate as ev                      # sets the published seed on import
from forecast_io import read_forecasts, write_forecasts

Q = 0.975
df = read_forecasts("results/forecasts.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
r = df["realized"].values


# --------------------------------------------------------------------------
# 1. Derive the 97.5% forecasts
# --------------------------------------------------------------------------
def gpd_var(v95, v99, xi, q):
    a = lambda p: (1 - p) ** (-xi)
    B = (v99 - v95) / (a(0.99) - a(0.95))
    A = v95 - B * a(0.95)
    return A + B * a(q)


def gpd_es(var_q, es99, var99, xi):
    C = (1 - xi) * es99 - var99
    return (var_q + C) / (1 - xi)


def t_var(v95, sig, nu, q):
    c = lambda p: stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)
    return v95 - sig * c(0.95) + sig * c(q)


out = pd.DataFrame({"date": df["date"]})
print("=== 1. Derivation check: stored 99.5% values recovered from the 95% and 99% values ===")
for pfx, xi in [("eevt", "e_xi"), ("gevt", "g_xi"), ("s", "s_xi")]:
    v995 = gpd_var(df[f"{pfx}_VaR_0.95"], df[f"{pfx}_VaR_0.99"], df[xi], 0.995)
    e995 = gpd_es(v995, df[f"{pfx}_ES_0.99"], df[f"{pfx}_VaR_0.99"], df[xi])
    print(f"  {pfx:<5} max |VaR error| {np.abs(v995 - df[f'{pfx}_VaR_0.995']).max():.1e}   "
          f"max |ES error| {np.abs(e995 - df[f'{pfx}_ES_0.995']).max():.1e}")
    v = gpd_var(df[f"{pfx}_VaR_0.95"], df[f"{pfx}_VaR_0.99"], df[xi], Q)
    out[f"{pfx}_VaR_{Q}"] = v
    out[f"{pfx}_ES_{Q}"] = gpd_es(v, df[f"{pfx}_ES_0.99"], df[f"{pfx}_VaR_0.99"], df[xi])
for pfx, sig, nu in [("et", "e_sig", "e_nu"), ("gt", "g_sig", "g_nu")]:
    v995 = t_var(df[f"{pfx}_VaR_0.95"], df[sig], df[nu], 0.995)
    print(f"  {pfx:<5} max |VaR error| {np.abs(v995 - df[f'{pfx}_VaR_0.995']).max():.1e}")
    out[f"{pfx}_VaR_{Q}"] = t_var(df[f"{pfx}_VaR_0.95"], df[sig], df[nu], Q)
# ordering check: the 97.5% forecast lies between the 95% and 99% forecasts every day
for pfx in ["eevt", "gevt", "s", "et", "gt"]:
    ok = ((out[f"{pfx}_VaR_{Q}"] > df[f"{pfx}_VaR_0.95"]) & (out[f"{pfx}_VaR_{Q}"] < df[f"{pfx}_VaR_0.99"])).all()
    print(f"  {pfx:<5} 95% < 97.5% < 99% on every day: {ok}")
out.to_csv("results/forecasts_975.csv", index=False)
print()

# --------------------------------------------------------------------------
# 2. Coverage tests (evaluate.py implementation)
# --------------------------------------------------------------------------
MODELS = {"EGARCH-EVT": "eevt", "GARCH-EVT": "gevt", "Static-POT": "s", "EGARCH-t": "et", "GARCH-t": "gt"}
print(f"=== 2. Coverage at q={Q} (T={len(r)}, expected {len(r)*(1-Q):.1f} violations) ===")
print(f"{'model':<11}{'viol':>6}{'rate%':>8}{'p_uc':>9}{'pi11%':>8}{'p_ind':>9}{'p_cc':>9}  consecutive pairs")
for name, pfx in MODELS.items():
    I = (r > out[f"{pfx}_VaR_{Q}"].values).astype(int)
    x, n = int(I.sum()), len(I)
    _, p_uc = ev.kupiec(x, n, 1 - Q)
    lr_i, p_i, _, pi11 = ev.christoffersen_ind(I)
    lr_uc, _ = ev.kupiec(x, n, 1 - Q)
    p_cc = 1 - stats.chi2.cdf(lr_uc + lr_i, 2)
    pairs = [f"{df.date[i-1].date()}/{df.date[i].date()}" for i in range(1, n) if I[i] and I[i - 1]]
    print(f"{name:<11}{x:>6}{100*x/n:>8.3f}{p_uc:>9.4f}{100*pi11:>8.2f}{p_i:>9.4f}{p_cc:>9.4f}  {len(pairs)}: {', '.join(pairs)}")
print()

# --------------------------------------------------------------------------
# 3. Pinball loss and Diebold-Mariano (var_accuracy.py formulas)
# --------------------------------------------------------------------------
def pinball(r, v, a):
    d = r - v
    return np.where(d >= 0, a * d, (a - 1) * d)


def dm(d, lag=5):
    d = np.asarray(d, float); T = len(d); m = d.mean(); s = np.var(d)
    for L in range(1, lag + 1):
        c = np.mean((d[L:] - m) * (d[:-L] - m)); s += 2 * (1 - L / (lag + 1)) * c
    st = m / np.sqrt(s / T)
    return st, 2 * (1 - stats.norm.cdf(abs(st)))


print(f"=== 3. Pinball loss at q={Q} (mean x1e4; lower is better) ===")
loss = {pfx: pinball(r, out[f"{pfx}_VaR_{Q}"].values, Q) for pfx in MODELS.values()}
print("  " + "  ".join(f"{name}={loss[p].mean()*1e4:.4f}" for name, p in MODELS.items()))
st, p = dm(loss["eevt"] - loss["gevt"])
adv = 100 * (1 - loss["eevt"].mean() / loss["gevt"].mean())
print(f"  EGARCH-EVT vs GARCH-EVT: advantage {adv:.2f}%  DM {st:.3f}  p {p:.4f}  (negative favours EGARCH-EVT)")
print()

# --------------------------------------------------------------------------
# 4. McNeil-Frey bootstrap ES test (evaluate.py implementation)
# --------------------------------------------------------------------------
print("=== 4. McNeil-Frey bootstrap ES test (B=10,000; one-sided, small p => ES too low) ===")
ext = df.copy()
for c in out.columns:
    if c != "date":
        ext[c] = out[c].values
ev.df = ext                                  # evaluate.py functions read ev.df
print("  Published sequence repeated with seed 20260901 (must match log_20260901_backtest_eval.txt):")
np.random.seed(20260901)
for pfx, name in [("eevt", "EGARCH-EVT"), ("gevt", "GARCH-EVT")]:
    for q in [0.95, 0.99, 0.995]:
        nD, obs, pv = ev.mf_es_test(pfx, q)
        print(f"    {name} q={q}: n_viol={nD}  mean_discrepancy={obs}  p={pv}")
print("  97.5% tests, reseeded with 20260901:")
np.random.seed(20260901)
for pfx, name in [("eevt", "EGARCH-EVT"), ("gevt", "GARCH-EVT")]:
    nD, obs, pv = ev.mf_es_test(pfx, Q)
    se = np.sqrt(pv * (1 - pv) / 10000)
    print(f"    {name} q={Q}: n_viol={nD}  mean_discrepancy={obs}  p={pv}  (Monte Carlo s.e. {se:.4f})")
print()

# --------------------------------------------------------------------------
# 5. Size of the 97.5% ES relative to the 99% VaR (the Basel calibration question)
# --------------------------------------------------------------------------
print("=== 5. 97.5% ES against 99% VaR, EGARCH-EVT (both one-day log returns) ===")
ratio = out[f"eevt_ES_{Q}"] / df["eevt_VaR_0.99"]
print(f"  ES_0.975 / VaR_0.99: median {ratio.median():.4f}  5th pct {ratio.quantile(0.05):.4f}  "
      f"95th pct {ratio.quantile(0.95):.4f}")
