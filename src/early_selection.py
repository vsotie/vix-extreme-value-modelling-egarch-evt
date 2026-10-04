"""Would the specification have been chosen with the information available at the start of the backtest?

The backtest re-estimates the parameters on each rolling window, but the specification
(EGARCH rather than GARCH, an AR(1) mean, Student-t innovations, a GPD tail above the
90th percentile) was chosen with the full sample. This script repeats the model
comparison of Table 5.1 and the tail fit of Section 6.4 on two early information sets:
  A  the first 1,000 returns, the window available when the first forecast is made
     (return dates 2007-01-04 to 2010-12-21);
  B  all returns before 6 October 2014 (1,952 returns).
It reports whether the same model would have been preferred, the size and
significance of the asymmetry parameter, the Engle-Ng sign-bias test of the symmetric
filter, the residual diagnostics and the GPD shape estimate. This does not make the
backtest an untouched out-of-sample test: the choices were made by an author who also
saw the later data. It shows whether the evidence for the choices was already
present in the early data.

Run from the repository root:
    python src/early_selection.py > results/log_20261002_early_selection.txt
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
sys.path.insert(0, "src")
from evt_common import load_returns
from utils import engle_ng_sign_bias, gpd_fit

warnings.filterwarnings("ignore")
r = load_returns(verbose=False)
sets = {"A: first 1,000 returns": r.iloc[:1000], "B: returns before 6 Oct 2014": r[r.index < "2014-10-06"],
        "Full sample (Table 5.1)": r}
for label, rr in sets.items():
    rc = rr.iloc[1:] * 100                       # common sample for all three models
    f_g = arch_model(rc, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    f_e = arch_model(rc, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    f_a = arch_model(rr * 100, mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    print(f"=== {label}: {len(rr)} returns, {rr.index[0].date()} to {rr.index[-1].date()} ===")
    print(f"  {'model':<22}{'LL':>10}{'BIC':>10}{'params':>8}")
    for nm, f in (("GARCH(1,1)-t const", f_g), ("EGARCH(1,1)-t const", f_e), ("EGARCH(1,1)-t AR(1)", f_a)):
        print(f"  {nm:<22}{f.loglikelihood:>10.1f}{f.bic:>10.1f}{f.num_params:>8}")
    best = min((("GARCH(1,1)-t const", f_g.bic), ("EGARCH(1,1)-t const", f_e.bic), ("EGARCH(1,1)-t AR(1)", f_a.bic)), key=lambda t: t[1])[0]
    print(f"  Lowest BIC: {best}")
    print(f"  EGARCH-AR(1): theta {f_a.params['gamma[1]']:+.3f} (t {f_a.tvalues['gamma[1]']:.1f}), "
          f"AR(1) coefficient {f_a.params.iloc[1]:+.3f} (t {f_a.tvalues.iloc[1]:.1f}), nu {f_a.params['nu']:.2f}")
    zg = (f_g.resid / f_g.conditional_volatility).dropna().values
    _, lm, p = engle_ng_sign_bias(zg)
    print(f"  Engle-Ng sign-bias test of the symmetric GARCH-t residuals: LM {lm:.2f}, p {p:.4f}")
    z = (f_a.resid / f_a.conditional_volatility).dropna().values
    lb_a = acorr_ljungbox(np.abs(z), lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    lb_s = acorr_ljungbox(z ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    print(f"  Adopted-model residuals: Ljung-Box (10 lags) p, |z| {lb_a:.3f}, z^2 {lb_s:.3f}")
    u = np.quantile(z, 0.90); e = z[z > u] - u; g = gpd_fit(e)
    print(f"  GPD above the 90th percentile of the residuals: u {u:.3f}, k {len(e)}, xi {g['xi']:.3f} (se {g['se_xi']:.3f}), psi {g['beta']:.3f}\n")
