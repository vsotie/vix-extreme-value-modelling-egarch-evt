"""
Rolling out-of-sample backtest for the VIX EGARCH-EVT tail-risk model.
Objectives III-IV. Protocol follows McNeil & Frey (2000): n=1000-day rolling
window, daily re-estimation, k=100 exceedances per window (10% tail),
one-day-ahead upper-tail VaR/ES. Applied to VIX (FRED VIXCLS), not MF's data.

Models:
  EGARCH-EVT  : AR(1)-EGARCH(1,1)-t filter  + GPD on residual upper tail  (ours)
  GARCH-EVT   : AR(1)-GARCH(1,1)-t  filter  + GPD on residual upper tail  (MF, symmetric)
  Static-POT  : unconditional GPD on raw-return upper tail in the window    (benchmark)
Free parametric-t columns (no EVT) are also recorded for the MF-table analogy.

Risk direction: upper tail (a VIX spike is the loss). Violation: r_{t+1} > VaR_q.
"""
import os, sys, time, warnings
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, pandas as pd
from arch import arch_model
from scipy import stats
from utils import gpd_fit, gpd_tail_measures
warnings.filterwarnings("ignore")

N       = 1000                      # MF rolling window length
Q_THR   = 0.90                      # threshold quantile -> k~100 in a 1000-day window
LEVELS  = [0.95, 0.99, 0.995]
DATA    = "data/vix_raw.csv"

vix = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date")
r   = np.log(vix["Close"] / vix["Close"].shift(1)).dropna()
rr  = r.values
dates = r.index
G   = len(rr)

def _fit(mean, vol, w100):
    """Fit an arch model on a window; return (mu_next, sig_next, z, nu, ok).
    mu_next, sig_next on the x100 scale; z = unit-free standardised residuals."""
    kw = dict(mean="AR", lags=1, dist="t")
    if vol == "EGARCH":
        am = arch_model(w100, vol="EGARCH", p=1, o=1, q=1, **kw)
    else:
        am = arch_model(w100, vol="GARCH", p=1, q=1, **kw)
    try:
        fit = am.fit(disp="off", show_warning=False)
        ok = getattr(fit, "convergence_flag", 0) == 0
    except Exception:
        return None
    try:
        fc = fit.forecast(horizon=1, reindex=False)
        mu = float(fc.mean.values[0, 0])
        sig = float(np.sqrt(fc.variance.values[0, 0]))
    except Exception:
        return None
    z = (fit.resid / fit.conditional_volatility)
    z = z[np.isfinite(z)]
    nu = float(fit.params["nu"])
    if not (np.isfinite(mu) and np.isfinite(sig) and sig > 0):
        return None
    return mu, sig, np.asarray(z, float), nu, ok

def _evt_q(z, qs):
    """GPD upper-tail quantiles/ES on a unit-free residual (or raw return) array."""
    u = np.quantile(z, Q_THR)
    exc = z[z > u] - u
    gf = gpd_fit(exc)
    tm = gpd_tail_measures(u, gf["xi"], gf["beta"], n=len(z), n_u=len(exc), qs=tuple(qs))
    return tm, gf, len(exc)

def _t_q(nu, q):
    return stats.t.ppf(q, nu) * np.sqrt((nu - 2) / nu)

def forecast_one(i):
    """One-day-ahead forecast for day i, using the n returns strictly before i."""
    w = rr[i - N:i]
    realized = rr[i]
    out = {"date": dates[i], "realized": realized}
    w100 = w * 100.0

    # --- conditional filters ---
    for tag, vol in [("e", "EGARCH"), ("g", "GARCH")]:
        f = _fit("AR", vol, w100)
        if f is None:
            out[f"{tag}_ok"] = False
            continue
        mu, sig, z, nu, ok = f
        out[f"{tag}_ok"] = True
        out[f"{tag}_sig"] = sig / 100.0          # return-scale conditional sd
        out[f"{tag}_nu"]  = nu
        try:
            tm, gf, k = _evt_q(z, LEVELS)
            out[f"{tag}_xi"] = gf["xi"]; out[f"{tag}_k"] = k
            for q in LEVELS:
                out[f"{tag}evt_VaR_{q}"] = (mu + sig * tm.loc[q, "z_q"]) / 100.0
                out[f"{tag}evt_ES_{q}"]  = (mu + sig * tm.loc[q, "es_q"]) / 100.0
        except Exception:
            out[f"{tag}_ok"] = False
        for q in LEVELS:                          # parametric-t, no EVT (free)
            out[f"{tag}t_VaR_{q}"] = (mu + sig * _t_q(nu, q)) / 100.0

    # --- static unconditional POT on raw returns ---
    try:
        tm_r, gf_r, k_r = _evt_q(w, LEVELS)
        out["s_ok"] = True; out["s_xi"] = gf_r["xi"]; out["s_k"] = k_r
        for q in LEVELS:
            out[f"s_VaR_{q}"] = tm_r.loc[q, "z_q"]
            out[f"s_ES_{q}"]  = tm_r.loc[q, "es_q"]
    except Exception:
        out["s_ok"] = False
    return out

if __name__ == "__main__":
    nwin = int(sys.argv[1]) if len(sys.argv) > 1 else (G - N)
    origins = list(range(N, N + nwin))
    t0 = time.time()
    res = forecast_one(origins[0])
    dt = time.time() - t0
    print(f"cols: {sorted(res.keys())}")
    print(f"first window date {res['date'].date()}  realized {res['realized']:.4f}")
    print(f"  EGARCH-EVT VaR: " + ", ".join(f"{q}:{res.get(f'eevt_VaR_{q}', float('nan')):.4f}" for q in LEVELS))
    print(f"  GARCH-EVT  VaR: " + ", ".join(f"{q}:{res.get(f'gevt_VaR_{q}', float('nan')):.4f}" for q in LEVELS))
    print(f"  Static-POT VaR: " + ", ".join(f"{q}:{res.get(f's_VaR_{q}', float('nan')):.4f}" for q in LEVELS))
    print(f"  EGARCH sig {res.get('e_sig'):.4f}  xi {res.get('e_xi'):.4f} k {res.get('e_k')}  | GARCH sig {res.get('g_sig'):.4f}")
    print(f"per-window ~{dt:.2f}s  -> full {G-N} windows ~{dt*(G-N)/60:.1f} min serial, ~{dt*(G-N)/120:.1f} min on 2 cores")
