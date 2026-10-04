"""Asymmetric-tail benchmarks for the rolling backtest (reviewer request).

The main comparison shows that EVT beats the fitted symmetric Student-t. This
script adds two benchmarks that allow an asymmetric or empirical tail, so that the
comparison is not only against a symmetric parametric tail:

  FHS     : the adopted AR(1)-EGARCH(1,1)-t filter, with VaR and ES taken from the
            empirical distribution of the window's standardised residuals
            (filtered historical simulation; no tail model). Same filter, same
            mu and sigma as EGARCH-EVT, so only the tail stage differs.
  EGARCH-skt : AR(1)-EGARCH(1,1) with Hansen's skewed Student-t innovations
            (arch 'skewt'), estimated by maximum likelihood on each window;
            VaR and ES are the model's own quantile and tail mean.

Protocol identical to backtest.py: 1,000-return window, one-day-ahead forecast,
upper tail, levels 95, 97.5, 99 and 99.5%.

Run from the repository root (Python 3.13, requirements.txt):
    python src/bench_asym.py > results/log_20261002_bench_asym.txt
Output: results/bench_asym_forecasts.csv
"""
import sys, time, platform, warnings
import numpy as np, pandas as pd
from arch import arch_model
from arch.univariate.distribution import SkewStudent
from joblib import Parallel, delayed
sys.path.insert(0, "src")
import sens_backtest as sb
from forecast_io import read_forecasts, write_forecasts

warnings.filterwarnings("ignore")
LEV = [0.95, 0.975, 0.99, 0.995]
_SKT = SkewStudent()
# Midpoint rule for ES = (1/(1-q)) int_q^1 ppf(u) du = int_0^1 ppf(1 - (1-q) s) ds.
# The grid stops short of s = 0, where ppf is unbounded; the omitted piece is
# below 2e-4 of the ES for nu > 4.5 (checked against a 2e6-point grid in the log).
_NS = 4000
_S = (np.arange(_NS) + 0.5) / _NS


def skt_es(q, nu, lam):
    """Tail mean of Hansen's standardised skewed t above its q-quantile."""
    x = _SKT.ppf(1.0 - (1.0 - q) * _S, [nu, lam])
    return float(x.mean())


def skt_var(q, nu, lam):
    return float(_SKT.ppf(np.array([q]), [nu, lam])[0])


def fit_skt(w100):
    """Skewed-t EGARCH fit. If the optimiser reports non-convergence, retry (1) with a tighter
    tolerance and more iterations, then (2) from the Student-t solution with zero skewness as
    starting values. The first converged fit is kept; if none converges the default fit is kept
    and flagged (k_ok False). Returns the fit and the retry that produced it (0 = default)."""
    spec = dict(mean="AR", lags=1, vol="EGARCH", p=1, o=1, q=1)
    f = arch_model(w100, dist="skewt", **spec).fit(disp="off", show_warning=False)
    if getattr(f, "convergence_flag", 0) == 0:
        return f, 0
    f1 = arch_model(w100, dist="skewt", **spec).fit(disp="off", show_warning=False,
                                                    tol=1e-9, options={"maxiter": 1000})
    if getattr(f1, "convergence_flag", 0) == 0:
        return f1, 1
    ft = arch_model(w100, dist="t", **spec).fit(disp="off", show_warning=False)
    f2 = arch_model(w100, dist="skewt", **spec).fit(disp="off", show_warning=False,
                                                    starting_values=np.r_[ft.params.values, 0.0])
    if getattr(f2, "convergence_flag", 0) == 0:
        return f2, 2
    return f, -1


def one(i):
    w100 = sb.rr[i - sb.N:i] * 100.0
    out = {"date": sb.dates[i], "realized": sb.rr[i]}
    # --- FHS on the adopted filter ---
    mu, sig, z = sb.fit_egarch(w100, 1, 1, 1)
    out["f_sig"] = sig / 100.0
    for q in LEV:
        zq = np.quantile(z, q)
        out[f"fhs_VaR_{q}"] = (mu + sig * zq) / 100.0
        out[f"fhs_ES_{q}"] = (mu + sig * z[z >= zq].mean()) / 100.0
    # --- EGARCH with skewed-t innovations ---
    try:
        f, retry = fit_skt(w100)
        fc = f.forecast(horizon=1, reindex=False)
        m2 = float(fc.mean.values[0, 0]); s2 = float(np.sqrt(fc.variance.values[0, 0]))
        nu, lam = float(f.params["eta"]), float(f.params["lambda"])
        out.update({"k_ok": getattr(f, "convergence_flag", 0) == 0, "k_retry": retry,
                    "k_sig": s2 / 100.0, "k_nu": nu, "k_lam": lam})
        for q in LEV:
            out[f"skt_VaR_{q}"] = (m2 + s2 * skt_var(q, nu, lam)) / 100.0
            out[f"skt_ES_{q}"] = (m2 + s2 * skt_es(q, nu, lam)) / 100.0
    except Exception:
        out["k_ok"] = False
    return out


if __name__ == "__main__":
    t0 = time.time()
    rows = Parallel(n_jobs=2)(delayed(one)(i) for i in range(sb.N, sb.G))
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    write_forecasts(df, "results/bench_asym_forecasts.csv")
    import arch, scipy
    print(f"ASYMMETRIC-TAIL BENCHMARKS: {len(df)} forecasts in {(time.time()-t0)/60:.1f} min")
    print(f"Python {platform.python_version()} ({platform.system()}), arch {arch.__version__}, scipy {scipy.__version__}")
    print(f"skew-t fit converged on {df['k_ok'].mean():.4f} of windows ({int((~df['k_ok']).sum())} not converged); "
          f"retries: {int((df['k_retry'] == 1).sum())} with a tighter tolerance, {int((df['k_retry'] == 2).sum())} from t starting values; "
          f"median nu {df['k_nu'].median():.2f}, median lambda {df['k_lam'].median():.3f}")
