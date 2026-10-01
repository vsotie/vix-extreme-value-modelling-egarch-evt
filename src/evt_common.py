"""Shared set-up for the EVT diagnostic scripts.

Loads the cached VIX closes, forms daily log returns and fits the adopted
AR(1)-EGARCH(1,1)-t filter exactly as main.py does (returns scaled by 100),
returning the returns, the standardised residuals and the fit.

Run every script from the Code/ directory, e.g. `python src/evt_mrl_regions.py`.
"""
import numpy as np
import pandas as pd
from arch import arch_model

DATA = "data/vix_raw.csv"


def load_returns(path=DATA, verbose=True):
    vix = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    if verbose:
        print(f"VIX loaded from cache: {path} ({len(vix)} rows, "
              f"{vix.index.min().date()} to {vix.index.max().date()})")
    return np.log(vix["Close"] / vix["Close"].shift(1)).dropna()


def fit_adopted(r):
    """AR(1)-EGARCH(1,1) with Student-t innovations on returns x100."""
    return arch_model(r * 100, mean="AR", lags=1, vol="EGARCH",
                      p=1, o=1, q=1, dist="t").fit(disp="off")


def residuals(fit):
    return (fit.resid / fit.conditional_volatility).dropna()


def mean_excess(x, u):
    """Excesses x - u of the observations above u."""
    return x[x > u] - u
