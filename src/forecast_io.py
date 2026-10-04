"""Reading and writing the forecast files without redistributing realised returns.

The series behind the dissertation is copyrighted by Cboe, so the published
forecast files in results/ do not contain the realised log returns. A realised
return can be reconstructed exactly from consecutive closes, and the closes can in
turn be reconstructed from the returns, so the column is left out of every public
file. The evaluation scripts rejoin it, by date, from the user's own copy of the
data (data/vix_raw.csv, which is not part of the repository).

write_forecasts(df, path) writes a forecast table without the 'realized' column.
read_forecasts(path, **kw) reads one and, if the column is absent, restores it.
"""
import os
import numpy as np
import pandas as pd

DATA = "data/vix_raw.csv"
_cache = {}
_NO_RETURN_FILES = {"forecasts_975.csv"}  # never held realised returns


def realised_returns(path=DATA):
    """Daily log returns indexed by date, from the local copy of the closes."""
    if path not in _cache:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} is needed to restore the realised returns that the "
                "published forecast files leave out. Obtain the series as described "
                "in the README and save it there.")
        vix = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        _cache[path] = np.log(vix["Close"] / vix["Close"].shift(1)).dropna()
    return _cache[path]


def read_forecasts(path, **kw):
    """Read a forecast file; restore 'realized' by date when it was left out."""
    df = pd.read_csv(path, **kw)
    if "realized" not in df.columns and os.path.basename(path) not in _NO_RETURN_FILES:
        r = realised_returns()
        dates = pd.to_datetime(df["date"])
        if not dates.isin(r.index).all():
            raise ValueError("a forecast date is missing from data/vix_raw.csv")
        df.insert(1, "realized", r.reindex(dates).to_numpy())
    return df


def write_forecasts(df, path):
    """Write a forecast table with the realised-return column removed."""
    df.drop(columns=["realized"], errors="ignore").to_csv(path, index=False)
