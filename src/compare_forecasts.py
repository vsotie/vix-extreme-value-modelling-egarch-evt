"""Compare a rerun of a rolling backtest with the published forecast file.

For every numeric column: the number of dates on which the two files differ
at all, by more than 1e-6 and by more than 1e-3 in relative terms, and the
largest relative difference. For every VaR column: the number of dates
on which the violation indicator (realised return above the VaR) differs.

Run from Code/, after rerunning the backtest into another folder, e.g.
    python src/compare_forecasts.py results/forecasts.csv ../rerun/results/forecasts.csv
"""
import sys
import numpy as np
import pandas as pd
from forecast_io import read_forecasts, write_forecasts


def compare(path_a, path_b):
    a = read_forecasts(path_a); b = read_forecasts(path_b)
    print(f"published: {path_a}  ({len(a)} rows)")
    print(f"rerun:     {path_b}  ({len(b)} rows)")
    assert list(a.columns) == list(b.columns), "column sets differ"
    assert (a["date"] == b["date"]).all(), "dates differ"
    rows = []
    for c in a.columns:
        if c == "date":
            continue
        x = a[c].to_numpy(dtype=float); y = b[c].to_numpy(dtype=float)
        nan_diff = int((np.isfinite(x) != np.isfinite(y)).sum())
        both = np.isfinite(x) & np.isfinite(y)
        rel = np.abs(x[both] - y[both]) / np.maximum(np.abs(x[both]), 1e-12)
        flips = ""
        if "_VaR_" in c:
            r = a["realized"].to_numpy(dtype=float)
            flips = int(((r > x) != (r > y)).sum())
        rows.append({"column": c, "dates differing": int((rel > 0).sum()) + nan_diff,
                     "rel diff > 1e-6": int((rel > 1e-6).sum()),
                     "rel diff > 1e-3": int((rel > 1e-3).sum()),
                     "max rel diff": f"{rel.max():.1e}" if rel.size else "n/a",
                     "violation flips": flips})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    flips = sum(v for v in t["violation flips"] if v != "")
    print(f"\nrows identical in every column: {int((a.fillna(0) == b.fillna(0)).all(axis=1).sum())} of {len(a)}")
    print(f"violation flips over all VaR columns: {flips}")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
