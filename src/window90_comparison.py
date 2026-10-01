"""Window-length study, part 2: the 90-day window as a lower bound.

Internal check of the 1,000-day window used in the backtest; not reported in
the dissertation. Reads results/window90_forecasts.csv (written by
sens_window90.py) and, for comparison on the same dates,
results/window_forecasts.csv (written by sens_window.py).

Run from Code/:  python src/window90_comparison.py
Output: results/log_20260912_window90_comparison.txt and
        Graphs/window_length_curve.png
"""
import os; os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from window_comparison import kupiec


def main():
    d9 = pd.read_csv("results/window90_forecasts.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    dw = pd.read_csv("results/window_forecasts.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    assert (d9.date.values == dw.date.values).all(), "the two files must cover the same dates"
    n = len(d9)
    nonconv = int((d9.conv == 0).sum())
    degenerate = int((d9.xi.abs() > 1).sum())
    k_valid = d9.k.dropna().unique()

    print(f"90-DAY WINDOW: estimator breakdown ({n} windows, {d9.date.min().year}-{d9.date.max().year})")
    print(f" EGARCH non-convergence: {100 * nonconv / n:.0f}% ({nonconv}) | hard failures: "
          f"{int(d9.fail_fit.sum())} filter, {int(d9.fail_gpd.sum())} GPD")
    print(f" GPD exceedances k={', '.join(f'{k:.0f}' for k in k_valid)} every window | "
          f"|xi|>1 (degenerate/ES undefined): {100 * degenerate / n:.0f}% ({degenerate})")
    print(f" nu collapsed to normal (>=499): {int((d9.nu >= 499).sum())} | nu<2.1: {int((d9.nu < 2.1).sum())}")
    print()
    ok = d9["e_90_VaR_0.99"].notna().values
    print(f"Violation rate vs target, EGARCH (n=90: the {ok.sum()} windows with a forecast):")
    rates = {}
    for q in (0.95, 0.99):
        for w in ("90", "1000", "1500"):
            if w == "90":
                v = d9[f"e_90_VaR_{q}"].values[ok]; rv = d9.realized.values[ok]
            else:
                v = dw[f"e_{w}_VaR_{q}"].values; rv = dw.realized.values
            x = int((rv > v).sum()); m = len(rv); _, p = kupiec(x, m, 1 - q)
            rates[(q, w)] = 100 * x / m
            print(f"  q={q} n={w}: {100 * x / m:.2f}% (target {100 * (1 - q):.0f}%)  Kupiec p={p:.4f}")
    print()
    with np.errstate(over="ignore", invalid="ignore"):
        sd90 = float(np.std(d9["e_90_VaR_0.99"].values[ok], ddof=1))
    vmax = float(np.nanmax(d9["e_90_VaR_0.99"].values))
    print(f"Forecast sd (99% VaR): n=90 {'diverges (inf' if not np.isfinite(sd90) else f'sd={sd90:.3f} ('}; "
          f"largest forecast {vmax:.1e}; degenerate GPD |xi|>1 gives near-infinite quantiles);")
    print(f"  n=1000 sd={dw['e_1000_VaR_0.99'].std():.3f}, n=1500 sd={dw['e_1500_VaR_0.99'].std():.3f}. "
          f"Pinball loss at n=90 is dominated by these outliers.")

    # ---- Figure: 99% violation rate against window length ----
    wins = ("90", "1000", "1500"); vals = [rates[(0.99, w)] for w in wins]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(3)
    ax.bar(x, vals, width=0.55, color=['#B8412B', '#26786A', '#324C7E'])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.05, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='0.2')
    ax.axhline(1.0, color='0.3', lw=1, ls='--')
    ax.text(2.4, 1.05, "target 1%", fontsize=8.5, color='0.3', ha='left', va='bottom')
    ax.text(0.35, 0.6 * max(vals),
            f"{100 * nonconv / n:.0f}% non-convergence\n{100 * degenerate / n:.0f}% degenerate tail (|ξ|>1)\n"
            f"k={', '.join(f'{k:.0f}' for k in k_valid)} exceedances", fontsize=8.5, color='#B8412B', va='center')
    ax.set_xticks(x); ax.set_xticklabels(wins)
    ax.set_xlabel("estimation window (days)"); ax.set_ylabel("99% VaR violation rate")
    ax.set_ylim(0, 1.11 * max(vals))
    ax.set_title(f"Too short breaks the estimator: 90-day window over-breaches {vals[0]:.0f}x", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig("Graphs/window_length_curve.png", dpi=150, bbox_inches="tight")
    print()
    print("saved Graphs/window_length_curve.png")


if __name__ == "__main__":
    main()
