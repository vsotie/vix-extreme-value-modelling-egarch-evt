"""Window-length study, part 1: 1000-day against 1500-day estimation windows.

Internal check of the 1,000-day window used in the backtest; not reported in
the dissertation. Reads results/window_forecasts.csv (written by
sens_window.py), which holds EGARCH-EVT and GARCH-EVT forecasts from both
window lengths on the same 3,433 dates.

Run from Code/:  python src/window_comparison.py
Output: results/log_20260912_window_comparison.txt and
        Graphs/window_regime_pinball.png
"""
import os; os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

REG = [("Recovery '13-16", "2013-01-01", "2016-12-31"),
       ("Volmageddon '17-19", "2017-01-01", "2019-12-31"),
       ("COVID '20-21", "2020-01-01", "2021-12-31"),
       ("Tightening '22-24", "2022-01-01", "2024-12-31"),
       ("Recent '25-26", "2025-01-01", "2026-12-31")]


def pinball(rv, v, a):
    d = rv - v
    return np.where(d >= 0, a * d, (a - 1) * d)


def kupiec(x, n, p):
    """Kupiec unconditional-coverage LR test; returns (LR, p-value)."""
    if x == 0:
        lr = -2 * (n * np.log(1 - p))
    else:
        pi = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def dm(d, lag=5):
    """Diebold-Mariano statistic with a Bartlett long-run variance (as in
    sens_backtest.py); returns (statistic, two-sided p-value)."""
    d = np.asarray(d); T = len(d); m = d.mean(); s = np.mean((d - m) ** 2)
    for L in range(1, lag + 1):
        c = np.mean((d[L:] - m) * (d[:-L] - m)); s += 2 * (1 - L / (lag + 1)) * c
    t = m / np.sqrt(s / T)
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def main():
    df = pd.read_csv("results/window_forecasts.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    r = df.realized.values; n = len(df)

    print("WINDOW-LENGTH SENSITIVITY: 1000 vs 1500 days")
    print(f"{n} forecasts, {df.date.min().date()} -> {df.date.max().date()} (common period; NO Euro crisis)")
    print()
    print("=== Overall coverage (viol / exp; Kupiec p) ===")
    for q in (0.95, 0.99):
        print(f" q={q} (exp {n * (1 - q):.0f}):")
        for m, lab in (("e", "EGARCH"), ("g", "GARCH ")):
            for w in ("1000", "1500"):
                v = df[f"{m}_{w}_VaR_{q}"].values; x = int((r > v).sum()); _, p = kupiec(x, n, 1 - q)
                print(f"   {lab} n={w}: viol={x} rate={100 * x / n:.3f}%  Kupiec p={p:.4f}  "
                      f"pinball={pinball(r, v, q).mean() * 1e4:.4f}")
    print()
    print("=== Does longer window change accuracy? DM on pinball, 1500 vs 1000 (neg => 1500 better) ===")
    for m, lab in (("e", "EGARCH-EVT"), ("g", "GARCH-EVT")):
        for q in (0.95, 0.99):
            d = (pinball(r, df[f"{m}_1500_VaR_{q}"].values, q)
                 - pinball(r, df[f"{m}_1000_VaR_{q}"].values, q))
            t, p = dm(d)
            print(f"   {lab} q={q}: mean diff {d.mean() * 1e4:+.4f}  DM {t:+.3f}  p {p:.4f}")
    print()
    print("=== By-regime breach ratio (actual/expected, 99%) and pinball, EGARCH ===")
    print(f"{'regime':<18}{'days':>7} | ratio 1000 / 1500 | pinball 1000 / 1500")
    gap = []                                   # pinball 1500 - 1000, for the figure
    for lab, a, b in REG:
        mk = ((df.date >= a) & (df.date <= b)).values; nd = int(mk.sum()); rr = r[mk]
        rat = [(rr > df.loc[mk, f"e_{w}_VaR_0.99"].values).sum() / (nd * 0.01) for w in ("1000", "1500")]
        pb = [pinball(rr, df.loc[mk, f"e_{w}_VaR_0.99"].values, 0.99).mean() * 1e4 for w in ("1000", "1500")]
        gap.append(pb[1] - pb[0])
        s = f"{rat[0]:.2f} / {rat[1]:.2f}"
        print(f"{lab:<18}{nd:>7} | {s:<18}| {pb[0]:.3f} / {pb[1]:.3f}  "
              f"({'1500 better' if pb[1] < pb[0] else '1000 better'})")
    print()
    print("=== EGARCH-vs-GARCH gap at each window (mean pinball diff EG-GA x1e4, 99%; neg=>EGARCH better) ===")
    for w in ("1000", "1500"):
        d = (pinball(r, df[f"e_{w}_VaR_0.99"].values, 0.99)
             - pinball(r, df[f"g_{w}_VaR_0.99"].values, 0.99)).mean() * 1e4
        print(f"   n={w}: EG-GA = {d:+.4f}")
    print()
    print("=== Forecast character: VaR_1500 - VaR_1000 (EGARCH 99%, return units) ===")
    a = df["e_1000_VaR_0.99"]; b = df["e_1500_VaR_0.99"]; d = b - a
    print(f"   mean {d.mean():+.5f}  sd {d.std():.5f}  |  sd(VaR_1000)={a.std():.4f}  sd(VaR_1500)={b.std():.4f}")
    print(f"   corr(VaR_1000,VaR_1500)={np.corrcoef(a, b)[0, 1]:.4f}")

    # ---- Figure: pinball difference 1500 - 1000 by regime ----
    labels = ["Recovery '13-16\n(calm)", "Volmageddon '17-19", "COVID '20-21\n(fast crash)",
              "Tightening '22-24", "Recent '25-26"]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.barh(y, gap, color=['#26786A' if g < 0 else '#B8412B' for g in gap], height=0.6)
    ax.axvline(0, color='0.3', lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    for yi, g in zip(y, gap):
        ax.text(g + (0.04 if g > 0 else -0.04), yi, f"{g:+.2f}", va='center',
                ha='left' if g > 0 else 'right', fontsize=8.5, color='0.25')
    ax.set_xlabel("pinball loss: 1500-day $-$ 1000-day window ($\\times10^{-4}$), EGARCH 99% VaR")
    ax.set_title("Longer window helps slightly in calm, hurts in fast regime shifts", fontsize=12)
    ax.text(0.98, 0.06, "right = 1500 worse (adaptivity cost)", transform=ax.transAxes,
            ha='right', fontsize=8.5, color='#B8412B', style='italic')
    ax.grid(axis='x', alpha=0.3); ax.set_xlim(-0.35, 0.9)
    fig.tight_layout(); fig.savefig("Graphs/window_regime_pinball.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
