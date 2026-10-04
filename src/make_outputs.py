"""Figures of the rolling backtest (Chapter 7) and the numbers quoted with them.

Run from Code/ after the backtest:  python src/make_outputs.py
Output: Graphs/backtest_var_path.png, Graphs/backtest_var_path_full.png;
printed summary = results/log_20260927_make_outputs.txt.
"""
import os; os.environ.setdefault("MPLBACKEND","Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from forecast_io import read_forecasts, write_forecasts
df=read_forecasts("results/forecasts.csv",parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# ---- Figure 7.1: adaptive vs static VaR with violations, 2019-2021 (calm -> COVID) ----
m=(df.date>=pd.Timestamp("2019-01-01"))&(df.date<=pd.Timestamp("2021-06-30"))
d=df.loc[m]; r=d["realized"].values; dt=d["date"].values
ve=d["eevt_VaR_0.99"].values; vs=d["s_VaR_0.99"].values
viol=r>ve
fig,ax=plt.subplots(figsize=(10,4.6))
ax.plot(dt,r,color="0.6",lw=0.7,label="VIX log return")
ax.plot(dt,ve,color="steelblue",lw=1.3,label="EGARCH-EVT 99% VaR")
ax.plot(dt,vs,color="firebrick",lw=1.2,ls="--",label="Static POT 99% VaR")
ax.scatter(dt[viol],r[viol],s=26,color="firebrick",zorder=5,label=f"Violation of EGARCH-EVT VaR ({viol.sum()})")
ax.set_ylabel("Daily log return",fontsize=10); ax.set_xlabel("")
ax.set_title("One-day-ahead 99% VaR, January 2019 to June 2021",fontsize=11)
ax.legend(fontsize=8.5,frameon=False,loc="upper left"); ax.grid(True,alpha=0.3)
fig.tight_layout(); fig.savefig("Graphs/backtest_var_path.png",dpi=150,bbox_inches="tight")
print("Fig 7.1 saved: Graphs/backtest_var_path.png  window",str(d.date.min().date()),"->",str(d.date.max().date()),"viol",int(viol.sum()))
print(f"  EGARCH-EVT 99% VaR in window: min {ve.min():.4f}  max {ve.max():.4f} on {pd.Timestamp(dt[ve.argmax()]).date()}")
print(f"  Static POT 99% VaR in window: min {vs.min():.4f}  max {vs.max():.4f}")
print("  EGARCH-EVT violations in window (date, return, VaR, previous day's return):")
for t_,rr,vv in zip(dt[viol],r[viol],ve[viol]):
    j=df.index[df.date==pd.Timestamp(t_)][0]
    print(f"    {pd.Timestamp(t_).date()}  r {rr:+.4f}  VaR {vv:.4f}  previous r {df.realized[j-1]:+.4f}")
sv=r>vs
print(f"  Static POT violations in window: {int(sv.sum())}")
for t_,rr,vv in zip(dt[sv],r[sv],vs[sv]):
    print(f"    {pd.Timestamp(t_).date()}  r {rr:+.4f}  VaR {vv:.4f}")
full_e=df["eevt_VaR_0.99"]; full_s=df["s_VaR_0.99"]
print(f"Full backtest 99% VaR, EGARCH-EVT: sd {full_e.std():.4f}  min {full_e.min():.4f}  max {full_e.max():.4f}")
print(f"Full backtest 99% VaR, Static POT:  sd {full_s.std():.4f}  min {full_s.min():.4f}  max {full_s.max():.4f}")

# ---- also full-sample conditional 99% VaR path (context) ----
fig,ax=plt.subplots(figsize=(11,4))
ax.plot(df.date,df.realized,color="0.7",lw=0.5)
ax.plot(df.date,df["eevt_VaR_0.99"],color="steelblue",lw=0.8,label="EGARCH-EVT 99% VaR")
vio=df.realized>df["eevt_VaR_0.99"]
ax.scatter(df.date[vio],df.realized[vio],s=10,color="firebrick",zorder=5,label=f"violations ({int(vio.sum())})")
ax.set_title("EGARCH-EVT 99% VaR and its violations, December 2010 to June 2026",fontsize=11)
ax.legend(fontsize=8.5,frameon=False); ax.grid(True,alpha=0.3)
fig.tight_layout(); fig.savefig("Graphs/backtest_var_path_full.png",dpi=150,bbox_inches="tight")
print("full path saved")

# ---- the four episodes of Chapter 1: forecasts issued the evening before ----
# ---- 99% violations: every EGARCH-EVT date, and consecutive-day pairs by model ----
print("EGARCH-EVT 99% violations (date, return, VaR):")
for _,row in df[df.realized>df["eevt_VaR_0.99"]].iterrows():
    print(f"  {row.date.date()}  r {row.realized:+.4f}  VaR {row['eevt_VaR_0.99']:.4f}")
print("Violations of the 99% VaR on consecutive forecast days (the pairs the first-order")
print("Christoffersen independence test counts):")
for mm,nm in [("eevt","EGARCH-EVT"),("gevt","GARCH-EVT"),("s","Static POT"),("et","EGARCH-t"),("gt","GARCH-t")]:
    v=(df.realized>df[f"{mm}_VaR_0.99"]).values
    pairs=[f"{df.date[i-1].date()}/{df.date[i].date()}" for i in range(1,len(df)) if v[i] and v[i-1]]
    print(f"  {nm:<11} {len(pairs)} pair(s): {', '.join(pairs) if pairs else '-'}")
print("Chapter 1 episodes: 99% VaR issued the previous close, and the next day's VaR")
for day in ["2018-02-05","2024-08-05","2024-12-18","2025-04-03","2025-04-04"]:
    i=df.index[df.date==pd.Timestamp(day)][0]
    row=df.loc[i]; nxt=df.loc[i+1]; prv=df.loc[i-1]
    print(f"  {day}: previous day {prv.date.date()} r {prv.realized:+.4f}, EGARCH-EVT VaR that day {prv['eevt_VaR_0.99']:.4f}")
    print(f"  {day}: r {row.realized:+.4f} | EGARCH-EVT VaR {row['eevt_VaR_0.99']:.4f} "
          f"(viol {int(row.realized>row['eevt_VaR_0.99'])}), next day {nxt['eevt_VaR_0.99']:.4f} | "
          f"Static POT VaR {row['s_VaR_0.99']:.4f} (viol {int(row.realized>row['s_VaR_0.99'])}), "
          f"next day {nxt['s_VaR_0.99']:.4f}")
