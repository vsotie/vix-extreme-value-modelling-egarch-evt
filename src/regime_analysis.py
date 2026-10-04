"""Regime-based data observation on the backtest forecasts (Ch.7 supplement).
Surfaces nuances the aggregate coverage/accuracy tests hide.

Run from Code/ after the backtest:  python src/regime_analysis.py
Printed output = results/log_20260912_regime_analysis.txt.
Note: the first regime starts on 2010-01-01, so it also contains the seven
forecasts of 22-31 Dec 2010 (the first forecast is 22 Dec 2010); none of the
models has a violation on those days."""
import os; os.environ.setdefault("MPLBACKEND","Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from forecast_io import read_forecasts, write_forecasts
df=read_forecasts("results/forecasts.csv",parse_dates=["date"]).sort_values("date").reset_index(drop=True)
r=df["realized"].values
def pinball(rv,v,a): d=rv-v; return np.where(d>=0,a*d,(a-1)*d)
REG=[("Euro '11-12","2010-01-01","2012-12-31"),("Recovery '13-16","2013-01-01","2016-12-31"),
     ("Volmageddon '17-19","2017-01-01","2019-12-31"),("COVID '20-21","2020-01-01","2021-12-31"),
     ("Tightening '22-24","2022-01-01","2024-12-31"),("Recent '25-26","2025-01-01","2026-12-31")]
labels=[x[0] for x in REG]
# Figure labels: the report names the backtest regimes by years, as in Table 3.3;
# the first starts at the first forecast, 22 Dec 2010. The printed tables keep
# the short labels above.
FIG_LABELS=["2010\u201312\n(from 22 Dec 2010)","2013\u201316\nRecovery","2017\u201319\nLow volatility",
            "2020\u201321\nCOVID-19","2022\u201324\nTightening","2025\u201326\nRecent"]
MODELS=('eevt','gevt','s')   # EGARCH-EVT, GARCH-EVT, Static POT
def in_regime(a,b): return ((df.date>=a)&(df.date<=b)).values

print("REGIME-BASED DATA OBSERVATION ON BACKTEST FORECASTS")
print(f"Source: results/forecasts.csv ({len(df):,} one-day forecasts, {df.date.min().year}-{df.date.max().year})")
print()

# ---- Breach ratio (actual / expected violations) ----
print("=== Breach ratio (actual/expected) by regime & model ===")
print(f"{'regime':<18}{'days':>7} | q=0.95 EGEVT/GAEVT/Static    | q=0.99 EGEVT/GAEVT/Static")
ratio={m:[] for m in MODELS}          # 99% ratios, used by Chart 2
for lab,a,b in REG:
    m=in_regime(a,b); nd=int(m.sum()); rr=r[m]; parts=[]
    for q in (0.95,0.99):
        exp=nd*(1-q)
        rs=[(rr>df.loc[m,f'{mm}_VaR_{q}'].values).sum()/exp for mm in MODELS]
        parts.append(" / ".join(f"{x:.2f}" for x in rs))
        if q==0.99:
            for mm,x in zip(MODELS,rs): ratio[mm].append(x)
    print(f"{lab:<18}{nd:>7} | {parts[0]:<28} | {parts[1]}")
print()

# ---- Pinball loss at 99% ----
print("=== EGARCH vs GARCH vs Static: mean pinball loss x1e4 (q=0.99) ===")
diff=[]                               # EGARCH - GARCH, used by Chart 1
for lab,a,b in REG:
    m=in_regime(a,b); rr=r[m]
    L={mm:pinball(rr,df.loc[m,f'{mm}_VaR_0.99'].values,0.99).mean()*1e4 for mm in MODELS}
    d=L['eevt']-L['gevt']; diff.append(d)
    print(f"  {lab:<20} EGARCH {L['eevt']:.3f}  GARCH {L['gevt']:.3f}  Static {L['s']:.3f} | EG-GA {d:+.3f} ({'EGARCH better' if d<0 else 'GARCH better'})")
print()

# ---- ES shortfall on EGARCH-EVT breach days, in units of the forecast sigma ----
print("=== ES shortfall by regime: mean (realized-ES)/sigma on EGARCH-EVT breach days, q=0.99 ===")
print("  (positive => ES too low; note small breach counts in late regimes)")
for lab,a,b in REG:
    sub=df.loc[in_regime(a,b)]; br=sub.realized>sub['eevt_VaR_0.99']
    disc=((sub.realized-sub['eevt_ES_0.99'])/sub.e_sig)[br]
    print(f"  {lab:<20} breaches {int(br.sum()):>2}  mean_disc {disc.mean():+.3f}")
print()
print("Charts: Graphs/regime_egarch_vs_garch.png, Graphs/regime_breach_ratio.png")

# ---- 97.5% values for the charts (results/forecasts_975.csv, written by level975.py) ----
# Added 28 Sep 2026. Nothing is printed here, so the printed log is unchanged.
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
d975=pd.read_csv("results/forecasts_975.csv",parse_dates=["date"],float_precision="round_trip").sort_values("date").reset_index(drop=True)
assert (d975.date.values==df.date.values).all()
ratio975={m:[] for m in MODELS}; diff975=[]
for lab,a,b in REG:
    m=in_regime(a,b); nd=int(m.sum()); rq=r[m]
    for mm in MODELS:
        ratio975[mm].append((rq>d975.loc[m,f'{mm}_VaR_0.975'].values).sum()/(nd*0.025))
    L={mm:pinball(rq,d975.loc[m,f'{mm}_VaR_0.975'].values,0.975).mean()*1e4 for mm in ('eevt','gevt')}
    diff975.append(L['eevt']-L['gevt'])

def minus(v): return f"{v:+.2f}".replace("-","−")

# ---- Chart 1: EGARCH - GARCH pinball by regime, 99% (dark) and 97.5% (light) ----
y=np.arange(len(labels))[::-1]; h=0.36
fig,ax=plt.subplots(figsize=(8.4,5.4))
for vals,off,dark in [(diff,+h/2,True),(diff975,-h/2,False)]:
    cols=[('#26786A' if d<0 else '#B8412B') if dark else ('#9CCBC0' if d<0 else '#E6A796') for d in vals]
    ax.barh(y+off,vals,color=cols,height=h)
    for yi,d in zip(y+off,vals):
        ax.text(d+(0.08 if d>0 else -0.08),yi,minus(d),va='center',
                ha='left' if d>0 else 'right',fontsize=8,color='0.25')
ax.axvline(0,color='0.3',lw=1)
ax.set_yticks(y); ax.set_yticklabels(FIG_LABELS,fontsize=9)
ax.set_xlabel("Mean VaR pinball loss of EGARCH-EVT minus that of GARCH-EVT ($\\times10^{-4}$)")
ax.set_title("Difference in VaR pinball loss between the two filters, by regime",fontsize=12)
ax.text(0.02,0.97,"Negative: EGARCH-EVT more accurate",transform=ax.transAxes,
        ha='left',va='top',fontsize=8.5,color='#26786A',style='italic')
ax.legend(handles=[Patch(color='0.35',label='99% VaR'),Patch(color='0.75',label='97.5% VaR')],
          fontsize=9,frameon=False,loc='lower left')
ax.grid(axis='x',alpha=0.3); ax.set_xlim(-5.2,3.4)
fig.tight_layout(); fig.savefig("Graphs/regime_egarch_vs_garch.png",dpi=150,bbox_inches="tight")

# ---- Chart 2: violation ratio by regime x model; bars 99%, diamonds 97.5% ----
fig,ax=plt.subplots(figsize=(9.2,4.4))
x=np.arange(len(labels)); w=0.26
styles=[('eevt','EGARCH-EVT','#324C7E'),('gevt','GARCH-EVT','#8A93A2'),('s','Static POT','#B8412B')]
for i,(mm,nm,c) in enumerate(styles):
    ax.bar(x+(i-1)*w,ratio[mm],w,label=nm,color=c)
    ax.plot(x+(i-1)*w,ratio975[mm],'D',ms=5.5,mfc='white',mec='black',mew=1.1,zorder=3)
ax.axhline(1.0,color='0.3',lw=1,ls='--')
ax.set_xticks(x); ax.set_xticklabels(FIG_LABELS,fontsize=8.5)
ax.set_ylabel("Breaches / expected")
ax.set_title("VaR breach ratio by regime: bars 99%, diamonds 97.5%",fontsize=12)
handles,_=ax.get_legend_handles_labels()
handles+= [Line2D([],[],ls='none',marker='D',ms=5.5,mfc='white',mec='black',mew=1.1,label='97.5% VaR'),
           Line2D([],[],color='0.3',lw=1,ls='--',label='target = 1.0')]
ax.legend(handles=handles,fontsize=9,frameon=False,ncol=5,loc='upper center',bbox_to_anchor=(0.5,-0.17))
ax.grid(axis='y',alpha=0.3)
fig.tight_layout(); fig.savefig("Graphs/regime_breach_ratio.png",dpi=150,bbox_inches="tight")
