"""Objective V, out-of-sample sensitivity of the rolling backtest.
V-a: EGARCH(1,1,1)-EVT VaR at threshold quantiles 0.85/0.875/0.90/0.925.
V-b: EGARCH(1,1,1) vs the richest alternative EGARCH(2,1,2), both EVT at 0.90.
One pass; per window fit the baseline and the alternative filter."""
import os; os.environ.setdefault("MPLBACKEND","Agg")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time
from arch import arch_model
from scipy import stats
from utils import gpd_fit, gpd_tail_measures
from joblib import Parallel, delayed
from forecast_io import read_forecasts, write_forecasts

N=1000; THRS=[0.850,0.875,0.900,0.925]; LEV=[0.95,0.99]; ALT=(2,1,2)
vix=pd.read_csv("data/vix_raw.csv",parse_dates=["Date"]).set_index("Date")
r=np.log(vix["Close"]/vix["Close"].shift(1)).dropna()
rr=r.values; dates=r.index; G=len(rr)

def fit_egarch(w100,p,o,q):
    f=arch_model(w100,mean="AR",lags=1,vol="EGARCH",p=p,o=o,q=q,dist="t").fit(disp="off",show_warning=False)
    fc=f.forecast(horizon=1,reindex=False)
    mu=float(fc.mean.values[0,0]); sig=float(np.sqrt(fc.variance.values[0,0]))
    z=(f.resid/f.conditional_volatility); z=np.asarray(z[np.isfinite(z)],float)
    return mu,sig,z

def varset(z,mu,sig,thr,levels):
    u=np.quantile(z,thr); exc=z[z>u]-u
    gf=gpd_fit(exc); tm=gpd_tail_measures(u,gf["xi"],gf["beta"],n=len(z),n_u=len(exc),qs=tuple(levels))
    return {q:(mu+sig*tm.loc[q,"z_q"])/100.0 for q in levels}

def one(i):
    w100=rr[i-N:i]*100.0; out={"date":dates[i],"realized":rr[i]}
    mu,sig,z=fit_egarch(w100,1,1,1)
    for thr in THRS:
        v=varset(z,mu,sig,thr,LEV)
        for q in LEV: out[f"base_{thr:.3f}_{q}"]=v[q]
    mu2,sig2,z2=fit_egarch(w100,*ALT)
    v2=varset(z2,mu2,sig2,0.900,LEV)
    for q in LEV: out[f"alt_{q}"]=v2[q]
    return out

def kupiec(x,n,p):
    if x==0: lr=-2*n*np.log(1-p)
    else:
        pi=x/n; lr=-2*((n-x)*np.log(1-p)+x*np.log(p)-(n-x)*np.log(1-pi)-x*np.log(pi))
    return lr,1-stats.chi2.cdf(lr,1)
def pinball(rv,v,a): d=rv-v; return np.where(d>=0,a*d,(a-1)*d)
def dm(d,lag=5):
    d=np.asarray(d,float);T=len(d);m=d.mean();s=np.var(d)
    for L in range(1,lag+1):
        c=np.mean((d[L:]-m)*(d[:-L]-m));s+=2*(1-L/(lag+1))*c
    t=m/np.sqrt(s/T);return t,2*(1-stats.norm.cdf(abs(t)))

if __name__=="__main__":
    t0=time.time()
    rows=Parallel(n_jobs=2,verbose=1)(delayed(one)(i) for i in range(N,G))
    df=pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    write_forecasts(df, "results/sens_forecasts.csv")
    rv=df["realized"].values; n=len(rv)
    print(f"\nDONE {n} forecasts in {(time.time()-t0)/60:.1f} min\n")
    print("=== V-a: THRESHOLD sensitivity, EGARCH(1,1,1)-EVT ===")
    for q in LEV:
        print(f" q={q} (target {100*(1-q):.1f}%, exp {n*(1-q):.1f}):")
        for thr in THRS:
            v=df[f"base_{thr:.3f}_{q}"].values; x=int((rv>v).sum()); lr,p=kupiec(x,n,1-q)
            print(f"   thr={thr:.3f} (k~{int(round((1-thr)*999))}): viol={x} rate={100*x/n:.3f}% Kupiec_p={p:.4f} pinball={pinball(rv,v,q).mean()*1e4:.4f}")
    print("\n=== V-b: LAG-STRUCTURE sensitivity, EGARCH(1,1,1) vs EGARCH(2,1,2), thr 0.90 ===")
    for q in LEV:
        vb=df[f"base_0.900_{q}"].values; va=df[f"alt_{q}"].values
        xb=int((rv>vb).sum()); xa=int((rv>va).sum())
        st_,p=dm(pinball(rv,va,q)-pinball(rv,vb,q))
        _,pb=kupiec(xb,n,1-q); _,pa=kupiec(xa,n,1-q)
        print(f" q={q}: (1,1,1) viol={xb} Kupiec_p={pb:.3f} pinball={pinball(rv,vb,q).mean()*1e4:.4f} | "
              f"(2,1,2) viol={xa} Kupiec_p={pa:.3f} pinball={pinball(rv,va,q).mean()*1e4:.4f} | DM(alt-base) {st_:.3f} p {p:.4f}")
    print("\n(DM ~0 / p>>0.05 => lag structure does not change forecast performance)")
