"""90-day window: lower-bound of the window-length study.
At n=90 the EGARCH-t filter fits 7 params on 90 obs and the GPD tail on ~9
exceedances (90th pct). Track failures; compare to 1000/1500 on identical dates."""
import os; os.environ.setdefault("MPLBACKEND","Agg")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time
from arch import arch_model
from utils import gpd_fit, gpd_tail_measures
from joblib import Parallel, delayed
from forecast_io import read_forecasts, write_forecasts
N=90; QT=0.90; LEV=[0.95,0.99]
vix=pd.read_csv("data/vix_raw.csv",parse_dates=["Date"]).set_index("Date")
r=np.log(vix["Close"]/vix["Close"].shift(1)).dropna(); rr=r.values; dates=r.index; G=len(rr)

def one(i):
    o={"date":dates[i],"realized":rr[i],"fail_fit":0,"fail_gpd":0,"nu":np.nan,"k":np.nan,"xi":np.nan}
    w100=rr[i-N:i]*100.0
    try:
        f=arch_model(w100,mean="AR",lags=1,vol="EGARCH",p=1,o=1,q=1,dist="t").fit(disp="off",show_warning=False)
        conv=getattr(f,"convergence_flag",0)==0
        fc=f.forecast(horizon=1,reindex=False)
        mu=float(fc.mean.values[0,0]); sig=float(np.sqrt(fc.variance.values[0,0]))
        z=(f.resid/f.conditional_volatility); z=np.asarray(z[np.isfinite(z)],float)
        o["nu"]=float(f.params["nu"]); o["conv"]=int(conv)
        if not (np.isfinite(mu) and np.isfinite(sig) and sig>0): raise ValueError
    except Exception:
        o["fail_fit"]=1; return o
    try:
        u=np.quantile(z,QT); exc=z[z>u]-u; o["k"]=len(exc)
        gf=gpd_fit(exc); o["xi"]=gf["xi"]
        tm=gpd_tail_measures(u,gf["xi"],gf["beta"],n=len(z),n_u=len(exc),qs=tuple(LEV))
        for q in LEV: o[f"e_90_VaR_{q}"]=(mu+sig*tm.loc[q,"z_q"])/100.0
    except Exception:
        o["fail_gpd"]=1
    return o

if __name__=="__main__":
    origins=list(range(1500,G))   # same date set as window_forecasts.csv
    t0=time.time()
    rows=Parallel(n_jobs=2,verbose=1)(delayed(one)(i) for i in origins)
    df=pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    write_forecasts(df, "results/window90_forecasts.csv")
    print(f"DONE {len(df)} in {(time.time()-t0)/60:.1f} min | fit failures {df.fail_fit.sum()} | gpd failures {df.fail_gpd.sum()}")
    print(f"non-convergence (conv flag): {(df.get('conv',pd.Series([1]*len(df)))==0).sum()}")
    print(f"k (exceedances) median {df.k.median()}, min {df.k.min()}, max {df.k.max()}")
    print(f"xi range: {df.xi.min():.2f} .. {df.xi.max():.2f}, median {df.xi.median():.3f}, |xi|>1 in {(df.xi.abs()>1).sum()} windows")
    print(f"nu range: {df.nu.min():.2f} .. {df.nu.max():.2f}")
