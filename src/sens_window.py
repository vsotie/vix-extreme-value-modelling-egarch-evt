"""Window-length sensitivity: 1000 vs 1500 day estimation window.
Same model (AR(1)-EGARCH-t + GPD upper tail) and same forecast dates; only the
memory length differs. Common period = from the 1500-day scheme's first forecast.
Also fits GARCH-EVT at both lengths to see if the EGARCH-vs-GARCH gap shifts."""
import os; os.environ.setdefault("MPLBACKEND","Agg")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time
from arch import arch_model
from utils import gpd_fit, gpd_tail_measures
from joblib import Parallel, delayed
N1,N2=1000,1500; QT=0.90; LEV=[0.95,0.99]
vix=pd.read_csv("data/vix_raw.csv",parse_dates=["Date"]).set_index("Date")
r=np.log(vix["Close"]/vix["Close"].shift(1)).dropna(); rr=r.values; dates=r.index; G=len(rr)

def fit_evt(w100,vol):
    am=arch_model(w100,mean="AR",lags=1,vol=("EGARCH" if vol=="E" else "GARCH"),
                  **({"p":1,"o":1,"q":1} if vol=="E" else {"p":1,"q":1}),dist="t")
    f=am.fit(disp="off",show_warning=False)
    fc=f.forecast(horizon=1,reindex=False)
    mu=float(fc.mean.values[0,0]); sig=float(np.sqrt(fc.variance.values[0,0]))
    z=(f.resid/f.conditional_volatility); z=np.asarray(z[np.isfinite(z)],float)
    u=np.quantile(z,QT); exc=z[z>u]-u; gf=gpd_fit(exc)
    tm=gpd_tail_measures(u,gf["xi"],gf["beta"],n=len(z),n_u=len(exc),qs=tuple(LEV))
    out={"sig":sig/100.0}
    for q in LEV:
        out[f"VaR_{q}"]=(mu+sig*tm.loc[q,"z_q"])/100.0
        out[f"ES_{q}"]=(mu+sig*tm.loc[q,"es_q"])/100.0
    return out

def one(i):
    o={"date":dates[i],"realized":rr[i]}
    for N,tag in [(N1,"1000"),(N2,"1500")]:
        w100=rr[i-N:i]*100.0
        for vol,mtag in [("E","e"),("G","g")]:
            d=fit_evt(w100,vol)
            for k,v in d.items(): o[f"{mtag}_{tag}_{k}"]=v
    return o

if __name__=="__main__":
    origins=list(range(N2,G))   # both windows available
    t0=time.time()
    rows=Parallel(n_jobs=2,verbose=1)(delayed(one)(i) for i in origins)
    df=pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv("results/window_forecasts.csv",index=False)
    print(f"DONE {len(df)} forecasts in {(time.time()-t0)/60:.1f} min  span {df.date.min().date()} -> {df.date.max().date()}")
