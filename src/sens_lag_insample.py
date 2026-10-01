"""Objective V(b), in-sample: EGARCH lag-structure sensitivity on the full sample.
Does the choice of (p,o,q) change model preference or the tail estimates?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from arch import arch_model
from utils import gpd_fit, gpd_tail_measures
vix=pd.read_csv("data/vix_raw.csv",parse_dates=["Date"]).set_index("Date")
r=np.log(vix["Close"]/vix["Close"].shift(1)).dropna()
GRID=[(1,1,1),(2,1,1),(1,1,2),(2,1,2),(1,2,1)]
rows=[]
for (p,o,q) in GRID:
    f=arch_model(r*100,mean="AR",lags=1,vol="EGARCH",p=p,o=o,q=q,dist="t").fit(disp="off",show_warning=False)
    z=(f.resid/f.conditional_volatility).dropna().values
    u=np.quantile(z,0.90); exc=z[z>u]-u; gf=gpd_fit(exc)
    tm=gpd_tail_measures(u,gf["xi"],gf["beta"],n=len(z),n_u=len(exc),qs=(0.99,0.995))
    gammas=[f.params[k] for k in f.params.index if k.startswith("gamma")]
    rows.append({"spec":f"EGARCH({p},{o},{q})","k":f.num_params,"LL":f.loglikelihood,
                 "AIC":f.aic,"BIC":f.bic,"theta_sum":sum(gammas),"nu":f.params["nu"],
                 "resid_xi":gf["xi"],"z_0.99":tm.loc[0.99,"z_q"],"z_0.995":tm.loc[0.995,"z_q"]})
df=pd.DataFrame(rows)
pd.set_option("display.width",200,"display.max_columns",20)
print("=== EGARCH lag-structure grid, full sample (AR(1) mean, Student-t) ===")
print(df.round({"LL":1,"AIC":1,"BIC":1,"theta_sum":4,"nu":3,"resid_xi":4,"z_0.99":4,"z_0.995":4}).to_string(index=False))
base=df.iloc[0]
print("\nBIC gap to EGARCH(1,1) (positive => worse than baseline):")
for _,x in df.iterrows():
    print(f"  {x['spec']}: dBIC {x['BIC']-base['BIC']:+.1f}   z_0.995 {x['z_0.995']:.4f} (base {base['z_0.995']:.4f}, {100*(x['z_0.995']/base['z_0.995']-1):+.2f}%)")
