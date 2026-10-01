import numpy as np, pandas as pd, time
from backtest import forecast_one, N, G, LEVELS
t0=time.time()
rows=[forecast_one(i) for i in range(N, N+50)]
df=pd.DataFrame(rows); dt=time.time()-t0
print(f"{len(df)} windows in {dt:.1f}s  ({dt/len(df):.3f}s/win)")
print("convergence: e_ok",df.e_ok.mean()," g_ok",df.g_ok.mean()," s_ok",df.s_ok.mean())
# EVT tail must be >= parametric-t tail at 0.995 (heavier tail), same mu/sig
bad=(df["eevt_VaR_0.995"] < df["et_VaR_0.995"]).sum()
print("EGARCH: EVT<t at 0.995 (should be ~0):", bad)
# ordering within a model: VaR increases with q
mono=((df["eevt_VaR_0.95"]<df["eevt_VaR_0.99"])&(df["eevt_VaR_0.99"]<df["eevt_VaR_0.995"])).mean()
print("EGARCH-EVT VaR monotone in q (frac):", mono)
# static should not depend on next-day sigma: check it's constant-ish vs conditional which varies
print("cond EGARCH 0.99 VaR std:", round(df["eevt_VaR_0.99"].std(),4), " static 0.99 VaR std:", round(df["s_VaR_0.99"].std(),4))
# violation counts at 0.95 on this mini-sample
for m,lab in [("eevt","EGARCH-EVT"),("gevt","GARCH-EVT"),("s","Static")]:
    v=(df["realized"]>df[f"{m}_VaR_0.95"]).sum()
    print(f"  {lab} 0.95 violations in 50: {v} (exp {50*0.05:.1f})")
# no-lookahead spot check: recompute window for row 10 independently
i=N+10
import numpy as np
from arch import arch_model
vix=pd.read_csv("data/vix_raw.csv",parse_dates=["Date"]).set_index("Date")
r=np.log(vix["Close"]/vix["Close"].shift(1)).dropna().values
w=r[i-N:i]*100
f=arch_model(w,mean="AR",lags=1,vol="EGARCH",p=1,o=1,q=1,dist="t").fit(disp="off",show_warning=False)
fc=f.forecast(horizon=1,reindex=False)
print("indep sigma_next:",round(float(np.sqrt(fc.variance.values[0,0]))/100,5)," engine:",round(df.iloc[10]["e_sig"],5))
