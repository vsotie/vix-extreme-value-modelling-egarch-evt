"""Objective IV: backtest evaluation of the rolling forecasts.
Reads results/forecasts.csv. Kupiec POF, Christoffersen independence + CC,
RMSE/QLIKE (+ Diebold-Mariano), McNeil-Frey bootstrap ES test, by-regime."""
import numpy as np, pandas as pd
from scipy import stats
np.random.seed(20260901)

df = pd.read_csv("results/forecasts.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
r = df["realized"].values
LEVELS = [0.95, 0.99, 0.995]
# headline 3 + MF-analog parametric-t columns
MODELS = {"EGARCH-EVT":"eevt", "GARCH-EVT":"gevt", "Static-POT":"s",
          "EGARCH-t":"et", "GARCH-t":"gt"}

def kupiec(x, n, p):
    if x == 0:
        lr = -2*(n*np.log(1-p))
    else:
        pi = x/n
        lr = -2*((n-x)*np.log(1-p)+x*np.log(p) - (n-x)*np.log(1-pi)-x*np.log(pi))
    return lr, 1-stats.chi2.cdf(lr, 1)

def christoffersen_ind(I):
    I = I.astype(int)
    n00=n01=n10=n11=0
    for a,b in zip(I[:-1], I[1:]):
        if a==0 and b==0: n00+=1
        elif a==0 and b==1: n01+=1
        elif a==1 and b==0: n10+=1
        else: n11+=1
    pi01 = n01/(n00+n01) if (n00+n01)>0 else 0.0
    pi11 = n11/(n10+n11) if (n10+n11)>0 else 0.0
    pi   = (n01+n11)/(n00+n01+n10+n11)
    def term(nx, p):
        return nx*np.log(p) if (nx>0 and p>0) else 0.0
    ln_null = term(n00+n10, 1-pi)+term(n01+n11, pi)
    ln_alt  = term(n00,1-pi01)+term(n01,pi01)+term(n10,1-pi11)+term(n11,pi11)
    lr = -2*(ln_null - ln_alt)
    lr = max(lr, 0.0)
    return lr, 1-stats.chi2.cdf(lr,1), pi01, pi11

def coverage_table():
    rows=[]
    for name,pfx in MODELS.items():
        for q in LEVELS:
            col=f"{pfx}_VaR_{q}"
            if col not in df: continue
            var=df[col].values; p=1-q
            I=(r>var).astype(int); x=int(I.sum()); n=len(I)
            lr_uc,p_uc=kupiec(x,n,p)
            lr_i,p_i,pi01,pi11=christoffersen_ind(I)
            lr_cc=lr_uc+lr_i; p_cc=1-stats.chi2.cdf(lr_cc,2)
            rows.append({"model":name,"q":q,"viol":x,"exp":round(n*p,1),
                         "rate%":round(100*x/n,3),"p_uc":round(p_uc,4),
                         "pi11%":round(100*pi11,2),"p_ind":round(p_i,4),
                         "p_cc":round(p_cc,4)})
    return pd.DataFrame(rows)

def accuracy():
    # one-step variance forecast loss vs r^2 proxy (return scale), EGARCH vs GARCH
    proxy=r**2
    out={}; losses={}
    for name,pfx in [("EGARCH-t","e"),("GARCH-t","g")]:
        s2=df[f"{pfx}_sig"].values**2
        rmse=np.sqrt(np.mean((s2-proxy)**2))
        qlike=np.mean(np.log(s2)+proxy/s2)
        out[name]={"RMSE_var":rmse,"QLIKE":qlike}
        losses[name]={"se":(s2-proxy)**2,"ql":np.log(s2)+proxy/s2}
    # Diebold-Mariano (EGARCH - GARCH); negative => EGARCH better
    def dm(d):
        d=np.asarray(d); T=len(d); dbar=d.mean()
        # NW variance, lag 5
        g0=np.var(d,ddof=0); s=g0
        for L in range(1,6):
            c=np.mean((d[L:]-dbar)*(d[:-L]-dbar)); s+=2*(1-L/6)*c
        stat=dbar/np.sqrt(s/T)
        return stat, 2*(1-stats.norm.cdf(abs(stat)))
    dm_se=dm(losses["EGARCH-t"]["se"]-losses["GARCH-t"]["se"])
    dm_ql=dm(losses["EGARCH-t"]["ql"]-losses["GARCH-t"]["ql"])
    return out, dm_se, dm_ql

def mf_es_test(pfx, q, B=10000):
    """McNeil-Frey bootstrap ES test on a conditional model.
    Discrepancy on violation days: (realized - ES)/sigma. H0 mean=0 vs >0
    (ES too low). One-sided bootstrap p-value."""
    sig=df[{"eevt":"e","gevt":"g"}[pfx]+"_sig"].values
    var=df[f"{pfx}_VaR_{q}"].values; es=df[f"{pfx}_ES_{q}"].values
    viol=r>var
    D=((r-es)/sig)[viol]
    if len(D)<2: return len(D), np.nan, np.nan
    obs=D.mean(); Dc=D-obs
    bs=np.array([np.random.choice(Dc,len(D),replace=True).mean() for _ in range(B)])
    p=np.mean(bs>=obs)   # one-sided: ES underestimated
    return len(D), round(obs,4), round(p,4)

if __name__=="__main__":
    print("=== ROLLING BACKTEST EVALUATION (Objectives III-IV) ===")
    print(f"forecasts: {len(df)}   span {df.date.min().date()} -> {df.date.max().date()}")
    print(f"convergence: EGARCH {df.e_ok.mean():.4f}  GARCH {df.g_ok.mean():.4f}  static {df.s_ok.mean():.4f}\n")
    ct=coverage_table()
    print("=== Coverage (Kupiec p_uc, Christoffersen p_ind / p_cc; pi11=P(viol|viol)) ===")
    print(ct.to_string(index=False),"\n")
    acc,dm_se,dm_ql=accuracy()
    print("=== Variance-forecast accuracy (proxy = r^2, return scale) ===")
    for k,v in acc.items(): print(f"  {k}: RMSE_var={v['RMSE_var']:.6e}  QLIKE={v['QLIKE']:.5f}")
    print(f"  Diebold-Mariano EGARCH-GARCH  squared-error: stat={dm_se[0]:.3f} p={dm_se[1]:.4f}")
    print(f"  Diebold-Mariano EGARCH-GARCH  QLIKE:         stat={dm_ql[0]:.3f} p={dm_ql[1]:.4f}")
    print("  (negative DM stat => EGARCH lower loss)\n")
    print("=== McNeil-Frey bootstrap ES test (H0: ES correct; small p => ES too low) ===")
    for pfx,name in [("eevt","EGARCH-EVT"),("gevt","GARCH-EVT")]:
        for q in LEVELS:
            nD,obs,p=mf_es_test(pfx,q)
            print(f"  {name} q={q}: n_viol={nD}  mean_discrepancy={obs}  p={p}")
    print()
    # by-regime violations at 0.99
    WINDOWS=[('2011-12 Euro','2011-01-01','2012-12-31'),('2013-16 Recovery','2013-01-01','2016-12-31'),
             ('2017-19 incl Volmageddon','2017-01-01','2019-12-31'),('2020-21 COVID','2020-01-01','2021-12-31'),
             ('2022-24 Tightening','2022-01-01','2024-12-31'),('2025-26 Recent','2025-01-01','2026-12-31')]
    print("=== Violations at q=0.99 by regime (exp = 1% of window days) ===")
    hdr=f"{'regime':<26}{'days':>6}{'exp':>6}{'EGEVT':>7}{'GAEVT':>7}{'Static':>7}"
    print(hdr)
    for lab,a,b in WINDOWS:
        m=(df.date>=a)&(df.date<=b); nd=int(m.sum())
        if nd==0: continue
        ve=int((r[m]>df.loc[m,'eevt_VaR_0.99']).sum())
        vg=int((r[m]>df.loc[m,'gevt_VaR_0.99']).sum())
        vs=int((r[m]>df.loc[m,'s_VaR_0.99']).sum())
        print(f"{lab:<26}{nd:>6}{nd*0.01:>6.1f}{ve:>7}{vg:>7}{vs:>7}")
