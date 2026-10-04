import time, pandas as pd
from joblib import Parallel, delayed
from backtest import forecast_one, N, G
from forecast_io import read_forecasts, write_forecasts
origins = list(range(N, G))
t0 = time.time()
rows = Parallel(n_jobs=2, verbose=1)(delayed(forecast_one)(i) for i in origins)
df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
write_forecasts(df, "results/forecasts.csv")
print(f"DONE {len(df)} forecasts in {(time.time()-t0)/60:.1f} min")
print("convergence rates: e_ok=%.4f g_ok=%.4f s_ok=%.4f" %
      (df.e_ok.mean(), df.g_ok.mean(), df.s_ok.mean()))
print("date span:", df.date.min(), "->", df.date.max())
