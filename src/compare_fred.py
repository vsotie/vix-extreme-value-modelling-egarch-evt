"""Compare the stored dataset with the VIXCLS series FRED publishes today.

The results of the dissertation are computed from data/vix_raw.csv, the file
supplied with the dissertation: the series as retrieved on 1 August 2026. FRED
can revise past values, so this script downloads the same window afresh with
load_vix, the function that wrote the stored file, into
data/vix_fred_latest.csv, and reports every difference. The stored file is
never modified.

Needs the supplied file in data/ and a free FRED API key in the environment
variable FRED_API_KEY (https://fred.stlouisfed.org/docs/api/api_key.html).
Run from the repository root: python src/compare_fred.py
"""
import os
import sys

import pandas as pd

from utils import load_vix

STORED = "data/vix_raw.csv"
LATEST = "data/vix_fred_latest.csv"
START, END = "2007-01-01", "2026-06-30"   # the window requested by main.py

if not os.path.exists(STORED):
    sys.exit(f"{STORED} not found. Put the file supplied with the "
             "dissertation there first (see README).")
api_key = os.environ.get("FRED_API_KEY")
if not api_key:
    sys.exit("Set FRED_API_KEY first (free key: "
             "https://fred.stlouisfed.org/docs/api/api_key.html).")

# Always compare with a fresh download, never with an earlier one
if os.path.exists(LATEST):
    os.remove(LATEST)
latest = load_vix(api_key, START, END, cache_path=LATEST)["Close"]
stored = pd.read_csv(STORED, parse_dates=["Date"]).set_index("Date")["Close"]

both = pd.concat({"stored": stored, "latest": latest}, axis=1)
only_stored = both.index[both["latest"].isna()]
only_latest = both.index[both["stored"].isna()]
common = both.dropna()
changed = common[common["stored"] != common["latest"]]

print(f"Stored file: {len(stored)} closes; latest download: {len(latest)} closes")
print(f"Dates only in the stored file:      {len(only_stored)}")
print(f"Dates only in the latest download:  {len(only_latest)}")
print(f"Dates with a different close:       {len(changed)}")
for d in only_stored.union(only_latest):
    print(f"  {d.date()}  stored {both.at[d, 'stored']}  latest {both.at[d, 'latest']}")
if len(changed):
    print(changed.to_string())
if len(only_stored) + len(only_latest) + len(changed) == 0:
    print("The latest download matches the stored data exactly.")
