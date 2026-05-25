"""
Generates 01_data_preparation.ipynb and 02_lstm_model.ipynb in the same directory.
Run once: /Users/Dave/miniforge3/envs/py312/bin/python gen_notebooks.py
"""

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from pathlib import Path

HERE = Path(__file__).parent


def mk_nb(cells):
    n = new_notebook()
    n.cells = cells
    n.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.9"},
    }
    return n


c = new_code_cell
m = new_markdown_cell

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 1 — Data Preparation (Steps 1–3)
# ─────────────────────────────────────────────────────────────────────────────

nb1_cells = [

m("""# NZ Electricity Price — Data Preparation (Steps 1–3)

Loads, cleans, resamples, merges, and feature-engineers all raw datasets into a
single hourly master DataFrame saved to `master.parquet`.

**Target:** `HAY2201` (Haywards reference node)
**Time window:** `2021-05-01 → 2024-12-31` (intersection of all sources)
**Prediction holdout:** `2024-12-25 → 2024-12-31` (last 7 days — kept for model demo)
"""),

c("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

DATA = "../"   # source data folder — all reads here, never written
OUT  = "./"    # claude_code/ — all outputs written here

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 120)
"""),

# ── Step 1: Load ──────────────────────────────────────────────────────────────
m("## Step 1 — Load Raw Data"),

c("""\
price_raw    = pd.read_csv(f"{DATA}Wholesale_price_trends_20260518210002.csv", skiprows=11)
demand_raw   = pd.read_csv(f"{DATA}Demand_trends_20260518215015.csv",          skiprows=11)
gen_raw      = pd.read_csv(f"{DATA}Generation_trends_20260519182544.csv",      skiprows=11)
hvdc_raw     = pd.read_csv(f"{DATA}HVDC_transfer_20260518215532.csv",          skiprows=10)
lake_t_raw   = pd.read_csv(f"{DATA}NI_TPO_Storage_LakeTaupo.csv")
lake_w_raw   = pd.read_csv(f"{DATA}NI_WKA_Storage_LakeWaikaremoana.csv")
wind_raw     = pd.read_csv(f"{DATA}Wind_data_100m.csv")

datasets = {
    "price": price_raw, "demand": demand_raw, "generation": gen_raw,
    "HVDC": hvdc_raw, "lake_taupo": lake_t_raw, "lake_waik": lake_w_raw,
    "wind": wind_raw,
}
for name, df in datasets.items():
    print(f"  {name:12s}  {str(df.shape):15s}  columns: {list(df.columns)[:4]}")
"""),

c("""\
# Quick date-range check for the 30-min datasets
for name, df, col, skip in [
    ("price",      price_raw,  "Period start", True),
    ("demand",     demand_raw, "Period start", True),
    ("generation", gen_raw,    "Period start", True),
    ("HVDC",       hvdc_raw,   "Period start", True),
]:
    ts = pd.to_datetime(df[col], dayfirst=True)
    print(f"  {name:12s}  {ts.min()}  →  {ts.max()}")

wind_ts = pd.to_datetime(wind_raw["time"])
print(f"  {'wind':12s}  {wind_ts.min()}  →  {wind_ts.max()}")

for name, df in [("lake_taupo", lake_t_raw), ("lake_waik", lake_w_raw)]:
    ts = pd.to_datetime(df["Date"] + " " + df["Time"])
    ts = ts[ts >= "2021-01-01"]
    print(f"  {name:12s}  {ts.min()}  →  {ts.max()}  ({len(ts)} daily readings)")
"""),

# ── Step 2: Clean & Resample ───────────────────────────────────────────────────
m("## Step 2 — Clean & Resample to Hourly"),

m("### 2a. Wholesale Price\n\nPivot 8 nodes wide, resample 30-min → hourly (mean $/MWh)."),

c("""\
price = price_raw.copy()
price["Period start"] = pd.to_datetime(price["Period start"], dayfirst=True)
price = (price
         .pivot(index="Period start", columns="Region ID", values="Price ($/MWh)")
         .rename_axis("datetime")
         .rename_axis(None, axis=1))
price_h = price.resample("1h").mean()
print("Price (hourly):", price_h.shape, "  nodes:", list(price_h.columns))
price_h.head(3)
"""),

c("""\
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(price_h.index, price_h["HAY2201"], lw=0.4, alpha=0.9, color="steelblue")
ax.set_ylabel("Price ($/MWh)")
ax.set_title("HAY2201 Haywards — hourly wholesale price")
ax.set_ylim(0, 800)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.savefig(f"{OUT}fig_price_overview.png", dpi=120)
plt.show()
"""),

m("### 2b. Demand\n\nPivot 5 zones wide, resample 30-min → hourly (sum GWh)."),

c("""\
demand = demand_raw.copy()
demand["Period start"] = pd.to_datetime(demand["Period start"], dayfirst=True)
demand = (demand
          .pivot(index="Period start", columns="Region ID", values="Demand (GWh)")
          .rename_axis("datetime")
          .rename_axis(None, axis=1))
demand.columns = ["demand_" + c for c in demand.columns]
demand_h = demand.resample("1h").sum()   # sum GWh across the hour
print("Demand (hourly):", demand_h.shape, list(demand_h.columns))
demand_h.head(3)
"""),

m("### 2c. Generation\n\nPivot 5 zones wide, resample 30-min → hourly (sum GWh)."),

c("""\
gen = gen_raw.copy()
gen["Period start"] = pd.to_datetime(gen["Period start"], dayfirst=True)
gen = (gen
       .pivot(index="Period start", columns="Region ID", values="Generation (GWh)")
       .rename_axis("datetime")
       .rename_axis(None, axis=1))
gen.columns = ["gen_" + c for c in gen.columns]
gen_h = gen.resample("1h").sum()
print("Generation (hourly):", gen_h.shape, list(gen_h.columns))
gen_h.head(3)
"""),

m("### 2d. HVDC Transfer\n\nSign-encode direction: Northward = +MW, Southward = −MW. Resample to hourly (mean)."),

c("""\
hvdc = hvdc_raw.copy()
hvdc["Period start"] = pd.to_datetime(hvdc["Period start"], dayfirst=True)
hvdc = hvdc.set_index("Period start").rename_axis("datetime")
hvdc["hvdc_flow_mw"] = np.where(
    hvdc["Direction"] == "Northward at Haywards",
     hvdc["Average flow (MW)"],
    -hvdc["Average flow (MW)"],
)
hvdc_h = hvdc[["hvdc_flow_mw"]].resample("1h").mean()
print("HVDC (hourly):", hvdc_h.shape)
hvdc_h.head(3)
"""),

m("""\
### 2e. Lake Storage — Daily → Hourly

Both lakes have exactly **1 reading per day** (at 23:59:59).
Linear time-interpolation creates smooth hourly values between daily measurements.
"""),

c("""\
def lake_to_hourly(path, col_name):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = (df.sort_values("datetime")
            .set_index("datetime")
            [["Active storage (Mm³)"]]
            .rename(columns={"Active storage (Mm³)": col_name}))
    # Resample 23:59:59 daily readings → clean midnight daily index
    df_daily = df.resample("D").last()
    # Build hourly index, reindex, then linearly interpolate gaps
    hourly_idx = pd.date_range(
        df_daily.index.min(),
        df_daily.index.max() + pd.Timedelta(hours=23),
        freq="1h",
    )
    return df_daily.reindex(hourly_idx).interpolate(method="linear")

lake_t_h = lake_to_hourly(f"{DATA}NI_TPO_Storage_LakeTaupo.csv",         "lake_taupo_mm3")
lake_w_h = lake_to_hourly(f"{DATA}NI_WKA_Storage_LakeWaikaremoana.csv",  "lake_waik_mm3")

print("Lake Taupo (hourly):", lake_t_h.shape,
      lake_t_h.index.min(), "→", lake_t_h.index.max())
print("Lake Waik  (hourly):", lake_w_h.shape,
      lake_w_h.index.min(), "→", lake_w_h.index.max())
lake_t_h.head(3)
"""),

c("""\
fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
axes[0].plot(lake_t_h, lw=0.8, color="teal",   label="Lake Taupo")
axes[1].plot(lake_w_h, lw=0.8, color="indigo", label="Lake Waikaremoana")
for ax in axes:
    ax.set_ylabel("Active storage (Mm³)")
    ax.legend(loc="upper right")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.savefig(f"{OUT}fig_lake_storage.png", dpi=120)
plt.show()
"""),

m("### 2f. Wind Data\n\nAlready hourly. Clean column names; encode wind direction as sin/cos."),

c("""\
wind = wind_raw.copy()
wind["time"] = pd.to_datetime(wind["time"])
wind = wind.set_index("time").rename_axis("datetime")
if "Unnamed: 0" in wind.columns:
    wind = wind.drop(columns=["Unnamed: 0"])

# Normalise column names → lowercase, spaces → underscores
wind.columns = (wind.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False)
                .str.replace("(", "", regex=False)
                .str.replace(")", "", regex=False))

# Encode wind directions as sin/cos, drop raw degree columns
dir_cols = [c for c in wind.columns if "direction_deg" in c]
for col in dir_cols:
    prefix = col.replace("_direction_deg", "")
    rad = np.deg2rad(wind[col])
    wind[prefix + "_dir_sin"] = np.sin(rad)
    wind[prefix + "_dir_cos"] = np.cos(rad)
wind = wind.drop(columns=dir_cols)

print("Wind (hourly):", wind.shape, list(wind.columns))
wind.head(3)
"""),

# ── Step 3: Align & Merge ──────────────────────────────────────────────────────
m("""\
## Step 3 — Align to Common Window & Merge

| Dataset   | Start       | End         |
|-----------|-------------|-------------|
| Price     | 2021-05-01  | 2026-04-30  |
| Demand    | 2021-01-01  | 2025-12-31  |
| Generation| 2021-05-01  | 2026-04-30  |
| HVDC      | 2021-01-01  | 2025-12-31  |
| Lake      | 1980-01-01  | **2024-12-31** |
| Wind      | 2021-01-01  | 2026-04-30  |

**Intersection → 2021-05-01 to 2024-12-31** (driven by lake data end-date).
"""),

c("""\
START = "2021-05-01 00:00:00"
END   = "2024-12-31 23:00:00"
hourly_idx = pd.date_range(START, END, freq="1h")
print(f"Common window: {START}  →  {END}  ({len(hourly_idx):,} hourly rows)")

def clip(df):
    return df.reindex(hourly_idx)

price_h  = clip(price_h)
demand_h = clip(demand_h)
gen_h    = clip(gen_h)
hvdc_h   = clip(hvdc_h)
lake_t_h = clip(lake_t_h)
lake_w_h = clip(lake_w_h)
wind_h   = clip(wind)

print("\\nNaN % after alignment:")
for name, df in [("price", price_h), ("demand", demand_h), ("gen", gen_h),
                 ("hvdc", hvdc_h), ("lake_taupo", lake_t_h), ("lake_waik", lake_w_h),
                 ("wind", wind_h)]:
    pct = df.isna().mean().mean() * 100
    print(f"  {name:12s}  {pct:.3f}%")
"""),

c("""\
master = pd.concat([price_h, demand_h, gen_h, hvdc_h, lake_t_h, lake_w_h, wind_h], axis=1)
print("Master after merge:", master.shape)
master.head(3)
"""),

c("""\
# Forward-fill up to 3 hours for any small gaps, then drop residual NaNs
master = master.ffill(limit=3)
remaining_nan = master.isna().sum()
print("Residual NaN per column (non-zero only):")
print(remaining_nan[remaining_nan > 0])
"""),

# ── Step 4: Feature Engineering ───────────────────────────────────────────────
m("## Step 4 — Feature Engineering"),

m("### 4a. Cyclical Time Encodings"),

c("""\
def cyc(s, period):
    return np.sin(2 * np.pi * s / period), np.cos(2 * np.pi * s / period)

idx = master.index
master["hour_sin"],  master["hour_cos"]  = cyc(idx.hour,        24)
master["dow_sin"],   master["dow_cos"]   = cyc(idx.dayofweek,    7)
master["month_sin"], master["month_cos"] = cyc(idx.month,        12)
master["is_weekend"] = (idx.dayofweek >= 5).astype(int)
print("Added cyclical time features.")
"""),

m("### 4b. Price Lags & Rolling Statistics (target: HAY2201)"),

c("""\
TARGET = "HAY2201"
# Intra-day lags + full Mon-Sun weekly shape (24 h steps) + 1-week anchor
for lag in [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168]:
    master[f"price_lag_{lag}h"] = master[TARGET].shift(lag)

master["price_roll_mean_24h"]  = master[TARGET].shift(1).rolling(24).mean()
master["price_roll_std_24h"]   = master[TARGET].shift(1).rolling(24).std()
master["price_roll_mean_168h"] = master[TARGET].shift(1).rolling(168).mean()
print("Added price lag and rolling features.")
"""),

m("### 4c. Aggregate Demand / Generation Features"),

c("""\
d_cols = [c for c in master.columns if c.startswith("demand_")]
g_cols = [c for c in master.columns if c.startswith("gen_")]

master["demand_total_gwh"]  = master[d_cols].sum(axis=1)
master["gen_total_gwh"]     = master[g_cols].sum(axis=1)
master["gen_demand_ratio"]  = master["gen_total_gwh"] / master["demand_total_gwh"].replace(0, np.nan)
print("Added demand/generation aggregate features.")
"""),

m("### 4d. Wind Power Proxy  (speed³, capped at 99th percentile)"),

c("""\
speed_cols = [c for c in master.columns if c.endswith("_wind_kmh")]
for col in speed_cols:
    cap = master[col].quantile(0.99)
    master[col.replace("_wind_kmh", "_wind_power")] = master[col].clip(upper=cap) ** 3

power_cols = [c for c in master.columns if "_wind_power" in c]
master["wind_power_total"] = master[power_cols].sum(axis=1)
print("Wind power proxy columns:", power_cols)
"""),

m("### 4e. NI–SI Price Spread  (HVDC congestion proxy)"),

c("""\
ni_nodes = ["OTA2201", "WKM2201", "SFD2201", "HAY2201", "KIK2201"]
si_nodes = ["RDF2201", "ISL2201", "BEN2201"]
master["price_ni_mean"]       = master[ni_nodes].mean(axis=1)
master["price_si_mean"]       = master[si_nodes].mean(axis=1)
master["price_ni_si_spread"]  = master["price_ni_mean"] - master["price_si_mean"]
print("Added NI/SI spread features.")
"""),

m("### 4f. Lake Storage Change  (24 h delta)"),

c("""\
master["lake_taupo_delta_24h"] = master["lake_taupo_mm3"].diff(24)
master["lake_waik_delta_24h"]  = master["lake_waik_mm3"].diff(24)
print("Added lake delta features.")
"""),

# ── Final check & save ─────────────────────────────────────────────────────────
m("## Final Check & Save"),

c("""\
# Drop rows where 168h lag is still NaN (warm-up period at start)
master = master.dropna(subset=["price_lag_168h"])
print("Master shape (after warm-up drop):", master.shape)
print(f"Date range: {master.index.min()} → {master.index.max()}")
print(f"Total features: {master.shape[1]}")
print("\\nColumn list:")
print(list(master.columns))
"""),

c("""\
# Check residual NaNs
nan_summary = master.isna().sum()
nan_remaining = nan_summary[nan_summary > 0]
if len(nan_remaining):
    print("Remaining NaN columns:")
    print(nan_remaining)
else:
    print("No NaN values remaining — master DataFrame is clean.")
"""),

c("""\
# Summary statistics for key columns
master[["HAY2201", "demand_total_gwh", "gen_total_gwh",
        "hvdc_flow_mw", "lake_taupo_mm3", "wind_power_total"]].describe().round(2)
"""),

c("""\
master.to_parquet(f"{OUT}master.parquet")
print(f"Saved → claude_code/master.parquet  ({master.shape[0]:,} rows × {master.shape[1]} cols)")
"""),

]  # end nb1_cells


# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 2 — LSTM Model (Steps 4–5)
# ─────────────────────────────────────────────────────────────────────────────

nb2_cells = [

m("""\
# NZ Electricity Price — LSTM Model (Steps 4–5)

Lightweight single-layer LSTM on CPU. Predicts the next **24 hours** of wholesale
electricity price at Haywards (`HAY2201`) from a **24-hour lookback window**.

**Architecture:** 2-layer LSTM (hidden=128) + attention pooling → 24-step forecast.
**Goal:** assess whether the dataset is sufficient for an okayish 24 h price forecast.

**Splits**
| Set      | Period                    | Purpose                        |
|----------|---------------------------|--------------------------------|
| Train    | 2021-05-08 → 2024-06-30   | Gradient updates               |
| Val      | 2024-07-01 → 2024-10-31   | Early stopping (patience 8)    |
| Test     | 2024-11-01 → 2024-12-24   | Final evaluation w/ labels     |
| Predict  | Dec 25, 2024              | 24 h demo — no labels          |
"""),

# ── Imports ───────────────────────────────────────────────────────────────────
c("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
"""),

# ── Load data ─────────────────────────────────────────────────────────────────
m("## Load Master DataFrame"),

c("""\
master = pd.read_parquet("./master.parquet")

# Fill residual NaNs: forward-fill (e.g. HVDC gaps), then zero for any remaining
master = master.ffill().fillna(0)

print("Loaded:", master.shape)
print("Range: ", master.index.min(), "→", master.index.max())
print("Remaining NaNs:", master.isna().sum().sum())
master.head(3)
"""),

# ── Splits ────────────────────────────────────────────────────────────────────
m("## Train / Val / Test / Predict Splits"),

c("""\
TARGET   = "HAY2201"
LOOKBACK = 48    # 2-day context window — captures two full daily cycles
HORIZON  = 24    # predict next 24 h
STRIDE   = 2     # sample every 2nd sequence

TRAIN_END  = "2024-06-30 23:00"
VAL_END    = "2024-10-31 23:00"
TEST_END   = "2024-12-24 23:00"

train_df = master[master.index <= TRAIN_END]
val_df   = master[(master.index > TRAIN_END) & (master.index <= VAL_END)]
test_df  = master[(master.index > VAL_END)   & (master.index <= TEST_END)]

for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"  {name:6s}  {len(df):6,} rows   {df.index.min().date()} → {df.index.max().date()}")
"""),

# ── Features & scaling ────────────────────────────────────────────────────────
m("## Feature Selection & Scaling"),

c("""\
# Include HAY2201 as a feature — the model sees the full price history in its
# lookback window and only needs to learn WHEN to deviate from it.
# Exclude the other raw node prices (kept only as spread/NI-SI aggregates).
raw_nodes    = ["OTA2201", "WKM2201", "RDF2201", "SFD2201", "KIK2201", "ISL2201", "BEN2201"]
feature_cols = [c for c in master.columns if c not in raw_nodes]   # HAY2201 now included
print(f"Features: {len(feature_cols)}")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(train_df[feature_cols].values)
y_train = scaler_y.fit_transform(train_df[[TARGET]].values)

X_val  = scaler_X.transform(val_df[feature_cols].values)
y_val  = scaler_y.transform(val_df[[TARGET]].values)

X_test = scaler_X.transform(test_df[feature_cols].values)
y_test = scaler_y.transform(test_df[[TARGET]].values)

print(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
"""),

# ── Sequence dataset ──────────────────────────────────────────────────────────
m("## Sequence Builder  (stride-sampled to reduce memory)"),

c("""\
class ElecDataset(Dataset):
    \"\"\"Returns (lookback_window, horizon_targets) pairs, sampled with stride.\"\"\"
    def __init__(self, X, y, lookback, horizon, stride=1):
        self.X, self.y = X, y
        self.lb, self.h, self.s = lookback, horizon, stride
        n = len(X) - lookback - horizon + 1
        self.indices = list(range(0, n, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = torch.tensor(self.X[i : i + self.lb],                       dtype=torch.float32)
        y = torch.tensor(self.y[i + self.lb : i + self.lb + self.h].flatten(), dtype=torch.float32)
        return x, y

train_ds = ElecDataset(X_train, y_train, LOOKBACK, HORIZON, stride=STRIDE)
val_ds   = ElecDataset(X_val,   y_val,   LOOKBACK, HORIZON, stride=1)
test_ds  = ElecDataset(X_test,  y_test,  LOOKBACK, HORIZON, stride=1)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_dl  = DataLoader(test_ds,  batch_size=64, shuffle=False)

print(f"Sequences — Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
"""),

# ── Model ─────────────────────────────────────────────────────────────────────
m("""\
## LSTM Model  (2-layer + attention pooling)

Instead of discarding all LSTM outputs and only keeping the final hidden state,
**attention pooling** computes a weighted average over all 48 timesteps — letting
the model learn *which hours in the lookback window matter most*.
"""),

c("""\
class AttentionPool(nn.Module):
    \"\"\"Soft attention over LSTM output sequence → single context vector.\"\"\"
    def __init__(self, hidden):
        super().__init__()
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, lstm_out):          # lstm_out: (batch, seq, hidden)
        scores = self.w(lstm_out)         # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)
        return (weights * lstm_out).sum(dim=1)   # (batch, hidden)


class PriceForecaster(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.2, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, n_layers,
            batch_first=True,
            dropout=dropout,   # applied between layers
        )
        self.attn = AttentionPool(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
        )

    def forward(self, x):
        out, _ = self.lstm(x)      # (batch, seq, hidden) — keep all timesteps
        ctx    = self.attn(out)    # attention-weighted sum → (batch, hidden)
        return self.head(ctx)


DEVICE = "cpu"
model  = PriceForecaster(n_features=len(feature_cols)).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"\\nParameters: {n_params:,}  |  Device: {DEVICE}")
"""),

# ── Training ──────────────────────────────────────────────────────────────────
m("## Training  (40 epochs max, early stopping patience 8)"),

c("""\
EPOCHS   = 40
LR       = 1e-3
PATIENCE = 8

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
best_val, wait_count = float("inf"), 0

for epoch in range(1, EPOCHS + 1):

    # ── train
    model.train()
    t_loss = 0.0
    for Xb, yb in train_dl:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item()
    t_loss /= len(train_dl)

    # ── validate
    model.eval()
    v_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            v_loss += criterion(model(Xb), yb).item()
    v_loss /= len(val_dl)

    train_losses.append(t_loss)
    val_losses.append(v_loss)

    tag = ""
    if v_loss < best_val:
        best_val, wait_count = v_loss, 0
        torch.save(model.state_dict(), "./best_model.pt")
        tag = " ✓"
    else:
        wait_count += 1

    print(f"Epoch {epoch:2d}  train={t_loss:.5f}  val={v_loss:.5f}{tag}")

    if wait_count >= PATIENCE:
        print("Early stopping.")
        break
"""),

c("""\
model.load_state_dict(torch.load("./best_model.pt", map_location=DEVICE))
model.eval()

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(train_losses, label="Train loss")
ax.plot(val_losses,   label="Val loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (scaled)")
ax.set_title("Training curve"); ax.legend()
plt.tight_layout()
plt.savefig("./fig_training_curve.png", dpi=120)
plt.show()
"""),

# ── Evaluation ────────────────────────────────────────────────────────────────
m("## Evaluation on Test Set"),

c("""\
def run_inference(model, dl, scaler_y, device):
    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in dl:
            p = model(Xb.to(device)).cpu().numpy()
            preds.append(p)
            actuals.append(yb.numpy())
    preds   = scaler_y.inverse_transform(np.vstack(preds))
    actuals = scaler_y.inverse_transform(np.vstack(actuals))
    return preds, actuals

test_preds, test_actual = run_inference(model, test_dl, scaler_y, DEVICE)
print("Test predictions shape:", test_preds.shape)  # (n_sequences, 24)
"""),

c("""\
flat_pred   = test_preds.flatten()
flat_actual = test_actual.flatten()

mae  = mean_absolute_error(flat_actual, flat_pred)
rmse = np.sqrt(mean_squared_error(flat_actual, flat_pred))
mape = np.mean(np.abs((flat_actual - flat_pred) / (np.abs(flat_actual) + 1e-6))) * 100

print("━" * 40)
print(f"LSTM — Test set metrics")
print(f"  MAE  : {mae:.2f}  $/MWh")
print(f"  RMSE : {rmse:.2f}  $/MWh")
print(f"  MAPE : {mape:.1f}%")
print("━" * 40)
"""),

m("### Naive Baseline — same hour, previous day"),

c("""\
naive_preds, naive_actuals = [], []
for i in range(len(test_ds)):
    lb = LOOKBACK
    # actual horizon
    y_true = y_test[i + lb : i + lb + HORIZON].flatten()
    # naive: repeat the values exactly 24 h before the forecast window
    y_naive = y_test[i + lb - 24 : i + lb - 24 + HORIZON].flatten()
    if len(y_true) == HORIZON and len(y_naive) == HORIZON:
        naive_preds.append(y_naive)
        naive_actuals.append(y_true)

naive_preds   = scaler_y.inverse_transform(np.vstack(naive_preds))
naive_actuals = scaler_y.inverse_transform(np.vstack(naive_actuals))

n_mae  = mean_absolute_error(naive_actuals.flatten(), naive_preds.flatten())
n_rmse = np.sqrt(mean_squared_error(naive_actuals.flatten(), naive_preds.flatten()))

print(f"Naive baseline (same hour –24 h)")
print(f"  MAE  : {n_mae:.2f}  $/MWh")
print(f"  RMSE : {n_rmse:.2f}  $/MWh")
print()
print(f"LSTM MAE improvement over naive : {(n_mae - mae) / n_mae * 100:.1f}%")
print(f"LSTM RMSE improvement over naive: {(n_rmse - rmse) / n_rmse * 100:.1f}%")
"""),

m("### Test Set — Visual Comparison (first 14 days)"),

c("""\
n_show = min(14 * 24, len(test_preds) * HORIZON)
actual_flat = test_actual[:n_show // HORIZON].flatten()[:n_show]
pred_flat   = test_preds[:n_show // HORIZON].flatten()[:n_show]

# First sequence starts after the initial lookback window
test_start = test_df.index[LOOKBACK]
plot_idx   = pd.date_range(test_start, periods=len(actual_flat), freq="1h")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(plot_idx, actual_flat, lw=1.0, color="steelblue", label="Actual HAY2201")
ax.plot(plot_idx, pred_flat,   lw=1.0, color="tomato",    label="LSTM forecast", alpha=0.85)
ax.set_ylabel("Price ($/MWh)")
ax.set_title("LSTM 24 h forecasts vs actual — first 14 days of test set")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.legend()
plt.tight_layout()
plt.savefig("./fig_test_predictions.png", dpi=120)
plt.show()
"""),

m("### Error distribution by hour-of-day"),

c("""\
errors = (test_preds - test_actual)   # shape (n_seq, 24)
hourly_mae = np.abs(errors).mean(axis=0)

fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(range(1, 25), hourly_mae, color="steelblue", alpha=0.8)
ax.set_xlabel("Forecast horizon (h ahead)")
ax.set_ylabel("MAE ($/MWh)")
ax.set_title("MAE by forecast horizon step")
plt.tight_layout()
plt.savefig("./fig_error_by_horizon.png", dpi=120)
plt.show()
"""),

# ── Feature importance ────────────────────────────────────────────────────────
m("""\
## Feature Importance

Two views:
1. **Permutation importance** — shuffle one feature at a time across all test sequences
   and measure the RMSE increase. Larger = more important.
2. **Attention weights** — average attention score per timestep in the 48 h lookback
   window, showing *when* (how many hours ago) the model focuses most.
"""),

c("""\
# ── 1. Permutation importance ──────────────────────────────────────────
baseline_rmse = np.sqrt(mean_squared_error(test_actual.flatten(), test_preds.flatten()))
rng = np.random.default_rng(42)

importances = {}
for j, col in enumerate(feature_cols):
    X_perm = X_test.copy()
    rng.shuffle(X_perm[:, j])                   # destroy this feature's temporal order
    perm_ds = ElecDataset(X_perm, y_test, LOOKBACK, HORIZON, stride=1)
    perm_dl = DataLoader(perm_ds, batch_size=64, shuffle=False)
    p, a    = run_inference(model, perm_dl, scaler_y, DEVICE)
    importances[col] = np.sqrt(mean_squared_error(a.flatten(), p.flatten())) - baseline_rmse

imp = pd.Series(importances).sort_values(ascending=False)
print(f"Baseline RMSE: {baseline_rmse:.2f} $/MWh")
print("\\nTop 15 features by permutation importance:")
print(imp.head(15).round(2).to_string())
"""),

c("""\
fig, ax = plt.subplots(figsize=(10, 7))
top_n = imp.head(20)
colors = ["tomato" if v > 0 else "steelblue" for v in top_n.values]
ax.barh(top_n.index[::-1], top_n.values[::-1], color=colors[::-1])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("RMSE increase when feature permuted ($/MWh)")
ax.set_title("Top 20 features — permutation importance")
plt.tight_layout()
plt.savefig("./fig_feature_importance.png", dpi=120)
plt.show()
"""),

c("""\
# ── 2. Attention weights over the 48 h lookback ───────────────────────
model.eval()
attn_weights = []
with torch.no_grad():
    for Xb, _ in test_dl:
        out, _ = model.lstm(Xb.to(DEVICE))          # (batch, 48, hidden)
        scores  = model.attn.w(out)                 # (batch, 48, 1)
        weights = torch.softmax(scores, dim=1).squeeze(-1)  # (batch, 48)
        attn_weights.append(weights.cpu().numpy())

avg_attn = np.vstack(attn_weights).mean(axis=0)    # (48,)
hours_ago = np.arange(LOOKBACK, 0, -1)             # 48 … 1

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(hours_ago, avg_attn, color="steelblue", alpha=0.8, width=0.8)
ax.invert_xaxis()
ax.set_xlabel("Hours ago (relative to forecast start)")
ax.set_ylabel("Average attention weight")
ax.set_title("Temporal attention — which hours in the 48 h window matter most")
ax.axvline(24, color="tomato", lw=1.2, ls="--", label="24 h ago (yesterday)")
ax.axvline(48, color="orange", lw=1.2, ls="--", label="48 h ago")
ax.legend()
plt.tight_layout()
plt.savefig("./fig_attention_weights.png", dpi=120)
plt.show()
"""),

# ── Predict Dec 25–31 ─────────────────────────────────────────────────────────
m("""\
## Predict Dec 25, 2024 (24 h demo)

We use the **48 h lookback window Dec 23 00:00 → Dec 24 23:00** (fully known history)
to forecast all 24 hours of **Dec 25, 2024**.

> *Extending to Dec 26–31 would require rolling actual prices forward as each day
> becomes known — out of scope for this assessment run.*
"""),

c("""\
# 48 h lookback window ending at 2024-12-24 23:00
lb_start = "2024-12-23 00:00"
lb_end   = "2024-12-24 23:00"
context  = master.loc[lb_start:lb_end, feature_cols].values
assert len(context) == LOOKBACK, f"Expected {LOOKBACK} rows, got {len(context)}"

x_input = torch.tensor(
    scaler_X.transform(context), dtype=torch.float32
).unsqueeze(0).to(DEVICE)   # shape (1, 48, n_features)

with torch.no_grad():
    pred_scaled = model(x_input).cpu().numpy()       # (1, 24)

pred_prices = scaler_y.inverse_transform(pred_scaled).flatten()

dec25_idx  = pd.date_range("2024-12-25 00:00", periods=24, freq="1h")
pred_series = pd.Series(pred_prices, index=dec25_idx, name="predicted_HAY2201_$/MWh")

print("Dec 25, 2024 — 24 h price forecast (HAY2201):")
print(pred_series.to_string())
"""),

c("""\
# Pull actual Dec 25 prices from master (they exist — just weren't used in training)
actual_dec25 = master.loc["2024-12-25 00:00":"2024-12-25 23:00", TARGET]

mae_dec25  = np.abs(pred_series.values - actual_dec25.values).mean()
rmse_dec25 = np.sqrt(((pred_series.values - actual_dec25.values) ** 2).mean())
print(f"Dec 25 holdout  MAE: {mae_dec25:.2f} $/MWh   RMSE: {rmse_dec25:.2f} $/MWh")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(actual_dec25.index, actual_dec25.values, lw=2, color="steelblue",
        marker="o", ms=4, label="Actual HAY2201")
ax.plot(pred_series.index, pred_series.values,   lw=2, color="darkorange",
        marker="o", ms=4, label="Predicted HAY2201", alpha=0.85)
ax.fill_between(pred_series.index,
                pred_series.values, actual_dec25.values,
                alpha=0.15, color="tomato", label="Error")
ax.set_ylabel("Price ($/MWh)")
ax.set_title(f"Dec 25, 2024 — Predicted vs Actual  |  MAE {mae_dec25:.1f} $/MWh  RMSE {rmse_dec25:.1f} $/MWh")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_xlabel("Hour (Dec 25 NZT)")
ax.legend()
plt.tight_layout()
plt.savefig("./fig_prediction_dec25.png", dpi=120)
plt.show()
"""),

c("""\
# Save prediction to CSV
pred_df_out = pred_series.reset_index()
pred_df_out.columns = ["datetime", "predicted_price_HAY2201_$/MWh"]
pred_df_out.to_csv("./predictions_dec25_2024.csv", index=False)
print("Saved → claude_code/predictions_dec25_2024.csv")
pred_df_out
"""),

# ── Summary ───────────────────────────────────────────────────────────────────
m("## Results Summary"),

c("""\
print("=" * 55)
print(" NZ Electricity Price LSTM — Results Summary")
print("=" * 55)
print(f" Target node    : HAY2201 (Haywards)")
print(f" Training data  : 2021-05-08 → 2024-06-30")
print(f" Lookback       : {LOOKBACK} h  |  Horizon: {HORIZON} h")
print(f" Features       : {len(feature_cols)}")
print()
print(f" Test set (Nov 1 – Dec 24, 2024)")
print(f"   LSTM  MAE  : {mae:.2f}  $/MWh")
print(f"   LSTM  RMSE : {rmse:.2f}  $/MWh")
print(f"   LSTM  MAPE : {mape:.1f}%")
print()
print(f" Naive baseline (same-hour-yesterday)")
print(f"   Naive MAE  : {n_mae:.2f}  $/MWh")
print(f"   Naive RMSE : {n_rmse:.2f}  $/MWh")
print()
print(f" LSTM vs naive MAE  : {(n_mae-mae)/n_mae*100:+.1f}%")
print(f" LSTM vs naive RMSE : {(n_rmse-rmse)/n_rmse*100:+.1f}%")
print("=" * 55)
print()
print(" Verdict:")
if mae < n_mae * 0.85:
    print("  The model shows meaningful improvement (>15%) over the")
    print("  naive baseline — the dataset is sufficient for an okayish")
    print("  24 h price forecast without fine-tuning.")
elif mae < n_mae:
    print("  The model beats the naive baseline but only marginally.")
    print("  The dataset is partially sufficient; additional features")
    print("  (e.g. fuel prices, outage schedules) could help.")
else:
    print("  The model does not beat the naive baseline on this test set.")
    print("  Possible causes: data coverage, feature relevance,")
    print("  or extreme price spikes dominating the loss.")
print("=" * 55)
"""),

]  # end nb2_cells


# ─────────────────────────────────────────────────────────────────────────────
# Write notebooks
# ─────────────────────────────────────────────────────────────────────────────

nb1 = mk_nb(nb1_cells)
nb2 = mk_nb(nb2_cells)

out1 = HERE / "01_data_preparation.ipynb"
out2 = HERE / "02_lstm_model.ipynb"

with open(out1, "w") as f:
    nbformat.write(nb1, f)
with open(out2, "w") as f:
    nbformat.write(nb2, f)

print(f"Written: {out1}")
print(f"Written: {out2}")
