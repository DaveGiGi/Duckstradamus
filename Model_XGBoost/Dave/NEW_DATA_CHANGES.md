# New Data Integration — Changes Summary

This document describes every structural change made to the XGBoost model
when the new preprocessed dataset was introduced.
It is written to be fed to the LSTM model session so the same changes
can be applied there. Apply what is relevant; skip what is LSTM-specific.

---

## 1. Data loading — completely replaced

### Before (old)
```python
df = pd.read_csv('../../Data_Processing/preprocessed_data.csv')
df["datetime_utc12"] = pd.to_datetime(df["datetime_utc12"])
df = df.sort_values("datetime_utc12").reset_index(drop=True)
target_col = "el_price_dol_MWh_OTA2201"
```

### After (new)
```python
target_col = "el_price_dol_MWh_OTA2201"

df = run_full_preprocessing(
    data_folder = "/Users/Dave/code/DaveGiGi/98-Project-Duckstradamus/Data_Processing/data_input",
    start_date  = "2019-01-01",
    end_date    = "2024-12-31"
)

df = prepare_xgboost_features(
    df                  = df,
    target              = "el_price_dol_MWh_OTA2201",
    skew_threshold      = 0.9,
    target_corr_cutoff  = 0.20,
    top_n_lags          = 3,
    multicoll_threshold = 0.9,
    verbose             = True,
)

df = build_xgboost_dataset(df)
```

**What this means for the LSTM:**
The same three function calls should be used to load the data.
The functions handle all cleaning, lag creation, feature selection and
log-transformation internally. No manual preprocessing is needed afterwards.

---

## 2. Datetime is now the index — not a column

### Before
`datetime_utc12` was a regular column. All datetime references used `df["datetime_utc12"]`.

### After
`datetime_utc12` is the **DataFrame index**. All datetime references use `df.index`.

```python
# Sorting
df = df.sort_index()

# Date range print
print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")

# Extracting timestamps in CV loop
timestamps = df.index[test_idx]

# Filtering by date (e.g. seasonal plots)
mask = (df.index >= start_ts) & (df.index < end_ts)
```

**Time feature extraction from index:**
```python
df["hour"]       = df.index.hour
df["dayofweek"]  = df.index.dayofweek
df["month"]      = df.index.month
df["dayofyear"]  = df.index.dayofyear
df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
```

---

## 3. Target is log-transformed

The target column `el_price_dol_MWh_OTA2201` is now in **log-space**.
Raw values look like `4.31`, `3.96`, `4.12` (= log of ~74, 52, 62 NZD/MWh).

**Every metric and every plot must back-transform with `np.exp()` before use.**
The model trains on log-price — only the output needs converting.

```python
# Predictions — back-transform before metrics
y_pred   = np.exp(model.predict(X_test))
y_actual = np.exp(y_test.values)

mae  = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2   = r2_score(y_actual, y_pred)
```

Do NOT apply `np.exp()` to raw feature columns (weather, lake levels, generation etc.)
— only to things that are lagged/rolled versions of the log-target, and to
model predictions.

---

## 4. Naive baseline column — new name

### Before
```python
naive_col = "target_lag_24h"
y_naive   = X_te[naive_col].values   # raw price
```

### After
The lag of the target is now named after the target column itself,
and is **already in log-space** (it is a lag of the log-transformed target).
Back-transform it exactly as you would a prediction.

```python
NAIVE_COL = "el_price_dol_MWh_OTA2201__lag_24h"   # log-scale

y_naive = np.exp(X_te[NAIVE_COL].values)   # → NZD/MWh
```

Verify the column exists:
```python
print([c for c in df.columns if "OTA2201" in c])
```

---

## 5. Feature column naming convention — changed

### Before
```python
"target_lag_24h"
"target_lag_168h"
"rolling_mean_24h"
"rolling_std_24h"
```

### After
All feature columns follow the pattern `{original_col}__{operation}`:
```
el_price_dol_MWh_OTA2201__lag_24h
el_price_dol_MWh_OTA2201__roll_72h
el_price_dol_MWh_OTA2201__roll_168h
Coal__lag_24h
Gas__roll_72h
```

The double underscore `__` separates the column name from the operation.
Operations include `lag_Xh` and `roll_Xh`.

---

## 6. No manual preprocessing needed

### Before
The model file contained:
- `df.ffill()`
- 24h lag shifts for production, demand, other price nodes
- Rolling mean/std computation (24h, 168h, 8760h windows)
- `df.dropna()`

### After
All of the above is handled by the three pipeline functions in step 1.
The model file only adds **time features** (step 2 above) and does a **NaN check**:

```python
df = df.sort_index()

# time features from index (see step 2)

# NaN check — merge or pipeline edge cases only
missing = df.isna().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    df = df.dropna()
```

---

## 7. Feature matrix shape

Old data: **99 features**, 43,825 rows (2020–2024)
New data: **38 features**, 50,221 rows (2019–2024)

The feature selection step (`prepare_xgboost_features`) automatically selects
only features with sufficient correlation to the target and removes
multicollinear ones. The LSTM should use the same feature set.

---

## 8. Back-transform checklist

Apply `np.exp()` to:
- ✅ Model predictions
- ✅ Actual target values (`y_test`, `y_train`, etc.)
- ✅ Naive baseline column (`el_price_dol_MWh_OTA2201__lag_24h`)
- ✅ Any other column that is a lag/roll of the log-target

Do NOT apply `np.exp()` to:
- ❌ Raw feature columns (Coal, Gas, lake levels, weather, demand, etc.)
- ❌ Feature importance scores
- ❌ Model weights / parameters

---

## 9. Imports required

```python
import sys
sys.path.append('../../Data_Processing')
from full_cleaning_preprocessing_script import run_full_preprocessing
from xgboost_feature_selection_script import prepare_xgboost_features, build_xgboost_dataset
```
