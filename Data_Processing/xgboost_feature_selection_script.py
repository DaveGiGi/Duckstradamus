import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_xgboost_features(
    df,
    target,
    skew_threshold=0.9,
    target_corr_cutoff=0.20,
    top_n_lags=3,
    multicoll_threshold=0.9,
    verbose=True,
):
    """
    Full XGBoost feature preparation pipeline.

    Takes a cleaned DataFrame (post-`run_full_preprocessing`) and returns
    a feature matrix `X_final` ready for chronological splitting and training.

    Pipeline:
      1. Add aggregate features (total_demand, total_lakes, total_wind, total_solar)
      2. Distribution review → log1p transform of skewed features and target
      3. Lagged correlation analysis (24h, 72h, 168h, 336h, rolling 72h/168h)
      4. Feature selection: keep top N lags per feature above target-corr cutoff
      5. Multicollinearity pruning between selected features

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, datetime-indexed dataframe with all numeric features.
    target : str
        Name of the target column (e.g. 'el_price_dol_MWh_OTA2201').
    skew_threshold : float
        |skew| above which a feature is log1p-transformed. Default 0.9.
    target_corr_cutoff : float
        Minimum |corr| between a lagged feature and the target to be kept. Default 0.20.
    top_n_lags : int
        Max number of lags retained per feature. Default 3.
    multicoll_threshold : float
        |corr| between features above which one of the pair is dropped. Default 0.9.
    verbose : bool
        If True, prints progress and shows visualizations.

    Returns
    -------
    dict with keys:
      'X_final'                : pd.DataFrame — final feature matrix for XGBoost
      'df_transformed'         : pd.DataFrame — the working df after aggregates + log transforms
      'target'                 : str
      'target_was_logged'      : bool — IMPORTANT for inverse-transforming predictions
      'features_logged'        : list of str — features that received log1p
      'dist_summary'           : pd.DataFrame — distribution review output
      'feature_selection_dict' : dict — selected lags per feature
      'features_dropped_multicoll' : set of str — features dropped during multicollinearity
    """

    df = df.copy()
    df = df.set_index('datetime_utc12')

    if verbose:
        print(f"Starting pipeline: df shape {df.shape}, target = {target}")
        print("="*70)

    # ---------------- Step 1: Aggregate features ----------------
    if verbose: print("\n[1/5] Adding aggregate features...")

    demand_cols = [c for c in df.columns if 'demand' in c.lower()]
    df['total_demand'] = df[demand_cols].sum(axis=1)

    lake_cols = [c for c in df.columns if 'storage' in c.lower()]
    df['total_lakes'] = df[lake_cols].sum(axis=1)

    wind_cols = [
        'palmerston_north_wind_kmh',
        'wellington_wind_kmh',
        'harapaki_hawkesbay_wind_kmh',
        'te_uku_waikato_wind_kmh',
        'kaiwera_downs_southland_wind_kmh',
    ]
    df['total_wind'] = df[wind_cols].sum(axis=1)

    solar_cols = [c for c in df.columns if 'sunshine' in c.lower() or 'shortwave' in c.lower()]
    df['total_solar'] = df[solar_cols].sum(axis=1)

    if verbose:
        print(f"   Added 4 aggregate features. df shape: {df.shape}")

    # ---------------- Step 2: Distribution review + log1p ----------------
    if verbose: print(f"\n[2/5] Distribution review (skew threshold = {skew_threshold})...")

    dist_rows = []
    for col in df.select_dtypes(include='number').columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        sk      = skew(s)
        min_val = s.min()
        can_log = min_val >= 0
        needs_log = (abs(sk) >= skew_threshold) and can_log
        dist_rows.append({
            'feature':             col,
            'skew':                round(sk, 2),
            'min':                 round(min_val, 2),
            'max':                 round(s.max(), 2),
            'can_log':             can_log,
            'needs_log_transform': needs_log,
        })

    dist_summary = (
        pd.DataFrame(dist_rows)
        .sort_values('skew', key=abs, ascending=False)
        .reset_index(drop=True)
    )

    features_to_log = dist_summary.loc[dist_summary['needs_log_transform'], 'feature'].tolist()
    target_was_logged = target in features_to_log

    for col in features_to_log:
        df[col] = np.log1p(df[col])

    if verbose:
        print(f"   Log-transformed {len(features_to_log)} features.")
        print(f"   Target log-transformed: {target_was_logged}")
        if target_was_logged:
            print(f"   ⚠️  REMEMBER: predictions will be in log-space — use np.expm1() to recover dollars")

    # ---------------- Step 3: Lagged correlation analysis ----------------
    if verbose: print(f"\n[3/5] Lagged correlation analysis...")

    numeric_df = df.select_dtypes(include='number')

    def _corr_for_lag(lag_hours, kind='shift'):
        if kind == 'shift':
            df_lagged = numeric_df.shift(lag_hours).copy()
        else:  # rolling mean
            df_lagged = numeric_df.shift(1).rolling(window=lag_hours).mean().copy()
        df_lagged['_y_now_'] = df[target]
        return df_lagged.corr()['_y_now_'].drop('_y_now_')

    corr_today = numeric_df.corr()[target].sort_values()

    lag_specs = {
        'corr_24h':         ('shift',    24),
        'corr_72h':         ('shift',    72),
        'corr_168h':        ('shift',   168),
        'corr_336h':        ('shift',   336),
        'corr_rolling_72':  ('rolling',  72),
        'corr_rolling_168': ('rolling', 168),
    }

    all_corr_dfs = {}
    for lag_name, (kind, n) in lag_specs.items():
        corr_lagged = _corr_for_lag(n, kind=kind)
        corr_lagged_aligned = corr_lagged[corr_today.index]
        all_df = pd.DataFrame({
            'feature': corr_today.index,
            lag_name:  corr_lagged_aligned.values,
        })
        all_corr_dfs[lag_name] = (all_df, lag_name)

    if verbose: print(f"   Computed 6 lag correlations.")

    # ---------------- Step 4: Feature selection (top N lags) ----------------
    if verbose:
        print(f"\n[4/5] Feature selection (|corr| ≥ {target_corr_cutoff}, top {top_n_lags} lags)...")

    feature_lags = {}
    for lag_name, (df_corr, lag_col) in all_corr_dfs.items():
        for _, row in df_corr.iterrows():
            feat = row['feature']
            corr_lag_val = row[lag_col]
            feature_lags.setdefault(feat, []).append((lag_name, corr_lag_val))

    rows = []
    for feat, lag_list in feature_lags.items():
        lag_list_sorted = sorted(lag_list, key=lambda x: abs(x[1]), reverse=True)
        kept = [(lag, c) for lag, c in lag_list_sorted[:top_n_lags] if abs(c) >= target_corr_cutoff]
        if not kept:
            continue
        for rank, (lag_name, c) in enumerate(kept, start=1):
            rows.append({'feature': feat, 'rank': rank, 'lag': lag_name, 'lag_corr': c})

    feature_selection_df = pd.DataFrame(rows).sort_values(['feature', 'rank']).reset_index(drop=True)

    feature_selection_dict = {}
    for feat, group in feature_selection_df.groupby('feature', sort=False):
        feature_selection_dict[feat] = group[['rank', 'lag', 'lag_corr']].to_dict(orient='records')

    if verbose:
        print(f"   Features kept: {feature_selection_df['feature'].nunique()}")
        print(f"   Total lagged features: {len(feature_selection_df)}")

    # ---------------- Step 5: Build feature matrix + multicollinearity pruning ----------------
    if verbose:
        print(f"\n[5/5] Building feature matrix + multicollinearity pruning (|corr| > {multicoll_threshold})...")

    lag_map = {
        'corr_24h':         ('shift',    24),
        'corr_72h':         ('shift',    72),
        'corr_168h':        ('shift',   168),
        'corr_336h':        ('shift',   336),
        'corr_rolling_72':  ('rolling',  72),
        'corr_rolling_168': ('rolling', 168),
    }

    lagged_features = {}
    target_corr_lookup = {}

    for feat, lag_records in feature_selection_dict.items():
        for record in lag_records:
            lag_name = record['lag']
            kind, n = lag_map[lag_name]
            if kind == 'shift':
                col_name = f'{feat}__lag_{n}h'
                lagged_features[col_name] = df[feat].shift(n)
            else:
                col_name = f'{feat}__roll_{n}h'
                lagged_features[col_name] = df[feat].shift(1).rolling(window=n).mean()
            target_corr_lookup[col_name] = abs(record['lag_corr'])

    X_features = pd.DataFrame(lagged_features, index=df.index)
    X_clean = X_features.dropna()
    feat_corr = X_clean.corr().abs()

    upper = feat_corr.where(np.triu(np.ones(feat_corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'feat_a', 'level_1': 'feat_b', 0: 'abs_corr'})
        .query('abs_corr >= @multicoll_threshold')
        .sort_values('abs_corr', ascending=False)
        .reset_index(drop=True)
    )

    to_drop = set()
    target_prefix = f"{target}__"
    for _, row in pairs.iterrows():

        a = row['feat_a']
        b = row['feat_b']

        if a in to_drop or b in to_drop:
            continue

        a_is_target_lag = a.startswith(target_prefix)
        b_is_target_lag = b.startswith(target_prefix)

        # -------------------------------------------------
        # NEVER PRUNE TARGET LAGS
        # -------------------------------------------------

        # Target lag vs target lag
        if a_is_target_lag and b_is_target_lag:
            continue

        # Target lag vs non-target lag
        if a_is_target_lag and not b_is_target_lag:
            to_drop.add(b)
            continue

        # Non-target lag vs target lag
        if b_is_target_lag and not a_is_target_lag:
            to_drop.add(a)
            continue

        if target_corr_lookup[a] >= target_corr_lookup[b]:
            to_drop.add(b)
        else:
            to_drop.add(a)

    if verbose:
        target_lags = [
            c for c in X_features.columns
            if c.startswith(target_prefix)
        ]

        print("\nTarget lag features generated:")
        for c in target_lags:
            print(f"   {c}")

        print("\nTarget lag features dropped:")
        for c in target_lags:
            if c in to_drop:
                print(f"   {c}")

    X_final = X_features.drop(columns=list(to_drop))

    if verbose:
        print(f"   Highly-correlated pairs: {len(pairs)}")
        print(f"   Features dropped: {len(to_drop)}")
        print(f"   X_final shape: {X_final.shape}")
        print("\n" + "="*70)
        print(f"DONE. X_final has {X_final.shape[1]} features and {X_final.shape[0]} rows.")
        if target_was_logged:
            print(f"⚠️  Target was log-transformed. Use np.expm1(predictions) to recover dollars.")

        # ---------- Optional visualizations ----------
        plt.figure(figsize=(20, 18))
        sns.heatmap(X_clean.corr(), cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, cbar_kws={'label': 'correlation'})
        plt.title(f'Feature-feature correlation (before pruning) — {X_clean.shape[1]} features')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(16, 14))
        sns.heatmap(X_final.dropna().corr(), cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, cbar_kws={'label': 'correlation'})
        plt.title(f'Feature-feature correlation (after pruning) — {X_final.shape[1]} features')
        plt.tight_layout()
        plt.show()

    return {
        'X_final':                    X_final,
        'df_transformed':             df,
        'target':                     target,
        'target_was_logged':          target_was_logged,
        'features_logged':            features_to_log,
        'dist_summary':               dist_summary,
        'feature_selection_dict':     feature_selection_dict,
        'features_dropped_multicoll': to_drop,
    }



"""
How to call it
pythonresult = prepare_xgboost_features(
    df,
    target='el_price_dol_MWh_OTA2201',
    verbose=True,
)

X_final           = result['X_final']
df_transformed    = result['df_transformed']
target_was_logged = result['target_was_logged']
features_logged   = result['features_logged']

The df_transformed is what you'll use to extract y (the target column). It contains the log-transformed
values — so y = df_transformed[target] gives you log-prices if logging happened.
Three things worth knowing
1. Why the function returns df_transformed too. You need the transformed target alongside X_final
for training. Returning both keeps them in sync — no risk of accidentally using a raw-target series
with a log-feature matrix.
2. The features_logged list is your record. If you ever ask "did demand_north get logged?", just
check 'demand_north' in result['features_logged']. The function records every decision so you can audit.
3. NaN handling for XGBoost. X_final will have ~336 NaN rows at the top (from the longest
rolling/shift). XGBoost can handle NaN natively, so this isn't strictly an error — but for
clean training, do X_final = X_final.dropna() and align y accordingly before splitting. Same
with chronological train/test split.
A small style note for production scripts
Once you're done debugging, you can drop verbose=False and the function becomes silent for use
in pipelines (the visualizations only render when verbose=True). That's the typical pattern:
notebooks call it with verbose=True for exploration, scripts and production code call with verbose=False.
Want to do a quick test run to confirm it produces the same X_final.shape as your manual
notebook execution? That would catch any bug before we move on.
"""


def build_xgboost_dataset(result):
    """
    Takes the dict returned by prepare_xgboost_features and assembles a single
    DataFrame ready for XGBoost training.

    Returns
    -------
    df_xgb : pd.DataFrame
        Datetime-indexed, containing:
          - All lagged feature columns (e.g. demand_north__lag_24h, etc.)
          - The target column (log-transformed if applicable)
        NaN warmup rows from the longest lag (~336 hours) are dropped.
        Ready to split chronologically into X/y for training.
    """
    X_final         = result['X_final']
    df_transformed  = result['df_transformed']
    target          = result['target']

    # Align target with X_final's index (same datetime index either way, but explicit)
    y = df_transformed.loc[X_final.index, target]

    # Combine into a single DataFrame
    df_xgb = X_final.copy()
    df_xgb[target] = y

    # Drop the warmup NaN rows (~336 from the longest shift/rolling)
    df_xgb = df_xgb.dropna()

    return df_xgb
