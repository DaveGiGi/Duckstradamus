
# ==========================================
# DUCKSTRADAMUS — CONSOLIDATED PREPROCESSING PIPELINE
# ==========================================
#
# PURPOSE:
#   Single-file version of all cleaning and preprocessing logic.
#   Replaces the four separate modules (clean_emi_data.py,
#   preprocess_emi_data.py, clean_generation_output_data.py,
#   preprocessing_Anu.py) and the main_preprocess.ipynb orchestration.
#
# HOW TO USE:
#   from full_cleaning_preprocessing_script import run_full_preprocessing
#
#   df_master = run_full_preprocessing(
#       data_folder       = "data_input",
#       start_date        = "2019-01-01",
#       end_date          = "2024-12-31",
#       save_path         = "preprocessed_data.csv", # it is optional, only if you want to save
#       save_intermediate = False   # set True to also save per-source CSVs
#   )
#
# WHAT CHANGED vs. ORIGINAL MODULES:
#   1. Hardcoded file paths in clean_emi_data.py → replaced with data_folder arg
#   2. Hardcoded save paths in all modules → optional via save_intermediate flag
#   3. Date filtering for wholesale price was inline in the notebook → moved into
#      preprocess_wholesale_price() as optional start_date / end_date params
#   4. Wind, Solar, Temperature had no date filtering → added optional params
#   5. All underlying logic is otherwise UNCHANGED
#
# EXPECTED data_folder STRUCTURE:
#   data_folder/
#     wholesale_prices.csv
#     demand_by_zone.csv
#     hvdc_transfer.csv
#     scheduled_outages.csv
#     generation_output_merged.csv
#     Wind_data_100m.csv
#     Solar_data.csv
#     Temperature_data.csv
#     Lakes storage levels/     ← subfolder, one CSV per lake
# ==========================================

import os
import pandas as pd
import holidays


# ==========================================
# SECTION 1 — WHOLESALE PRICE
# Original sources: clean_emi_data.py, preprocess_emi_data.py
# ==========================================

def clean_wholesale_price(data_folder, save_intermediate=False):
    """
    Loads raw wholesale price CSV, unpivots region nodes into columns,
    and converts timestamps from NZ local time (Pacific/Auckland) to
    UTC+12 fixed offset (Etc/GMT-12), removing daylight saving variation.

    DST handling:
      - Rows at :40 and :50 are extra-hour rows produced when clocks go back.
        They are extracted, remapped to :00 and :30, and reinserted after
        the main conversion.
      - Rows that fall exactly on DST boundary (NaT after localize) are dropped.

    Args:
        data_folder (str): Folder containing raw input files.
        save_intermediate (bool): If True, saves cleaned CSV to data_output/.

    Returns:
        pd.DataFrame: Cleaned wholesale price with datetime column in UTC+12.
    """
    file_path = os.path.join(data_folder, "wholesale_prices.csv")
    wholesale_price = pd.read_csv(file_path)

    # Select and rename relevant columns
    wholesale_price = wholesale_price[
        ['Period start', 'Region ID', 'Price ($/MWh)']
    ].rename(columns={
        'Period start':   'datetime',
        'Region ID':      'region_id',
        'Price ($/MWh)':  'price_dol/MWh'
    })

    # Parse datetime strings (NZ data uses day-first format)
    wholesale_price['datetime'] = pd.to_datetime(
        wholesale_price['datetime'], dayfirst=True
    )

    # Pivot: one column per region node
    wholesale_price = wholesale_price.pivot(
        index='datetime', columns='region_id', values='price_dol/MWh'
    )
    wholesale_price.columns = [
        f'el_price_dol/MWh_{col}' for col in wholesale_price.columns
    ]

    # --- DST handling ---
    # :40 and :50 rows are the extra half-hours from clock going back (DST end)
    extra_hour_data   = wholesale_price[wholesale_price.index.minute.isin([40, 50])].copy()
    wholesale_price_clean = wholesale_price[~wholesale_price.index.minute.isin([40, 50])].copy()

    # Convert NZ local → UTC+12 fixed offset
    wholesale_price_utc12 = wholesale_price_clean.copy()
    wholesale_price_utc12.index = (
        pd.to_datetime(wholesale_price_utc12.index)
        .tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='NaT')
        .tz_convert('Etc/GMT-12')
        .tz_localize(None)
    )
    # Drop rows that landed on a DST boundary (NaT after localize)
    wholesale_price_utc12 = wholesale_price_utc12[wholesale_price_utc12.index.notna()]

    # Remap extra-hour rows: :40 → :00, :50 → :30
    extra_hour_data.index = extra_hour_data.index.map(
        lambda ts: ts.replace(minute=0 if ts.minute == 40 else 30)
    )

    # Reinsert extra-hour rows and sort chronologically
    wholesale_price_utc12 = pd.concat(
        [wholesale_price_utc12, extra_hour_data]
    ).sort_index()

    # Sanity check: flag any remaining duplicate timestamps
    dupes = wholesale_price_utc12[wholesale_price_utc12.index.duplicated(keep=False)]
    if not dupes.empty:
        print(f"⚠️  Duplicate timestamps in wholesale price after conversion:\n{dupes}")

    # Reset index so datetime becomes a regular column
    wholesale_price_utc12.reset_index(inplace=True)

    if save_intermediate:
        out = "data_output/wholesale_price_utc12.csv"
        wholesale_price_utc12.to_csv(out, index=False)
        print(f"   ↳ saved intermediate → {out}")

    return wholesale_price_utc12


def preprocess_wholesale_price(df, start_date=None, end_date=None, save_intermediate=False):
    """
    Aggregates 30-min wholesale price data to hourly buckets using mean.

    Bucketing logic:
      timestamp + 30min → floor to hour
      e.g. 01:00 → 01:30 → 02:00  and  01:30 → 02:00 → 02:00
      Both half-hours are averaged into the 02:00 bucket.

    Date filtering is applied here (was previously done inline in the notebook).

    Args:
        df (pd.DataFrame): Output of clean_wholesale_price().
        start_date (str): Optional lower bound, e.g. "2019-01-01".
        end_date (str): Optional upper bound, e.g. "2024-12-31".
        save_intermediate (bool): If True, saves preprocessed CSV to data_output/.

    Returns:
        pd.DataFrame: Hourly wholesale prices with datetime_utc12 column.
    """
    df['hour_bucket'] = (
        df['datetime'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly = (
        df
        .groupby('hour_bucket')
        .agg(
            datetime_utc12             = ('datetime',                   'max'),
            el_price_dol_MWh_BEN2201  = ('el_price_dol/MWh_BEN2201',  'mean'),
            el_price_dol_MWh_HAY2201  = ('el_price_dol/MWh_HAY2201',  'mean'),
            el_price_dol_MWh_INV2201  = ('el_price_dol/MWh_INV2201',  'mean'),
            el_price_dol_MWh_ISL2201  = ('el_price_dol/MWh_ISL2201',  'mean'),
            el_price_dol_MWh_KIK2201  = ('el_price_dol/MWh_KIK2201',  'mean'),
            el_price_dol_MWh_OTA2201  = ('el_price_dol/MWh_OTA2201',  'mean'),
            el_price_dol_MWh_RDF2201  = ('el_price_dol/MWh_RDF2201',  'mean'),
            el_price_dol_MWh_SFD2201  = ('el_price_dol/MWh_SFD2201',  'mean'),
            el_price_dol_MWh_WKM2201  = ('el_price_dol/MWh_WKM2201',  'mean'),
        )
        .reset_index(drop=True)
        .sort_values('datetime_utc12')
        .reset_index(drop=True)
    )

    # Apply date range filter (standardised — was previously done inline in notebook)
    if start_date is not None:
        df_hourly = df_hourly[df_hourly['datetime_utc12'] >= start_date]
    if end_date is not None:
        df_hourly = df_hourly[df_hourly['datetime_utc12'] <= end_date]
    df_hourly = df_hourly.reset_index(drop=True)

    if save_intermediate:
        out = "data_output/wholesale_price_preprocessed.csv"
        df_hourly.to_csv(out, index=False)
        print(f"   ↳ saved intermediate → {out}")

    return df_hourly


# ==========================================
# SECTION 2 — DEMAND PER ZONE
# Original sources: clean_emi_data.py, preprocess_emi_data.py
# ==========================================

def clean_demand_per_zone(data_folder, save_intermediate=False):
    """
    Loads raw demand CSV, unpivots zone columns, and converts to UTC+12.
    Uses identical DST-handling logic as clean_wholesale_price().

    Args:
        data_folder (str): Folder containing raw input files.
        save_intermediate (bool): If True, saves cleaned CSV to data_output/.

    Returns:
        pd.DataFrame: Cleaned demand data with datetime column in UTC+12.
    """
    file_path = os.path.join(data_folder, "demand_by_zone.csv")
    demand = pd.read_csv(file_path)

    demand = demand[
        ['Period start', 'Region ID', 'Demand (GWh)']
    ].rename(columns={
        'Period start':  'datetime',
        'Region ID':     'region_id',
        'Demand (GWh)':  'demand_GWh'
    })
    demand['datetime'] = pd.to_datetime(demand['datetime'], dayfirst=True)

    # Pivot: one column per zone
    demand = demand.pivot(index='datetime', columns='region_id', values='demand_GWh')
    demand.columns = [f'demand_GWh_{col}' for col in demand.columns]

    # --- DST handling (same logic as wholesale price) ---
    extra_hour_data = demand[demand.index.minute.isin([40, 50])].copy()
    demand_clean    = demand[~demand.index.minute.isin([40, 50])].copy()

    demand_utc12 = demand_clean.copy()
    demand_utc12.index = (
        pd.to_datetime(demand_utc12.index)
        .tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='NaT')
        .tz_convert('Etc/GMT-12')
        .tz_localize(None)
    )
    demand_utc12 = demand_utc12[demand_utc12.index.notna()]

    extra_hour_data.index = extra_hour_data.index.map(
        lambda ts: ts.replace(minute=0 if ts.minute == 40 else 30)
    )
    demand_utc12 = pd.concat([demand_utc12, extra_hour_data]).sort_index()

    dupes = demand_utc12[demand_utc12.index.duplicated(keep=False)]
    if not dupes.empty:
        print(f"⚠️  Duplicate timestamps in demand after conversion:\n{dupes}")

    demand_utc12.reset_index(inplace=True)

    if save_intermediate:
        out = "data_output/demand_utc12.csv"
        demand_utc12.to_csv(out, index=False)
        print(f"   ↳ saved intermediate → {out}")

    return demand_utc12


def preprocess_demand_per_zone(df, save_intermediate=False):
    """
    Aggregates 30-min demand data to hourly buckets using sum.

    Demand is summed (not averaged) because GWh values represent energy
    consumed in each half-hour — summing gives the correct total per hour.

    Args:
        df (pd.DataFrame): Output of clean_demand_per_zone().
        save_intermediate (bool): If True, saves preprocessed CSV to data_output/.

    Returns:
        pd.DataFrame: Hourly demand DataFrame with datetime_utc12 column.
    """
    df['hour_bucket'] = (
        df['datetime'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly_demand = (
        df
        .groupby('hour_bucket')
        .agg(
            datetime_utc12  = ('datetime',       'max'),
            demand_GWh_CNI  = ('demand_GWh_CNI', 'sum'),
            demand_GWh_LNI  = ('demand_GWh_LNI', 'sum'),
            demand_GWh_LSI  = ('demand_GWh_LSI', 'sum'),
            demand_GWh_UNI  = ('demand_GWh_UNI', 'sum'),
            demand_GWh_USI  = ('demand_GWh_USI', 'sum'),
        )
        .reset_index(drop=True)
        .sort_values('datetime_utc12')
        .reset_index(drop=True)
    )

    if save_intermediate:
        out = "data_output/demand_preprocessed.csv"
        df_hourly_demand.to_csv(out, index=False)
        print(f"   ↳ saved intermediate → {out}")

    return df_hourly_demand


# ==========================================
# SECTION 3 — HVDC TRANSFER
# Original sources: clean_emi_data.py, preprocess_emi_data.py
# ==========================================

def clean_hvdc(data_folder, save_intermediate=False):
    """
    Loads HVDC transfer data, handles duplicate timestamps, converts to UTC+12,
    and forward-fills maintenance gaps (NaN → 0) on a 30-min grid.

    The duplicate handling is more nuanced than for other sources: a second pass
    is done with ambiguous=True (rather than NaT) to recover duplicate rows that
    the first pass dropped, and they are only reinserted if a slot is still empty.

    Args:
        data_folder (str): Folder containing raw input files.
        save_intermediate (bool): If True, saves cleaned CSV to data_output/.

    Returns:
        pd.DataFrame: Cleaned HVDC data with datetime column in UTC+12.
    """
    file_path = os.path.join(data_folder, "hvdc_transfer.csv")
    hvdc = pd.read_csv(file_path)

    hvdc = hvdc[
        ['Period start', 'Direction', 'Peak flow (MW)', 'Average flow (MW)']
    ].rename(columns={
        'Period start':       'datetime',
        'Peak flow (MW)':     'peak_flow_MW',
        'Average flow (MW)':  'avg_flow_MW'
    })
    hvdc['datetime'] = pd.to_datetime(hvdc['datetime'], dayfirst=True)

    # Separate duplicates before converting — duplicates get ambiguous=True
    hvdc_dupes = hvdc[hvdc['datetime'].duplicated(keep='first')].copy()
    hvdc_clean = hvdc[~hvdc['datetime'].duplicated(keep='first')].copy()

    # Convert main batch to UTC+12
    hvdc_clean = hvdc_clean.set_index('datetime')
    hvdc_clean.index = (
        pd.to_datetime(hvdc_clean.index)
        .tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='NaT')
        .tz_convert('Etc/GMT-12')
        .tz_localize(None)
    )
    hvdc_clean = hvdc_clean[hvdc_clean.index.notna()]

    # Convert duplicate batch (ambiguous=True allows DST-end duplicates through)
    hvdc_dupes = hvdc_dupes.set_index('datetime')
    hvdc_dupes.index = (
        pd.to_datetime(hvdc_dupes.index)
        .tz_localize('Pacific/Auckland', ambiguous=True, nonexistent='NaT')
        .tz_convert('Etc/GMT-12')
        .tz_localize(None)
    )

    # Only reinsert duplicates into slots that are still empty
    new_slots = hvdc_dupes.index[~hvdc_dupes.index.isin(hvdc_clean.index)]
    hvdc_clean = pd.concat([hvdc_clean, hvdc_dupes.loc[new_slots]]).sort_index()

    # Fill maintenance gaps: resample to 30-min grid, forward-fill then zero-fill
    hvdc_utc12 = hvdc_clean.resample('30min').asfreq().fillna(0)
    hvdc_utc12.reset_index(inplace=True)

    if save_intermediate:
        out = "data_output/hvdc_utc12.csv"
        hvdc_utc12.to_csv(out, index=False)
        print(f"   ↳ saved intermediate → {out}")

    return hvdc_utc12


def preprocess_hvdc(df, save_intermediate=False):
    """
    Aggregates 30-min HVDC transfer data to hourly buckets.
    Derives a directional indicator from the sign of avg_flow_MW:
      +1 = North-to-South,  -1 = South-to-North,  0 = no flow

    Peak flow uses the reading with the largest absolute value in the hour,
    because directional magnitude matters more than the simple max.

    Args:
        df (pd.DataFrame): Output of clean_hvdc().
        save_intermediate (bool): If True, saves preprocessed CSV to data_output/.

    Returns:
        pd.DataFrame: Hourly HVDC DataFrame with datetime_utc12 column.
    """
    df['hour_bucket'] = (
        df['datetime'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly_hvdc = (
        df
        .groupby('hour_bucket')
        .agg(
            datetime_utc12  = ('datetime',      'max'),
            avg_flow_MW     = ('avg_flow_MW',   'mean'),
            # Peak flow: take the reading with the largest absolute MW value
            peak_flow_MW    = ('peak_flow_MW',  lambda x: x.loc[x.abs().idxmax()]),
        )
        .reset_index(drop=True)
        .sort_values('datetime_utc12')
        .reset_index(drop=True)
    )

    # Derive direction from sign of avg_flow
    df_hourly_hvdc['Direction'] = df_hourly_hvdc['avg_flow_MW'].map(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    if save_intermediate:
        out = "data_output/hvdc_preprocessed.csv"
        df_hourly_hvdc.to_csv(out, index=False)
        print(f"   ↳ saved intermediate → {out}")

    return df_hourly_hvdc


# ==========================================
# SECTION 4 — SCHEDULED OUTAGES
# Original sources: clean_emi_data.py, preprocess_emi_data.py
# ==========================================

def clean_outages(data_folder, save_intermediate=False):
    """
    Loads scheduled outage data, pivots by technology type (SeriesCode),
    builds a full continuous 30-min index, and forward-fills capacity values.

    Outage data is event-driven (a row = a change event, not a reading per period),
    so forward-filling is the correct way to populate the gaps.

    Args:
        data_folder (str): Folder containing raw input files.
        save_intermediate (bool): If True, saves cleaned CSV to data_output/.

    Returns:
        pd.DataFrame: Cleaned outages data with Timestamp column in UTC+12.
    """
    file_path = os.path.join(data_folder, "scheduled_outages.csv")
    df = pd.read_csv(file_path)
    df.columns = ['Timestamp', 'SeriesCode', 'Series', 'MW']

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)

    # Convert to UTC+12 (same approach as other EMI sources)
    df['Timestamp'] = (
        df['Timestamp']
        .dt.tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='NaT')
        .dt.tz_convert('Etc/GMT-12')
        .dt.tz_localize(None)
    )
    df = df[df['Timestamp'].notna()]

    # Pivot: one column per technology code (NZ_G, NZ_H, NZ_T, NZ_W, UNKN)
    df_wide = df.pivot_table(
        index='Timestamp', columns='SeriesCode', values='MW', aggfunc='first'
    )
    df_wide.columns.name = None

    # Build full 30-min grid from first to last event, then forward-fill
    full_index = pd.date_range(
        start=df_wide.index.min(),
        end=df_wide.index.max(),
        freq='30min'
    )
    df_outages = df_wide.reindex(full_index).ffill().fillna(0)
    df_outages.index.name = 'Timestamp'
    df_outages.reset_index(inplace=True)

    if save_intermediate:
        out = "data_output/scheduled_outages_utc12.csv"
        df_outages.to_csv(out, index=False)
        print(f"   ↳ saved intermediate → {out}")

    return df_outages


def preprocess_outages(df, save_intermediate=False):
    """
    Aggregates 30-min outage data to hourly buckets using mean MW per technology.

    Args:
        df (pd.DataFrame): Output of clean_outages().
        save_intermediate (bool): If True, saves preprocessed CSV to data_output/.

    Returns:
        pd.DataFrame: Hourly outages DataFrame with datetime_utc12 column.
    """
    df['hour_bucket'] = (
        df['Timestamp'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly_outages = (
        df
        .groupby('hour_bucket')
        .agg(
            datetime_utc12  = ('Timestamp', 'max'),
            outage_Gas_MW   = ('NZ_G',      'mean'),
            outage_Hyd_MW   = ('NZ_H',      'mean'),
            outage_Ter_MW   = ('NZ_T',      'mean'),
            outage_Win_MW   = ('NZ_W',      'mean'),
            outage_UNKN_MW  = ('UNKN',      'mean'),
        )
        .reset_index(drop=True)
        .sort_values('datetime_utc12')
        .reset_index(drop=True)
    )

    if save_intermediate:
        out = "data_output/outages_preprocessed.csv"
        df_hourly_outages.to_csv(out, index=False)
        print(f"   ↳ saved intermediate → {out}")

    return df_hourly_outages


# ==========================================
# SECTION 5 — GENERATION OUTPUT
# Original source: clean_generation_output_data.py
# ==========================================

def preprocess_generation_data(file_path, start_date, end_date):
    """
    Full pipeline for generation output data. Combines cleaning and preprocessing
    in a single function (the source data requires them to be tightly coupled).

    Steps:
      1. Load and fix the dual Trading_Date schema (column name changed Jan 2020)
      2. Melt from wide format (one row per plant per day, TP1..TP50 as columns)
         to long format (one row per plant per trading period)
      3. Convert NZ local wall-clock time → UTC and UTC+12 with full DST handling
      4. Aggregate 30-min periods to hourly by summing kWh per plant
      5. Standardise Fuel_Code labels (e.g. 'HYD' → 'Hydro')
      6. Pivot by Fuel_Code so each fuel type becomes a column
      7. Filter to the requested date range

    Args:
        file_path (str): Path to generation_output_merged.csv.
        start_date (str): Lower bound of date filter, e.g. "2019-01-01".
        end_date (str): Upper bound of date filter, e.g. "2024-12-31".

    Returns:
        pd.DataFrame: Hourly generation by fuel type, indexed by datetime_utc12.
    """
    df = pd.read_csv(file_path, low_memory=False)

    # --- Fix dual-schema Trading_Date column ---
    # Before 2020 the column was 'Trading_date' (lowercase d),
    # after 2020 it became 'Trading_Date' (uppercase D).
    # Consolidate into a single 'Trading_Date' column.
    df['Trading_date'] = df['Trading_date'].fillna(df['Trading_Date'])
    df = df.drop(columns=['Trading_Date'])
    df = df.rename(columns={'Trading_date': 'Trading_Date'})

    # --- Melt: wide → long ---
    # Each row becomes one plant × one trading period (30 min)
    metadata_cols = [
        'Site_Code', 'POC_Code', 'Nwk_Code', 'Gen_Code',
        'Fuel_Code', 'Tech_Code', 'Trading_Date'
    ]
    tp_cols = [f'TP{i}' for i in range(1, 51)]

    df_long = df.melt(
        id_vars=metadata_cols,
        value_vars=tp_cols,
        var_name='TP',
        value_name='generation_kwh'
    )

    # Drop rows where the trading period didn't exist
    # (TP49 and TP50 only exist on DST-end days; TP47/48 only on DST-start days)
    df_long = df_long.dropna(subset=['generation_kwh'])

    # Convert TP label to an integer offset from midnight
    # TP1 = 0 min, TP2 = 30 min, TP3 = 60 min, ...
    df_long['tp_num'] = df_long['TP'].str.replace('TP', '').astype(int)
    df_long = df_long.drop(columns=['TP'])

    # --- Datetime conversion ---

    # Step 1: NZ wall-clock time = Trading_Date + (tp_num - 1) × 30 minutes
    df_long['datetime_nz'] = (
        pd.to_datetime(df_long['Trading_Date'])
        + pd.to_timedelta((df_long['tp_num'] - 1) * 30, unit='m')
    )

    # Step 2: Localize the naive NZ times to Pacific/Auckland (handles DST)
    #   nonexistent: DST spring-forward gap → shift forward 1 hour
    #   ambiguous=True: DST fall-back overlap → treat as the first occurrence
    nz_tz_aware = df_long['datetime_nz'].dt.tz_localize(
        'Pacific/Auckland',
        nonexistent=pd.Timedelta(hours=1),
        ambiguous=True
    )

    # Step 3: Convert to UTC and UTC+12, then strip the tz label for clean naive datetimes
    df_long['datetime_utc']   = nz_tz_aware.dt.tz_convert('UTC').dt.tz_localize(None)
    df_long['datetime_utc12'] = nz_tz_aware.dt.tz_convert('Etc/GMT-12').dt.tz_localize(None)

    # Clean up helper columns no longer needed
    df_long = df_long.drop(columns=['Trading_Date', 'tp_num'])

    # Reorder columns
    final_order = [
        'datetime_nz', 'datetime_utc12', 'datetime_utc',
        'Site_Code', 'POC_Code', 'Nwk_Code', 'Gen_Code',
        'Fuel_Code', 'Tech_Code', 'generation_kwh',
    ]
    df_long = (
        df_long[final_order]
        .sort_values(['datetime_utc12', 'POC_Code'])
        .reset_index(drop=True)
    )

    # --- Aggregate 30-min → hourly ---
    # Bucket: (timestamp + 30min).floor('h')
    #   01:00 → 01:30 → floor → 02:00
    #   01:30 → 02:00 → floor → 02:00
    # Generation is summed (kWh are additive)
    df_long['hour_bucket'] = (
        df_long['datetime_utc12'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly = (
        df_long
        .groupby(
            ['hour_bucket', 'Site_Code', 'POC_Code', 'Nwk_Code',
             'Gen_Code', 'Fuel_Code', 'Tech_Code'],
            observed=True
        )
        .agg(
            datetime_nz    = ('datetime_nz',    'max'),
            datetime_utc12 = ('datetime_utc12', 'max'),
            datetime_utc   = ('datetime_utc',   'max'),
            generation_kwh = ('generation_kwh', 'sum'),
        )
        .reset_index()
        .drop(columns=['hour_bucket'])
    )

    # Final column order and sort
    final_cols = [
        'datetime_nz', 'datetime_utc12', 'datetime_utc',
        'Site_Code', 'POC_Code', 'Nwk_Code', 'Gen_Code',
        'Fuel_Code', 'Tech_Code', 'generation_kwh',
    ]
    df_hourly = (
        df_hourly[final_cols]
        .sort_values(['datetime_utc12', 'POC_Code'])
        .reset_index(drop=True)
    )

    # Drop NZ local and true UTC — only UTC+12 is used for merging
    df_hourly = df_hourly.drop(columns=['datetime_nz', 'datetime_utc'])

    # --- Standardise Fuel_Code labels ---
    # Source data has inconsistent codes across years
    fuel_mapping = {
        'HYD': 'Hydro',
        'SOL': 'Solar',
        'GEO': 'Geo',
        'WIN': 'Wind',
        'ELE': 'Ele',
    }
    df_hourly['Fuel_Code'] = df_hourly['Fuel_Code'].replace(fuel_mapping)

    # --- Pivot by Fuel_Code ---
    # Result: one row per datetime_utc12, one column per fuel type
    # Cell value = total kWh generated by that fuel in that hour across all plants
    # fill_value=0: if no plants of that fuel ran in that hour → 0 (not NaN)
    df_by_fuel = df_hourly.pivot_table(
        index='datetime_utc12',
        columns='Fuel_Code',
        values='generation_kwh',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    df_by_fuel.columns.name = None

    # Apply date range filter
    df_by_fuel = df_by_fuel[
        (df_by_fuel['datetime_utc12'] >= start_date) &
        (df_by_fuel['datetime_utc12'] <= end_date)
    ].reset_index(drop=True)

    return df_by_fuel


# ==========================================
# SECTION 6 — WIND DATA
# Original source: preprocessing_Anu.py
# ==========================================

def preprocess_wind_data(file_path, start_date=None, end_date=None):
    """
    Loads and cleans wind speed/direction data.
    Standardises column names, parses datetime, removes NaNs and duplicates.
    Date filtering added (was missing in original).

    Args:
        file_path (str): Path to Wind_data_100m.csv.
        start_date (str): Optional lower bound filter.
        end_date (str): Optional upper bound filter.

    Returns:
        pd.DataFrame: Cleaned wind DataFrame with datetime column.
    """
    df = pd.read_csv(file_path)

    # Standardise column names (lowercase, underscores, no brackets)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    df.drop(columns=["time"], inplace=True)

    # Move datetime to first column for consistency
    df = df[["datetime"] + [col for col in df.columns if col != "datetime"]]

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.sort_values("datetime")

    # Date filter (added — was missing in original module)
    if start_date is not None:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


# ==========================================
# SECTION 7 — SOLAR DATA
# Original source: preprocessing_Anu.py
# ==========================================

def preprocess_solar_data(file_path, start_date=None, end_date=None):
    """
    Loads and cleans solar irradiance data.
    Identical structure to preprocess_wind_data().
    Date filtering added (was missing in original).

    Args:
        file_path (str): Path to Solar_data.csv.
        start_date (str): Optional lower bound filter.
        end_date (str): Optional upper bound filter.

    Returns:
        pd.DataFrame: Cleaned solar DataFrame with datetime column.
    """
    df = pd.read_csv(file_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    df.drop(columns=["time"], inplace=True)
    df = df[["datetime"] + [col for col in df.columns if col != "datetime"]]

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.sort_values("datetime")

    # Date filter (added — was missing in original module)
    if start_date is not None:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


# ==========================================
# SECTION 8 — TEMPERATURE DATA
# Original source: preprocessing_Anu.py
# ==========================================

def preprocess_temperature_data(file_path, start_date=None, end_date=None):
    """
    Loads and cleans temperature data.
    Identical structure to preprocess_wind_data() and preprocess_solar_data().
    Date filtering added (was missing in original).

    Args:
        file_path (str): Path to Temperature_data.csv.
        start_date (str): Optional lower bound filter.
        end_date (str): Optional upper bound filter.

    Returns:
        pd.DataFrame: Cleaned temperature DataFrame with datetime column.
    """
    df = pd.read_csv(file_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    df.drop(columns=["time"], inplace=True)
    df = df[["datetime"] + [col for col in df.columns if col != "datetime"]]

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.sort_values("datetime")

    # Date filter (added — was missing in original module)
    if start_date is not None:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


# ==========================================
# SECTION 9 — LAKE STORAGE
# Original source: preprocessing_Anu.py
# ==========================================

def preprocess_lake_storage(file_path):
    """
    Loads a single lake storage CSV and returns key columns with a clean datetime.
    Each lake has its own file — this is called once per file inside the main loop.

    Args:
        file_path (str): Path to a single lake CSV file.

    Returns:
        pd.DataFrame: Cleaned lake storage DataFrame with columns:
                      [datetime, lake_level_m, active_storage_mm³]
    """
    df = pd.read_csv(file_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    # Combine separate date and time columns into a single datetime
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        errors="coerce"
    )

    df = df[["datetime", "lake_level_m", "active_storage_mm³"]]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def make_lake_hourly(df, start_date=None, end_date=None):
    """
    Resamples a single lake storage DataFrame to hourly frequency
    and linearly interpolates numeric columns (time-weighted).

    Why interpolate? Lake level data is sparse (often one reading per day),
    so we interpolate rather than forward-fill to produce smooth hourly values.

    Edge case handled: source data sometimes uses 23:59:59 for day-boundary
    readings — these are shifted forward by 1 second to 00:00:00 of the next day
    before resampling, to avoid misalignment.

    Args:
        df (pd.DataFrame): Output of preprocess_lake_storage().
        start_date (str): Optional lower bound filter, applied before resampling.
        end_date (str): Optional upper bound filter, applied before resampling.

    Returns:
        pd.DataFrame: Hourly lake storage DataFrame with datetime column.
    """
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Shift 23:59:59 boundary timestamps to midnight of the next day
    mask = df["datetime"].dt.time == pd.to_datetime("23:59:59").time()
    df.loc[mask, "datetime"] = df.loc[mask, "datetime"] + pd.Timedelta(seconds=1)

    df = df.sort_values("datetime")

    # Apply date filter before resampling (more efficient than filtering after)
    if start_date is not None:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]

    # Resample to 1-hour frequency
    df = df.set_index("datetime")
    hourly_df = df.resample("h").asfreq()

    # infer_objects avoids FutureWarning about dtype inference in newer pandas
    hourly_df = hourly_df.infer_objects(copy=False)

    # Interpolate numeric columns using time-based weighting
    numeric_cols = hourly_df.select_dtypes(include="number").columns
    hourly_df[numeric_cols] = hourly_df[numeric_cols].interpolate(method="time")

    return hourly_df.reset_index()


# ==========================================
# SECTION 10 — NZ HOLIDAY DATA
# Original source: preprocessing_Anu.py
# ==========================================

def create_nz_holiday_data(start_date, end_date):
    """
    Generates an hourly DataFrame from start_date to end_date with a binary
    'holiday' column that is 1 on weekends and New Zealand public holidays.

    Public holidays come from the `holidays` library (NZ subset).
    Weekends are detected via dt.weekday >= 5.

    Args:
        start_date (str): Start date, e.g. "2019-01-01".
        end_date (str): End date, e.g. "2024-12-31".

    Returns:
        pd.DataFrame: Hourly DataFrame with columns [datetime, holiday].
    """
    df = pd.DataFrame({
        "datetime": pd.date_range(start=start_date, end=end_date, freq="h")
    })

    nz_holidays = holidays.NewZealand(
        years=range(
            pd.to_datetime(start_date).year,
            pd.to_datetime(end_date).year + 1
        )
    )

    is_holiday = df["datetime"].dt.date.isin(nz_holidays)
    is_weekend = df["datetime"].dt.weekday >= 5

    # Binary flag: 1 if public holiday OR weekend, else 0
    df["holiday"] = (is_holiday | is_weekend).astype(int)

    return df


# ==========================================
# MASTER ORCHESTRATOR
# ==========================================

def run_full_preprocessing(
    data_folder,
    start_date,
    end_date,
    save_path=None,
    save_intermediate=False
):
    """
    Runs the complete Duckstradamus preprocessing pipeline end-to-end.

    All raw sources are loaded from data_folder, cleaned, aggregated to hourly,
    and left-joined onto a master DataFrame anchored on the wholesale price timeline.
    The wholesale price is used as the anchor because it defines the full date/time
    grid — every other source is joined onto it, so gaps in other sources become NaN
    rather than missing rows.

    Args:
        data_folder (str): Path to the folder containing all raw input files.
                           Expected structure:
                             data_folder/
                               wholesale_prices.csv
                               demand_by_zone.csv
                               hvdc_transfer.csv
                               scheduled_outages.csv
                               generation_output_merged.csv
                               Wind_data_100m.csv
                               Solar_data.csv
                               Temperature_data.csv
                               Lakes storage levels/   ← one CSV per lake

        start_date (str): Start of target date range, e.g. "2019-01-01".
        end_date (str): End of target date range, e.g. "2024-12-31".
        save_path (str): Output path for the final master CSV.
        save_intermediate (bool): If True, each step also saves its own CSV
                                  to data_output/. Useful for debugging.

    Returns:
        pd.DataFrame: Master DataFrame with all features joined on datetime_utc12.
    """
    print("=" * 60)
    print("DUCKSTRADAMUS — FULL PREPROCESSING PIPELINE")
    print(f"  data_folder : {data_folder}")
    print(f"  date range  : {start_date}  →  {end_date}")
    print(f"  save_path   : {save_path}")
    print("=" * 60)

    # ----------------------------------------------------------
    # STEP 1 — Wholesale price (anchor DataFrame)
    # Defines the full datetime_utc12 index that all other sources
    # are joined onto. Date filter applied here so the master df
    # spans exactly start_date → end_date from the beginning.
    # ----------------------------------------------------------
    print("\n[1/9] Wholesale price...")
    df = preprocess_wholesale_price(
        clean_wholesale_price(data_folder, save_intermediate=save_intermediate),
        start_date=start_date,
        end_date=end_date,
        save_intermediate=save_intermediate
    )
    print(f"      Anchor df shape: {df.shape}")

    # ----------------------------------------------------------
    # STEP 2 — Generation output
    # Handles its own date filter internally (start_date/end_date
    # are required args in preprocess_generation_data).
    # ----------------------------------------------------------
    print("\n[2/9] Generation output...")
    df_gen = preprocess_generation_data(
        file_path=os.path.join(data_folder, "generation_output_merged.csv"),
        start_date=start_date,
        end_date=end_date
    )
    df = df.merge(df_gen, how='left', on='datetime_utc12')
    print(f"      Shape after merge: {df.shape}")

    # ----------------------------------------------------------
    # STEP 3 — Wind
    # Joined on datetime (right) = datetime_utc12 (left).
    # The 'datetime' column is dropped after merging to avoid duplication.
    # ----------------------------------------------------------
    print("\n[3/9] Wind...")
    df_wind = preprocess_wind_data(
        file_path=os.path.join(data_folder, "Wind_data_100m.csv"),
        start_date=start_date,
        end_date=end_date
    )
    df = df.merge(df_wind, how='left', left_on='datetime_utc12', right_on='datetime')
    df = df.drop(columns=['datetime'])
    print(f"      Shape after merge: {df.shape}")

    # ----------------------------------------------------------
    # STEP 4 — Solar
    # ----------------------------------------------------------
    print("\n[4/9] Solar...")
    df_solar = preprocess_solar_data(
        file_path=os.path.join(data_folder, "Solar_data.csv"),
        start_date=start_date,
        end_date=end_date
    )
    df = df.merge(df_solar, how='left', left_on='datetime_utc12', right_on='datetime')
    df = df.drop(columns=['datetime'])
    print(f"      Shape after merge: {df.shape}")

    # ----------------------------------------------------------
    # STEP 5 — Temperature
    # ----------------------------------------------------------
    print("\n[5/9] Temperature...")
    df_temp = preprocess_temperature_data(
        file_path=os.path.join(data_folder, "Temperature_data.csv"),
        start_date=start_date,
        end_date=end_date
    )
    df = df.merge(df_temp, how='left', left_on='datetime_utc12', right_on='datetime')
    df = df.drop(columns=['datetime'])
    print(f"      Shape after merge: {df.shape}")

    # ----------------------------------------------------------
    # STEP 6 — Lake storage (loop over all lake files)
    # Each lake CSV is processed independently and merged separately.
    # Column names are prefixed with the first 7 characters of the filename
    # (e.g. "SI_WPU_") to avoid name collisions across lakes.
    # ----------------------------------------------------------
    print("\n[6/9] Lake storage...")
    lake_folder = os.path.join(data_folder, "Lakes storage levels")
    for filename in sorted(os.listdir(lake_folder)):
        filepath = os.path.join(lake_folder, filename)
        prefix = filename[:7]   # e.g. "SI_WPU_" from "SI_WPU_Storage_Lake.csv"

        df_lake = make_lake_hourly(
            preprocess_lake_storage(filepath),
            start_date=start_date,
            end_date=end_date
        )
        # Prefix all value columns (not the datetime key used for merging)
        df_lake = df_lake.rename(columns={
            col: f'{prefix}_{col}'
            for col in df_lake.columns
            if col != 'datetime'
        })
        df = df.merge(df_lake, how='left', left_on='datetime_utc12', right_on='datetime')
        df = df.drop(columns=['datetime'])
        print(f"      Merged: {filename}  (prefix: '{prefix}')")

    print(f"      Shape after all lake merges: {df.shape}")

    # ----------------------------------------------------------
    # STEP 7 — Holidays
    # Generated programmatically — no file to load.
    # ----------------------------------------------------------
    print("\n[7/9] Holidays...")
    df_holiday = create_nz_holiday_data(start_date, end_date)
    df = df.merge(df_holiday, how='left', left_on='datetime_utc12', right_on='datetime')
    df = df.drop(columns=['datetime'])
    print(f"      Shape after merge: {df.shape}")

    # ----------------------------------------------------------
    # STEP 8 — Demand per zone
    # ----------------------------------------------------------
    print("\n[8/9] Demand per zone...")
    df_demand = preprocess_demand_per_zone(
        clean_demand_per_zone(data_folder, save_intermediate=save_intermediate),
        save_intermediate=save_intermediate
    )
    df = df.merge(df_demand, how='left', on='datetime_utc12')
    print(f"      Shape after merge: {df.shape}")

    # ----------------------------------------------------------
    # STEP 9 — HVDC transfer + Scheduled outages
    # Both are merged on datetime_utc12 (produced by their preprocess functions).
    # ----------------------------------------------------------
    print("\n[9/9] HVDC and scheduled outages...")
    df_hvdc = preprocess_hvdc(
        clean_hvdc(data_folder, save_intermediate=save_intermediate),
        save_intermediate=save_intermediate
    )
    df = df.merge(df_hvdc, how='left', on='datetime_utc12')

    df_outages = preprocess_outages(
        clean_outages(data_folder, save_intermediate=save_intermediate),
        save_intermediate=save_intermediate
    )
    df = df.merge(df_outages, how='left', on='datetime_utc12')
    print(f"      Shape after merge: {df.shape}")

    # ----------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"   Saved to    : {save_path}")

    print(f"\n{'=' * 60}")
    print(f"✅ Pipeline complete.")
    print(f"   Final shape : {df.shape}")
    print(f"   Saved to    : {save_path}")
    print(f"{'=' * 60}\n")

    return df
