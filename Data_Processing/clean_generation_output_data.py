
# ==========================================
# FINAL CLEANING CODE
# GENERATION DATA PREPROCESSING
# ==========================================


def preprocess_generation_data(file_path):

    # Loading the Data
    df = pd.read_csv(file_path, low_memory = False)

    # Cleaning schema renaming in Jan 1st 2020. COnsolidating into Trading Date
    df['Trading_date'] = df['Trading_date'].fillna(df['Trading_Date'])
    df = df.drop(columns=['Trading_Date'])
    df = df.rename(columns={'Trading_date': 'Trading_Date'})

    # Unpivot TP Data into multiple rows, each row now corresponds to one date and hour
    # Melt from wide to long
    # id_vars: columns that stay fixed (will be repeated for each TP)
    # value_vars: columns to unpivot (TP1 through TP50)

    metadata_cols = ['Site_Code', 'POC_Code', 'Nwk_Code', 'Gen_Code',
                    'Fuel_Code', 'Tech_Code', 'Trading_Date']
    tp_cols = [f'TP{i}' for i in range(1, 51)]

    df_long = df.melt(
        id_vars=metadata_cols,
        value_vars=tp_cols,
        var_name='TP',
        value_name='generation_kwh'
    )

    # Drop NaN rows (non-existent trading periods)
    df_long = df_long.dropna(subset=['generation_kwh'])

    # Extract the TP number as an integer in new column, drop old TP column
    df_long['tp_num'] = df_long['TP'].str.replace('TP', '').astype(int)
    df_long = df_long.drop(columns=['TP'])

    # Convert times to local NZ time, UTC + 12, and true UTC
        # 1. NZ local wall clock = Trading_Date + (tp_num - 1) * 30 minutes
    df_long['datetime_nz'] = (
        pd.to_datetime(df_long['Trading_Date'])
        + pd.to_timedelta((df_long['tp_num'] - 1) * 30, unit='m')
    )

        # 2. Use pandas timezone tools to make Pandas aware of timezone and
        # derive true UTC and true UTC+12
            # Step A: localize the naive datetime as Pacific/Auckland (handles DST)
    nz_tz_aware = df_long['datetime_nz'].dt.tz_localize(
        'Pacific/Auckland',
        nonexistent=pd.Timedelta(hours=1),  # shift DST-gap times forward 1 hour
        ambiguous=True                       # treat DST-end duplicates as first occurrence
    )

            # Step B: convert to UTC and UTC+12, then drop the timezone label for clean naive datetimes
    df_long['datetime_utc']   = nz_tz_aware.dt.tz_convert('UTC').dt.tz_localize(None)
    df_long['datetime_utc12'] = nz_tz_aware.dt.tz_convert('Etc/GMT-12').dt.tz_localize(None)

            # Step C: drop the Trading_Date and tp_num Columns
    df_long = df_long.drop(columns=['Trading_Date'])
    df_long = df_long.drop(columns=['tp_num'])

    # Edit new order
    final_order = [
        'datetime_nz', 'datetime_utc12', 'datetime_utc',    # the three datetimes
        'Site_Code', 'POC_Code', 'Nwk_Code', 'Gen_Code',
        'Fuel_Code', 'Tech_Code',
        'generation_kwh',
    ]
    df_long = df_long[final_order].sort_values(['datetime_utc12', 'POC_Code']).reset_index(drop=True)


    # Average half hour times into one hour. Ex: 1:30 and 2:00 will be averaged into 2:00.
    # Centered hourly aggregation
    # Each "bucket H" contains TPs at (H-0:30) and H, representing the hour centered on H

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
            datetime_nz    = ('datetime_nz',    'max'),   # 👈 max instead of mean
            datetime_utc12 = ('datetime_utc12', 'max'),   # 👈
            datetime_utc   = ('datetime_utc',   'max'),   # 👈
            generation_kwh = ('generation_kwh', 'sum'),  # still mean for the value
        )
        .reset_index()
        .drop(columns=['hour_bucket'])   # duplicate of datetime_utc12 after the groupby
    )

    # Reorder columns
    final_cols = [
        'datetime_nz', 'datetime_utc12', 'datetime_utc',
        'Site_Code', 'POC_Code', 'Nwk_Code', 'Gen_Code',
        'Fuel_Code', 'Tech_Code', 'generation_kwh',
    ]
    df_hourly = (
        df_hourly[final_cols]                           # 1. column order
        .sort_values(['datetime_utc12', 'POC_Code'])    # 2. row order
        .reset_index(drop=True)                         # 3. clean index
    )


    # Drop the local NZ tiem and true UTC as it is not needed. DELETE THIS LINE IF WANT TO KEEP
    df_hourly = df_hourly.drop(columns=['datetime_nz'])
    df_hourly = df_hourly.drop(columns=['datetime_utc'])


    # Fix the Fuel_Codes labelling inconsistencies
    fuel_mapping = {
        'HYD': 'Hydro',
        'SOL': 'Solar',
        'GEO': 'Geo',
        'WIN': 'Wind',
        'ELE': 'Ele',     # renamed but kept separate
    }
    df_hourly['Fuel_Code'] = df_hourly['Fuel_Code'].replace(fuel_mapping)


    # Make UTC + 12 Primary Key by grouping by Fuel_code
    df_by_fuel = df_hourly.pivot_table(
        index='datetime_utc12',     # rows = unique datetimes
        columns='Fuel_Code',        # columns = each fuel code
        values='generation_kwh',    # cell values
        aggfunc='sum',              # sum kWh across all plants of that fuel
        fill_value=0                # if no plants of that fuel ran in that hour → 0
    ).reset_index()
    df_by_fuel.columns.name = None
