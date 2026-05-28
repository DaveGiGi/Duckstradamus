# Project Duckstradamus
# David Giger

#27.05.2026

import pandas as pd

def preprocess_wholesale_price(df):

    save_path = 'data_output/wholesale_price_preprocessed.csv'

    df['hour_bucket'] = (
    df['datetime'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly = (
    df
    .groupby('hour_bucket')
    .agg(
        datetime_utc12              = ('datetime',                    'max'),
        el_price_dol_MWh_BEN2201   = ('el_price_dol/MWh_BEN2201',   'mean'),
        el_price_dol_MWh_HAY2201   = ('el_price_dol/MWh_HAY2201',   'mean'),
        el_price_dol_MWh_INV2201   = ('el_price_dol/MWh_INV2201',   'mean'),
        el_price_dol_MWh_ISL2201   = ('el_price_dol/MWh_ISL2201',   'mean'),
        el_price_dol_MWh_KIK2201   = ('el_price_dol/MWh_KIK2201',   'mean'),
        el_price_dol_MWh_OTA2201   = ('el_price_dol/MWh_OTA2201',   'mean'),
        el_price_dol_MWh_RDF2201   = ('el_price_dol/MWh_RDF2201',   'mean'),
        el_price_dol_MWh_SFD2201   = ('el_price_dol/MWh_SFD2201',   'mean'),
        el_price_dol_MWh_WKM2201   = ('el_price_dol/MWh_WKM2201',   'mean'),
    )
    .reset_index(drop=True)
    .sort_values('datetime_utc12')
    .reset_index(drop=True)
    )

    df_hourly.to_csv(save_path, index=False)
    print(f"✅ wholesale price has been preprocessed and saved under {save_path} ✅")

    return df_hourly


def preprocess_demand_per_zone(df):

    save_path = 'data_output/demand_preprocessed.csv'

    df['hour_bucket'] = (
    df['datetime'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly_demand = (
    df
    .groupby('hour_bucket')
    .agg(
        datetime_utc12  = ('datetime',        'max'),
        demand_GWh_CNI  = ('demand_GWh_CNI',  'sum'),
        demand_GWh_LNI  = ('demand_GWh_LNI',  'sum'),
        demand_GWh_LSI  = ('demand_GWh_LSI',  'sum'),
        demand_GWh_UNI  = ('demand_GWh_UNI',  'sum'),
        demand_GWh_USI  = ('demand_GWh_USI',  'sum'),
    )
    .reset_index(drop=True)
    .sort_values('datetime_utc12')
    .reset_index(drop=True)
    )

    df_hourly_demand.to_csv(save_path, index=False)
    print(f"✅ demand has been preprocessed and saved under {save_path} ✅")

    return df_hourly_demand

def preprocess_hvdc(df):

    save_path = 'data_output/hvdc_preprocessed.csv'

    df['hour_bucket'] = (
    df['datetime'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly_hvdc = (
    df
    .groupby('hour_bucket')
    .agg(
        datetime_utc12      = ('datetime',      'max'),
        avg_flow_MW         = ('avg_flow_MW',   'mean'),
        peak_flow_MW        = ('peak_flow_MW',  lambda x: x.loc[x.abs().idxmax()]),
    )
    .reset_index(drop=True)
    .sort_values('datetime_utc12')
    .reset_index(drop=True)
    )

    # Derive direction from sign of avg_flow
    df_hourly_hvdc['Direction'] = df_hourly_hvdc['avg_flow_MW'].map(
    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    df_hourly_hvdc.to_csv(save_path, index=False)
    print(f"✅ hvdc has been preprocessed and saved under {save_path} ✅")

    return df_hourly_hvdc

def preprocess_outages(df):
    save_path = 'data_output/outages_preprocessed.csv'

    df['hour_bucket'] = (
    df['Timestamp'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly_outages = (
    df
    .groupby('hour_bucket')
    .agg(
        datetime_utc12 = ('Timestamp', 'max'),
        outage_Gas_MW          = ('NZ_G',      'mean'),
        outage_Hyd_MW           = ('NZ_H',      'mean'),
        outage_Ter_MW        = ('NZ_T',      'mean'),
        outage_Win_MW           = ('NZ_W',      'mean'),
        outage_UNKN_MW           = ('UNKN',      'mean'),
    )
    .reset_index(drop=True)
    .sort_values('datetime_utc12')
    .reset_index(drop=True)
    )

    df_hourly_outages.to_csv(save_path, index=False)
    print(f"✅ outages has been preprocessed and saved under {save_path} ✅")

    return df_hourly_outages
