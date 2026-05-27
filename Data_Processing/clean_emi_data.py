# Project Duckstradamus
# David Giger

#27.05.2026

import pandas as pd

def clean_wholesale_price():
    """unstacks the wholeprice data (the nodes) and converts to utc+12"""

    save_path="data_output/wholesale_price_utc12.csv"

    #### LOAD DATA ####
    wholesale_price = pd.read_csv("data_input/wholesale_prices.csv")

    #Rename column name and drop irrelevant columns (by not including)
    wholesale_price = wholesale_price[['Period start', 'Region ID', 'Price ($/MWh)']].rename(columns={
    'Period start': 'datetime',
    'Region ID': 'region_id',
    'Price ($/MWh)': 'price_dol/MWh'
    })

    #convert object to datetime
    wholesale_price['datetime'] = pd.to_datetime(wholesale_price['datetime'], dayfirst=True)

    #unstack nodes, so each one has a own column
    wholesale_price = wholesale_price.pivot(index='datetime', columns='region_id', values='price_dol/MWh')
    wholesale_price.columns = [f'el_price_dol/MWh_{col}' for col in wholesale_price.columns]

    # convert into utc12 timezone (remove daylight saving)
    # Step 1: extract the sub-30min rows FROM ORIGINAL
    extra_hour_data = wholesale_price[wholesale_price.index.minute.isin([40, 50])].copy()

    # Step 2: drop them from original before we work on it
    wholesale_price_clean = wholesale_price[~wholesale_price.index.minute.isin([40, 50])].copy()

    # Step 3: convert clean original to fixed UTC+12
    wholesale_price_utc12 = wholesale_price_clean.copy()
    wholesale_price_utc12.index = (
        pd.to_datetime(wholesale_price_utc12.index)
        .tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='NaT')
        .tz_convert('Etc/GMT-12')
        .tz_localize(None)
    )

    # Step 4: drop NaT rows
    wholesale_price_utc12 = wholesale_price_utc12[wholesale_price_utc12.index.notna()]

    # Step 5: remap 02:40 → 02:00, 02:50 → 02:30
    extra_hour_data.index = extra_hour_data.index.map(
        lambda ts: ts.replace(minute=0 if ts.minute == 40 else 30)
    )

    # Step 6: reinsert and sort
    wholesale_price_utc12 = pd.concat([wholesale_price_utc12, extra_hour_data]).sort_index()

    # Step 7: verify
    print(wholesale_price_utc12[wholesale_price_utc12.index.duplicated(keep=False)])


    #reset index so datetime is a column
    wholesale_price_utc12.reset_index(inplace=True)


    # save under path_save
    wholesale_price_utc12.to_csv(save_path, index=False)
    print(f"✅ wholesale_price has been cleaned and saved under {save_path} ✅")

    return wholesale_price_utc12



def clean_demand_per_zone():
    """unstacks the wholeprice data (the nodes) and converts to utc+12"""

    save_path="data_output/demand_utc12.csv"

    #### LOAD DATA ####
    demand = pd.read_csv("data_input/demand_by_zone.csv")

    #Rename column name and drop irrelevant columns (by not including)
    demand = demand[['Period start', 'Region ID', 'Demand (GWh)']].rename(columns={
    'Period start': 'datetime',
    'Region ID': 'region_id',
    'Demand (GWh)': 'demand_GWh'
    })

    #convert object to datetime
    demand['datetime'] = pd.to_datetime(demand['datetime'], dayfirst=True)

    #unstack nodes, so each one has a own column
    demand = demand.pivot(index='datetime', columns='region_id', values='demand_GWh')
    demand.columns = [f'demand_GWh_{col}' for col in demand.columns]


    # convert into utc12 timezone (remove daylight saving)
    # Step 1: extract the sub-30min rows FROM ORIGINAL
    extra_hour_data = demand[demand.index.minute.isin([40, 50])].copy()

    # Step 2: drop them from original before we work on it
    demand_clean = demand[~demand.index.minute.isin([40, 50])].copy()

    # Step 3: convert clean original to fixed UTC+12
    demand_utc12 = demand_clean.copy()
    demand_utc12.index = (
        pd.to_datetime(demand_utc12.index)
        .tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='NaT')
        .tz_convert('Etc/GMT-12')
        .tz_localize(None)
    )

    # Step 4: drop NaT rows
    demand_utc12 = demand_utc12[demand_utc12.index.notna()]

    # Step 5: remap 02:40 → 02:00, 02:50 → 02:30
    extra_hour_data.index = extra_hour_data.index.map(
        lambda ts: ts.replace(minute=0 if ts.minute == 40 else 30)
    )

    # Step 6: reinsert and sort
    demand_utc12 = pd.concat([demand_utc12, extra_hour_data]).sort_index()

    # Step 7: verify
    print(demand_utc12[demand_utc12.index.duplicated(keep=False)])

    #reset index so datetime is a column
    demand_utc12.reset_index(inplace=True)

    # save under path_save
    demand_utc12.to_csv(save_path, index=False)
    print(f"✅ demand has been cleaned and saved under {save_path} ✅")


    return demand_utc12

clean_demand_per_zone()
