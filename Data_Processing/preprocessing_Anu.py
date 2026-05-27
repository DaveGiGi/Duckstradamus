import pandas as pd

# ==========================================
# LAKE STORAGE DATA PREPROCESSING
# ==========================================
def preprocess_lake_storage(file_path):

    # Load dataset
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    # Create datetime column
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        errors="coerce"
    )

    # Keep useful columns
    df = df[
        ["datetime", "lake_level_m", "active_storage_mm³"]
    ]

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Sort values
    df = df.sort_values("datetime")

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


# ==========================================
# SOLAR DATA PREPROCESSING
# ==========================================

def preprocess_solar_data(file_path):

    # Load dataset
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    # Convert time column to datetime
    df["datetime"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    # Drop old time column
    df.drop(columns=["time"], inplace=True)

    # Move datetime to first column
    df = df[
        ["datetime"] + [col for col in df.columns if col != "datetime"]
    ]

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Sort values
    df = df.sort_values("datetime")

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df



# ==========================================
# TEMPERATURE DATA PREPROCESSING
# ==========================================

def preprocess_temperature_data(file_path):

    # Load dataset
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    # Convert time column to datetime
    df["datetime"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    # Drop old time column
    df.drop(columns=["time"], inplace=True)

    # Move datetime to first column
    df = df[
        ["datetime"] + [col for col in df.columns if col != "datetime"]
    ]

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Sort by datetime
    df = df.sort_values("datetime")

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


# ==========================================
# WIND DATA PREPROCESSING
# ==========================================

def preprocess_wind_data(file_path):

    # Load dataset
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    # Convert time column to datetime
    df["datetime"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    # Drop old time column
    df.drop(columns=["time"], inplace=True)

    # Move datetime to first column
    df = df[
        ["datetime"] + [col for col in df.columns if col != "datetime"]
    ]

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Sort by datetime
    df = df.sort_values("datetime")

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df



# ==========================================
# HOLIDAY DATA PREPROCESSING
# ==========================================

import pandas as pd
import holidays

def create_nz_holiday_data(
    start_date="2014-01-01",
    end_date="2026-12-30",
    save_path=None
    ):
    import holidays
    # Create hourly datetime range
    df = pd.DataFrame({
        "datetime": pd.date_range(
            start=start_date,
            end=end_date,
            freq="h"
        )
    })

    # New Zealand official holidays
    nz_holidays = holidays.NewZealand(
        years=range(
            pd.to_datetime(start_date).year,
            pd.to_datetime(end_date).year + 1
        )
    )

    # Official holidays
    is_holiday = df["datetime"].dt.date.isin(nz_holidays)

    # Weekends
    is_weekend = df["datetime"].dt.weekday >= 5

    # Combine both into binary holiday column
    df["holiday"] = (
        is_holiday | is_weekend
    ).astype(int)

    # Save only if path is given
    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df




# ==========================================
# make_lake_hourly data and interpolate the other columns
# ==========================================

import pandas as pd

def make_lake_hourly(
    file_path,
    start_date=None,
    end_date=None,
    save=True
):

    # Load dataset
    df = pd.read_csv(file_path)

    # Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Convert 23:59:59 -> next day 00:00:00
    mask = (
        df["datetime"].dt.time
        == pd.to_datetime("23:59:59").time()
    )

    df.loc[mask, "datetime"] = (
        df.loc[mask, "datetime"]
        + pd.Timedelta(seconds=1)
    )

    # Sort datetime
    df = df.sort_values("datetime")

    # Optional date filtering
    if start_date is not None:
        df = df[
            df["datetime"] >= pd.to_datetime(start_date)
        ]

    if end_date is not None:
        df = df[
            df["datetime"] <= pd.to_datetime(end_date)
        ]

    # Set datetime index
    df = df.set_index("datetime")

    # Create hourly timestamps
    hourly_df = df.resample("h").asfreq()

    # Improve datatype handling
    hourly_df = hourly_df.infer_objects(copy=False)

    # Select numeric columns
    numeric_cols = (
        hourly_df
        .select_dtypes(include="number")
        .columns
    )

    # Time interpolation
    hourly_df[numeric_cols] = (
        hourly_df[numeric_cols]
        .interpolate(method="time")
    )

    # Reset index
    hourly_df = hourly_df.reset_index()

    # Save file if requested
    if save:
        hourly_df.to_csv(file_path, index=False)

    return hourly_df
