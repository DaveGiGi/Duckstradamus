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

def preprocess_holiday_data(file_path):

    # Load dataset
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Convert date to datetime
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Drop missing values
    df.dropna(inplace=True)

    # Convert daily data to hourly data
    df_hourly = (
        df
        .set_index("date")
        .resample("h")
        .ffill()
        .reset_index()
    )

    # Rename date to datetime
    df_hourly = df_hourly.rename(columns={"date": "datetime"})

    # Sort and reset index
    df_hourly = df_hourly.sort_values("datetime")
    df_hourly.reset_index(drop=True, inplace=True)

    return df_hourly
