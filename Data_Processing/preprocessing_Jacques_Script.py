
# ==========================================
# GENERATION DATA PREPROCESSING
# ==========================================


def preprocess_generation_data(file_path):

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
