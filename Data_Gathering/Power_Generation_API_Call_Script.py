"""
fetch_generation_data.py

Downloads NZ electricity generation output data from the Electricity Authority's
public Azure blob storage and combines monthly CSV files into a single dataset.

Source: https://www.ea.govt.nz/data-and-insights/datasets/wholesale/generation/generation-output/
"""

import pandas as pd
from pathlib import Path


def fetch_generation_data(
    start=(2014, 1),
    end=(2026, 4),
    output_path="data/generation_output_merged.csv",
):
    """Download NZ generation output CSVs for a date range and merge them.

    Parameters
    ----------
    start : tuple of (year, month)
        First month to include (inclusive). Earliest available is (1997, 8).
    end : tuple of (year, month)
        Last month to include (inclusive).
    output_path : str or Path
        Where to save the merged CSV.

    Returns
    -------
    pd.DataFrame
        All monthly files concatenated into a single DataFrame.
    """
    start_year, start_month = start
    end_year, end_month = end

    # Build the list of YYYYMM strings for the date range
    years_and_months = []

    for i in range(start_year, end_year + 1):
        for y in range(1, 13):
            if i == start_year and y < start_month:
                continue
            if i == end_year and y > end_month:
                break
            if y < 10:
                years_and_months.append(f"{i}0{y}")
            else:
                years_and_months.append(f"{i}{y}")

    print(f"Downloading {len(years_and_months)} files...")

    # Make the call for each file
    base_url = "https://emidatasets.blob.core.windows.net/publicdata/Datasets/Wholesale/Generation/Generation_MD/"
    end_url = "_Generation_MD.csv"

    df_list = []

    for date in years_and_months:
        url = f"{base_url}{date}{end_url}"
        df_date = pd.read_csv(url)
        df_list.append(df_date)

    # Compile all files into a single master DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"Merged: {len(merged_df):,} rows x {len(merged_df.columns)} columns")

    # Save to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return merged_df


if __name__ == "__main__":
    fetch_generation_data(
        start=(2014, 1),
        end=(2026, 4),
        output_path="data/generation_output_merged.csv",
    )
