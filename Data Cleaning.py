from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# File and folder settings
# ---------------------------------------------------------------------
# INPUT_FILE:
# Path to the raw ICES contaminants-in-seawater CSV file.
# This file is expected to live in a folder called "data" relative to
# where the script is run.
INPUT_FILE = Path("data/ContaminantsSeawater_2026031810085210.csv")

# OUTPUT_DIR:
# Folder where all cleaned and aggregated output CSV files will be written.
OUTPUT_DIR = Path("output")

# ---------------------------------------------------------------------
# Geographic filter: rough North Sea bounding box
# ---------------------------------------------------------------------
# Only rows with coordinates inside this latitude/longitude box are kept.
# This is a simple spatial filter to isolate observations from the North Sea.
MIN_LAT = 50.5
MAX_LAT = 62.0
MIN_LON = -5.5
MAX_LON = 10.5

# ---------------------------------------------------------------------
# Target parameter
# ---------------------------------------------------------------------
# PARAM values identify which environmental variable is being measured.
# Here we only keep CPHL, which represents chlorophyll-related measurements.
TARGET_PARAM = "CPHL"


def is_valid_qflag(qflag: str) -> bool:
    """
    Return True if the quality flag is acceptable, otherwise False.

    In this project, we exclude rows where the QFLAG suggests the value
    should not be used as a regular observation.

    Rules used here:
    - Missing QFLAG is accepted.
    - Any flag containing "<" is rejected
      (often indicates below detection limit / censored value).
    - Any flag containing "Q" is rejected
      (typically a questionable-quality indicator).
    - Any flag containing "D" is rejected
      (often indicates a problematic or doubtful record).

    Parameters
    ----------
    qflag : str
        The quality flag from the source dataset.

    Returns
    -------
    bool
        True if the row should be kept, False otherwise.
    """
    if pd.isna(qflag):
        return True
    qflag = str(qflag).strip().upper()
    if "<" in qflag:
        return False
    if "Q" in qflag:
        return False
    if "D" in qflag:
        return False
    return True


def is_valid_vflag(vflag: str) -> bool:
    """
    Return True if the validation flag is acceptable, otherwise False.

    In this script, only records with VFLAG == "A" are accepted.
    Missing VFLAG is also accepted.

    Parameters
    ----------
    vflag : str
        The validation flag from the source dataset.

    Returns
    -------
    bool
        True if the row should be kept, False otherwise.
    """
    if pd.isna(vflag):
        return True
    vflag = str(vflag).strip().upper()
    return vflag == "A"


def main():
    """
    Main cleaning and aggregation workflow.

    What this script does:
    1. Reads a subset of columns from the raw CSV.
    2. Keeps only observations inside the North Sea bounding box.
    3. Keeps only the target parameter (CPHL).
    4. Converts key columns to proper data types.
    5. Removes invalid, incomplete, or low-quality records.
    6. Computes a representative sample depth.
    7. Removes duplicate analytical records.
    8. Saves:
       - a cleaned row-level dataset
       - annual station summaries
       - overall station summaries
    """
    # Create the output folder if it does not already exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Only read the columns needed for this project.
    # This reduces memory use and makes the script faster than loading
    # the entire raw file.
    usecols = [
        "STATN",
        "Country",
        "MYEAR",
        "DATE",
        "Latitude",
        "Longitude",
        "DEPHU",
        "DEPHL",
        "MATRX",
        "PARGROUP",
        "PARAM",
        "BASIS",
        "QFLAG",
        "Value",
        "MUNIT",
        "VFLAG",
        "tblAnalysisID",
        "tblParamID",
        "tblSampleID",
    ]

    print("Reading file...")
    df = pd.read_csv(INPUT_FILE, usecols=usecols, low_memory=False)
    print(f"Rows loaded: {len(df):,}")

    # -----------------------------------------------------------------
    # Step 1: Geographic filter
    # -----------------------------------------------------------------
    # Keep only rows whose coordinates fall inside the North Sea bounding box.
    df = df[
        df["Latitude"].between(MIN_LAT, MAX_LAT)
        & df["Longitude"].between(MIN_LON, MAX_LON)
    ].copy()
    print(f"Rows after North Sea filter: {len(df):,}")

    # -----------------------------------------------------------------
    # Step 2: Parameter filter
    # -----------------------------------------------------------------
    # Keep only the target parameter (CPHL).
    df = df[df["PARAM"] == TARGET_PARAM].copy()
    print(f"Rows after PARAM == {TARGET_PARAM}: {len(df):,}")

    # -----------------------------------------------------------------
    # Step 3: Convert columns to usable data types
    # -----------------------------------------------------------------
    # DATE is stored as day/month/year text in the source file.
    # errors='coerce' turns invalid dates into NaT, which can be removed later.
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d/%m/%Y", errors="coerce")

    # Convert measurement and coordinate columns to numeric values.
    # Non-numeric entries become NaN.
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["DEPHU"] = pd.to_numeric(df["DEPHU"], errors="coerce")
    df["DEPHL"] = pd.to_numeric(df["DEPHL"], errors="coerce")

    # -----------------------------------------------------------------
    # Step 4: Remove incomplete records
    # -----------------------------------------------------------------
    # These fields are considered essential for downstream analysis.
    df = df.dropna(subset=["DATE", "Latitude", "Longitude", "Value", "STATN", "MUNIT"])

    # -----------------------------------------------------------------
    # Step 5: Keep only expected measurement units
    # -----------------------------------------------------------------
    # We only keep values reported in micrograms per liter (ug/l)
    # so all retained measurements are on the same scale.
    df = df[df["MUNIT"].str.lower() == "ug/l"].copy()

    # -----------------------------------------------------------------
    # Step 6: Remove physically invalid values
    # -----------------------------------------------------------------
    # Negative chlorophyll values are not meaningful here, so remove them.
    df = df[df["Value"] >= 0].copy()

    # -----------------------------------------------------------------
    # Step 7: Apply quality filters
    # -----------------------------------------------------------------
    df = df[df["QFLAG"].apply(is_valid_qflag)].copy()
    df = df[df["VFLAG"].apply(is_valid_vflag)].copy()

    # -----------------------------------------------------------------
    # Step 8: Create useful time features
    # -----------------------------------------------------------------
    # These extra columns are helpful for later analysis and modeling.
    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month
    df["dayofyear"] = df["DATE"].dt.dayofyear

    # -----------------------------------------------------------------
    # Step 9: Estimate sampling depth
    # -----------------------------------------------------------------
    # sample_depth_m is used as a single representative depth value.
    #
    # Logic:
    # - Start with DEPHU (upper depth).
    # - If both upper and lower depth are available, use the midpoint.
    #
    # This is useful because some observations represent a depth interval
    # instead of a single exact depth.
    df["sample_depth_m"] = df["DEPHU"]

    both_depths = df["DEPHU"].notna() & df["DEPHL"].notna()
    df.loc[both_depths, "sample_depth_m"] = (
        df.loc[both_depths, "DEPHU"] + df.loc[both_depths, "DEPHL"]
    ) / 2

    # -----------------------------------------------------------------
    # Step 10: Remove duplicate analytical records
    # -----------------------------------------------------------------
    # The combination of analysis ID, parameter ID, and sample ID is treated
    # as the unique identifier for one measurement record.
    df = df.drop_duplicates(
        subset=["tblAnalysisID", "tblParamID", "tblSampleID"],
        keep="first",
    )

    # Sort records in a predictable order for easier inspection.
    df = df.sort_values(["DATE", "STATN"]).reset_index(drop=True)

    print(f"Rows after cleaning: {len(df):,}")
    print(f"Unique stations: {df['STATN'].nunique():,}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")

    # -----------------------------------------------------------------
    # Output 1: Cleaned row-level dataset
    # -----------------------------------------------------------------
    # This file contains all retained measurements after filtering/cleaning.
    cleaned_file = OUTPUT_DIR / "north_sea_cphl_cleaned.csv"
    df.to_csv(cleaned_file, index=False)

    # -----------------------------------------------------------------
    # Output 2: Station-by-year summary
    # -----------------------------------------------------------------
    # For each station and year, compute:
    # - mean chlorophyll
    # - median chlorophyll
    # - number of observations
    station_year = (
        df.groupby(["STATN", "Latitude", "Longitude", "year"], as_index=False)
        .agg(
            cphl_mean_ug_l=("Value", "mean"),
            cphl_median_ug_l=("Value", "median"),
            n_obs=("Value", "size"),
        )
        .sort_values(["year", "STATN"])
    )

    station_year_file = OUTPUT_DIR / "north_sea_cphl_station_year_mean.csv"
    station_year.to_csv(station_year_file, index=False)

    # -----------------------------------------------------------------
    # Output 3: Overall station summary
    # -----------------------------------------------------------------
    # For each station across the full dataset, compute:
    # - overall mean chlorophyll
    # - overall median chlorophyll
    # - total number of observations
    # - first and last year observed
    station_mean = (
        df.groupby(["STATN", "Latitude", "Longitude"], as_index=False)
        .agg(
            cphl_mean_ug_l=("Value", "mean"),
            cphl_median_ug_l=("Value", "median"),
            n_obs=("Value", "size"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .sort_values("STATN")
    )

    station_mean_file = OUTPUT_DIR / "north_sea_cphl_station_mean.csv"
    station_mean.to_csv(station_mean_file, index=False)

    # Final summary for the user
    print("\nSaved files:")
    print(f"- {cleaned_file}")
    print(f"- {station_year_file}")
    print(f"- {station_mean_file}")


if __name__ == "__main__":
    main()