from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/ContaminantsSeawater_2026031810085210.csv")
OUTPUT_DIR = Path("output")

MIN_LAT = 50.5
MAX_LAT = 62.0
MIN_LON = -5.5
MAX_LON = 10.5

TARGET_PARAM = "CPHL"

def is_valid_qflag(qflag: str) -> bool:
    """
    Keep rows with:
    - no QFLAG
    - or simple flags you decide to accept later

    For a first clean dataset:
    - reject rows with '<' because they are below detection / censored
    - reject 'Q' and 'D' flagged rows for simplicity
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
    Keep rows where VFLAG is:
    - missing
    - or 'A' (acceptable)

    Reject other flags for the first pass.
    """
    if pd.isna(vflag):
        return True

    vflag = str(vflag).strip().upper()
    return vflag == "A"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    df = df[
        df["Latitude"].between(MIN_LAT, MAX_LAT)
        & df["Longitude"].between(MIN_LON, MAX_LON)
    ].copy()

    print(f"Rows after North Sea filter: {len(df):,}")

    df = df[df["PARAM"] == TARGET_PARAM].copy()

    print(f"Rows after PARAM == {TARGET_PARAM}: {len(df):,}")

    df["DATE"] = pd.to_datetime(df["DATE"], format="%d/%m/%Y", errors="coerce")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["DEPHU"] = pd.to_numeric(df["DEPHU"], errors="coerce")
    df["DEPHL"] = pd.to_numeric(df["DEPHL"], errors="coerce")

    df = df.dropna(subset=["DATE", "Latitude", "Longitude", "Value", "STATN", "MUNIT"])

    df = df[df["MUNIT"].str.lower() == "ug/l"].copy()

    df = df[df["Value"] >= 0].copy()

    df = df[df["QFLAG"].apply(is_valid_qflag)].copy()
    df = df[df["VFLAG"].apply(is_valid_vflag)].copy()

    df["year"] = df["DATE"].dt.year
    df["month"] = df["DATE"].dt.month
    df["dayofyear"] = df["DATE"].dt.dayofyear

    df["sample_depth_m"] = df["DEPHU"]
    both_depths = df["DEPHU"].notna() & df["DEPHL"].notna()
    df.loc[both_depths, "sample_depth_m"] = (
        df.loc[both_depths, "DEPHU"] + df.loc[both_depths, "DEPHL"]
    ) / 2

    df = df.drop_duplicates(
        subset=["tblAnalysisID", "tblParamID", "tblSampleID"],
        keep="first",
    )

    df = df.sort_values(["DATE", "STATN"]).reset_index(drop=True)

    print(f"Rows after cleaning: {len(df):,}")
    print(f"Unique stations: {df['STATN'].nunique():,}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")

    cleaned_file = OUTPUT_DIR / "north_sea_cphl_cleaned.csv"
    df.to_csv(cleaned_file, index=False)

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

    print("\nSaved files:")
    print(f"- {cleaned_file}")
    print(f"- {station_year_file}")
    print(f"- {station_mean_file}")


if __name__ == "__main__":
    main()