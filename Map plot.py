from pathlib import Path

import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------
# File and folder settings
# ---------------------------------------------------------------------
# INPUT_FILE:
# This is the station-by-year summary CSV.
# It contains one row per station per year, including average chlorophyll values.
INPUT_FILE = Path("output/north_sea_cphl_station_year_mean.csv")

# OUTPUT_DIR:
# Folder where the GeoPackage output file will be saved.
OUTPUT_DIR = Path("output")

# ---------------------------------------------------------------------
# Export settings
# ---------------------------------------------------------------------
# YEAR_TO_EXPORT:
# Select which single year should be exported to GIS format.
# Only records from this year will be kept.
YEAR_TO_EXPORT = 2023

# ---------------------------------------------------------------------
# Geographic extent: North Sea bounding box
# ---------------------------------------------------------------------
# This extent is used as a final safety filter so only points inside the
# expected North Sea region are written to the GIS output.
MIN_LON = -5.5
MAX_LON = 10.5
MIN_LAT = 50.5
MAX_LAT = 62.0


def main():
    """
    Export annual chlorophyll station points to a GeoPackage file.

    What this script does:
    1. Reads the station-by-year chlorophyll summary CSV.
    2. Checks that all required columns are present.
    3. Keeps only rows for a chosen year.
    4. Applies a North Sea geographic extent filter.
    5. Converts the table into a GeoDataFrame with point geometry.
    6. Saves the result as a GeoPackage (.gpkg) for use in GIS software
       such as QGIS.

    The output coordinate reference system is EPSG:4326
    (WGS84 longitude/latitude).
    """
    # Create the output directory if it does not already exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the station-year summary table from CSV.
    df = pd.read_csv(INPUT_FILE)

    # -----------------------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------------------
    # These are the columns needed to create the GIS point layer and
    # preserve the chlorophyll summary information.
    required_cols = {
        "STATN",
        "Latitude",
        "Longitude",
        "year",
        "cphl_mean_ug_l",
        "cphl_median_ug_l",
        "n_obs",
    }

    # Check whether any required columns are missing.
    # If so, stop immediately with a clear error message.
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # -----------------------------------------------------------------
    # Filter to one target year
    # -----------------------------------------------------------------
    # Keep only rows for the selected export year.
    df = df[df["year"] == YEAR_TO_EXPORT].copy()

    # Stop if there is no data for that year.
    if df.empty:
        raise ValueError(f"No rows found for year {YEAR_TO_EXPORT}")

    # -----------------------------------------------------------------
    # Apply geographic extent filter
    # -----------------------------------------------------------------
    # Keep only stations whose coordinates fall inside the North Sea
    # bounding box. This helps remove accidental out-of-region points.
    df = df[
        df["Longitude"].between(MIN_LON, MAX_LON)
        & df["Latitude"].between(MIN_LAT, MAX_LAT)
    ].copy()

    # Stop if no rows remain after filtering.
    if df.empty:
        raise ValueError("No rows left after extent filter")

    # -----------------------------------------------------------------
    # Convert the tabular data into geospatial point data
    # -----------------------------------------------------------------
    # points_from_xy creates point geometries from longitude (x) and
    # latitude (y). The CRS is set to EPSG:4326, which is the standard
    # WGS84 geographic coordinate system used by GPS and many web maps.
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )

    # -----------------------------------------------------------------
    # Write output as a GeoPackage
    # -----------------------------------------------------------------
    # A GeoPackage (.gpkg) is a GIS-friendly file format supported well
    # by QGIS and other spatial software. It can store both attributes
    # and geometry in one file.
    gpkg_file = OUTPUT_DIR / f"north_sea_cphl_points_{YEAR_TO_EXPORT}.gpkg"
    gdf.to_file(gpkg_file, layer="cphl_points", driver="GPKG")

    # Print a short summary so the user knows where the file was saved
    # and what coordinate reference system it uses.
    print("Saved:")
    print(gpkg_file)
    print("CRS:", gdf.crs)


if __name__ == "__main__":
    main()