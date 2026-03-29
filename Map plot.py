from pathlib import Path

import pandas as pd
import geopandas as gpd

INPUT_FILE = Path("output/north_sea_cphl_station_year_mean.csv")
OUTPUT_DIR = Path("output")

YEAR_TO_EXPORT = 2023

MIN_LON = -5.5
MAX_LON = 10.5
MIN_LAT = 50.5
MAX_LAT = 62.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_cols = {
        "STATN",
        "Latitude",
        "Longitude",
        "year",
        "cphl_mean_ug_l",
        "cphl_median_ug_l",
        "n_obs",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["year"] == YEAR_TO_EXPORT].copy()

    if df.empty:
        raise ValueError(f"No rows found for year {YEAR_TO_EXPORT}")

    df = df[
        df["Longitude"].between(MIN_LON, MAX_LON)
        & df["Latitude"].between(MIN_LAT, MAX_LAT)
    ].copy()

    if df.empty:
        raise ValueError("No rows left after extent filter")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )

    gpkg_file = OUTPUT_DIR / f"north_sea_cphl_points_{YEAR_TO_EXPORT}.gpkg"
    gdf.to_file(gpkg_file, layer="cphl_points", driver="GPKG")

    print("Saved:")
    print(gpkg_file)
    print("CRS:", gdf.crs)


if __name__ == "__main__":
    main()