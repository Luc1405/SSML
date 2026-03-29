from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, mapping
from shapely.ops import transform
from tqdm import tqdm

try:
    from sentinelhub import CRS, DataCollection, Geometry, SHConfig, SentinelHubStatistical
except ImportError as exc:
    raise SystemExit(
        "sentinelhub is not installed. Run:\n"
        "pip install sentinelhub"
    ) from exc

DEFAULT_INPUT_CSV = r"output/north_sea_cphl_cleaned.csv"
DEFAULT_OUTPUT_DIR = r"data/sentinel_chunks"
DEFAULT_FINAL_OUTPUT_CSV = r"data/sentinel_points_combined.csv"

DEFAULT_START_DATE = "2016-04-01"
DEFAULT_BUFFER_M = 1200
DEFAULT_DAY_WINDOW = 7
DEFAULT_MAX_CLOUD = 100
DEFAULT_MOSAICKING_ORDER = "mostRecent"
DEFAULT_MAX_WORKERS = 1

CHUNK_BY = "month"         # "year", "month", or "rows"
ROWS_PER_CHUNK = 200
SLEEP_BETWEEN_CHUNKS = 30
REQUEST_SLEEP_SECONDS = 2.0
SKIP_EXISTING_CHUNKS = True

DEFAULT_FEATURES = [
    "CHL_OC4ME",
    "CHL_NN",
    "TSM_NN",
    "KD490_M07",
    "ADG443_NN",
    "B03",
    "B04",
    "B06",
    "B07",
    "B08",
    "B10",
    "B11",
    "B17",
]

TEST_MODE = False
TEST_SAMPLE_AFTER = "2018-01-01"
TEST_N_ROWS = 10
TEST_RANDOM_STATE = 42
TEST_STATION = None
PRINT_RAW_RESPONSE = False

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"

S3_OLCI_L2 = DataCollection.define(
    name="SENTINEL3_OLCI_L2_CUSTOM",
    api_id="sentinel-3-olci-l2",
    catalog_id="sentinel-3-olci-l2",
)

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"
_TO_3857 = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True).transform

@dataclass(frozen=True)
class QueryKey:
    date_str: str
    lat: float
    lon: float
    buffer_m: int
    day_window: int
    max_cloud: int
    mosaicking_order: str
    features_signature: str


def safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "nan":
            return None
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def make_buffer_geometry(lon: float, lat: float, buffer_m: int) -> dict:
    point_wgs84 = Point(lon, lat)
    point_3857 = transform(_TO_3857, point_wgs84)
    buffered_3857 = point_3857.buffer(buffer_m)
    return mapping(buffered_3857)


def build_evalscript(features: List[str]) -> str:
    bands_csv = ", ".join(f'"{b}"' for b in features)

    return f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: [{bands_csv}, "dataMask"]
    }}],
    output: [
      {{
        id: "features",
        bands: {len(features)},
        sampleType: "FLOAT32"
      }},
      {{
        id: "dataMask",
        bands: 1,
        sampleType: "UINT8"
      }}
    ]
  }};
}}

function evaluatePixel(sample) {{
  return {{
    features: [{", ".join(f"sample.{b}" for b in features)}],
    dataMask: [sample.dataMask]
  }};
}}
""".strip()


def parse_interval_stat(
    stat: dict,
    features: List[str],
    sample_datetime: pd.Timestamp,
) -> Optional[dict]:
    try:
        interval_from = pd.to_datetime(stat["interval"]["from"], utc=True)
        interval_to = pd.to_datetime(stat["interval"]["to"], utc=True)
        outputs = stat["outputs"]
        features_out = outputs["features"]["bands"]
    except Exception:
        return None

    mask_stats = {}
    try:
        data_mask_bands = outputs.get("dataMask", {}).get("bands", {})
        if data_mask_bands:
            first_mask_band = next(iter(data_mask_bands))
            mask_stats = data_mask_bands[first_mask_band].get("stats", {})
    except Exception:
        mask_stats = {}

    if not mask_stats:
        mask_stats = features_out.get("B0", {}).get("stats", {})

    sample_count = mask_stats.get("sampleCount")
    no_data_count = mask_stats.get("noDataCount")

    result: Dict[str, Any] = {
        "sat_interval_from": interval_from.isoformat(),
        "sat_interval_to": interval_to.isoformat(),
        "sat_interval_mid_utc": (interval_from + (interval_to - interval_from) / 2).isoformat(),
        "sampleCount": sample_count,
        "noDataCount": no_data_count,
    }

    if sample_count is not None and no_data_count is not None:
        result["valid_pixel_count"] = max(int(sample_count - no_data_count), 0)
    else:
        result["valid_pixel_count"] = None

    has_any_numeric_value = False

    for idx, feat in enumerate(features):
        band_key = f"B{idx}"
        band_stats = features_out.get(band_key, {}).get("stats", {})

        mean_val = safe_number(band_stats.get("mean"))
        stdev_val = safe_number(band_stats.get("stDev"))
        min_val = safe_number(band_stats.get("min"))
        max_val = safe_number(band_stats.get("max"))

        result[f"s3_{feat}_mean"] = mean_val
        result[f"s3_{feat}_stDev"] = stdev_val
        result[f"s3_{feat}_min"] = min_val
        result[f"s3_{feat}_max"] = max_val

        if mean_val is not None:
            has_any_numeric_value = True

    sample_utc = (
        sample_datetime.tz_convert("UTC")
        if sample_datetime.tzinfo is not None
        else sample_datetime.tz_localize("UTC")
    )
    mid = pd.to_datetime(result["sat_interval_mid_utc"], utc=True)
    result["sat_time_diff_hours"] = abs((mid - sample_utc).total_seconds()) / 3600.0
    result["has_any_numeric_value"] = has_any_numeric_value

    return result


def empty_result(features: List[str], status: str, error: Optional[str], valid_pixel_count: Optional[int]) -> dict:
    return {
        "query_status": status,
        "query_error": error,
        "sat_interval_from": None,
        "sat_interval_to": None,
        "sat_interval_mid_utc": None,
        "sat_time_diff_hours": None,
        "sampleCount": None,
        "noDataCount": None,
        "valid_pixel_count": valid_pixel_count,
        "has_any_numeric_value": None,
        **{
            f"s3_{feat}_{suffix}": None
            for feat in features
            for suffix in ["mean", "stDev", "min", "max"]
        },
    }


def choose_best_interval(intervals: List[dict]) -> Optional[dict]:
    valid = [
        x for x in intervals
        if x is not None
        and x.get("valid_pixel_count") is not None
        and x.get("valid_pixel_count", 0) > 0
        and x.get("has_any_numeric_value") is True
    ]
    if valid:
        valid.sort(key=lambda x: x["sat_time_diff_hours"])
        return valid[0]

    fallback = [x for x in intervals if x is not None]
    if fallback:
        fallback.sort(key=lambda x: x["sat_time_diff_hours"])
        return fallback[0]

    return None


def build_request(
    geometry_geojson: dict,
    date_value: pd.Timestamp,
    day_window: int,
    max_cloud: int,
    mosaicking_order: str,
    features: List[str],
    config: SHConfig,
) -> SentinelHubStatistical:
    start = (date_value - pd.Timedelta(days=day_window)).strftime("%Y-%m-%d")
    end = (date_value + pd.Timedelta(days=day_window + 1)).strftime("%Y-%m-%d")

    aggregation = SentinelHubStatistical.aggregation(
        evalscript=build_evalscript(features),
        time_interval=(start, end),
        aggregation_interval="P1D",
        resolution=(300, 300),
    )

    input_data = SentinelHubStatistical.input_data(
        data_collection=S3_OLCI_L2,
        maxcc=max_cloud / 100.0,
        mosaicking_order=mosaicking_order,
    )

    return SentinelHubStatistical(
        aggregation=aggregation,
        input_data=[input_data],
        geometry=Geometry(geometry_geojson, crs=CRS.POP_WEB),
        config=config,
    )


def fetch_one(
    key: QueryKey,
    sample_datetime: pd.Timestamp,
    geometry_geojson: dict,
    features: List[str],
    config: SHConfig,
    max_retries: int = 5,
) -> dict:
    request = build_request(
        geometry_geojson=geometry_geojson,
        date_value=sample_datetime,
        day_window=key.day_window,
        max_cloud=key.max_cloud,
        mosaicking_order=key.mosaicking_order,
        features=features,
        config=config,
    )

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(REQUEST_SLEEP_SECONDS)
            response = request.get_data()[0]

            if PRINT_RAW_RESPONSE:
                print("\n--- RAW RESPONSE START ---")
                print(response)
                print("--- RAW RESPONSE END ---\n")

            data = response.get("data", [])
            parsed = [parse_interval_stat(x, features, sample_datetime) for x in data]
            parsed = [x for x in parsed if x is not None]

            if not parsed:
                return empty_result(
                    features=features,
                    status="no_intervals_returned",
                    error=f"Raw intervals returned: {len(data)}, but none could be parsed.",
                    valid_pixel_count=None,
                )

            best = choose_best_interval(parsed)

            if best is None:
                return empty_result(
                    features=features,
                    status="no_intervals_returned",
                    error="Intervals parsed, but no best interval could be selected.",
                    valid_pixel_count=None,
                )

            if best.get("valid_pixel_count") is None:
                best["query_status"] = "interval_missing_counts"
                best["query_error"] = None
                return best

            if best.get("valid_pixel_count", 0) <= 0:
                best["query_status"] = "no_valid_pixels"
                best["query_error"] = None
                return best

            if not best.get("has_any_numeric_value", False):
                best["query_status"] = "interval_all_nan"
                best["query_error"] = None
                return best

            best["query_status"] = "ok"
            best["query_error"] = None
            return best

        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(120, 10 * attempt))
            else:
                return empty_result(
                    features=features,
                    status="error",
                    error=last_error,
                    valid_pixel_count=None,
                )

    return empty_result(features=features, status="error", error=last_error, valid_pixel_count=None)


def ensure_credentials(config: SHConfig) -> None:
    client_id = os.getenv("SH_CLIENT_ID") or config.sh_client_id
    client_secret = os.getenv("SH_CLIENT_SECRET") or config.sh_client_secret

    if not client_id or not client_secret:
        raise SystemExit(
            "Missing credentials.\n"
            "Set SH_CLIENT_ID and SH_CLIENT_SECRET in your environment first."
        )

    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_token_url = CDSE_TOKEN_URL
    config.sh_base_url = CDSE_BASE_URL


def prepare_unique_queries(
    df: pd.DataFrame,
    buffer_m: int,
    day_window: int,
    max_cloud: int,
    mosaicking_order: str,
    features: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()

    work["__query_date"] = pd.to_datetime(work["DATE"], errors="coerce").dt.floor("D")
    work["__lat_round"] = work["Latitude"].round(6)
    work["__lon_round"] = work["Longitude"].round(6)

    features_signature = hashlib.md5(",".join(features).encode("utf-8")).hexdigest()[:12]

    work["__query_key"] = work.apply(
        lambda r: QueryKey(
            date_str=r["__query_date"].strftime("%Y-%m-%d"),
            lat=float(r["__lat_round"]),
            lon=float(r["__lon_round"]),
            buffer_m=buffer_m,
            day_window=day_window,
            max_cloud=max_cloud,
            mosaicking_order=mosaicking_order,
            features_signature=features_signature,
        ),
        axis=1,
    )

    unique_queries = (
        work[["__query_key", "__query_date", "__lat_round", "__lon_round"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return work, unique_queries


def run_queries(
    unique_queries: pd.DataFrame,
    features: List[str],
    config: SHConfig,
    buffer_m: int,
    max_workers: int,
) -> Dict[QueryKey, dict]:
    results: Dict[QueryKey, dict] = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in unique_queries.iterrows():
            key = row["__query_key"]
            geom = make_buffer_geometry(
                lon=float(row["__lon_round"]),
                lat=float(row["__lat_round"]),
                buffer_m=buffer_m,
            )
            fut = executor.submit(
                fetch_one,
                key=key,
                sample_datetime=row["__query_date"],
                geometry_geojson=geom,
                features=features,
                config=config,
            )
            futures[fut] = key

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Querying Sentinel-3"):
            key = futures[fut]
            results[key] = fut.result()

    return results


def build_chunks(df: pd.DataFrame) -> List[tuple[str, pd.DataFrame]]:
    if CHUNK_BY == "year":
        chunks = []
        years = sorted(df["DATE"].dt.year.dropna().astype(int).unique().tolist())
        for year in years:
            part = df[df["DATE"].dt.year == year].copy()
            if not part.empty:
                chunks.append((f"year_{year}", part))
        return chunks

    if CHUNK_BY == "month":
        chunks = []
        df = df.copy()
        df["__chunk_month"] = df["DATE"].dt.to_period("M").astype(str)
        months = sorted(df["__chunk_month"].dropna().unique().tolist())
        for month in months:
            part = df[df["__chunk_month"] == month].copy()
            part.drop(columns=["__chunk_month"], inplace=True, errors="ignore")
            safe_name = month.replace("-", "_")
            if not part.empty:
                chunks.append((f"month_{safe_name}", part))
        return chunks

    if CHUNK_BY == "rows":
        chunks = []
        total = len(df)
        start = 0
        chunk_idx = 1
        while start < total:
            end = min(start + ROWS_PER_CHUNK, total)
            part = df.iloc[start:end].copy()
            chunks.append((f"rows_{chunk_idx:03d}", part))
            start = end
            chunk_idx += 1
        return chunks

    raise SystemExit("CHUNK_BY must be 'year', 'month', or 'rows'.")


def process_chunk(
    chunk_name: str,
    chunk_df: pd.DataFrame,
    args: argparse.Namespace,
    config: SHConfig,
    output_dir: Path,
) -> Path:
    work, unique_queries = prepare_unique_queries(
        df=chunk_df,
        buffer_m=args.buffer_m,
        day_window=args.day_window,
        max_cloud=args.max_cloud,
        mosaicking_order=args.mosaicking_order,
        features=args.features,
    )

    print(f"\nProcessing chunk: {chunk_name}")
    print(f"Rows in chunk: {len(work)}")
    print(f"Unique point/date queries: {len(unique_queries)}")

    query_results = run_queries(
        unique_queries=unique_queries,
        features=args.features,
        config=config,
        buffer_m=args.buffer_m,
        max_workers=args.max_workers,
    )

    sat_df = pd.DataFrame(
        [{"__query_key": key, **value} for key, value in query_results.items()]
    )

    out = work.merge(sat_df, on="__query_key", how="left")
    out.drop(columns=["__query_key", "__query_date", "__lat_round", "__lon_round"], inplace=True)

    out["s3_buffer_m"] = args.buffer_m
    out["s3_day_window"] = args.day_window
    out["s3_max_cloud"] = args.max_cloud
    out["s3_mosaicking_order"] = args.mosaicking_order
    out["s3_chunk_name"] = chunk_name

    chunk_path = output_dir / f"{chunk_name}.csv"
    out.to_csv(chunk_path, index=False)

    ok_rows = int((out["query_status"] == "ok").sum())
    err_rows = int((out["query_status"] == "error").sum())
    print(f"Finished chunk: {chunk_name}")
    print(f"Wrote: {chunk_path}")
    print(f"Matched rows: {ok_rows} | Error rows: {err_rows}")

    if err_rows > 0:
        error_examples = (
            out.loc[out["query_status"] == "error", [c for c in ["DATE", "query_error", "STATN"] if c in out.columns]]
            .drop_duplicates()
            .head(5)
        )
        print("Example errors:")
        print(error_examples.to_string(index=False))

    return chunk_path


def combine_chunk_csvs(chunk_paths: List[Path], final_output_path: Path) -> None:
    frames = []
    for path in chunk_paths:
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        raise SystemExit("No chunk CSVs were created, so nothing could be combined.")

    combined = pd.concat(frames, ignore_index=True)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(final_output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunked point-query Sentinel-3 OLCI L2 Water data for in-situ points.")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Path to input CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for chunk CSVs.")
    parser.add_argument("--final-output", default=DEFAULT_FINAL_OUTPUT_CSV, help="Combined final CSV path.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Keep only rows with DATE >= this value.")
    parser.add_argument("--buffer-m", type=int, default=DEFAULT_BUFFER_M, help="Point buffer radius in meters.")
    parser.add_argument("--day-window", type=int, default=DEFAULT_DAY_WINDOW, help="Search window in days on each side.")
    parser.add_argument("--max-cloud", type=int, default=DEFAULT_MAX_CLOUD, help="Maximum cloud coverage percentage.")
    parser.add_argument(
        "--mosaicking-order",
        default=DEFAULT_MOSAICKING_ORDER,
        choices=["mostRecent", "leastRecent", "leastCC"],
        help="Sentinel Hub mosaicking order.",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel request count.")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, help="Bands/variables to extract.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    final_output_path = Path(args.final_output)

    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_cols = {"DATE", "Latitude", "Longitude"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {sorted(missing)}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df[df["DATE"].notna()].copy()

    start_date = pd.Timestamp(args.start_date)
    df = df[df["DATE"] >= start_date].copy()
    df = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()

    if TEST_MODE:
        df = df[df["DATE"] >= TEST_SAMPLE_AFTER].copy()

        if TEST_STATION:
            df = df[df["STATN"].astype(str) == TEST_STATION].copy()

        if df.empty:
            raise SystemExit("No rows remain after TEST_MODE station/date filtering.")

        df = df.sample(min(TEST_N_ROWS, len(df)), random_state=TEST_RANDOM_STATE).copy()

    if df.empty:
        raise SystemExit("No rows remain after filtering. Nothing to query.")

    config = SHConfig()
    ensure_credentials(config)

    chunks = build_chunks(df)

    print(f"Total filtered rows: {len(df)}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk mode: {CHUNK_BY}")
    print(f"Buffer: {args.buffer_m} m | Day window: ±{args.day_window} days | Max cloud: {args.max_cloud}")
    print(f"Request sleep: {REQUEST_SLEEP_SECONDS} s | Sleep between chunks: {SLEEP_BETWEEN_CHUNKS} s")

    chunk_paths: List[Path] = []

    for idx, (chunk_name, chunk_df) in enumerate(chunks, start=1):
        chunk_path = output_dir / f"{chunk_name}.csv"

        print(f"\nChunk {idx}/{len(chunks)}: {chunk_name}")

        if SKIP_EXISTING_CHUNKS and chunk_path.exists():
            print(f"Skipping existing chunk file: {chunk_path}")
            chunk_paths.append(chunk_path)
            continue

        produced_path = process_chunk(
            chunk_name=chunk_name,
            chunk_df=chunk_df,
            args=args,
            config=config,
            output_dir=output_dir,
        )
        chunk_paths.append(produced_path)

        if idx < len(chunks) and SLEEP_BETWEEN_CHUNKS > 0:
            print(f"Sleeping {SLEEP_BETWEEN_CHUNKS} seconds before next chunk...")
            time.sleep(SLEEP_BETWEEN_CHUNKS)

    combine_chunk_csvs(chunk_paths, final_output_path)

    combined = pd.read_csv(final_output_path)
    total_rows = len(combined)
    ok_rows = int((combined["query_status"] == "ok").sum())
    no_intervals_rows = int((combined["query_status"] == "no_intervals_returned").sum())
    no_valid_rows = int((combined["query_status"] == "no_valid_pixels").sum())
    all_nan_rows = int((combined["query_status"] == "interval_all_nan").sum())
    missing_counts_rows = int((combined["query_status"] == "interval_missing_counts").sum())
    err_rows = int((combined["query_status"] == "error").sum())

    print("\nChunks complete.")
    print(f"Combined file written: {final_output_path.resolve()}")
    print(f"Rows written: {total_rows}")
    print(f"Matched rows: {ok_rows}")
    print(f"No-interval rows: {no_intervals_rows}")
    print(f"No-valid-pixel rows: {no_valid_rows}")
    print(f"All-NaN interval rows: {all_nan_rows}")
    print(f"Missing-count rows: {missing_counts_rows}")
    print(f"Error rows: {err_rows}")

    if err_rows > 0:
        error_examples = (
            combined.loc[
                combined["query_status"] == "error",
                [c for c in ["DATE", "query_error", "STATN", "s3_chunk_name"] if c in combined.columns]
            ]
            .drop_duplicates()
            .head(10)
        )
        print("\nExample errors:")
        print(error_examples.to_string(index=False))


if __name__ == "__main__":
    main()