from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Input/output settings
# ---------------------------------------------------------------------
# INPUT_CSV:
# File containing observed values and model predictions.
# In this case it uses the XGBoost cross-validation prediction output.
INPUT_CSV = "outputs_feb/cv_predictions_xgboost.csv"

# OUTPUT_BASENAME:
# Base name used for the output files (CSV, PNG, JSON summary).
OUTPUT_BASENAME = "xgboost_semivariogram_feb"

# OUTDIR:
# Folder where output files will be written.
OUTDIR = "Images"

# ---------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------
# These define which columns in the input CSV contain:
# - station longitude
# - station latitude
# - observed chlorophyll value
# - predicted chlorophyll value
LON_COL = "Longitude"
LAT_COL = "Latitude"
VALUE_COL = "Value"
PRED_COL = "xgboost_pred"

# ---------------------------------------------------------------------
# Semivariogram settings
# ---------------------------------------------------------------------
# N_LAGS:
# Number of distance bins (lag classes) used in the empirical semivariogram.
N_LAGS = 15

# MAX_DIST_KM:
# Maximum distance included in the semivariogram.
# If None, the script automatically uses the 95th percentile of sampled
# pair distances, which helps avoid very long-distance pairs dominating
# the plot.
MAX_DIST_KM = None

# SAMPLE_PAIRS:
# Maximum number of point pairs to evaluate.
# This is useful because the number of all possible pairs grows very
# quickly with dataset size. Sampling keeps computation manageable.
SAMPLE_PAIRS = 200000

# RANDOM_STATE:
# Seed used for reproducible random pair sampling.
RANDOM_STATE = 42

# PLOT_TITLE:
# Base title shown on the semivariogram plot.
PLOT_TITLE = "Residual semivariogram"


def haversine_km(lon1, lat1, lon2, lat2):
    """
    Compute great-circle distance between two geographic points in kilometers.

    This uses the haversine formula, which is appropriate for longitude/
    latitude coordinates on the Earth.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : array-like or scalar
        Longitude and latitude coordinates in decimal degrees.

    Returns
    -------
    array-like or scalar
        Distance(s) in kilometers.
    """
    r = 6371.0088
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def compute_empirical_semivariogram(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    value_col: str,
    pred_col: str,
    n_lags: int,
    max_dist_km: float | None,
    sample_pairs: int | None,
    random_state: int,
) -> pd.DataFrame:
    """
    Compute an empirical semivariogram of model residuals.

    Residuals are defined here as:
        observed value - predicted value

    The semivariogram measures how similar or dissimilar residuals are as
    a function of distance between observation points.

    Interpretation:
    - Low semivariance at short distances suggests nearby residuals are similar.
    - Increasing semivariance with distance suggests spatial autocorrelation.
    - A flat semivariogram suggests little spatial structure remains.

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing coordinates, observed values, and predictions.
    lon_col : str
        Name of the longitude column.
    lat_col : str
        Name of the latitude column.
    value_col : str
        Name of the observed-value column.
    pred_col : str
        Name of the prediction column.
    n_lags : int
        Number of distance bins.
    max_dist_km : float or None
        Maximum distance to include. If None, use the 95th percentile
        of pairwise distances.
    sample_pairs : int or None
        Number of point pairs to sample. If None, use all possible pairs.
    random_state : int
        Random seed for reproducible pair sampling.

    Returns
    -------
    pd.DataFrame
        Table with one row per lag bin, including:
        - distance range
        - midpoint distance
        - number of point pairs
        - mean semivariance
    """
    # Keep only rows with complete coordinate, observed, and predicted values.
    work = df[[lon_col, lat_col, value_col, pred_col]].dropna().copy()
    if len(work) < 3:
        raise ValueError("Need at least 3 complete rows to compute a semivariogram.")

    # Residual = observation - prediction
    # This is the quantity whose spatial structure we want to evaluate.
    work["residual"] = work[value_col] - work[pred_col]

    lon = work[lon_col].to_numpy(dtype=float)
    lat = work[lat_col].to_numpy(dtype=float)
    resid = work["residual"].to_numpy(dtype=float)
    n = len(work)

    # Generate all unique point pairs using the upper triangle of the
    # pairwise matrix.
    iu = np.triu_indices(n, k=1)
    pair_count = len(iu[0])

    if pair_count == 0:
        raise ValueError("No point pairs available.")

    # If the total number of pairs is too large, randomly sample a subset
    # for efficiency.
    if sample_pairs is not None and sample_pairs < pair_count:
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(pair_count, size=sample_pairs, replace=False)
        i_idx = iu[0][chosen]
        j_idx = iu[1][chosen]
    else:
        i_idx, j_idx = iu

    # Compute geographic distance between all chosen pairs.
    dists = haversine_km(lon[i_idx], lat[i_idx], lon[j_idx], lat[j_idx])

    # Semivariance for each pair:
    # 0.5 * (difference in residuals)^2
    semiv = 0.5 * (resid[i_idx] - resid[j_idx]) ** 2

    # If no max distance is supplied, use the 95th percentile of pairwise
    # distances so extreme long-distance pairs do not dominate the result.
    if max_dist_km is None:
        max_dist_km = float(np.quantile(dists, 0.95))
    if max_dist_km <= 0:
        raise ValueError("MAX_DIST_KM must be positive.")

    # Keep only pairs within the chosen maximum distance.
    mask = dists <= max_dist_km
    dists = dists[mask]
    semiv = semiv[mask]

    if len(dists) == 0:
        raise ValueError("No point pairs remain after distance filtering.")

    # Create equally spaced distance bins from 0 to max_dist_km.
    bins = np.linspace(0, max_dist_km, n_lags + 1)

    # Assign each pairwise distance to a lag bin.
    bin_ids = np.digitize(dists, bins, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_lags - 1)

    # Summarize semivariance within each lag bin.
    rows = []
    for b in range(n_lags):
        m = bin_ids == b
        count = int(np.sum(m))
        if count == 0:
            rows.append(
                {
                    "lag_bin": b + 1,
                    "dist_from_km": float(bins[b]),
                    "dist_to_km": float(bins[b + 1]),
                    "dist_mid_km": float((bins[b] + bins[b + 1]) / 2.0),
                    "n_pairs": 0,
                    "semivariance": np.nan,
                }
            )
            continue

        rows.append(
            {
                "lag_bin": b + 1,
                "dist_from_km": float(bins[b]),
                "dist_to_km": float(bins[b + 1]),
                # Use the actual mean distance of pairs in the bin
                # instead of the simple bin midpoint.
                "dist_mid_km": float(np.mean(dists[m])),
                "n_pairs": count,
                "semivariance": float(np.mean(semiv[m])),
            }
        )

    return pd.DataFrame(rows)


def plot_semivariogram(semivar_df: pd.DataFrame, title: str, out_png: Path) -> None:
    """
    Plot and save the empirical semivariogram.

    Only lag bins containing at least one point pair are plotted.

    Parameters
    ----------
    semivar_df : pd.DataFrame
        Output of compute_empirical_semivariogram().
    title : str
        Plot title.
    out_png : Path
        Output PNG file path.
    """
    plot_df = semivar_df[semivar_df["n_pairs"] > 0].copy()

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["dist_mid_km"], plot_df["semivariance"], marker="o")
    plt.xlabel("Distance (km)")
    plt.ylabel("Semivariance")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    """
    Main workflow for computing and exporting a residual semivariogram.

    Steps:
    1. Read the input CSV containing observed and predicted values.
    2. Compute the empirical semivariogram of residuals.
    3. Save the semivariogram table to CSV.
    4. Save a PNG plot of the semivariogram.
    5. Save a JSON summary of the run settings and output paths.
    """
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read model prediction results.
    df = pd.read_csv(INPUT_CSV)

    # Compute semivariogram on residuals.
    semivar_df = compute_empirical_semivariogram(
        df=df,
        lon_col=LON_COL,
        lat_col=LAT_COL,
        value_col=VALUE_COL,
        pred_col=PRED_COL,
        n_lags=N_LAGS,
        max_dist_km=MAX_DIST_KM,
        sample_pairs=SAMPLE_PAIRS,
        random_state=RANDOM_STATE,
    )

    # Add a small amount of metadata to the semivariogram output table.
    semivar_df["pred_col"] = PRED_COL
    semivar_df["residual_definition"] = f"{VALUE_COL} - {PRED_COL}"

    csv_out = outdir / f"{OUTPUT_BASENAME}.csv"
    png_out = outdir / f"{OUTPUT_BASENAME}.png"
    summary_out = outdir / f"{OUTPUT_BASENAME}_summary.json"

    # Save semivariogram values and plot.
    semivar_df.to_csv(csv_out, index=False)
    plot_semivariogram(semivar_df, f"{PLOT_TITLE} ({PRED_COL})", png_out)

    # Save a compact JSON summary so the run settings are documented.
    summary = {
        "input_csv": INPUT_CSV,
        "prediction_column": PRED_COL,
        "rows_used": int(df[[LON_COL, LAT_COL, VALUE_COL, PRED_COL]].dropna().shape[0]),
        "n_lags": N_LAGS,
        "max_dist_km": MAX_DIST_KM,
        "sample_pairs": SAMPLE_PAIRS,
        "outputs": {
            "csv": str(csv_out),
            "png": str(png_out),
            "summary": str(summary_out),
        },
    }
    summary_out.write_text(__import__("json").dumps(summary, indent=2))

    print(f"Wrote: {csv_out}")
    print(f"Wrote: {png_out}")
    print(f"Wrote: {summary_out}")


if __name__ == "__main__":
    main()