from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = "outputs_feb/cv_predictions_xgboost.csv"
OUTPUT_BASENAME = "xgboost_semivariogram_feb"

OUTDIR = "Images"
LON_COL = "Longitude"
LAT_COL = "Latitude"
VALUE_COL = "Value"
PRED_COL = "xgboost_pred"

N_LAGS = 15
MAX_DIST_KM = None
SAMPLE_PAIRS = 200000
RANDOM_STATE = 42
PLOT_TITLE = "Residual semivariogram"


def haversine_km(lon1, lat1, lon2, lat2):
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
    work = df[[lon_col, lat_col, value_col, pred_col]].dropna().copy()
    if len(work) < 3:
        raise ValueError("Need at least 3 complete rows to compute a semivariogram.")

    work["residual"] = work[value_col] - work[pred_col]

    lon = work[lon_col].to_numpy(dtype=float)
    lat = work[lat_col].to_numpy(dtype=float)
    resid = work["residual"].to_numpy(dtype=float)
    n = len(work)

    iu = np.triu_indices(n, k=1)
    pair_count = len(iu[0])

    if pair_count == 0:
        raise ValueError("No point pairs available.")

    if sample_pairs is not None and sample_pairs < pair_count:
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(pair_count, size=sample_pairs, replace=False)
        i_idx = iu[0][chosen]
        j_idx = iu[1][chosen]
    else:
        i_idx, j_idx = iu

    dists = haversine_km(lon[i_idx], lat[i_idx], lon[j_idx], lat[j_idx])
    semiv = 0.5 * (resid[i_idx] - resid[j_idx]) ** 2

    if max_dist_km is None:
        max_dist_km = float(np.quantile(dists, 0.95))
    if max_dist_km <= 0:
        raise ValueError("MAX_DIST_KM must be positive.")

    mask = dists <= max_dist_km
    dists = dists[mask]
    semiv = semiv[mask]

    if len(dists) == 0:
        raise ValueError("No point pairs remain after distance filtering.")

    bins = np.linspace(0, max_dist_km, n_lags + 1)
    bin_ids = np.digitize(dists, bins, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_lags - 1)

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
                "dist_mid_km": float(np.mean(dists[m])),
                "n_pairs": count,
                "semivariance": float(np.mean(semiv[m])),
            }
        )

    return pd.DataFrame(rows)


def plot_semivariogram(semivar_df: pd.DataFrame, title: str, out_png: Path) -> None:
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
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

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

    semivar_df["pred_col"] = PRED_COL
    semivar_df["residual_definition"] = f"{VALUE_COL} - {PRED_COL}"

    csv_out = outdir / f"{OUTPUT_BASENAME}.csv"
    png_out = outdir / f"{OUTPUT_BASENAME}.png"
    summary_out = outdir / f"{OUTPUT_BASENAME}_summary.json"

    semivar_df.to_csv(csv_out, index=False)
    plot_semivariogram(semivar_df, f"{PLOT_TITLE} ({PRED_COL})", png_out)

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
