
#python north_sea_chla_models.py --csv sentinel_points_combined.csv --season-months 4 5 6 --block-size-km 50 --outdir outputs_all_months


#Needs    pandas, numpy, scikit-learn, xgboost

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TARGET_COL = "Value"

NON_PREDICTOR_COLS = [
    "Country", "STATN", "MYEAR", "DATE", "DEPHU", "DEPHL", "MATRX", "PARGROUP",
    "PARAM", "BASIS", "QFLAG", "MUNIT", "VFLAG", "tblAnalysisID", "tblParamID",
    "tblSampleID", "sample_depth_m",
    "Value",
    "sat_interval_from", "sat_interval_to", "sat_interval_mid_utc",
    "query_error", "query_status", "s3_chunk_name", "s3_mosaicking_order", "sat_time_diff_hours",
    "year",
    "s3_CHL_OC4ME_max", "s3_CHL_OC4ME_min", "s3_CHL_NN_max", "s3_CHL_NN_min", "s3_CHL_OC4ME_mean", "s3_CHL_NN_mean"
]

PREFERRED_SENTINEL_PREFIXES = [
    "s3_"
]

OPTIONAL_FEATURE_COLS = [
]

LINEAR_REDUCED_FEATURE_CANDIDATES = [
    "s3_CHL_OC4ME_mean",
    "s3_CHL_NN_mean",
    "s3_TSM_NN_mean",
    "s3_KD490_M07_mean",
    "s3_ADG443_NN_mean",
    "s3_B06_mean",
    "s3_B07_mean",
    "s3_B08_mean",
    "s3_B11_mean",
    "doy_sin",
    "doy_cos",
]

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def haversine_km_scale(lat_deg: float) -> Tuple[float, float]:
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat_deg))
    return km_per_deg_lon, km_per_deg_lat


def make_spatial_blocks(df: pd.DataFrame, block_size_km: float = 50.0) -> pd.Series:
    lat0 = float(df["Latitude"].mean())
    km_per_deg_lon, km_per_deg_lat = haversine_km_scale(lat0)

    x_km = df["Longitude"] * km_per_deg_lon
    y_km = df["Latitude"] * km_per_deg_lat

    bx = np.floor(x_km / block_size_km).astype(int)
    by = np.floor(y_km / block_size_km).astype(int)

    return bx.astype(str) + "_" + by.astype(str)


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dayofyear" in out.columns:
        theta = 2 * np.pi * out["dayofyear"].astype(float) / 365.25
        out["doy_sin"] = np.sin(theta)
        out["doy_cos"] = np.cos(theta)
    return out


def filter_training_rows(df: pd.DataFrame, season_months: List[int] | None = None) -> pd.DataFrame:
    out = df.copy()

    if "query_status" in out.columns:
        out = out[out["query_status"].eq("ok")]
    if "has_any_numeric_value" in out.columns:
        out = out[out["has_any_numeric_value"].fillna(False)]

    out = out[out[TARGET_COL].notna()]
    out = out[out[TARGET_COL] >= 0]

    if season_months:
        out = out[out["month"].isin(season_months)]

    return out.reset_index(drop=True)


def drop_constant_numeric_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    kept = []
    for c in cols:
        if c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        nunique = df[c].nunique(dropna=True)
        if nunique > 1:
            kept.append(c)
    return kept


def drop_highly_correlated_features(
    df: pd.DataFrame,
    cols: List[str],
    threshold: float = 0.95
) -> List[str]:
    if len(cols) <= 1:
        return cols

    corr = df[cols].corr().abs()
    keep = []
    for col in cols:
        is_too_correlated = False
        for kept_col in keep:
            if pd.notna(corr.loc[col, kept_col]) and corr.loc[col, kept_col] >= threshold:
                is_too_correlated = True
                break
        if not is_too_correlated:
            keep.append(col)
    return keep


def select_tree_predictor_columns(
    df: pd.DataFrame,
    include_month: bool = False,
    include_dayofyear_raw: bool = False,
    include_latlon: bool = False,
) -> List[str]:
    cols = []

    for c in df.columns:
        if c in NON_PREDICTOR_COLS:
            continue
        if c in ["Latitude", "Longitude"] and not include_latlon:
            continue
        if c == TARGET_COL:
            continue

        if c.startswith(tuple(PREFERRED_SENTINEL_PREFIXES)):
            cols.append(c)

    for c in OPTIONAL_FEATURE_COLS:
        if c in df.columns and c not in cols:
            cols.append(c)

    for c in ["doy_sin", "doy_cos"]:
        if c in df.columns and c not in cols:
            cols.append(c)

    if include_month and "month" in df.columns:
        cols.append("month")
    if include_dayofyear_raw and "dayofyear" in df.columns:
        cols.append("dayofyear")

    if include_latlon:
        for c in ["Latitude", "Longitude"]:
            if c in df.columns:
                cols.append(c)

    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = sorted(set(cols))
    cols = drop_constant_numeric_columns(df, cols)
    return cols


def select_linear_predictor_columns(
    df: pd.DataFrame,
    include_latlon: bool = False,
) -> List[str]:
    cols = []

    for c in LINEAR_REDUCED_FEATURE_CANDIDATES:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    if include_latlon:
        for c in ["Latitude", "Longitude"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)

    cols = drop_constant_numeric_columns(df, cols)
    cols = drop_highly_correlated_features(df, cols, threshold=0.95)
    return cols

def build_linear_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])


def build_rf_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            min_samples_split=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ))
    ])


def build_xgb_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        ))
    ])

def compute_smearing_factor(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> float:
    resid = y_log_true - y_log_pred
    smear = float(np.mean(np.exp(resid)))
    if not np.isfinite(smear) or smear <= 0:
        smear = 1.0
    return smear


def backtransform_with_smearing(pred_log: np.ndarray, smear_factor: float) -> np.ndarray:
    pred_raw = np.exp(pred_log) * smear_factor - 1.0
    pred_raw = np.clip(pred_raw, 0, None)
    return pred_raw

def evaluate_fold(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Dict[str, float]:
    return {
        "rmse_raw": rmse(y_true_raw, y_pred_raw),
        "mae_raw": float(mean_absolute_error(y_true_raw, y_pred_raw)),
        "r2_raw": float(r2_score(y_true_raw, y_pred_raw)),
        "bias_raw": float(np.mean(y_pred_raw - y_true_raw)),
    }


def spatial_group_kfold_splits(groups: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(42)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    folds = np.array_split(shuffled, n_splits)
    all_splits = []

    for fold_groups in folds:
        test_mask = groups.isin(set(fold_groups)).to_numpy()
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        all_splits.append((train_idx, test_idx))

    return all_splits


def run_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_pipeline: Pipeline,
    model_name: str,
    spatial_groups: pd.Series,
    outdir: Path,
    n_splits: int = 5
) -> pd.DataFrame:
    X = df[feature_cols].copy()
    y_raw = df[TARGET_COL].to_numpy(dtype=float)
    y_log = np.log1p(y_raw)

    splits = spatial_group_kfold_splits(spatial_groups, n_splits=n_splits)

    rows = []
    smear_rows = []
    oof = np.full(shape=len(df), fill_value=np.nan, dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        model = clone(model_pipeline)
        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_log_tr = y_log[tr_idx]
        y_raw_te = y_raw[te_idx]

        model.fit(X_tr, y_log_tr)

        pred_log_tr = model.predict(X_tr)
        smear_factor = compute_smearing_factor(y_log_tr, pred_log_tr)

        pred_log_te = model.predict(X_te)
        pred_raw_te = backtransform_with_smearing(pred_log_te, smear_factor)

        oof[te_idx] = pred_raw_te
        metrics = evaluate_fold(y_raw_te, pred_raw_te)
        metrics["fold"] = fold
        metrics["model"] = model_name
        metrics["n_train"] = len(tr_idx)
        metrics["n_test"] = len(te_idx)
        rows.append(metrics)

        smear_rows.append({
            "fold": fold,
            "model": model_name,
            "smear_factor": smear_factor,
        })

    fold_df = pd.DataFrame(rows)
    overall = evaluate_fold(y_raw[~np.isnan(oof)], oof[~np.isnan(oof)])
    overall_row = {
        "fold": "overall",
        "model": model_name,
        "n_train": np.nan,
        "n_test": int((~np.isnan(oof)).sum()),
        **overall
    }
    fold_df = pd.concat([fold_df, pd.DataFrame([overall_row])], ignore_index=True)

    pred_df = df[["Latitude", "Longitude", "year", "month", "dayofyear", TARGET_COL]].copy()
    pred_df[f"{model_name}_pred"] = oof
    pred_df.to_csv(outdir / f"cv_predictions_{model_name}.csv", index=False)

    smear_df = pd.DataFrame(smear_rows)
    smear_df.to_csv(outdir / f"cv_smearing_factors_{model_name}.csv", index=False)

    return fold_df


def fit_final_models(
    df: pd.DataFrame,
    linear_feature_cols: List[str],
    tree_feature_cols: List[str],
    outdir: Path
) -> Dict[str, Dict[str, object]]:
    X_linear = df[linear_feature_cols].copy()
    X_tree = df[tree_feature_cols].copy()
    y_raw = df[TARGET_COL].to_numpy(dtype=float)
    y_log = np.log1p(y_raw)

    linear_model = build_linear_model()
    rf_model = build_rf_model()
    xgb_model = build_xgb_model()

    linear_model.fit(X_linear, y_log)
    rf_model.fit(X_tree, y_log)
    xgb_model.fit(X_tree, y_log)

    linear_smear = compute_smearing_factor(y_log, linear_model.predict(X_linear))
    rf_smear = compute_smearing_factor(y_log, rf_model.predict(X_tree))
    xgb_smear = compute_smearing_factor(y_log, xgb_model.predict(X_tree))

    lin = linear_model.named_steps["model"]
    coef_df = pd.DataFrame({
        "feature": linear_feature_cols,
        "coefficient_standardized": lin.coef_
    }).sort_values("coefficient_standardized", key=np.abs, ascending=False)
    coef_df.to_csv(outdir / "linear_model_coefficients.csv", index=False)

    rf = rf_model.named_steps["model"]
    rf_imp_df = pd.DataFrame({
        "feature": tree_feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    rf_imp_df.to_csv(outdir / "rf_feature_importances.csv", index=False)

    xgb = xgb_model.named_steps["model"]
    xgb_imp_df = pd.DataFrame({
        "feature": tree_feature_cols,
        "importance_gain": xgb.feature_importances_
    }).sort_values("importance_gain", ascending=False)
    xgb_imp_df.to_csv(outdir / "xgb_feature_importances.csv", index=False)

    calibration_df = pd.DataFrame([
        {"model": "linear", "smear_factor": linear_smear},
        {"model": "random_forest", "smear_factor": rf_smear},
        {"model": "xgboost", "smear_factor": xgb_smear},
    ])
    calibration_df.to_csv(outdir / "model_calibration_factors.csv", index=False)

    return {
        "linear": {"model": linear_model, "smear_factor": linear_smear, "feature_cols": linear_feature_cols},
        "random_forest": {"model": rf_model, "smear_factor": rf_smear, "feature_cols": tree_feature_cols},
        "xgboost": {"model": xgb_model, "smear_factor": xgb_smear, "feature_cols": tree_feature_cols},
    }


def predict_on_feature_table(
    model: Pipeline,
    feature_table: pd.DataFrame,
    feature_cols: List[str],
    smear_factor: float = 1.0
) -> pd.DataFrame:
    X = feature_table[feature_cols].copy()
    pred_log = model.predict(X)
    pred_raw = backtransform_with_smearing(pred_log, smear_factor)

    out = feature_table.copy()
    out["pred_chla"] = pred_raw
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument(
        "--season-months",
        nargs="*",
        type=int,
        default=None,
        help="Restrict training to these months. Default: None"
    )
    ap.add_argument("--block-size-km", type=float, default=50.0)
    ap.add_argument(
        "--include-month",
        action="store_true",
        help="Include raw month as predictor for tree models."
    )
    ap.add_argument(
        "--include-dayofyear-raw",
        action="store_true",
        help="Include raw dayofyear in addition to cyclic terms for tree models."
    )
    ap.add_argument(
        "--include-latlon",
        action="store_true",
        help="Include Latitude and Longitude as predictors."
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = add_cyclic_time_features(df)
    df = filter_training_rows(df, season_months=args.season_months)

    linear_feature_cols = select_linear_predictor_columns(
        df,
        include_latlon=args.include_latlon,
    )
    tree_feature_cols = select_tree_predictor_columns(
        df,
        include_month=args.include_month,
        include_dayofyear_raw=args.include_dayofyear_raw,
        include_latlon=args.include_latlon,
    )

    if len(linear_feature_cols) == 0:
        raise ValueError("No valid linear-model features were found in the reduced feature set.")
    if len(tree_feature_cols) == 0:
        raise ValueError("No valid tree-model features were found.")

    pd.Series(linear_feature_cols, name="feature").to_csv(outdir / "selected_features_linear.csv", index=False)
    pd.Series(tree_feature_cols, name="feature").to_csv(outdir / "selected_features_tree.csv", index=False)

    spatial_groups = make_spatial_blocks(df, block_size_km=args.block_size_km)
    df = df.copy()
    df["spatial_block"] = spatial_groups

    df.to_csv(outdir / "training_table_prepared.csv", index=False)

    linear_pipe = build_linear_model()
    rf_pipe = build_rf_model()
    xgb_pipe = build_xgb_model()

    cv_linear = run_cv(df, linear_feature_cols, linear_pipe, "linear", spatial_groups, outdir)
    cv_rf = run_cv(df, tree_feature_cols, rf_pipe, "random_forest", spatial_groups, outdir)
    cv_xgb = run_cv(df, tree_feature_cols, xgb_pipe, "xgboost", spatial_groups, outdir)
    cv_all = pd.concat([cv_linear, cv_rf, cv_xgb], ignore_index=True)
    cv_all.to_csv(outdir / "spatial_cv_metrics.csv", index=False)

    fit_final_models(df, linear_feature_cols, tree_feature_cols, outdir)

    summary = {
        "n_rows_after_filtering": int(len(df)),
        "season_months": args.season_months,
        "block_size_km": args.block_size_km,
        "include_latlon": bool(args.include_latlon),
        "n_linear_features": len(linear_feature_cols),
        "linear_features": linear_feature_cols,
        "n_tree_features": len(tree_feature_cols),
        "tree_features": tree_feature_cols,
        "models": ["linear_multiple_regression", "random_forest", "xgboost"],
        "validation": "spatial_cv_only_all_years",
        "calibration": "duan_smearing_on_log1p_backtransform",
        "year_used_as_predictor": False,
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("Done.")
    print(f"Rows after filtering: {len(df)}")
    print(f"Linear features: {len(linear_feature_cols)}")
    print(f"Tree features: {len(tree_feature_cols)}")
    print(f"Include lat/lon: {args.include_latlon}")
    print("Models trained: linear_multiple_regression, random_forest, xgboost")
    print("Validation run: spatial_cv_only_all_years")
    print("Calibration: duan_smearing_on_log1p_backtransform")
    print("Year used as predictor: False")
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()