from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from pandas.api.types import is_datetime64_any_dtype

from .config import CreateSqlAlchemyEngine
from .data_loader import LoadTables
from .feature_engineering import CleanAndEngineerFeatures
from .utils.reporting import ComputeMape, SaveJson
from .utils.paths import ToSafeName


def _build_Xy_for_city(Xy: pd.DataFrame, city_name: str) -> pd.DataFrame:
    if "city" not in Xy.columns:
        return Xy.copy()
    m = Xy["city"].astype(str) == str(city_name)
    return Xy.loc[m].copy()


def _prepare_splits(Xy: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame, datetime]:
    if "listing_date" in Xy.columns:
        Xy["listing_date"] = pd.to_datetime(Xy["listing_date"], errors="coerce")
        # obetnij tz
        if getattr(Xy["listing_date"].dt, "tz", None) is not None:
            Xy["listing_date"] = Xy["listing_date"].dt.tz_convert("UTC").dt.tz_localize(None)
        max_date = Xy["listing_date"].max()
        time_cutoff = max_date - timedelta(days=test_days)
        train_df = Xy[Xy["listing_date"] < time_cutoff].copy()
        test_df = Xy[Xy["listing_date"] >= time_cutoff].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            Xy_sorted = Xy.sort_values("listing_date").reset_index(drop=True)
            idx = int(0.8 * len(Xy_sorted))
            train_df = Xy_sorted.iloc[:idx].copy()
            test_df = Xy_sorted.iloc[idx:].copy()
    else:
        time_cutoff = pd.Timestamp.utcnow().to_pydatetime()
        idx = int(0.8 * len(Xy))
        train_df = Xy.iloc[:idx].copy()
        test_df = Xy.iloc[idx:].copy()
    return train_df, test_df, time_cutoff


def _train_single_city(
    city: str,
    Xy_city: pd.DataFrame,
    cat_cols: list[str],
    random_seed: int,
    test_days: int,
    gpu: bool,
) -> dict:
    # log-target
    Xy_city = Xy_city.copy()
    Xy_city["TargetLog"] = np.log1p(Xy_city["price"].astype(float))
    training_target_col = "TargetLog"

    train_df, test_df, cutoff = _prepare_splits(Xy_city, test_days)

    # GroupKFold po dzielnicy jeśli dostępna, w obrębie miasta
    group_col = "district" if "district" in train_df.columns else None
    if group_col is not None:
        groups = train_df[group_col].astype(str).values
        n_unique_groups = max(1, train_df[group_col].astype(str).nunique())
    else:
        groups = np.arange(len(train_df))
        n_unique_groups = len(train_df)

    n_splits = int(min(5, n_unique_groups, len(train_df)))
    n_splits = max(2, n_splits) if len(train_df) >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)

    def build_XY(df: pd.DataFrame):
        y = df[training_target_col].values
        X = df.drop(columns=[training_target_col, "price"], errors="ignore")
        # usuń datetime
        datetime_cols = [c for c in X.columns if is_datetime64_any_dtype(X[c])]
        if datetime_cols:
            X = X.drop(columns=datetime_cols)
        # kategorie
        auto_cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype.name == "string"]
        for c in set(cat_cols).union(auto_cat_cols):
            if c in X.columns:
                X[c] = X[c].astype("string").fillna("Unknown")
        final_cat_cols = [c for c in X.columns if X[c].dtype.name == "string"]
        cat_idx = [X.columns.get_loc(c) for c in final_cat_cols]
        return X, y, cat_idx

    best_model = None
    best_mape = float("inf")
    best_params = None

    param_grid = [
        {"depth": d, "learning_rate": lr, "l2_leaf_reg": l2}
        for d in (6, 8, 10)
        for lr in (0.03, 0.05, 0.08)
        for l2 in (3.0, 5.0, 8.0)
    ]

    for params in param_grid:
        cv_mapes = []
        for tr_idx, val_idx in gkf.split(train_df, groups=groups):
            tr_df, val_df = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            X_tr, y_tr, cat_idx = build_XY(tr_df)
            X_val, y_val_log, _ = build_XY(val_df)

            loss = "RMSE"
            cb_params = dict(
                loss_function=loss,
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                l2_leaf_reg=params["l2_leaf_reg"],
                random_seed=random_seed,
                iterations=20000,
                od_type="Iter",
                od_wait=400,
                eval_metric="RMSE",
                verbose=False,
            )
            if gpu:
                cb_params.update(task_type="GPU", devices=os.getenv("CUDA_DEVICES", "0"))

            model = CatBoostRegressor(**cb_params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val_log), use_best_model=True, cat_features=cat_idx)

            val_pred_log = model.predict(X_val)
            val_pred = np.expm1(val_pred_log)
            y_val = np.expm1(y_val_log)
            mape = float(ComputeMape(y_val, val_pred))
            cv_mapes.append(mape)

        avg_mape = float(np.mean(cv_mapes)) if cv_mapes else float("inf")
        if avg_mape < best_mape:
            best_mape = avg_mape
            best_params = params
            best_model = None  # wymuś ponowny fit na pełnym treningu

    # Trenowanie finalnego modelu dla miasta
    X_tr_full, y_tr_full, cat_idx = build_XY(train_df)
    X_te, y_te_log, _ = build_XY(test_df)
    cb_params = dict(
        loss_function="RMSE",
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        random_seed=random_seed,
        iterations=20000,
        od_type="Iter",
        od_wait=400,
        eval_metric="RMSE",
        verbose=False,
    )
    if gpu:
        cb_params.update(task_type="GPU", devices=os.getenv("CUDA_DEVICES", "0"))

    best_model = CatBoostRegressor(**cb_params)
    best_model.fit(X_tr_full, y_tr_full, eval_set=(X_te, y_te_log), use_best_model=True, cat_features=cat_idx)

    # Test
    test_pred_log = best_model.predict(X_te)
    test_pred = np.expm1(test_pred_log)
    y_test = np.expm1(y_te_log)
    test_mae = float(mean_absolute_error(y_test, test_pred))
    test_mape = float(ComputeMape(y_test, test_pred))

    return {
        "model": best_model,
        "test": {"MAE": test_mae, "MAPE": test_mape},
        "best_params": best_params,
        "cutoff": str(cutoff),
        "feature_names": list(X_tr_full.columns),
        "categorical_names": [c for c in X_tr_full.columns if X_tr_full[c].dtype.name == "string"],
        "rows": {"train": int(len(train_df)), "test": int(len(test_df))},
    }


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-days", type=int, default=int(os.getenv("TEST_DAYS", "60")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument("--gpu", action="store_true", help="Użyj GPU (CatBoost task_type=GPU)")
    parser.add_argument("--limit-cities", type=int, default=0, help="Opcjonalny limit liczby miast do trenowania")
    args = parser.parse_args()

    engine = CreateSqlAlchemyEngine()
    raw_df, geo_df = LoadTables(engine)
    Xy, cat_cols, target_col = CleanAndEngineerFeatures(raw_df)

    # Priorytet: miasta ze słownika w geographic_features (gf_city)
    if "gf_city" in geo_df.columns:
        cities = [c for c in geo_df["gf_city"].astype(str).dropna().unique().tolist()]
    else:
        # Fallback do miast z danych ogłoszeń
        if "city" in Xy.columns:
            cities = [c for c in Xy["city"].astype(str).dropna().unique().tolist()]
        else:
            cities = ["GLOBAL"]

    if args.limit_cities and args.limit_cities > 0:
        cities = cities[: args.limit_cities]

    artifacts_root = Path("artifacts")
    (artifacts_root / "per_city").mkdir(parents=True, exist_ok=True)

    overall = {
        "DataTreningu": pd.Timestamp.utcnow().isoformat(),
        "Miasta": cities,
        "Wersja": 1,
        "GPU": bool(args.gpu),
        "TestDays": int(args.test_days),
        "Modele": [],
    }

    for city in cities:
        safe = ToSafeName(city)
        out_dir = artifacts_root / "per_city" / safe
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Trening miasta: {city} → {out_dir} ===")
        # Filtruj wyłącznie po nazwie miasta w danych ogłoszeń
        Xy_city = _build_Xy_for_city(Xy, city)
        if len(Xy_city) < 200:
            print(f"Pominięto {city}: zbyt mało danych ({len(Xy_city)})")
            continue
        res = _train_single_city(city, Xy_city, cat_cols, args.seed, args.test_days, args.gpu)

        # Zapis modelu i schematu
        model_path = out_dir / "model.cbm"
        res["model"].save_model(model_path.as_posix())
        SaveJson({
            "feature_names": res["feature_names"],
            "categorical_names": res["categorical_names"],
        }, out_dir / "feature_schema.json")

        # Raport per miasto
        city_report = {
            "Miasto": str(city),
            "SafeName": safe,
            "Test": res["test"],
            "Parametry": res["best_params"],
            "Cutoff": res["cutoff"],
            "Rows": res["rows"],
            "ModelPath": str(model_path.as_posix()),
        }
        SaveJson(city_report, out_dir / "report.json")
        overall["Modele"].append(city_report)

    # Indeks modeli
    SaveJson(overall, artifacts_root / "per_city_index.json")
    print("\nZakończono trening per-miasto. Indeks: artifacts/per_city_index.json")


if __name__ == "__main__":
    Main()


