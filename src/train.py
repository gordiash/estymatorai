from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import joblib
from pandas.api.types import is_datetime64_any_dtype

from .config import CreateSqlAlchemyEngine
from .data_loader import LoadTables
from .feature_engineering import CleanAndEngineerFeatures
from .utils.reporting import ComputeMape, SaveJson


def PrepareData(df: pd.DataFrame, target_col: str):
    """Przygotowuje dane dla Random Forest"""
    drop_cols = [target_col]
    if "price" in df.columns and target_col != "price":
        drop_cols.append("price")
    X = df.drop(columns=drop_cols)
    # Usuń kolumny datetime
    datetime_cols = [c for c in X.columns if is_datetime64_any_dtype(X[c])]
    if datetime_cols:
        X = X.drop(columns=datetime_cols)
    # Wykryj kolumny kategoryczne
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype.name == "string"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    y = df[target_col].values
    return X, y, categorical_cols, numeric_cols


def Train(time_cutoff: datetime | None, test_days: int, random_seed: int) -> tuple[Pipeline, dict]:
    engine = CreateSqlAlchemyEngine()
    raw_df, geo_df = LoadTables(engine)

    Xy, cat_cols, target_col = CleanAndEngineerFeatures(raw_df)

    # Ujednolicenie strefy czasu: zawsze porównujemy daty w wersji NAIVE (UTC odcięte z tz)
    if "listing_date" in Xy.columns:
        Xy["listing_date"] = pd.to_datetime(Xy["listing_date"], errors="coerce")
        if getattr(Xy["listing_date"].dt, "tz", None) is not None:
            Xy["listing_date"] = Xy["listing_date"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Ustal cutoff
    if time_cutoff is None:
        if "listing_date" in Xy.columns and Xy["listing_date"].notna().any():
            max_date = Xy["listing_date"].max()
            time_cutoff = max_date - timedelta(days=test_days)
        else:
            time_cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=test_days))
    # Zapewnij NAIVE (bez tz)
    time_cutoff = pd.Timestamp(time_cutoff)
    if time_cutoff.tz is not None:
        time_cutoff = time_cutoff.tz_convert("UTC").tz_localize(None)
    time_cutoff = time_cutoff.to_pydatetime()

    if "listing_date" in Xy.columns:
        train_df = Xy[Xy["listing_date"] < time_cutoff].copy()
        test_df = Xy[Xy["listing_date"] >= time_cutoff].copy()
        # Fallback: jeśli któryś zbiór pusty, zrób 80/20 po dacie (sort)
        if len(train_df) == 0 or len(test_df) == 0:
            Xy_sorted = Xy.sort_values("listing_date").reset_index(drop=True)
            idx = int(0.8 * len(Xy_sorted))
            train_df = Xy_sorted.iloc[:idx].copy()
            test_df = Xy_sorted.iloc[idx:].copy()
    else:
        # Jeśli brak daty, zrób prosty split 80/20
        idx = int(0.8 * len(Xy))
        train_df = Xy.iloc[:idx].copy()
        test_df = Xy.iloc[idx:].copy()

    # Przygotuj dane treningowe i testowe
    X_train, y_train, cat_cols_train, num_cols_train = PrepareData(train_df, target_col)
    X_test, y_test, cat_cols_test, num_cols_test = PrepareData(test_df, target_col)
    
    # Stwórz preprocessor dla kolumn kategorycznych
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols_train),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_train),
        ],
        remainder="drop",
    )
    
    # Stwórz Random Forest z TransformedTargetRegressor (log transform)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=1,
        random_state=random_seed,
    )
    
    # Pipeline z transformacją log na target
    model = TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    
    # GroupKFold dla walidacji krzyżowej
    group_col = "city" if "city" in train_df.columns else ("district" if "district" in train_df.columns else None)
    if group_col is not None:
        groups = train_df[group_col].astype(str).values
        n_unique_groups = max(1, train_df[group_col].astype(str).nunique())
    else:
        groups = np.arange(len(train_df))
        n_unique_groups = len(train_df)

    n_splits = int(min(5, n_unique_groups, len(train_df)))
    n_splits = max(2, n_splits) if len(train_df) >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)
    
    # Parametry do RandomizedSearchCV (zmniejszone dla szybszego treningu)
    param_dist = {
        "model__regressor__n_estimators": [200, 400, 600],
        "model__regressor__max_depth": [None, 10, 15],
        "model__regressor__min_samples_split": [2, 5],
        "model__regressor__min_samples_leaf": [1, 2],
        "model__regressor__max_features": ["sqrt", "log2"],
        "model__regressor__bootstrap": [True],
    }
    
    # Scorer MAPE
    def mape_scorer(y_true, y_pred):
        return ComputeMape(y_true, y_pred)
    
    from sklearn.metrics import make_scorer
    mape_scorer = make_scorer(mape_scorer, greater_is_better=False)
    
    # RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring=mape_scorer,
        cv=gkf.split(X_train, groups=groups),
        n_jobs=1,
        verbose=1,
        random_state=random_seed,
        refit=True,
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_

    # Test
    test_pred = best_model.predict(X_test)
    y_test = test_df[target_col].values.astype(float)

    # Kalibracja per-miasto (prosta korekta liniowa y = a*x + b)
    if "city" in train_df.columns:
        calibrated = np.array(test_pred, copy=True)
        for city, tr_city_df in train_df.groupby("city"):
            # Dopasuj korektę na zbiorze treningowym dla miasta
            Xc, yc, _, _ = PrepareData(tr_city_df, target_col)
            pc = best_model.predict(Xc)
            yc = tr_city_df[target_col].values.astype(float)
            if len(pc) >= 10:
                A = np.vstack([pc, np.ones_like(pc)]).T
                a, b = np.linalg.lstsq(A, yc, rcond=None)[0]
                mask = (test_df.get("city") == city).values
                if mask.any():
                    calibrated[mask] = a * test_pred[mask] + b
        test_pred = calibrated

    test_mae = float(mean_absolute_error(y_test, test_pred))
    test_mape = float(ComputeMape(y_test, test_pred))

    # Per-miasto (liczone z gotowych predykcji X_test, bez ponownego przewidywania)
    per_city = []
    if "city" in test_df.columns:
        # Mapowanie pozycji w X_test do wierszy test_df (te same kolejności)
        test_positions = pd.Series(range(len(test_df)), index=test_df.index)
        for city, sdf in test_df.groupby("city"):
            pos = test_positions.loc[sdf.index].values
            y_c = y_test[pos]
            pred_c = test_pred[pos]
            per_city.append({
                "Miasto": str(city),
                "MAE": float(mean_absolute_error(y_c, pred_c)),
                "MAPE": float(ComputeMape(y_c, pred_c)),
                "Rows": int(len(sdf)),
            })

    # Feature importance z Random Forest
    feature_importance = best_model.named_steps['model'].regressor_.feature_importances_
    feature_names = best_model.named_steps['prep'].get_feature_names_out()
    top_features = [(name, float(importance)) for name, importance in zip(feature_names, feature_importance)]
    top_features.sort(key=lambda x: x[1], reverse=True)
    top_features = top_features[:10]

    # Raport
    report = {
        "DataTreningu": pd.Timestamp.utcnow().isoformat(),
        "ZakresDanych": {
            "Miasta": sorted([str(x) for x in Xy.get("city", pd.Series(dtype=str)).dropna().unique().tolist()]) if "city" in Xy.columns else [],
            "LiczbaRekordow": int(len(Xy)),
            "ZakresDat": {
                "Min": str(Xy["listing_date"].min()) if "listing_date" in Xy.columns else None,
                "Max": str(Xy["listing_date"].max()) if "listing_date" in Xy.columns else None,
            },
        },
        "Split": {
            "Metoda": f"Time-based + GroupKFold({gkf.n_splits})",
            "TimeCutoff": str(time_cutoff),
            "TrainRows": int(len(train_df)),
            "TestRows": int(len(test_df)),
        },
        "Model": {
            "Typ": "RandomForestRegressor (with TransformedTargetRegressor log1p)",
            "Parametry": best_params,
        },
        "Wyniki": {
            "Test": {"MAE": test_mae, "MAPE": test_mape},
            "PerMiasto": per_city,
        },
        "Wyjasnienia": {
            "TopCechyFeatureImportance": top_features,
        },
    }

    # Zapis artefaktów
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True, parents=True)
    model_path = artifacts / "model_rf.joblib"
    joblib.dump(best_model, model_path.as_posix())
    SaveJson(report, artifacts / "report_rf.json")

    print({"Test": {"MAE": test_mae, "MAPE": test_mape}})

    return best_model, report


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-cutoff", type=str, default=os.getenv("TIME_CUTOFF"), help="Data odcięcia walidacji/testu, np. 2024-12-01 00:00:00")
    parser.add_argument("--test-days", type=int, default=int(os.getenv("TEST_DAYS", "60")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    args = parser.parse_args()

    cutoff = None
    if args.time_cutoff:
        cutoff = pd.to_datetime(args.time_cutoff).to_pydatetime()

    Train(cutoff, args.test_days, args.seed)


if __name__ == "__main__":
    Main()
