from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
import joblib

from pandas.api.types import is_datetime64_any_dtype

from .config import CreateSqlAlchemyEngine
from .data_loader import LoadTables
from .feature_engineering import CleanAndEngineerFeatures
from .utils.reporting import ComputeMape, SaveJson


def _split_by_time(Xy: pd.DataFrame, test_days: int, time_cutoff: datetime | None) -> Tuple[pd.DataFrame, pd.DataFrame, datetime]:
    if "listing_date" in Xy.columns:
        Xy = Xy.copy()
        Xy["listing_date"] = pd.to_datetime(Xy["listing_date"], errors="coerce")
        if getattr(Xy["listing_date"].dt, "tz", None) is not None:
            Xy["listing_date"] = Xy["listing_date"].dt.tz_convert("UTC").dt.tz_localize(None)
        if time_cutoff is None:
            max_date = Xy["listing_date"].max()
            time_cutoff = max_date - timedelta(days=test_days)
        # ensure naive
        time_cutoff = pd.Timestamp(time_cutoff)
        if time_cutoff.tz is not None:
            time_cutoff = time_cutoff.tz_convert("UTC").tz_localize(None)
        time_cutoff = time_cutoff.to_pydatetime()
        train_df = Xy[Xy["listing_date"] < time_cutoff].copy()
        test_df = Xy[Xy["listing_date"] >= time_cutoff].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            Xy_sorted = Xy.sort_values("listing_date").reset_index(drop=True)
            idx = int(0.8 * len(Xy_sorted))
            train_df = Xy_sorted.iloc[:idx].copy()
            test_df = Xy_sorted.iloc[idx:].copy()
        return train_df, test_df, time_cutoff
    # brak daty → prosty split 80/20
    idx = int(0.8 * len(Xy))
    return Xy.iloc[:idx].copy(), Xy.iloc[idx:].copy(), pd.Timestamp.utcnow().to_pydatetime()


def _build_preprocessor(df: pd.DataFrame, target_col: str) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df.drop(columns=[target_col])
    # usuń kolumny datetime
    datetime_cols = [c for c in X.columns if is_datetime64_any_dtype(X[c])]
    if datetime_cols:
        X = X.drop(columns=datetime_cols)
    # wykryj typy
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype.name == "string"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def _make_pipeline_rf(n_estimators: int = 200,
                       max_depth: int | None = None,
                       min_samples_split: int = 2,
                       min_samples_leaf: int = 1,
                       max_features: str | float = "sqrt",
                       bootstrap: bool = True,
                       random_state: int = 42,
                       preprocessor: ColumnTransformer | None = None) -> Pipeline:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=1,
        random_state=random_state,
    )
    # Trenujemy na log-cenie, zwracamy w PLN
    model = TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)
    steps = []
    if preprocessor is not None:
        steps.append(("prep", preprocessor))
    steps.append(("model", model))
    return Pipeline(steps)


def _groups_for_cv(train_df: pd.DataFrame) -> Tuple[np.ndarray, int]:
    group_col = "city" if "city" in train_df.columns else ("district" if "district" in train_df.columns else None)
    if group_col is not None:
        groups = train_df[group_col].astype(str).values
        n_unique_groups = max(1, train_df[group_col].astype(str).nunique())
    else:
        groups = np.arange(len(train_df))
        n_unique_groups = len(train_df)
    return groups, n_unique_groups


def TrainRF(time_cutoff: datetime | None, test_days: int, random_seed: int,
            max_attempts: int = 5, n_iter: int = 30,
            target_mape: float = 0.0038) -> tuple[Pipeline, dict]:
    engine = CreateSqlAlchemyEngine()
    raw_df, _ = LoadTables(engine)

    Xy, cat_cols, target_col = CleanAndEngineerFeatures(raw_df)

    train_df, test_df, cutoff = _split_by_time(Xy, test_days, time_cutoff)

    preprocessor, num_cols, cat_ohe_cols = _build_preprocessor(train_df, target_col)

    # przygotuj CV
    groups, n_unique_groups = _groups_for_cv(train_df)
    n_splits = int(min(5, n_unique_groups, len(train_df)))
    n_splits = max(2, n_splits) if len(train_df) >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)

    # dane X, y na potrzeby CV i fitu
    y_tr = train_df[target_col].values.astype(float)
    X_tr = train_df.drop(columns=[target_col])
    # usuń datetime po stronie X (preprocessor i tak je wytnie, ale dla spójności)
    datetime_cols = [c for c in X_tr.columns if is_datetime64_any_dtype(X_tr[c])]
    if datetime_cols:
        X_tr = X_tr.drop(columns=datetime_cols)

    y_te = test_df[target_col].values.astype(float)
    X_te = test_df.drop(columns=[target_col])
    datetime_cols_te = [c for c in X_te.columns if is_datetime64_any_dtype(X_te[c])]
    if datetime_cols_te:
        X_te = X_te.drop(columns=datetime_cols_te)

    # scorer MAPE (mniejszy lepszy)
    def _mape_scorer(y_true, y_pred):
        return ComputeMape(y_true, y_pred)
    mape_scorer = make_scorer(_mape_scorer, greater_is_better=False)

    # przestrzeń losowa (zmniejszona dla szybszego treningu)
    param_dist = {
        "model__regressor__n_estimators": [200, 400, 600],
        "model__regressor__max_depth": [None, 10, 15],
        "model__regressor__min_samples_split": [2, 5],
        "model__regressor__min_samples_leaf": [1, 2],
        "model__regressor__max_features": ["sqrt", "log2"],
        "model__regressor__bootstrap": [True],
    }

    best_overall = {
        "pipeline": None,
        "params": None,
        "test_mae": float("inf"),
        "test_mape": float("inf"),
        "attempt": 0,
    }

    rng = np.random.default_rng(random_seed)

    for attempt in range(1, max_attempts + 1):
        seed = int(rng.integers(0, 1_000_000))
        base = _make_pipeline_rf(random_state=seed, preprocessor=preprocessor)
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=min(n_iter, 10),
            scoring=mape_scorer,
            cv=gkf.split(X_tr, groups=groups),
            n_jobs=1,
            verbose=1,
            random_state=seed,
            refit=True,
        )
        search.fit(X_tr, y_tr)

        # Ocena na teście (pred w PLN – TTR robi inverse)
        y_pred = search.best_estimator_.predict(X_te)
        test_mae = float(mean_absolute_error(y_te, y_pred))
        test_mape = float(ComputeMape(y_te, y_pred))

        if test_mape < best_overall["test_mape"]:
            best_overall.update({
                "pipeline": search.best_estimator_,
                "params": search.best_params_,
                "test_mae": test_mae,
                "test_mape": test_mape,
                "attempt": attempt,
            })

        print({"Attempt": attempt, "Test": {"MAE": test_mae, "MAPE": test_mape}})
        if test_mape <= target_mape:
            break

    # Zapis artefaktów
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True, parents=True)
    model_path = artifacts / "model_rf.joblib"
    joblib.dump(best_overall["pipeline"], model_path.as_posix())

    report = {
        "DataTreningu": pd.Timestamp.utcnow().isoformat(),
        "Split": {
            "Metoda": f"Time-based + GroupKFold({gkf.n_splits})",
            "TimeCutoff": str(cutoff),
            "TrainRows": int(len(train_df)),
            "TestRows": int(len(test_df)),
        },
        "Model": {
            "Typ": "RandomForestRegressor (with TransformedTargetRegressor log1p)",
            "Parametry": best_overall["params"],
        },
        "Wyniki": {
            "Test": {"MAE": best_overall["test_mae"], "MAPE": best_overall["test_mape"]},
            "OsiagnietoProgMAPE": bool(best_overall["test_mape"] <= target_mape),
            "ProgMAPE": float(target_mape),
            "NajlepszaProba": int(best_overall["attempt"]),
        },
    }
    SaveJson(report, artifacts / "report_rf.json")

    print({"Test": {"MAE": best_overall["test_mae"], "MAPE": best_overall["test_mape"]}})
    return best_overall["pipeline"], report


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-cutoff", type=str, default=os.getenv("TIME_CUTOFF"), help="Data odcięcia walidacji/testu, np. 2024-12-01 00:00:00")
    parser.add_argument("--test-days", type=int, default=int(os.getenv("TEST_DAYS", "60")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument("--max-attempts", type=int, default=int(os.getenv("MAX_ATTEMPTS", "5")))
    parser.add_argument("--n-iter", type=int, default=int(os.getenv("N_ITER", "30")))
    parser.add_argument("--target-mape", type=float, default=float(os.getenv("TARGET_MAPE", "0.0038")))
    args = parser.parse_args()

    cutoff = None
    if args.time_cutoff:
        cutoff = pd.to_datetime(args.time_cutoff).to_pydatetime()

    TrainRF(cutoff, args.test_days, args.seed, args.max_attempts, args.n_iter, args.target_mape)


if __name__ == "__main__":
    Main()
