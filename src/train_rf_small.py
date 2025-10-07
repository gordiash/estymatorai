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

from src.config import CreateSqlAlchemyEngine
from src.data_loader import LoadTables
from src.feature_engineering import CleanAndEngineerFeatures
from src.utils.reporting import ComputeMape, SaveJson


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
    train_df = Xy.iloc[:idx].copy()
    test_df = Xy.iloc[idx:].copy()
    return train_df, test_df, datetime.now()


def _make_pipeline_rf(n_estimators: int = 50,  # ZMNIEJSZONE z 200
                       max_depth: int | None = 10,  # ZMNIEJSZONE z None
                       n_jobs: int = 1) -> Pipeline:
    """Tworzy pipeline Random Forest z preprocessing"""
    
    # Definicja kolumn kategorycznych
    categorical_features = [
        "city", "district", "province", "building_type", 
        "heating_type", "standard_of_finish", "market", "gf_city"
    ]
    
    # Preprocessing dla kolumn kategorycznych
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=n_jobs
    )
    
    # Pipeline z log transformation
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor=rf,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])
    
    return pipeline


def PrepareData(Xy: pd.DataFrame, categorical_features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Przygotowuje dane do treningu Random Forest"""
    
    # Usuń kolumny które nie są potrzebne (ale zostaw price dla y)
    columns_to_drop = ['listing_date', 'scraped_at', 'created_at', 'updated_at', 'gf_created_at', 'gf_updated_at']
    Xy_clean = Xy.drop(columns=[col for col in columns_to_drop if col in Xy.columns])
    
    # Upewnij się, że wszystkie kolumny kategoryczne są string
    for col in categorical_features:
        if col in Xy_clean.columns:
            Xy_clean[col] = Xy_clean[col].astype("string").fillna("Unknown")
    
    # Upewnij się, że wszystkie kolumny numeryczne są float
    numeric_cols = Xy_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Xy_clean[col] = pd.to_numeric(Xy_clean[col], errors='coerce').fillna(0.0).astype(float)
    
    # Podziel na X i y
    X = Xy_clean.drop(columns=['price'])
    y = Xy_clean['price']
    
    return X, y


def Train(Xy: pd.DataFrame, test_days: int = 30) -> Pipeline:
    """Trenuje model Random Forest"""
    
    print("Przygotowywanie danych...")
    
    # Podziel dane na train/test
    train_df, test_df, time_cutoff = _split_by_time(Xy, test_days, None)
    
    print(f"Train: {len(train_df)} próbek")
    print(f"Test: {len(test_df)} próbek")
    print(f"Cutoff: {time_cutoff}")
    
    # Definicja kolumn kategorycznych
    categorical_features = [
        "city", "district", "province", "building_type", 
        "heating_type", "standard_of_finish", "market", "gf_city"
    ]
    
    # Przygotuj dane treningowe
    X_train, y_train = PrepareData(train_df, categorical_features)
    X_test, y_test = PrepareData(test_df, categorical_features)
    
    print(f"Features: {X_train.shape[1]}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Stwórz pipeline
    pipeline = _make_pipeline_rf(n_estimators=50, max_depth=10, n_jobs=1)
    
    print("Trenowanie modelu...")
    
    # Hyperparameter tuning z mniejszymi parametrami
    param_dist = {
        "model__regressor__n_estimators": [30, 50, 70],  # ZMNIEJSZONE
        "model__regressor__max_depth": [8, 10, 12],      # ZMNIEJSZONE
        "model__regressor__min_samples_split": [3, 5, 7],
        "model__regressor__min_samples_leaf": [1, 2, 3],
        "model__regressor__max_features": ['sqrt', 'log2']
    }
    
    # RandomizedSearchCV z mniejszą liczbą iteracji
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=5,  # ZMNIEJSZONE z 20
        cv=3,      # ZMNIEJSZONE z 5
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1,
        verbose=1
    )
    
    # Trenuj model
    random_search.fit(X_train, y_train)
    
    print("Najlepsze parametry:")
    print(random_search.best_params_)
    
    # Ewaluacja
    y_pred = random_search.predict(X_test)
    mape = ComputeMape(y_test, y_pred)
    
    print(f"MAPE na test set: {mape:.2f}%")
    
    # Zapisz model
    model_path = Path("artifacts/model_rf_small.joblib")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(random_search.best_estimator_, model_path)
    
    print(f"Model zapisany: {model_path}")
    
    # Raport
    report = {
        "model_type": "RandomForestRegressor (small)",
        "best_params": random_search.best_params_,
        "mape": mape,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X_train.shape[1],
        "model_size_mb": model_path.stat().st_size / (1024*1024)
    }
    
    SaveJson(report, Path("artifacts/report_small.json"))
    
    return random_search.best_estimator_


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest model (small version)")
    parser.add_argument("--test-days", type=int, default=30, help="Number of days for test set")
    
    args = parser.parse_args()
    
    print("=== Random Forest Training (Small Version) ===")
    print(f"Test days: {args.test_days}")
    
    # Załaduj dane
    engine = CreateSqlAlchemyEngine()
    listings_df, buildings_df = LoadTables(engine)
    
    # Przygotuj dane
    Xy, categorical_features, target_column = CleanAndEngineerFeatures(listings_df)
    
    print(f"Total samples: {len(Xy)}")
    
    # Trenuj model
    model = Train(Xy, args.test_days)
    
    print("Trening zakończony!")


if __name__ == "__main__":
    main()

